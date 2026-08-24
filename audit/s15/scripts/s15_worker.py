"""S15 worker: one unit = one grid row, all 29 folds.
J1 no dynamics assert kills a job. J2 only Gate-C halts. J3 per-fold try/except.
J4 resume-safe. J5 USR1 handler. J6 STATUS.json heartbeat."""
import sys, os, json, time, socket, signal, hashlib, traceback, threading
import numpy as np, torch
sys.path.insert(0, "/users/3171356m/A-GCL/audit/s15/scripts")
import s15_data as DAT, s15_models as MO, s15_train as TR, s15_grid as G
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix,
    brier_score_loss)
from sklearn.linear_model import LogisticRegression

S15 = DAT.S15
_STOP = {"flag": False}

def _usr1(signum, frame):
    _STOP["flag"] = True
    print("USR1 received: finishing current fold then exiting 0", flush=True)
signal.signal(signal.SIGUSR1, _usr1)

# ------------------------------------------------------------------ metrics
def _boot(y, s, B=2000, seed=DAT.BASE):
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed); n = len(y)
    idx = rng.integers(0, n, (B, n)); Y = y[idx].astype(np.float64); S = s[idx]
    npos = Y.sum(1); nneg = n - npos; ok = (npos > 0) & (nneg > 0)
    r = rankdata(S, method="average", axis=1)
    return (((r * Y).sum(1) - npos * (npos + 1) / 2) / np.maximum(npos * nneg, 1))[ok]

def metrics(y, score, boot=2000):
    y = np.asarray(y); score = np.asarray(score, dtype=np.float64)
    if len(np.unique(y)) < 2: return dict(auc=float("nan"), n=int(len(y)))
    yh = (score > 0).astype(int)
    p = 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    bs = _boot(y, score, boot)
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    cal = LogisticRegression(C=1e10, max_iter=5000).fit(lg[:, None], y)
    return dict(auc=float(roc_auc_score(y, score)),
        auc_ci_lo=float(np.percentile(bs, 2.5)), auc_ci_hi=float(np.percentile(bs, 97.5)),
        auprc=float(average_precision_score(y, score)), acc=float(accuracy_score(y, yh)),
        bal_acc=float(balanced_accuracy_score(y, yh)),
        sens=float(tp / max(tp + fn, 1)), spec=float(tn / max(tn + fp, 1)),
        ppv=float(tp / max(tp + fp, 1)), npv=float(tn / max(tn + fn, 1)),
        f1=float(f1_score(y, yh)), mcc=float(matthews_corrcoef(y, yh)),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        brier=float(brier_score_loss(y, p)), calib_slope=float(cal.coef_[0, 0]),
        calib_intercept=float(cal.intercept_[0]),
        threshold="score>0 (sigmoid 0.5 head; LinearSVC boundary probe)", n=int(len(y)))

# ------------------------------------------------------------------ features
def alff_z(ALFF, tr):
    A = ALFF.astype(np.float64)
    mu, sd = A[tr].mean(0, keepdims=True), A[tr].std(0, keepdims=True)
    return ((A - mu) / np.maximum(sd, 1e-6)).astype(np.float32)

def alff_minmax(ALFF):
    A = ALFF.astype(np.float64)
    mn, mx = A.min(1, keepdims=True), A.max(1, keepdims=True)
    sp = mx - mn
    return np.where(sp > 0, (A - mn) / np.where(sp > 0, sp, 1.0), A).astype(np.float32)

def build_X(spec, FC, ALFF, tr, control=None, alff_mode="z"):
    """FC diagonal STAYS 1.0; no row z-scoring, no /max|FC|, no sparsify."""
    R = FC.astype(np.float32)
    if control == "C-SHUF":                       # FC-row columns shuffled per subject
        Rs = np.empty_like(R)
        for s in range(len(R)):
            Rs[s] = R[s][:, np.random.default_rng(DAT.BASE + s).permutation(90)]
        R = Rs
    A = alff_z(ALFF, tr) if alff_mode == "z" else alff_minmax(ALFF)
    I90 = np.repeat(np.eye(90, dtype=np.float32)[None], len(R), 0)
    X = {"alff": A, "fcrow": R, "fcrow+alff": np.concatenate([R, A], 2),
         "alff+onehot": np.concatenate([A, I90], 2)}[spec]
    if control == "C-ROI":                        # ROI order shuffled per subject
        Xs = np.empty_like(X); FCs = np.empty_like(FC)
        for s in range(len(X)):
            p = np.random.default_rng(DAT.BASE + 7000 + s).permutation(90)
            Xs[s] = X[s][p]; FCs[s] = FC[s][p][:, p]
        return Xs, FCs
    return X, FC

# ------------------------------------------------------------------ status
def write_status(jd, state, done, total, extra=None):
    r = dict(state=state, folds_done=done, folds_total=total,
             host=socket.gethostname(), updated=time.strftime("%F %T"))
    if extra: r.update(extra)
    json.dump(r, open(f"{jd}/STATUS.json.tmp", "w"), indent=1)
    os.replace(f"{jd}/STATUS.json.tmp", f"{jd}/STATUS.json")

def heartbeat(jd, stop):
    while not stop.is_set():
        open(f"{jd}/HEARTBEAT", "w").write(time.strftime("%F %T"))
        stop.wait(60)

def atomic_json(obj, path):
    json.dump(obj, open(path + ".tmp", "w"), indent=1, default=str)
    json.load(open(path + ".tmp")); os.replace(path + ".tmp", path)

# ------------------------------------------------------------------ run
def run(branch, idx):
    units = {"main": G.MAIN, "ctrl": G.CTRL, "tran": G.TRAN}[branch]
    u = units[idx]; uid = G.unit_id(u)
    jd = f"{S15}jobs/{uid}"; os.makedirs(jd, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.use_deterministic_algorithms(True)
    t_unit = time.time()
    d, man = DAT.load(where=uid)                     # GATE-C, every job (J2)
    FC, ALFF, y_true = d["FC"], d["ALFF"], d["y"].astype(np.int64)
    all_folds = DAT.all_folds(d)
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(jd, stop), daemon=True).start()
    write_status(jd, "running", 0, len(all_folds), dict(unit=uid, config=u))
    arch = u["arch"]; spec = G.ARMS[u["arm"]][1]
    control = u.get("control"); mode = u.get("mode")
    y_use = y_true.copy()
    if control == "C-PERM":
        y_use = np.random.default_rng(DAT.BASE).permutation(y_true)
        assert y_use.sum() == y_true.sum()
    cfg = dict(K_or_hidden=u["K_or_hidden"], lr=u["lr"], wd=u["wd"], loss=u["loss"],
               freeze_encoder=(control == "C-RAND"), readout="roi",
               scaled_softmax=True, dropout=0.10, H=128)
    seed = G.SEEDS[u["seed_idx"]]
    done = 0
    for tag, tr, te in all_folds:
        fp = f"{S15}feat/{uid}__{tag}.npz"
        if os.path.exists(fp):
            done += 1; write_status(jd, "running", done, len(all_folds)); continue
        t0 = time.time()
        try:
            X, FCu = build_X(spec, FC, ALFF, tr, control)
            need_graph = (arch == "WGIN")
            tr_use = np.arange(954) if mode == "T3" else tr     # T3 = LABEL LEAKAGE
            model, ema_sd, curve, info = TR.train_fold(
                arch, X, FCu, y_use, tr_use, cfg, seed, log=f"{uid}/{tag}")
            R, S = TR.extract(model, X, FCu, np.arange(954), need_graph)
            ema_model = MO.build_model(arch, X.shape[2], seed, cfg["K_or_hidden"],
                        freeze_encoder=cfg["freeze_encoder"], p=cfg["dropout"],
                        H=cfg["H"])
            ema_model.load_state_dict(ema_sd)
            _, S_ema = TR.extract(ema_model, X, FCu, np.arange(954), need_graph)
            pd_, poof = K.probe_pipe(R.astype(np.float64), y_use, [(tr, te)], [])
            ck = f"{S15}ckpt/{uid}__{tag}.pt"
            torch.save(model.state_dict(), ck + ".tmp"); os.replace(ck + ".tmp", ck)
            rec = {**u, **dict(status="OK", unit=uid, branch=branch,
                       fold=tag, fold_protocol=tag[:-1] if tag[-1].isdigit() else tag,
                       seed=seed, head=metrics(y_use[te], S[te]),
                       head_ema=metrics(y_use[te], S_ema[te]),
                       probe=metrics(y_use[te], poof[te]),
                       ema_delta=float(roc_auc_score(y_use[te], S_ema[te]) -
                                       roc_auc_score(y_use[te], S[te]))
                                 if len(np.unique(y_use[te])) > 1 else float("nan"),
                       **{k: v for k, v in info.items() if k != "movement"},
                       movement=info["movement"],
                       label_convention="ASD=1 NC=0 (A-GCL uses ASD=0/HC=1: AUC same,"
                                        " SENS/SPEC SWAPPED vs the paper)",
                       leakage=bool(u.get("leakage", False)),
                       h_fc=man["h_fc"], h_labels=man["h_labels"],
                       h_folds_lab=man["h_folds_lab"], cache_file=man["cache_file"],
                       node=socket.gethostname(),
                       ckpt_sha=hashlib.sha256(open(ck, "rb").read()).hexdigest()[:16],
                       wall_s=round(time.time() - t0, 1),
                       peak_rss_mb=round(__import__("resource").getrusage(
                           __import__("resource").RUSAGE_SELF).ru_maxrss / 1024.0, 1))}
            atomic_json(dict(rec=rec, curve=curve), f"{jd}/fold_{tag}.json")  # JSON FIRST
            tmp = fp + ".tmp.npz"
            np.savez_compressed(tmp[:-4], repr=R.astype(np.float32),
                                head=S.astype(np.float32), head_ema=S_ema.astype(np.float32),
                                probe_oof=poof, y_true=y_true, y_used=y_use,
                                tr=np.asarray(tr), te=np.asarray(te))
            zz = np.load(tmp); assert np.isfinite(zz["repr"]).all(); os.replace(tmp, fp)
            print(f"DONE {uid} {tag} head {rec['head']['auc']:.4f} "
                  f"probe {rec['probe']['auc']:.4f} val {info['best_val_auc']:.4f} "
                  f"ep{info['best_epoch']} steps {info['total_steps']} "
                  f"mv {info['movement_max']:.3f} clip {info['clip_rate']:.2f} "
                  f"{info['verdict']} {rec['wall_s']}s", flush=True)
        except Exception as e:                                   # J3: never abort
            rec = {**u, **dict(status="FAILED", unit=uid, branch=branch, fold=tag,
                       error=repr(e), traceback=traceback.format_exc(),
                       node=socket.gethostname(), wall_s=round(time.time() - t0, 1))}
            atomic_json(dict(rec=rec, curve=[]), f"{jd}/fold_{tag}.json")
            print(f"FAILED {uid} {tag}: {e}", flush=True)
        done += 1
        write_status(jd, "running", done, len(all_folds))
        if _STOP["flag"]:
            write_status(jd, "requeued", done, len(all_folds)); stop.set(); sys.exit(0)
    stop.set()
    write_status(jd, "done", done, len(all_folds),
                 dict(wall_s=round(time.time() - t_unit, 1)))
    open(f"{jd}/UNIT.done", "w").write("done")
    print(f"UNIT_COMPLETE {uid} {time.time()-t_unit:.0f}s", flush=True)

if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
