"""S12B Track 1 job. One SLURM array task = (arm, emb_dim); 30 forward families
(norm x mrelu x seed) probed at stages S2..S6; S0/S1 arm-level in the emb32 task.
NO optimizer imported anywhere. SKIP-keyed; atomic writes; Gate-0 re-verified."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12b/scripts"); import s12b_core as B
from joblib import Parallel, delayed

TASKS = [(a, e) for a in B.ARMS for e in B.EMBS]          # 18
NWORK = int(os.environ.get("S12B_WORKERS", "6"))

def flat(H):  # [954,90,e] -> [954, 90e]
    return H.reshape(len(H), -1)

def stage_mats(H1, H2):
    """Distinct per-family eval matrices from one forward. DECLARED DEDUP:
    S6_nnT == S4 and S6_nnF == S3 as matrices (probing a node stage 'flattened by
    ROI' IS the ROI-aware readout), so S6 results are aliased at consolidation,
    never recomputed."""
    out = {"S2": flat(H1), "S3": flat(H2)}
    dT = B.derived_stages(H2, True); dF = B.derived_stages(H2, False)
    out["S4"] = flat(dT["S4"]); out["S5_nnT"] = dT["S5"]; out["S5_nnF"] = dF["S5"]
    return out

def run_eval(name, get_X, folds, y, XFC, conf, scodes, extra):
    ev = B.StageEval(y, XFC=XFC, conf=conf, keep_proj=True)
    for t, tr, te in folds:
        ev.fold_fit(get_X(t), tr, te, t, njobs=1, site_codes=scodes)
    r = ev.finalize(scodes); r.update(extra)
    return r, ev

def run_family(arm, emb, norm, mrelu, sidx, X, FC, y, XFC, conf, scodes, folds):
    tag = f"{arm}_e{emb}_{norm}_mr{int(mrelu)}_s{sidx}"
    jp = f"{B.S12B}jobs/t1_{tag}.json"
    if os.path.exists(jp): return f"skip {tag}"
    t0 = time.time(); torch.set_num_threads(1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda": torch.cuda.reset_peak_memory_stats()
    seed = B.BASE + sidx; d_in = X.shape[2]
    evs, res = {}, {}
    if norm == "bn":                                       # fold-calibrated running stats
        cache = {}
        def stages_for(t):
            if t not in cache:
                tr = next(f[1] for f in folds if f[0] == t)
                enc = B.S12BEncoder(d_in, emb, "bn", mrelu, seed)
                H1, H2 = B.extract_stages(enc, X, FC, dev, bn_train_idx=tr)
                cache.clear(); cache[t] = stage_mats(H1, H2)
            return cache[t]
        for st in ("S2", "S3", "S4", "S5_nnT", "S5_nnF"):
            evs[st] = B.StageEval(y, XFC=XFC, conf=conf, keep_proj=True)
        for t, tr, te in folds:
            M = stages_for(t)
            for st, ev in evs.items():
                ev.fold_fit(M[st], tr, te, t, njobs=1, site_codes=scodes)
    else:
        enc = B.S12BEncoder(d_in, emb, norm, mrelu, seed)
        H1, H2 = B.extract_stages(enc, X, FC, dev)
        M = stage_mats(H1, H2)
        for st, Xs in M.items():
            evs[st] = B.StageEval(y, XFC=XFC, conf=conf, keep_proj=True)
            for t, tr, te in folds:
                evs[st].fold_fit(Xs, tr, te, t, njobs=1, site_codes=scodes)
    gpu_mb = (torch.cuda.max_memory_allocated() / 1e6) if dev == "cuda" else 0.0
    npz = {}
    for st, ev in evs.items():
        res[st] = ev.finalize(scodes)
        npz[f"{st}__p_ord"] = ev.p_ord; npz[f"{st}__p_loso"] = ev.p_loso
        npz[f"{st}__proj"] = ev.proj
    res["S6_nnT"] = {"alias": "S4"}; res["S6_nnF"] = {"alias": "S3"}  # declared dedup
    cfg = dict(arm=arm, emb=emb, norm=norm, message_relu=bool(mrelu), seed=seed,
               post_bn_relu=True, dropout="inert(eval)", d_in=int(d_in),
               bn_protocol=("per-fold TRAIN-only calibration, momentum=None,"
                            " batch128, manifest order") if norm == "bn" else "n/a")
    fp = f"{B.S12B}feat/t1_{tag}.npz"
    tmp = fp + ".tmp.npz"
    np.savez_compressed(tmp[:-4], **npz)
    zz = np.load(tmp); assert f"S3__p_ord" in zz.files; os.replace(tmp, fp)
    B.atomic_json(dict(unit=tag, cfg=cfg, results=res,
                       wall_s=round(time.time() - t0, 1), gpu_mem_mb=round(gpu_mb, 1),
                       provenance=B.provenance()), jp)
    return f"done {tag} {time.time()-t0:.0f}s"

def run_arm_level(arm, X, FC, y, XFC, conf, scodes, folds):
    """S0 (input) and S1 (A1 = X + FC^T@X, mrelu=F, param-free) — once per arm."""
    outs = []
    for st, Xs in (("S0", flat(X)), ("S1", flat(B.a1_stage(X, FC)))):
        tag = f"{arm}_{st}"
        jp = f"{B.S12B}jobs/t1_{tag}.json"
        if os.path.exists(jp): outs.append(f"skip {tag}"); continue
        t0 = time.time()
        r, ev = run_eval(tag, lambda t: Xs, folds, y, XFC, conf, scodes,
                         dict(stage=st, arm=arm))
        fp = f"{B.S12B}feat/t1_{tag}.npz"; tmp = fp + ".tmp.npz"
        np.savez_compressed(tmp[:-4], p_ord=ev.p_ord, p_loso=ev.p_loso, proj=ev.proj)
        zz = np.load(tmp); assert "p_ord" in zz.files; os.replace(tmp, fp)
        B.atomic_json(dict(unit=tag, cfg=dict(arm=arm, stage=st, d=int(Xs.shape[1])),
                           results={st: r}, wall_s=round(time.time() - t0, 1),
                           provenance=B.provenance()), jp)
        outs.append(f"done {tag}")
    return outs

def main(task):
    arm, emb = TASKS[task]
    d = B.load_all()                                       # Gate-0 re-verification
    y = d["y"]; FC = d["FC"]
    X = B.arm_features(arm, d["M1B"], FC)
    XFC = d["X_fc"].astype(np.float64)
    conf = B.confound_dict(d); scodes, sites = B.site_codes(d)
    folds = B.folds_all(y)
    print(f"task {task}: arm={arm} emb={emb} d_in={X.shape[2]} folds={len(folds)}",
          flush=True)
    if emb == 32:
        for m in run_arm_level(arm, X, FC, y, XFC, conf, scodes, folds):
            print(m, flush=True)
    fams = [(norm, mr, s) for norm in B.NORMS for mr in (True, False)
            for s in range(len(B.SEEDS))]
    msgs = Parallel(n_jobs=NWORK, backend="loky", verbose=5)(
        delayed(run_family)(arm, emb, norm, mr, s, X, FC, y, XFC, conf, scodes, folds)
        for norm, mr, s in fams)
    for m in msgs: print(m, flush=True)
    open(f"{B.S12B}jobs/T1_{arm}_e{emb}.done", "w").write("done")

if __name__ == "__main__":
    main(int(sys.argv[1]))
