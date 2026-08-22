"""S12A5 Wave-1 job: one unit = (arm in A/B/C, seed_idx). 24 frozen folds.
Atomic order (S12A4b lesson): fold JSON before npz resume marker. Trainable-only
parameter movement recorded in-job."""
import sys, os, json, time, socket, hashlib, numpy as np, torch
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a5/scripts"); import s12a5_core as M
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
import s11_core as K, s7_core as C7
from sklearn.metrics import roc_auc_score

UNITS = [(a, s) for a in "ABC" for s in (0, 1, 2)]
SKIP = ("running_mean", "running_var", "num_batches_tracked")

def movement(init, fin):
    out = {}
    for grp, pref in (("encoder", ("roi.gim.encoder.",)),
                      ("readout", ("roi.readout.",)),
                      ("edge", ("edge.",)), ("head", ("head.",))):
        di, df = [], []
        for k in fin:
            if not any(k.startswith(p) for p in pref): continue
            if any(s in k for s in SKIP) or not torch.is_floating_point(fin[k]): continue
            di.append(init[k].float().flatten()); df.append(fin[k].float().flatten())
        if di:
            i0, f0 = torch.cat(di), torch.cat(df)
            out[grp] = float((f0 - i0).norm() / i0.norm().clamp(min=1e-12))
    return out

def run(arm, sidx, device):
    seed = M.BASE + sidx; tagj = f"{arm}_s{sidx}"
    df, X_fc, y, ids, gh = A1.load_gate()
    FC, Xold, stats = A1.load_tensors(df)
    I90 = np.eye(90, dtype=np.float32)
    Xid = np.concatenate([Xold, np.repeat(I90[None], 954, 0)], axis=2)
    Xe = X_fc.astype(np.float32)                       # canonical frozen edge order
    F = [("o%d" % i, tr, te) for i, (tr, te) in enumerate(K.folds_ordinary())]
    F += [("l%d" % i, tr, te) for i, (tr, te) in enumerate(K.folds_loso(y))]
    for tag, tr, te in F:
        fp = f"{M.S12A5}feat/{tagj}_{tag}.npz"
        if os.path.exists(fp): print("skip", tagj, tag, flush=True); continue
        t0 = time.time()
        model, curve, info, init_state, (R0, S0) = M.train_fold5(
            arm, seed, np.asarray(tr), Xid, FC, Xe, y, device, log=f"{tagj}/{tag}")
        R, S = M.extract5(model, Xid, FC, Xe, np.arange(954), device)
        d, o = K.probe_pipe(R.astype(np.float64), y,
                            [(np.asarray(tr), np.asarray(te))], [])
        mv = movement(init_state, {k: v.cpu() for k, v in model.state_dict().items()})
        ck = f"{M.S12A5}ckpt/{tagj}_{tag}.pt"
        torch.save(model.state_dict(), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        rec = dict(unit=f"{tagj}_{tag}", arm=arm, seed=seed, fold=tag,
                   fold_auc_head=(float(roc_auc_score(y[te], S[te]))
                                  if len(np.unique(y[te])) > 1 else None),
                   fold_auc_repr=d["fold_auc"][0] if d["fold_auc"] else None,
                   best_val_auc=info["best_val_auc"], best_epoch=info["best_epoch"],
                   epochs_run=info["epochs_run"], movement=mv,
                   ckpt_sha=hashlib.sha256(open(ck, "rb").read()).hexdigest()[:16],
                   hostname=socket.gethostname(),
                   gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                   wall_s=round(time.time() - t0, 1), git="c0dd9f2",
                   cfg=dict(arm=arm, lr=M.LR, wd=M.WD, batch=M.BATCH,
                            enc_drop=0.3 if arm in ("A", "B") else None,
                            edge_drop=0.3 if arm in ("B", "C") else None,
                            clf_drop=0.0, max_epochs=M.MAX_EPOCHS, patience=M.PATIENCE))
        jp = f"{M.S12A5}jobs/fold_{tagj}_{tag}.json"
        json.dump(dict(rec=rec, curve=curve,
                       provenance=C7.provenance({"unit": rec["unit"]})),
                  open(jp + ".tmp", "w"), indent=1, default=str)
        assert json.load(open(jp + ".tmp"))["rec"]; os.replace(jp + ".tmp", jp)
        tmp = fp + ".tmp.npz"
        np.savez_compressed(tmp[:-4], repr=R.astype(np.float32),
                            repr0=R0.astype(np.float32), head=S.astype(np.float32),
                            head0=S0.astype(np.float32), oof_repr=o, y=y,
                            tr=np.asarray(tr), te=np.asarray(te))
        zz = np.load(tmp); assert np.isfinite(zz["repr"]).all(); os.replace(tmp, fp)
        print("DONE", tagj, tag, "head", rec["fold_auc_head"], "repr",
              rec["fold_auc_repr"], f"{rec['wall_s']}s", flush=True)
    open(f"{M.S12A5}jobs/JOB_w1_{tagj}.done", "w").write("done")

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a, s = UNITS[int(sys.argv[1])]
    run(a, s, dev)
