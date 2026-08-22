"""S12A4 winner validation: label-permutation control for arm 1 (CE), seed 20260818.
Global permutation via RNG(BASE). Frozen folds; full per-fold pipeline rerun.
Expected pooled OOF AUC ~0.50; PASS if <= 0.55 (pre-registered)."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a4/scripts"); import s12a4_core as M
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
import s11_core as K, s7_core as C7
from sklearn.metrics import roc_auc_score

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df, X_fc, y, ids, gh = A1.load_gate()
    FC, Xold, stats = A1.load_tensors(df)
    I90 = np.eye(90, dtype=np.float32)
    Xid = np.concatenate([Xold, np.repeat(I90[None], 954, 0)], axis=2)
    yp = np.random.default_rng(M.BASE).permutation(y)     # global label permutation
    assert (yp == y).mean() < 1.0 and yp.sum() == y.sum()
    oof_head = np.full(954, np.nan); oof_h = np.full(954, np.nan)
    recs = []
    for i, (tr, te) in enumerate(K.folds_ordinary()):
        t0 = time.time()
        model, curve, info = M.train_fold(1, M.BASE, np.asarray(tr), Xid, FC, yp,
                                          dev, log=f"perm/o{i}")
        H, Z, ND = M.extract_all(model, Xid, FC, dev)
        with torch.no_grad():
            sc = model.head(torch.from_numpy(H).to(dev)).squeeze(-1).cpu().numpy()
        oof_head[te] = sc[te]
        d, o = K.probe_pipe(H.astype(np.float64), yp, [(np.asarray(tr), np.asarray(te))], [])
        oof_h[te] = o[te]
        recs.append(dict(fold=i, best_val=info["best_val_auc"],
                         wall_s=round(time.time() - t0, 1)))
        print("perm fold", i, "done", flush=True)
    auc_head = float(roc_auc_score(yp, oof_head))
    auc_h = float(roc_auc_score(yp, oof_h))
    res = dict(perm_auc_head=auc_head, perm_auc_h=auc_h,
               PASS=bool(auc_head <= 0.55 and auc_h <= 0.55),
               permuted_frac_equal=float((yp == y).mean()), folds=recs,
               provenance=C7.provenance({"unit": "S12A4_perm_control"}))
    fp = M.S12A4 + "out/PERMUTATION.json"
    json.dump(res, open(fp + ".tmp", "w"), indent=1, default=str)
    os.replace(fp + ".tmp", fp)
    print("PERMUTATION:", json.dumps(res)[:200])

if __name__ == "__main__":
    main()
