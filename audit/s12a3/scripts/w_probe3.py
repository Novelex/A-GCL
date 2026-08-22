"""S12A3 probe worker: one unit = (arm, seed_idx, kind) + plumbing sentinel.
Flatten ROI features -> S11 harness (pooled OOF AUC). NO global_add_pool."""
import sys, os, json, time, numpy as np
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a3/scripts"); import s12a3_core as M
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
import s11_core as K, s7_core as C7

UNITS = ([(a, s, k) for a in "ABCDEF" for s in (0, 1, 2) for k in ("ord", "loso")]
         + [("plumb", 0, "ord")])

def run(arm, sidx, kind):
    tag = f"{arm}_s{sidx}_{kind}" if arm != "plumb" else "plumb"
    fp = f"{M.S12A3}out/unit_{tag}.json"
    if os.path.exists(fp): print("DONE(skip)", tag); return
    t0 = time.time()
    df, X_fc, y, ids, gh = A1.load_gate()
    if arm == "plumb":
        d, _ = K.probe_pipe(X_fc, y, K.folds_ordinary(), [])
        res = dict(auc=d["auc"], expected=0.7565, delta=abs(d["auc"] - 0.7565)); meta = {}
    else:
        z = np.load(M.emb_path(arm, sidx))
        flat = z["flat"].astype(np.float64)
        assert np.array_equal(z["y"], y) and flat.shape[0] == 954
        folds = K.folds_ordinary() if kind == "ord" else K.folds_loso(y)
        d, _ = K.probe_pipe(flat, y, folds, [])
        res = {k: d[k] for k in ("auc", "bacc", "sens", "spec", "fold_auc",
                                 "fold_sd", "ci_lo", "ci_hi")}
        meta = dict(cfg=json.loads(str(z["cfg"])), seed=int(z["seed"]),
                    sd_hash=str(z["sd_hash"]), flat_dim=flat.shape[1],
                    flat_sha=M.arr_sha(flat.astype(np.float32)))
    out = dict(unit=tag, arm=arm, kind=kind, result=res, meta=meta,
               wall_s=round(time.time() - t0, 1),
               provenance=C7.provenance({"unit": f"S12A3_{tag}"}))
    tmp = fp + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1, default=str)
    assert json.load(open(tmp))["result"]; os.replace(tmp, fp)
    print("DONE", tag, round(res["auc"], 4), f"{out['wall_s']}s")

if __name__ == "__main__":
    a, s, k = UNITS[int(sys.argv[1])]
    run(a, s, k)
