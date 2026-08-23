"""S12B Track 3 — ALFF normalization check (CPU). Gate-1 probe on frozen folds.
Documented: production abideDataset v3 = per-subject PER-BAND min-max
(x.min(dim=0)); frozen audit M1_B = per-subject JOINT min-max (bitwise == S5
cache). Both delete between-subject amplitude."""
import sys, os, time, numpy as np
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12b/scripts"); import s12b_core as B

NJ = int(os.environ.get("S11_NJOBS", "4"))

def perband_minmax(M):  # [954,90,3] raw -> v3 formula
    mn = M.min(1, keepdims=True); mx = M.max(1, keepdims=True)
    span = np.where((mx - mn) > 0, mx - mn, 1.0)
    return ((M - mn) / span).astype(np.float64)

def main():
    jp = B.S12B + "jobs/t3.json"
    if os.path.exists(jp): print("skip"); return
    d = B.load_all(); y = d["y"]; folds = B.folds_all(y)
    Mraw = d["M1raw"]; Mjoint = d["M1B"].astype(np.float64); Xfc = d["X_fc"].astype(np.float64)
    arms = {
        "raw_270": Mraw.reshape(954, -1),
        "joint_minmax_270(frozen M1_B)": Mjoint.reshape(954, -1),
        "perband_minmax_270(production v3)": perband_minmax(Mraw).reshape(954, -1),
        "zscore_across_subjects_270(==raw under probe scaler; invariance check)":
            Mraw.reshape(954, -1).copy(),
        "alff_raw+FC_4275": np.concatenate([Mraw.reshape(954, -1), Xfc], 1),
        "FC_4005(=R1)": Xfc,
    }
    res, t0 = {}, time.time()
    for name, X in arms.items():
        ev = B.StageEval(y, XFC=None, conf=None, keep_proj=False)
        for t, tr, te in folds:
            ev.fold_fit(X, tr, te, t, njobs=NJ)
        res[name] = ev.finalize()
        print(name, res[name]["pooled_ordinary"]["auc"], flush=True)
    B.atomic_json(dict(arms=res, wall_s=round(time.time() - t0, 1),
                       note="abideDataset.py lines 71-74: per-band min-max;"
                            " frozen M1_B: joint min-max over 90x3 (S12A1 assert)",
                       provenance=B.provenance()), jp)
    print("DONE t3", flush=True)

if __name__ == "__main__":
    main()
