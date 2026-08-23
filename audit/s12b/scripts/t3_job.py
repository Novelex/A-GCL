"""S12B Track 3 — ALFF normalization check (CPU). Gate-1 probe on frozen folds.
Documented: production abideDataset v3 = per-subject PER-BAND min-max
(x.min(dim=0)); frozen audit M1_B = per-subject JOINT min-max (bitwise == S5
cache). Both delete between-subject amplitude."""
import sys, os, time, numpy as np
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12b/scripts"); import s12b_core as B

NJ = int(os.environ.get("S11_NJOBS", "4"))

def perband_minmax(M):
    """[954,90,3] raw -> production v3 formula, VERBATIM: abideDataset.py does
    torch.where(span > 0, (x-x_min)/span, x) — degenerate bands keep RAW values,
    they do not become zeros (review R6)."""
    mn = M.min(1, keepdims=True); mx = M.max(1, keepdims=True)
    span = mx - mn
    out = np.where(span > 0, (M - mn) / np.where(span > 0, span, 1.0), M)
    return out.astype(np.float64)

def main():
    jp = B.S12B + "jobs/t3.json"
    if os.path.exists(jp): print("skip"); return
    d = B.load_all(); y = d["y"]; folds = B.folds_all(y)
    Mraw = d["M1raw"]; Mjoint = d["M1B"].astype(np.float64); Xfc = d["X_fc"].astype(np.float64)
    Rf = Mraw.reshape(954, -1)
    # genuine across-subject z-score (review R8/R17): falsifiable invariance check,
    # not a copy of raw. Degenerate columns (sd==0) pass through unscaled.
    mu = Rf.mean(0, keepdims=True); sd = Rf.std(0, keepdims=True)
    Zs = (Rf - mu) / np.where(sd > 0, sd, 1.0)
    n_degenerate = int((sd <= 0).sum())
    arms = {
        "raw_270": Rf,
        "joint_minmax_270(frozen M1_B)": Mjoint.reshape(954, -1),
        "perband_minmax_270(production v3)": perband_minmax(Mraw).reshape(954, -1),
        "zscore_across_subjects_270(invariance check vs raw)": Zs,
        "alff_raw+FC_4275": np.concatenate([Rf, Xfc], 1),
        "FC_4005(=R1)": Xfc,
    }
    print(f"degenerate (sd==0) ALFF columns: {n_degenerate}/270", flush=True)
    res, t0 = {}, time.time()
    for name, X in arms.items():
        ev = B.StageEval(y, XFC=None, conf=None, keep_proj=False)
        for t, tr, te in folds:
            ev.fold_fit(X, tr, te, t, njobs=NJ)
        res[name] = ev.finalize()
        print(name, res[name]["pooled_ordinary"]["auc"], flush=True)
    inv = abs(res["raw_270"]["pooled_ordinary"]["auc"] -
              res["zscore_across_subjects_270(invariance check vs raw)"]
              ["pooled_ordinary"]["auc"])
    B.atomic_json(dict(arms=res, wall_s=round(time.time() - t0, 1),
                       invariance_raw_vs_zscore_absdiff=float(inv),
                       degenerate_columns=n_degenerate,
                       note="abideDataset.py lines 71-74: per-band min-max (verbatim,"
                            " incl. span==0 -> keep raw); frozen M1_B: joint min-max"
                            " over 90x3 (S12A1 assert)",
                       provenance=B.provenance()), jp)
    print("DONE t3", flush=True)

if __name__ == "__main__":
    main()
