"""S15 collector: ONE command assembles every CSV, plot and report from whatever
finished. Asserts cache-hash identity across all rows (Section 2.5)."""
import sys, os, json, glob, hashlib, numpy as np, pandas as pd
sys.path.insert(0, "/users/3171356m/A-GCL/audit/s15/scripts"); import s15_data as DAT
S15 = DAT.S15
REF = [("LinearSVC 4005 FC edges", 0.7565, 0.7432), ("ridge logistic", 0.7561, 0.7406),
       ("BNT S13 winner", 0.6583, 0.6619), ("WGIN S12A5 arm A", 0.6307, float("nan")),
       ("RANDOM WGIN S12A3 (untrained watermark)", 0.6539, float("nan"))]

def rows():
    R, E = [], []
    for f in sorted(glob.glob(f"{S15}jobs/*/fold_*.json")):
        j = json.load(open(f)); r = j["rec"]
        if r.get("status") == "FAILED":
            R.append(dict(status="FAILED", unit=r["unit"], branch=r.get("branch"),
                          arm=r.get("arm"), fold=r.get("fold"), error=r.get("error")))
            continue
        base = {k: r.get(k) for k in ("unit","branch","arm","arch","loss","lr","wd",
            "K_or_hidden","seed","fold","fold_protocol","control","mode","leakage",
            "status","n_params","repr_dim","best_epoch","epochs_run","total_steps",
            "best_val_auc","train_val_gap","verdict","movement_max","flag_untrained",
            "clip_rate","flag_clip_high","adaptive_clip_final","flag_best_epoch_1",
            "integrity_loss_decreased","ocread_entropy","ema_delta","h_fc","h_labels",
            "h_folds_lab","cache_file","node","wall_s","peak_rss_mb","ckpt_sha")}
        for g, v in (r.get("movement") or {}).items(): base[f"movement_{g}"] = v
        for pt in ("head", "head_ema", "probe"):
            if pt in r and isinstance(r[pt], dict):
                R.append({**base, "eval_point": pt, **r[pt]})
        for c in j.get("curve", []): E.append(dict(unit=r["unit"], fold=r["fold"], **c))
    return pd.DataFrame(R), pd.DataFrame(E)

def main():
    df, ep = rows()
    if len(df) == 0: print("no results yet"); return
    df.to_csv(S15 + "s15_results.csv", index=False)
    if len(ep): ep.to_csv(S15 + "s15_epochs.csv", index=False)
    ok = df[df.status == "OK"] if "status" in df else df
    # Section 2.5: identical cache hashes across every row, or the wave is invalid
    bad = []
    for c in ("h_fc", "h_labels", "h_folds_lab", "cache_file"):
        u = ok[c].dropna().unique()
        if len(u) > 1: bad.append((c, list(u)))
    print(f"rows {len(df)} | OK {len(ok)} | FAILED {int((df.status=='FAILED').sum())}")
    if bad:
        print("*** WAVE INVALID: differing cache hashes ***", bad); return
    print("cache-hash identity: OK (single value across every row)")
    pr = ok[(ok.eval_point == "probe") & (~ok.leakage.fillna(False))]
    print("\n=== PROBE AUC by arm x fold protocol (mean over folds and seeds) ===")
    for proto in ("lab", "site", "loso"):
        s = pr[pr.fold_protocol == proto]
        if not len(s): continue
        print(f"\n-- F-{proto.upper()} --")
        g = s.groupby(["branch", "arm", "K_or_hidden"]).auc.agg(["mean", "std", "count"])
        print(g.sort_values("mean", ascending=False).head(12).round(4).to_string())
    print("\n=== FROZEN REFERENCE LINES ===")
    for n, a, l in REF: print(f"  {n:<42} F-LAB {a:.4f}  LOSO {l:.4f}")
    print("\n=== VALIDITY GATE ===")
    v = ok[ok.eval_point == "probe"]
    print(f"  folds with movement>0.10 : {int((~v.flag_untrained.fillna(True)).sum())}/{len(v)}")
    print(f"  folds with clip_rate<30% : {int((~v.flag_clip_high.fillna(True)).sum())}/{len(v)}")
    print(f"  median movement_max      : {v.movement_max.median():.4f}  (S13: 0.016-0.039)")
    print(f"  median clip_rate         : {v.clip_rate.median():.4f}  (S13: 0.92)")
    print(f"  median total_steps       : {v.total_steps.median():.0f}  (S13: ~500)")

if __name__ == "__main__":
    main()
