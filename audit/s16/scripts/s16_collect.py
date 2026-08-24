"""S16 collector. REFUSES to write any results file if the wave is contaminated."""
import sys, os, json, glob, numpy as np, pandas as pd
S16="/users/3171356m/A-GCL/audit/s16/"

def guard():
    """No results are written if ANY unit is poisoned or carries a failure."""
    problems=[]
    if os.path.exists(S16+"jobs/POISON"):
        problems.append("GLOBAL POISON MARKER: "+open(S16+"jobs/POISON").read().strip())
    for t in sorted(glob.glob(S16+"jobs/*/POISON")):
        problems.append("unit poison: "+open(t).read().strip())
    tot_a=tot_f=0
    for t in sorted(glob.glob(S16+"jobs/*/TALLY.json")):
        d=json.load(open(t)); tot_a+=d["attempted"]; tot_f+=d["failed"]
        if d["failed"]>0:
            problems.append(f"unit {d['unit']}: {d['failed']}/{d['attempted']} folds FAILED")
    return problems, tot_a, tot_f

def main():
    problems, tot_a, tot_f = guard()
    if problems:
        print("*** REFUSING TO WRITE RESULTS — THE WAVE IS CONTAMINATED ***", file=sys.stderr)
        print(f"total folds attempted {tot_a}, failed {tot_f}", file=sys.stderr)
        for p in problems[:25]: print("  "+p, file=sys.stderr)
        if len(problems)>25: print(f"  ... and {len(problems)-25} more", file=sys.stderr)
        print("\nFix the cause and re-run. No CSV, no report, no plots were produced.",
              file=sys.stderr)
        sys.exit(3)
    R=[]
    for f in sorted(glob.glob(S16+"jobs/*/fold_*.json")):
        r=json.load(open(f))["rec"]
        if r.get("status")!="OK": continue
        base={k:r.get(k) for k in ("unit","branch","arm","E","arch","mode","seed","fold",
              "fold_protocol","control","alff_mode","kh","svm_tr_enc","n_tr","n_tr_enc",
              "n_tr_probe","movement_max","clip_rate","verdict","best_epoch","total_steps",
              "repr_dim_used","sparse","h_fc","cache_file","node","wall_s")}
        for g,v in (r.get("movement") or {}).items(): base["movement_"+g]=v
        for pt in ("head","probe_honest","probe_old_full"):
            if isinstance(r.get(pt),dict):
                R.append({**base,"eval_point":pt,**r[pt]})
        fu=r.get("fusion")
        if fu:
            R.append({**base,"eval_point":"fused","auc":fu["fused_auc"],
                      "alpha_selected":fu["alpha_selected"],"stack_auc":fu["stack_auc"],
                      "delta_vs_svm_tr_enc":fu["delta_vs_svm_tr_enc"],
                      "delta_vs_0p7565_SECONDARY":fu["delta_vs_0p7565_SECONDARY"],
                      "alpha1_ok":fu["alpha1_equals_svm_tr_enc"],
                      "alpha1_bitwise":fu["alpha1_bitwise_equals_zsFC"]})
    if not R: print("no OK folds yet"); return
    df=pd.DataFrame(R); df.to_csv(S16+"s16_results.csv",index=False)
    print(f"CLEAN: {tot_a} folds attempted, 0 failed. rows {len(df)} -> s16_results.csv")
    bad=[c for c in ("h_fc","cache_file") if df[c].dropna().nunique()>len(df.E.unique())]
    if bad: print("*** cache-hash inconsistency:",bad, file=sys.stderr); sys.exit(3)
    fu=df[df.eval_point=="fused"]
    if len(fu) and not fu.alpha1_ok.all():
        print("*** alpha=1 assertion FAILED on some folds ***", file=sys.stderr); sys.exit(3)
    print("\n=== delta vs svm_tr_enc (headline), by arm x E ===")
    if len(fu):
        print(fu.pivot_table(index="arm",columns="E",values="delta_vs_svm_tr_enc",
                             aggfunc="mean").round(4).to_string())
if __name__=="__main__": main()
