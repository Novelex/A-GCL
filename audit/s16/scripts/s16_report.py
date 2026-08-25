"""S16 POST-C6 REPORTING AND DECISION. Written BEFORE C6 results exist.

Refuses to emit a headline if any required cell, prediction, provenance field or
validity decision is unresolved."""
import sys, os, json, glob, collections, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_prov as P, s16_ledger as L, s16_grid as G
NS=os.environ.get("S16_NS","prod")

PROTOCOLS = ("lab","site","loso")          # THREE SEPARATE ESTIMANDS. NEVER POOLED.
MOVEMENT_MIN, CLIP_MAX, CRAND_MIN = 0.10, 0.30, 0.03
CRAND_REF = {"WGIN":"A4","BNT":"A6"}       # see CRAND_MAPPING.md
EXACT_MATCH = {"A4","A6"}                  # inputs identical to their reference
HISTORICAL_REFERENCE_ONLY = 0.7565         # NEVER subtracted from a fold or site

AGG_RULE = ("Deterministic aggregation: within a protocol, first average the per-fold "
            "value across the folds of a given seed, then report the mean and the "
            "spread ACROSS SEEDS. Fold-level and seed-level dispersions are reported "
            "SEPARATELY and are never collapsed into a single number.")

def refusals(df):
    r=collections.defaultdict(list)
    if df is None or len(df)==0: r["no_results"].append("results CSV absent or empty"); return r
    _,cells,units,tags = L.expected_ledger()
    need = {"svm_tr_enc","svm_tr_full","size_delta_paired","delta_vs_svm_tr_enc",
            "delta_vs_svm_tr_full","alpha_selected","fold_protocol","seed","arm","E",
            "mode","control","alff_mode","eval_point","movement_max","clip_rate"}
    miss = need - set(df.columns)
    if miss: r["missing_columns"].append(sorted(miss))
    got = {(u,f) for u,f in zip(df.unit, df.fold)}
    if got != cells:
        r["incomplete_ledger"].append(f"{len(cells-got)} cells absent, "
                                      f"{len(got-cells)} unexpected")
    for pr in glob.glob(P.jobs_dir(NS)+"*/fold_*.json"):
        uid=os.path.basename(os.path.dirname(pr))
        tag=os.path.basename(pr)[5:-5]
        if not os.path.exists(P.feat_dir(NS)+f"{uid}__{tag}.pred.json"):
            r["missing_prediction"].append(f"{uid}/{tag}")
    fu=df[df.eval_point=="fused"]
    if len(fu) and "alpha1_equals_svm_tr_enc" in fu.columns and not fu.alpha1_equals_svm_tr_enc.fillna(False).all():
        r["alpha1_identity_failed"].append("at least one fused cell")
    return r

def validity(df):
    """Per (arm,E,mode) validity decision. Returns a table and any unresolved cases."""
    out=[]; unresolved=[]
    pr = df[(df.eval_point=="probe_honest") & (df.control.isna())]
    rand = df[(df.control=="C-RAND") & (df.eval_point=="probe_honest")]
    rand_by_arch = rand.groupby("arch").auc.mean().to_dict()
    for (arm,E,mode),g in pr.groupby(["arm","E","mode"]):
        arch=g.arch.iloc[0]
        mv=float(g.movement_max.mean()); cl=float(g.clip_rate.mean()); auc=float(g.auc.mean())
        ref=rand_by_arch.get(arch)
        if arch=="EDGEMLP":
            crand="NO REFERENCE IN GRID"; crand_ok=None
            note="DESCRIPTIVE ONLY — A7 cannot satisfy C-RAND (see CRAND_MAPPING.md)"
        else:
            crand=(auc-ref) if ref is not None else None
            crand_ok=(crand is not None and crand>=CRAND_MIN)
            note=("exact reference" if arm in EXACT_MATCH else "INPUT-MISMATCHED REFERENCE")
            if ref is None: unresolved.append(f"{arm}/{E}/{mode}: no C-RAND value for {arch}")
        verdict = ("UNTRAINED" if (mv<=MOVEMENT_MIN or cl>=CLIP_MAX or crand_ok is False)
                   else ("INTERPRETABLE" if crand_ok else "DESCRIPTIVE"))
        out.append(dict(arm=arm,E=E,mode=mode,arch=arch,movement_max=mv,clip_rate=cl,
                        crand_delta=crand,crand_reference=CRAND_REF.get(arch,"none"),
                        verdict=verdict,note=note))
    return pd.DataFrame(out), unresolved

def summarise(df):
    """THREE separate estimands. Never averaged together."""
    blocks={}
    for proto in PROTOCOLS:
        s=df[(df.fold_protocol==proto)&(df.eval_point=="fused")]
        if not len(s): continue
        per_seed=s.groupby(["arm","E","mode","seed"]).delta_vs_svm_tr_enc.mean().reset_index()
        agg=per_seed.groupby(["arm","E","mode"]).delta_vs_svm_tr_enc.agg(
            mean="mean", sd_across_seeds="std", n_seeds="count").reset_index()
        fold_sd=s.groupby(["arm","E","mode"]).delta_vs_svm_tr_enc.std().rename("sd_across_folds")
        blocks[proto]=agg.merge(fold_sd,on=["arm","E","mode"],how="left")
    return blocks

def shift_vs_signed(df):
    """Evaluated SEPARATELY for BNT and WGIN, per the addendum: the identity is exact
    for BNT (affine absorbed by Linear) and only approximate for WGIN (the constant
    term varies per subject, so LayerNorm does not absorb it)."""
    out=[]
    for arch in ("BNT","WGIN"):
        s=df[(df.arch==arch)&(df.eval_point=="probe_honest")&(df.control.isna())]
        for proto in PROTOCOLS:
            q=s[s.fold_protocol==proto]
            a=q[q.E=="signed"].auc.mean(); b=q[q.E=="shift"].auc.mean()
            if np.isfinite(a) and np.isfinite(b):
                out.append(dict(arch=arch,protocol=proto,signed=a,shift=b,diff=b-a,
                    expectation=("EXACT: Linear(D,H) can represent the affine map"
                        if arch=="BNT" else
                        "APPROXIMATE: per-subject constant not absorbed by LayerNorm"),
                    within_0p01=bool(abs(b-a)<=0.01)))
    return pd.DataFrame(out)

def main():
    path=P.results_path(NS)
    df=pd.read_csv(path) if os.path.exists(path) else None
    r=refusals(df)
    if r:
        print("*** NO HEADLINE. Unresolved items: ***", file=sys.stderr)
        for k,v in r.items(): print(f"  [{k}] {v[:3]}", file=sys.stderr)
        sys.exit(3)
    vt, unres = validity(df)
    if unres:
        print("*** NO HEADLINE: unresolved validity decisions ***", file=sys.stderr)
        for u in unres[:10]: print("  "+u, file=sys.stderr)
        sys.exit(3)
    print(AGG_RULE); print()
    for proto,b in summarise(df).items():
        print(f"=== ESTIMAND E-{proto.upper()} (never pooled with the others) ===")
        print(b.round(4).to_string(index=False)); print()
    print("=== VALIDITY ==="); print(vt.round(4).to_string(index=False)); print()
    print("=== C-PERM (must sit near 0.50) ===")
    cp=df[(df.control=="C-PERM")&(df.eval_point=="probe_honest")]
    if len(cp): print(cp.groupby(["arch","fold_protocol"]).auc.agg(["mean","std","count"]).round(4).to_string())
    print("\n=== shift vs signed, BY ARCHITECTURE ===")
    print(shift_vs_signed(df).round(4).to_string(index=False))
    print(f"\nHISTORICAL REFERENCE {HISTORICAL_REFERENCE_ONLY} is a labelled line only; "
          f"it is NEVER subtracted from a fold or site.")
    print("Negative fused deltas are reported unchanged. Any C7 candidate selected "
          "from these results is EXPLORATORY.")

if __name__=="__main__": main()
