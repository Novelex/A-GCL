"""S16 POST-C6 REPORTING AND DECISION. Written BEFORE C6 results exist.

Refuses to emit a headline if any required cell, prediction, provenance field or
validity decision is unresolved."""
import sys, os, json, glob, collections, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_prov as P, s16_ledger as L, s16_grid as G
NS=os.environ.get("S16_NS","prod")

PROTOCOLS = ("lab","site","loso")          # THREE SEPARATE ESTIMANDS. NEVER POOLED.
MOVEMENT_MIN, CLIP_MAX, CRAND_MIN = 0.10, 0.30, 0.03

# ---- PRE-REGISTERED DECISION BANDS (declared 2026-08-25, BEFORE any C6 result
# exists; no C6 run has ever produced a result). Defect D40: C-PERM and the
# shift-vs-signed identity were PRINTED as diagnostics and never enforced, so a
# broken control or a violated identity could not stop the report.
CPERM_BAND    = (0.45, 0.55)   # protocol-level mean AUC under permuted labels
# SHIFT_BNT_TOL IS WITHDRAWN AS A SCIENTIFIC GATE (defect D49). Affine
# representational equivalence — Linear(D,H) can express x -> (x+1)/2 — says the two
# PARAMETERISATIONS are equivalent. It does NOT say two INDEPENDENTLY initialised and
# independently optimised models must reach nearly the same AUC. Measured seed-to-seed
# noise on identical data at a fixed E level is ~0.08 AUC, EIGHT TIMES the old +/-0.01
# tolerance, so the gate could not distinguish a broken identity from ordinary
# optimisation variance and would have failed a CORRECT run. The claim is now tested
# deterministically by transporting the first affine layer
# (W' = 2W, b' = b - W.1) in test_pass3.py, and the trained AUC difference is reported
# as a DESCRIPTIVE diagnostic only.
SHIFT_EXCLUDE_ARMS = ("A3",)   # A3's node features ARE the FC rows, so E rewrites
                               # the inputs themselves rather than only the graph
                               # weights; the affine-absorption argument does not
                               # apply and A3 is excluded from the identity check.
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
    cells,units,tags = L.expected_ledger()   # 3 values, not 4 (defect D37)
    need = {"svm_tr_enc","svm_tr_full","size_delta_paired","delta_vs_svm_tr_enc",
            "delta_vs_svm_tr_full","alpha_selected","fold_protocol","seed","arm","E",
            "mode","control","alff_mode","eval_point","movement_max","clip_rate"}
    miss = need - set(df.columns)
    if miss: r["missing_columns"].append(sorted(miss))
    got = {(u,f) for u,f in zip(df.unit, df.fold)}
    if got != cells:
        r["incomplete_ledger"].append(f"{len(cells-got)} cells absent, "
                                      f"{len(got-cells)} unexpected")
    # PER-CELL EVALUATION-POINT SET (defect D55). Presence of (unit, fold) SOMEWHERE
    # in the CSV proved nothing: a cell missing three of its four evaluation rows,
    # or carrying a duplicate or an unexpected one, satisfied the old check. The
    # exact set is now verified independently of the collector.
    mode_of = {uid: u["mode"] for uid, _, u in units}
    if "eval_point" not in df.columns:
        r["eval_points_missing_column"].append("results CSV has no eval_point column")
    else:
        for (uid, fold), g in df.groupby(["unit", "fold"]):
            want = P.expected_eval_points(mode_of.get(uid, "plain"))
            pts = list(g.eval_point)
            have = set(pts)
            dup = sorted({p_ for p_ in pts if pts.count(p_) > 1})
            if dup:
                r["duplicate_eval_point"].append(f"{uid}/{fold}: {dup}")
            if have != want:
                r["eval_point_set_wrong"].append(
                    f"{uid}/{fold}: missing {sorted(want-have)}, "
                    f"unexpected {sorted(have-want)}")
    for pr in glob.glob(P.jobs_dir(NS)+"*/fold_*.json"):
        uid=os.path.basename(os.path.dirname(pr))
        tag=os.path.basename(pr)[5:-5]
        if not os.path.exists(P.feat_dir(NS)+f"{uid}__{tag}.pred.json"):
            r["missing_prediction"].append(f"{uid}/{tag}")
    fu=df[df.eval_point=="fused"]
    if len(fu) and "alpha1_equals_svm_tr_enc" in fu.columns and not fu.alpha1_equals_svm_tr_enc.fillna(False).all():
        r["alpha1_identity_failed"].append("at least one fused cell")
    return r

def _seed_then_mean(g, col="auc"):
    """The pre-registered aggregation: average across the folds of a seed FIRST,
    then across seeds. Never a single pooled fold-average."""
    per_seed = g.groupby("seed")[col].mean()
    return float(per_seed.mean()), int(per_seed.size)

def validity(df):
    """Per (arm, E, mode, fold_protocol) validity decision — ONE VERDICT PER
    ESTIMAND.

    Defect D38: this grouped by (arm,E,mode) only and compared against a C-RAND
    reference pooled over all three protocols. lab / site / loso are three separate
    estimands that are never pooled (AGGREGATION_SPEC.md), and LOSO runs ~0.044
    lower than the ordinary protocols, so a pooled reference silently imported that
    protocol shift into every verdict.

    Pass rule, stated explicitly:
      * decision level  : (arm, E, mode, fold_protocol)
      * statistic       : per-seed fold-mean, then mean across seeds
      * C-RAND reference: same arch AND same fold_protocol, same aggregation
      * INTERPRETABLE   : movement_max > 0.10 AND clip_rate < 0.30 AND
                          (auc - crand) >= 0.03
      * UNTRAINED       : movement_max <= 0.10 OR clip_rate >= 0.30 OR crand < 0.03
      * DESCRIPTIVE     : no admissible C-RAND reference (A7, or arch absent)
    """
    out=[]; unresolved=[]
    pr = df[(df.eval_point=="probe_honest") & (df.control.isna())]
    rand = df[(df.control=="C-RAND") & (df.eval_point=="probe_honest")]
    rand_ref = {}
    for (arch, proto), g in rand.groupby(["arch","fold_protocol"]):
        rand_ref[(arch, proto)] = _seed_then_mean(g)[0]
    for (arm,E,mode,proto),g in pr.groupby(["arm","E","mode","fold_protocol"]):
        arch=g.arch.iloc[0]
        auc, n_seeds = _seed_then_mean(g)
        mv=float(g.groupby("seed").movement_max.mean().mean())
        cl=float(g.groupby("seed").clip_rate.mean().mean())
        ref=rand_ref.get((arch, proto))
        if arch=="EDGEMLP":
            crand="NO REFERENCE IN GRID"; crand_ok=None
            note="DESCRIPTIVE ONLY — A7 cannot satisfy C-RAND (see CRAND_MAPPING.md)"
        else:
            crand=(auc-ref) if ref is not None else None
            crand_ok=(crand is not None and crand>=CRAND_MIN)
            note=("exact reference" if arm in EXACT_MATCH else "INPUT-MISMATCHED REFERENCE")
            if ref is None:
                unresolved.append(f"{arm}/{E}/{mode}/{proto}: no C-RAND value for "
                                  f"{arch} within protocol {proto!r}")
        verdict = ("UNTRAINED" if (mv<=MOVEMENT_MIN or cl>=CLIP_MAX or crand_ok is False)
                   else ("INTERPRETABLE" if crand_ok else "DESCRIPTIVE"))
        out.append(dict(arm=arm,E=E,mode=mode,fold_protocol=proto,arch=arch,
                        auc=auc,n_seeds=n_seeds,movement_max=mv,clip_rate=cl,
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

# alff_mode is part of the key (found in Pass 4): without it the ABLATION units
# (A1 with alff-raw / alff-joint) collide with the main A1 units on
# (arch, arm, mode, seed, fold), producing a many-to-one merge — 351 "pairs" out of
# 270 shift cells, and a negative unpaired count.
PAIR_KEY = ["arch","arm","mode","seed","alff_mode","fold_protocol","fold"]
UNIT_KEY = ["arch","arm","mode","seed","alff_mode"]

def shift_vs_signed(df):
    """PAIRED shift-vs-signed identity, over the domain the GRID actually pairs.

    D39: this once averaged the signed and shift sets separately and subtracted the
    two means, which mixes the identity with whatever composition differed between
    them. The identity is a within-cell claim and is tested within the cell.

    D54 (refined here): `unpaired` must mean "a cell that SHOULD have had a partner
    and does not" — never "the signed set is larger than the shift set". Only 30 of
    the 69 signed unit-keys have a shift twin in the frozen grid: controls and
    ablations exist at E=signed ONLY, by design. Counting those as unpaired would
    make the gate fire on every correct run, which is the same over-strictness that
    made the withdrawn +/-0.01 AUC gate unusable. The pairing DOMAIN is therefore the
    set of unit-keys present at BOTH E levels; within that domain every fold must
    pair, and a missing fold-level partner is a real defect."""
    s_=df[(df.eval_point=="probe_honest")&(df.control.isna())
          &(~df.arm.isin(SHIFT_EXCLUDE_ARMS))]
    lhs=s_[s_.E=="signed"][PAIR_KEY+["auc"]].rename(columns={"auc":"signed"})
    rhs=s_[s_.E=="shift" ][PAIR_KEY+["auc"]].rename(columns={"auc":"shift"})
    if not len(lhs) or not len(rhs):
        return pd.DataFrame(), ["no shift/signed pairs exist — identity UNTESTED"]
    # the domain: unit-keys observed at BOTH levels
    lk=set(map(tuple, lhs[UNIT_KEY].drop_duplicates().values.tolist()))
    rk=set(map(tuple, rhs[UNIT_KEY].drop_duplicates().values.tolist()))
    domain=lk & rk
    def in_domain(d):
        return d[d[UNIT_KEY].apply(tuple, axis=1).isin(domain)]
    L2, R2 = in_domain(lhs), in_domain(rhs)
    m=L2.merge(R2, on=PAIR_KEY, how="inner")
    unpaired=[]
    only_signed = len(L2) - len(m)
    only_shift  = len(R2) - len(m)
    if only_signed or only_shift:
        unpaired.append(f"inside the pairable domain ({len(domain)} unit-keys), "
                        f"{only_signed} signed cell(s) and {only_shift} shift cell(s) "
                        f"have no counterpart ({len(m)} complete pairs)")
    if not len(m):
        return pd.DataFrame(), unpaired or ["no complete shift/signed pairs"]
    m["diff"]=m["shift"]-m["signed"]
    rows=[]
    for (arch,proto),g in m.groupby(["arch","fold_protocol"]):
        per_seed=g.groupby("seed")["diff"].mean()
        rows.append(dict(arch=arch,protocol=proto,n_pairs=int(len(g)),
            mean_signed=float(g["signed"].mean()), mean_shift=float(g["shift"].mean()),
            paired_diff=float(per_seed.mean()),
            max_abs_pair=float(g["diff"].abs().max()),
            expectation=("EXACT: Linear(D,H) can represent the affine map"
                if arch=="BNT" else
                "APPROXIMATE: per-subject constant not absorbed by LayerNorm"),
            interpretation="DESCRIPTIVE ONLY - independently trained models; see "
                           "test_pass4.py for the deterministic affine-transport test"))
    return pd.DataFrame(rows), unpaired

def cperm_gate(df):
    """HARD GATE. Under permuted labels the honest probe must sit at chance.
    Band [0.45, 0.55] on the protocol-level mean, pre-registered above."""
    cp=df[(df.control=="C-PERM")&(df.eval_point=="probe_honest")]
    if not len(cp):
        return pd.DataFrame(), ["C-PERM cells ABSENT — the permutation control was "
                                "not run, so the pipeline is unverified"]
    rows=[]; fails=[]
    lo,hi=CPERM_BAND
    for (arch,proto),g in cp.groupby(["arch","fold_protocol"]):
        mu,n=_seed_then_mean(g)
        ok=(lo<=mu<=hi)
        rows.append(dict(arch=arch,protocol=proto,mean_auc=mu,n_seeds=n,
                         n_cells=int(len(g)),band=f"[{lo},{hi}]",passed=ok))
        if not ok:
            fails.append(
                f"C-PERM {arch}/{proto}: mean AUC {mu:.4f} is outside the predeclared "
                f"operational band [{lo},{hi}]. STOP headline generation and "
                f"investigate permutation variance, "
                f"class balance, "
                f"aggregation, "
                f"optimisation and possible leakage. "
                f"This result alone does NOT prove leakage - it proves that "
                f"investigation is required (defect D50)")
    return pd.DataFrame(rows), fails

def shift_gate(tab, unpaired):
    """PAIR-COMPLETENESS gate (defects D49, D54).

    The AUC MAGNITUDE never decides headline validity — affine equivalence does not
    require two independently trained models to match in AUC, and the deterministic
    proof lives in test_pass4.py / test_pass3.py. What IS mandatory is that the
    comparison be well posed.

    D54: `unpaired` was accepted as a parameter and then ignored, so a table built
    from a partly-unpaired set — signed cells with no shift counterpart, or the
    reverse — reported a clean paired difference computed over whichever cells
    happened to match, and nothing stopped the headline."""
    fails=[]
    if not len(tab):
        fails.append("shift-vs-signed produced no rows - the diagnostic is UNTESTED")
        return fails
    z=tab[tab.n_pairs<=0]
    if len(z):
        fails.append(f"{len(z)} arch x protocol group(s) have zero pairs - the "
                     f"shift/signed comparison is not well posed")
    for u in (unpaired or []):
        fails.append(f"unpaired shift/signed cells: {u}. The comparison is only "
                     f"defined on complete pairs, so the headline is withheld until "
                     f"every signed cell has its shift counterpart and vice versa")
    return fails

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
    gate_failures=[]
    print("=== C-PERM OPERATIONAL GATE (predeclared band "
          f"[{CPERM_BAND[0]}, {CPERM_BAND[1]}]; hold-and-investigate, "
          "not a proof of leakage) ===")
    ct, cfails = cperm_gate(df)
    if len(ct): print(ct.round(4).to_string(index=False))
    gate_failures += cfails

    print("\n=== shift vs signed, PAIRED — DESCRIPTIVE DIAGNOSTIC ONLY ===")
    print("    (affine equivalence is tested deterministically in test_pass3.py;")
    print("     independently trained models are NOT required to match in AUC)")
    st, unpaired = shift_vs_signed(df)
    if len(st): print(st.round(4).to_string(index=False))
    for u in unpaired: print("  NOTE: "+u)
    gate_failures += shift_gate(st, unpaired)

    if gate_failures:
        print("\n*** NO HEADLINE: HARD GATE FAILURE ***", file=sys.stderr)
        for g_ in gate_failures: print("  "+g_, file=sys.stderr)
        sys.exit(4)

    print(f"\nHISTORICAL REFERENCE {HISTORICAL_REFERENCE_ONLY} is a labelled line only; "
          f"it is NEVER subtracted from a fold or site.")
    print("Negative fused deltas are reported unchanged. Any C7 candidate selected "
          "from these results is EXPLORATORY.")

if __name__=="__main__": main()
