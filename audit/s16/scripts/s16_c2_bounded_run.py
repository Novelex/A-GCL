"""S16 BOUNDED C2 RUNNER. Executes the site x label matched retrospective estimator.

NOT RUN during the correction pass. Writes ONLY new outputs; never overwrites the
superseded historical reports."""
import sys, os, json, glob, time, numpy as np
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")

# RUNTIME OUTPUT DIR (defect D36). These were written to audit/s16/out/, which is
# NOT gitignored, so a C2 run dirtied the worktree — and a dirty tree is exactly
# what the provenance guard refuses. Runtime products belong under runs/, which is
# ignored; only curated documents are ever committed.
OUT = "/users/3171356m/A-GCL/audit/s16/runs/c2/"
def _out(name):
    os.makedirs(OUT, exist_ok=True); return OUT + name
def _atomic(obj, name):
    """TEMP -> validate -> rename, so a killed run never leaves a half-written file."""
    p_ = _out(name); t_ = p_ + ".tmp"
    with open(t_, "w") as fh: json.dump(obj, fh, indent=1, default=str)
    json.load(open(t_)); os.replace(t_, p_); return p_
import s16_c2_bounded as CB, s16_data as DAT
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
S16 = DAT.S16

# --- FROZEN AGGREGATION DEFINITION, declared before any execution -----------------
PRIMARY = "pooled_oof_per_seed"
PRIMARY_DEF = ("PRIMARY QUANTITY = ONE POOLED OUT-OF-FOLD AUC DIFFERENCE PER SEED. "
               "For a given seed, honest and biased out-of-fold scores are accumulated "
               "across ALL ordinary folds and BOTH half directions, then a single AUC "
               "is computed for each and subtracted. Fold-level values are preserved as "
               "SUPPORTING output only. This choice is frozen here and is not switched "
               "after seeing results.")

SOURCES = [
 ("RANDOM WGIN (S12A5 A repr0, epoch-0) [CALIBRATION]",
  "/users/3171356m/agcl_audit_s0/s12a5/feat/A_s0_o*.npz","y","repr0",32),
 ("trained WGIN (S12A4 arm1 h)",
  "/users/3171356m/agcl_audit_s0/s12a4/feat/a1_s0_o*.npz","y","h",32),
 ("S12A5 arm A (WGIN)","/users/3171356m/agcl_audit_s0/s12a5/feat/A_s0_o*.npz","y","repr",32),
 ("S12A5 arm B (WGIN+edge skip)","/users/3171356m/agcl_audit_s0/s12a5/feat/B_s0_o*.npz","y","repr",64),
 ("S12A5 arm C (edge MLP)","/users/3171356m/agcl_audit_s0/s12a5/feat/C_s0_o*.npz","y","repr",32),
 ("S13 BNT winner (K=2 wd1e-4)",
  "/users/3171356m/A-GCL/audit/s13/feat/T2_K2_wd0.0001_s0__o*.npz","y_true","repr",256),
 ("S15 B1 BNT K=32 (terminated)",
  "/users/3171356m/A-GCL/audit/s15/feat/main_B1_BNT_kh32_L-BCE_lr0.0003_wd0.001_s0__lab*.npz",
  "y_true","repr",4096),
]

def validate_source(name, pat, ykey, rkey, exp_dim, y_ref, ids_ref):
    """Validate ONE source. The caller validates ALL of them before ANY fitting.

    Defect D42: sources were validated and fitted in the same loop, so sources one
    to six were already fitted by the time source seven was found to be invalid.
    A run could therefore burn hours and then report a partial source set, which
    invites choosing the estimate after seeing it."""
    fs = sorted(glob.glob(pat)); prob=[]; no_ids=[]
    if not fs: return None, [f"{name}: no saved folds match {pat}"], None
    folds=[]
    for f in fs:
        z=np.load(f)
        if rkey not in z.files: prob.append(f"{f}: representation key {rkey!r} absent"); continue
        R=z[rkey]
        if R.shape[0]!=954: prob.append(f"{f}: {R.shape[0]} subjects != 954")
        if R.shape[1]!=exp_dim: prob.append(f"{f}: repr dim {R.shape[1]} != expected {exp_dim}")
        yy=z[ykey].astype(np.int64)
        if not np.array_equal(yy,y_ref): prob.append(f"{f}: labels differ from the frozen cohort")
        # ids_ref is USED, not decorative (defect D43): when a source records its
        # subject order, it must equal the frozen cohort order exactly. Equal labels
        # are NOT sufficient — two different orderings can share a label vector.
        idk = next((k for k in ("ids","subject_ids","subjects") if k in z.files), None)
        if idk is not None:
            try: got=[str(x) for x in z[idk].tolist()]
            except Exception as e:
                prob.append(f"{f}: subject-id array unreadable ({e.__class__.__name__}); "
                            f"an id array that cannot be read cannot be verified")
                got=None
        if idk is not None and got is not None:
            if got != [str(x) for x in ids_ref]:
                prob.append(f"{f}: subject order differs from the frozen cohort "
                            f"(first mismatch at "
                            f"{next((i for i,(a,b) in enumerate(zip(got,ids_ref)) if str(a)!=str(b)), 'length')})")
        if idk is None:
            no_ids.append(os.path.basename(f))
        # EXACT PARTITION of 0..953 (defect D44). Disjointness plus a total of 954
        # does not prove a partition: a duplicate inside tr can offset a missing
        # subject, so a subject could be scored twice and another never scored.
        tr,te=z["tr"],z["te"]
        tl,el=tr.tolist(),te.tolist()
        if len(set(tl))!=len(tl): prob.append(f"{f}: duplicate indices inside tr")
        if len(set(el))!=len(el): prob.append(f"{f}: duplicate indices inside te")
        if set(tl)&set(el): prob.append(f"{f}: tr and te overlap")
        if set(tl)|set(el) != set(range(954)):
            miss=sorted(set(range(954))-(set(tl)|set(el)))
            prob.append(f"{f}: tr|te is not an exact partition of 0..953 "
                        f"({len(miss)} subjects never appear, e.g. {miss[:5]})")
        folds.append((f,R,yy,tr,te))
    tags=[os.path.basename(f) for f,_,_,_,_ in folds]
    if len(set(tags))!=len(tags): prob.append(f"{name}: duplicate fold files")
    if len(folds)!=5: prob.append(f"{name}: {len(folds)} ordinary folds, expected 5")

    # ---- CROSS-FOLD STRUCTURE (defect D47). Every fold above was validated in
    # ISOLATION, so five files each holding the SAME te passed: each one is a legal
    # partition on its own. The consequence is severe and silent — the same 154
    # subjects are "held out" five times and 800 subjects are never scored once,
    # while the source reports five valid folds.
    sigs = [tuple(sorted(int(x) for x in te)) for _,_,_,_,te in folds]
    if len(set(sigs)) != len(sigs):
        from collections import Counter
        dup = [n for n,c in Counter(sigs).items() if c>1]
        prob.append(f"{name}: the 5 folds do NOT have 5 distinct test sets "
                    f"({len(set(sigs))} distinct); {len(dup)} test set(s) repeat")
    if len(folds)==5:
        allte=[set(g) for g in sigs]
        for i in range(5):
            for j in range(i+1,5):
                ov=allte[i]&allte[j]
                if ov:
                    prob.append(f"{name}: folds {i} and {j} overlap in {len(ov)} test "
                                f"subjects (e.g. {sorted(ov)[:5]}) - test sets must be "
                                f"pairwise disjoint")
                    break
            else: continue
            break
        uni=set().union(*allte)
        if uni != set(range(954)):
            miss=sorted(set(range(954))-uni); extra=sorted(uni-set(range(954)))
            prob.append(f"{name}: the union of the 5 test sets is not exactly 0..953 "
                        f"({len(miss)} subjects never tested e.g. {miss[:5]}, "
                        f"{len(extra)} out of range)")
        for k,(f_,_,_,tr_,te_) in enumerate(folds):
            if set(int(x) for x in tr_) != set(range(954)) - allte[k]:
                prob.append(f"{name}: fold {k} tr is not exactly the complement of its te")
                break
    if no_ids:
        print(f"NOTE {name}: {len(no_ids)} fold file(s) carry no subject-id array; "
              f"order is verified by label vector only", flush=True)
    return folds, prob, sigs

def run_source(name, folds, sites, y):
    per_seed=[]; ledger=[]
    for seed in CB.SEEDS:
        rng=np.random.default_rng(seed)
        oh=np.full(len(y),np.nan); ob=np.full(len(y),np.nan); feasible=True
        for f,R,yy,tr,te in folds:
            for te_a,te_b in CB.half_split(te,y,seed):
                sel=CB.matched_draw(te_a,tr,y,sites,rng)
                if sel is None: feasible=False; break
                _,o1=K.probe_pipe(np.asarray(R,float),y,[(te_a,te_b)],[])
                _,o2=K.probe_pipe(np.asarray(R,float),y,[(sel, te_b)],[])
                oh[te_b]=o1[te_b]; ob[te_b]=o2[te_b]
                ledger.append(dict(source=name,seed=int(seed),fold=os.path.basename(f),
                    n_honest_train=int(len(te_a)),n_biased_train=int(len(sel)),
                    n_scored=int(len(te_b)),
                    biased_train_subjects=sorted(int(x) for x in sel)))
            if not feasible: break
        if not feasible: per_seed.append(None); continue
        from sklearn.metrics import roc_auc_score
        m=np.isfinite(oh)&np.isfinite(ob)
        h=float(roc_auc_score(y[m],oh[m])); b=float(roc_auc_score(y[m],ob[m]))
        per_seed.append(dict(seed=int(seed),honest=h,biased=b,paired_difference=b-h,
                             n_scored=int(m.sum())))
    diffs=[p["paired_difference"] if p else None for p in per_seed]
    return dict(source=name, primary=PRIMARY, per_seed=per_seed,
                summary=CB.monte_carlo_summary(diffs)), ledger

def main():
    t0=time.time()
    y, sites, ids = CB.cohort()
    all_res=[]; all_led=[]; problems=[]

    # ---- PHASE 1: validate ALL seven sources. Nothing is fitted in this phase.
    validated=[]; sig_map={}
    for name,pat,ykey,rkey,dim in SOURCES:
        folds, prob, sigs = validate_source(name,pat,ykey,rkey,dim,y,ids)
        if prob: problems.extend(prob)
        validated.append((name, folds, prob))
        if sigs: sig_map[name]=tuple(sorted(sigs))
        print(f"validate {name}: {'OK' if not prob and folds else str(len(prob))+' problems'}",
              flush=True)

    # ---- CANONICAL FOLD MEMBERSHIP ACROSS SOURCES (defect D47). The seven sources
    # are compared to one another, so they must be the SAME five test partitions. If
    # two sources disagree, the paired difference is computed over different subjects
    # and the comparison is meaningless.
    if len(set(sig_map.values())) > 1:
        ref_name = next(iter(sig_map))
        ref = sig_map[ref_name]
        for nm, sg in sig_map.items():
            if sg != ref:
                d = [i for i in range(min(len(sg),len(ref))) if sg[i]!=ref[i]]
                problems.append(f"{nm}: fold membership differs from {ref_name!r} "
                                f"(first differing fold index {d[0] if d else 'n/a'}) - "
                                f"all seven sources must share the SAME five test sets")
    # The halt must consider BOTH per-source problems AND cross-source problems.
    # Looking only at per-source problems let a cross-source fold-membership mismatch
    # through to fitting (found by test_pass3.py while testing the D47 fix itself).
    bad=[n for n,f_,p_ in validated if p_ or f_ is None]
    cross=[p for p in problems if "fold membership differs" in p]
    if cross: bad = bad + [f"<cross-source: {len(cross)} mismatch(es)>"]
    if bad:
        # NOTHING is fitted. A partial source set is not reported, because choosing
        # which sources to believe after seeing which ones failed is a selection.
        for name,f_,p_ in validated:
            all_res.append(dict(source=name,
                status=("SOURCE_VALIDATION_FAILED" if (p_ or f_ is None) else "NOT_RUN"),
                problems=p_))
        _atomic(dict(primary=PRIMARY, primary_definition=PRIMARY_DEF, results=all_res,
                     problems=problems, halted=(
                        f"{len(bad)} validation failure(s) across {len(SOURCES)} "
                        f"sources ({len(cross)} cross-source fold-membership "
                        f"mismatch(es)); NO source was fitted and NO estimate is "
                        f"reported"),
                     wall_s=round(time.time()-t0,1)), "C2_BOUNDED.json")
        print(f"HALTED before fitting: {len(bad)} invalid source(s): {bad}", file=sys.stderr)
        for pr_ in problems[:10]: print("  "+pr_, file=sys.stderr)
        sys.exit(6)

    # ---- PHASE 2: every source is valid; fit them.
    for name, folds, _ in validated:
        res, led = run_source(name, folds, sites, y)
        res["status"]="OK"; all_res.append(res); all_led.extend(led)
        s_=res["summary"]
        print(f"== {name}: mean {s_.get('mean_paired_difference')} "
              f"MCse {s_.get('monte_carlo_se_of_mean')} flips "
              f"{s_.get('sign_flips_descriptive')}/{s_.get('n_seeds')}", flush=True)
        # CALIBRATION GATE: the random encoder runs FIRST and must pass
        if "CALIBRATION" in name:
            v = CB.calibration_verdict(s_.get("mean_paired_difference"))
            print("CALIBRATION:", v["consequence"], flush=True)
            if not v["passed"]:
                _atomic(dict(primary=PRIMARY, primary_definition=PRIMARY_DEF,
                    calibration=v, results=all_res, problems=problems,
                    halted="calibration failed; the other six sources were NOT "
                           "interpreted and are NOT reported"), "C2_BOUNDED.json")
                print("HALTED: retrospective estimates remain UNRESOLVED", file=sys.stderr)
                sys.exit(5)
    a=_atomic(dict(primary=PRIMARY, primary_definition=PRIMARY_DEF, results=all_res,
                   problems=problems, wall_s=round(time.time()-t0,1)), "C2_BOUNDED.json")
    b=_atomic(all_led, "C2_BOUNDED_LEDGER.json")
    print(f"wrote {a} + {b}", flush=True)

if __name__=="__main__": main()
