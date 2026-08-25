"""S16 BOUNDED C2 RUNNER. Executes the site x label matched retrospective estimator.

NOT RUN during the correction pass. Writes ONLY new outputs; never overwrites the
superseded historical reports."""
import sys, os, json, glob, time, numpy as np
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
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
    """Every saved source is validated BEFORE any fitting."""
    fs = sorted(glob.glob(pat)); prob=[]
    if not fs: return None, [f"{name}: no saved folds match {pat}"]
    folds=[]
    for f in fs:
        z=np.load(f)
        if rkey not in z.files: prob.append(f"{f}: representation key {rkey!r} absent"); continue
        R=z[rkey]
        if R.shape[0]!=954: prob.append(f"{f}: {R.shape[0]} subjects != 954")
        if R.shape[1]!=exp_dim: prob.append(f"{f}: repr dim {R.shape[1]} != expected {exp_dim}")
        yy=z[ykey].astype(np.int64)
        if not np.array_equal(yy,y_ref): prob.append(f"{f}: labels differ from the frozen cohort")
        tr,te=z["tr"],z["te"]
        if len(set(tr.tolist())&set(te.tolist())): prob.append(f"{f}: tr and te overlap")
        if len(tr)+len(te)!=954: prob.append(f"{f}: tr+te = {len(tr)+len(te)} != 954")
        folds.append((f,R,yy,tr,te))
    tags=[os.path.basename(f) for f,_,_,_,_ in folds]
    if len(set(tags))!=len(tags): prob.append(f"{name}: duplicate fold files")
    if len(folds)!=5: prob.append(f"{name}: {len(folds)} ordinary folds, expected 5")
    return folds, prob

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
    for name,pat,ykey,rkey,dim in SOURCES:
        folds, prob = validate_source(name,pat,ykey,rkey,dim,y,ids)
        if prob: problems.extend(prob)
        if prob or folds is None:
            all_res.append(dict(source=name, status="SOURCE_VALIDATION_FAILED",
                                problems=prob)); continue
        res, led = run_source(name, folds, sites, y)
        res["status"]="OK"; all_res.append(res); all_led.extend(led)
        s=res["summary"]
        print(f"== {name}: mean {s.get('mean_paired_difference')} "
              f"MCse {s.get('monte_carlo_se_of_mean')} flips "
              f"{s.get('sign_flips_descriptive')}/{s.get('n_seeds')}", flush=True)
        # CALIBRATION GATE: the random encoder runs FIRST and must pass
        if "CALIBRATION" in name:
            v = CB.calibration_verdict(s.get("mean_paired_difference"))
            print("CALIBRATION:", v["consequence"], flush=True)
            if not v["passed"]:
                json.dump(dict(primary=PRIMARY, primary_definition=PRIMARY_DEF,
                    calibration=v, results=all_res, problems=problems,
                    halted="calibration failed; the other six sources were NOT "
                           "interpreted and are NOT reported"),
                    open(S16+"out/C2_BOUNDED.json","w"), indent=1, default=str)
                print("HALTED: retrospective estimates remain UNRESOLVED", file=sys.stderr)
                sys.exit(5)
    json.dump(dict(primary=PRIMARY, primary_definition=PRIMARY_DEF, results=all_res,
                   problems=problems, wall_s=round(time.time()-t0,1)),
              open(S16+"out/C2_BOUNDED.json","w"), indent=1, default=str)
    json.dump(all_led, open(S16+"out/C2_BOUNDED_LEDGER.json","w"), indent=1, default=str)
    print("wrote C2_BOUNDED.json + C2_BOUNDED_LEDGER.json", flush=True)

if __name__=="__main__": main()
