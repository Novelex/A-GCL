"""Pass 2, P6-P9: the report must run, decide per protocol, pair the identity,
and STOP on a broken control.

D37 (P6): refusals() unpacked 4 values from expected_ledger(), which returns 3 —
         the report crashed with ValueError on its FIRST statement, so it could
         never have emitted a headline on any input.
D38 (P7): validity grouped by (arm,E,mode) and used a C-RAND reference pooled over
         lab+site+loso, importing the ~0.044 LOSO shift into every verdict.
D39 (P8): shift-vs-signed subtracted two independently-averaged sets instead of
         differencing paired cells.
D40 (P9): C-PERM and the shift identity were printed, never enforced."""
import sys, os, subprocess, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s16_prov as P, s16_ledger as L, s16_grid as G, s16_report as RP
OK=[]
def check(c,m): OK.append(bool(c)); print(f"  [{'PASS' if c else 'FAIL'}] {m}")
HERE=os.path.dirname(os.path.abspath(__file__)); NS="test"

def synth_csv(cperm_auc=0.50, shift_delta=0.0, rng_seed=1):
    """A COMPLETE, ledger-exact results table: every unit, every fold, every
    eval_point the collector emits."""
    rng=np.random.default_rng(rng_seed)
    cells,units,tags=L.expected_ledger(); R=[]
    for uid,br,u in units:
        for tag in tags:
            proto=tag.rstrip("0123456789")
            base=dict(unit=uid,branch=br,arm=u["arm"],E=u["E"],arch=u["arch"],
                mode=u["mode"],seed=G.SEEDS[u["seed_idx"]],fold=tag,
                fold_protocol=proto,control=u.get("control"),
                alff_mode=u.get("alff_mode"),kh=u["kh"],svm_tr_enc=0.7565,
                svm_tr_full=0.7565,size_delta_paired=0.0,movement_max=0.50,
                clip_rate=0.05,ocread_entropy=np.nan,evaluated_state="raw",
                n_tr=763,n_tr_enc=610,n_tr_probe=153)
            ctl=u.get("control")
            # LOSO is a harder estimand for EVERY model, the random encoder
            # included, so the -0.044 protocol shift is applied to C-RAND too.
            # That is the whole point: a within-protocol reference cancels it.
            shift_p = -0.044 if proto=="loso" else 0.0
            if ctl=="C-PERM":   a=cperm_auc            # chance, protocol-invariant
            elif ctl=="C-RAND": a=0.5578 + shift_p
            else:               a=0.62   + shift_p
            # the E-level effect: shift differs from signed by exactly shift_delta
            if ctl is None and u["E"]=="shift": a += shift_delta
            a += rng.normal(0, 1e-4)
            for pt in ("head","head_ema","probe_honest","probe_old_full"):
                R.append({**base,"eval_point":pt,"auc":a,"alpha_selected":None,
                          "alpha1_equals_svm_tr_enc":None,
                          "alpha1_bitwise_equals_zsFC":None,
                          "delta_vs_svm_tr_enc":None,"delta_vs_svm_tr_full":None})
            if u["mode"]=="fused":
                R.append({**base,"eval_point":"fused","auc":0.76,
                    "alpha_selected":1.0,"delta_vs_svm_tr_enc":0.0035,
                    "delta_vs_svm_tr_full":0.0035,"alpha1_equals_svm_tr_enc":True,
                    "alpha1_bitwise_equals_zsFC":True,"delta_is_unclamped":True})
    return pd.DataFrame(R)

def run_report(df):
    os.makedirs(P.root(NS), exist_ok=True)
    df.to_csv(P.results_path(NS), index=False)
    env={**os.environ,"S16_NS":NS,"PYTHONPYCACHEPREFIX":
         "/users/3171356m/agcl_audit_s0/pycache"}
    return subprocess.run([sys.executable, f"{HERE}/s16_report.py"],
                          capture_output=True, text=True, env=env)

print("=== 1. P6: the report RUNS and emits a headline on a complete table ===")
df=synth_csv()
r=run_report(df)
check(r.returncode==0, f"report exits 0 (was ValueError on the unpack); rc={r.returncode}")
check(len(r.stdout) > 500, f"report is nonempty ({len(r.stdout)} chars)")
for want in ("ESTIMAND E-LAB","ESTIMAND E-SITE","ESTIMAND E-LOSO","VALIDITY",
             "C-PERM GATE","shift vs signed, PAIRED"):
    check(want in r.stdout, f"section present: {want}")

print("\n=== 2. P7: validity is decided PER PROTOCOL ===")
vt,unres=RP.validity(df)
check("fold_protocol" in vt.columns, "validity table carries fold_protocol")
check(not unres, f"no unresolved validity decisions ({unres[:2]})")
n_combo=vt.groupby(["arm","E","mode"]).size()
check(bool((n_combo==3).all()), f"every (arm,E,mode) has all 3 protocol rows "
                                f"(min {n_combo.min()}, max {n_combo.max()})")
lo=vt[vt.fold_protocol=="loso"].crand_delta.replace("NO REFERENCE IN GRID",np.nan)
la=vt[vt.fold_protocol=="lab"].crand_delta.replace("NO REFERENCE IN GRID",np.nan)
check(abs(float(pd.to_numeric(lo,errors="coerce").mean())
        - float(pd.to_numeric(la,errors="coerce").mean())) < 1e-3,
      "LOSO uses a LOSO C-RAND reference, so the -0.044 protocol shift cancels "
      "instead of contaminating the verdict")

print("\n=== 3. P8: the identity is PAIRED and A3 is excluded ===")
st,unp=RP.shift_vs_signed(df)
check(len(st)>0, f"paired table built ({len(st)} arch x protocol rows)")
check(bool((st.n_pairs>0).all()), f"every row rests on real pairs "
                                  f"(min {int(st.n_pairs.min())})")
src=open(f"{HERE}/s16_report.py").read()
check('SHIFT_EXCLUDE_ARMS = ("A3",)' in src, "A3 excluded explicitly, with the reason")
keys=["arch","arm","mode","seed","fold_protocol","fold"]
check(all(k in src for k in keys) and 'keys=["arch","arm","mode","seed","fold_protocol","fold"]' in src,
      "pairing key is (arch, arm, mode, seed, fold_protocol, fold)")

print("\n=== 4. P9: C-PERM is a HARD GATE ===")
r2=run_report(synth_csv(cperm_auc=0.62))     # a leaking permutation control
check(r2.returncode==4, f"leaking C-PERM (0.62) exits 4, not 0 (got {r2.returncode})")
check("HARD GATE FAILURE" in r2.stderr, "stderr names the hard-gate failure")
check("permuted labels are being predicted" in r2.stderr,
      f"reason is explicit: {[l.strip() for l in r2.stderr.splitlines() if 'C-PERM' in l][:1]}")
r3=run_report(synth_csv(cperm_auc=0.50))
check(r3.returncode==0, f"C-PERM at exactly 0.500 PASSES (got {r3.returncode}) "
                        f"— an exact 0.5 is the expected outcome, not a failure")

print("\n=== 5. P9: the BNT shift identity is a HARD GATE ===")
r4=run_report(synth_csv(shift_delta=0.05))   # identity violated by 0.05
check(r4.returncode==4, f"violated identity exits 4 (got {r4.returncode})")
check("affine-absorption claim is false" in r4.stderr,
      f"reason is explicit: {[l.strip() for l in r4.stderr.splitlines() if 'BNT' in l][:1]}")
r5=run_report(synth_csv(shift_delta=0.005)) # inside +/-0.01
check(r5.returncode==0, f"a 0.005 difference stays inside the band (got {r5.returncode})")

if os.path.exists(P.results_path(NS)): os.remove(P.results_path(NS))
print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
