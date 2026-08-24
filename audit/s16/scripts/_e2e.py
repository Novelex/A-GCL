"""S16 E2E GATE. Runs ONE REAL FOLD OF EVERY ARM THROUGH THE REAL WORKER and then
ASSERTS SUBSTANCE, not merely the absence of an exception. This is the gate whose
absence caused two failed submissions."""
import sys, os, json, glob, shutil, numpy as np
sys.path.insert(0,'/users/3171356m/A-GCL/audit/s16/scripts')
import s16_train as TR, s16_data as DAT
TR.MAX_EPOCHS, TR.MIN_EPOCHS, TR.PATIENCE = 4, 2, 2       # short: correctness only
import s16_worker as W, s16_grid as G, s16_feat as FT
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
S16=DAT.S16
DAT.folds_orig = DAT.folds
DAT.folds = lambda d,p: (DAT.folds_orig(d,p)[:1] if p=='lab' else [])   # ONE fold total

FAILS=[]
def check(name, ok, det=""):
    print(("PASS " if ok else "FAIL ")+name+(" | "+det if det else ""), flush=True)
    if not ok: FAILS.append(f"{name}: {det}")

targets=[]
for arm in ("A1","A3","A4","A5","A6","A7"):
    for mode in ("plain","fused"):
        i=next((k for k,u in enumerate(G.MAIN) if u["arm"]==arm and u["mode"]==mode
                and u["E"]=="signed" and u["seed_idx"]==0), None)
        if i is not None: targets.append(("main",i,f"{arm}-{mode}"))
for c in ("C-RAND","C-PERM","C-SHUF","C-ROI"):
    i=next((k for k,u in enumerate(G.CTRL) if u["control"]==c and u["seed_idx"]==0), None)
    if i is not None: targets.append(("ctrl",i,c))
i=next((k for k,u in enumerate(G.ABL) if u["seed_idx"]==0), None)
if i is not None: targets.append(("abl",i,"ALFF-abl"))

# ---- A7 PARITY: the bridge is only valid if this holds ----
d,MAN,ent = DAT.load("signed", where="e2e")
Xfc,_y,_i,_m = K.load_Xfc()
tri,_ = FT.build_X("edgetri", d["FC"], d["ALFF"], np.arange(700))
check("A7_parity_bitwise_Xfc",
      tri.shape==(954,4005) and np.array_equal(tri.astype(np.float64), Xfc),
      f"A7 input at E=signed is the frozen X_fc BITWISE (shape {tri.shape}); this is "
      f"what makes the C6<->C2 bridge valid")

rows_seen=[]
for branch,idx,label in targets:
    try:
        W.run(branch, idx)
    except SystemExit as e:
        check(f"run_{label}", False, f"worker exited {e.code}"); continue
    except Exception as e:
        check(f"run_{label}", False, repr(e)[:200]); continue
    uid=G.unit_id({"main":G.MAIN,"ctrl":G.CTRL,"abl":G.ABL}[branch][idx])
    fs=sorted(glob.glob(f"{S16}jobs/{uid}/fold_*.json"))
    if not fs: check(f"row_{label}", False, "no results row written"); continue
    rec=json.load(open(fs[0]))["rec"]                     # re-read from disk
    rows_seen.append((label,rec))
    ok=rec.get("status")=="OK"
    check(f"status_{label}", ok, rec.get("error","")[:120] if not ok else "OK")
    if not ok: continue
    for pt in ("probe_honest","probe_old_full","head"):
        a=rec.get(pt,{}).get("auc")
        check(f"{pt}_{label}", a is not None and np.isfinite(a) and 0.0<a<1.0
              and abs(a-0.5)>1e-9, f"AUC {a}")
    sv=rec.get("svm_tr_enc")
    check(f"svm_tr_enc_{label}", sv is not None and np.isfinite(sv) and 0.0<sv<1.0,
          f"{sv}")
    if rec.get("mode")=="fused":
        fu=rec.get("fusion")
        check(f"fusion_present_{label}", isinstance(fu,dict) and "alpha_curve" in fu,
              f"{len(fu.get('alpha_curve',[])) if fu else 0} alpha points")
        if fu:
            check(f"alpha1_exact_{label}", fu["alpha1_equals_svm_tr_enc"] is True,
                  f"|a1_auc {fu['alpha1_auc']:.10f} - svm_tr_enc {sv:.10f}| = "
                  f"{abs(fu['alpha1_auc']-sv):.2e} (need <1e-12)")
            check(f"alpha1_bitwise_{label}", fu["alpha1_bitwise_equals_zsFC"] is True,
                  "fused at alpha=1 is bitwise z(s_FC) on te")
            check(f"fused_auc_{label}", np.isfinite(fu["fused_auc"])
                  and 0.0<fu["fused_auc"]<1.0, f"{fu['fused_auc']:.4f} "
                  f"alpha={fu['alpha_selected']} delta_vs_svm_tr_enc="
                  f"{fu['delta_vs_svm_tr_enc']:+.4f}")

# ---- one full row printed verbatim ----
if rows_seen:
    lbl,rec = next(((l,r) for l,r in rows_seen if r.get("mode")=="fused"), rows_seen[0])
    print("\n=== ONE FULL RESULTS ROW, VERBATIM ("+lbl+") ===", flush=True)
    print(json.dumps(rec, indent=1, default=str)[:6000], flush=True)

print("\n=== E2E SUMMARY ===")
print(f"{len(targets)} arms exercised through the real worker; {len(FAILS)} assertion failures")
for f in FAILS: print("  FAIL "+f)
sys.exit(1 if FAILS else 0)
