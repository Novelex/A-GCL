"""E2E assertions: read every row from disk and check SUBSTANCE."""
import sys, os, json, glob, numpy as np
sys.path.insert(0,'/users/3171356m/A-GCL/audit/s16/scripts')
import s16_data as DAT, s16_feat as FT, s16_grid as G, _e2e_run as R, s16_prov as P
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
S16=DAT.S16; F=[]
def ck(n,ok,d=""):
    print(("PASS " if ok else "FAIL ")+n+(" | "+d if d else ""),flush=True)
    if not ok: F.append(f"{n}: {d}")

d,MAN,ent = DAT.load("signed", where="e2echeck")
Xfc,_,_,_ = K.load_Xfc()
tri,_ = FT.build_X("edgetri", d["FC"], d["ALFF"], np.arange(700))
# S12A5 arm C consumed Xe = X_fc.astype(np.float32) (w_wave1.py:34), so the correct
# parity target is BITWISE AT FLOAT32. Comparing an f32 array upcast to f64 against the
# f64 original can never hold; that was an error in the assertion, not in the data.
_b32 = np.array_equal(tri, Xfc.astype(np.float32))
_d64 = float(np.abs(tri.astype(np.float64)-Xfc).max())
ck("A7_parity_f32_bitwise", tri.shape==(954,4005) and _b32 and _d64 < 3e-8,
   f"A7 input at E=signed == X_fc.astype(float32) BITWISE (what S12A5 arm C consumed); "
   f"f64 residual {_d64:.2e} is pure float32 quantisation — validates the C6<->C2 bridge")

T=R.targets(); rows=[]
for b,i,label in T:
    uid=G.unit_id({"main":G.MAIN,"ctrl":G.CTRL,"abl":G.ABL}[b][i])
    fs=sorted(glob.glob(P.jobs_dir("e2e")+f"{uid}/fold_*.json"))
    if not fs: ck(f"row_{label}",False,"no results row on disk"); continue
    rec=json.load(open(fs[0]))["rec"]; rows.append((label,rec))
    ok=rec.get("status")=="OK"
    ck(f"status_{label}", ok, "OK" if ok else rec.get("error","")[:130])
    if not ok: continue
    for pt in ("probe_honest","probe_old_full","head"):
        a=rec.get(pt,{}).get("auc")
        # AUC EXACTLY 0.5 IS VALID (defect D9). C-PERM on permuted labels can legitimately
        # produce it, and a correct control was previously reported as a gate failure.
        # Require only: present, finite, and within [0,1].
        ck(f"{pt}_{label}", a is not None and np.isfinite(a) and 0.0<=a<=1.0,
           f"AUC {a}")
    sv=rec.get("svm_tr_enc")
    ck(f"svm_tr_enc_{label}", sv is not None and np.isfinite(sv) and 0.0<sv<1.0, f"{sv}")
    if rec.get("mode")=="fused":
        fu=rec.get("fusion")
        ck(f"fusion_{label}", isinstance(fu,dict) and len(fu.get("alpha_curve",[]))==21,
           f"{len(fu.get('alpha_curve',[])) if fu else 0} alpha points")
        if fu:
            ck(f"alpha1_exact_{label}", fu["alpha1_equals_svm_tr_enc"] is True,
               f"|{fu['alpha1_auc']:.10f} - {sv:.10f}| = {abs(fu['alpha1_auc']-sv):.2e} (<1e-12)")
            ck(f"alpha1_bitwise_{label}", fu["alpha1_bitwise_equals_zsFC"] is True, "z(s_FC) on te")
            ck(f"fused_{label}", np.isfinite(fu["fused_auc"]) and 0<fu["fused_auc"]<1,
               f"AUC {fu['fused_auc']:.4f} alpha={fu['alpha_selected']} "
               f"delta_vs_svm_tr_enc={fu['delta_vs_svm_tr_enc']:+.4f}")
if rows:
    lbl,rec=next(((l,r) for l,r in rows if r.get("mode")=="fused"), rows[0])
    print(f"\n=== ONE FULL RESULTS ROW, VERBATIM ({lbl}) ===")
    print(json.dumps(rec,indent=1,default=str)[:6500])
print(f"\n=== E2E SUMMARY: {len(T)} arms, {len(F)} assertion failures ===")
for f in F: print("  FAIL "+f)
sys.exit(1 if F else 0)
