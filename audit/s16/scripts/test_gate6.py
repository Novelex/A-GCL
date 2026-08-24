"""S16 Gate 6: fusion semantics, tie-breaking, fold baselines, prediction schema."""
import sys, os, json, numpy as np
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_data as DAT, s16_feat as FT
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
from sklearn.metrics import roc_auc_score
F=[]
def ck(n,ok,d=""):
    print(("PASS " if ok else "FAIL ")+n+(" | "+d if d else ""),flush=True)
    if not ok: F.append(n)

TB = lambda inner: max(inner, key=lambda r:(r["auc"], r["alpha"]))["alpha"]

print("--- A. conservative tie-breaking ---")
uniq=[dict(alpha=a,auc=(0.70 if abs(a-0.35)<1e-9 else 0.60)) for a in np.round(np.arange(0,1.001,0.05),4)]
ck("A1_unique_optimum", abs(TB(uniq)-0.35)<1e-9, f"selected {TB(uniq)} (the unique max)")
tie=[dict(alpha=a,auc=0.65) for a in np.round(np.arange(0,1.001,0.05),4)]
ck("A2_complete_tie_picks_FC_end", abs(TB(tie)-1.0)<1e-9,
   f"selected {TB(tie)} — a total tie resolves to alpha=1, the most FC-favouring end")
plateau=[dict(alpha=a,auc=(0.68 if a>=0.60 else 0.55)) for a in np.round(np.arange(0,1.001,0.05),4)]
ck("A3_partial_plateau_picks_largest", abs(TB(plateau)-1.0)<1e-9,
   f"selected {TB(plateau)} — the largest alpha on the plateau [0.60,1.0]")
plat2=[dict(alpha=a,auc=(0.68 if 0.20<=a<=0.40 else 0.55)) for a in np.round(np.arange(0,1.001,0.05),4)]
ck("A4_bounded_plateau_picks_upper_edge", abs(TB(plat2)-0.40)<1e-9,
   f"selected {TB(plat2)} — upper edge of the plateau, not the learned-favouring edge")
old_tb = lambda inner: max(inner, key=lambda r:r["auc"])["alpha"]
ck("A5_old_rule_was_learned_favouring", abs(old_tb(tie)-0.0)<1e-9,
   f"the pre-Gate-6 rule would have picked {old_tb(tie)} on a total tie (defect D7)")

print("\n--- A. alpha=1 endpoint identity, and negative deltas are legal ---")
d,MAN,ent = DAT.load("signed", where="gate6")
y = d["y"].astype(np.int64); tag,tr,te = DAT.folds(d,"lab")[0]
tr_enc,tr_prb = FT.honest_split(tr,y); Xfc,_,_,_ = K.load_Xfc()
R = np.random.default_rng(0).standard_normal((954,32))       # deliberately useless
s_fc,s_le = FT.scores_for_fusion(R,Xfc,y,tr_enc,tr_prb,te)
svm_tr_enc = float(roc_auc_score(y[te], s_fc[te]))
f1 = FT.fuse_scores(s_fc,s_le,1.0,tr_prb)
mu,sd = FT.zfit(s_fc,tr_prb)
ck("A6_endpoint_exists_and_equals_standardised_FC",
   np.array_equal(f1[te], FT.zapply(s_fc,mu,sd)[te]), "fused(alpha=1) == z(s_FC) bitwise")
ck("A7_endpoint_preserves_FC_ranking",
   np.array_equal(np.argsort(f1[te]), np.argsort(s_fc[te])), "identical ordering")
ck("A8_endpoint_auc_equals_fold_svm_tr_enc",
   abs(float(roc_auc_score(y[te],f1[te]))-svm_tr_enc)<1e-12,
   f"{roc_auc_score(y[te],f1[te]):.10f} == {svm_tr_enc:.10f}")
inner=[dict(alpha=float(a),auc=float(roc_auc_score(y[tr_prb],
        FT.fuse_scores(s_fc,s_le,a,tr_prb)[tr_prb]))) for a in FT.ALPHA_GRID]
a_sel=TB(inner); f_sel=FT.fuse_scores(s_fc,s_le,a_sel,tr_prb)
delta=float(roc_auc_score(y[te],f_sel[te]))-svm_tr_enc
ck("A9_negative_delta_is_reportable", np.isfinite(delta),
   f"alpha={a_sel} delta={delta:+.4f} — reported AS MEASURED, never clamped "
   f"({'negative, and that is a result' if delta<0 else 'non-negative here'})")

print("\n--- B. fold-specific baselines ---")
d_full,_ = K.probe_pipe(Xfc.astype(np.float64),y,[(np.asarray(tr),np.asarray(te))],[])
svm_tr_full=float(d_full["auc"]); paired=svm_tr_full-svm_tr_enc
ck("B1_two_baselines_same_test_subjects", np.isfinite(svm_tr_full),
   f"svm_tr_enc {svm_tr_enc:.4f} (n={len(tr_enc)}) vs svm_tr_full {svm_tr_full:.4f} "
   f"(n={len(tr)}) on the SAME {len(te)} test subjects")
ck("B2_paired_size_delta", np.isfinite(paired),
   f"size_delta_paired {paired:+.4f} — the ONLY defensible size estimate")
hist = 0.7565-0.7319
ck("B3_historical_diff_is_not_clean", abs(hist-paired)>1e-9,
   f"historical 0.7565-0.7319 = {hist:+.4f} differs from this fold's paired "
   f"{paired:+.4f}; the historical pair spans different fold designs, training-set "
   f"definitions and code versions and is NOT a clean estimate")

print("\n--- E. estimands stay separate ---")
spec=open("/users/3171356m/A-GCL/audit/s16/AGGREGATION_SPEC.md").read()
ck("E1_spec_forbids_pooling", "NEVER AVERAGED TOGETHER" in spec and "E-LOSO" in spec)
ck("E2_spec_forbids_0p7565_subtraction",
   "Do not subtract 0.7565 from an individual fold or site" in spec)
ck("E3_spec_marks_c7_exploratory", "exploratory" in spec)
ck("E4_a7_bridge_withdrawn", "bridge" in spec.lower() and "withdrawn" in spec.lower())

print(f"\n=== GATE 6 SUMMARY: {len(F)} failures ===")
for f in F: print("  FAIL "+f)
sys.exit(1 if F else 0)
