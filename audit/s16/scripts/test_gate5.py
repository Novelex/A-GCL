"""S16 Gate 5: preprocessing isolation + C-ROI semantics. No jobs, no results."""
import sys, os, copy, numpy as np
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_data as DAT, s16_feat as FT
F=[]
def ck(n,ok,d=""):
    print(("PASS " if ok else "FAIL ")+n+(" | "+d if d else ""),flush=True)
    if not ok: F.append(n)

d,MAN,ent = DAT.load("signed", where="gate5")
FC,ALFF,y = d["FC"],d["ALFF"],d["y"].astype(np.int64)
tag,tr,te = DAT.folds(d,"lab")[0]
tr_enc,tr_prb = FT.honest_split(tr,y)
print(f"n_tr {len(tr)} n_tr_enc {len(tr_enc)} n_tr_prb {len(tr_prb)} n_te {len(te)}")

# ================= A. SPLIT BEFORE COHORT PREPROCESSING =================
print("\n--- A. preprocessing isolation ---")
A0 = FT.alff_scaled(ALFF, tr_enc, "z")
# perturb ONLY tr_prb rows of the raw input
ALFF_pert = ALFF.copy()
rng = np.random.default_rng(0)
ALFF_pert[np.asarray(tr_prb)] += rng.normal(0, 5.0, size=(len(tr_prb),90,3)).astype(np.float32)
A1 = FT.alff_scaled(ALFF_pert, tr_enc, "z")
# fitted parameters must be identical
mu0 = ALFF.astype(np.float64)[tr_enc].mean(0); sd0 = ALFF.astype(np.float64)[tr_enc].std(0)
mu1 = ALFF_pert.astype(np.float64)[tr_enc].mean(0); sd1 = ALFF_pert.astype(np.float64)[tr_enc].std(0)
ck("A1_z_params_unchanged", np.array_equal(mu0,mu1) and np.array_equal(sd0,sd1),
   f"max|dmu| {np.abs(mu0-mu1).max():.1e} max|dsd| {np.abs(sd0-sd1).max():.1e} "
   f"(tr_prb perturbed by N(0,5))")
ck("A2_transformed_tr_enc_unchanged",
   np.array_equal(A0[np.asarray(tr_enc)], A1[np.asarray(tr_enc)]),
   "z-scored tr_enc rows bitwise identical after perturbing tr_prb")
ck("A3_te_unchanged", np.array_equal(A0[np.asarray(te)], A1[np.asarray(te)]),
   "te rows also unchanged (they depend only on tr_enc-fitted parameters)")
ck("A4_tr_prb_did_change", not np.array_equal(A0[np.asarray(tr_prb)],A1[np.asarray(tr_prb)]),
   "the perturbation IS present in tr_prb — the test perturbs something real")
# per-subject modes estimate no cohort statistics: tr_enc is irrelevant to them
for m in ("raw","joint","perband"):
    a = FT.alff_scaled(ALFF, tr_enc, m); b = FT.alff_scaled(ALFF, te, m)
    ck(f"A5_{m}_independent_of_split", np.array_equal(a,b),
       f"'{m}' gives identical output for tr_enc vs te as the fitting index — it "
       f"estimates NO cohort statistic")
ck("A6_cohort_modes_declared", FT.COHORT_LEVEL_MODES==("z",),
   f"COHORT_LEVEL_MODES={FT.COHORT_LEVEL_MODES}")

# ================= B. C-ROI SEMANTICS =================
print("\n--- B. C-ROI ---")
Xc, FCc = FT.build_X("fcrow+alff", FC, ALFF, tr_enc, control="C-ROI")
Xp, FCp = FT.build_X("fcrow+alff", FC, ALFF, tr_enc, control=None)
ck("B1_symmetry", float(np.abs(FCc-FCc.transpose(0,2,1)).max())<1e-6,
   f"max|FC-FC^T| {float(np.abs(FCc-FCc.transpose(0,2,1)).max()):.1e} after relabeling")
dia = FCc[:, np.arange(90), np.arange(90)]
ck("B2_diagonal", float(np.abs(dia-1.0).max())==0.0,
   "diag(FC)==1.0 exactly preserved (a permutation maps the diagonal to itself)")
# coordinate consistency: node i's profile row must equal the permuted adjacency row i
okc=True
for si in [3, 77, 512, 901]:
    if not np.allclose(Xc[si][:, :90], FCc[si], atol=0): okc=False
ck("B3_feature_adjacency_consistent", okc,
   "X[:, :90] == the relabeled adjacency, bitwise, for every checked subject")
# inverse reconstruction
oki=True
for si in [3, 77, 512, 901]:
    p = FT.roi_perm(si); inv = np.argsort(p)
    if not np.array_equal(FCc[si][inv][:,inv], FC[si]): oki=False
    if not np.array_equal(Xc[si][inv][:, :90][:, inv], Xp[si][:, :90]): oki=False
    if not np.array_equal(Xc[si][inv][:, 90:], Xp[si][:, 90:]): oki=False
ck("B4_inverse_reconstruction", oki,
   "applying p^-1 to adjacency, profile rows+cols and ALFF rows restores the original")
# independent deterministic subject permutations
perms = [FT.roi_perm(si) for si in range(8)]
ck("B5_independent_permutations",
   len({tuple(p) for p in perms})==8 and np.array_equal(FT.roi_perm(5),FT.roi_perm(5)),
   "8 distinct permutations; roi_perm is deterministic on repeat")
# fixed ROI correspondence destroyed
same = sum(1 for si in range(200) if np.array_equal(FT.roi_perm(si), np.arange(90)))
ck("B6_correspondence_destroyed", same==0,
   f"{same}/200 subjects retain the identity permutation — cross-subject ROI "
   f"correspondence is gone")
# ALFF band columns must NOT be permuted
okb=True
for si in [3, 77, 512, 901]:
    p = FT.roi_perm(si)
    if not np.array_equal(Xc[si][:, 90:], Xp[si][p][:, 90:]): okb=False
ck("B7_no_band_permutation", okb,
   "ALFF block = original ALFF with ROWS permuted by p and its 3 BAND COLUMNS untouched")
# alff-only arm: rows permuted, no profile block to permute
Xa,_ = FT.build_X("alff", FC, ALFF, tr_enc, control="C-ROI")
Xa0,_= FT.build_X("alff", FC, ALFF, tr_enc, control=None)
ck("B8_alff_arm_rows_only",
   all(np.array_equal(Xa[si], Xa0[si][FT.roi_perm(si)]) for si in [3,77,512]),
   "spec='alff' has n_profile=0: rows permuted, nothing else")
# A7 edge triangle follows the relabeled adjacency
t_c,_ = FT.build_X("edgetri", FC, ALFF, tr_enc, control="C-ROI")
t_p,_ = FT.build_X("edgetri", FC, ALFF, tr_enc, control=None)
ck("B9_a7_control_now_active", not np.array_equal(t_c,t_p),
   "C-ROI now changes A7's input (defect D6b: it was previously inert)")

print(f"\n=== GATE 5 SUMMARY: {len(F)} failures ===")
for f in F: print("  FAIL "+f)
sys.exit(1 if F else 0)
