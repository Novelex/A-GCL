# S5.5 — CLASSICAL INPUT CALIBRATION BEFORE ANY GNN (evidence only)
2026-08-19 | baseline commit 8cac2358ff12bcfa7452c38c4f4ef5e058814289 | tree CLEAN
No A-GCL training, no graph-cache modification, no ComBat, no data_dense_v3.pt.

## SETUP
Cohort: 954 (455 ASD / 499 NC), 90 ROIs. Cohort sha256 asserted at build time ==
aca3d945f7d89ccb1cc6fb46ca01f6036ccf036d81d120d2b09fe4bd0d1d68c9 (S1/S3C frozen).
Splits: the EXACT S3C frozen splits reused (splits.json sha256
28fed44dc4666066cc0621f329392e58050b39d5ef1371ec5327830518d98916), subject order asserted
identical to S3C's X_sources.npz ordering before any feature was built.
Node branches B / C / D each computed directly from frozen RAW M1 (never chained).
FC primary representation: unique UPPER-TRIANGLE OFF-DIAGONAL edges, 90*89/2 = 4005.
No diagonal (constant 1.0) and no duplicated (u,v)/(v,u) pairs.
LEAKAGE CONTROL: every pipeline is Pipeline([StandardScaler, clf]) inside GridSearchCV, so
the population-level scaler AND C are refit on each inner training fold only. Threshold is
fixed at decision_function > 0 and never tuned. No test fold influences any fitted quantity.

## ORDINARY NESTED 5-FOLD CV
  featset       clf     n_feat  AUC      foldSD   bacc    acc     sens    spec    F1      prec
  FC_only       linsvm    4005  0.7565   0.0229  0.6907  0.6908  0.6901  0.6914  0.6804  0.6709
  FC_only       logreg    4005  0.7561   0.0238  0.6921  0.6939  0.6527  0.7315  0.6704  0.6891
  M1B_only      linsvm     270  0.6286   0.0213  0.5884  0.5901  0.5516  0.6253  0.5622  0.5731
  M1B_only      logreg     270  0.6278   0.0212  0.5957  0.5996  0.5121  0.6794  0.5495  0.5929
  M1C_only      linsvm     270  0.6250   0.0171  0.5965  0.5985  0.5516  0.6413  0.5672  0.5837
  M1C_only      logreg     270  0.6322   0.0164  0.6111  0.6153  0.5209  0.7014  0.5636  0.6140
  M1D_only      linsvm     270  0.6486   0.0197  0.6021  0.6038  0.5648  0.6393  0.5762  0.5881
  M1D_only      logreg     270  0.6497   0.0190  0.6113  0.6164  0.5011  0.7214  0.5547  0.6213
  FC+M1B        linsvm    4275  0.7522   0.0238  0.6838  0.6845  0.6681  0.6994  0.6689  0.6696
  FC+M1B        logreg    4275  0.7523   0.0240  0.6886  0.6908  0.6418  0.7355  0.6644  0.6887
  FC+M1C        linsvm    4275  0.7527   0.0202  0.6851  0.6855  0.6747  0.6954  0.6718  0.6688
  FC+M1C        logreg    4275  0.7530   0.0210  0.6930  0.6950  0.6505  0.7355  0.6704  0.6916
  FC+M1D        linsvm    4275  0.7566   0.0209  0.6872  0.6876  0.6769  0.6974  0.6740  0.6710
  FC+M1D        logreg    4275  0.7566   0.0219  0.6927  0.6950  0.6440  0.7415  0.6682  0.6943
  CTRL_FCfull   linsvm    8010  0.7537   0.0175  0.6919  0.6918  0.6945  0.6894  0.6825  0.6709
  CTRL_FCfull   logreg    8010  0.7533   0.0197  0.6964  0.6981  0.6593  0.7335  0.6757  0.6928

## SYMMETRY CONTROL (explicitly not the primary representation)
  linsvm: 4005 upper-tri 0.7565  vs  8010 full symmetric 0.7537   delta -0.0027
  logreg: 4005 upper-tri 0.7561  vs  8010 full symmetric 0.7533   delta -0.0028
Duplicating the symmetric half changes nothing material (it is exactly redundant
information, and slightly hurts by doubling dimensionality). The 4005 unique
upper-triangle representation is confirmed as the correct primary choice.

## SITE-HELD-OUT (LEAVE-ONE-SITE-OUT, 19 sites)
  featset      best LOSO AUC   ordinary   drop     bacc_loso   site-fold AUC sd
  FC_only         0.7432        0.7565   +0.0133    0.6766        0.0864
  CTRL_FCfull     0.7404        0.7537   +0.0133    0.6790        0.0849
  FC+M1D          0.7397        0.7566   +0.0170    0.6724        0.0707
  FC+M1C          0.7394        0.7530   +0.0133    0.6724        0.0755
  FC+M1B          0.7391        0.7523   +0.0131    0.6723        0.0748
  M1D_only        0.6187        0.6497   +0.0311    0.5903        0.0959
  M1B_only        0.6159        0.6286   +0.0127    0.5802        0.0911
  M1C_only        0.6153        0.6322   +0.0096    0.5829        0.0770
FC is remarkably robust to site hold-out: it loses only 0.013 AUC and remains far above
every ALFF condition. M1D has the LARGEST drop of any condition (+0.031).

## DOES ALFF ADD INFORMATION BEYOND FC?  ANSWER: NO
Paired bootstrap (2000 resamples) on identical out-of-fold predictions.
  ORDINARY CV, vs FC_only:
    linsvm  FC+M1B -0.0043 [-0.0107,+0.0020] p=0.182 ns
            FC+M1C -0.0038 [-0.0104,+0.0028] p=0.260 ns
            FC+M1D +0.0001 [-0.0069,+0.0068] p=0.976 ns
    logreg  FC+M1B -0.0038 [-0.0102,+0.0023] p=0.225 ns
            FC+M1C -0.0031 [-0.0096,+0.0033] p=0.362 ns
            FC+M1D +0.0005 [-0.0063,+0.0072] p=0.881 ns
  SITE-HELD-OUT, vs FC_only:
    linsvm  FC+M1B -0.0041 ns ; FC+M1C -0.0038 ns ; FC+M1D -0.0036 ns
    logreg  FC+M1B -0.0038 ns ; FC+M1C -0.0036 ns ; FC+M1D -0.0034 ns
Every increment is statistically indistinguishable from zero, and under site hold-out ALL
SIX are negative. The best case anywhere is +0.0005.
For contrast, FC vs the best ALFF alone is large and significant:
  FC_only - M1D_only = +0.1080 [+0.0707,+0.1452] (linsvm), +0.1065 [+0.0691,+0.1429] (logreg)

## B vs C vs D
  ORDINARY CV (paired):
    linsvm  D-B +0.0198 [+0.0028,+0.0372] p=0.020 SIG ; D-C +0.0234 [+0.0078,+0.0392] p=0.001 SIG
            C-B -0.0036 [-0.0159,+0.0085] p=0.582 ns
    logreg  D-B +0.0217 [+0.0056,+0.0384] p=0.009 SIG ; D-C +0.0175 [+0.0044,+0.0309] p=0.012 SIG
            C-B +0.0043 [-0.0103,+0.0192] p=0.574 ns
  SITE-HELD-OUT (paired) — the advantage DISAPPEARS:
    linsvm  D-B +0.0027 [-0.0118,+0.0169] ns ; D-C +0.0031 [-0.0068,+0.0141] ns
    logreg  D-B +0.0097 [-0.0031,+0.0232] ns ; D-C +0.0067 [-0.0038,+0.0174] ns
This is the decisive normalization result: D's ordinary-CV superiority (the first
significant normalization difference anywhere in this audit) does NOT survive site
hold-out, and D simultaneously shows the largest ordinary->LOSO drop of any condition
(+0.0311 vs +0.0096 for C and +0.0127 for B). The most parsimonious reading is that part
of D's ordinary-CV edge was site-exploitable. Per the stated rule, no normalization winner
is declared from ordinary-CV AUC.

## NEGATIVE CONTROLS (1920 full nested-CV re-runs, 480 permutations x 4 conditions)
  condition          observed   null mean   null sd   null 97.5%   null max   p
  FC+M1D|linsvm       0.7566     0.4996     0.0245     0.5491      0.5755   0.0021
  FC_only|linsvm      0.7565     0.4990     0.0241     0.5499      0.5811   0.0021
  FC_only|logreg      0.7561     0.4973     0.0235     0.5456      0.5657   0.0021
  M1D_only|linsvm     0.6486     0.5006     0.0250     0.5529      0.5728   0.0021
  overall null AUC mean 0.4992 (sd 0.0243); null balanced accuracy mean 0.4988
  largest AUC produced by ANY of the 1920 null runs = 0.5811, below every observed value.
  p = 0.0021 is the floor for 480 permutations -> no permutation ever reached the observed
  AUC. NO LEAKAGE DETECTED.

## CONFOUND ENCODING (no diagnosis label used; site chance bacc = 0.0526, 19 sites)
  featset     site_bacc  site_acc   r2_TR    r2_T   r2_age  r2_meanFD
  FC_only       0.6356    0.7264   0.2928  0.4996  0.4956    0.4527
  M1B_only      0.5345    0.6111   0.4335  0.5030  0.3098    0.2656
  M1C_only      0.5151    0.5964   0.4264  0.4779  0.3200    0.2581
  M1D_only      0.5261    0.6090   0.4300  0.4828  0.3163    0.2569
  FC+M1D        0.7230    0.8040   0.4814  0.6160  0.5308    0.4568
FC encodes site at 12.1x chance — MORE strongly than any ALFF branch — and also encodes
age (r2 0.50) and head motion (r2 0.45) more strongly. Concatenating FC with ALFF makes
site MORE decodable (0.723) than either alone, i.e. the two carry partly complementary
site information even though ALFF adds no diagnostic information.
IMPORTANT NUANCE: high site encoding did NOT translate into fragility here. FC loses only
0.013 AUC under leave-one-site-out. So FC's site information is largely separable from its
diagnostic information — the opposite of the S3C M2 pattern, where high site encoding came
with the largest LOSO drop.

## PAPER REFERENCE (external only — NOT a target)
Zhang et al. 2023, Table 2, ABIDE I, AAL1 atlas, 5-fold CV, linear SVM row:
  Accuracy 66.37 +/- 3.82 | AUC 64.08 +/- 3.19 | Precision 62.30 +/- 4.83
  Recall 70.57 +/- 2.84   | F1 66.18 +/- 3.45  | Avg 65.90 +/- 3.63
Our leakage-safe FC-only linear SVM: AUC 75.65, Accuracy 69.08, Precision 67.09,
Recall/Sens 69.01, F1 68.04.
WHY THESE ARE NOT COMPARABLE, explicitly:
 1. Cohort: ours is a frozen 954 (455/499) derived from a 956-subject C-PAC download; the
    paper's ABIDE I sample is a different size and selection (the repo documents 987/116 as
    the paper's configuration).
 2. Atlas: ours is AAL1 restricted to the 90 cerebrum ROIs (labels < 9001); the paper's
    AAL1 uses all 116 (cerebrum + cerebellum + vermis). Different feature space entirely
    (4005 vs 6670 unique edges).
 3. Preprocessing: our FC comes from C-PAC filt_noglobal rois_aal; the paper's pipeline is
    fMRIPrep with subject-native AAL registration (S3A, original README).
 4. The paper does not state the SVM competitor's exact input representation, so it is not
    certain their SVM row is FC-only.
 5. Their AUC is reported as a 5-fold mean +/- SD; ours is a pooled out-of-fold AUC from
    nested CV with all hyperparameters chosen inside training folds.
The comparison is recorded for orientation only. Our number being higher is NOT evidence of
a better method and MUST NOT be read as reproducing or beating the paper.

## DECISION-RELEVANT SUMMARY
FC-only diagnostic regime : AUC 0.7565 (ordinary), 0.7432 (site-held-out), bacc 0.691/0.677,
                            fold SD 0.023, permutation p 0.0021. Robust and by far the
                            strongest classical signal available.
M1-only regime            : AUC 0.629-0.650 (ordinary), 0.615-0.619 (LOSO), bacc 0.588-0.611.
                            Real but weak; ~0.11 AUC below FC, a significant gap.
Does M1 add beyond FC     : NO. Every FC+ALFF increment is non-significant, and all six
                            site-held-out increments are negative. Best case +0.0005.
B vs C vs D               : D > B and D > C significantly under ordinary CV, but NOT under
                            site hold-out, where all differences vanish and D has the
                            largest drop. B vs C is indistinguishable everywhere.
                            NO WINNER DECLARED.
Best leakage-safe classical condition : FC_only with a linear SVM (4005 upper-triangle
                            edges, scaler inside the pipeline). It is the highest on both
                            ordinary and site-held-out AUC and needs no ALFF at all.
Site robustness           : FC drop +0.013, ALFF drops +0.010 to +0.031. FC is the most
                            site-robust despite encoding site most strongly.
Negative control          : PASS. 1920 nulls centre on 0.4992; max null 0.5811 < every
                            observed AUC; p = 0.0021 (floor) for all four tested conditions.

## COMPUTE
partition gpu-h100, CPU cores only (no GPU requested — sklearn linear models are CPU-bound).
  1869101 s55-cv    16 tasks x 4 CPU  COMPLETED 16/16    0.56 CPU-h
  1869102 s55-loso  16 tasks x 4 CPU  COMPLETED 16/16    2.38 CPU-h
  1869103 s55-perm  48 tasks x 4 CPU  COMPLETED 48/48   86.43 CPU-h
  1869109 s55-conf   1 task  x 8 CPU  COMPLETED          0.47 CPU-h
  total 89.84 CPU-hours, 81/81 tasks COMPLETED, 0 failures. seed 20260818 throughout.
outputs (outside the repo): /users/3171356m/agcl_audit_s0/s55/
  features.npz, s55_cv_all.csv, s55_cv_loso_merged.csv, s55_delta.txt, s55_loso_summary.txt,
  s55_perm_all.csv, s55_perm_summary.csv, s55_confound.csv, cv/ loso/ perm/ (fold predictions)

## UNRESOLVED / LIMITATIONS
1. ALFF adding nothing beyond FC is established only for LINEAR models on concatenated
   features. It does NOT prove ALFF is useless inside a GNN, where node features and edges
   interact non-linearly through message passing. That is precisely what S6 would test, and
   this result is the honest baseline it must beat.
2. FC's strong site encoding (12.1x chance) is not currently harmful under LOSO, but it is a
   standing risk for any model with more capacity than a linear SVM.
3. The 4 all-positive-FC subjects flagged in S4 were retained (no additional exclusions, as
   instructed); their influence was not separately quantified.
4. Nodes and edges still derive from different C-PAC strategies (M1 nofilt, FC filt).
5. Permutations covered 4 of 16 conditions (480 each). The untested conditions are close
   relatives of tested ones, but their nulls were not measured directly.

S5.5 STATUS: EVIDENCE COMPLETE — no normalization winner declared, S6 not begun.
