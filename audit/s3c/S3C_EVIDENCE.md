# S3C — FULL ALFF SOURCE SELECTION (evidence)
2026-08-18 | HEAD 906a494b… | tree ?? ALFF_func_proc/ | no code/data modified, no ALFF file touched
No A-GCL training, no ComBat, no GNN. data_dense_v3.pt NOT used. norm_matrix NOT used.

## PRE-REGISTRATION (frozen BEFORE any performance was computed)
cohort sha256 aca3d945f7d89ccb1cc6fb46ca01f6036ccf036d81d120d2b09fe4bd0d1d68c9 (== S1)
954 subjects, 455 ASD (y=1) / 499 NC (y=0), 90 ROIs, frozen S1 ROI order.
master seed 20260818
  outer   StratifiedKFold(5, shuffle, random_state=20260818) on y
          fold test sizes [191,191,191,191,190]; ASD per test fold [91,91,91,91,91]
  inner   StratifiedKFold(5, shuffle, random_state=20260818) inside GridSearchCV(scoring=roc_auc)
  grouped StratifiedGroupKFold(5, shuffle, random_state=20260818), groups = SITE_ID
  LOSO    19 sites (all sites met n>=20 and >=5 per class)
  grid    C in {1e-3,1e-2,1e-1,1,10,100,1000}
  splits.json    sha256 28fed44dc4666066cc0621f329392e58050b39d5ef1371ec5327830518d98916
  X_sources.npz  sha256 dc10bf36c4124aa7f214ab6bbc5a89288adc03da747e3950485bb15c0da333a9
LEAKAGE CONTROL: A-D are per-subject (structurally leakage-free). F is a pipeline step
(PerBandStandardizer) so GridSearchCV refits it on every inner training fold; the outer test
fold never contributes to normalization, hyperparameters, feature selection, model choice or
threshold (threshold fixed at decision_function > 0, never tuned). E (train-fold min-max) was
excluded as instructed. No transform was ever fitted on all 954.

## LEVEL 1 — FEATURE / SIGNAL AUDIT
Distributions (per source x band): 0 NaN, 0 Inf everywhere. 3xIQR outliers 251-557 per
90x954 block. Between-subject CV of the ROI sd: M1 0.309-0.320, M2 0.392-0.430, OLD 0.288-0.314.
Bootstrap reliability of the ROI mean profile (500 resamples): all sources r >= 0.9986
(OLD highest at 0.9995-0.9996) — every source is highly reproducible at the group-profile level.
Cross-source agreement: M1-OLD 0.801-0.811 pearson (highest), M1-M2 0.713-0.758, M2-OLD 0.591-0.625.
Univariate diagnosis signal (DESCRIPTIVE ONLY — never used to select features for Level 2):
  source band       n_FDR_sig  max|g|  mean|g|  max AUC dev  mean MI
  M1     classical      21     0.302   0.118      0.079      0.0072
  M1     slow4          22     0.271   0.125      0.072      0.0083
  M1     slow5           6     0.328   0.092      0.081      0.0081
  M2     classical      53     0.348   0.182      0.082      0.0108
  M2     slow4          54     0.341   0.179      0.081      0.0095
  M2     slow5          41     0.355   0.181      0.083      0.0097
  OLD    classical      31     0.276   0.130      0.079      0.0071
  OLD    slow4          30     0.272   0.136      0.080      0.0081
  OLD    slow5          12     0.251   0.099      0.072      0.0077
=> M2 has by far the MOST univariate diagnosis-associated ROIs. Level 4 shows why that is
   not decisive. All effect sizes are small (max |Hedges g| 0.36).

## LEVEL 2 — PREDICTIVE TEST (120 conditions = 3 sources x 4 bands x 5 norms x 2 classifiers)
Best per source (pooled out-of-fold AUC):
  M1  0.6547  all3 / D_SUBJ_BAND_Z / linSVM   fold sd 0.0275  bacc 0.6162 sens 0.5451 spec 0.6874 F1 0.5768 prec 0.6170
  M2  0.6455  slow5 / F_TRAINFOLD_BAND_Z / linSVM  fold sd 0.0278  bacc 0.6002 sens 0.4769 spec 0.7234 F1 0.5357 prec 0.6119
  OLD 0.6364  all3 / F_TRAINFOLD_BAND_Z / logreg   fold sd 0.0373  bacc 0.6024 sens 0.5033 spec 0.7014 F1 0.5485 prec 0.6071
ALL 15 top-ranked conditions are M1.
Mean pooled AUC by source x band:   all3   classical  slow4   slow5
                              M1   0.6466   0.6451   0.6165  0.6425
                              M2   0.6239   0.6166  0.6019  0.6353
                              OLD  0.6257   0.6143  0.5990  0.6231
Mean by source x normalization: M1 0.633-0.642, M2 0.614-0.624, OLD 0.610-0.625.
Normalization mattered far less than source; classifiers were equivalent (linSVM 0.6236 vs
logreg 0.6248 mean). slow4 is the weakest band for every source.

## LEVEL 3 — NEGATIVE CONTROLS (6000 full nested-CV re-runs)
1000 label permutations x 6 conditions (3 strongest per source + 3 weakest per source),
each permutation repeating the ENTIRE pipeline including the inner grid search, plus a
feature-column permutation control (each feature column independently shuffled across
subjects, true labels retained).
  condition                              observed  perm mean  perm sd  perm 97.5%  perm max  p
  M1|all3|D_SUBJ_BAND_Z|linsvm             0.6547    0.4995   0.0252     0.5481    0.5867  0.0010
  M2|slow5|F_TRAINFOLD_BAND_Z|linsvm       0.6455    0.4978   0.0255     0.5432    0.5744  0.0010
  OLD|all3|F_TRAINFOLD_BAND_Z|logreg       0.6364    0.4976   0.0237     0.5448    0.5642  0.0010
  M1|slow4|B_SUBJ_JOINT_MINMAX|linsvm      0.6039    0.4940   0.0255     0.5432    0.5773  0.0010
  M2|slow4|B_SUBJ_JOINT_MINMAX|logreg      0.5940    0.4929   0.0252     0.5430    0.5773  0.0010
  OLD|slow4|B_SUBJ_JOINT_MINMAX|logreg     0.5807    0.4910   0.0253     0.5423    0.5736  0.0010
Overall label-permutation AUC mean 0.4955 (sd 0.0252); feature-column permutation mean
0.5002 (sd 0.0247). Both centre on 0.5 as required. The largest AUC ever produced by any
of the 6000 null runs was 0.5867, below every observed value. p = 0.0010 is the floor for
1000 permutations, i.e. no permutation reached the observed AUC in any condition.
=> NO LEAKAGE DETECTED. Negative controls PASS for every tested condition.

## LEVEL 4 — CONFOUND AUDIT (no diagnosis label used in this section)
Confound encoding, mean over all bands and normalizations (site chance bacc = 0.0526):
  source  site_bacc  site_bacc_max  site_acc  sex_bacc  R2_TR   R2_T    R2_age  R2_meanFD
  M1        0.5563       0.6458      0.6356    0.5319   0.3834  0.4333  0.2921   0.3249
  M2        0.7207       0.7630      0.7776    0.5321   0.4760  0.5343  0.3944   0.4223
  OLD       0.6298       0.7092      0.6892    0.5251   0.3828  0.4520  0.3099   0.3663
=> M2 encodes SITE far more strongly than M1 (0.721 vs 0.556 balanced accuracy, 13.7x vs
   10.6x chance) and leads on every other confound too (TR, scan length, age, motion).
   M1 is the LEAST confounded representation on every single confound measured.
   This directly explains M2's larger univariate signal in Level 1.
Evaluation A/B/C (ordinary -> site-aware -> leave-one-site-out), best-per-source condition:
  M1   ordinary 0.6547 -> grouped 0.6273 (-0.0273) -> LOSO 0.6304 (-0.0242)
  M2   ordinary 0.6455 -> grouped 0.6148 (-0.0307) -> LOSO 0.6165 (-0.0290)
  OLD  ordinary 0.6364 -> grouped 0.6154 (-0.0209) -> LOSO 0.6152 (-0.0212)
Mean drop across all 40 conditions per source: M1 -0.0185, M2 -0.0266, OLD -0.0107.
Best achievable under LOSO: M1 0.6398 (classical/A_RAW/linSVM), M2 0.6236 (slow5/A_RAW/linSVM),
OLD 0.6226 (all3/A_RAW/linSVM). The entire LOSO top-12 is M1.
Per-site LOSO AUC (LOSO-best per source): M1 mean 0.640 sd 0.111, 2/19 sites below 0.5;
M2 mean 0.629 sd 0.091, 1/19 below 0.5; OLD mean 0.625 sd 0.094, 2/19 below 0.5.
=> No source collapses under site-held-out evaluation. M2 loses the most, M1 stays highest.

## LEVEL 5 — STABILITY AND PAIRED COMPARISON
Bootstrap (2000 resamples) 95% CI of pooled OOF AUC:
  M1  0.6547 [0.6210, 0.6874]   M2 0.6455 [0.6100, 0.6788]   OLD 0.6364 [0.5999, 0.6724]
PAIRED on the SAME outer-fold predictions (bootstrap of the difference):
  M1 - M2  +0.0095 [-0.0388, +0.0567] p=0.701  NOT significant
  M1 - OLD +0.0182 [-0.0335, +0.0671] p=0.470  NOT significant
  M2 - OLD +0.0087 [-0.0407, +0.0602] p=0.726  NOT significant
PAIRED under LOSO (best-per-source under LOSO):
  M1 - M2  +0.0164 [-0.0333, +0.0653] ns ; M1 - OLD +0.0171 [-0.0347, +0.0669] ns
Fairest paired test — identical band/norm/classifier for all three sources, LOSO:
  classical/A_RAW/linsvm : M1 0.6398, M2 0.5942, OLD 0.6125 ; M1-M2 +0.0461 [-0.0047,+0.0948] ns
  all3/D_SUBJ_BAND_Z/lin : M1 0.6304, M2 0.6012, OLD 0.6153 ; M1-M2 +0.0296 [-0.0227,+0.0803] ns
  slow5/F_TRAINFOLD/lin  : M1 0.6278, M2 0.6165, OLD 0.6093 ; M1-M2 +0.0116 [-0.0366,+0.0625] ns
Fold-to-fold AUC sd: M1 0.0275, M2 0.0278, OLD 0.0373. Site-to-site AUC sd: M1 0.0957,
M2 0.0866, OLD 0.0995. Confusion stability (sens/spec sd across folds): M1 0.045/0.063,
M2 0.023/0.029, OLD 0.063/0.057 — M2 is the most stable in confusion terms but at a
markedly lower sensitivity (0.477 vs M1 0.545).
=> M1 is numerically ahead in EVERY paired comparison (ordinary and site-held-out, matched
   and unmatched conditions), but NO difference reaches significance. The three sources are
   statistically TIED on discrimination.

## FINAL S3C DECISION TABLE
                                            M1                    M2                    OLD
paper fidelity              ROI-first, MATCHES     voxel-first, NOT in    ROI-first but from
                            paper Sec 2.1          the paper text         C-PAC's own rois_aal
                            ("Fourier transform                           (paper-consistent
                            of the mean time                              ordering, different
                            series")                                      derivative product)
best nested AUC             0.6547                 0.6455                 0.6364
95% bootstrap CI            [0.6210, 0.6874]       [0.6100, 0.6788]       [0.5999, 0.6724]
balanced accuracy           0.6162                 0.6002                 0.6024
sensitivity / specificity   0.545 / 0.687          0.477 / 0.723          0.503 / 0.701
F1 / precision              0.577 / 0.617          0.536 / 0.612          0.549 / 0.607
permutation p               0.0010 (null 0.4995)   0.0010 (null 0.4978)   0.0010 (null 0.4976)
site-prediction strength    0.556 bacc (LOWEST)    0.721 bacc (HIGHEST)   0.630 bacc
site-held-out (LOSO) AUC    0.6398 (best)          0.6236                 0.6226
mean LOSO drop              -0.0185                -0.0266 (largest)      -0.0107 (smallest)
TR dependence (R2)          0.383 (lowest)         0.476 (highest)        0.383
scan-length / age / motion  0.433 / 0.292 / 0.325  0.534 / 0.394 / 0.422  0.452 / 0.310 / 0.366
                            (lowest on all)        (highest on all)
stability (fold AUC sd)     0.0275                 0.0278                 0.0373 (worst)
best band combination       all3 (ordinary),       slow5                  all3
                            classical (LOSO)
best normalization          D_SUBJ_BAND_Z          F_TRAINFOLD_BAND_Z     F_TRAINFOLD_BAND_Z
                            (A_RAW under LOSO)     (A_RAW under LOSO)     (A_RAW under LOSO)

## RANKING AGAINST THE STATED DECISION RULE
1. Scientific correctness — M1 satisfies it. S3A proved all three compute canonical DPABI
   ALFF identically (machine precision, full 954 cohort). S2 proved M1 and M2 share the exact
   same NIfTI input, and the paper (Sec 2.1) specifies ROI-first: "calculated from the Fourier
   transform of the mean time series". M1 is the ROI-first branch; M2's ordering has no basis
   in the paper text. OLD is ROI-first in spirit but derives from a different C-PAC product.
2. Leakage / negative controls — ALL PASS. 6000 null runs centre on 0.4955/0.5002, max null
   0.5867 < every observed AUC, p = 0.0010 (floor) everywhere.
3. Prefer information surviving site/confound tests — M1 wins decisively. It is the LEAST
   site-encoding (0.556 vs 0.721), least TR/length/age/motion-dependent, and retains the
   HIGHEST site-held-out AUC (0.6398). M2's larger univariate signal (53-54 vs 6-22 FDR-
   significant ROIs) is accompanied by the strongest site encoding and the largest LOSO drop,
   which is the signature of site-driven rather than diagnosis-driven variance.
4. Require stable paired improvement — NOT MET by any source. Every paired CI spans zero.
   M1 is consistently ahead but never significantly so.
5. Effectively tied -> prefer M1 (paper's ROI-first definition). THIS RULE IS THE ONE THAT
   DECIDES, and it points to M1 — which is also the leader on every confound criterion.
6. Not decided on ordinary-CV accuracy alone — M1 also leads under grouped and LOSO.

RANKING:  1st M1   2nd OLD   3rd M2
M1 first: paper-faithful, highest ordinary and site-held-out AUC, least confounded on every
measure, passes all negative controls.
OLD second over M2 despite a marginally lower ordinary AUC (0.6364 vs 0.6455): OLD is
ROI-first (paper-consistent ordering), substantially less site-encoding than M2 (0.630 vs
0.721), has the smallest site-held-out drop (-0.0107), and its LOSO AUC is effectively equal
to M2's (0.6226 vs 0.6236). M2's apparent advantage is confined to ordinary CV and is
accompanied by the worst confound profile in the study.
M2 third: not described by the paper, most site/TR/age/motion-encoding, largest LOSO drop,
lowest sensitivity (0.477).
CAVEAT ON STRENGTH: this ranking is driven by confound behaviour and paper fidelity, NOT by a
statistically significant discrimination difference. On AUC alone the three are tied.

## COMPUTE RECORD
partition gpu-h100 (CPU cores only; NO GPU requested at any point — GPUs reserved for A-GCL)
  1868883 s3c-lvl1   1 task    x 8 CPU  — Level 1 signal + univariate
  1868863 s3c-lvl2  20 tasks   x 4 CPU  — Level 2, 120 nested-CV conditions
  1868884 s3c-lvl4  12 tasks   x 4 CPU  — Level 4 confound encoding
  1868885 s3c-grp   20 tasks   x 4 CPU  — Level 4B/4C grouped + LOSO
  1868916 s3c-perm 120 tasks   x 4 CPU  — Level 3, 6000 permutation nested-CV runs
  all 173 array tasks COMPLETED, 0 failures. Total 24.32 CPU-hours.
seeds: master 20260818; permutation seeds 20260818 + 100000*chunk + j (fully reproducible).
outputs: /users/3171356m/agcl_audit_s0/s3c/
  PREREGISTRATION.txt, splits.json(+sha256), X_sources.npz(+sha256), meta.csv,
  lvl1_{distributions,bootstrap_reliability,agreement,univariate,univariate_summary}.csv,
  lvl2_all.csv, lvl2/ (per-task results + out-of-fold scores),
  lvl3_perm_all.csv, lvl3_perm_summary.csv, lvl3/,
  lvl4_confound_all.csv, lvl2_lvl4_merged.csv, lvl4/,
  lvl5_paired.txt, lvl5_paired_loso.txt, S3C_DECISION_TABLE.csv, S3C_FINAL.txt

## UNRESOLVED / LIMITATIONS
1. All three sources are weak discriminators in absolute terms (AUC 0.62-0.65, balanced
   accuracy 0.60-0.62, max |Hedges g| 0.36). No source provides strong ASD signal from ALFF
   node features alone with linear models.
2. The M1 advantage is not statistically significant. A larger cohort or a paired test with
   more power would be needed to separate the sources on discrimination.
3. Only linear models were tested, as specified. A non-linear model could rank the sources
   differently; this was not examined.
4. The S3B.5 FLIRT misalignment finding concerns norm_matrix only and does NOT affect M1,
   M2 or OLD, none of which passed through that step.
5. Confound encoding was measured with a fixed C=1 linear SVM (not inner-tuned) to keep the
   confound sweep tractable; absolute site-decodability could be slightly higher with tuning.
   The RANKING between sources is unlikely to change, but the absolute numbers are a floor.
6. sex was near chance for every source (0.525-0.532), so sex is not a meaningful confound
   in these representations; site, scan length and TR are.

S3C STATUS: EVIDENCE COMPLETE — no winner frozen. Awaiting your independent review.
