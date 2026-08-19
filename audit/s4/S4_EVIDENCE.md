# S4 — FUNCTIONAL-CONNECTIVITY / EDGE AUDIT (evidence only)
2026-08-18 | HEAD 8cac2358 (see GIT STATE below) | no loader/cache modified, no A-GCL training, no ComBat,
M1 not normalized, raw FC files unchanged.
Frozen inputs: M1 ROI-first raw ALFF node features, 954 subjects (455 ASD / 499 NC),
90 AAL ROIs, cohort sha256 aca3d945f7d89ccb1cc6fb46ca01f6036ccf036d81d120d2b09fe4bd0d1d68c9.

## 1. FC SUBSET TO THE FROZEN 954 — ALIGNMENT
cohort sha256 re-verified against S1: MATCH
  FC files found            954 / 954     missing 0     extra-in-cohort 0
  duplicate IDs             0
  FC directories hold 956; the 2 beyond the cohort are exactly the S0 exclusions
  CMU_b_0050669 and Leuven_1_0050706 — neither enters the 954.
  labels: storage-derived ASD/NC agrees with the frozen S1 label for all 954 (True)
  ROI order: FC files carry NO embedded labels. Order is inherited from the .1D column
  order, which S1 proved identical to the frozen AAL90 axis (same loader, same 9001
  cutoff, order-preserving mask): position 0 = Precentral_L … position 89 = Temporal_Inf_R.

## 2. VALIDITY OF ALL 954 STORED cropped_matrix
  shape (90,90) for all            : True      dtype float64 for all
  finite                           : True      0 NaN, 0 Inf
  raw range (global)               : min -0.760031   max 1.000000
  symmetry max|FC - FC.T|          : max over 954 = 2.220e-16 (mean 1.551e-16) — symmetric
                                     to machine precision
  diagonal                         : exactly 1.0 everywhere; max|diag - 1| = 2.220e-16
  exact-zero entries               : 0 across the entire dataset (0 subjects affected)
  negative off-diagonal edges      : mean 731.7 / 8010 (9.1%), range [0, 3086]
  positive off-diagonal edges      : mean 7278.3 / 8010
  constant matrices                : 0
  duplicate identical matrices     : 0 (954 distinct SHA-256 over the matrix bytes)
  max|FC| per subject              : exactly 1.000000000000 for all 954

## 3. INDEPENDENT RECOMPUTATION — ALL 954 (not just a sample)
Source used: /mnt/scratch/.../muhammad-GraSTIACL/data/raw/rois_aal — the cpac/**filt_noglobal**
rois_aal .1D set that S2 proved (by byte-size match against the download manifests) is the
one the FC generator actually read. Method: parse the '#' header per file, keep labels < 9001
(90 columns, order preserved), np.corrcoef(ts, rowvar=False) over the full time series.
  subjects recomputed            954 / 954, covering the frozen cohort exactly
  max_abs_error                  0.0000e+00   (max over all 954)
  mean_abs_error                 0.0000e+00
  correlation stored vs recomputed  min 1.000000000000, mean 1.000000000000
  entries differing by > 1e-10   0 of 954 x 8100 = 7,727,400
  subjects with any differing entry  0 / 954
  dead ROIs (near-zero variance) in the filt source: 0 (0 subjects)
  T range 116..316
=> BITWISE EXACT reproduction of every stored entry. The FC generator chain
   (02_local_global_pcc.ipynb) is independently CONFIRMED for the whole cohort.
   Note this is a stronger result than S3A's ALFF check (which matched to ~1e-14);
   here the error is identically zero.

## 4. WHAT THE FC ACTUALLY IS — PROVEN PROPERTIES
  Pearson definition   : np.corrcoef over the full (unwindowed) ROI time series, signed.
                         Confirmed by exact reproduction in §3 and by the generator source
                         (notebook cell 7: W = np.corrcoef(ts90, rowvar=False)).
  thresholding         : NONE. 0 exact zeros anywhere in 7.7M entries; the value
                         distribution is continuous down to -0.760. No sparsification,
                         no top-k, no proportional threshold.
  diagonal / self-corr : RETAINED, exactly 1.0. Not zeroed, not removed.
  negative correlations: RETAINED and signed. 9.1% of off-diagonal edges are negative
                         (mean 731.7 per subject). No absolute value was taken.
  Fisher-z             : NOT applied. Values are bounded in [-0.760, 1.000]; a Fisher-z
                         transform would be unbounded and would map the diagonal to +inf.
                         The diagonal is finite and exactly 1.0, which is itself proof.

## 5. LOADER EDGE NORMALIZATION  fc / max(abs(fc))  — PROVEN NO-OP
datasets/abideDataset.py:85-87
    max_abs = np.abs(fc).max()
    if max_abs > 0: fc = fc / max_abs
  max|FC| BEFORE : min over 954 = 1.000000000000000, max = 1.000000000000000
  subjects where max|FC| deviates from 1.0 by > 1e-12 : 0 / 954
  max|FC| AFTER  : 1.0 (unchanged)
  MATHEMATICAL PROOF: Pearson r satisfies |r| <= 1, and the self-correlation r_ii = 1 is
  retained on the diagonal. Hence max|FC| = 1 exactly for every subject, and fc / 1.0 = fc.
  EMPIRICAL CONFIRMATION: verified BITWISE IDENTICAL before/after for ALL 954 subjects
  (np.array_equal true in every case).
  => the paper's stated edge rule ("normalized to [-1,1] by dividing by the maximum of the
     absolute values") is implemented faithfully but has zero effect on this data. It would
     only bite if the diagonal were removed/zeroed or a non-PCC edge weight were used.

## 6. DENSE EDGE CONSTRUCTION — 8100 DIRECTED ENTRIES CONFIRMED
loader lines 92-99 build nodes=arange(90); src=repeat_interleave(90); dst=repeat(90).
  edge_index shape (2, 8100); edge_weight shape (8100,)
  total directed entries        8100 == 90 x 90                       True
  self-loops (u == v)           90 present                            True
  both (u,v) and (v,u)          present for every ordered pair        True
  ordering                      row-major, matches fc.reshape(-1)     True
  exact-zero weights            0 in this cohort
  Because construction is dense rather than .nonzero(), a 0.0 weight WOULD still be emitted
  as an edge. Since no subject has any exact zero, dense and .nonzero() coincide here — a
  verified no-op for this data and a correctness generalization for future data.

## 7. FC CONFOUND DIAGNOSTICS (global summaries only; no feature selection, no prediction)
  global FC mean over 954: 0.3313 (sd 0.1103), range [0.0958, 0.6899]
  fraction of negative edges: mean 0.0913, range [0.0000, 0.3853]
  summary        eta2_site   r_TR     r_T   r_meanFD  r_dvars   r_age
  fc_mean          0.1054  -0.143   0.153     0.382    0.040    0.082
  fc_median        0.1051  -0.136   0.132     0.378    0.044    0.086
  fc_sd            0.2763   0.164  -0.409    -0.153    0.108   -0.032
  fc_absmean       0.0950  -0.123   0.112     0.386    0.055    0.078
  frac_neg         0.1549   0.190  -0.267    -0.301    0.012   -0.092
  MOTION is the strongest single correlate of global FC level (r = +0.382 with mean FD) —
  the classic motion-inflates-connectivity effect, expected here because the source is
  *noglobal* (no global signal regression).
  SITE explains 9.5-27.6% of between-subject variance in these summaries; fc_sd is the most
  site-driven (eta2 0.276). Site means of fc_mean span 0.265 (YALE) to 0.398 (CMU).
  SCAN LENGTH: longer scans give lower edge-wise sd (r = -0.409), as expected from
  estimator noise in r with more timepoints.
  ASD vs NC on global summaries is negligible: fc_mean d = -0.028, frac_neg d = -0.005,
  fc_sd d = -0.042. Global FC level is therefore NOT a trivial diagnosis proxy.

## MAJOR ANOMALIES
1. FOUR subjects have ZERO negative edges: Stanford_0051191, UCLA_1_0051251,
   UM_1_0050366, USM_0050453. Their off-diagonal minima are +0.057, +0.084, +0.003, +0.023
   and their median FC is 0.56-0.64. 71 subjects have fewer than 50 negative edges
   (cohort median is 575). An all-positive correlation matrix with a high median is the
   signature of a dominant global/motion component that no GSR has removed. These are
   valid Pearson matrices, not corrupt files, but they are extreme.
2. The opposite tail exists too: Yale_0050628 has 3086 negative edges (38.5%).
   The 20-fold spread in negative-edge fraction across subjects is itself a between-subject
   heterogeneity that any edge-weighted GNN will see.
3. FC and the frozen M1 node features come from DIFFERENT C-PAC strategies
   (FC = filt_noglobal, M1 = nofilt_noglobal func_preproc, established in S2). The graph
   edges are band-pass filtered 0.01-0.1 Hz while the node ALFF is not. This is internally
   consistent for each object in isolation but means edges and nodes describe differently
   filtered versions of the same scan.

## 8. RAW FILES PRESERVED
sha256sum -c against the S0 manifest over all 2868 raw files: exit 0, zero mismatches.
FC .mat files unchanged. All S4 evidence written to /users/3171356m/agcl_audit_s0/s4/
(outside the repo): s4_alignment_validity.txt, s4_fc_validity.csv (954 rows),
s4_anomaly_norm_dense.txt, s4_recompute_all.csv (954 rows), s4_recompute_summary.txt,
s4_confounds.txt, s4_fc_global_summaries.csv, w_fc_recompute.py, j_fc.slurm.

## COMPUTE RECORD
SLURM job 1869052, partition gpu-h100 (CPU only, no GPU requested), array 0-15,
2 CPU / 8 GB per task, 16/16 COMPLETED, 0 failures.

## UNRESOLVED ISSUES
1. The node/edge filtering mismatch (anomaly 3) is a design question for later stages, not
   a defect: nothing here proves it is wrong, but it is not what a single-pipeline design
   would produce.
2. FC ROI order is inherited, not embedded. It rests on the S1 proof that the FC generator
   used the same .1D header parsing as the ALFF route. The .mat files themselves still carry
   no labels, so an independent label check is impossible from the FC files alone.
3. The four all-positive subjects were not excluded or flagged by any existing QC in the
   repository; whether they should be is a scientific decision that belongs to a later stage.
4. Only the static/global PCC (cropped_matrix) was audited here. The dynamic-window
   matrices (correlation_matrices, 3x(90,90) per subject in *_DW) were confirmed present and
   correctly shaped in S1 but their values were NOT audited in S4.

S4 STATUS: EVIDENCE COMPLETE

## GIT STATE (changed during S4 — recorded accurately)
The S0 working-tree freeze ENDED partway through S4: you pushed the audit archive.
  before S4 wrote its report : HEAD 906a494b076968768573a24c31804c6b0b1dd65b, tree
                               dirty only with untracked ALFF_func_proc/
  now                        : HEAD 8cac2358ff12bcfa7452c38c4f4ef5e058814289
                               "Add S0-S3C ALFF audit evidence + ALFF_func_proc branches"
                               374 files, working tree CLEAN, in sync with
                               origin/main (github.com/Novelex/A-GCL)
  ALFF_func_proc/method1/alff_roi_first.npz and method2/alff_voxel_first.npz are now TRACKED.
IMPORTANT: this commit added audit artefacts and the two ALFF .npz files. It did NOT touch
any pipeline code, any file under data/, or the processed cache. Verified independently:
sha256sum -c over all 2868 raw files -> exit 0, zero mismatches; data_dense_v3.pt unchanged.
So the S0 *data* baseline is intact even though the *working-tree* freeze has ended.
The S4 findings above were computed against the same bytes throughout.
