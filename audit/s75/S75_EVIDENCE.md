# S7.5 — INFORMATION-LOCALIZATION + PERMUTATION-INVARIANT BASELINE AUDIT
2026-08-20 | git HEAD f6c49d49761ef3bf82b3d59798ee1e8ace6b62dc | tree CLEAN
CPU only. NO A-GCL TRAINING. NO ViewLearner training. NO S8. NO 116-ROI branch built.
Production sources verified byte-identical to 8cac2358 (0 non-audit files changed).
All 50 SLURM tasks COMPLETED; 160/160 stage probes + 20/20 ROI permutations + 4 section units.

## PRIMARY RESULT — WHERE LINEAR ACCESSIBILITY IS LOST
  stage                       n_feat    P: AUC (10 seeds)    O: AUC (10 seeds)
  X (input)                      270    0.6286  (single)     same
  Q1 = (I + E^T)X                270    0.6397  (single)     same
  WGIN1 pre-BN                  2880    0.6268 +- 0.0173     0.6268 +- 0.0173
  BN1                           2880    0.6268 +- 0.0173     0.6268 +- 0.0173
  post-BN1 -> WGIN2             2880    0.6268 +- 0.0173     0.6033 +- 0.0196
  WGIN2 pre-BN                  2880    0.6281 +- 0.0050     0.6278 +- 0.0038
  BN2 final nodes (pre-norm)    2880    0.6281 +- 0.0050     0.6278 +- 0.0038
  final nodes (post-norm)       2880    0.6281 +- 0.0050     0.5134 +- 0.0167
  h (after global_add_pool)       32    0.4933 +- 0.0231     0.5081 +- 0.0174
  z (after projection)            32    0.4993 +- 0.0237     0.4973 +- 0.0243

THE ANSWER TO THE KEY QUESTION IS UNAMBIGUOUS AND DIFFERS BY ARCHITECTURE:
  The WGIN/BN transformations lose essentially NOTHING. Accessibility is flat at
  0.627-0.628 from Q1 all the way through WGIN1, BN1, WGIN2 and BN2 in BOTH paths.
  The loss happens at exactly ONE step, and it is a DIFFERENT step in each path:
    P (normalize_nodes=False): lost at SUM POOLING
        final_nodes - h = +0.1347 +- 0.0249, 95% CI [+0.1205, +0.1495]   SIGNIFICANT
    O (normalize_nodes=True):  lost at the FINAL PER-NODE L2 NORMALIZATION, before pooling
        preNorm - postNorm     = +0.1144 +- 0.0192, 95% CI [+0.1024, +0.1253]  SIGNIFICANT
        final_nodes - h        = +0.0053 +- 0.0275, CI [-0.0111, +0.0213]      ns
  h vs z is negligible in both: P -0.0060 (ns), O +0.0108 (ns).
PERMITTED STATEMENT (pre-registered wording):
  "fixed-ROI regional information is more linearly accessible before permutation-invariant
   sum pooling at random initialization."
  For O the same information is already removed one step earlier, by F.normalize(x, dim=1),
  which discards per-node magnitude. NO claim is made about what training would do.

## 3. REGIONAL FC-STRENGTH BASELINES (off-diagonal only)
  feature          n_feat   AUC     95% CI            bacc     LOSO
  signed_s             90   0.6535  [0.6184,0.6864]   0.6069   0.6253
  absolute_a           90   0.6223  [0.5862,0.6543]   0.5998   0.6078
  positive_p           90   0.6509  [0.6170,0.6844]   0.5992   0.6217
  negative_n           90   0.6161  [0.5810,0.6513]   0.5826   0.6059
  concat [s,a,p,n]    360   0.6605  [0.6245,0.6910]   0.6319   0.6430
  FULL_FC_4005       4005   0.7565  [0.7251,0.7834]   0.6907   0.7432   <- reproduces S5.5 EXACTLY
  paired delta vs full FC (bootstrap 95% CI):
    signed -0.1030 [-0.1288,-0.0765] ; absolute -0.1341 ; positive -0.1055
    negative -0.1404 ; concat -0.0960 [-0.1226,-0.0696]      ALL SIGNIFICANT
=> 90-dimensional regional strength recovers a substantial part of the signal but is
   SIGNIFICANTLY below the full 4005-edge representation. Edge-level detail carries
   ~0.10 AUC that node-strength summaries do not. These are low-dimensional structural
   baselines, NOT architectural ceilings.

## 4. PERMUTATION-INVARIANT HAND-DESIGNED BASELINES
  feature                       n_feat   AUC     95% CI            bacc     LOSO
  sorted_signed_strength_90         90   0.5266  [0.4905,0.5618]   0.5373   0.5099
  sorted_absolute_strength_90       90   0.4884  [0.4516,0.5244]   0.5042    -
  FC_eigenvalues_90                 90   0.5392  [0.5024,0.5770]   0.5267   0.5190
  FC_eig_zerodiag_90                90   0.5392  [0.5023,0.5771]   0.5267    -
  sorted_Q1_perchannel_270         270   0.4696  [0.4351,0.5047]   0.4477    -
  sorted_Q1_nodenorm_90             90   0.4852  [0.4469,0.5242]   0.5028    -
  Q1_singular_values_3               3   0.4760  [0.4433,0.5150]   0.4787    -
  COMBINED_invariant               540   0.5339  [0.4973,0.5718]   0.5315   0.4939
=> every hand-designed permutation-invariant summary is at or near chance; every 95% CI
   includes or nearly includes 0.50. Sorting destroys the anatomical correspondence that
   the 4005-edge and 90-strength representations rely on.
   THIS DOES NOT BOUND ALL PERMUTATION-INVARIANT NEURAL NETWORKS. It bounds only these
   specific summaries. A trained invariant GNN is not constrained by this number.

## 5. ROI-ALIGNMENT DEPENDENCE DIAGNOSTIC
  aligned full FC (frozen)              AUC 0.7565   LOSO 0.7432
  independent per-subject permutation   AUC 0.4999 +- 0.0220, range [0.4601,0.5423], n=20
  paired delta                          -0.2566
  common-permutation sanity check       AUC 0.7565  <- reproduces the aligned result exactly
=> destroying cross-subject anatomical edge correspondence, while preserving each subject's
   own graph spectrum and intrinsic structure, drives the classifier to exact chance.
   The 4005-edge classifier depends entirely on consistent ROI identity across subjects.
   This is an ROI-ALIGNMENT DIAGNOSTIC, not a biological model.
   The common-permutation control confirms a single shared relabeling is a no-op, as
   mathematically required.

## 6. GNN PERMUTATION EQUIVARIANCE / INVARIANCE (fixed random weights, eval mode)
  path  h max_abs   h max_rel   z max_abs   node-equivariance max_abs
  P     3.906e-03   1.331e-06   4.150e-03   1.068e-04
  O     1.335e-05   2.303e-06   7.629e-06   1.937e-07
=> h and z are permutation-INVARIANT and the node matrix is permutation-EQUIVARIANT
   (P^T H'_nodes == H_nodes) to float32 tolerance. P's larger absolute error simply
   reflects its larger un-normalized activations; relative error is ~1e-6 in both.
   This is expected GNN behaviour, not a defect.

## 7. FULL 90-ROI FC RECOVERABILITY (leakage-safe CV ridge; NO diagnosis labels)
  path repr target    mean   median      sd     min      Q1      Q3     max  >0  >=.25 >=.50 >=.75
  P    h    signed   0.5905  0.6273  0.1410 -0.1836  0.5474  0.6743  0.7550  89   88    72     1
  P    h    absolute 0.5668  0.6374  0.2268 -0.8885  0.5074  0.6929  0.7698  88   84    69     3
  P    z    signed   0.6054  0.6442  0.1177  0.1674  0.5542  0.6802  0.7630  90   88    75     1
  P    z    absolute 0.6092  0.6531  0.1357  0.0933  0.5645  0.6965  0.7776  90   87    75     5
  O    h    signed   0.1384  0.1395  0.0266  0.0616  0.1214  0.1580  0.1958  90    0     0     0
  O    h    absolute 0.0888  0.0891  0.0308  0.0147  0.0682  0.1102  0.1596  90    0     0     0
  O    z    signed   0.1902  0.1877  0.0336  0.0932  0.1683  0.2152  0.2667  90    4     0     0
  O    z    absolute 0.1338  0.1274  0.0388  0.0342  0.1088  0.1636  0.2216  90    0     0     0
  MAJOR ARCHITECTURE EFFECT: under P, 75 of 90 regional strengths are recoverable at
  R2 >= 0.50 from a 32-dim embedding. Under O, ZERO regions reach 0.50 and the mean falls
  from 0.605 to 0.190. The final L2 node normalization removes the magnitude information
  that regional strength decoding depends on.
  O/z/signed most recoverable : Temporal_Sup_R .267, Temporal_Sup_L .257, Insula_L .252,
    Insula_R .251, Cingulum_Mid_R .248, Supp_Motor_Area_R .246, Putamen_L .244, Putamen_R .242
  O/z/signed least recoverable: Temporal_Pole_Mid_L .093, Amygdala_L .120, Caudate_R .128,
    Paracentral_Lobule_L .129, Cingulum_Post_L .134
  Distribution is DISTRIBUTED, not global-only: under P the interquartile range is
  0.55-0.68 across all 90 regions, i.e. most of the cortex is individually decodable.

## 8. IS THE STRENGTH CLASSIFIER DRIVEN BY A FEW ROIs?  (fold-safe coefficients)
  top-5 ROIs   14.2% of |coef| mass   (uniform would be  5.6%)
  top-10 ROIs  25.2%                  (uniform 11.1%)
  top-20 ROIs  43.5%                  (uniform 22.2%)
  ROIs needed for 50% of mass: 25 of 90   (uniform 45)
  top 10: Temporal_Inf_R .0374, Heschl_R .0317, Precuneus_R .0257, Occipital_Inf_R .0238,
          Temporal_Mid_R .0234, Fusiform_L .0229, Thalamus_L .0220, Temporal_Mid_L .0220,
          Occipital_Mid_R .0218, Insula_L .0214
  Coefficients were fitted INSIDE training folds only and averaged; no full-cohort selection.
=> MODERATELY CONCENTRATED BUT BROADLY DISTRIBUTED. It takes 25 regions to reach half the
   coefficient mass (uniform would need 45), so there is real concentration, but no small
   handful of ROIs drives the classification.

## 9. PAPER / ORIGINAL INTERPRETATION (maintained, per instruction)
  - A-GCL uses trainable edge-level augmentation.
  - ViewLearner consumes source and destination node embeddings and emits one edge logit,
    so the trained model CAN learn edge-specific structure, not merely strength summaries.
  - global_add_pool makes the graph representation permutation-invariant.
  - The paper's exact ablation percentage was NOT re-verified in S7.5: UNKNOWN / NOT VERIFIED.
  NONE of the banned phrases are used anywhere in this report.

## 10. 116-ROI BRANCH
  NOT BUILT. Feasibility note only: S75_FUTURE_116_PLAN.md (26 regions = AAL 91-116,
  18 cerebellar + 8 vermis; the .1D files already contain all 116 columns so FC and the
  OLD ALFF route are cheap to extend, but M1/M2 need a ~15-20 CPU-hour recompute, and the
  zero-valid-voxel exclusion set would change, so the 954 cohort would no longer hold).

## WHAT IS AND IS NOT PROVEN
PROVEN HERE
  - Accessibility is flat through WGIN1/BN1/WGIN2/BN2 and drops at exactly one step:
    sum pooling for P, final L2 node normalization for O. Both deltas are significant.
  - The full 4005-edge classifier depends entirely on consistent cross-subject ROI identity
    (independent permutation -> exact chance, 0.4999).
  - 90-dim regional strength is significantly weaker than full edges (-0.10 AUC).
  - h/z are permutation-invariant and nodes permutation-equivariant to float tolerance.
  - Under P, most regional FC strengths are individually decodable from the 32-dim
    embedding; under O they are not.
NOT PROVEN (and not claimed)
  - Nothing about what a TRAINED encoder can represent. Every encoder here is random.
  - No bound on permutation-invariant neural networks; only on the specific hand-designed
    summaries tested.
  - No claim that information is destroyed, irreversible, or that A-GCL has a ceiling.

## COMPUTE
  smoke: local, 8 subjects / 2 seeds, all stages + readers, PASS (caught np.trapezoid
  NumPy-2-only API bug, fixed).
  A first serial submission (1869716) was CANCELLED after measuring 111 s per 2880-dim
  probe -> 4.9 h serial, exceeding walltime; root cause was n_jobs=1 on a 16-core request.
  A second design (1869741) was also cancelled: SLURM packed 6 tasks/node and throughput
  was 0.44 probes/min, heading for per-task timeout with total loss of in-flight units.
  FINAL DESIGN: one probe = one resumable unit.
    1869821  probe array, 160 units over 40 tasks, 2 CPU each   COMPLETED
    1869822  roiperm array, 20 perms over 10 tasks              COMPLETED
    1869743/4/5 strength / invariant / recov                    COMPLETED
    misc (sections 6+8) run locally                             COMPLETED
  50/50 SLURM tasks COMPLETED, 0 failures.

S7.5 STATUS: EVIDENCE COMPLETE — NOT FROZEN. Awaiting independent review.
