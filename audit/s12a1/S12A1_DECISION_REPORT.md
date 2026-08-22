# S12A1 DECISION REPORT — ROI-IDENTITY INPUT / FIRST-WGIN INFORMATION GATE
2026-08-22 | git HEAD 1ece8166... (audit/r0 local commits; production surface unchanged)
One intervention only: X_id = [M1_B | I90] (90x93). Corrected-C MAIN encoder, random init,
eval()+no_grad, NO training, NO ViewLearner. Seeds 20260818/19/20. Authoritative S11
manifest order; frozen S3C folds; exact S11 probe implementation.

## 1-3. STATE / GATE / HASHES
Data gate: ALL PASS on all 954 (S12A1_DATA_GATE.md). Hashes asserted: manifest 4f01b0ab,
X_fc 5e0780c9, pair-map aba8e09f, splits 28fed44d, dataset 312266b2, ROI a7632cd9.
Encoder contract recorded from instantiation: emb 32, 2 layers, normalize_nodes=T,
message_relu=T, post_bn_relu=T, drop 0.3 (inert in eval), eps=0.0 buffer, standard pooling.
State-dict hashes per seed recorded; donor reused the EXACT saved real-FC encoder states.

## 4. Q1 MATHEMATICAL PROOF (all 954, actual conv.propagate — S6 method)
max_abs = 0.00e+00, max_rel = 0.00e+00, mismatching subjects = 0, identity-block
diagonal = 2.0 exactly (FC diag 1 + root 1). Orientation retained explicitly:
Q1_identity = I90 + FC^T. message_relu=True is a layer-1 no-op (inputs non-negative) —
confirmed by the exactness above. PASS.

## 5-6. RECOVERED FC + PLUMBING
Recovered 4005 pairs (transpose retained, S11 pair order) vs canonical X_fc:
max_abs 2.98e-08 (float32). PASS.
Q1-FC plumbing SVM: AUC 0.7565 — |delta| 0.0000 vs the frozen baseline. PASS. (Plumbing only.)

## 7-10. PRIMARY RESULTS (H1_BN, [90x32]->2880, ordinary CV)
                    s0       s1       s2      mean
  old (3-feat)    0.6328   0.5955   0.5947   0.6076
  identity        0.6873   0.6943   0.7083   0.6966
  donor (mean-FC) 0.6159   0.6070   0.6132   0.6120
per-seed identity-old : +0.0545 +0.0988 +0.1136  (mean +0.0890)
per-seed real-donor   : +0.0714 +0.0872 +0.0951
PRE-REGISTERED CRITERIA:
  A. mean identity H1_BN 0.6966 >= 0.65                        MET
  B. mean(identity-old) +0.0890 >= +0.02, positive all seeds   MET
  C. real-donor >= +0.03 in EVERY seed                         MET
ALL PLUMBING + ALL PRIMARY CRITERIA: **S12A1 PASS**

## 11. LOSO (primary H1_BN only)
  old 0.5973 | identity 0.6763 | donor 0.6077 (means of 3 seeds)
  LOSO real-donor gaps: +0.0571 +0.0730 +0.0757 — the effect SURVIVES site hold-out.
Donor protocol was strictly fold-safe (inner-train mean for C selection; outer-train mean
for the final fit; never a global mean).

## 12. SECONDARY OBSERVATIONS (diagnostic, random-init, not verdict-bearing)
  H2_BN: id 0.6539 vs old 0.6270 — the gain persists through layer 2.
  final post-norm nodes: id 0.6217 vs old 0.5300.
  pooled h: id 0.5498 vs old 0.5020 — READOUT REMAINS THE BOTTLENECK (h loses ~0.14
  from the node stages even with identity input). z similar (0.5410 vs 0.5090).
  H1_BN == H1_preBN as pre-noted (untrained BN in eval is affine-identity).
  H1_to_layer2 (post-BN ReLU) costs the identity condition -0.039 (0.6966 -> 0.6575).

## 13. PREREGISTERED VERDICT: PASS (Outcome 1)
Scientific statement (as pre-registered): "Adding fixed ROI identity allows
subject-specific FC information to survive the first actual WGIN MLP/BatchNorm
compression." The donor control proves the surviving information is SUBJECT-SPECIFIC FC,
not merely constant anatomical identity or ALFF: removing subject-specificity (donor)
costs 0.07-0.10 AUC in every seed, ordinary and LOSO.

## 14. EXACTLY ONE NEXT ACTION
S12A2 — READOUT ONLY: keep the proven identity input fixed; compare current
global_add_pool versus a fixed-order ROI-aware readout. No other changes.
NOT implemented. Awaiting review/authorization. STOP.
