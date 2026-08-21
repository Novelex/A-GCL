# S9 DECISION REPORT — ONE DECISIVE DIAGNOSTIC
2026-08-20 | git HEAD f6c49d49761ef3bf82b3d59798ee1e8ace6b62dc | production untouched
Corrected-C only | seed 20260818 | 200 epochs | frozen S8 dataset (M1_B, sha 312266b2...) |
frozen S3C splits | S7.5 probe implementation | one model, one seed.

## 1. WHAT PASSED
Smoke (8 subj / 1 epoch / CPU / 2 threads): ALL checks — epoch-0 extraction provably before
any optimizer update (asserted n_updates==0), shapes, finiteness, edge_weight asserts,
checkpoint save/reload, atomic writes + DONE, production tree unchanged. See S9_SMOKE_REPORT.md.
Full run: job 1870264 on node07 NVIDIA L40S (1 CPU / 16 GB), 200 epochs in 153.9 s,
5800 update pairs, gradients finite in every epoch, checkpoint reload verified
(bitwise state_dict equality + forward agreement within the recorded CUDA tolerance 5e-3;
an earlier run 1870261 failed ONLY on an audit-side atol=1e-5 reload assert calibrated for
CPU — CUDA scatter_add non-determinism, known since S6/S8; outputs discarded, rerun clean).
Probes: 9/9 COMPLETED on the frozen splits, scaling and C-selection inside training folds only.

## 2. RESULTS (pooled OOF AUC, bootstrap 95% CI)
  representation                 ep0                    ep200
  pre_norm_nodes  [2880]   0.6194 [0.584,0.655]   0.5531 [0.516,0.591]
  post_norm_nodes [2880]   0.5531 [0.515,0.588]   0.5819 [0.543,0.619]
  h (sum-pooled)  [32]     0.4925 [0.457,0.528]   0.4592 [0.422,0.495]
  z (projected)   [32]     0.4918 [0.456,0.528]   0.4778 [0.443,0.514]
  pre_norm ROI-PERMUTED    —                      0.4408 [0.405,0.477]  (perm seed 20260818+9000, recorded)
Cross-check: ep0 pre-norm 0.6194 ≈ S7.5's untrained stage plateau (~0.62-0.63) — independent
replication of the S7.5 pipeline through a different extraction path.

## 3. PRE-REGISTERED VERDICT (rules fixed before results)
CONTROL: epoch-0 pre-norm = 0.6194 >= 0.60  -> S9 VALID.
PRIMARY: epoch-200 pre-norm = 0.5531, in (0.53, 0.60)
  -> "BOTH training and readout contribute."
Decomposition:
  TRAINING effect: node-level accessibility fell 0.6194 -> 0.5531 (-0.066) over 200 epochs.
    The contrastive/view objective actively DEGRADED regional diagnostic accessibility,
    but did not erase it.
  READOUT effect: the single largest cliff is SUM POOLING: post-norm nodes 0.5819 ->
    h 0.4592 (-0.123, landing below chance). Registered branch: "post-norm high, h near
    chance -> sum pooling is the immediate loss."
  L2 NORMALIZATION at ep200 is NOT the loss point (pre 0.5531 -> post 0.5819, no drop;
    at ep0 it did cost -0.066). Projection h->z is flat at chance.
  ROI PERMUTATION: 0.4408 — destroying cross-subject anatomical correspondence removes
    everything. Registered branch: information depends on FIXED ANATOMICAL IDENTITY,
    which global_add_pool provably discards (S7.5 invariance proof).

## 4. EXACT INFORMATION-LOSS LOCATION
  X/Q1 (~0.62-0.64) --encoder+training--> pre-norm nodes 0.5531   (moderate, -0.066 from training)
  pre-norm --F.normalize--> post-norm 0.5819                      (no loss at ep200)
  post-norm --GLOBAL_ADD_POOL--> h 0.4592                         (<<< PRIMARY CLIFF, -0.123)
  h --projection--> z 0.4778                                      (chance -> chance)

## 5. MASK BEHAVIOUR
mu 0.4638 -> 0.0830; sampled keep 0.4758 -> 0.1026: converged to a stable ~8-10% sparse
regime (no paper-style collapse to 0, no O-style saturation to 1). Deterministic mu for the
20 pre-registered subjects (indices 0-9 ASD block, 455-464 NC block): EXACTLY symmetric
(max|mu-mu^T| = 0.0e+00), strongly bimodal (mean 0.078, sd 0.191, range [0,1]) — near-binary
edge selection. Losses: model InfoNCE 4.14 -> 3.81 while the memory term rose 8.28 -> 8.90.

## 6. LIMITATIONS
One seed (20260818), one config (Corrected-C), current 954/AAL90 dataset, linear probes,
GPU training (CUDA scatter_add non-determinism documented; CPU retrain would differ ~1e-3).
Attention/mean/max pooling would remain permutation-invariant — none is "ROI-aware" without
explicit ROI identity; no such claim is made.

## 7. RECOMMENDED NEXT EXPERIMENT (strictly from the registered decision)
The verdict is "both contribute": training costs -0.066 at the node level and pooling
costs -0.123 on top. The registered decision therefore requires characterizing BOTH:
one experiment — retrain Corrected-C with embedding snapshots every 25 epochs across
5 seeds (CPU array, ~50 jobs, all parallel), probing pre-norm nodes at each snapshot.
This yields (a) the trajectory of the training-side degradation (does -0.066 saturate,
grow, or oscillate?), (b) seed variance for every number in this report, and (c) the
epoch at which node-level accessibility peaks — all without implementing any new pooling,
ROI embedding, or masking constraint (not authorized).

Single next action: epoch-resolved multi-seed node-accessibility trace (5 seeds x
snapshots every 25 epochs, probe pre-norm nodes on the frozen splits).

STOP. Nothing further implemented. S9 NOT declared perfect; awaiting independent review.
