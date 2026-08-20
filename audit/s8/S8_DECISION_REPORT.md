# S8 DECISION REPORT — View Learner Audit + Training Readiness
2026-08-20 | git HEAD f6c49d49761ef3bf82b3d59798ee1e8ace6b62dc | tree CLEAN | production untouched
Pipeline: START_REPORT -> SMOKE PASS (2x) -> adversarial mirror verification (3 agents) ->
diagnostics A-F -> pre-registered policy -> authorized pilot training (first A-GCL training
in this audit). All outputs in agcl_audit_s0/s8/. Nothing written to data//processed//cache.

## 1. CURRENT VERIFIED STATE
Baseline: 0 non-audit files changed vs 8cac2358; all input hashes asserted at start
(cohort aca3d945..., X_sources dc10bf36..., M1_B cache 312266b2...).
The three training contracts were reconstructed FROM SOURCE and then adversarially
verified line-by-line by three independent agents (38 confirmed points total):
  P refuted once (drop_last) -> fixed. O refuted (3 real divergences) -> fixed.
  C NOT refuted (0 contract divergences).
Fixes applied BEFORE any accuracy existed; stale pre-fix pilots were cancelled unseen.

## 2. IMPLEMENTATION MISMATCHES DISCOVERED (in the PRODUCTION/ORIGINAL code itself)
M1. ORIGINAL augmentation is not edge-dropping: 08339b7 passes sigmoid(gate) ALONE as the
    augmented view's edge weights (REPLACES FC values). The fork multiplies mask*FC.
    These are different algorithms, both now faithfully mirrored.
M2. ORIGINAL evaluation slip: get_embeddings passed edge_weight positionally into the
    ignored edge_attr slot -> the original evaluated embeddings on UNWEIGHTED graphs.
M3. ORIGINAL model update SUBTRACTS the memory term (model_loss = InfoNCE - 0.4*cr);
    both fork profiles ADD it.
M4. Paper-exact view learner initializes SATURATED: normalize_nodes=False -> unbounded
    node embeddings -> |logits| huge -> mu is all-0 or all-1 depending on seed
    (seed means 0.0002 / 1.0000 / 0.14; per-edge cross-seed variance ~ max possible).
M5. Asymmetric masks are applied TRANSPOSED relative to the paper's (I+A o E) reading
    (proven err 0.0; consequence of S6's E^T finding). Not a crash; a semantics gap.
M6. Paper-exact directional sampling is incoherent as edge-dropping: 49.15% of (i,j)/(j,i)
    pairs receive OPPOSITE keep/drop decisions (0% under the corrected symmetric path).

## 3. VIEW LEARNER FINDINGS (diagnostics A-F, corrected O semantics)
A  Init masks: P saturated (M4). C healthy: mu ~ 0.46-0.60, tight.
B  No structural bias at init: |pearson| <= 0.17 vs FC value/|FC|/node strength.
C  One controlled step, BN-drift-free, common random numbers:
     O: both players improve their own objectives (view +0.296, model -0.418). Sanest.
     C: view improves (+0.028); encoder lowers InfoNCE (-0.113) as intended
        (composite before/after not comparable: the bank fills between probes; recorded).
     P: view step ~flat (-0.0008, saturated gradients); encoder composite WORSENS (+0.318),
        dominated by the paper-literal memory term that already contains the batch itself.
D  50 steps on one fixed batch: P kept% 2.1% -> 0.000 (collapse). C -> 0 on a fixed batch
    (adversary overfits one batch; different in real training). O (replace-aug) -> 77% kept.
E  Orientation proof (err 0.0): mask on directed (i,j) scales the message INTO target j;
    operator is (I+(E o B)^T)X. No silent transpose bug; see M5.
F  mu corr asym-vs-sym 0.863; sampled-keep pair disagreement 49.15% asym vs 0% sym.

## 4. PILOT TRAINING RESULT (authorized; 200 epochs, seed 20260818, full 954, labels unseen)
policy locked BEFORE results (S8_TRAINING_POLICY.txt): final-epoch eval only, h AND z both
reported, frozen S3C nested 5-fold CV, no selection of any kind.
  cfg  repr                    AUC     95% CI              bacc    sens    spec
  P    h                     0.4764  [0.4417,0.5119]     0.4874  0.3275  0.6473
  P    z                     0.4933  [0.4587,0.5286]     0.5011  0.3890  0.6132
  O    h                     0.5289  [0.4933,0.5640]     0.5359  0.4044  0.6673
  O    z                     0.4925  [0.4572,0.5305]     0.4987  0.3341  0.6633
  C    h                     0.5091  [0.4716,0.5457]     0.5193  0.4615  0.5772
  C    z                     0.4685  [0.4309,0.5038]     0.4747  0.4264  0.5230
  (unweighted-eval variants mirroring the original slip: 0.462-0.525, same conclusion)
  curves: P keep 0.29->0.03 (mask collapse); O keep 0.51->0.97 (mask saturation, InfoNCE
  driven to -6.1 = trivial task); C keep 0.48->0.10 with genuine slow loss decline.
  runtimes: 44-66 min/config on 1 CPU thread (bitwise-reproducible).
VERDICT: EVERY trained configuration is at CHANCE. Best value 0.529 (O_h); every CI
includes or nearly includes 0.50. Identical to the S7 RANDOM-encoder envelope (0.49-0.51).
200 epochs of the paper's contrastive scheme produced representations carrying NO MORE
linearly accessible diagnosis information than untrained random weights.
REFERENCES (frozen): FC-only linear SVM 0.7565 / LOSO 0.7432; Q1 0.6397; M1-only 0.6497.
LIMITS: one seed; linear probes; paper hyperparameters. This does NOT prove no
hyperparameter setting could do better. It DOES show the as-specified mechanism, run
faithfully in all three historical variants, does not beat a zero-training baseline
that is 0.23 AUC higher.

## 5. TRAINING READINESS
Machinery: READY and verified (smoke PASS x2, bitwise-deterministic, atomic outputs,
faithful contracts). Scientific case for scaling THESE contracts up: NOT SUPPORTED by
the pilot — two of three collapse into degenerate augmentation regimes exactly as the
mechanistic diagnostics predicted, and none beats chance.

## 6. ARCHITECTURE RISKS (carried forward)
- Sum pooling (P) / final L2 norm (O,C) discard the linearly accessible regional signal
  BEFORE the graph readout (S7.5: the single lossy step, ~0.13 AUC at random init).
- The paper regularizer has no opposing term -> mask collapse (confirmed in training).
- The O/budget direction saturates masks -> augmentation vanishes (confirmed in training).
- E^T transpose semantics for any asymmetric mask (M5).

## 7. WHAT REMAINS UNKNOWN
- Whether ANY contrastive variant (different lambda, temperature, symmetric InfoNCE,
  no L2-norm, mean-pool, supervised or hybrid head) can beat 0.7565 on this data.
- Multi-seed variance of the pilot numbers (single seed by design).
- Non-linear probes of trained embeddings (linear only, per policy).

## 8. RECOMMENDED NEXT EXPERIMENT
The audit's own evidence says the bottleneck is the readout, not the encoder: node-level
representations remain ~0.63 accessible through every layer while h/z are ~0.50, and
FC edges alone are 0.7565. The highest-information next step is therefore NOT more
contrastive tuning but a controlled comparison on the SAME frozen splits:
  (a) trained-C encoder + FLATTENED final-node readout (bypasses the lossy pooling),
  (b) supervised GNN baseline (same WGIN backbone, cross-entropy, nested CV),
  (c) FC-only SVM 0.7565 as the bar.
This isolates whether the deficit is the SSL objective or the pooled readout, with
one seed-matched run each, before any hyperparameter search is contemplated.

Single next action: run experiment (a) — probe the trained-C pilot's FLATTENED final-node
representation [90x32] on the frozen splits; the embeddings are already saved, so it
requires no new training and directly tests the pooling-bottleneck hypothesis.

NO winner declared. S8 NOT frozen. Independent review required.
