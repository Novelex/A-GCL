# S12B RESULTS — ENCODER AUTOPSY
All three blocking gates PASS. 552/552 Track-1 forward configurations, 7020
config x stage rows, Tracks 2/3/4 complete, both winner controls PASS.
Ceiling: CEILING_PROBE (R1) = 0.7481. Retention ratio = (AUC-0.5)/(0.7481-0.5).

## GATES
G0 data: n=954 (455/499), 90 nodes / 8100 directed edges, FC rebuilt from .mat ==
frozen X_fc BITWISE, cache-vs-mat mismatches 0, symmetry 2.2e-16, diag dev 2.2e-16,
splits sha 28fed44d (frozen, 5 ordinary + 19 LOSO).
G1 instrument: R2 LinearSVC 0.7565 (frozen anchor reproduced EXACTLY), R1 probe
0.7481 (only -0.008 vs SVC -> probe is not weak), R3 mean-of-10 permutations 0.4877
(draws 0.465-0.514) -> no leakage, R4 ALFF floor 0.6315.
G2 forward: 14/14 checks. WGINConv hand-verified on a 4-node toy graph with a
negative edge weight, both message_relu branches, f64 <1e-6 and f32 <1e-4.
Production parity <1e-5. CUDA scatter-add non-determinism (~6e-7) FIXED to bitwise
0.0 via deterministic algorithms + cuBLAS workspace pin.

## TRACK 1 — WHERE THE SIGNAL DIES (retention ratio)
Production default (emb32, BatchNorm, mrelu=T, normalize_nodes=T):
| arm | S0 | S1 A1 | S2 H1 | S3 H2 | S4 norm | S5 pool | S6 ROI |
|---|---|---|---|---|---|---|---|
| A ALFF(3)        | 0.530 | 0.564 | 0.401 | 0.333 | 0.371 | -0.033 | 0.371 |
| B ALFF+onehot    | 0.530 | **0.967** | 0.590 | 0.505 | 0.550 | 0.026 | 0.550 |
| C ALFF+FC-row    | **0.996** | 0.751 | 0.628 | 0.447 | 0.498 | 0.118 | 0.498 |
| D ALFF+both      | 0.996 | 0.928 | 0.675 | 0.500 | 0.518 | 0.118 | 0.518 |
| C-shuf (control) | 0.469 | 0.404 | 0.358 | 0.330 | 0.383 | -0.046 | 0.383 |
| B-rand (control) | 0.530 | 0.933 | 0.525 | 0.345 | 0.374 | -0.051 | 0.374 |
Best family (emb128, LayerNorm): B 0.967 -> 0.707 at S2; C 0.751 -> 0.693; best S6
overall 0.690; best at ANY encoder stage 0.834 (D, emb32, bn, S2).

TWO SERIAL KILLERS, one recoverable and one not:
1. FIRST WGIN BLOCK (S1->S2), Linear(d->emb)+Norm+ReLU. Arm B enters at 0.967 and
   leaves at 0.590 (production) / 0.707 (best): a PERMANENT loss of 0.26-0.38
   retention in ONE operation. No later stage recovers it.
2. global_add_pool (S4->S5), collapse to -0.05..0.12 retention (chance). This one
   IS recoverable: ROI-flatten (S6) restores +0.274.

## BOTTLENECK VERDICTS (arm C at H2 unless noted)
B1 agg rank-3: PARTIAL. Kills arm A only (0.530->0.564, +0.03). With a full-rank
   node basis (B/D) A1 reaches 0.928-0.967 -> the aggregation is NOT the wall.
B2 emb width: CONFIRMED, small. 32=0.490, 64=0.503, 128=0.552 (+0.062).
B3 normalisation: CONFIRMED, largest knob. bn=0.450, ln=0.537, none=0.557 (+0.107).
   BatchNorm is the worst of the three.
B4 F.normalize: REJECTED. nn=T 0.510 vs nn=F 0.515 (-0.005, within noise).
B5 global_add_pool: CONFIRMED. pool 0.239 vs ROI-flatten 0.512 (+0.274).

## CONTROLS (all PASS)
C-shuf collapses 0.996 -> 0.469 at S0 -> the gain is genuine connectivity TOPOLOGY,
not global FC strength (decision rule 9 does NOT fire).
B-rand ~= B at S1 (0.933 vs 0.967) -> "ROI identity" was a FULL-RANK NODE BASIS,
not anatomy (decision rule 10 FIRES).
P-lab (winner, labels permuted) 0.4885 PASS. P-roi (winner, per-subject ROI
permutation of features AND FC) 0.4974 PASS.

## WHAT THE ENCODER ENCODES INSTEAD (production default, mean arms A-D)
| stage | AUC | reten | FC R2 | site F1 | age R2 | motion R2 | meanFC R2 |
|---|---|---|---|---|---|---|---|
| S0 | 0.689 | 0.763 | 0.469 | 0.668 | 0.408 | 0.328 | 0.716 |
| S1 | 0.699 | 0.803 | 0.675 | 0.529 | 0.391 | 0.350 | **0.945** |
| S2 | 0.642 | 0.573 | 0.564 | 0.374 | 0.262 | 0.269 | **0.943** |
| S3 | 0.611 | 0.446 | 0.464 | 0.310 | 0.150 | 0.202 | **0.908** |
| S5 | 0.514 | 0.057 | 0.311 | 0.100 | 0.010 | 0.122 | **0.890** |
| S6 | 0.620 | 0.484 | 0.477 | 0.283 | 0.178 | 0.203 | **0.908** |
Diagnosis retention falls 0.80 -> 0.06 while MEAN FC STRENGTH survives at R2 0.89
through every stage INCLUDING pooling. The encoder degrades into a global-signal-
strength detector. Site F1 also decays (0.67 -> 0.10), so it is not scanner either.

## TRACK 2 — HONEST CEILING (frozen folds, AdamW, 36 runs + classical)
| model | ordinary | LOSO | overfit folds |
|---|---|---|---|
| **LinearSVC (frozen S11)** | **0.7565** | **0.7432** | — |
| ridge logistic | 0.7561 | 0.7406 | — |
| best MLP h64 wd1e-3 | 0.7246 +-0.008 | 0.7090 +-0.003 | 97% |
| h256 wd1e-4 | 0.7237 +-0.005 | 0.7087 | 97% |
| h512 (2.07M params) | 0.7110-0.7201 | <=0.7060 | 99-100% |
| elastic-net logistic | 0.7140 | 0.6910 | — |
Every learned model loses to the linear one by ~0.03; 96-100% of folds are OVERFIT
by the pre-registered rule; MORE CAPACITY IS WORSE; weight decay over four orders
of magnitude changes almost nothing. p >> n as predicted (763 train, 4005 features).

## TRACK 3 — ALFF NORMALISATION (a second, independent leak)
raw 0.6423 / LOSO 0.6174 > per-band min-max (production v3) 0.6322 / 0.6119 ~=
joint min-max (frozen M1_B) 0.6315 / 0.6062. Per-subject normalisation costs ~0.010
ordinary and ~0.011 LOSO. Instrument invariance check |raw - z-across-subjects| =
0.000000 exactly, 0 degenerate columns. ALFF adds only +0.004 on top of FC
(4275: 0.7522 vs FC-only 0.7481).

## TRACK 4 — PROTOCOL INFLATION (production corrected-C backbone, h only)
Transductive final 0.4906 / 0.5097 / 0.5073; inductive final 0.4927 / 0.4829 /
0.4793 -- the production backbone sits AT CHANCE after 100 SSL epochs.
T-trans minus T-ind = +0.0176 +- 0.0139.
E-best minus E-final = **+0.0442 +- 0.0060 (inductive)**, +0.0180 +- 0.0100 (trans).
Reported as measured deltas only; no claim is made about the paper's number.
