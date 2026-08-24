# S15 SMOKE SUITE

host node02.cognition.gla.alces.network | 2026-08-24 15:06:15 | wall 514s | CEILING_PROBE 0.7565

## FROZEN REFERENCE LINES (never recomputed)
| reference | F-LAB | LOSO |
|---|---|---|
| LinearSVC 4005 FC edges | 0.7565 | 0.7432 |
| ridge logistic | 0.7561 | 0.7406 |
| BNT S13 winner | 0.6583 | 0.6619 |
| WGIN S12A5 arm A | 0.6307 | — |
| RANDOM WGIN S12A3 (untrained watermark) | 0.6539 | — |

## GATES
- [PASS] **S1_gate_c** — 954 subjects, ASD 455/NC 499, FC sym 0.0e+00, diag dev 0.0e+00, |FC|max 1.000000, folds lab 5/site 5/loso 19 = 29; all S11 hashes match
- [PASS] **S2_instrument** — R1/R2 F-LAB 0.7565 (need 0.7565+-0.015), LOSO 0.7432 (need 0.7432+-0.015); R3 permuted 0.4802 (need 0.50+-0.03); R4 ALFF floor 0.6286
- [PASS] (recorded, not gating) **S3_capacity_sweep** — 16 configs; S13 POST-MORTEM K=2/H=128 repr_dim 256 FC-recon R2 0.5102; best BNT R2 0.5560
- [PASS] **S4_overfit_BNT** — train AUC 1.0000 (need 1.000), loss 0.000001 (need <0.01), 500 steps, dropout 0, wd 0, no clip
- [PASS] **S4_overfit_WGIN** — train AUC 1.0000 (need 1.000), loss 0.000003 (need <0.01), 500 steps, dropout 0, wd 0, no clip
- [PASS] **S5a_orthonormal_buffer** — max|EE^T-I| 4.17e-07; E in state_dict, buffer, requires_grad False
- [PASS] **S5b_soft_assignment** — P sums to 1 (1.8e-07); Z_G (8, 32, 128); repr (8, 4096); entropy 3.4627 (max 3.4657)
- [PASS] **S5c_attention** — rows sum to 1 (2.4e-07); shape (8, 4, 90, 90); 2 layers x 4 heads; FC NEVER injected into scores
- [PASS] **S5d_roi_equivariance** — profile==FC row bitwise (True); FC symmetry EXACTLY 0.0e+00 so a transposed profile is provably a no-op; permuting data AND inp.weight cols -> max|Z_L(perm)-perm(Z_L)| 1.73e-06 <1e-4 (guards a [B,D,90] axis swap)
- [PASS] **S5e_K1_equals_mean** — max|Z_G/90 - mean| 7.15e-07 (declared factor 90; P==1)
- [PASS] **S5f_gradient_flow** — 0 dead trainable params; E.grad is None (True)
- [PASS] **S5g_determinism_reload** — same seed bitwise identical; checkpoint reload bitwise identical (CPU)
- [PASS] **S6a_wgin_hand_h.float64** — max err 0.00e+00 < 1e-06 (negative edge weight and sign-flip node included)
- [PASS] **S6a_wgin_hand_h.float32** — max err 0.00e+00 < 1e-04 (negative edge weight and sign-flip node included)
- [PASS] **S6b_wgin_features_selfloop** — x[i,:90]==FC[i,:] bitwise (8x8); LOGGED FORK: 90 self-loops with FC[i,i]=1.0 AND (1+eps)x_r -> own features counted TWICE, kept as S12A5 ran it
- [PASS] (recorded, not gating) **S7_clip_calibration** — BNT: p50 6.291 p90 13.379 p95 15.233 p99 22.338 max 26.96, 100% above S13's fixed 1.0 | WGIN: p50 10.388 p90 22.954 p95 26.882 p99 33.978 max 42.84, 100% above S13's fixed 1.0
- [PASS] **S8_determinism_env** — use_deterministic_algorithms(True); OMP_NUM_THREADS=8; torch threads 8

## S3 CAPACITY SWEEP (forward-only, random init, 3 seeds)

| arch | K/hidden | H/readout | repr_dim | trainable | FC-recon R2 | entropy |
|---|---|---|---|---|---|---|
| BNT | 2 | 128 | 256 | 277,249 | 0.5102 | 0.691/0.693 |
| BNT | 2 | 256 | 512 | 1,078,785 | 0.5506 | 0.693/0.693 |
| BNT | 4 | 128 | 512 | 278,017 | 0.5062 | 1.382/1.386 |
| BNT | 4 | 256 | 1024 | 1,080,321 | 0.5557 | 1.385/1.386 |
| BNT | 8 | 128 | 1024 | 279,553 | 0.5056 | 2.077/2.079 |
| BNT | 8 | 256 | 2048 | 1,083,393 | 0.5532 | 2.078/2.079 |
| BNT | 16 | 128 | 2048 | 282,625 | 0.5124 | 2.769/2.773 |
| BNT | 16 | 256 | 4096 | 1,089,537 | 0.5512 | 2.771/2.773 |
| BNT | 32 | 128 | 4096 | 288,769 | 0.5038 | 3.462/3.466 |
| BNT | 32 | 256 | 8192 | 1,101,825 | 0.5560 | 3.464/3.466 |
| BNT | 64 | 128 | 8192 | 301,057 | 0.5110 | 4.155/4.159 |
| BNT | 64 | 256 | 16384 | 1,126,401 | 0.5539 | 4.157/4.159 |
| WGIN | 128 | sum | 128 | 78,977 | 0.3909 | — |
| WGIN | 128 | roi | 11520 | 113,153 | 0.4292 | — |
| WGIN | 256 | sum | 256 | 289,025 | 0.4534 | — |
| WGIN | 256 | roi | 23040 | 357,377 | 0.4683 | — |

## S7 GRADIENT SURVEY (clipping OFF) — the S13 post-mortem

| arch | p50 | p90 | p95 | p99 | max | % above S13's fixed 1.0 |
|---|---|---|---|---|---|---|
| BNT | 6.291 | 13.379 | 15.233 | 22.338 | 26.96 | 100% |
| WGIN | 10.388 | 22.954 | 26.882 | 33.978 | 42.84 | 100% |
