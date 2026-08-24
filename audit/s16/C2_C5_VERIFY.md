# S16 C2-C5 VERIFICATION

host login1.cognition.gla.alces.network | 2026-08-24 17:35:42 | wall 676s

## FROZEN ANCHORS (never recomputed)
| reference | ord | LOSO |
|---|---|---|
| LinearSVC 4005 FC edges | 0.7565 | 0.7432 |
| random WGIN S12A3 (untrained watermark) | 0.6539 | — |
| trained WGIN S12A4b | 0.6429 | — |
| WGIN S12A5 arm A | 0.6307 | — |
| BNT S13 winner | 0.6583 | 0.6619 |

## CHECKS
- [PASS] **C2a_ruler** — FC ord 0.7565 (need 0.7565+-0.015); FC LOSO 0.7432 (need 0.7432+-0.015); permuted 0.4802 (need 0.50+-0.03); ALFF floor 0.6286
- [PASS] **C4_floor_exact** — learned block zeroed -> 0.7490109890 vs FC-alone 0.7490109890 on the SAME fold; diff 0.00e+00 (need bitwise). StandardScaler standardises each column independently, so the two blocks' different scales cannot distort one another.
- [PASS] (recorded) **C4_blocks_independent** — FC block zeroed, random learned block -> 0.5576 (chance-like, confirms the blocks are separate and correctly positioned)
- [PASS] **C4_floor_pooled** — pooled 5-fold, learned block zeroed -> 0.7565 (need 0.7565)
- [PASS] **T1_orthonormal** — max|EE^T-I| 4.17e-07; buffer in state_dict, requires_grad False
- [PASS] **T2_ocread** — P sums to 1 (1.8e-07); Z_G (8, 32, 128); repr (8, 4096); entropy 3.4627 / max 3.4657
- [PASS] **T3_attention** — rows sum to 1; shape (8, 4, 90, 90); 2 layers x 4 heads; FC never in scores
- [PASS] **T4_roi_equivariance** — permute data AND inp.weight cols -> max|Z_L(perm)-perm(Z_L)| 1.73e-06 (<1e-4); FC symmetry EXACTLY 0.0e+00 so a transposed profile is provably a no-op; guards a [B,D,90] axis swap
- [PASS] **T5_K1_mean** — max|Z_G/90 - mean| 7.15e-07 (declared factor 90)
- [PASS] **T6_wgin_hand_h.float64** — max err 0.00e+00 (negative edge weight + sign-flip node; self-loop double count is a LOGGED FORK, not fixed)
- [PASS] **T6_wgin_hand_h.float32** — max err 0.00e+00 (negative edge weight + sign-flip node; self-loop double count is a LOGGED FORK, not fixed)
- [PASS] **T7_sparse_positive_only** — 8 subjects: every sparse edge has FC>0 and equals the original value bitwise; retained 91.0% of 8100, min degree 3, isolated nodes 0
- [PASS] **T8_profile_is_row** — x[i,:90] == FC[i,:] bitwise, 8 subjects x 8 ROIs
- [PASS] **T9_gradients** — 0 dead trainable params; E.grad is None
- [PASS] **T10_determinism_reload** — same seed bitwise identical; checkpoint reload bitwise identical (CPU)
- [PASS] **T11_overfit_BNT** — train AUC 1.0000 (need 1.000), loss 0.000001 (need <0.01)
- [PASS] **T11_overfit_WGIN** — train AUC 1.0000 (need 1.000), loss 0.000003 (need <0.01)
- [PASS] (recorded) **T12_params** — A1(WGIN,D=3) 101,633 repr 11520 | A3(WGIN,D=90) 112,769 repr 11520 | A4(WGIN,D=93) 113,153 repr 11520 | A5(BNT,D=90) 288,385 repr 4096 | A6(BNT,D=93) 288,769 repr 4096
