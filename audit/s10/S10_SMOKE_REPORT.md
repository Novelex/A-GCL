# S10 SMOKE REPORT — PASS (arms B, C, D; 8 subjects, 2 epochs, CPU)
date: 2026-08-20T22:26:54+01:00 | git HEAD d671467bce16260c682ff4ebdd466af5535c1b6d | production clean
| check | B | C | D |
|---|---|---|---|
| data gate (dataset sha 312266b2..., splits sha 28fed44d..., N/labels) | PASS | PASS | PASS |
| ROI-order: 90 nodes/graph, contiguous blocks, x == frozen M1_B rows, manifest sha a7632cd9... | PASS | PASS | PASS |
| edge_weight non-None + finite at extraction | PASS | PASS | PASS |
| budget unit test: mean(mu)=0.119 -> ascent pushes keep UP; 0.881 -> DOWN; sym; finite | n/a | PASS | PASS |
| mask exactly symmetric during training (max err logged per epoch) | PASS | PASS | PASS |
| shapes post_norm[8,90,32] h[8,32] z[8,32]; all finite; grads finite | PASS | PASS | PASS |
| checkpoint save/reload bitwise state_dict | PASS | PASS | PASS |
| atomic TEMP->validate->rename->DONE | PASS | PASS | PASS |
ROI-aware wrapper: replaces ONLY global_add_pool (flatten frozen-order [90x32] -> Linear(2880,32)
-> existing projection head); encoder/proj init stream identical to arm A at the same seed
(roi_linear initialized AFTER from a recorded offset seed). Single encoder pass per forward.
Arm A infrastructure = S9's (already smoke-proven); its seed 20260818 run is REUSED from S9.
