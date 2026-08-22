# S12A1 SMOKE REPORT — PASS (8 subjects, seed BASE, CPU)
| check | result |
|---|---|
| exact manifest lookup, x_old [90,3], x_id [90,93], edge_weight [8100], finite | PASS |
| Q1 extraction via ACTUAL conv.propagate | PASS |
| Q1 identity analytic error (I90 + FC^T) | max_abs <= 1e-5 PASS; identity-block diag == 2 exactly |
| H1_preBN / H1_BN / H1_to_layer2 / H2 [90,32]; h [32]; z [32] | PASS |
| no gradients created (requires_grad False everywhere, no_grad) | PASS |
| no production writes; outputs under audit/s12a1 only | PASS |
| encoder state hash stable across rebuild (same seed) | PASS |
| outputs reloadable, bitwise | PASS |
| DONE written only after all validation | PASS |
