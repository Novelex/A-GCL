# S12A2 RESULTS — Readout Bottleneck Test
Input: S12A1 identity final_postnorm nodes [954,90,32], seeds 20260818/19/20 (gated).
Harness: frozen S3C folds, S11 pipeline (scaler + LinearSVC C-grid inner CV, pooled OOF AUC).
Plumbing sentinel: FC baseline reproduced 0.7565, delta 0.0000.

| arm | readout                          | s0     | s1     | s2     | mean   |
|-----|----------------------------------|--------|--------|--------|--------|
| P   | global_add_pool (current A-GCL)  | 0.5100 | 0.5848 | 0.5546 | 0.5498 |
| F   | learned Linear(2880->32), ROI-aware | 0.5763 | 0.6011 | 0.6335 | **0.6036** |
| C   | PCA-32 of flattened nodes (diag) | 0.6029 | 0.6026 | 0.5930 | 0.5995 |
| X   | flatten 2880 -> SVM (ceiling)    | 0.6046 | 0.6224 | 0.6381 | 0.6217 |
| P LOSO |                               | 0.4886 | 0.5723 | 0.5522 | 0.5377 |
| F LOSO |                               | 0.5918 | 0.6059 | 0.6001 | **0.5992** |

Pre-registered criteria:
- mean3(F)-mean3(P) = +0.0538 >= +0.05  MET (marginal; seed-1 gap only +0.0163)
- F > P in every seed: +0.0662 / +0.0163 / +0.0789  MET
- F >= X - 0.02: 0.6036 >= 0.6017  MET  (32-d learned readout recovers ~all node signal)
- LOSO corroborates: F-P = +0.0615 (per-seed +0.1031/+0.0336/+0.0478, all positive)
Diagnostics: C (unsupervised 32-d) ~= F (supervised 32-d): the deficit of P is not lack of
supervision — sum pooling itself destroys ROI-indexed signal. F head direct AUCs 0.57-0.63
consistent with SVM-on-features. Early stopping used validation AUC only (epochs 1-36).
Internal consistency: P ord per-seed == S12A1 pooled-h id probes exactly; X == S12A1
final_postnorm probe (0.6217). All F readouts trained per outer fold; scaler on inner-train;
no leakage. Wall: 19 units parallel, longest 7.3 min (X ceiling SVMs); F-LOSO 41 s/unit.
