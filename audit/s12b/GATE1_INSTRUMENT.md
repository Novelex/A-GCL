# S12B GATE 1 — INSTRUMENT CALIBRATION: **PASS**
Probe (frozen for the audit): per-fold Scaler -> PCA(min(200,d,n-1), randomized, rs=20260818) -> LogisticRegression(l2, C by inner 5-fold CV on [0.001, 0.01, 0.1, 1, 10, 100]); threshold 0.5; bootstrap 2000.

| ref | pooled OOF AUC | 95% CI | LOSO AUC | criterion | ok |
|---|---|---|---|---|---|
| R1 raw FC probe (CEILING_PROBE) | 0.7481 | [0.7168,0.7773] | 0.7342 | >= R2-0.03 | True |
| R2 raw FC LinearSVC (frozen S11 path) | 0.7565 | [0.7250,0.7857] | — | in [0.74,0.77] | True |
| R3 labels permuted (mean of 10 draws, A1) | 0.4877 | draws [0.465,0.514] | — | mean in [0.47,0.53] & max<=0.55 | True |
| R4 ALFF(joint-minmax, 270) probe | 0.6315 | [0.5976,0.6657] | 0.6062 | floor (report) | — |

RETENTION RATIO denominator: CEILING_PROBE = 0.7481 (above-chance 0.2481).
Old FC benchmark 0.7565 is NOT compared to Arm D directly; R1 defines the new measurement ceiling (validation rule 12).
- wall 211.9s | folds (computed from the frozen authority): 5 ordinary + 19 LOSO
