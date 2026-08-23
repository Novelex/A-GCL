# S13 GATE 1 — INSTRUMENT CALIBRATION: **PASS**
THE probe for all of S13 is `K.probe_pipe`, used UNCHANGED (scaler + LinearSVC grid-C, all fitted inside each fold). R1 and R2 are therefore the same code path — R2 is the frozen-anchor reading of it.

| ref | OOF AUC | 95% CI | LOSO | criterion | ok |
|---|---|---|---|---|---|
| R1 raw FC 4005 (CEILING_PROBE) | 0.7565 | [0.7250,0.7857] | 0.7432 | >= R2-0.03 | True |
| R2 raw FC 4005 LinearSVC grid-C | 0.7565 | [0.7250,0.7857] | 0.7432 | in [0.741,0.772] | True |
| R3 raw FC, LABELS PERMUTED | 0.4802 | [0.4435,0.5176] | — | in [0.47,0.53] | True |
| R4 ALFF(3) flattened, node-only floor | 0.6286 | [0.5946,0.6640] | 0.6159 | record | — |

FROZEN REFERENCE LINES (never recomputed): LinearSVC 0.7565 / LOSO 0.7432; ridge logistic 0.7561 / 0.7406; best MLP 0.7246 / 0.7090; best WGIN 0.6307.
- folds: 5 ordinary + 19 LOSO | wall 552.3s
