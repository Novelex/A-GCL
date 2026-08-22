# S12A5 RESULTS
All gates PASS; smoke PASS; 3-lens pre-launch review 0 blockers (2 metric gaps fixed
pre-launch); 9 Wave-1 jobs (216 fold-trainings) + 3 Wave-2 transductive jobs complete;
grad norms finite throughout.

## WAVE 1 (supervised CE + weight decay 1e-4; pooled OOF AUC; 3 seeds)
| arm | head ord | repr(SVM) ord | head LOSO | repr LOSO |
|---|---|---|---|---|
| A WGIN control        | 0.6307 | 0.6488 | 0.6340 | 0.6515 |
| B WGIN + FC edge skip | 0.6789 | 0.7044 | 0.6805 | 0.6896 |
| C pure FC edge MLP    | **0.7124** | **0.7239** | **0.7080** | **0.7145** |
Reference: frozen FC-SVM 0.7565 / LOSO 0.7432. Winner C detail (head, ord): acc/bacc/F1 in
out/CONSOLIDATED.json; train-val gaps A +0.220, B +0.231, C +0.195 (regularization did not
eliminate overfitting; median best epoch 23-24). Movement (trainable): A enc 0.14/readout
0.79; B enc 0.12/edge 0.90; C edge 0.88/head 0.58 — in B the EDGE branch does the learning.
Site-wise (C): strongest USM 0.833, OLIN 0.825, UM_2 0.804; weakest CALTECH 0.486,
STANFORD 0.550 (small sites). Epoch-0 and final embeddings + all ckpts saved (manifest).

## WAVE 2 (A-GCL corrected-C on ROI stack; NEVER mixed with inductive)
TRANSDUCTIVE (SSL on all 954, fixed 200 ep): h 0.6026 (h0 0.5903), z 0.5651 (z0 0.5641),
flat 0.6408 vs flat0 0.6539 (training REDUCES retention again), LOSO h 0.5903 mean;
mask keep collapses to 0.016-0.035; encoder moved 0.56-0.61 (trained hard, still fails).
INDUCTIVE (citation, S12A4 arm 3, identical stack/folds/seeds): h 0.5672, z 0.5055.
Transductive - inductive = +0.035 < +0.05 pre-registered threshold.

## CONTROLS
ROI-permutation (winner C, per-subject FC row+col permutation, retrained): AUC 0.5020 —
COLLAPSE, PASS: the signal is anatomical ROI-identity-indexed edge values, not artifacts.
