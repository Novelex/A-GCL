# S13 GATE 0 — DATA AND CACHE: **PASS**
Every line below is an EXECUTED assert; the run exits non-zero on any failure.

- cache (new namespace, delete-and-rebuilt): `data_s13_4f01b0abc1b160bd.pt` sha256=f74fe2c087f61a86 (90.7 MB). data.pt / data_dense_v3.pt / earlier .pt are NEVER loaded.
- len(dataset) == 954 | ASD 455 / NC 499 == S11 manifest exactly
- FC.shape (954, 90, 90) | ALFF.shape (954, 90, 3)
- FC symmetry max |FC-FC^T| = 2.220e-16 (< 1e-6) | max |diag(FC)-1| = 2.220e-16 (exactly 1.0)
- no NaN/Inf anywhere in FC or ALFF
- subject-ID order sha256 3b7c7993707e74a1 | label sha256 c3389e38c0f87793
- FC row-major edge order == S11 X_fc BITWISE (max abs diff 0.0); X_fc sha 5e0780c9d99b238a
- ALFF band order: frozen M1_B, sha de63049c08ac023c; x_vs_M1B max 0.0; .mat-vs-cache mismatches 0
- folds LOADED from frozen S3C splits sha 28fed44dc4666066 (5 ordinary + 19 LOSO), never regenerated
- M1_B dataset sha 312266b23ecf1348

## Environment
- git 9ea8e5f | host login1.cognition.gla.alces.network | 2026-08-23 19:11:05
- python 3.12.13 | torch 2.5.0+cu121 | numpy 1.26.4 | sklearn 1.9.0
- OMP_NUM_THREADS=4 | cores 32 | pip freeze -> out/pip_freeze.txt
- wall 12.0s
