# S12B GATE 0 — DATA / CACHE INTEGRITY: **PASS** (every item below is an executed assert; the run dies on any failure)
- ASSERTED (review R7): FC rebuilt from .mat == frozen X_fc bitwise; .mat-vs-S5-graph-cache mismatches == 0; FC symmetric and diag==1; node features == canonical M1_B bitwise; X_fc sha256 == frozen S11 sha
- cache (fresh namespace, delete-and-rebuilt): `data_s12b_4f01b0abc1b160bd.pt` sha256=9a851d47003ee327  (93.6 MB)
- n=954 (ASD 455 / NC 499); 90 nodes and 8100 directed edges per graph (self-loops included, FC diag=1)
- subject-order sha256: 3b7c7993707e74a1 | label sha256: c3389e38c0f87793
- S11 manifest sha: 4f01b0abc1b160bd | X_fc sha: 5e0780c9d99b238a
- frozen splits sha: 28fed44dc4666066 (== S3C authority) | M1_B dataset sha: 312266b23ecf1348
- FC source stats: sym_max=2.22e-16, diag_dev=2.22e-16, cache_vs_mat max=2.98e-08, x_vs_M1B max=0.0, mismatches=0
- folds: 5 ordinary + 19 LOSO (frozen, loaded not regenerated)
- ALFF band order: frozen M1 axis-2 order (ALFF/fALFF/mALFF as S3A), raw M1 loaded via exact FILE_ID reindex; joint-minmax M1_B bitwise == S5 cache (asserted in load_tensors)

## Environment
- git SHA 3047cf7 | host node10.cognition.gla.alces.network | 2026-08-23 03:04:32
- python 3.12.13 | packaged by Anaconda, Inc. | (main, Jul  9 2026, 14:38:16) [GCC 14.3.0]
- torch 2.5.0+cu121 | CUDA 12.1 | GPU(s): NVIDIA H100 NVL, 95830 MiB
- CPU count 32 | pip freeze -> out/pip_freeze.txt

## Cohort by site (n, ASD, mean age, %male, mean FD)
            n  asd  age_mean  pct_male  fd_mean
site                                           
CALTECH    28   12    28.150     1.214    0.070
CMU        20    9    25.750     1.150    0.302
KKI        47   19    10.002     1.255    0.132
LEUVEN_1   28   14    22.607     1.000    0.079
LEUVEN_2   33   15    14.094     1.242    0.101
MAX_MUN    49   23    26.245     1.082    0.109
NYU       173   74    15.105     1.208    0.069
OLIN       33   18    16.576     1.152    0.161
PITT       55   29    18.729     1.145    0.144
SBL        29   15    34.621     1.000    0.161
SDSU       31   11    14.322     1.226    0.085
STANFORD   39   19     9.976     1.205    0.107
TRINITY    45   21    16.985     1.000    0.096
UCLA_1     66   37    13.342     1.136    0.117
UCLA_2     23   10    12.484     1.087    0.122
UM_1       98   45    13.580     1.255    0.115
UM_2       33   13    16.064     1.061    0.078
USM        69   44    22.703     1.000    0.123
YALE       55   27    12.709     1.291    0.109

- wall 2.6s
