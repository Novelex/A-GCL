# S16 C6 — COMPLETE WAVE, PROVISIONAL (NOT collector-validated)

Generated 2026-08-27T20:21:51Z, read-only, from `audit/s16/runs/prod/jobs/*/fold_*.json`.

## STATUS

- **1431 / 1,431 cells**, all `status=OK`. 159/159 units. 0 POISON. 0 requeues.
- `s16_collect.py` and `s16_report.py` have **NOT** been run. No rejection class,
  sealed-bundle check, ledger validation, validity gate, C-PERM gate or
  pair-completeness gate has been applied. **Directional only — do not quote.**
- Raw artifacts (32 GB, 30 GB of it `feat/*.npz`) are gitignored, not in this commit.

| job | id | outcome | max elapsed |
|---|---|---|---|
| WGIN | 1875174 | 63/63 COMPLETED | 6:25:10 |
| BNT + EdgeMLP | 1875175 | 72/72 COMPLETED | 1:14:19 |
| Controls | 1875176 | 24/24 COMPLETED | 3:06:45 |
| C2 bounded | 1875173 | exit 5 — calibration gate halted it | 0:11:39 |

Total ~3,829 CPU-hours.

## Frozen reference

LinearSVC on 4005 FC edges: **0.7565** ordinary / **0.7432** LOSO.  
In-run `svm_tr_enc` (matched 80% encoder subset): **0.7142**

## Controls — probe_honest

| control | mean | sd | n |
|---|---|---|---|
| C-PERM | 0.4755 | 0.0520 | 54 |
| C-RAND | 0.5108 | 0.0945 | 54 |
| C-SHUF | 0.4796 | 0.0917 | 54 |
| C-ROI | 0.4688 | 0.0833 | 54 |

C-PERM is inside the pre-registered operational band [0.45, 0.55]. No leakage.

## Learned arms — probe_honest

| arm | arch | lab | site | loso |
|---|---|---|---|---|
| A1 | WGIN | 0.5270 | 0.5278 | 0.3875 |
| A3 | WGIN | 0.5904 | 0.5299 | 0.5205 |
| A4 | WGIN | 0.5758 | 0.5463 | 0.4954 |
| A5 | BNT | 0.6343 | 0.5935 | 0.6188 |
| A6 | BNT | 0.5938 | 0.5682 | 0.5072 |
| A7 | EDGEMLP | 0.6978 | 0.6767 | 0.5995 |

- pooled **lab**: 0.5998 ± 0.0738 (n=405)
- pooled **site**: 0.5765 ± 0.0708 (n=405)
- pooled **loso**: 0.5127 ± 0.1619 (n=405)

## Fusion — primary endpoint

| metric | value |
|---|---|
| fused folds | 567 |
| fused AUC | 0.7192 ± 0.0825 |
| **delta vs svm_tr_enc** | **-0.0027** (median +0.0000, sd 0.0142) |
| delta vs svm_tr_full | -0.0140 |
| folds beating FC | **85/567 = 15.0%** |
| **alpha = 1.0 (pure FC chosen)** | **333/567 = 58.7%** |

alpha=1 is the FC fallback ENDPOINT, not a floor. Deltas are unclamped.

| arm | fused AUC | delta | beat-FC |
|---|---|---|---|
| A1 | 0.7172 | -0.0046 | 14.8% |
| A3 | 0.7181 | -0.0038 | 7.4% |
| A4 | 0.7210 | -0.0009 | 16.7% |
| A5 | 0.7217 | -0.0001 | 18.5% |
| A6 | 0.7191 | -0.0028 | 13.9% |
| A7 | 0.7171 | -0.0048 | 13.0% |

## Edge treatment (E)

- signed: 0.5447 (n=405)
- abs: 0.5699 (n=270)
- pos_zero: 0.5799 (n=270)
- shift: 0.5666 (n=270)

## Training health — READ THIS

- movement_max median **0.065**, max 0.295
- **1031 of 1431 folds at or below the 0.10 validity threshold** — most of the grid would be classified UNTRAINED by the validity gate.
- clip_rate median 0.081, max 0.110 (threshold 0.30, never breached)
- best_epoch median 34 of 400; 34 folds at 1
- verdicts: {'OVERFIT': 1027, 'HEALTHY': 258, 'UNDERFIT': 146}

## C2

Halted at its calibration gate. All seven sources validated; the random encoder read **+0.02143** (MC se 0.00523, 3/20 sign flips) where it must read 0, so no retrospective estimate was interpreted. Independent of C6.
