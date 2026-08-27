# S16 C6 — PROVISIONAL SNAPSHOT (NOT collector-validated)
Generated 2026-08-27T14:37:50Z from `audit/s16/runs/prod/jobs/*/fold_*.json`, read-only.
Repo SHA `e2805c837992d43f1145c5ca5751378a4be5fb58`
## STATUS: INCOMPLETE AND UNVALIDATED
- **1338 of 1,431 cells** present. The WGIN array (`1875174`) was still running.
- `s16_collect.py` and `s16_report.py` have **NOT** been run. None of the 20 collector
  rejection classes, the sealed-bundle validator, the ledger check, the validity gate,
  the C-PERM gate or the pair-completeness gate has been applied to these numbers.
- **These figures are directional only and must not be quoted as results.**
- Raw artifacts (32 GB, 30 GB of it `feat/*.npz`) are gitignored and NOT in this commit.

## Job IDs
| job | id | state |
|---|---|---|
| C2 bounded | 1875173 | exit 5 — calibration gate halted the run |
| WGIN | 1875174 | running at snapshot time |
| BNT + EdgeMLP | 1875175 | all 72 tasks COMPLETED |
| Controls | 1875176 | all 24 tasks COMPLETED |

## Frozen reference (never recomputed)
- LinearSVC 4005 FC edges: **0.7565** ordinary / **0.7432** LOSO
- `svm_tr_enc` observed in this run (80% encoder subset): **0.7144**

## Controls

| control | probe_honest | n |
|---|---|---|
| C-PERM | 0.4755 | 54 |
| C-RAND | 0.5108 | 54 |
| C-SHUF | 0.4796 | 54 |
| C-ROI | 0.4688 | 54 |

C-PERM sits inside the pre-registered operational band [0.45, 0.55].

## Learned arms — probe_honest

| arm | arch | lab+site | n |
|---|---|---|---|
| A1 | WGIN | 0.5313 | 151 |
| A3 | WGIN | 0.5601 | 36 |
| A4 | WGIN | 0.5609 | 142 |
| A5 | BNT | 0.6139 | 144 |
| A6 | BNT | 0.5810 | 144 |
| A7 | EDGEMLP | 0.6872 | 144 |

| protocol | probe_honest | n |
|---|---|---|
| lab | 0.6046 | 384 |
| site | 0.5807 | 377 |
| loso | 0.5215 | 361 |

## Fusion — the primary endpoint

- fused folds: **556**
- delta vs `svm_tr_enc`: mean **-0.0027**, median **+0.0000**, range -0.1615 to +0.0865
- folds beating FC: **84/556 (15.1%)**
- **alpha = 1.0 selected in 327/556 (58.8%)** — the inner selection chose PURE FC, discarding the learned representation

alpha=1 is the FC fallback ENDPOINT, not a floor. Deltas are reported unclamped.

## Training health — READ THIS

- movement_max: median **0.063**, max 0.295
- **984 of 1338 folds have movement_max <= 0.10**, the validity threshold. On these data most of the grid would be classified **UNTRAINED** by the validity gate.
- clip_rate: median 0.079, max 0.110 (threshold 0.30 — not breached)
- best_epoch: median 34, max 362 of a 400 budget; 33 folds at best_epoch==1
- verdicts: {'OVERFIT': 965, 'UNDERFIT': 132, 'HEALTHY': 241}

## C2

Halted at its pre-registered calibration gate (exit 5). All seven sources validated, then the random encoder read **+0.02143 (MC se 0.00523, 3/20 sign flips)** where it must read 0, so no retrospective estimate was interpreted. C2 is scientifically independent of C6.
