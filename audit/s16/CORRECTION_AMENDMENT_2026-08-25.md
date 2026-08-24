# S16 PRE-RESULT CORRECTION AMENDMENT — 2026-08-25

> **The scientific question and arm grid are unchanged; implementation and evaluation
> defects were corrected before valid C6 results existed.**

No C6 scientific result has ever been produced. Every artifact on disk is either a
4-epoch end-to-end smoke run or a failed submission. Twenty defects were found and
corrected across Correction Gates 1-8. This amendment records what changed in the
CLAIMS, so that nothing written earlier is left standing uncorrected.

## A5 — grid arithmetic
Arm **A7 (EdgeMLP) at all four E** was added after the original arithmetic was agreed.
| | earlier | current |
|---|---|---|
| configs | 17 | **21** |
| units | 135 | **159** (126 MAIN + 24 CTRL + 9 ABL) |
| fold-runs | 1,134 | **1,431** |
Expected-ledger hash **`8587b1ca36553408`**, generated from `s16_grid` + `s16_data`.

## A6 — "guaranteed floor" withdrawn
The alpha=1 endpoint was described as a **guaranteed floor**, implying the fused delta
could not be negative. It can. Correct statement: alpha=1 is the **FC FALLBACK
ENDPOINT** — it exists, equals standardised FC, preserves the FC ranking, and its AUC
equals **that fold's** `svm_tr_enc`. Nothing constrains the SELECTED alpha on the outer
test set. **The delta may be negative and is never clamped, floored or replaced.**

## A7 — the A7 "bridge" claim withdrawn
A7 is an **ordinary arm**. It is architecturally identical to S12A5 arm C (bitwise:
same layer shapes, dropout 0.3, plain `Linear(32,1)`, 1,033,793 trainable parameters)
but the training recipes differ (Adam coupled / no warmup / fixed clip / 200 epochs
versus AdamW decoupled / warmup / cosine / adaptive clip / min-80 max-400). **Any
C6-vs-C2 difference reflects training-set size AND recipe jointly and isolates
neither.**

## A8 — 0.7319 is not a constant
It was one fold's reading of `svm_tr_enc`. Each fold now reports its own `svm_tr_enc`
and `svm_tr_full` on identical test subjects, plus the paired `size_delta_paired`.
Measured on fold lab0: **0.7319 vs 0.7490, paired +0.0171**. The historical
`0.7565 - 0.7319 = +0.0246` is **NOT a clean estimate** — it spans different fold
designs, training-set definitions and code versions. **0.7565 must not be subtracted
from any individual fold or site**; it is a labelled historical reference only.

## A9 — array status corrected
The C6 arrays submitted on 2026-08-24 reported SLURM **COMPLETED** while producing
**1,431 failed folds and zero results**. A unit exits 0 even when every fold inside it
fails, so **a scheduler COMPLETED state is not scientific completion**. Completion is
now decided cell by cell against the expected ledger, and the collector refuses to
write anything unless every cell is present, unique, successful and provenance-backed.

## A10 — EMA wording
EMA was computed every optimiser step and **discarded**, while the frozen recipe
(S15 PROTOCOL.md:186, :200) requires "EMA and raw both evaluated ... both reported
with the delta". Restored: raw is the validation-best checkpoint, EMA(0.999) is
evaluated **alongside** it, both reported with `ema_delta`, selection by VALIDATION
only. Each row records `evaluated_state` literally.

## A11 — C2 superseded
`C2_PROBE.md` and `C2_PRECISION.md` are marked **SUPERSEDED — DO NOT QUOTE**. Their
estimator's calibration FAILED: the random encoder, which cannot memorise, read
**+0.0231** against a predeclared equivalence band of **[-0.01, +0.01]**. **All
retrospective pure-bias estimates are UNRESOLVED**, and the earlier claim that "only
the two BNTs show real memorisation bias" is **WITHDRAWN**. A site x label matched
replacement exists (`s16_c2_bounded.py`), is confirmed feasible (6,671 checks, 0
infeasible), and **has not been run**.

## A12 — stale SHA
`s16_data.GIT` was hard-coded to `d52798c` while HEAD had moved on, misattributing
every provenance record. It is now read from `git rev-parse` at runtime, and the
**cache builder SHA is recorded separately** so a cache built at one commit and
consumed at another is visible rather than conflated.

## A13 — wall-time statements
Every wall-time figure quoted before 2026-08-25 was an estimate, and the ones given as
"4-7 hours" and "6-10 hours" were wrong: the measured cost is **~17 min per fold**,
dominated by probing an 11,520-dimensional WGIN representation. No wall-time claim
should be quoted unless it cites a measurement. Current honest figure for 1,431
fold-runs: **~6 h at 40 concurrent slots, ~16 h at 15**.

## Unchanged
The scientific question, the five architectures/arms, the four E treatments, the three
fold protocols, the seeds, the frozen cohort and splits, and every pre-registered
decision rule are **unchanged**.
