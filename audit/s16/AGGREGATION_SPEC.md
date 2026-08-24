# S16 AGGREGATION SPECIFICATION — written before any C6 result is read

## 1. THREE SEPARATE ESTIMANDS. NEVER AVERAGED TOGETHER.
The three fold protocols answer different questions on different populations and
**must never be pooled, averaged, or placed in a single cell**.

| estimand | folds | question it answers | comparator |
|---|---|---|---|
| **E-LAB** | lab0, lab1, lab2 | performance under label-stratified resampling, the historical anchor's design | that fold's own `svm_tr_enc` |
| **E-SITE** | site0, site1, site2 | performance when (label, site) is balanced across folds | that fold's own `svm_tr_enc` |
| **E-LOSO** | loso0, loso1, loso2 | generalisation to an ENTIRELY UNSEEN SITE | that fold's own `svm_tr_enc` |
Any table, plot or sentence mixing them is invalid. E-SITE is expected to read HIGHER
than E-LOSO because site information becomes exploitable within folds; that is a
property of the design, not a result.

## 2. THE COMPARATOR IS FOLD-SPECIFIC. 0.7565 AND 0.7319 ARE NOT CONSTANTS.
`svm_tr_enc` is computed **per fold, on that fold's own test subjects**. 0.7319 was one
fold's reading of it and is NOT universal. `0.7565` is a HISTORICAL full-cohort,
5-fold, full-`tr` number produced under a different split design.
**Do not subtract 0.7565 from an individual fold or site.** It may appear only as a
labelled historical reference line on a plot of the E-LAB pooled figure.
Each fold additionally reports `svm_tr_full` (same pipeline, same test subjects,
trained on the complete outer `tr`) and the **paired** `size_delta_paired =
svm_tr_full - svm_tr_enc`. The historical difference `0.7565 - 0.7319` is **NOT** a
clean estimate of the training-size cost: it compares two different fold designs, two
different training-set definitions and two different code versions. The paired
per-fold delta is the only defensible size estimate.

## 3. THE HEADLINE QUANTITY
For every cell: `delta = AUC(fused at selected alpha) - svm_tr_enc(that fold)`.
Reported per estimand as mean +/- SE **across folds**, and separately **across seeds**,
never collapsed into one number. **The delta may be negative and is never clamped,
floored, or replaced after test evaluation.** A negative delta is a result.

## 4. FULL GRID REPORTED
All 21 configs x 4 E (where applicable) x {plain, fused} are reported, not only the
best. Arms failing the validity gate (movement <= 0.10, clip_rate >= 30%, or failing to
beat their C-RAND twin by >= 0.03) are printed as **UNTRAINED** and excluded from any
architecture verdict while still appearing in the grid.

## 5. C7 CANDIDATE SELECTION IS EXPLORATORY
Any configuration chosen from C6 to carry into C7 is **selected on C6 data and is
therefore exploratory**. Its C6 number is not an unbiased estimate of its C7
performance and must never be quoted as a final result. C7 scores it once, on folds
C6 did not use for selection.

## 6. A7 IS AN ORDINARY ARM
A7 (EdgeMLP) is reported as one arm among the others. **Its former "bridge" role is
withdrawn**: S12A5 arm C and S16 A7 share an architecture (verified bitwise: identical
layer shapes, dropout 0.3, plain `Linear(32,1)` head, 1,033,793 trainable parameters)
but NOT a training recipe. S12A5 used `Adam(lr 5e-4, wd 1e-4)`, coupled decay, no
warmup, no cosine schedule, fixed clipping, 200 max epochs. S16 uses
`AdamW(lr 3e-4, wd 1e-3)`, decoupled, linear warmup over 10% of steps, cosine decay,
min 80 / max 400 epochs, adaptive p90 clipping. **Any A7 C6-vs-C2 difference reflects
training-set size AND recipe jointly and isolates neither.**

## 7. PREDICTION-LEVEL OUTPUT
Every cell writes `<unit>__<fold>.pred.json` (schema `s16-pred-1`) carrying subject IDs,
test indices, true and used labels, FC score, learned score, head and EMA scores, the
selected fused score, the selected alpha, the full inner alpha curve, both fold
baselines, and the git SHA / config hash / data hashes / checkpoint and feature SHAs.
Every reported aggregate must be reproducible from these files alone.
