# S17 Wave 1 — PRE-REGISTERED PLAN

Written and committed **before** Wave 1 is submitted. No Wave-1 production result
exists at the time of writing. Every decision rule below is fixed here so that it
cannot be chosen after seeing the numbers.

Parent commit: `784051c` (S17 Wave 1 implementation).
Policy: the **frozen S16 PROD policy** `798ed7790c1ddabc`, used exactly as it is.

---

## 1. The question

S16 C6 answered the original question: across 1,431 validated cells, no learned
encoder beat a linear SVM on raw FC edges, and score-level fusion selected pure FC
(alpha = 1.0) in 58.7% of folds. Every graph encoder sat at 0.53-0.63 probe_honest
against an SVM reference of 0.7565.

Branch R asks one narrow follow-up: **does keeping the 90 ROI rows separate and
ordered, instead of pooling them, recover the FC signal that S12B showed was
destroyed at `global_add_pool`?** RowMLPR applies one shared MLP to each ROI row
independently and flattens all 90 outputs in order. It never sums or pools.

## 2. Arms

| arm | spec | input | role |
|---|---|---|---|
| R1s | `fcrow_signed` | FC rows, signed, diagonal zeroed | Branch R |
| R1a | `fcrow_abs` | \|FC\| rows | Branch R |
| R1p | `fcrow_split` | `concat[max(R,0), max(-R,0)]`, 180 cols | Branch R |
| A7 | `edgetri` | 4005-edge upper triangle (EdgeMLP) | **S16 reference** |

A7 is included so Branch R is judged against a contemporaneous baseline run under
the identical policy, rather than against a number copied from the C6 table.

4 arms x 3 seeds x 9 folds (3 LAB + 3 SITE + 3 LOSO) = **108 folds**.

## 3. Pre-registered decision gates

**G1 — keep or drop Branch R.**
Keep if the best Branch-R arm reaches **probe_honest >= 0.70**.
Drop if it falls **below 0.65**.
Between 0.65 and 0.70 is an explicit no-decision band: the branch is neither kept nor
dropped on Wave 1 alone and the outcome is reported as inconclusive.

**G2 — channel winner.**
The winner is **R1p** unless **R1s or R1a beats it by more than 0.01** probe_honest.
A margin of 0.01 or less is a tie and R1p is retained, because the split channel is
the more expressive parameterisation and ties should not be broken by noise.

**G3 — "adds value".**
An arm adds value only if its **fused delta > 0 in more than half of its folds**.
The delta is measured against **each fold's own `svm_tr_enc`**, never against a
pooled or historical baseline. Alpha = 1.0 is the FC fallback ENDPOINT, not a floor;
the selected-alpha outer-test delta may be negative and is reported unclamped.

**G4 — training validity.**
A fold is valid only if **movement_max > 0.10 AND clip_rate < 0.30**.
Folds failing G4 are reported but are not used to satisfy G1, G2 or G3.
This gate is stated in advance because S16 C6 returned movement_max median 0.063,
with 980 of 1,431 folds at or below 0.10 — if Wave 1 repeats that, the arms are
inert and no AUC comparison between them means anything.

**LOSO is descriptive, not decisive.** Test sets are 20 (CMU) / 28 (CALTECH) /
47 (KKI) subjects, so per-fold AUC is high-variance. Gates G1-G3 are decided on LAB
and SITE; LOSO is reported alongside but does not by itself keep or drop a branch.

**Reporting rule.** probe_honest and the fused delta are reported for
**LAB, SITE and LOSO separately and are never pooled.** They are three distinct
estimands; LOSO ran ~0.044 below the ordinary protocols in S16, and averaging them
would import that shift into every number.

## 4. What is frozen and must not change

- PROD policy `798ed7790c1ddabc`: `max_epochs=400`, `min_epochs=80`, `patience=50`,
  `batch=32`, warmup 0.10, cosine floor 0.05, label smoothing 0.05.
- **`max_epochs` in particular.** `total_steps = steps_per_epoch x max_epochs` sets
  the cosine LR schedule, so changing it would alter the learning rate at every step
  and destroy comparability with S16. It is asserted at submission time.
- Optimizer: AdamW, **lr = 3e-4, wd = 1e-3**, betas (0.9, 0.999), eps 1e-8, loss
  L-BCE — identical to every S16 arm, taken from `s16_prov.model_cfg`.
- The frozen cohort, the four E caches, and all fold definitions.
- The S16 ledger hash `8587b1ca36553408`.

Nothing in Wave 1 writes under `audit/s16/`. S17 has its own namespace root, which
asserts `/audit/s17/` and rejects `/audit/s16/`.

## 5. Failure handling

Each array task runs one fold inside try/except. A failure writes
`runs/prod/failed/<unit>__<fold>.json` with the traceback. Once more than **5%** of
attempted folds have failed, a `POISON` marker is written and the whole array is
cancelled, because a systematic fault is far more likely than 6 independent ones.

## 6. What Wave 1 does NOT decide

Wave 1 covers the three input channels and one architecture at E = signed only. It
does not test groups, ALFF branches, attention, other E levels, or Waves 2-4. No
result here authorises any of those.
