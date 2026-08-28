# S17 Wave 1 — RESULTS

Generated 2026-08-28T18:14:54Z. Repo `8db11677fe114fc860b5e88095f44d9c05f62cd9`

SLURM array **1875862**, 108/108 folds, **0 failures**, no POISON. PROD policy `798ed7790c1ddabc` (max_epochs 400, lr 3e-4, wd 1e-3), unchanged.

Gates are those pre-registered in `W1_PLAN.md`, committed before submission.

## probe_honest — LAB / SITE / LOSO never pooled

| arm | LAB | SITE | LOSO *(descriptive)* | LAB+SITE |
|---|---|---|---|---|
| R1s | 0.6788 | 0.6142 | 0.5228 | **0.6465** |
| R1a | 0.6631 | 0.5975 | 0.5426 | **0.6303** |
| R1p | 0.6428 | 0.5979 | 0.5326 | **0.6204** |
| A7 *(S16 ref)* | 0.7190 | 0.6959 | 0.5991 | **0.7075** |

LOSO is descriptive only: 20 (CMU) / 28 (CALTECH) / 47 (KKI) test subjects.

## Gate outcomes

**G1 — DROP.** Best Branch-R arm is R1s at **0.6465** on LAB+SITE. Rule: keep >= 0.70, drop < 0.65. Below the no-decision band.

**G2 — winner R1s, not R1p.** R1s 0.6465 beats R1p 0.6204 by **+0.0261**, exceeding the 0.01 margin. The split channel came last.

**G3 — no arm adds value.**

| arm | folds with delta > 0 | mean delta | alpha=1.0 |
|---|---|---|---|
| R1s | 6/18 | +0.0001 | 50.0% |
| R1a | 3/18 | -0.0043 | 55.6% |
| R1p | 5/18 | -0.0016 | 50.0% |
| A7 | 2/18 | -0.0002 | 77.8% |

**G4 — all Branch-R folds valid.**

| arm | valid | movement median | movement min | clip max |
|---|---|---|---|---|
| R1s | **27/27** | 0.426 | 0.223 | 0.074 |
| R1a | **27/27** | 0.566 | 0.157 | 0.071 |
| R1p | **27/27** | 0.520 | 0.225 | 0.073 |
| A7 | **19/27** | 0.125 | 0.046 | 0.099 |

## What is new here

G4 is the substantive finding. S16 C6 returned movement median 0.063 with ~74% of folds at or below the 0.10 threshold, so 'inert encoder' and 'no signal' could not be told apart. Wave 1 returns movement **0.43-0.57 with 27/27 folds valid for every RowMLP arm** — roughly 8x higher. These arms genuinely trained and still landed at 0.62-0.65, below A7's 0.7075 and far below the 0.7565 SVM reference.

Keeping the 90 ROI rows ordered and unpooled did **not** recover the signal S12B localised to `global_add_pool`. The loss is not only pooling.

Note the reference arm is the marginal one: A7 is valid in just 19/27 folds (movement median 0.125, at the threshold), while the arms under test are healthy.

## Runtime

- per fold: median 47s (min 36, max 65)
- best_epoch median 43, max 167, none at 1; ~79 epochs actually ran of the 400 ceiling
- fast because RowMLPR is 13,249-19,009 params (A7 is 1,033,793), there is no graph message passing, and early stopping halted around epoch 80

## Scope

Wave 1 is E=signed, one architecture, three channels. It does not test groups, ALFF branches, attention, other E levels, or Waves 2-4.
