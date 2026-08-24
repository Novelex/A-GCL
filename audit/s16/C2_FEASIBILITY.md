# S16 C2 FEASIBILITY REPORT — counting only, NO refit executed

This report fits no probe, trains no model and scores nothing. It answers one
question: can the site x label matched biased comparator be drawn from `tr`
WITHOUT REPLACEMENT for every fold, seed and half?

Folds: 5 ordinary. Seeds: 20 predefined (20260818..20260837). Halves per seed: 2. Cell checks performed: 6671.

## Per-fold summary

| fold | n_tr | n_te | halves checked | halves feasible | fold feasible |
|---|---|---|---|---|---|
| lab0 | 763 | 191 | 40 | 40 | YES |
| lab1 | 763 | 191 | 40 | 40 | YES |
| lab2 | 763 | 191 | 40 | 40 | YES |
| lab3 | 763 | 191 | 40 | 40 | YES |
| lab4 | 764 | 190 | 40 | 40 | YES |

## Site x label cells — worst case across all seeds and halves

`requested` is the maximum any half asked for; `available in tr` is the minimum
the pool ever held. A cell is infeasible if requested > available at any point.

| fold | site | label | requested (max) | available in tr (min) | feasible |
|---|---|---|---|---|---|
| lab0 | CALTECH | 0 | 4 | 12 | yes |
| lab0 | CALTECH | 1 | 2 | 10 | yes |
| lab0 | CMU | 0 | 1 | 10 | yes |
| lab0 | CMU | 1 | 3 | 6 | yes |
| lab0 | KKI | 0 | 6 | 22 | yes |
| lab0 | KKI | 1 | 5 | 13 | yes |
| lab0 | LEUVEN_1 | 0 | 1 | 13 | yes |
| lab0 | LEUVEN_1 | 1 | 2 | 12 | yes |
| lab0 | LEUVEN_2 | 0 | 1 | 17 | yes |
| lab0 | LEUVEN_2 | 1 | 1 | 14 | yes |
| lab0 | MAX_MUN | 0 | 5 | 21 | yes |
| lab0 | MAX_MUN | 1 | 7 | 16 | yes |
| lab0 | NYU | 0 | 18 | 76 | yes |
| lab0 | NYU | 1 | 14 | 54 | yes |
| lab0 | OLIN | 0 | 4 | 11 | yes |
| lab0 | OLIN | 1 | 4 | 14 | yes |
| lab0 | PITT | 0 | 5 | 21 | yes |
| lab0 | PITT | 1 | 6 | 22 | yes |
| lab0 | SBL | 0 | 2 | 12 | yes |
| lab0 | SBL | 1 | 3 | 12 | yes |
| lab0 | SDSU | 0 | 5 | 15 | yes |
| lab0 | SDSU | 1 | 3 | 8 | yes |
| lab0 | STANFORD | 0 | 4 | 16 | yes |
| lab0 | STANFORD | 1 | 3 | 16 | yes |
| lab0 | TRINITY | 0 | 4 | 20 | yes |
| lab0 | TRINITY | 1 | 2 | 19 | yes |
| lab0 | UCLA_1 | 0 | 8 | 21 | yes |
| lab0 | UCLA_1 | 1 | 5 | 31 | yes |
| lab0 | UCLA_2 | 0 | 1 | 12 | yes |
| lab0 | UCLA_2 | 1 | 2 | 8 | yes |
| lab0 | UM_1 | 0 | 8 | 44 | yes |
| lab0 | UM_1 | 1 | 5 | 39 | yes |
| lab0 | UM_2 | 0 | 4 | 16 | yes |
| lab0 | UM_2 | 1 | 2 | 11 | yes |
| lab0 | USM | 0 | 5 | 18 | yes |
| lab0 | USM | 1 | 6 | 37 | yes |
| lab0 | YALE | 0 | 6 | 22 | yes |
| lab0 | YALE | 1 | 4 | 22 | yes |
| lab1 | CALTECH | 0 | 1 | 15 | yes |
| lab1 | CALTECH | 1 | 2 | 10 | yes |
| lab1 | CMU | 0 | 3 | 8 | yes |
| lab1 | CMU | 1 | 1 | 8 | yes |
| lab1 | KKI | 0 | 7 | 20 | yes |
| lab1 | KKI | 1 | 1 | 18 | yes |
| lab1 | LEUVEN_1 | 0 | 1 | 13 | yes |
| lab1 | LEUVEN_1 | 1 | 3 | 11 | yes |
| lab1 | LEUVEN_2 | 0 | 5 | 13 | yes |
| lab1 | LEUVEN_2 | 1 | 3 | 12 | yes |
| lab1 | MAX_MUN | 0 | 2 | 24 | yes |
| lab1 | MAX_MUN | 1 | 3 | 20 | yes |
| lab1 | NYU | 0 | 18 | 75 | yes |
| lab1 | NYU | 1 | 11 | 61 | yes |
| lab1 | OLIN | 0 | 1 | 14 | yes |
| lab1 | OLIN | 1 | 3 | 15 | yes |
| lab1 | PITT | 0 | 3 | 23 | yes |
| lab1 | PITT | 1 | 7 | 22 | yes |
| lab1 | SBL | 0 | 4 | 10 | yes |
| lab1 | SBL | 1 | 2 | 13 | yes |
| lab1 | SDSU | 0 | 3 | 17 | yes |
| lab1 | SDSU | 1 | 1 | 10 | yes |
| lab1 | STANFORD | 0 | 5 | 14 | yes |
| lab1 | STANFORD | 1 | 4 | 15 | yes |
| lab1 | TRINITY | 0 | 5 | 18 | yes |
| lab1 | TRINITY | 1 | 3 | 17 | yes |
| lab1 | UCLA_1 | 0 | 4 | 25 | yes |
| lab1 | UCLA_1 | 1 | 9 | 27 | yes |
| lab1 | UCLA_2 | 0 | 2 | 11 | yes |
| lab1 | UCLA_2 | 1 | 2 | 8 | yes |
| lab1 | UM_1 | 0 | 10 | 41 | yes |
| lab1 | UM_1 | 1 | 9 | 32 | yes |
| lab1 | UM_2 | 0 | 3 | 16 | yes |
| lab1 | UM_2 | 1 | 2 | 11 | yes |
| lab1 | USM | 0 | 3 | 22 | yes |
| lab1 | USM | 1 | 7 | 35 | yes |
| lab1 | YALE | 0 | 7 | 20 | yes |
| lab1 | YALE | 1 | 7 | 19 | yes |
| lab2 | CALTECH | 0 | 2 | 14 | yes |
| lab2 | CALTECH | 1 | 3 | 9 | yes |
| lab2 | CMU | 0 | 4 | 7 | yes |
| lab2 | CMU | 1 | 1 | 8 | yes |
| lab2 | KKI | 0 | 3 | 25 | yes |
| lab2 | KKI | 1 | 3 | 16 | yes |
| lab2 | LEUVEN_1 | 0 | 4 | 10 | yes |
| lab2 | LEUVEN_1 | 1 | 3 | 11 | yes |
| lab2 | LEUVEN_2 | 0 | 4 | 14 | yes |
| lab2 | LEUVEN_2 | 1 | 3 | 12 | yes |
| lab2 | MAX_MUN | 0 | 7 | 18 | yes |
| lab2 | MAX_MUN | 1 | 2 | 21 | yes |
| lab2 | NYU | 0 | 13 | 80 | yes |
| lab2 | NYU | 1 | 11 | 59 | yes |
| lab2 | OLIN | 0 | 4 | 11 | yes |
| lab2 | OLIN | 1 | 2 | 16 | yes |
| lab2 | PITT | 0 | 5 | 21 | yes |
| lab2 | PITT | 1 | 7 | 21 | yes |
| lab2 | SBL | 0 | 3 | 11 | yes |
| lab2 | SBL | 1 | 4 | 11 | yes |
| lab2 | SDSU | 0 | 3 | 17 | yes |
| lab2 | SDSU | 1 | 2 | 9 | yes |
| lab2 | STANFORD | 0 | 1 | 19 | yes |
| lab2 | STANFORD | 1 | 3 | 16 | yes |
| lab2 | TRINITY | 0 | 6 | 18 | yes |
| lab2 | TRINITY | 1 | 6 | 14 | yes |
| lab2 | UCLA_1 | 0 | 4 | 25 | yes |
| lab2 | UCLA_1 | 1 | 9 | 26 | yes |
| lab2 | UCLA_2 | 0 | 4 | 8 | yes |
| lab2 | UCLA_2 | 1 | 2 | 8 | yes |
| lab2 | UM_1 | 0 | 8 | 42 | yes |
| lab2 | UM_1 | 1 | 8 | 37 | yes |
| lab2 | UM_2 | 0 | 4 | 16 | yes |
| lab2 | UM_2 | 1 | 3 | 10 | yes |
| lab2 | USM | 0 | 4 | 21 | yes |
| lab2 | USM | 1 | 7 | 36 | yes |
| lab2 | YALE | 0 | 5 | 22 | yes |
| lab2 | YALE | 1 | 3 | 24 | yes |
| lab3 | CALTECH | 0 | 3 | 13 | yes |
| lab3 | CALTECH | 1 | 2 | 10 | yes |
| lab3 | CMU | 1 | 2 | 7 | yes |
| lab3 | KKI | 0 | 8 | 20 | yes |
| lab3 | KKI | 1 | 5 | 14 | yes |
| lab3 | LEUVEN_1 | 0 | 5 | 8 | yes |
| lab3 | LEUVEN_1 | 1 | 3 | 11 | yes |
| lab3 | LEUVEN_2 | 0 | 4 | 13 | yes |
| lab3 | LEUVEN_2 | 1 | 1 | 14 | yes |
| lab3 | MAX_MUN | 0 | 4 | 22 | yes |
| lab3 | MAX_MUN | 1 | 3 | 20 | yes |
| lab3 | NYU | 0 | 11 | 85 | yes |
| lab3 | NYU | 1 | 8 | 65 | yes |
| lab3 | OLIN | 0 | 3 | 12 | yes |
| lab3 | OLIN | 1 | 4 | 13 | yes |
| lab3 | PITT | 0 | 7 | 18 | yes |
| lab3 | PITT | 1 | 3 | 26 | yes |
| lab3 | SBL | 0 | 4 | 10 | yes |
| lab3 | SBL | 1 | 4 | 11 | yes |
| lab3 | SDSU | 0 | 4 | 16 | yes |
| lab3 | SDSU | 1 | 2 | 9 | yes |
| lab3 | STANFORD | 0 | 6 | 13 | yes |
| lab3 | STANFORD | 1 | 6 | 12 | yes |
| lab3 | TRINITY | 0 | 2 | 22 | yes |
| lab3 | TRINITY | 1 | 3 | 18 | yes |
| lab3 | UCLA_1 | 0 | 5 | 23 | yes |
| lab3 | UCLA_1 | 1 | 6 | 31 | yes |
| lab3 | UCLA_2 | 0 | 3 | 10 | yes |
| lab3 | UCLA_2 | 1 | 1 | 9 | yes |
| lab3 | UM_1 | 0 | 8 | 44 | yes |
| lab3 | UM_1 | 1 | 8 | 35 | yes |
| lab3 | UM_2 | 0 | 5 | 14 | yes |
| lab3 | UM_2 | 1 | 3 | 9 | yes |
| lab3 | USM | 0 | 5 | 19 | yes |
| lab3 | USM | 1 | 9 | 31 | yes |
| lab3 | YALE | 0 | 2 | 26 | yes |
| lab3 | YALE | 1 | 7 | 19 | yes |
| lab4 | CALTECH | 0 | 5 | 10 | yes |
| lab4 | CALTECH | 1 | 3 | 9 | yes |
| lab4 | CMU | 0 | 3 | 8 | yes |
| lab4 | CMU | 1 | 2 | 7 | yes |
| lab4 | KKI | 0 | 3 | 25 | yes |
| lab4 | KKI | 1 | 4 | 15 | yes |
| lab4 | LEUVEN_1 | 0 | 2 | 12 | yes |
| lab4 | LEUVEN_1 | 1 | 3 | 11 | yes |
| lab4 | LEUVEN_2 | 0 | 3 | 15 | yes |
| lab4 | LEUVEN_2 | 1 | 7 | 8 | yes |
| lab4 | MAX_MUN | 0 | 6 | 19 | yes |
| lab4 | MAX_MUN | 1 | 6 | 15 | yes |
| lab4 | NYU | 0 | 14 | 80 | yes |
| lab4 | NYU | 1 | 12 | 57 | yes |
| lab4 | OLIN | 0 | 3 | 12 | yes |
| lab4 | OLIN | 1 | 3 | 14 | yes |
| lab4 | PITT | 0 | 5 | 21 | yes |
| lab4 | PITT | 1 | 4 | 25 | yes |
| lab4 | SBL | 0 | 1 | 13 | yes |
| lab4 | SBL | 1 | 2 | 13 | yes |
| lab4 | SDSU | 0 | 5 | 15 | yes |
| lab4 | SDSU | 1 | 3 | 8 | yes |
| lab4 | STANFORD | 0 | 2 | 18 | yes |
| lab4 | STANFORD | 1 | 2 | 17 | yes |
| lab4 | TRINITY | 0 | 5 | 18 | yes |
| lab4 | TRINITY | 1 | 5 | 16 | yes |
| lab4 | UCLA_1 | 0 | 7 | 22 | yes |
| lab4 | UCLA_1 | 1 | 3 | 33 | yes |
| lab4 | UCLA_2 | 0 | 2 | 11 | yes |
| lab4 | UCLA_2 | 1 | 3 | 7 | yes |
| lab4 | UM_1 | 0 | 10 | 41 | yes |
| lab4 | UM_1 | 1 | 7 | 37 | yes |
| lab4 | UM_2 | 0 | 2 | 18 | yes |
| lab4 | UM_2 | 1 | 2 | 11 | yes |
| lab4 | USM | 0 | 4 | 20 | yes |
| lab4 | USM | 1 | 7 | 37 | yes |
| lab4 | YALE | 0 | 6 | 22 | yes |
| lab4 | YALE | 1 | 3 | 24 | yes |

## VERDICT: FEASIBLE

Every site x label cell can be matched without replacement for every
fold, seed and half. The bounded estimator is runnable. **It has NOT
been run.**

## Residual confounding NOT addressed in this pass
Matching site x label removes site composition and class balance as explanations
for the paired difference. The following are **deliberately NOT matched** and
remain residual confounds: sex, age, mean framewise displacement (motion), TR.

## Predeclared calibration band
Random-encoder equivalence band **[-0.01, 0.01]**,
declared before any estimate is produced. If the random encoder's mean paired
difference falls outside it, ALL retrospective pure-bias estimates remain
UNRESOLVED and no arm may be described as memorising.
