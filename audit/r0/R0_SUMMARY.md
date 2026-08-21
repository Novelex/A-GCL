# R0 SUMMARY — released_code_compat_08339b7 (FROZEN)
run: seed 123, 100 epochs, 21 eval events, 1312 s, job 1870556, tests 8/8 PASS

==============================================================================
R0 UPSTREAM-FIRST RESULTS (test, released 10-fold protocol)
==============================================================================
pretraining (event 0, ep0): acc=0.5190+-0.0639 f1=0.4070 sen=0.3515 spe=0.6642
final (ep100)             : acc=0.4947+-0.0426 f1=0.3748 sen=0.3244 spe=0.6530

per-metric INDEPENDENT MAXIMIZATION over the 100-entry upstream curve:
  acc: max=0.5241 at curve_epoch 30 (source ckpt ep30) | final-epoch value 0.4947 | max-final +0.0294
  f1: max=0.4366 at curve_epoch 25 (source ckpt ep25) | final-epoch value 0.3748 | max-final +0.0617
  sen: max=0.4002 at curve_epoch 15 (source ckpt ep15) | final-epoch value 0.3244 | max-final +0.0759
  spe: max=0.7068 at curve_epoch 85 (source ckpt ep85) | final-epoch value 0.6530 | max-final +0.0538

20 post-training events (acc): mean=0.5032 sd=0.0113 min=0.4822 max=0.5241
  ^ labelled: Combined checkpoint and regenerated-split volatility
100-entry curve (acc)        : mean=0.5042 sd=0.0114 min=0.4822 max=0.5241
AUC diagnostic_not_upstream_parity: mean=0.5086 range [0.4709,0.5283]  (never used for selection)

RNG: 21 distinct RNG-state hashes over 21 events (seeded once, advancing stream)
fold-hash sets all distinct: True
paper Table 2 (AAL1, 5-fold): accuracy 80.65 claimed elsewhere / SVM row 66.37 — R0 released procedure on verified data: 52.41% best, 49.47% final

## PERMANENT STOP
Reproduction is CLOSED. R0 is frozen: no F2-F7, no further bug hunting, no tuning,
no claim of corrected accuracy, no claim that 80.65% was reproduced. Accuracy above is
an OBSERVATION under the released protocol, not a reproduction pass criterion and not
an unbiased generalization estimate (see R0_PREREGISTRATION.md disclaimer).
