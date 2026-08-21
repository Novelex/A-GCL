# R0 PREREGISTRATION — written AFTER parity PASS, BEFORE the full run
date: 2026-08-21T23:25:55+01:00 | live HEAD 6ec18cda2f0dd8d4dbc774026b43fb164d60cfab
config: released_code_compat_08339b7 | seed 123 | 100 epochs (README invocation) |
eval before training + after epochs 5..100 (21 evaluator invocations, 100-entry curve).
STATEMENT (verbatim, required): R0 deliberately reproduces transductive SSL, changing
random 10-fold partitions, unweighted downstream embeddings and test-set-based
checkpoint/metric selection. Its score is not an unbiased generalization estimate and
cannot validate Table 2, which reports a different five-fold protocol.
HASHES:
  upstream evaluator (committed) : b71e982d0cee03bd
  upstream queue (committed)     : fdd61b4ff4cea87a
  shipped pyc 310                : 6ccbfb4453275621
  repairs doc                    : a6f4930ece8ab5ea
  r0_core.py                     : 61bbd01eb7c3f427
  upstream_step.py               : 9ddb24e93e5d8fd0
  w_r0_run.py                    : 89f2d096149f28c1
  parity report                  : 1f96397f3254abde
  dataset cache (M1_B)           : 312266b23ecf1348
  environment                    : python 3.12.13, torch 2.5.0+cu121, sklearn 1.9.0
Split-related code: KFold(10,shuffle,random_state=None)+train_test_split(0.2,None) on the
GLOBAL NumPy stream — fold hashes and RNG-state hashes logged per evaluator invocation.
27cafb2932828fc9
