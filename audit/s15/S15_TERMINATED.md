# S15 — TERMINATED 2026-08-24 18:14:11

## What completed
- **1325 fold-runs** of a planned 9570 (13.8%), salvaged to `audit/s15/PARTIAL/`.
- Arms reached: **{'B1': 1240, 'B2': 85}** — the array never got past arm B1 (BNT, FC-row nodes).
  Arms B2, B3, W1, W2, W3 produced **nothing**. All controls and all transductive
  units produced **nothing**.
- Fold protocols: {'lab': 278, 'site': 258, 'loso': 463, 'loso1': 326}.

## Reason for termination
Superseded by S16, which fixes two things S15 structurally cannot:
1. **Fusion floor.** S16's `-fused` arms set repr = concat(raw FC 4005, learned), so
   the probe can always recover the SVM answer by zeroing the learned block. The
   floor is 0.7565 BY CONSTRUCTION (verified bitwise in S16 C4: learned-block-zeroed
   reads 0.7490109890 vs FC-alone 0.7490109890, diff 0.00e+00; pooled 5-fold exactly
   0.7565). S15 has no floor, so a compressing encoder scores below baseline and the
   result is uninformative.
2. **probe_honest.** S16 splits train into tr_enc (80%, encoder only) and tr_probe
   (20%, probe only), so both sides of the probe are out-of-sample.

S15 and S16 were competing for the same ~25 CPU slots; terminating S15 reallocates
the cluster to the better-designed experiment.

## *** ALL S15 NUMBERS WERE PRODUCED WITH THE BIASED PROBE ***
S15 used the S12A5 pattern (audit/s12a5/scripts/w_wave1.py:40-46): `train_fold5`
trains on `tr`; `extract5` then embeds ALL 954 subjects with that model; `probe_pipe`
fits on R[tr] — subjects the encoder MEMORISED — and scores R[te], subjects it never
saw. The two sides come from different distributions and the SVM's C is selected on
the wrong one. Raw FC has no such shift because nothing was fitted to produce it.
**Every AUC below is therefore an upper-biased reading and must not be quoted as a
clean out-of-sample number.** S16 C2 re-scores these same saved representations under
probe_honest; the delta is the measured size of this bias.

## The numbers, recorded as biased
| config | F-LAB probe_old | F-SITE | F-LOSO |
|---|---|---|---|
| B1 K=8 | 0.6540+-0.0310 | 0.6712+-0.0425 | 0.6312+-0.1019 |
| B1 K=32 | 0.6702+-0.0393 | 0.6569+-0.0545 | 0.6395+-0.1027 |
| B2 K=8 | 0.6418+-0.0304 | 0.6434+-0.0387 | 0.6201+-0.1210 |
| B2 K=32 | 0.6512+-0.0366 | 0.6298+-0.0596 | — |

Best single F-LAB fold seen: **0.7607** (biased probe).

## Validity gate on what completed
- movement > 0.10: **310/1325** folds, median 0.0544 (S13 was 0.016-0.039)
- clip_rate < 30%: **1325/1325** folds, median 0.0751 (S13 was 0.92)
- median total optimizer steps 1840 (S13 ~500)
The repaired recipe demonstrably fixed clipping and step count. Parameter movement
improved but **most folds still fail the >0.10 gate**, so most S15 arms would have been
reported UNTRAINED even had the run completed.

## Plain English
We stopped this run early. It had finished about 13% of its work and had only tested
one of its six model setups. We stopped it because a newer, better-designed run (S16)
does the same job with two important fixes, and the two were fighting each other for
the same computers. Everything S15 produced has been kept. But every accuracy number
it produced was measured with a scoring method we have since shown is optimistically
biased, so none of them should be quoted on their own — S16 re-measures the same saved
results properly and reports how large that bias was.
