# S16 C6 — SHORT RUN: design, confirmations, and how to read it
Submitted 2026-08-24. Arrays 1873487 (A), 1873488 (B), 1873497 (C). CPU only.
Results appended when the run completes.

## THREE CONFIRMATIONS REQUESTED BEFORE SUBMISSION

**1. Cross-fit standardisation — KEPT AS BUILT. Reasoning recorded here.**
s_FC and s_learned live on different scales, so an unstandardised alpha would measure
scale, not information. Both are z-scored with mean/sd from the inner split (tr_prb).
The subtlety: the learned probe is FITTED on tr_prb, so its scores there would be
IN-SAMPLE. In-sample scores have inflated spread; z-scoring by an inflated sd shrinks
the learned block's standardised magnitude, which would bias the selected alpha toward
FC and understate the learned contribution. Both sides are therefore made
out-of-sample on tr_prb:
  s_FC      SVM fitted on tr_enc -> tr_prb and te are both unseen.
  s_learned 2-fold cross-fit WITHIN tr_prb for the tr_prb values; fitted on the whole
            of tr_prb for the te values.
Implemented in `s16_feat.scores_for_fusion`. Without it, alpha would measure a fitting
artefact rather than information.

**2. alpha = 1.0 assertion targets `svm_tr_enc`, NOT 0.7565. CONFIRMED.**
`s16_worker.py:111`  ->  `a1_exact = bool(abs(a1_auc - svm_tr_enc) < 1e-12)`
Verified live on ordinary fold 0: alpha=1 AUC **0.7318681319**, svm_tr_enc
**0.7318681319**, difference 0.0, and the fused vector is BITWISE equal to z(s_FC) on
te. Comparing to 0.7565 would have failed on every fold, by construction.

**3. A7 architectural parity with S12A5 arm C — FAILED AS BUILT, NOW FIXED.**
Three differences were found and corrected before submission:
| | S12A5 arm C | S16 A7 as first built | S16 A7 now |
|---|---|---|---|
| net | Linear(4005,256)-ReLU-Dropout(**0.3**)-Linear(256,32) | dropout **0.10** | dropout **0.3** |
| head | plain `Linear(32,1)` | LayerNorm+Dropout+Linear | plain `Linear(32,1)` |
| trainable params | 1,033,793 | (differed) | **1,033,793** |
Dropout is now HARDCODED for A7; the `p` argument is ignored so the S16 default cannot
leak in. Verified: layer shapes identical, dropout 0.3 == 0.3, parameter count equal.

## *** RESIDUAL CONFOUND — ARCHITECTURE PARITY IS NECESSARY BUT NOT SUFFICIENT ***
A7's BRIDGE ROLE IS WITHDRAWN (Gate 6). It is reported as an ordinary EdgeMLP arm.
Bridging would require training-set size to be the ONLY difference. It is not. The TRAINING RECIPE also
differs:
  S12A5 arm C : Adam(lr 5e-4, weight_decay 1e-4) — COUPLED decay, no warmup,
                no cosine schedule, fixed clip, 200 max epochs
  S16 A7      : AdamW(lr 3e-4, wd 1e-3) — DECOUPLED, linear warmup over 10% of steps,
                cosine decay to 0.05*lr, min 80 / max 400 epochs, adaptive p90 clipping
**A7's C6-minus-C2 difference therefore measures training-set size AND recipe
JOINTLY.** It bounds the size effect rather than isolating it. Isolating size alone
would require an additional A7 run under S12A5's exact recipe (~216 extra fast folds);
that has NOT been run and is not claimed.
A directly measured, recipe-free component IS available and should be used as the
reference point. **CORRECTED (Gate 6):** 0.7319 was ONE FOLD's reading, not a
constant. Every fold now reports its own `svm_tr_enc` AND `svm_tr_full` on identical
test subjects, with the paired `size_delta_paired` as the only defensible size
estimate (fold lab0: 0.7319 vs 0.7490, paired +0.0171 — NOT the historical
0.7565-0.7319 = +0.0246, which spans different fold designs and code versions).
The superseded sentence read: the SVM itself trained on tr_enc reads **0.7319** versus **0.7565** on
full tr, i.e. **-0.0246** purely from the 763 -> 610 reduction, with no encoder and no
recipe involved.

## THE GRID
| array | contents | units | fold-runs |
|---|---|---|---|
| A `1873487` | BNT (A5,A6) x 4 E + **EdgeMLP A7 x 4 E (ordinary arm)** | 72 | 648 |
| B `1873488` | WGIN (A1,A4) x 4 E + A3 signed + ALFF ablation | 63 | 567 |
| C `1873497` | controls: C-RAND, C-PERM, C-SHUF, C-ROI x 2 arch x 3 seeds | 24 | 216 |
| | | **159** | **1431** |
E in {signed, abs, pos_zero, shift}, applied to FC EVERYWHERE it appears.
Folds: ordinary 0-2, site-stratified 0-2, LOSO 0-2. Seeds 20260818/19/20.
CPU only; `--gres` absent from every submit script.

## WHAT EACH FOLD REPORTS
`probe_honest` (encoder on tr_enc, probe on tr_prb, scored on te — both sides
out-of-sample) · `probe_old_full` (the historical biased reading, for the delta) ·
**`svm_tr_enc`** (the fold-specific FC comparator; NOT a floor) and **`svm_tr_full`**
with the paired `size_delta_paired` · the full pooled **alpha curve**, 21 points
0->1 in 0.05 steps, on BOTH te and the inner split · alpha selected on the INNER SPLIT
ONLY · the **stacking** variant (logistic regression on [s_FC, s_learned] fitted on the
inner split) with coefficients · the alpha=1.0 bitwise and exact-AUC assertions ·
**`head_ema` and `ema_delta`** — EMA(0.999) evaluated ALONGSIDE the raw
validation-best weights, both reported with the delta, selection by VALIDATION only
(frozen rule, S15 PROTOCOL.md:186 and :200; restored at Correction Gate 2 after being
computed every step and discarded) · `evaluated_state`, a literal record of which
weights produced the reported numbers.

**HEADLINE:  delta = AUC(fused at chosen alpha) - svm_tr_enc(THAT FOLD)**
**alpha=1 is the FC FALLBACK ENDPOINT, not a guaranteed floor. The outer-test delta
MAY BE NEGATIVE and is never clamped or replaced after evaluation.**
**SECONDARY:  delta_vs_svm_tr_full**, the same fold's full-tr comparator.
**WITHDRAWN (final preflight): `AUC(fused) - 0.7565` is NOT reported per cell.**
0.7565 must never be subtracted from an individual fold or site; it survives only as a
labelled historical reference line on a pooled E-LAB plot.  (0.7565 is a full-tr HISTORICAL
reference; it is NOT the floor for C6 arms, whose encoders saw only tr_enc.)

## PRECISION
C2's companion refit (job 1873482, 20 repeats of both random draws) supplies mean±SE
for every pure-bias figure. Single-draw values are noise-dominated at ~95 scoring
subjects (SE ~ +-0.02-0.03) and are not quotable individually.

## PLAIN ENGLISH
This run asks one question: does anything the models learn add value on top of plain
connectivity? To answer it fairly we blend each model's score with the simple linear
model's score, sliding a dial from "all linear" to "all learned", and check whether any
setting of the dial beats pure linear. The dial is set using held-out data only, never
the final test subjects, and we prove that at the "all linear" end the blend reproduces
the linear model exactly — so the comparison cannot be rigged.
One honest caveat is recorded above: we added the edge model as a yardstick to connect
this run's numbers to earlier ones, and we made its architecture identical so the only
change would be how much training data it sees. But its training procedure is also
different from the earlier run, so that yardstick measures both things together, not
data size alone. Where a clean number was available we used it: shrinking the training
set from 763 to 610 subjects costs the linear model 0.0246 of accuracy by itself.
