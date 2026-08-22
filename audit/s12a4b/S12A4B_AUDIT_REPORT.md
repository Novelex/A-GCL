# S12A4B — TRAINING VALIDITY AUDIT (audit only; zero new training runs)
STEP 0 — INTEGRITY: PASS. 288/288 checkpoints match their (arm, seed, fold, config); fold
indices bitwise-equal frozen S3C splits; labels/cohort/dataset hashes verified; no 956 cache.

STEP 1 — EARLY STOPPING: max allowed 200. Median best epoch: arm1 28, arm2 28, arm3 18,
arm4 15 (median stopped epoch 35-48; 79-86% of folds best before epoch 50). Per protocol
wording this alone would say: "Training convergence is uncertain." Steps 2/3/5 resolve it.

STEP 2 — PARAMETER MOVEMENT (trainable params only): encoder 15-18% relative change
(GIN weight matrices 17-20%) — genuinely trained, not frozen; readout ~100%; head 8-15%.
Readout dominated movement ~6-7x. METHOD NOTE: a naive state_dict aggregate gives 70-137x
"encoder change" — that is BN running-stat BUFFERS (forward-pass statistics, not gradient
learning) plus zero-init biases; it was identified and excluded. Init states reconstructed
deterministically from the seeded constructors (verified: they bitwise-reproduce the S12A3
arm-B random encoders, 0.6509/0.6565/0.6542 on the same folds).

STEP 3 — GAPS: arm1 train 0.930 / val 0.692 / OOF 0.644 (gap +0.24); arm2 +0.17;
arms3/4 +0.05-0.07. Large supervised gap = OVERFITTING, not capacity-to-fit limitation:
optimization had ample power to memorize train data.

STEP 4 — TRAINED vs RANDOM (exact same-weights init, same frozen folds): flatten retention
init 0.6539 vs trained 0.6429 -> delta -0.0110 (negative in 2/3 seeds). Supervised training
did NOT improve WGIN retention; it slightly degraded it while the readout/head learned.

STEP 5 — CURVES: validation peaked at 27-58% of each run then DECLINED (tail slopes -0.003
to -0.038); early stopping fired on overfitting onset, not truncation. Gradient norms were
not logged in S12A4 (finiteness only); parameter movement is the substitute evidence and
shows encoder gradients flowed.

## PRE-REGISTERED DECISION -> OUTCOME 2
"S12A4 is a valid architecture retention result."
Rationale against Outcome 1: the early best-epochs reflect fast convergence of a small
model, not undertraining — train AUC 0.93, encoder weights moved 15-18%, validation was
already declining (more epochs = more overfitting, not more signal), and the decisive
same-weights test shows training the encoder does not raise retention (delta -0.011).
Caveat kept on record: readout dominated learning 6-7x; a fixed-epoch encoder-heavy
protocol would move encoder weights more — but the declining validation curves and the
negative step-4 delta indicate it would not raise retention.

## RECOMMENDED NEXT (evidence-based only)
S12A5: architecture-side retention intervention — give the ROI-aware readout direct access
to input information (e.g., input-skip/residual concatenation past WGIN propagation),
since every objective tested optimizes the readout against a ~0.65 retention ceiling that
the propagation layer sets at initialization. NOT more epochs; NOT objective/mask tuning.
