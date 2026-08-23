# S13 PROTOCOL AMENDMENTS (timestamped, justified, never silent)

## A1 — 2026-08-23 — Gate-2 test 4 re-specified (BEFORE any training result)
OBSERVED: the pre-registered test 4 ("permute ROI order AND the connection-profile
columns identically -> Z_L must be the same up to that permutation, <1e-4") FAILED
at 1.88e+00. The gate stopped the run as designed. No training had been launched.

DIAGNOSIS (measured, not argued):
 (i) The frozen FC is EXACTLY symmetric: max|FC - FC^T| = 0.0, and
     FC[:8] == FC[:8].transpose bitwise. Row profile and column profile are the
     SAME VECTOR. A transposed profile therefore cannot change any number in this
     audit, and the test as worded is VACUOUS for this data — it cannot detect the
     bug it was written to catch.
 (ii) The failure has a different cause. Permuting the connection-profile COLUMNS
     changes which scalar feeds which input unit of Linear(D,H). A model is
     equivariant under that joint permutation only if the input projection's weight
     COLUMNS are permuted with it. This is true of ANY non-equivariant input
     projection and is not a defect.
 (iii) Permuting data AND inp.weight columns together gives
     max|Z_L(perm) - perm(Z_L)| = 1.67e-06, comfortably < 1e-4.

AMENDMENT — test 4 becomes THREE sub-checks, jointly STRICTER than the original
(the original could not fail for a symmetric-FC bug; these can):
 4a PROFILE IDENTITY (bitwise): X[s,i,:90] == FC[s,i,:] for 8 random subjects x 8
    random ROIs. Directly asserts the node feature block IS the FC row.
 4b SYMMETRY PROOF: assert max|FC - FC^T| == 0.0, establishing (not assuming) that
    row and column profiles coincide, so a transposition is provably a no-op.
 4c TRUE EQUIVARIANCE: permute ROI order in the data AND the corresponding columns
    of inp.weight -> max|Z_L(perm) - perm(Z_L)| < 1e-4. This is the check that
    actually guards the dangerous silent bug: an AXIS SWAP feeding [B,D,90] instead
    of [B,90,D]. (Gate-2 test 3 independently guards the same failure by asserting
    the attention map is [B,4,90,90] and not [B,4,93,93].)
Nothing else changes. No threshold, arm, decision rule, or training setting is
touched. Written and committed before Gate 2 was re-run and before any unit trained.

## A2 — 2026-08-23 — T3 (116 ROIs) NOT RUN (recorded in PROTOCOL.md before results)
The frozen ALFF exists only for 90 ROIs: M1 is computed from the 4D NIfTI via atlas
labels 1..90 (s3a_recompute.py), and the 26 cerebellar/vermis regions (codes
>= 9001) were never computed. The 116-column C-PAC ROI timeseries DO exist
(data/ALFF_need/rois_aal/*.1D, 956 subjects, verified), so a 116-ROI build is
technically possible, but it would derive NEW unfrozen data — and the standing
instruction for S13 is to use only what already exists. CONSEQUENCE, STATED
PLAINLY: the 116-vs-90 ROI contribution (the pre-registered T3 - T2 contrast) is
NOT MEASURED by this audit and no claim about it will be made.

## A3 — 2026-08-23 — TRAINING_INTEGRITY: recorded, not fatal (BEFORE any result read)
OBSERVED: 8 of 30 units died on `assert integ["loss_decreased"]`; 545/720 folds
completed. Units affected: T2_K2_wd1e-4_s0, T2_K2_wd1e-3_s0, and ALL of T5 (x3)
and T6 (x3).

DIAGNOSIS (reproduced exactly, fold o2 of T2_K2_wd1e-4_s0):
 (i) THE ASSERTION IS ILL-FORMED AT best_epoch == 1. It tests
     curve[best_epoch-1]["train_loss"] < curve[0]["train_loss"]; when best_epoch
     is 1 this compares epoch 1's loss TO ITSELF (0.90407 < 0.90407 = False). It
     can NEVER pass in that case, whatever the model does. Measured trace: val AUC
     peaks at epoch 1 (0.6531) and training then DEGRADES it (0.5224-0.6051 over
     the next epochs) while train_loss falls monotonically to 0.45027 by epoch 44.
     The model is training correctly; validation simply peaks immediately.
 (ii) FOR NEGATIVE CONTROLS THE ASSERT INVERTS THE CONTROL'S PURPOSE. T5
     (columns shuffled) and T6 (labels permuted) are DESIGNED to carry no signal;
     a model that fails to improve on them is the EXPECTED, REQUIRED result. All
     six control units were killed for behaving exactly as pre-registered.
 (iii) KILLING A 24-FOLD UNIT OVER ONE FOLD DESTROYS VALID EVIDENCE from the
     other 23 folds, which had already been computed.

SELECTION BIAS DISCLOSED: because the assert crashed the unit at the first
violating fold, the 545 completed folds SYSTEMATICALLY EXCLUDE folds whose best
epoch was 1. An earlier inspection reading "zero T2 folds with best_epoch <= 2"
was therefore an artifact of that censoring, not a property of training. All
affected units are re-run under the amended rule and the count of best_epoch==1
folds is reported as a first-class diagnostic in RESULTS.md.

AMENDMENT:
 - ALL integrity flags are still COMPUTED AND RECORDED for every fold.
 - `loss_decreased` is evaluated only when best_epoch > 1; at best_epoch == 1 it is
   recorded as "N/A (self-comparison)" — declared, not silently skipped.
 - HARD KILL is retained ONLY for genuine corruption: non-finite loss, non-finite
   gradients, and checkpoint-reload mismatch. These still abort the run.
 - Integrity violations and best_epoch==1 counts are COUNTED AND REPORTED
   PROMINENTLY in RESULTS.md instead of silently removing folds.
No threshold, arm, decision rule, model setting or training setting is changed.
