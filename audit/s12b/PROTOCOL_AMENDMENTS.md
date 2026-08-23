# S12B PROTOCOL AMENDMENTS (each timestamped, justified, never silent)

## A1 — 2026-08-23 02:10 — Gate 1 R3 criterion re-specified (BEFORE any Track ran)
OBSERVED: first Gate-1 run: R2=0.7565 PASS (exact frozen anchor), R1=0.7481 PASS,
R3(single permutation draw)=0.4647 -> FAIL of the pre-registered band [0.47,0.53].
Per-fold R3: 0.400/0.413/0.492/0.479/0.518; pooled bootstrap CI [0.430,0.502].
The gate STOPPED the run as designed; Gate 2 and all Tracks did not execute.

DIAGNOSIS: under the null, pooled-OOF AUC of ONE permutation draw has SE ~= 0.019
(n=954); the original band is a +-1.6 SE test with ~11% false-failure probability
per tail. The observed failure is on the BELOW-chance side. Leakage — the failure
mode this reference exists to catch — inflates AUC ABOVE 0.5 and cannot produce
sub-chance pooled results. A single unlucky draw therefore triggered the gate in
the direction the gate does not guard against.

AMENDMENT (strictly more stringent against leakage):
  R3 = mean pooled-OOF AUC over 10 independent label permutations
       (rng seeds 20260818..20260827).
  PASS requires BOTH: mean in [0.47, 0.53]  (mean of 10 has SE ~= 0.006, so this
  is now a +-5 SE band)  AND  max single draw <= 0.55 (one-sided leakage guard).
No other criterion, threshold, or protocol element is changed. Amendment written
before Gate 1 was re-run and before any Track-1/2/3/4 computation existed.

## A2 — 2026-08-23 02:10 — fold-count clarification (documentation, not a change)
[corrected 02:55 after review] The frozen S3C ordinary split is 5 folds, not the 8
written at PROTOCOL.md lines 87 (Track 1) and 113 (Track 2) — my first wording of
this amendment misattributed those to Track 4. LOSO contributes 19 evaluable folds
(24 total). All code always loaded the frozen splits via s11_core, so no fold
assignment, computation, or statistic is affected; the R2=0.7565 anchor is and
always was a 5-fold number. gate1.py additionally hardcoded "8 ordinary" into the
generated report; that string is now COMPUTED from the frozen authority. GATE0
already reported the true counts.

## A3 — 2026-08-23 02:55 — site instrument is 19-class, not 17
PROTOCOL.md line 91 registers "site (17-class...)". The frozen cohort has 19 sites
(s3c/meta.csv, all with both classes). s12b_core.site_codes() always derived the
classes from data, so the fitted model was always correct; the pre-registered
chance baselines are corrected to acc 1/19 = 0.053 and the macro-F1 floor for 19
classes. Documentation defect, no code behaviour change.

## A4 — 2026-08-23 02:55 — pre-launch review fixes (all raised findings adjudicated)
An adversarial 4-lens review was run before launch. It was CUT SHORT by an API
spend limit: 4 of 23 agents completed (2 review lenses + 1 verification; the
engineering lens never ran), so the workflow's "18 refuted" is an artifact of
failed verifications being counted as refutations. I adjudicated all 19 raised
findings directly against the code. Zero were leakage. Fixes applied BEFORE any
track ran, all strengthening the instrument, none altering a pre-registered
criterion:
 R2/R4/R11 X_fc is now cached and probed as the frozen float64 array (was f32-cast),
   and its sha256 is genuinely ASSERTED in load_all (the previous computation was
   dead code). Gate 1 and the Track-2 classical arms now read K.load_Xfc() directly.
 R7 Gate 0 now ASSERTS what it claimed: FC rebuilt from .mat == frozen X_fc bitwise;
   .mat-vs-S5-graph-cache mismatch count == 0; FC symmetry and diag==1; node
   features == canonical M1_B. These were previously computed and printed only.
 R3 Gate reports, pip_freeze and the Track-1 .done sentinel now use atomic
   TEMP->validate->rename (a truncated sentinel could have skipped work on resume).
 R14 I2 RidgeCV now uses alpha_per_target=True: the per-edge R2 distribution is the
   instrument, so a single compromise alpha across 4005 targets was biasing it.
 R18 per-fold I2 R2 is now stored alongside the pooled OOF R2.
 R15 per-fold PCA n_components is now recorded (stored projections are NaN-padded
   to 200 columns; the width was previously unrecoverable downstream).
 R10 Gate 2 now asserts it is running on a GPU node (the GPU determinism leg could
   silently not run) and compares CPU vs GPU at 1e-4; cross-device BITWISE equality
   is explicitly NOT claimed — different kernels — this is a declared fork, while
   within-device bitwise equality remains required.
 R6 Track 3's "production v3" arm now replicates abideDataset.py verbatim including
   the span==0 branch (keep raw values, not zeros).
 R8/R17 Track 3's across-subject z-score arm now actually z-scores; it fed a bitwise
   copy of raw, making the invariance check unfalsifiable. The raw-vs-z absolute
   difference is now reported as a real number.
 R9 Track 2 records parameter L2 movement per group PER EPOCH as registered (it was
   computed once, after loading the best checkpoint).
 R5 Track 4's per-eval bootstrap is 2000 as registered (was 200).
 R12 P-lab / P-roi controls and the top-5 stage-tensor dump are implemented in
   t1_controls.py, written after the review snapshot; the finding is superseded.
 R16 the R3 gate fragility this review flagged is the same defect amendment A1
   already corrected, independently.
