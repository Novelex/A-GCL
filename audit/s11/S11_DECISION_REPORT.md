# S11 DECISION REPORT — DATA-IMPORT GATE + FC COMPRESSION CAPACITY
2026-08-20 | git HEAD f6c49d49761ef3bf82b3d59798ee1e8ace6b62dc | production untouched
All learned transforms (scaler/PCA/RP/PLS/SVM-C) fitted INSIDE inner-CV pipelines; frozen
S3C outer folds + 19-site LOSO; every worker asserted manifest/X_fc/pair-map/split hashes.

## ANSWERS TO THE NINE REGISTERED QUESTIONS
1. CORRECT DATASET IMPORTED? YES — proven, not assumed. 954/455/499/90, exact-ID loading,
   FC source==graph round-trip 0 mismatches (954 subjects), x==frozen M1_B bitwise,
   X_fc==frozen S5.5 representation bitwise (max_abs 0.0). The gate also CAUGHT AND STOPPED
   a real subject-ordering bug on its first run (S5-cache vs split-defining order) — see
   S11_DATA_GATE.md for the recorded S8-S10 fold-drift finding.
2. BASELINE REPRODUCED? YES, exactly: ord 0.7565 (delta -1.6e-5), LOSO 0.7432 (delta -3.5e-5).
3. PCA COMPRESSION: 32 -> 0.7166 | 64 -> 0.7281 | 128 -> 0.7434 | 256 -> 0.7512 (ord);
   LOSO tracks 0.014-0.016 lower throughout. Cost of compression vs RAW: -0.040 / -0.028 /
   -0.013 / -0.005. Retained variance 61/71/81/90%.
4. RANDOM PROJECTION: 32 -> 0.6242 | 64 -> 0.6560 | 128 -> 0.6868 | 256 -> 0.7220
   (3 seeds averaged, no seed selection; seed SD < 0.01). Generic projection retains
   substantial but not full discriminative geometry, strongly dimension-dependent.
   Per the registered caution: this does NOT establish that the signal is "diffuse".
5. SUPERVISED PLS: PEAKS AT 8 COMPONENTS (0.7267/0.7127), then DEGRADES: 16 -> 0.7070,
   32 -> 0.6949, 64 -> 0.6930, 128 (triggered extension) -> 0.6931. Later diagnosis-fitted
   components overfit inner folds rather than add signal. Registered caveat honoured:
   PLS being supervised, PLS-8's strength shows only that a LOW-DIMENSIONAL DIAGNOSIS-AWARE
   basis retains the signal — it does not certify any unsupervised 8-D representation.
6. IS 32-D INTRINSICALLY TOO SMALL? NO. PCA-32 on identity-aligned edges scores 0.7166 —
   a 32-D UNSUPERVISED linear compression retains 95% of the RAW AUC margin over chance,
   while trained A-GCL's 32-D z sat at 0.49 on the same data. Width was never the problem.
   32-D does carry a real, quantified cost (-0.040), essentially eliminated by 128-256.
7. DOMINANT PROBLEM? Not dimensionality (Q6). Not forced compression (PCA-256 = RAW-0.005).
   It is REPRESENTATION + OBJECTIVE ALIGNMENT: the signal lives in anatomically indexed
   FC-pair features (S7.5 ROI-permutation -> chance), and A-GCL both discards that indexing
   (sum pooling) and optimizes an objective that does not select for diagnosis (S8-S10).
8. S12 REPRESENTATION: per the registered table this is CASE A (PCA-32/64 >= 0.70; PCA-64
   within 0.03 of RAW). S12 should be an EDGE-AWARE COMPACT ENCODER PRESERVING ROI-PAIR
   IDENTITY, latent width 64-128 (64 costs ~0.03, 128 ~0.01; do not force 32), evaluated
   inductively and leakage-safe against the 0.7565/0.7432 bar.
9. HARD 80% MASK — S10 STATUS (mandated section): S10 requested an 80% keep target; the
   soft penalty (target_keep 0.80, lambda 2.0) DID NOT BIND — actual keep stayed ~2-25%.
   An exactly enforced 80% hard top-k mask therefore remains SCIENTIFICALLY UNTESTED. It is
   deferred as a forensic reproduction follow-up — NOT because S10 proved it cannot work,
   but because the accuracy-critical evidence points far more strongly to the anatomically
   indexed 4005-edge representation while node-based A-GCL z remained near chance.

## CASE EVALUATION (pre-registered)
  CASE A: MET  (PCA-32 0.7166 >= 0.70; PCA-64 0.7281, within 0.028 of RAW 0.7565)
  CASE B: not met (PCA-32/64 are not weak; PLS-32/64 are NOT close to RAW)
  CASE C: not met (PLS-128 == PLS-64; no late-component gain)
  CASE D: not met (PCA-256 within 0.005 of RAW)
  CASE E: partially observed (RP-256 0.7220 approaches PCA-32; stated per the caution only)

## SCIENTIFIC RULE COMPLIANCE
No method was selected on held-out performance for re-reporting: the decision uses the
pre-registered case table, and any S12 model must be evaluated in a fresh leakage-safe
inductive protocol against the frozen bar.

SINGLE NEXT ACTION: design S12 as an edge-aware compact encoder that preserves fixed
ROI-pair identity (input: the audited 4005-edge representation; latent width 64-128;
supervised or hybrid objective; nested-CV + LOSO evaluation against 0.7565/0.7432).
Await authorization before implementing.

STOP.
