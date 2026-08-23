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
