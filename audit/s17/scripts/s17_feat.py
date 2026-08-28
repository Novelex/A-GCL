"""S17 Wave 1 input specs: fcrow_signed, fcrow_abs, fcrow_split.

Everything else is imported from S16. No S16 code is copied or forked.

All three specs keep 90 COLUMNS per ROI block so that "column j = ROI j" holds. The
diagonal is ZEROED, never deleted: deleting it would shift every column index by one
past the diagonal and silently break both the column identity and the C-ROI control,
which permutes profile columns by ROI.

Column standardisation is fitted on tr_enc ONLY and applied unchanged to tr_prb and
te, so no cohort statistic ever sees held-out subjects (S16 defect D5).
"""
import sys
import numpy as np
sys.path.insert(0, "/users/3171356m/A-GCL/audit/s16/scripts")
import s16_feat as FT                      # imported, not copied

SPECS = ("fcrow_signed", "fcrow_abs", "fcrow_split")

# n_profile per spec — the number of leading FC-profile columns C-ROI must permute.
# Getting this wrong silently breaks the control, so it is declared explicitly.
N_PROFILE = {"fcrow_signed": 90, "fcrow_abs": 90, "fcrow_split": 180}


def zero_diagonal(R):
    """FC[i,i] -> 0 for every subject, WITHOUT dropping the column."""
    Rz = R.astype(np.float32, copy=True)
    idx = np.arange(Rz.shape[1])
    Rz[:, idx, idx] = 0.0
    return Rz


def zscore_columns(X, tr_enc, eps=1e-8):
    """Z-score each column across subjects AND rows, fitted on tr_enc only.

    X is [N, 90, C]. Statistics are taken over (subjects x rows) for each column c,
    so column c keeps one mean and one sd — the ROI-identity semantics of the column
    are preserved. Returns (Xz, mu, sd) with mu/sd shaped [C] for inspection."""
    tr = np.asarray(tr_enc)
    blk = X[tr].reshape(-1, X.shape[2])            # tr_enc rows only
    mu = blk.mean(axis=0)
    sd = blk.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)               # constant column -> leave centred
    Xz = ((X - mu) / sd).astype(np.float32)
    return Xz, mu.astype(np.float32), sd.astype(np.float32)


def build_X(spec, FCt, tr_enc, control=None, return_stats=False):
    """Build one S17 Wave-1 input.

    FCt is ALREADY E-transformed by the caller, exactly as in s16_feat.build_X.
    tr_enc is the ENCODER training index set; the split happens BEFORE this call.

    Returns (X, FCu) — or (X, FCu, mu, sd) when return_stats=True.
    """
    if spec not in SPECS:
        raise ValueError(f"unknown S17 spec {spec!r}; expected one of {SPECS}")
    R = zero_diagonal(np.asarray(FCt, dtype=np.float32))

    if spec == "fcrow_signed":
        X = R
    elif spec == "fcrow_abs":
        X = np.abs(R)
    else:                                          # fcrow_split
        X = np.concatenate([np.maximum(R, 0.0), np.maximum(-R, 0.0)], axis=2)

    X, mu, sd = zscore_columns(X, tr_enc)

    FCu = FCt
    if control == "C-ROI":
        # n_profile MUST match the number of FC-profile columns for this spec:
        # 90 for signed/abs, 180 for split. apply_c_roi permutes node rows AND the
        # first n_profile columns; passing 90 for the split spec would permute only
        # the positive half and leave the negative half misaligned.
        X, FCu = FT.apply_c_roi(X, FCt, N_PROFILE[spec])

    return (X, FCu, mu, sd) if return_stats else (X, FCu)
