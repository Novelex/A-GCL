"""
Step 1.5 (docs/COMPLETE_PLAN.md.pdf): compare old ALFF (norm_matrix, the
existing per-subject per-band z-scored .mat node features -- see
correction.md's z-score confirmation) against new ALFF (data/ALFF_need/
alff_new.npz, computed from the ROI-averaged time series per Step 1.2-1.4).

For each ROI and band, correlate old against new across subjects.

Fairness note: old ALFF is already z-scored per subject per band (mean 0,
std 1 across the 90 ROIs). To make this a fair like-for-like comparison
(not an artifact of one side being normalized and the other not), the new
ALFF is z-scored the same way -- per subject, per band, across ROIs --
before correlating. Pearson r is invariant to this rescaling PROVIDED it's
applied consistently; the point is not to introduce a spurious mismatch by
comparing a normalized quantity to a raw one.

Usage:
    python scripts/compare_alff.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scipy.io as sio


def load_old_alff(raw_dir: Path, file_ids: list[str]) -> np.ndarray:
    """Returns old_alff [N, 90, 3] in the same subject order as file_ids,
    read from data/raw/{ASD,NC}_NF/{file_id}_nf.mat's 'norm_matrix' key."""
    # figure out which folder (ASD_NF or NC_NF) each file_id lives in, once
    location: dict[str, Path] = {}
    for folder in ("ASD_NF", "NC_NF"):
        folder_path = raw_dir / folder
        for fname in folder_path.iterdir():
            if fname.name.endswith("_nf.mat"):
                fid = fname.name[: -len("_nf.mat")]
                location[fid] = fname

    missing = [fid for fid in file_ids if fid not in location]
    if missing:
        raise FileNotFoundError(f"{len(missing)} subjects missing an _nf.mat file, e.g. {missing[:5]}")

    old = np.zeros((len(file_ids), 90, 3), dtype=np.float64)
    for i, fid in enumerate(file_ids):
        mat = sio.loadmat(location[fid])
        old[i] = np.nan_to_num(mat["norm_matrix"])
    return old


def zscore_per_subject_per_band(x: np.ndarray) -> np.ndarray:
    """x: [N, 90, 3]. Z-score across the ROI axis (90), independently per
    subject and per band -- matching how old ALFF (norm_matrix) was already
    normalized upstream (DPARSFA's zALFFMap convention)."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)  # guard divide-by-zero, shouldn't occur
    return (x - mean) / std


def per_roi_band_correlation(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """old, new: [N, 90, 3], already normalized comparably. Returns r [90, 3]
    -- Pearson correlation across the N subjects, for each ROI and band."""
    n_rois, n_bands = old.shape[1], old.shape[2]
    r = np.full((n_rois, n_bands), np.nan)
    for roi in range(n_rois):
        for band in range(n_bands):
            o = old[:, roi, band]
            nnew = new[:, roi, band]
            if o.std() == 0 or nnew.std() == 0:
                continue
            r[roi, band] = np.corrcoef(o, nnew)[0, 1]
    return r


BAND_NAMES = ["slow-5", "slow-4", "classical"]


def summarize(r: np.ndarray) -> None:
    flat = r[np.isfinite(r)]
    logging.info("Overall: %d/%d (ROI,band) pairs had finite r", flat.size, r.size)
    logging.info("Overall r: mean=%.4f median=%.4f min=%.4f max=%.4f",
                 flat.mean(), np.median(flat), flat.min(), flat.max())

    agree = (flat > 0.95).sum()
    mid = ((flat >= 0.6) & (flat <= 0.9)).sum()
    low = (flat < 0.5).sum()
    logging.info("r > 0.95 (agree):        %4d / %d (%.1f%%)", agree, flat.size, 100 * agree / flat.size)
    logging.info("r in [0.6, 0.9] (differ): %4d / %d (%.1f%%)", mid, flat.size, 100 * mid / flat.size)
    logging.info("r < 0.5 (very different): %4d / %d (%.1f%%)", low, flat.size, 100 * low / flat.size)

    logging.info("Per-band breakdown:")
    for b, name in enumerate(BAND_NAMES):
        band_r = r[:, b]
        band_r = band_r[np.isfinite(band_r)]
        logging.info("  %-10s mean=%.4f median=%.4f min=%.4f max=%.4f  "
                     "(>0.95: %d, 0.6-0.9: %d, <0.5: %d)",
                     name, band_r.mean(), np.median(band_r), band_r.min(), band_r.max(),
                     (band_r > 0.95).sum(), ((band_r >= 0.6) & (band_r <= 0.9)).sum(),
                     (band_r < 0.5).sum())

    if flat.mean() > 0.95:
        verdict = "AGREE (r > 0.95) -- old ALFF was fine, go to Phase 2 with it."
    elif flat.mean() >= 0.5:
        verdict = "MEANINGFULLY DIFFERENT (0.5 <= r <= 0.9-ish) -- continue to Step 1.6."
    else:
        verdict = "VERY DIFFERENT (r < 0.5) -- continue to Step 1.6, and double-check both."
    logging.info("Verdict (based on overall mean r): %s", verdict)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                         datefmt="%H:%M:%S")

    npz_path = Path("data/ALFF_need/alff_new.npz")
    d = np.load(npz_path, allow_pickle=True)
    file_ids = d["file_ids"].tolist()
    new_alff = d["alff"]
    logging.info("Loaded new ALFF: %s (%d subjects)", npz_path, len(file_ids))

    old_alff = load_old_alff(Path("data/raw"), file_ids)
    logging.info("Loaded old ALFF (norm_matrix) for %d subjects", len(file_ids))

    old_z = old_alff  # already z-scored per subject per band, as loaded
    new_z = zscore_per_subject_per_band(new_alff)

    # sanity: confirm old really is already ~z-scored (mean~0, std~1 per subject per band)
    old_mean_abs = np.abs(old_z.mean(axis=1)).mean()
    old_std_mean = old_z.std(axis=1).mean()
    logging.info("Old ALFF sanity: mean(|per-subject-band mean|)=%.6f (expect ~0), "
                 "mean(per-subject-band std)=%.4f (expect ~1)", old_mean_abs, old_std_mean)

    r = per_roi_band_correlation(old_z, new_z)
    summarize(r)

    out_path = Path("data/ALFF_need/alff_correlation.npz")
    np.savez(out_path, r=r, file_ids=np.array(file_ids))
    logging.info("Saved per-(ROI,band) correlation matrix: %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
