"""
ALFF computation from ROI-averaged time series -- docs/COMPLETE_PLAN.md.pdf
Phase 1, Steps 1.2-1.4.

Why this exists (see the plan's "WHY THIS CHANGED"): A-GCL computes ALFF from
the ROI-averaged time series. The node features currently in the dataset
(norm_matrix) were instead computed at every voxel and then averaged -- since
ALFF involves |FFT|, which is nonlinear, average(ALFF(voxels)) != ALFF(average
(voxels)). rois_aal.1D (downloaded by scripts/download_alff_rois.py, C-PAC
pipeline, nofilt_noglobal strategy) is already the ROI-averaged time series --
exactly A-GCL's input.

Usage:
    # Step 1.3 -- verify on ONE subject first. Do not run all subjects until
    # this passes.
    python scripts/compute_alff.py --verify-subject CMU_a_0050642

    # Step 1.4 -- run all subjects (only after verification passes)
    python scripts/compute_alff.py
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from scipy.signal import detrend

# ---------------------------------------------------------------------
# Step 1.2 -- the ALFF script, exactly as specified in
# docs/COMPLETE_PLAN.md.pdf
# ---------------------------------------------------------------------

BANDS = [(0.010, 0.027),   # slow-5
         (0.027, 0.073),   # slow-4
         (0.010, 0.080)]   # classical

# AAL labels >= 9001 are cerebellum (9001-9082) and vermis (9100-9170) --
# dropped per the plan's "Use the first 90 columns (labels < 9001)"
CEREBELLUM_VERMIS_THRESHOLD = 9001
N_ROIS = 90


def compute_alff(ts, tr):
    """ts: [T, 90] ROI-mean time series (unfiltered). Returns [90, 3]."""
    ts = detrend(ts, axis=0)                      # remove linear drift
    n = ts.shape[0]
    nfft = 2 ** int(np.ceil(np.log2(n)))           # zero-pad, matches DPABI
    amp = 2 * np.abs(np.fft.rfft(ts, n=nfft, axis=0)) / n
    freqs = np.fft.rfftfreq(nfft, d=tr)

    alff = np.zeros((ts.shape[1], 3))
    for b, (lo, hi) in enumerate(BANDS):
        m = (freqs >= lo) & (freqs <= hi)
        alff[:, b] = amp[m].mean(axis=0)           # mean amplitude in band
    return alff


def to_malff(alff):
    """Normalise by whole-brain mean, per band."""
    return alff / alff.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------------
# I/O -- rois_aal.1D loading, TR lookup (not specified verbatim in the
# plan's code block, but required by its Notes: "tr comes from your
# subject_tr.csv, per subject" / "Use the first 90 columns (labels < 9001)")
# ---------------------------------------------------------------------

def read_roi_labels(path: Path) -> list[int]:
    """rois_aal.1D's first line is a '#'-prefixed header, one AAL label per
    column (e.g. '#2001\\t#2002\\t...'). Returns the label ints in column order."""
    with open(path) as f:
        header = f.readline().strip()
    if not header.startswith('#'):
        raise ValueError(f"{path}: expected '#'-prefixed header line, got: {header[:50]!r}")
    return [int(tok.lstrip('#')) for tok in header.split()]


def load_roi_timeseries(path: Path) -> np.ndarray:
    """Returns ts [T, 90]: the cortical/subcortical AAL ROIs only (label <
    9001), cerebellum and vermis dropped. No nuisance regression -- C-PAC
    already applied it (per the plan's Notes)."""
    labels = read_roi_labels(path)
    raw = np.loadtxt(path)  # [T, 116] -- '#' header line auto-skipped as a comment
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != len(labels):
        raise ValueError(f"{path}: {raw.shape[1]} data columns but {len(labels)} header labels")

    keep = [i for i, lab in enumerate(labels) if lab < CEREBELLUM_VERMIS_THRESHOLD]
    if len(keep) != N_ROIS:
        raise ValueError(f"{path}: expected {N_ROIS} cortical/subcortical ROIs, found {len(keep)}")

    return raw[:, keep]


def load_subject_tr_table(path: Path) -> dict[str, dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {r['FILE_ID']: r for r in rows}


# ---------------------------------------------------------------------
# Step 1.3 -- verify on ONE subject first
# ---------------------------------------------------------------------

def verify_one_subject(file_id: str, rois_dir: Path, subject_tr: dict[str, dict]) -> bool:
    if file_id not in subject_tr:
        logging.error("%s: not found in subject_tr.csv", file_id)
        return False

    ts_path = rois_dir / f"{file_id}_rois_aal.1D"
    if not ts_path.exists():
        logging.error("%s: %s does not exist", file_id, ts_path)
        return False

    tr = float(subject_tr[file_id]['TR_seconds'])
    ts = load_roi_timeseries(ts_path)
    alff = compute_alff(ts, tr)
    malff = to_malff(alff)

    checks = {
        "shape == (90, 3)": alff.shape == (N_ROIS, 3),
        "all finite (no NaN, no Inf)": bool(np.isfinite(alff).all() and np.isfinite(malff).all()),
        "malff.mean(axis=0) ~= 1.0 for each band": bool(np.allclose(malff.mean(axis=0), 1.0, atol=1e-6)),
        "no zero rows (every ROI has a value)": bool(not (alff == 0).all(axis=1).any()),
    }

    logging.info("Verification for %s (TR=%.3fs, T=%d timepoints):", file_id, tr, ts.shape[0])
    all_passed = True
    for check_name, passed in checks.items():
        logging.info("  [%s] %s", "PASS" if passed else "FAIL", check_name)
        all_passed = all_passed and passed

    logging.info("alff shape=%s malff.mean(axis=0)=%s", alff.shape, malff.mean(axis=0))

    if all_passed:
        logging.info("All 4 checks passed for %s. Safe to run Step 1.4 (all subjects).", file_id)
    else:
        logging.error("One or more checks FAILED for %s. Do not proceed to Step 1.4 yet.", file_id)

    return all_passed


# ---------------------------------------------------------------------
# Step 1.4 -- run all subjects
# ---------------------------------------------------------------------

def run_all_subjects(rois_dir: Path, subject_tr: dict[str, dict], output_path: Path) -> None:
    file_ids = sorted(subject_tr.keys())
    n = len(file_ids)

    alff_all = np.full((n, N_ROIS, 3), np.nan, dtype=np.float64)
    malff_all = np.full((n, N_ROIS, 3), np.nan, dtype=np.float64)
    dx_group = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=bool)

    n_failed = 0
    for i, file_id in enumerate(file_ids):
        row = subject_tr[file_id]
        dx_group[i] = int(row['DX_GROUP'])
        ts_path = rois_dir / f"{file_id}_rois_aal.1D"
        try:
            tr = float(row['TR_seconds'])
            ts = load_roi_timeseries(ts_path)
            alff = compute_alff(ts, tr)
            malff = to_malff(alff)

            if not np.isfinite(alff).all() or not np.isfinite(malff).all():
                raise ValueError("non-finite value in computed ALFF/mALFF")
            if (alff == 0).all(axis=1).any():
                raise ValueError("at least one all-zero ROI row")

            alff_all[i] = alff
            malff_all[i] = malff
            ok[i] = True
        except Exception as exc:
            n_failed += 1
            logging.error("[%d/%d] %s FAILED: %s", i + 1, n, file_id, exc)

        if (i + 1) % 100 == 0 or (i + 1) == n:
            logging.info("[%d/%d] processed (%d failed so far)", i + 1, n, n_failed)

    n_nan = int(np.isnan(alff_all[ok]).any(axis=(1, 2)).sum()) if ok.any() else 0
    n_all_zero_roi = int((alff_all[ok] == 0).all(axis=2).any(axis=1).sum()) if ok.any() else 0

    logging.info("Finished: %d/%d succeeded, %d failed.", int(ok.sum()), n, n_failed)
    logging.info("Post-hoc check -- succeeded subjects with any NaN: %d (expected 0)", n_nan)
    logging.info("Post-hoc check -- succeeded subjects with an all-zero ROI: %d (expected 0)", n_all_zero_roi)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        file_ids=np.array(file_ids),
        alff=alff_all,
        malff=malff_all,
        dx_group=dx_group,
        ok=ok,
    )
    logging.info("Saved: %s (alff/malff shape %s)", output_path, alff_all.shape)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ALFF from rois_aal.1D (docs/COMPLETE_PLAN.md.pdf Phase 1).")
    parser.add_argument("--rois-dir", type=Path, default=Path("data/ALFF_need/rois_aal"),
                         help="Directory of *_rois_aal.1D files (default: data/ALFF_need/rois_aal)")
    parser.add_argument("--subject-tr", type=Path, default=Path("data/subject_tr.csv"),
                         help="Per-subject TR/DX_GROUP table (default: data/subject_tr.csv)")
    parser.add_argument("--output", type=Path, default=Path("data/ALFF_need/alff_new.npz"),
                         help="Where to save the combined [N,90,3] array (default: data/ALFF_need/alff_new.npz)")
    parser.add_argument("--verify-subject", type=str, default=None,
                         help="Step 1.3: run the 4 sanity checks on this one FILE_ID and exit, "
                              "instead of processing all subjects (Step 1.4).")
    parser.add_argument("--max-subjects", type=int, default=None,
                         help="Limit Step 1.4 to the first N subjects (sorted by FILE_ID) -- "
                              "for a quick end-to-end test of the run-all/save pathway before "
                              "committing to the full run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                         datefmt="%H:%M:%S")

    subject_tr = load_subject_tr_table(args.subject_tr)
    logging.info("Loaded TR table: %d subjects", len(subject_tr))

    if args.verify_subject:
        passed = verify_one_subject(args.verify_subject, args.rois_dir, subject_tr)
        return 0 if passed else 1

    if args.max_subjects is not None:
        limited_ids = sorted(subject_tr.keys())[: args.max_subjects]
        subject_tr = {fid: subject_tr[fid] for fid in limited_ids}
        logging.info("--max-subjects %d: limited to %d subjects", args.max_subjects, len(subject_tr))

    logging.warning("Running Step 1.4 (%d subjects). Step 1.3 (--verify-subject) "
                     "should have already passed on at least one subject first.", len(subject_tr))
    run_all_subjects(args.rois_dir, subject_tr, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
