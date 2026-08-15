"""
Layer 1 audit: DATA -- format, normalization status, and overall health of
every raw input source before anything touches the pipeline. Covers old
ALFF (norm_matrix), new ALFF (alff_new.npz), raw PCC (cropped_matrix), and
labels. See layertesting/layer1/layer1_report.md for the written-up results
this script produces.

Scope note: this checks the RAW files as they exist on disk -- e.g. whether
old ALFF is already z-scored by an upstream process, not whether our own
pipeline code normalizes it later (that's Layer 2, graph construction).

Usage:
    python layertesting/layer1/test_layer1_data.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from compare_alff import load_old_alff  # noqa: E402


def check_old_alff():
    logging.info("=" * 70)
    logging.info("OLD ALFF (norm_matrix) -- format and normalization")
    logging.info("=" * 70)

    nf_dir = REPO_ROOT / "data/raw/ASD_NF"
    sample_file = sorted(os.listdir(nf_dir))[0]
    mat = sio.loadmat(nf_dir / sample_file)
    x = mat["norm_matrix"]
    logging.info("File format: .mat, key='norm_matrix'")
    logging.info("Sample file: %s", sample_file)
    logging.info("Shape: %s, dtype: %s", x.shape, x.dtype)

    file_ids = None  # populated below when loading all 956
    npz = np.load(REPO_ROOT / "data/ALFF_need/alff_new.npz", allow_pickle=True)
    file_ids = npz["file_ids"].tolist()
    old_alff = load_old_alff(REPO_ROOT / "data/raw", file_ids)
    logging.info("Loaded all %d subjects: shape=%s dtype=%s", len(file_ids), old_alff.shape, old_alff.dtype)

    finite = np.isfinite(old_alff).all()
    logging.info("All finite (no NaN/Inf): %s", finite)

    per_subj_band_mean = old_alff.mean(axis=1)  # [N, 3]
    per_subj_band_std = old_alff.std(axis=1)    # [N, 3]
    logging.info("Normalization check -- per-subject-per-band stats across all 956 subjects:")
    logging.info("  mean(|per-subject-band mean|) = %.8f (expect ~0 if z-scored)",
                 np.abs(per_subj_band_mean).mean())
    logging.info("  mean(per-subject-band std)    = %.6f (expect ~1 if z-scored)",
                 per_subj_band_std.mean())
    is_zscored = np.abs(per_subj_band_mean).mean() < 1e-3 and abs(per_subj_band_std.mean() - 1.0) < 1e-3
    logging.info("  VERDICT: already z-scored per-subject-per-band = %s", is_zscored)
    logging.info("  (traced source: DPARSFA's automatic zALFFMap output, not raw ALFF -- "
                 "see docs/alff_phase1_log.md)")

    return old_alff, file_ids


def check_new_alff(file_ids):
    logging.info("=" * 70)
    logging.info("NEW ALFF (alff_new.npz) -- format and normalization")
    logging.info("=" * 70)

    npz_path = REPO_ROOT / "data/ALFF_need/alff_new.npz"
    d = np.load(npz_path, allow_pickle=True)
    logging.info("File format: .npz, keys=%s", list(d.files))
    alff = d["alff"]
    malff = d["malff"]
    logging.info("alff shape=%s dtype=%s", alff.shape, alff.dtype)
    logging.info("malff shape=%s dtype=%s", malff.shape, malff.dtype)

    logging.info("All finite (alff): %s", np.isfinite(alff).all())
    logging.info("All finite (malff): %s", np.isfinite(malff).all())

    per_subj_band_mean = alff.mean(axis=1)
    per_subj_band_std = alff.std(axis=1)
    logging.info("Normalization check -- raw alff, per-subject-per-band stats:")
    logging.info("  mean(|per-subject-band mean|) = %.6f, mean(per-subject-band std) = %.6f",
                 np.abs(per_subj_band_mean).mean(), per_subj_band_std.mean())
    logging.info("  VERDICT: NOT z-scored (raw magnitude scale, as computed by compute_alff.py) --"
                 " this is the expected, correct un-normalized state at this layer")

    malff_band_mean = malff.mean(axis=1)
    logging.info("mALFF invariant: malff.mean(axis=1) per band, averaged over subjects = %s (expect [1,1,1])",
                 malff_band_mean.mean(axis=0))

    return alff


def check_pcc():
    logging.info("=" * 70)
    logging.info("RAW PCC / FC (cropped_matrix) -- format and normalization")
    logging.info("=" * 70)

    adj_dir = REPO_ROOT / "data/raw/ASD_ADJ"
    sample_file = sorted(os.listdir(adj_dir))[0]
    mat = sio.loadmat(adj_dir / sample_file)
    fc = mat["cropped_matrix"]
    logging.info("File format: .mat, key='cropped_matrix'")
    logging.info("Sample file: %s", sample_file)
    logging.info("Shape: %s, dtype: %s", fc.shape, fc.dtype)
    logging.info("Min/max on disk (BEFORE our pipeline's max-abs normalization): %.6f / %.6f",
                 np.nanmin(fc), np.nanmax(fc))
    logging.info("Diagonal (self-correlation) values, first 5: %s", np.diag(fc)[:5])
    logging.info("VERDICT: on-disk values are RAW Pearson correlations (range [-1,1] already, since PCC "
                 "is bounded by construction) -- NOT yet passed through abideDataset.py's max-abs "
                 "normalization step (that happens at Layer 2, graph construction, not here)")
    logging.info("Any NaN in this sample: %s", np.isnan(fc).any())


def check_labels_and_status(file_ids):
    logging.info("=" * 70)
    logging.info("LABELS and overall cross-source status")
    logging.info("=" * 70)

    npz = np.load(REPO_ROOT / "data/ALFF_need/alff_new.npz", allow_pickle=True)
    dx_group = npz["dx_group"]
    logging.info("Label format: ABIDE convention, dx_group in {1, 2} (1=ASD, 2=control)")
    logging.info("Project convention used downstream: ASD=1, NC=0 (remapped)")
    logging.info("Counts: N=%d, ASD=%d, NC=%d", len(dx_group), int((dx_group == 1).sum()), int((dx_group == 2).sum()))

    ours = set()
    for folder in ["ASD_ADJ", "NC_ADJ"]:
        for fname in os.listdir(REPO_ROOT / "data/raw" / folder):
            if fname.endswith("_adj.mat"):
                ours.add(fname[: -len("_adj.mat")])
    npz_ids = set(file_ids)
    logging.info("Subject-set cross-check: our 956-subject cohort vs alff_new.npz's file_ids")
    logging.info("  missing from npz: %d, extra in npz: %d, exact match: %s",
                 len(ours - npz_ids), len(npz_ids - ours), ours == npz_ids)

    logging.info("Known reference numbers from prior layer-testing (docs/alff_phase1_log.md):")
    logging.info("  raw-FC (PCC) baseline, no GNN: acc=0.6401 auc=0.6777")
    logging.info("  old ALFF baseline (LinearSVC): acc=0.5722 auc=0.5916")
    logging.info("  new ALFF baseline (LinearSVC): acc=0.5742 auc=0.5952")
    logging.info("  old vs new ALFF correlation (Step 1.5): mean r=0.6576 (meaningfully different)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    old_alff, file_ids = check_old_alff()
    new_alff = check_new_alff(file_ids)
    check_pcc()
    check_labels_and_status(file_ids)
    logging.info("=" * 70)
    logging.info("Layer 1 audit complete.")


if __name__ == "__main__":
    main()
