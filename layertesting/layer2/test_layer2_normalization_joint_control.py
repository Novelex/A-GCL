"""
Layer 2 follow-up #2: the JOINT-min-max control.

test_layer2_normalization_fix.py showed per-band min-max beats the original
z-scored Step 1.6 baseline for BOTH old and new ALFF -- but that comparison
changed two things at once (z-score -> min-max normalization TYPE, and
joint -> per-band SCOPE), so it doesn't cleanly isolate whether the
joint-vs-per-band bug found in this layer is actually what matters.

This script fills in the missing cell: JOINT (global, all 3 bands combined)
min-max -- the exact normalization datasets/abideDataset.py used BEFORE the
fix -- for both old and new ALFF. Together with the already-saved per-band
results (normalization_fix_results.npz), this completes the real 2x2:

    {joint min-max, per-band min-max} x {old ALFF, new ALFF}

holding the normalization TYPE (min-max, not z-score) fixed and varying only
the one thing the fix actually changed: joint vs per-band scope.

LinearSVC only, by request.

Usage:
    python layertesting/layer2/test_layer2_normalization_joint_control.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from compare_alff import load_old_alff  # noqa: E402

SEED = 123
N_SPLITS = 5
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]


def joint_minmax(raw: np.ndarray) -> np.ndarray:
    """raw: [N, 90, 3]. Per-subject GLOBAL min-max to [0,1] -- a single
    min/max over all 3 bands combined, matching the ORIGINAL (pre-fix)
    datasets/abideDataset.py normalization (single x.min()/x.max() over the
    whole 90x3 tensor)."""
    x = torch.as_tensor(raw, dtype=torch.float64)
    x_min = x.amin(dim=(1, 2), keepdim=True)
    x_max = x.amax(dim=(1, 2), keepdim=True)
    span = x_max - x_min
    scaled = torch.where(span > 0, (x - x_min) / span, x)
    return scaled.numpy()


def evaluate(X: np.ndarray, y: np.ndarray, name: str) -> dict:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    accs, aucs = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        pipeline = Pipeline([("scaler", StandardScaler()),
                              ("clf", LinearSVC(dual=False, max_iter=10000))])
        clf = GridSearchCV(pipeline, {"clf__C": C_GRID}, cv=5, scoring="accuracy", n_jobs=4)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        dec = clf.decision_function(X[test_idx])
        acc = accuracy_score(y[test_idx], pred)
        auc = roc_auc_score(y[test_idx], dec)
        accs.append(acc)
        aucs.append(auc)
        logging.info("  [%s] fold %d/%d: acc=%.4f auc=%.4f best_C=%s",
                     name, fold, N_SPLITS, acc, auc, clf.best_params_["clf__C"])
    result = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
              "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}
    logging.info("[%s] acc: %.4f +- %.4f | auc: %.4f +- %.4f",
                 name, result["acc_mean"], result["acc_std"], result["auc_mean"], result["auc_std"])
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                         datefmt="%H:%M:%S")

    npz = np.load(REPO_ROOT / "data/ALFF_need/alff_new.npz", allow_pickle=True)
    file_ids = npz["file_ids"].tolist()
    dx_group = npz["dx_group"]
    y = (dx_group == 1).astype(int)

    old_raw = load_old_alff(REPO_ROOT / "data/raw", file_ids)  # already z-scored (Layer 1)
    new_raw = npz["alff"]  # raw (Layer 1)

    old_joint = joint_minmax(old_raw).reshape(len(file_ids), 270)
    new_joint = joint_minmax(new_raw).reshape(len(file_ids), 270)

    logging.info("N=%d, ASD=%d, NC=%d", len(y), int((y == 1).sum()), int((y == 0).sum()))
    logging.info("=== Old ALFF + JOINT min-max (original pre-fix normalization) | LinearSVC ===")
    old_result = evaluate(old_joint, y, "Old ALFF (joint norm)")

    logging.info("=== New ALFF + JOINT min-max (original pre-fix normalization) | LinearSVC ===")
    new_result = evaluate(new_joint, y, "New ALFF (joint norm)")

    # per-band results from the prior run (test_layer2_normalization_fix.py),
    # hardcoded here the same way that script hardcoded the z-scored Step 1.6
    # baseline, to print the full 2x2 in one place
    per_band = {
        "old": {"acc_mean": 0.5837, "acc_std": 0.0102, "auc_mean": 0.6085, "auc_std": 0.0250},
        "new": {"acc_mean": 0.5795, "acc_std": 0.0264, "auc_mean": 0.6031, "auc_std": 0.0291},
    }

    logging.info("")
    logging.info("=== FULL 2x2: {joint, per-band} min-max x {old, new} ALFF | LinearSVC ===")
    logging.info("%-28s | %-20s | %-20s", "Features", "Accuracy", "AUC")
    logging.info("%-28s | %.4f +- %.4f    | %.4f +- %.4f",
                 "Old ALFF (joint)", old_result["acc_mean"], old_result["acc_std"],
                 old_result["auc_mean"], old_result["auc_std"])
    logging.info("%-28s | %.4f +- %.4f    | %.4f +- %.4f",
                 "Old ALFF (per-band)", per_band["old"]["acc_mean"], per_band["old"]["acc_std"],
                 per_band["old"]["auc_mean"], per_band["old"]["auc_std"])
    logging.info("%-28s | %.4f +- %.4f    | %.4f +- %.4f",
                 "New ALFF (joint)", new_result["acc_mean"], new_result["acc_std"],
                 new_result["auc_mean"], new_result["auc_std"])
    logging.info("%-28s | %.4f +- %.4f    | %.4f +- %.4f",
                 "New ALFF (per-band)", per_band["new"]["acc_mean"], per_band["new"]["acc_std"],
                 per_band["new"]["auc_mean"], per_band["new"]["auc_std"])
    logging.info("")
    logging.info("Per-band minus joint (isolated joint-vs-per-band effect, holding min-max fixed):")
    logging.info("  Old ALFF: acc %+.4f, auc %+.4f",
                 per_band["old"]["acc_mean"] - old_result["acc_mean"],
                 per_band["old"]["auc_mean"] - old_result["auc_mean"])
    logging.info("  New ALFF: acc %+.4f, auc %+.4f",
                 per_band["new"]["acc_mean"] - new_result["acc_mean"],
                 per_band["new"]["auc_mean"] - new_result["auc_mean"])

    out_path = REPO_ROOT / "layertesting/layer2/normalization_joint_control_results.npz"
    np.savez(out_path, old_result=old_result, new_result=new_result, per_band=per_band)
    logging.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
