"""
Layer 2 follow-up: re-run the classical-ML LinearSVC baseline (old ALFF vs
new ALFF), using the FIXED per-band min-max normalization -- the same
normalization datasets/abideDataset.py now applies -- instead of the
z-score convention the original Step 1.6 comparison used
(scripts/compute_alff_baseline.py). This tests directly whether the
joint-vs-per-band normalization bug found in this layer actually matters
for downstream classification accuracy, using the graph-construction-
relevant normalization rather than a different one chosen for a different
purpose (fair z-score comparability, in the original Step 1.6).

LinearSVC only, by request -- not the ElasticNet-LR comparison this time.

Usage:
    python layertesting/layer2/test_layer2_normalization_fix.py
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


def per_band_minmax(raw: np.ndarray) -> np.ndarray:
    """raw: [N, 90, 3]. Per-subject, per-band min-max to [0,1] -- the FIX,
    matching datasets/abideDataset.py's corrected normalization."""
    x = torch.as_tensor(raw, dtype=torch.float64)
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
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

    old_fixed = per_band_minmax(old_raw).reshape(len(file_ids), 270)
    new_fixed = per_band_minmax(new_raw).reshape(len(file_ids), 270)

    logging.info("N=%d, ASD=%d, NC=%d", len(y), int((y == 1).sum()), int((y == 0).sum()))
    logging.info("=== Old ALFF + per-band min-max fix | LinearSVC ===")
    old_result = evaluate(old_fixed, y, "Old ALFF (fixed norm)")

    logging.info("=== New ALFF + per-band min-max fix | LinearSVC ===")
    new_result = evaluate(new_fixed, y, "New ALFF (fixed norm)")

    logging.info("")
    logging.info("=== SUMMARY: per-band min-max fix vs. original (z-scored) Step 1.6 ===")
    logging.info("%-28s | %-20s | %-20s", "Features", "Accuracy", "AUC")
    logging.info("%-28s | %.4f +- %.4f    | %.4f +- %.4f",
                 "Old ALFF (fixed norm)", old_result["acc_mean"], old_result["acc_std"],
                 old_result["auc_mean"], old_result["auc_std"])
    logging.info("%-28s | %.4f +- %.4f    | %.4f +- %.4f",
                 "New ALFF (fixed norm)", new_result["acc_mean"], new_result["acc_std"],
                 new_result["auc_mean"], new_result["auc_std"])
    logging.info("(original Step 1.6, z-scored, LinearSVC: Old=0.5722/0.5916, New=0.5742/0.5952)")
    logging.info("New ALFF delta from fix: acc %+.4f, auc %+.4f",
                 new_result["acc_mean"] - 0.5742, new_result["auc_mean"] - 0.5952)
    logging.info("Old ALFF delta from fix: acc %+.4f, auc %+.4f",
                 old_result["acc_mean"] - 0.5722, old_result["auc_mean"] - 0.5916)

    out_path = REPO_ROOT / "layertesting/layer2/normalization_fix_results.npz"
    np.savez(out_path, old_result=old_result, new_result=new_result)
    logging.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
