"""
Step 1.6 (docs/COMPLETE_PLAN.md.pdf) -- THE DECISION POINT.

Classical ML on old ALFF vs. new ALFF: same folds, same seed, ALFF features
only, no PCC. Costs zero GPU time, and decides everything downstream:
whichever wins is the ALFF used from here on.

Two classifiers, both run on both feature sets, for a 2x2 comparison:
  - LinearSVC (the plan's own spec, and what's used everywhere else in this
    project -- the raw-FC baseline, embedding_evaluation.py, the paper's own
    protocol)
  - Elastic Net logistic regression (L1+L2 mix) -- with 270 features and 956
    samples, its built-in sparsity can reveal whether the ALFF signal is
    concentrated in a subset of features or diffuse. If both classifiers
    agree on which ALFF wins, that's much stronger evidence than either
    algorithm's result alone.

Reference from the plan (a different cohort, 979 subjects -- ours is 956,
so exact numbers won't match, but this is the anchor to compare against):

    Old ALFF (voxelwise -> averaged): acc 59%, AUC 0.616  [LinearSVC]
    New ALFF (ROI-mean -> FFT):       ?

Usage:
    # one (feature-set, classifier) combination -- for the SLURM array
    # (see compute_alff_baseline_array.slurm), each of the 4 tasks runs
    # independently and saves its own result file
    python scripts/compute_alff_baseline.py --feature-set old --classifier LinearSVC

    # all 4 combinations sequentially in one process (no array) --
    # convenient for a single local/interactive run
    python scripts/compute_alff_baseline.py

    # once all 4 combination result files exist, print the combined summary
    python scripts/compute_alff_baseline.py --summarize-only
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from compare_alff import load_old_alff, zscore_per_subject_per_band

SEED = 123
N_SPLITS = 5
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
L1_RATIO_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

FEATURE_SETS = ["old", "new"]
FEATURE_LABELS = {"old": "Old ALFF", "new": "New ALFF"}
RESULTS_DIR = Path("data/ALFF_need")


def build_linear_svc():
    pipeline = Pipeline([("scaler", StandardScaler()),
                          ("clf", LinearSVC(dual=False, max_iter=10000))])
    param_grid = {"clf__C": C_GRID}
    return pipeline, param_grid


def build_elasticnet_lr():
    pipeline = Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                                      max_iter=10000, random_state=SEED))])
    param_grid = {"clf__C": C_GRID, "clf__l1_ratio": L1_RATIO_GRID}
    return pipeline, param_grid


CLASSIFIERS = {
    "LinearSVC": build_linear_svc,
    "ElasticNet-LR": build_elasticnet_lr,
}


def result_path(feature_set: str, clf_name: str) -> Path:
    return RESULTS_DIR / f"alff_baseline_{feature_set}_{clf_name}.npz"


def load_features(feature_set: str):
    npz_path = RESULTS_DIR / "alff_new.npz"
    d = np.load(npz_path, allow_pickle=True)
    file_ids = d["file_ids"].tolist()
    dx_group = d["dx_group"]  # ABIDE convention: 1=ASD, 2=control
    y = (dx_group == 1).astype(int)  # project convention: ASD=1, NC=0

    if feature_set == "new":
        raw = d["alff"]
    else:
        raw = load_old_alff(Path("data/raw"), file_ids)

    # "same normalisation" -- identical per-subject-per-band z-score applied
    # to both old and new (old is already z-scored upstream; re-applying is
    # idempotent, and keeps both paths through this code identical)
    z = zscore_per_subject_per_band(raw)
    X = z.reshape(len(file_ids), 270)
    return X, y


def evaluate(X: np.ndarray, y: np.ndarray, feature_name: str, clf_name: str, n_jobs: int) -> dict:
    """Same folds, same seed, varying only the classifier -- StratifiedKFold(5,
    seed=123) with GridSearchCV over each classifier's own grid."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    build_fn = CLASSIFIERS[clf_name]

    accs, aucs = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        pipeline, param_grid = build_fn()
        clf = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=n_jobs)
        clf.fit(X[train_idx], y[train_idx])

        pred = clf.predict(X[test_idx])
        dec = clf.decision_function(X[test_idx])

        acc = accuracy_score(y[test_idx], pred)
        auc = roc_auc_score(y[test_idx], dec)
        accs.append(acc)
        aucs.append(auc)
        logging.info("  [%s | %s] fold %d/%d: acc=%.4f auc=%.4f best_params=%s",
                     feature_name, clf_name, fold, N_SPLITS, acc, auc, clf.best_params_)

    result = {
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
    }
    logging.info("[%s | %s] acc: %.4f +- %.4f | auc: %.4f +- %.4f",
                 feature_name, clf_name, result["acc_mean"], result["acc_std"],
                 result["auc_mean"], result["auc_std"])
    return result


def run_one_combination(feature_set: str, clf_name: str, n_jobs: int = 4) -> None:
    X, y = load_features(feature_set)
    feature_name = FEATURE_LABELS[feature_set]
    logging.info("N=%d, ASD=%d, NC=%d", len(y), int((y == 1).sum()), int((y == 0).sum()))
    logging.info("=== %s | %s === (n_jobs=%d)", feature_name, clf_name, n_jobs)

    result = evaluate(X, y, feature_name, clf_name, n_jobs=n_jobs)

    out_path = result_path(feature_set, clf_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **result)
    logging.info("Saved: %s", out_path)


def print_summary() -> int:
    results = {}
    missing = []
    for feature_set in FEATURE_SETS:
        for clf_name in CLASSIFIERS:
            path = result_path(feature_set, clf_name)
            if not path.exists():
                missing.append(str(path))
                continue
            d = np.load(path)
            results[(FEATURE_LABELS[feature_set], clf_name)] = {k: float(d[k]) for k in d.files}

    if missing:
        logging.error("Missing result file(s), cannot summarize yet:")
        for m in missing:
            logging.error("  %s", m)
        return 1

    logging.info("=== SUMMARY (Step 1.6 decision table, 2x2) ===")
    logging.info("%-10s | %-14s | %-20s | %-20s", "Features", "Classifier", "Accuracy", "AUC")
    for (feature_name, clf_name), r in results.items():
        logging.info("%-10s | %-14s | %.4f +- %.4f    | %.4f +- %.4f",
                     feature_name, clf_name, r["acc_mean"], r["acc_std"], r["auc_mean"], r["auc_std"])
    logging.info("(plan's reference, different 979-subject cohort, LinearSVC only: "
                 "Old ALFF acc=0.59 auc=0.616)")

    logging.info("")
    for clf_name in CLASSIFIERS:
        old_acc = results[("Old ALFF", clf_name)]["acc_mean"]
        new_acc = results[("New ALFF", clf_name)]["acc_mean"]
        winner = "New ALFF" if new_acc > old_acc else "Old ALFF"
        logging.info("Winner by accuracy under %s: %s (old=%.4f, new=%.4f)",
                     clf_name, winner, old_acc, new_acc)

    both_agree = (
        (results[("New ALFF", "LinearSVC")]["acc_mean"] > results[("Old ALFF", "LinearSVC")]["acc_mean"])
        == (results[("New ALFF", "ElasticNet-LR")]["acc_mean"] > results[("Old ALFF", "ElasticNet-LR")]["acc_mean"])
    )
    logging.info("Both classifiers agree on the winner: %s", both_agree)
    logging.info("Per the plan: 'Whichever wins is your ALFF from here on.'")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1.6: classical ML, old ALFF vs new ALFF.")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default=None,
                         help="Run only this feature set (pairs with --classifier). "
                              "Omit to run all 4 combinations sequentially.")
    parser.add_argument("--classifier", choices=list(CLASSIFIERS), default=None,
                         help="Run only this classifier (pairs with --feature-set).")
    parser.add_argument("--summarize-only", action="store_true",
                         help="Skip computation -- just load the 4 saved result files and print "
                              "the combined summary table.")
    parser.add_argument("--n-jobs", type=int, default=4,
                         help="GridSearchCV parallelism (default: 4). Raise this to match "
                              "--cpus-per-task for a faster wall-clock time on the slower "
                              "ElasticNet-LR grid -- doesn't change the result, only the speed.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                         datefmt="%H:%M:%S")
    args = parse_args()

    if args.summarize_only:
        return print_summary()

    if args.feature_set is not None or args.classifier is not None:
        if args.feature_set is None or args.classifier is None:
            raise SystemExit("--feature-set and --classifier must be given together.")
        run_one_combination(args.feature_set, args.classifier, n_jobs=args.n_jobs)
        return 0

    # no args: run all 4 combinations sequentially, then summarize
    for feature_set in FEATURE_SETS:
        for clf_name in CLASSIFIERS:
            run_one_combination(feature_set, clf_name, n_jobs=args.n_jobs)
    logging.info("")
    return print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
