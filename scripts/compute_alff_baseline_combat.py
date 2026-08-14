"""
Step 1.6 variant -- ComBat site-harmonization before the classical ML
comparison, requested as a follow-up to compute_alff_baseline.py.

Why: ABIDE spans 20 different scanning sites (NYU=173 subjects down to
CMU=21), each with its own scanner/protocol. Site-driven variance in ALFF
could be swamping the (weaker) diagnostic signal, which would explain why
both old and new ALFF score only modestly above chance in the un-harmonized
comparison (see docs/alff_phase1_log.md). ComBat (Johnson et al. 2007,
via the neuroCombat package -- matches the reference R sva::ComBat)
estimates and removes per-site location/scale shifts using an empirical
Bayes model, while explicitly preserving a specified biological covariate
(here, DX_GROUP) so it doesn't remove diagnosis-related signal along with
the site effect.

Methodology, exactly one variable changed vs. compute_alff_baseline.py:
ComBat harmonization is inserted before the existing per-subject-per-band
z-score step; everything else (classifiers, folds, seed, grids) is
identical, so this is a clean A/B test against the already-saved
non-harmonized Step 1.6 results.

Explicit methodological note (per the user's direction): ComBat is fit ONCE
on the full 956-subject dataset, not per-CV-fold ("without splitting"). This
means the site-correction parameters are estimated using all subjects,
including those that later land in a test fold within a given CV split --
a mild form of leakage, though a much weaker one than leaking the label
itself (ComBat only uses site + the preserved DX_GROUP covariate, not
per-subject held-out information about the classifier's own task). Noted
here for transparency, not hidden.

Old ALFF is already z-scored per-subject-per-band (irreversibly, upstream --
see docs/alff_phase1_log.md) before we ever see it, so ComBat runs on that
already-normalized representation for old ALFF. New ALFF's true raw values
are available, so ComBat runs on those directly for new ALFF -- the more
standard order. Both are then z-scored the same way afterward, matching the
existing pipeline exactly.

Usage:
    python scripts/compute_alff_baseline_combat.py --feature-set old --classifier LinearSVC
    python scripts/compute_alff_baseline_combat.py --summarize-only
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

from compare_alff import load_old_alff, zscore_per_subject_per_band
from compute_alff_baseline import (CLASSIFIERS, FEATURE_SETS, FEATURE_LABELS,
                                    RESULTS_DIR, evaluate)


def result_path(feature_set: str, clf_name: str) -> Path:
    return RESULTS_DIR / f"alff_baseline_combat_{feature_set}_{clf_name}.npz"


def load_site_table(path: Path, file_ids: list[str]) -> pd.Series:
    with open(path) as f:
        rows = {r["FILE_ID"]: r["SITE_ID"] for r in csv.DictReader(f)}
    missing = [fid for fid in file_ids if fid not in rows]
    if missing:
        raise FileNotFoundError(f"{len(missing)} subjects missing a SITE_ID, e.g. {missing[:5]}")
    return pd.Series([rows[fid] for fid in file_ids], name="SITE_ID")


def combat_harmonize(raw: np.ndarray, site: pd.Series, dx_group: np.ndarray) -> np.ndarray:
    """raw: [N, 90, 3]. Returns ComBat-harmonized [N, 90, 3], site effects
    removed, DX_GROUP preserved as a categorical covariate so diagnosis
    signal isn't harmonized away along with the site effect."""
    n = raw.shape[0]
    X = raw.reshape(n, 270)

    covars = pd.DataFrame({"SITE_ID": site.values, "DX_GROUP": dx_group})
    out = neuroCombat(dat=X.T, covars=covars, batch_col="SITE_ID", categorical_cols=["DX_GROUP"])
    harmonized = out["data"].T  # back to [N, 270]

    if not np.isfinite(harmonized).all():
        raise ValueError("ComBat output contains non-finite values")

    return harmonized.reshape(n, 90, 3)


def load_features_combat(feature_set: str):
    npz_path = RESULTS_DIR / "alff_new.npz"
    d = np.load(npz_path, allow_pickle=True)
    file_ids = d["file_ids"].tolist()
    dx_group_abide = d["dx_group"]  # ABIDE convention: 1=ASD, 2=control
    y = (dx_group_abide == 1).astype(int)  # project convention: ASD=1, NC=0

    site = load_site_table(Path("data/subject_tr.csv"), file_ids)

    if feature_set == "new":
        raw = d["alff"]  # true raw values -- standard order for ComBat
    else:
        raw = load_old_alff(Path("data/raw"), file_ids)  # already z-scored upstream

    logging.info("Running ComBat: N=%d, %d sites, batch_col=SITE_ID, preserved=DX_GROUP",
                 len(file_ids), site.nunique())
    harmonized = combat_harmonize(raw, site, dx_group_abide)

    # same post-processing as the non-harmonized pipeline, so ComBat is the
    # only variable that differs from compute_alff_baseline.py
    z = zscore_per_subject_per_band(harmonized)
    X = z.reshape(len(file_ids), 270)
    return X, y


def run_one_combination(feature_set: str, clf_name: str, n_jobs: int = 4) -> None:
    X, y = load_features_combat(feature_set)
    feature_name = FEATURE_LABELS[feature_set] + " + ComBat"
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
            results[(feature_set, clf_name)] = {k: float(d[k]) for k in d.files}

    if missing:
        logging.error("Missing result file(s), cannot summarize yet:")
        for m in missing:
            logging.error("  %s", m)
        return 1

    baseline = {}
    for feature_set in FEATURE_SETS:
        for clf_name in CLASSIFIERS:
            from compute_alff_baseline import result_path as base_result_path
            path = base_result_path(feature_set, clf_name)
            if path.exists():
                d = np.load(path)
                baseline[(feature_set, clf_name)] = {k: float(d[k]) for k in d.files}

    logging.info("=== SUMMARY: ComBat-harmonized (Step 1.6 variant) ===")
    logging.info("%-20s | %-14s | %-20s | %-20s | %-14s", "Features", "Classifier", "Accuracy",
                 "AUC", "vs no-ComBat acc")
    for feature_set in FEATURE_SETS:
        for clf_name in CLASSIFIERS:
            r = results[(feature_set, clf_name)]
            base = baseline.get((feature_set, clf_name))
            delta = f"{(r['acc_mean'] - base['acc_mean']) * 100:+.2f} pp" if base else "n/a"
            logging.info("%-20s | %-14s | %.4f +- %.4f    | %.4f +- %.4f    | %s",
                         FEATURE_LABELS[feature_set] + " + ComBat", clf_name,
                         r["acc_mean"], r["acc_std"], r["auc_mean"], r["auc_std"], delta)

    logging.info("")
    for clf_name in CLASSIFIERS:
        old_acc = results[("old", clf_name)]["acc_mean"]
        new_acc = results[("new", clf_name)]["acc_mean"]
        winner = "New ALFF" if new_acc > old_acc else "Old ALFF"
        logging.info("Winner (ComBat-harmonized) by accuracy under %s: %s (old=%.4f, new=%.4f)",
                     clf_name, winner, old_acc, new_acc)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1.6 + ComBat: classical ML on site-harmonized ALFF.")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default=None)
    parser.add_argument("--classifier", choices=list(CLASSIFIERS), default=None)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
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

    for feature_set in FEATURE_SETS:
        for clf_name in CLASSIFIERS:
            run_one_combination(feature_set, clf_name, n_jobs=args.n_jobs)
    logging.info("")
    return print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
