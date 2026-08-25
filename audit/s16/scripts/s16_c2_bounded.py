"""S16 C2 BOUNDED ESTIMATOR — site x label matched retrospective bias measurement.

DESIGN (one outer fold, one repeated test half-split, one seed):
  te is split into halves te[a] | te[b] (stratified by label).
  HONEST comparator   : probe TRAINS on te[a]      -- subjects the encoder NEVER saw
  BIASED comparator   : probe TRAINS on a subset of tr -- subjects the encoder DID see
  BOTH                : score the IDENTICAL te[b]
  MATCHING            : the biased training sample reproduces te[a]'s site x label
                        counts EXACTLY, drawn WITHOUT REPLACEMENT from tr.
  paired difference   : biased_auc - honest_auc, per (fold, seed)

WHAT THIS BOUNDS AND WHAT IT DOES NOT
  Matching site x label removes site composition and class balance as explanations for
  the paired difference. It does NOT remove sex, age, head motion (mean FD) or TR:
  those remain RESIDUAL CONFOUNDS and are deliberately NOT matched in this pass.

INFEASIBILITY POLICY (no silent repair)
  If any site x label cell requests more subjects than tr contains, the estimator
  REFUSES for that fold. It will NOT sample with replacement, NOT pool sites, NOT
  weight, and NOT substitute a different estimator. Retrospective pure bias is then
  reported as UNRESOLVED for that fold.

UNCERTAINTY is Monte Carlo split variability ONLY: the spread produced by re-drawing
the split across predefined seeds. Reported as mean paired difference, SD across
seeds, Monte Carlo standard error of that mean, an empirical interval over the seeds,
and a descriptive count of sign flips. These are NOT population standard errors and
NO formal test is performed or implied."""
import sys, os, json, numpy as np
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_data as DAT
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEEDS = [DAT.BASE + i for i in range(20)]          # 20 predefined seeds
RANDOM_ENCODER_EQUIVALENCE_BAND = (-0.01, 0.01)    # PREDECLARED, before any result

def cohort():
    d, MAN, ent = DAT.load("signed", where="c2bounded")
    return d["y"].astype(np.int64), np.asarray(d["sites"]), d["ids"]

def half_split(te, y, seed):
    """Deterministic label-stratified halving of te."""
    te = np.asarray(te)
    skf = StratifiedKFold(2, shuffle=True, random_state=seed)
    return [(te[a], te[b]) for a, b in skf.split(np.zeros(len(te)), y[te])]

def cell_counts(idx, y, sites):
    c = {}
    for i in np.asarray(idx):
        c[(sites[i], int(y[i]))] = c.get((sites[i], int(y[i])), 0) + 1
    return c

def feasibility(te_a, tr, y, sites):
    """-> (rows, feasible). One row per site x label cell: requested vs available."""
    req = cell_counts(te_a, y, sites)
    avail = cell_counts(tr, y, sites)
    rows, ok = [], True
    for (st, lab), n in sorted(req.items()):
        have = avail.get((st, lab), 0)
        f = have >= n
        ok &= f
        rows.append(dict(site=st, label=int(lab), requested=int(n),
                         available_in_tr=int(have), feasible=bool(f)))
    return rows, ok

def matched_draw(te_a, tr, y, sites, rng):
    """Draw from tr reproducing te_a's site x label counts EXACTLY, WITHOUT
    replacement. Returns None if any cell is infeasible — never repairs."""
    req = cell_counts(te_a, y, sites)
    tr = np.asarray(tr, dtype=int)      # refuse, never crash, on a malformed pool
    if tr.size == 0: return None
    out = []
    for (st, lab), n in sorted(req.items()):
        pool = tr[(sites[tr] == st) & (y[tr] == lab)]
        if len(pool) < n: return None
        out.append(rng.choice(pool, size=n, replace=False))
    return np.concatenate(out) if out else None

def paired_difference(R, y, tr, te, seed, sites):
    """One seed: pooled honest and biased AUC over both halves, and their difference.
    Returns None if any half is infeasible."""
    rng = np.random.default_rng(seed)
    oh = np.full(len(y), np.nan); ob = np.full(len(y), np.nan)
    for te_a, te_b in half_split(te, y, seed):
        sel = matched_draw(te_a, tr, y, sites, rng)
        if sel is None: return None
        _, o1 = K.probe_pipe(np.asarray(R, float), y, [(te_a, te_b)], [])
        _, o2 = K.probe_pipe(np.asarray(R, float), y, [(sel,  te_b)], [])
        oh[te_b] = o1[te_b]; ob[te_b] = o2[te_b]
    m = np.isfinite(oh) & np.isfinite(ob)
    if len(np.unique(y[m])) < 2: return None
    h = float(roc_auc_score(y[m], oh[m])); b = float(roc_auc_score(y[m], ob[m]))
    return dict(honest=h, biased=b, paired_difference=b - h, n_scored=int(m.sum()))

def monte_carlo_summary(diffs):
    """Monte Carlo SPLIT VARIABILITY only. Not a population standard error; no test."""
    d = np.asarray([x for x in diffs if x is not None], float)
    if d.size == 0: return dict(n_seeds=0, note="no feasible seeds")
    return dict(n_seeds=int(d.size),
                mean_paired_difference=float(d.mean()),
                sd_across_seeds=float(d.std(ddof=1)) if d.size > 1 else float("nan"),
                monte_carlo_se_of_mean=(float(d.std(ddof=1)/np.sqrt(d.size))
                                        if d.size > 1 else float("nan")),
                empirical_interval_min=float(d.min()),
                empirical_interval_max=float(d.max()),
                sign_flips_descriptive=int((d < 0).sum()),
                sign_flip_fraction_descriptive=float((d < 0).mean()),
                interpretation=("Monte Carlo split variability across predefined "
                                "seeds. NOT a population standard error. No formal "
                                "test is performed or implied."))

def calibration_verdict(random_encoder_mean):
    lo, hi = RANDOM_ENCODER_EQUIVALENCE_BAND
    passed = (random_encoder_mean is not None) and (lo <= random_encoder_mean <= hi)
    return dict(band=[lo, hi], random_encoder_mean=random_encoder_mean, passed=bool(passed),
                consequence=("calibration PASSED: retrospective bias estimates may be "
                             "interpreted" if passed else
                             "calibration FAILED: ALL retrospective pure-bias estimates "
                             "remain UNRESOLVED and no arm may be described as "
                             "memorising"))

RESIDUAL_CONFOUNDS = ["sex", "age", "mean framewise displacement (motion)", "TR"]
