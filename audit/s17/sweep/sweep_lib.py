"""S17 ACCURACY SWEEP — shared library. CPU only.

DATA   frozen 954x90x90 FC (the S5.5 float64 edge array that produces 0.7565),
       mALFF classical band from TRUE raw M1 (never an S16 ALFF loader).
FOLDS  LAB = frozen S3C 5 folds (== S5.5 OUTER, == S16 F-LAB), SITE = S16 F-SITE 5,
       LOSO = 19 sites. Inner selection: StratifiedKFold(5, shuffle, seed 20260818)
       over the outer TRAIN block — exactly S5.5's GridSearchCV cv.
RULE   every representation is FITTED on the training block it is handed and applied
       to the held-out block. Nothing is cached across folds; the same subject gets
       different tangent values in different folds by construction.
"""
import os, sys, json, time, warnings
import numpy as np
from scipy import linalg
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

S16S = "/users/3171356m/A-GCL/audit/s16/scripts"; S55 = "/users/3171356m/agcl_audit_s0/s55"
sys.path.insert(0, S55); sys.path.insert(0, S16S)
import s55_core as C55                       # the exact S5.5 pipeline pieces
SEED = C55.SEED
ROOT = os.path.realpath(os.environ.get("SWEEP_ROOT", "/users/3171356m/A-GCL/audit/s17/runs/sweep/")) + "/"   # normalised; smoke may redirect
# Production root must live under audit/s17/. A smoke run may redirect ONLY into the
# session scratchpad. Nothing may ever resolve under audit/s16/.
assert ("/audit/s17/" in ROOT or ROOT.startswith("/tmp/claude-")) and "/audit/s16/" not in ROOT, ROOT
IU = np.triu_indices(90, k=1); I90 = np.eye(90)
RAW_M1 = "/users/3171356m/A-GCL/ALFF_func_proc/method1/alff_roi_first.npz"

def aj(o, p):
    """Atomic JSON write with a per-process temp name (concurrent writers of the same
    path — e.g. inputs.json from many tasks — cannot clobber each other's temp)."""
    os.makedirs(os.path.dirname(p), exist_ok=True); t = f"{p}.tmp.{os.getpid()}"
    json.dump(o, open(t, "w"), indent=1, default=str); json.load(open(t)); os.replace(t, p)

def anpz(p, **arrs):
    """Atomic npz write; returns the sha256[:16] of the final file."""
    t = f"{p}.tmp.{os.getpid()}.npz"; np.savez_compressed(t, **arrs); np.load(t).close(); os.replace(t, p)
    return sha_file(p)

def sha_file(p):
    import hashlib; h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()[:16]

# ================================================================= DATA
_DATA = None
def data():
    """Frozen inputs, verified every call-site process. Returns a dict."""
    global _DATA
    if _DATA is not None: return _DATA
    assert os.path.exists(C55.CACHE), "S5.5 features.npz absent - refusing to rebuild it from here"
    F = C55.build(); y = np.asarray(F["y"]).astype(int); ids = [str(x) for x in F["ids"]]
    FCu = np.asarray(F["FC"], dtype=np.float64)
    import hashlib
    h = hashlib.sha256(np.ascontiguousarray(FCu).tobytes()).hexdigest()[:16]
    assert h == "5e0780c9d99b238a", f"frozen FC identity changed: {h}"          # the array that gives 0.7565
    FCsq = np.zeros((954, 90, 90)); FCsq[:, IU[0], IU[1]] = FCu
    FCsq += FCsq.transpose(0, 2, 1); FCsq[:, np.arange(90), np.arange(90)] = 1.0
    assert np.array_equal(FCsq[:, IU[0], IU[1]], FCu)
    Z = np.load(RAW_M1, allow_pickle=True); M1 = Z["alff"].astype(np.float64)
    assert [str(x) for x in Z["file_ids"]] == ids, "raw M1 order drift"
    smin, smax = M1.min((1, 2)), M1.max((1, 2))
    assert not ((smin == 0.0).all() and (smax == 1.0).all()), "M1 is min-maxed: WRONG FILE"
    B2 = M1[:, :, 2]; assert (B2 > 0).all()
    A = B2 / B2.mean(axis=1, keepdims=True)                    # mALFF, 90 features
    assert np.abs(A.mean(1) - 1).max() < 1e-12
    import s16_data as DAT, pandas as pd
    d16, _, _ = DAT.load("signed", where="sweep")
    assert [str(x) for x in d16["ids"]] == ids
    lab16 = DAT.folds(d16, "lab")
    for (t, tr, te), (otr, ote) in zip(lab16, C55.OUTER):               # LAB == S5.5 as SETS
        assert np.array_equal(np.sort(te), np.sort(ote)) and np.array_equal(np.sort(tr), np.sort(otr))
    # LAB uses S5.5's OUTER arrays VERBATIM so the row ORDER - and therefore the inner
    # StratifiedKFold splits and liblinear's data order - are exactly S5.5's.
    folds = {"lab": [(t, np.asarray(otr), np.asarray(ote)) for (t, _, _), (otr, ote) in zip(lab16, C55.OUTER)],
             "site": DAT.folds(d16, "site"), "loso": DAT.folds(d16, "loso")}
    sites = np.asarray(d16["sites"])
    _DATA = dict(y=y, ids=ids, FCu=FCu, FCsq=FCsq, ALFF=A, folds=folds, sites=sites)
    return _DATA

def protocols():
    """Smoke may restrict to a subset; production always runs all three."""
    return tuple(os.environ.get("SWEEP_SMOKE_PROTOS", "lab,site,loso").split(","))

def outer_folds(protocol):
    return [(t, np.asarray(tr), np.asarray(te)) for t, tr, te in data()["folds"][protocol]]

def inner_splits(tr, y, n=5):
    """Relative to `tr`, exactly as GridSearchCV(cv=StratifiedKFold(5, shuffle, SEED))."""
    skf = StratifiedKFold(n, shuffle=True, random_state=SEED)
    return [(tr[a], tr[b]) for a, b in skf.split(np.zeros(len(tr)), y[tr])]

# ================================================================= SPD utilities
def _form_symmetric(function, eigenvalues, eigenvectors):
    return np.dot(eigenvectors * function(eigenvalues), eigenvectors.T)
def _map_eigenvalues(function, symmetric):
    vals, vecs = linalg.eigh(symmetric); return _form_symmetric(function, vals, vecs)

def geometric_mean(matrices, max_iter=10, tol=1e-7):
    """Affine-invariant (Karcher) mean. VENDORED from nilearn 0.14.0
    nilearn/connectome/connectivity_matrices.py::_geometric_mean (Fletcher 2007,
    Algorithm 3) — nilearn is NOT installed in this venv and installing it would
    risk altering the frozen sklearn/scipy that reproduce 0.7565."""
    matrices = np.asarray(matrices); gmean = matrices.mean(0); norm_old = np.inf; step = 1.0
    for _ in range(max_iter):
        vals_g, vecs_g = linalg.eigh(gmean)
        g_inv_sqrt = _form_symmetric(np.sqrt, 1.0 / vals_g, vecs_g)
        logs = np.stack([_map_eigenvalues(np.log, g_inv_sqrt @ m @ g_inv_sqrt) for m in matrices])
        logs_mean = logs.mean(0)
        if np.any(np.isnan(logs_mean)): raise FloatingPointError("nan in geometric mean")
        norm = np.linalg.norm(logs_mean)
        vals_l, vecs_l = linalg.eigh(logs_mean)
        g_sqrt = _form_symmetric(np.sqrt, vals_g, vecs_g)
        gmean = g_sqrt @ _form_symmetric(np.exp, vals_l * step, vecs_l) @ g_sqrt
        if norm < norm_old: norm_old = norm
        elif norm > norm_old: step /= 2.0; norm = norm_old
        if norm / gmean.size < tol: break
    return gmean

def logeuclid_mean(matrices):
    return _map_eigenvalues(np.exp, np.stack([_map_eigenvalues(np.log, m) for m in matrices]).mean(0))

def shrink(M, alpha):        return (1.0 - alpha) * M + alpha * I90
def eig_truncate(M, k):
    """Keep the top-k eigenvalues; the remaining (90-k) are FLOORED at
    max(lambda_k, 1e-6*lambda_1) so the matrix stays well-conditioned PD and the
    discarded directions carry no variation. The RELATIVE floor matters: a floor at
    a numerically-zero lambda_k (k=90 on these rank-deficient matrices) made the
    whitened matrix indefinite and the log map returned NaN — caught in the smoke
    test. This is an explicit interpretation of 'keep top k, discard the rest'."""
    vals, vecs = linalg.eigh(M); order = np.argsort(vals)[::-1]; vals = vals[order]; vecs = vecs[:, order]
    floor = max(vals[k - 1], 1e-6 * vals[0])
    vals = np.where(np.arange(90) < k, np.maximum(vals, floor), floor)
    return (vecs * vals) @ vecs.T
def partial_spd(M, alpha):
    """Unit-diagonal PRECISION Q = D^-1/2 (C_alpha)^-1 D^-1/2 : SPD by construction,
    off-diagonals equal MINUS the partial correlations. Projected to the tangent
    space like any SPD matrix; the sign convention is immaterial to a linear
    classifier. (The unit-diagonal partial-correlation matrix 2I - Q itself is not
    guaranteed PD, so it cannot be log-mapped safely.)"""
    P = linalg.inv(shrink(M, alpha)); d = 1.0 / np.sqrt(np.diag(P)); return (P * d[:, None]) * d[None, :]

# ================================================================= REPRESENTATIONS
class Rep:
    """fit(train_mats) -> self ; transform(mats) -> (n, 4005). NEVER cached across folds."""
    def __init__(self, kind, alpha=None, k=None, mean="logeuclid"):
        self.kind, self.alpha, self.k, self.mean = kind, alpha, k, mean
    def name(self):
        if self.kind == "flat": return "flat"
        tag = {"tangent_shrink": f"a{self.alpha}", "tangent_trunc": f"k{self.k}",
               "partial_tangent": f"a{self.alpha}"}[self.kind]
        return f"{self.kind}_{tag}_{self.mean}"
    def _pre(self, M):
        if self.kind == "tangent_shrink":  return shrink(M, self.alpha)
        if self.kind == "tangent_trunc":   return eig_truncate(M, self.k)
        if self.kind == "partial_tangent": return partial_spd(M, self.alpha)
        raise ValueError(self.kind)
    def fit(self, mats):
        if self.kind == "flat": return self
        pre = np.stack([self._pre(m) for m in mats])
        ref = geometric_mean(pre) if self.mean == "geometric" else logeuclid_mean(pre)
        vals, vecs = linalg.eigh(ref); self.Wh_ = _form_symmetric(lambda v: 1 / np.sqrt(v), vals, vecs)
        self.n_fit_ = len(mats); return self
    def transform(self, mats):
        if self.kind == "flat": return mats[:, IU[0], IU[1]]
        out = np.empty((len(mats), 4005))
        for j, m in enumerate(mats):
            T = _map_eigenvalues(np.log, self.Wh_ @ self._pre(m) @ self.Wh_)
            out[j] = ((T + T.T) / 2)[IU]
        return out

def rep_catalogue():
    R = [Rep("flat")]
    if os.environ.get("SWEEP_SMOKE_REPS"):      # smoke only: a tiny named subset
        want = os.environ["SWEEP_SMOKE_REPS"].split(",")
        full = _rep_catalogue_full(); out = [r for r in full if r.name() in want]
        assert [r.name() for r in out] == want, (want, [r.name() for r in full][:5]); return out
    return _rep_catalogue_full()

def _rep_catalogue_full():
    R = [Rep("flat")]
    for mean in ("logeuclid", "geometric"):
        for a in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8): R.append(Rep("tangent_shrink", alpha=a, mean=mean))
        for k in (10, 20, 30, 40, 50, 60, 90): R.append(Rep("tangent_trunc", k=k, mean=mean))
        for a in (0.1, 0.2, 0.4, 0.6): R.append(Rep("partial_tangent", alpha=a, mean=mean))
    names = [r.name() for r in R]; assert len(set(names)) == len(names)
    return R                                                           # 43

# ================================================================= CLASSIFIERS
C_WIDE = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
C_S55  = list(C55.GRID["clf__C"])                                       # [1e-4 .. 100]
RBF_C  = [0.01, 0.1, 1, 10, 100, 1000]
RBF_G  = [1e-5, 3e-5, 1e-4, 2.5e-4, 1e-3, 3e-3, 1e-2]                  # 2.5e-4 ~= 'scale' at 4005
EN_C   = [1e-3, 1e-2, 1e-1, 1, 10]; EN_L1 = [0.1, 0.5, 0.9]
def clf_catalogue():
    if os.environ.get("SWEEP_SMOKE_CLFS") == "s55linsvm":      # smoke only
        return [("linsvm", dict(C=c)) for c in C_S55]
    G = []
    for c in C_WIDE: G.append(("linsvm", dict(C=c)))
    for c in C_WIDE: G.append(("logreg", dict(C=c)))
    for c in RBF_C:
        for g in RBF_G: G.append(("rbf", dict(C=c, gamma=g)))
    for c in EN_C:
        for l1 in EN_L1: G.append(("elasticnet", dict(C=c, l1_ratio=l1)))
    return G                                                           # 9+9+42+15 = 75
def make_clf(kind, hp):
    if kind == "linsvm":     return LinearSVC(C=hp["C"], dual="auto", max_iter=20000, random_state=SEED)   # == C55.make_pipe
    if kind == "logreg":     return LogisticRegression(C=hp["C"], penalty="l2", solver="lbfgs", max_iter=3000, random_state=SEED)
    if kind == "rbf":        return SVC(C=hp["C"], gamma=hp["gamma"], kernel="rbf", random_state=SEED)
    if kind == "elasticnet": return LogisticRegression(C=hp["C"], l1_ratio=hp["l1_ratio"], penalty="elasticnet",
                                                       solver="saga", max_iter=2000, tol=1e-3, random_state=SEED)
    raise ValueError(kind)
def cfg_id(kind, hp): return kind + "|" + "|".join(f"{k}={v:g}" for k, v in sorted(hp.items()))

# ================================================================= NESTED CORE
def eval_rep_on_fold(rep, protocol, k_fold, y_override=None, clfs=None, inner_n=5, log=print):
    """For ONE outer fold and ONE representation: inner-CV score of EVERY classifier
    config (mean AUC over inner splits, exactly GridSearchCV's mean_test_score) and
    the outer-TEST decision scores of every config refit on the full outer train.
    Selection is NOT done here — the aggregator selects on inner scores only."""
    D = data(); y = D["y"] if y_override is None else y_override
    tag, tr, te = outer_folds(protocol)[k_fold]
    clfs = clfs or clf_catalogue(); mats = D["FCsq"]
    inner = inner_splits(tr, y, inner_n)
    inner_auc = {cfg_id(k, h): [] for k, h in clfs}
    t0 = time.time()
    for a, b in inner:                                         # inner: rep fit on a ONLY
        r = Rep(rep.kind, rep.alpha, rep.k, rep.mean).fit(mats[a])
        Xa, Xb = r.transform(mats[a]), r.transform(mats[b])
        sc = StandardScaler().fit(Xa); Xa, Xb = sc.transform(Xa), sc.transform(Xb)
        for kind, hp in clfs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = make_clf(kind, hp).fit(Xa, y[a])
            inner_auc[cfg_id(kind, hp)].append(float(roc_auc_score(y[b], m.decision_function(Xb))))
    t_in = time.time() - t0
    r = Rep(rep.kind, rep.alpha, rep.k, rep.mean).fit(mats[tr])   # outer: rep fit on tr ONLY
    Xtr, Xte = r.transform(mats[tr]), r.transform(mats[te])
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    test_scores = {}
    for kind, hp in clfs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = make_clf(kind, hp).fit(Xtr, y[tr])
        test_scores[cfg_id(kind, hp)] = m.decision_function(Xte).astype(np.float64)
    feat_stats = dict(min=float(Xte.min()), max=float(Xte.max()), mean=float(Xte.mean()), sd=float(Xte.std()))
    return dict(protocol=protocol, fold=tag, k_fold=k_fold, rep=rep.name(), n_tr=int(len(tr)), n_te=int(len(te)),
                n_ref_mean=int(getattr(r, "n_fit_", 0)), inner_n=inner_n,
                inner_auc={c: [float(v) for v in v_] for c, v_ in inner_auc.items()},
                inner_mean={c: float(np.mean(v_)) for c, v_ in inner_auc.items()},
                test_idx=te.astype(int), test_scores=test_scores, feat_stats_te=feat_stats,
                secs_inner=round(t_in, 1), secs_total=round(time.time() - t0, 1))

def select_inner(inner_mean, allowed=None):
    """GridSearchCV tie rule: first config (in catalogue order) attaining the max."""
    best, bc = -1.0, None
    for c, v in inner_mean.items():
        if allowed is not None and c not in allowed: continue
        if v > best: best, bc = v, c
    return bc, best
