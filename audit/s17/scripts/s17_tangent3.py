"""S17 tangent rerun 3: WIDENED GRIDS. Same as s17_tangent2.py, two changes only:
  alpha grid -> {0.1, 0.2, 0.4, 0.6, 0.8, 0.9}   (0.4 was pinned in 23/24 folds)
  C grid     -> extended two decades below       (1e-4 was pinned in 24/24 folds)
Everything else identical. Original s17_tangent2 docstring follows.

S17 tangent rerun WITH SHRINKAGE. Same as s17_tangent.py, one change:
C_shrunk = (1-alpha)*C + alpha*I before projecting, alpha selected from
{0.01, 0.05, 0.1, 0.2, 0.4} in the INNER folds only, exactly as C is selected
(same GridSearchCV, same StratifiedKFold, same seed), then the fold's chosen alpha
is applied IDENTICALLY to both arms: selected on the FC arm (T1), reused for FCamp
(T2) in that fold.

Shrinkage with alpha >= 0.01 makes every matrix PD (min eig >= alpha - 1.5e-14),
so the previous 1e-6 ridge is no longer applied; the flat L0/L1 conditions never
used it anyway (strict triangle excludes the diagonal).

Implementation unchanged: scipy log-Euclidean, reference mean expm(mean(logm(.)))
from the fold's TRAINING subjects only, recomputed fresh per fold; projection
logm(M^-1/2 C M^-1/2); strict upper triangle k=1. Writes ONLY audit/s17/runs/tangent2/.
"""
import sys, os, json, time, shutil
import numpy as np

sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s55")
import s55_core as C
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Memory

OUT = "/users/3171356m/A-GCL/audit/s17/runs/tangent3/"
os.makedirs(OUT, exist_ok=True)
assert "/audit/s17/" in OUT and "/audit/s16/" not in OUT
NJ = int(os.environ.get("TAN_NJOBS", "4"))
ALPHAS = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
BOOT_N = 2000
MEMDIR = os.environ.get("TAN_CACHE", "/tmp/claude-102000043/-users-3171356m-A-GCL/"
                        "a9e2c01c-7bf7-4c3e-b518-73a2642d296c/scratchpad/tan3_cache")

def aj(o, p):
    json.dump(o, open(p + ".tmp", "w"), indent=1, default=str)
    json.load(open(p + ".tmp")); os.replace(p + ".tmp", p)

log_lines = []
def log(s): print(s, flush=True); log_lines.append(s)

# ------------------------------------------------ BUILD (identical to before)
Z = np.load("/users/3171356m/A-GCL/ALFF_func_proc/method1/alff_roi_first.npz",
            allow_pickle=True)
M1 = Z["alff"].astype(np.float64); raw_ids = [str(x) for x in Z["file_ids"]]
smin, smax = M1.min((1, 2)), M1.max((1, 2))
assert not ((smin == 0.0).all() and (smax == 1.0).all()), "min-maxed tensor: WRONG FILE"
B2 = M1[:, :, 2]; assert (B2 > 0).all()
F = C.build(); y = np.asarray(F["y"]).astype(int)
assert raw_ids == [str(x) for x in F["ids"]], "order drift"
A = B2 / B2.mean(axis=1, keepdims=True)
assert np.abs(A.mean(axis=1) - 1.0).max() < 1e-12
iu = np.triu_indices(90, k=1)
FCu = np.asarray(F["FC"], dtype=np.float64)
FCsq = np.zeros((954, 90, 90)); FCsq[:, iu[0], iu[1]] = FCu
FCsq += FCsq.transpose(0, 2, 1); FCsq[:, np.arange(90), np.arange(90)] = 1.0
W = A[:, :, None] * A[:, None, :]
assert np.abs(W - W.transpose(0, 2, 1)).max() == 0.0
FCamp = FCsq * W
assert np.abs(FCamp - FCamp.transpose(0, 2, 1)).max() == 0.0
FCamp_u = FCamp[:, iu[0], iu[1]]
log("BUILD ok: raw M1 verified, A mean==1, FCamp symmetric 0.0")

# ------------------------------------------------ EIGENVALUE SPECTRUM (report item)
def min_eigs(Ms): return np.array([np.linalg.eigvalsh(m)[0] for m in Ms])
e_fc = min_eigs(FCsq); e_amp = min_eigs(FCamp)
rep = int(np.argsort(e_fc)[len(e_fc) // 2])          # median min-eig subject
ev = np.sort(np.linalg.eigvalsh(FCsq[rep]))
log(f"SPECTRUM subject index {rep} (median min-eig subject), FC, sorted eigenvalues "
    f"BEFORE shrinkage:")
for r in range(0, 90, 10):
    log("  " + " ".join(f"{v:10.3e}" for v in ev[r:r+10]))
for thr in (0.01, 0.001, 1e-6):
    log(f"SPECTRUM FC subj{rep}: eigenvalues below {thr:g}: {int((ev < thr).sum())}/90")
ev2 = np.sort(np.linalg.eigvalsh(FCamp[rep]))
for thr in (0.01, 0.001, 1e-6):
    log(f"SPECTRUM FCamp subj{rep}: eigenvalues below {thr:g}: {int((ev2 < thr).sum())}/90")
log(f"PD: min eig before shrinkage: FC {e_fc.min():.6e} | FCamp {e_amp.min():.6e}; "
    f"NO ridge — shrinkage alpha>=0.01 guarantees PD (min eig after >= alpha-1.5e-14)")

# ------------------------------------------------ tangent + shrinkage machinery
# TangentShrink lives in s17_tanlib (a real module): joblib.Memory hashes estimators
# with pickle-by-reference, and a __main__-defined class is unpicklable inside loky
# workers — that crash killed the first run right after its gates passed.
from s17_tanlib import TangentShrink, sym_fun, logm_spd, inv_sqrtm  # noqa: F401

shutil.rmtree(MEMDIR, ignore_errors=True)   # stale cache from a dead run
mem = Memory(MEMDIR, verbose=0)
def make_tan_pipe():
    return Pipeline([("tan", TangentShrink()), ("sc", StandardScaler()),
                     ("clf", LinearSVC(dual="auto", max_iter=20000,
                                       random_state=C.SEED))], memory=mem)
SKF = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
CGRID = [1e-6, 1e-5] + list(C.GRID["clf__C"])   # two decades below S5.5's 1e-4 floor
assert CGRID == [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

meta = pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv"); site = meta.site.values
loso_folds = [(np.where(site != s)[0], np.where(site == s)[0])
              for s in C.SPEC["loso_sites"]]

# ------------------------------------------------ GATES: L0, L1 (flat, unshrunk)
res, oofs = {}, {}
for name, X in (("L0_FC_flat", FCu), ("L1_FCamp_flat", FCamp_u)):
    for mode, folds in (("cv", C.OUTER), ("loso", loso_folds)):
        t0 = time.time()
        _, p, oof = C.nested(X, y, "linsvm", folds=folds, n_jobs=NJ)
        res[f"{name}|{mode}"] = p["auc"]; oofs[f"{name}|{mode}"] = oof
        log(f"RUN {name:14s} {mode:4s} AUC {p['auc']:.4f} ({time.time()-t0:.0f}s)")
if abs(res["L0_FC_flat|cv"] - 0.7565) > 5e-4:
    log("*** GATE L0 failed — STOP"); aj(dict(halted="L0", res=res), OUT+"TANGENT3.json"); sys.exit(3)
if abs(res["L1_FCamp_flat|cv"] - 0.7261) > 5e-4:
    log("*** GATE L1 failed — STOP"); aj(dict(halted="L1", res=res), OUT+"TANGENT3.json"); sys.exit(3)
log("GATES: L0 reproduces 0.7565 and L1 reproduces 0.7261 — continuing")

# ------------------------------------------------ T1, T2 with inner-selected alpha
FCsq_flat = FCsq.reshape(954, 8100)          # views for the transformer; sources
FCamp_flat = FCamp.reshape(954, 8100)        # are never modified
alpha_per_fold = {}
for mode, folds in (("cv", C.OUTER), ("loso", loso_folds)):
    oofT1 = np.full(954, np.nan); oofT2 = np.full(954, np.nan); alphas = []
    c1_list, c2_list = [], []
    for k, (tr, te) in enumerate(folds):
        tr = np.asarray(tr); te = np.asarray(te)
        assert len(set(tr.tolist()) & set(te.tolist())) == 0
        t0 = time.time()
        # T1: alpha AND C selected jointly in the inner folds, exactly as C alone
        # was selected in S5.5 (same GridSearchCV, SKF, scoring, seed).
        gs1 = GridSearchCV(make_tan_pipe(), {"tan__alpha": ALPHAS, "clf__C": CGRID},
                           cv=SKF, scoring="roc_auc", n_jobs=NJ, refit=True)
        gs1.fit(FCsq_flat[tr], y[tr])
        oofT1[te] = gs1.decision_function(FCsq_flat[te])
        a_k = float(gs1.best_params_["tan__alpha"]); alphas.append(a_k)
        c1_list.append(float(gs1.best_params_["clf__C"]))
        # T2: the SAME alpha, fixed; only C is searched.
        gs2 = GridSearchCV(make_tan_pipe(), {"tan__alpha": [a_k], "clf__C": CGRID},
                           cv=SKF, scoring="roc_auc", n_jobs=NJ, refit=True)
        gs2.fit(FCamp_flat[tr], y[tr])
        oofT2[te] = gs2.decision_function(FCamp_flat[te])
        c2_list.append(float(gs2.best_params_["clf__C"]))
        log(f"FOLD {mode}{k}: alpha={a_k} C_T1={gs1.best_params_['clf__C']} "
            f"C_T2={gs2.best_params_['clf__C']} n_ref={len(tr)} ({time.time()-t0:.0f}s)")
    assert np.isfinite(oofT1).all() and np.isfinite(oofT2).all()
    res[f"T1_FC_tangent|{mode}"] = float(roc_auc_score(y, oofT1))
    res[f"T2_FCamp_tangent|{mode}"] = float(roc_auc_score(y, oofT2))
    oofs[f"T1_FC_tangent|{mode}"] = oofT1; oofs[f"T2_FCamp_tangent|{mode}"] = oofT2
    alpha_per_fold[mode] = dict(alpha=alphas, C_T1=c1_list, C_T2=c2_list)
    pin = dict(alpha_at_lower=sum(a == min(ALPHAS) for a in alphas),
               alpha_at_upper=sum(a == max(ALPHAS) for a in alphas),
               C_T1_at_lower=sum(c == min(CGRID) for c in c1_list),
               C_T1_at_upper=sum(c == max(CGRID) for c in c1_list),
               C_T2_at_lower=sum(c == min(CGRID) for c in c2_list), n_folds=len(alphas))
    alpha_per_fold[mode]["boundary_pins"] = pin
    log(f"PIN {mode}: alpha@{min(ALPHAS)}={pin['alpha_at_lower']}/{pin['n_folds']} "
        f"alpha@{max(ALPHAS)}={pin['alpha_at_upper']}/{pin['n_folds']} | "
        f"C_T1@{min(CGRID):g}={pin['C_T1_at_lower']}/{pin['n_folds']} "
        f"C_T1@{max(CGRID):g}={pin['C_T1_at_upper']}/{pin['n_folds']} | "
        f"C_T2@{min(CGRID):g}={pin['C_T2_at_lower']}/{pin['n_folds']}")
    log(f"RUN T1_FC_tangent  {mode:4s} AUC {res[f'T1_FC_tangent|{mode}']:.4f}  alphas {alphas}  C_T1 {c1_list}")
    log(f"RUN T2_FCamp_tang. {mode:4s} AUC {res[f'T2_FCamp_tangent|{mode}']:.4f}  (same alphas)  C_T2 {c2_list}")

# ------------------------------------------------ paired bootstrap
def boot(a, b):
    rng = np.random.default_rng(C.SEED); oa, ob = oofs[a], oofs[b]
    m = np.isfinite(oa) & np.isfinite(ob); ya, sa, sb = y[m], oa[m], ob[m]; d = []
    while len(d) < BOOT_N:
        idx = rng.integers(0, len(ya), len(ya))
        if len(np.unique(ya[idx])) < 2: continue
        d.append(roc_auc_score(ya[idx], sa[idx]) - roc_auc_score(ya[idx], sb[idx]))
    d = np.array(d)
    return dict(delta=float(roc_auc_score(ya, sa) - roc_auc_score(ya, sb)),
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                p_boot=float(2 * min((d <= 0).mean(), (d >= 0).mean())))
boots = {}
for mode in ("cv", "loso"):
    for lbl, a, b in ((f"T1-L0|{mode}", f"T1_FC_tangent|{mode}", f"L0_FC_flat|{mode}"),
                      (f"T2-T1|{mode}", f"T2_FCamp_tangent|{mode}", f"T1_FC_tangent|{mode}")):
        boots[lbl] = boot(a, b)
        x = boots[lbl]; log(f"BOOT {lbl}: {x['delta']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}] p={x['p_boot']:.3f}")

aj(dict(implementation="TANGENT3 widened grids; scipy log-Euclidean + shrinkage (1-a)C+aI; alpha in "
        f"{ALPHAS} selected in the inner folds on the FC arm (T1) and applied "
        "identically to FCamp (T2) in the same fold; reference mean "
        "expm(mean(logm(.))) from TRAIN subjects only, fresh per fold; no ridge",
        results=res, bootstrap=boots, alpha_per_fold=alpha_per_fold, alpha_grid=ALPHAS, C_grid=CGRID,
        spectrum_subject=rep, spectrum_fc_sorted=[float(v) for v in ev],
        spectrum_counts=dict(fc={str(t): int((ev < t).sum()) for t in (0.01, 0.001, 1e-6)},
                             fcamp={str(t): int((ev2 < t).sum()) for t in (0.01, 0.001, 1e-6)}),
        min_eig_before=dict(fc=float(e_fc.min()), fcamp=float(e_amp.min())),
        log=log_lines), OUT + "TANGENT3.json")
np.savez_compressed(OUT + "TANGENT3_oof.npz", y=y, **oofs)
shutil.rmtree(MEMDIR, ignore_errors=True)
log("COMPLETE")
