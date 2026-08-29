"""S17 tangent-space measurement of FC and FCamp. NO training of encoders, NO new
architecture, CPU only, no wave.

IMPLEMENTATION USED: scipy log-Euclidean (nilearn is NOT installed in this venv).
  reference mean : LOG-EUCLIDEAN mean  M = expm( mean_s logm(C_s) ), TRAIN subjects
                   of the fold ONLY, recomputed fresh for every fold.
  projection     : S_s = logm( M^(-1/2) . C_s . M^(-1/2) )
  features       : strict upper triangle k=1, 4005 values.
All matrix functions via scipy/numpy eigendecomposition of symmetric matrices.

Gates: L0 must reproduce 0.7565 and L1 must reproduce 0.7261 or everything stops.
Writes ONLY under audit/s17/runs/tangent/.
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s55")
import s55_core as C
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

OUT = "/users/3171356m/A-GCL/audit/s17/runs/tangent/"
os.makedirs(OUT, exist_ok=True)
assert "/audit/s17/" in OUT and "/audit/s16/" not in OUT
NJ = int(os.environ.get("TAN_NJOBS", "4"))
BOOT_N = 2000

def aj(o, p):
    json.dump(o, open(p + ".tmp", "w"), indent=1, default=str)
    json.load(open(p + ".tmp")); os.replace(p + ".tmp", p)

log_lines = []
def log(s): print(s, flush=True); log_lines.append(s)

# ------------------------------------------------ BUILD (identical to s17_malff)
Z = np.load("/users/3171356m/A-GCL/ALFF_func_proc/method1/alff_roi_first.npz",
            allow_pickle=True)
M1 = Z["alff"].astype(np.float64); raw_ids = [str(x) for x in Z["file_ids"]]
smin, smax = M1.min((1, 2)), M1.max((1, 2))
assert not ((smin == 0.0).all() and (smax == 1.0).all()), "min-maxed tensor: WRONG FILE"
B2 = M1[:, :, 2]; assert (B2 > 0).all()
F = C.build(); y = np.asarray(F["y"]).astype(int)
s55_ids = [str(x) for x in F["ids"]]
assert raw_ids == s55_ids, "order drift"
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
log(f"BUILD ok: raw M1 verified, A mean==1, FCamp symmetric 0.0, shapes {FCsq.shape}")

# ------------------------------------------------ POSITIVE DEFINITENESS
def min_eigs(Ms):
    return np.array([np.linalg.eigvalsh(m)[0] for m in Ms])
t0 = time.time()
e_fc, e_amp = min_eigs(FCsq), min_eigs(FCamp)
gmin = min(e_fc.min(), e_amp.min())
log(f"PD: min eig FC {e_fc.min():.6e} | FCamp {e_amp.min():.6e} | global {gmin:.6e} "
    f"({time.time()-t0:.0f}s)")
RIDGE = 0.0
if gmin <= 0.0:
    RIDGE = -gmin + 1e-6
    FCsq_r = FCsq + RIDGE * np.eye(90)
    FCamp_r = FCamp + RIDGE * np.eye(90)
    e_fc2, e_amp2 = min_eigs(FCsq_r), min_eigs(FCamp_r)
    log(f"PD: ridge {RIDGE:.6e} applied IDENTICALLY to both arms; min eig after: "
        f"FC {e_fc2.min():.6e} | FCamp {e_amp2.min():.6e}")
    # the strict upper triangle is untouched by a diagonal ridge, so L0/L1 flat
    # features are bitwise identical with or without it — verified:
    assert np.array_equal(FCsq_r[:, iu[0], iu[1]], FCu)
    assert np.array_equal(FCamp_r[:, iu[0], iu[1]], FCamp_u)
    FCsq_t, FCamp_t = FCsq_r, FCamp_r
    after = dict(fc=float(e_fc2.min()), fcamp=float(e_amp2.min()))
else:
    log("PD: all matrices already positive definite — NO ridge applied")
    FCsq_t, FCamp_t = FCsq, FCamp
    after = dict(fc=float(e_fc.min()), fcamp=float(e_amp.min()))

# ------------------------------------------------ tangent machinery (scipy/eigh)
def sym_fun(Msym, fun):
    lam, V = np.linalg.eigh(Msym)
    return (V * fun(lam)) @ V.T

def logm_spd(Msym):  return sym_fun(Msym, np.log)
def inv_sqrtm(Msym): return sym_fun(Msym, lambda l: 1.0 / np.sqrt(l))

def tangent_fold(mats, tr, te):
    """Log-Euclidean mean from TRAIN ONLY, fresh; project train+test; strict iu."""
    assert len(set(tr.tolist()) & set(te.tolist())) == 0, "train/test overlap"
    logs_tr = np.stack([logm_spd(mats[i]) for i in tr])
    Mref = sym_fun(logs_tr.mean(0), np.exp)              # expm(mean of logs)
    Wh = inv_sqrtm(Mref)
    def proj(idx):
        out = np.empty((len(idx), 4005))
        for j, i in enumerate(idx):
            S = logm_spd(Wh @ mats[i] @ Wh)
            S = (S + S.T) / 2.0                          # kill 1-ulp asymmetry
            out[j] = S[iu]
        return out
    return proj(tr), proj(te), len(tr)

def nested_flat(X, folds):
    """EXACTLY S5.5's nested(), via its own make_pipe/GRID/SKF."""
    rows, p, oof = C.nested(X, y, "linsvm", folds=folds, n_jobs=NJ)
    return p["auc"], oof

def nested_tangent(mats, folds, tag):
    oof = np.full(954, np.nan); counts = []
    for k, (tr, te) in enumerate(folds):
        tr = np.asarray(tr); te = np.asarray(te)
        Xtr, Xte, ntr = tangent_fold(mats, tr, te)
        assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
        gs = GridSearchCV(C.make_pipe("linsvm"), C.GRID,
                          cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED),
                          scoring="roc_auc", n_jobs=NJ, refit=True)
        gs.fit(Xtr, y[tr]); oof[te] = gs.decision_function(Xte)
        counts.append(dict(fold=k, n_ref_mean=ntr, n_test=len(te),
                           overlap=0))                   # asserted disjoint above
    assert np.isfinite(oof).all()
    return float(roc_auc_score(y, oof)), oof, counts

meta = pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv"); site = meta.site.values
loso_folds = [(np.where(site != s)[0], np.where(site == s)[0])
              for s in C.SPEC["loso_sites"]]

# ------------------------------------------------ GATES: L0, L1 reproduce
res, oofs, counts_all = {}, {}, {}
for name, X in (("L0_FC_flat", FCu), ("L1_FCamp_flat", FCamp_u)):
    for mode, folds in (("cv", C.OUTER), ("loso", loso_folds)):
        t0 = time.time(); auc, oof = nested_flat(X, folds)
        res[f"{name}|{mode}"] = auc; oofs[f"{name}|{mode}"] = oof
        log(f"RUN {name:14s} {mode:4s} AUC {auc:.4f} ({time.time()-t0:.0f}s)")
if abs(res["L0_FC_flat|cv"] - 0.7565) > 5e-4:
    log(f"*** GATE: L0 {res['L0_FC_flat|cv']:.4f} != 0.7565 — STOP"); aj(dict(halted="L0", res=res), OUT+"TANGENT.json"); sys.exit(3)
if abs(res["L1_FCamp_flat|cv"] - 0.7261) > 5e-4:
    log(f"*** GATE: L1 {res['L1_FCamp_flat|cv']:.4f} != 0.7261 — STOP"); aj(dict(halted="L1", res=res), OUT+"TANGENT.json"); sys.exit(3)
log("GATES: L0 reproduces 0.7565 and L1 reproduces 0.7261 — continuing")

# ------------------------------------------------ T1, T2
tanvec = {}
for name, mats in (("T1_FC_tangent", FCsq_t), ("T2_FCamp_tangent", FCamp_t)):
    for mode, folds in (("cv", C.OUTER), ("loso", loso_folds)):
        t0 = time.time()
        auc, oof, counts = nested_tangent(mats, folds, name)
        res[f"{name}|{mode}"] = auc; oofs[f"{name}|{mode}"] = oof
        counts_all[f"{name}|{mode}"] = counts
        log(f"RUN {name:14s} {mode:4s} AUC {auc:.4f} ({time.time()-t0:.0f}s) "
            f"ref-mean n per fold: {[c['n_ref_mean'] for c in counts]}")
    # value distribution: each subject's TEST-side tangent vector from the CV folds
    pool = np.empty((954, 4005))
    for tr, te in C.OUTER:
        _, Xte, _ = tangent_fold(mats, np.asarray(tr), np.asarray(te))
        pool[np.asarray(te)] = Xte
    tanvec[name] = dict(min=float(pool.min()), max=float(pool.max()),
                        mean=float(pool.mean()), sd=float(pool.std()))
    log(f"DIST {name}: min {pool.min():+.4f} max {pool.max():+.4f} "
        f"mean {pool.mean():+.6f} sd {pool.std():.4f}")

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
    boots[f"T1-L0|{mode}"] = boot(f"T1_FC_tangent|{mode}", f"L0_FC_flat|{mode}")
    boots[f"T2-T1|{mode}"] = boot(f"T2_FCamp_tangent|{mode}", f"T1_FC_tangent|{mode}")
    for k in (f"T1-L0|{mode}", f"T2-T1|{mode}"):
        b = boots[k]; log(f"BOOT {k}: {b['delta']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}] p={b['p_boot']:.3f}")

aj(dict(implementation="scipy log-Euclidean (nilearn not installed); reference mean "
        "= LOG-EUCLIDEAN mean expm(mean(logm(C_train))); projection "
        "logm(M^-1/2 C M^-1/2); strict upper triangle k=1",
        ridge=RIDGE, min_eig_before=dict(fc=float(e_fc.min()), fcamp=float(e_amp.min())),
        min_eig_after=after, results=res, bootstrap=boots, ref_mean_counts=counts_all,
        FC_stats=dict(min=float(FCsq.min()), max=float(FCsq.max()),
                      mean=float(FCsq.mean()), sd=float(FCsq.std())),
        FCamp_stats=dict(min=float(FCamp.min()), max=float(FCamp.max()),
                         mean=float(FCamp.mean()), sd=float(FCamp.std())),
        tangent_stats=tanvec, log=log_lines), OUT + "TANGENT.json")
np.savez_compressed(OUT + "TANGENT_oof.npz", y=y, **oofs)
log("COMPLETE")
