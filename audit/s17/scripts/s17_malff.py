"""S17 mALFF amplitude-scaled FC — classical probe only. NO training, NO new
architecture, CPU only.

L0  FC upper triangle (4005)            must reproduce 0.7565 or STOP
L1  FCamp upper triangle (4005)         FCamp[s,i,j] = FC[s,i,j] * A[s,i] * A[s,j]
L2  [FC || FCamp]        (8010)

A[s,i] = M1_raw[s,i,classical] / mean_i M1_raw[s,:,classical]   (mALFF, band idx 2)

M1_raw is read DIRECTLY from ALFF_func_proc/method1/alff_roi_first.npz — never via
any S16 loader, because S16's frozen ALFF tensor is already per-subject min-max
scaled to [0,1] (proven: mode 'joint' is bitwise identical to 'raw' on it).

Pipelines, folds, grid and seed are S5.5's own, imported from s55_core — the exact
code that produced 0.7565. Writes ONLY under audit/s17/runs/malff/.
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s55")
import s55_core as C                                  # imported, never copied
import pandas as pd

OUT = "/users/3171356m/A-GCL/audit/s17/runs/malff/"
os.makedirs(OUT, exist_ok=True)
assert "/audit/s17/" in OUT and "/audit/s16/" not in OUT
NJ = int(os.environ.get("MALFF_NJOBS", "4"))
BOOT_N, BOOT_SEED = 2000, C.SEED

def aj(o, p):
    json.dump(o, open(p + ".tmp", "w"), indent=1, default=str)
    json.load(open(p + ".tmp")); os.replace(p + ".tmp", p)

log_lines = []
def log(s):
    print(s, flush=True); log_lines.append(s)

# ---------------------------------------------------------------- STEP 1: raw M1
Z = np.load("/users/3171356m/A-GCL/ALFF_func_proc/method1/alff_roi_first.npz",
            allow_pickle=True)
M1 = Z["alff"].astype(np.float64)                     # (954, 90, 3)
raw_ids = [str(x) for x in Z["file_ids"]]
smin, smax = M1.min((1, 2)), M1.max((1, 2))
n0, n1 = int((smin == 0.0).sum()), int((smax == 1.0).sum())
log(f"STEP1 raw M1: shape {M1.shape}  per-subj min in [{smin.min():.4f},{smin.max():.4f}]"
    f"  per-subj max in [{smax.min():.4f},{smax.max():.4f}]")
log(f"STEP1 subjects at min==0.0: {n0}/954   at max==1.0: {n1}/954")
if n0 == 954 and n1 == 954:
    log("STEP1 *** WRONG FILE: tensor is min-maxed. STOP."); sys.exit(2)
B2 = M1[:, :, 2]                                       # classical band 0.010-0.080 Hz
assert (B2 > 0).all(), "nonpositive ALFF in classical band"
log(f"STEP1 band2 (classical): min {B2.min():.4f} max {B2.max():.4f} — strictly positive, RAW confirmed")

# ---------------------------------------------------------------- alignment
F = C.build()                                          # S5.5's own frozen features
y = np.asarray(F["y"]).astype(int)
s55_ids = [str(x) for x in F["ids"]]
assert raw_ids == s55_ids, "subject order drift raw-M1 vs S5.5"
meta = pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv")
site = meta.site.values
assert len(site) == 954
log(f"ALIGN: raw-M1 ids == S5.5 ids == frozen cohort order (954). y: {int((y==1).sum())} ASD / {int((y==0).sum())} NC")

# ---------------------------------------------------------------- STEP 2: mALFF
A = B2 / B2.mean(axis=1, keepdims=True)                # A[s].mean() == 1 by construction
mres = np.abs(A.mean(axis=1) - 1.0).max()
assert mres < 1e-12, f"per-subject mean != 1 (max residual {mres:.3e})"
log(f"STEP2 mALFF A: shape {A.shape}  per-subject mean==1.0 verified (max residual {mres:.2e})")
log(f"STEP2 A stats: min {A.min():.4f} max {A.max():.4f} mean {A.mean():.6f} sd {A.std():.4f}")

# ---------------------------------------------------------------- STEP 3: FCamp
import scipy.io  # noqa: F401  (loaded via C.build already)
FCu = np.asarray(F["FC"], dtype=np.float64)            # THE array that produced 0.7565
iu = np.triu_indices(90, k=1)
# Build the full squares in FLOAT64 from S5.5's OWN edge array — the exact array
# that produced 0.7565 — plus the frozen diagonal of exactly 1.0. The s16 cache is
# NOT used as the source: it stores float32, and its triangle differs from the
# float64 .mat read by ~3e-08 (pure float32 quantisation, same constant documented
# in the A7 parity check). It is compared below as a cross-check only.
FCsq = np.zeros((954, 90, 90), dtype=np.float64)
FCsq[:, iu[0], iu[1]] = FCu
FCsq += FCsq.transpose(0, 2, 1)
FCsq[:, np.arange(90), np.arange(90)] = 1.0            # frozen diag == 1.0
assert np.abs(FCsq[:, iu[0], iu[1]] - FCu).max() == 0.0, "square rebuild broke the triangle"
assert np.abs(FCsq - FCsq.transpose(0, 2, 1)).max() == 0.0
sys.path.insert(0, "/users/3171356m/A-GCL/audit/s16/scripts")
import s16_data as DAT
d16, _, ent16 = DAT.load("signed", where="malff")
assert [str(x) for x in d16["ids"]] == s55_ids, "s16 cache order drift"
xchk = np.abs(np.asarray(d16["FC"], dtype=np.float64) - FCsq).max()
log(f"STEP3 cross-check vs s16 float32 cache: max|diff| = {xchk:.3e} "
    f"(float32 quantisation bound 6e-08; cache {ent16['cache_file']})")
assert xchk < 6e-8, "frozen FC chain mismatch beyond float32 quantisation"
# W[i,j] = A_i * A_j is EXACTLY symmetric (one multiply per entry); multiplying the
# symmetric FCsq by it elementwise keeps bitwise symmetry. The unparenthesised form
# (FCsq*A_i)*A_j rounds differently from its transpose by ~1 ulp.
W = A[:, :, None] * A[:, None, :]
assert np.abs(W - W.transpose(0, 2, 1)).max() == 0.0
FCamp = FCsq * W                                       # new array; FCsq untouched
sym = np.abs(FCamp - FCamp.transpose(0, 2, 1)).max()
log(f"STEP3 FCamp: shape {FCamp.shape} (== FC {FCsq.shape})  symmetry max|X-X.T| = {sym:.3e}")
assert sym == 0.0 and FCamp.shape == FCsq.shape
log(f"STEP3 FC    stats: min {FCsq.min():+.4f} max {FCsq.max():+.4f} mean {FCsq.mean():+.6f} sd {FCsq.std():.4f}")
log(f"STEP3 FCamp stats: min {FCamp.min():+.4f} max {FCamp.max():+.4f} mean {FCamp.mean():+.6f} sd {FCamp.std():.4f}")
FCamp_u = FCamp[:, iu[0], iu[1]]
log(f"STEP3 FCamp triangle: {FCamp_u.shape}")

# ---------------------------------------------------------------- STEP 4: L0/L1/L2
SETS = {"L0_FC": FCu, "L1_FCamp": FCamp_u,
        "L2_FC+FCamp": np.hstack([FCu, FCamp_u])}
loso_folds = [(np.where(site != s)[0], np.where(site == s)[0])
              for s in C.SPEC["loso_sites"]]
res, oofs = {}, {}
for name, X in SETS.items():
    for mode, folds in (("cv", C.OUTER), ("loso", loso_folds)):
        t0 = time.time()
        rows, pooled, oof = C.nested(X, y, "linsvm", folds=folds, n_jobs=NJ)
        res[f"{name}|{mode}"] = dict(pooled_auc=pooled["auc"], n_feat=int(X.shape[1]),
            per_fold_auc=[round(r["auc"], 5) for r in rows],
            best_C=[r["best_C"] for r in rows], secs=round(time.time() - t0, 1))
        oofs[f"{name}|{mode}"] = oof
        log(f"STEP4 {name:12s} {mode:4s} pooled AUC {pooled['auc']:.4f}  "
            f"({X.shape[1]} feats, {time.time()-t0:.0f}s)")
        if name == "L0_FC" and mode == "cv":
            if abs(pooled["auc"] - 0.7565) > 0.0005:
                log(f"STEP4 *** L0 = {pooled['auc']:.4f} != 0.7565 — PIPELINE WRONG. STOP.")
                aj(dict(halted="L0 failed to reproduce 0.7565", res=res), OUT + "MALFF.json")
                sys.exit(3)
            log("STEP4 L0 reproduces 0.7565 — pipeline verified, continuing")

# ---------------------------------------------------------------- paired bootstrap
def boot_ci(oa, ob, y, n=BOOT_N, seed=BOOT_SEED):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); m = np.isfinite(oa) & np.isfinite(ob)
    ya, sa, sb = y[m], oa[m], ob[m]; d = []
    while len(d) < n:
        b = rng.integers(0, len(ya), len(ya))
        if len(np.unique(ya[b])) < 2: continue
        d.append(roc_auc_score(ya[b], sa[b]) - roc_auc_score(ya[b], sb[b]))
    d = np.array(d)
    point = (roc_auc_score(ya, sa) - roc_auc_score(ya, sb))
    return dict(delta=float(point), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)),
                p_boot=float(2 * min((d <= 0).mean(), (d >= 0).mean())))
boots = {}
for mode in ("cv", "loso"):
    for name in ("L1_FCamp", "L2_FC+FCamp"):
        b = boot_ci(oofs[f"{name}|{mode}"], oofs[f"L0_FC|{mode}"], y)
        boots[f"{name}-L0|{mode}"] = b
        log(f"BOOT {name}-L0 [{mode:4s}]: {b['delta']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}] p={b['p_boot']:.3f}")

aj(dict(results=res, bootstrap=boots,
        A_stats=dict(min=float(A.min()), max=float(A.max()), mean=float(A.mean()),
                     sd=float(A.std())),
        FC_stats=dict(min=float(FCsq.min()), max=float(FCsq.max()),
                      mean=float(FCsq.mean()), sd=float(FCsq.std())),
        FCamp_stats=dict(min=float(FCamp.min()), max=float(FCamp.max()),
                         mean=float(FCamp.mean()), sd=float(FCamp.std())),
        log=log_lines), OUT + "MALFF.json")
np.savez_compressed(OUT + "MALFF_oof.npz", y=y, **oofs)
log("COMPLETE — written to " + OUT)
