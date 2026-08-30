"""Branch B, STEP 1: three functional coordinates for each of the 90 ROIs,
derived from the lab0 / tr_enc RAW Pearson FC.

No K-means, no training, no other fold. Reads the frozen S16 'signed' cache
directly with numpy so that neither torch nor pandas is required, and verifies
every frozen hash itself rather than delegating to the S16 loader. Nothing under
audit/s16/ or audit/s17/sweep/ is written; the only writes are the two output
files below, both asserted to live under /branchB/.

Raw FC, NOT tangent. The mean tangent matrix is zero by construction, which would
drive the normalized-angle affinity to a constant ~0.5; that is an explicit gate.
"""
import os, sys, json, csv, hashlib
import numpy as np

# ------------------------------------------------------------------ paths
S16    = "/users/3171356m/A-GCL/audit/s16/"
BRANCH = "/users/3171356m/A-GCL/audit/s17/branchB/"
OUTDIR = BRANCH + "runs/grad/"
NPY    = OUTDIR + "gradients_lab0.npy"
JSN    = OUTDIR + "gradients_lab0.json"
for _p in (NPY, JSN):
    assert "/branchB/" in _p, f"output path lacks /branchB/: {_p}"
    assert "/audit/s16/" not in _p and "/s17/sweep/" not in _p, _p
assert os.path.isdir(OUTDIR), OUTDIR

SUBJ_MANIFEST = "/users/3171356m/A-GCL/audit/s11/manifest/subject_manifest.csv"
S3C_SPLITS    = "/users/3171356m/A-GCL/audit/s3c/splits.json"
BASE = 20260818                      # s12a5_core.BASE, the frozen split seed

# Authoritative ID-list digests, produced by the S16 code path in the FROZEN venv
# (s16_data.load('signed') + s16_data.folds(d,'lab')[0] + s16_feat.honest_split).
# These are the "manifest lists" this script must reproduce exactly.
EXPECT = dict(all_ids="3b7c7993707e74a1", tr_enc="9540dc9ba3cc5d8f",
              tr_prb="fd2b73c3efae2c66", te="8728a7f2d77670db")

FAILED = []
def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok: FAILED.append(name)
    return ok

def h(x):
    if isinstance(x, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]
    return hashlib.sha256(str(x).encode()).hexdigest()[:16]
def hids(v): return hashlib.sha256("|".join(v).encode()).hexdigest()[:16]

# ------------------------------------------------------- frozen cache + hashes
print("\n=== FROZEN DATA ===")
MAN = json.load(open(S16 + "CACHE_MANIFEST.json"))
ent = MAN["caches"]["signed"]
cache = S16 + "cache/" + ent["cache_file"]
z = np.load(cache)
FC   = np.asarray(z["FC"])                       # (954,90,90) raw signed Pearson
ids  = [str(s) for s in z["subject_ids"]]
y    = np.asarray(z["y"]).astype(np.int64)
flab = np.asarray(z["fold_lab"])
print(f"  cache        : {ent['cache_file']}")
print(f"  FC shape     : {FC.shape}  dtype {FC.dtype}")
gate("frozen h_fc",            h(FC)   == ent["h_fc"],        f"{h(FC)} == {ent['h_fc']}")
gate("frozen h_labels",        h(y)    == MAN["h_labels"],    f"{h(y)} == {MAN['h_labels']}")
gate("frozen h_folds_lab",     h(flab) == MAN["h_folds_lab"], f"{h(flab)} == {MAN['h_folds_lab']}")
gate("frozen h_subject_order", h("|".join(ids)) == MAN["h_subject_order"])
gate("E=signed is raw (identity transform)", ent["sparse"] is False and ent["fc_absmax"] <= 1.0)

# ------------------------------------------ IDENTITY CHECK 1: full cohort list
print("\n=== IDENTITY 1: full raw-FC subject list vs frozen 954 cohort, in order ===")
with open(SUBJ_MANIFEST) as f:
    rows = list(csv.DictReader(f))
man_ids = [r["FILE_ID"] for r in rows]
man_row = [int(r["row_index"]) for r in rows]
gate("cohort size 954",              len(ids) == 954 == len(man_ids), f"cache {len(ids)}, manifest {len(man_ids)}")
gate("manifest row_index is 0..953", man_row == list(range(954)))
gate("ID lists equal IN ORDER",      ids == man_ids)
gate("ID digest vs authoritative",   hids(ids) == EXPECT["all_ids"], f"{hids(ids)} == {EXPECT['all_ids']}")
gate("IDs unique",                   len(set(ids)) == 954)

# ------------------------------------------------------------- lab0 fold + split
print("\n=== FOLD lab0 ===")
te_idx = np.where(flab == 0)[0]
tr_idx = np.where((flab != 0) & (flab >= 0))[0]
from sklearn.model_selection import train_test_split
_a, _b = train_test_split(np.arange(len(tr_idx)), test_size=0.20,
                          stratify=y[tr_idx], random_state=BASE)
tr_enc_idx, tr_prb_idx = np.asarray(tr_idx)[_a], np.asarray(tr_idx)[_b]
n_enc, n_prb, n_te = len(tr_enc_idx), len(tr_prb_idx), len(te_idx)
print(f"  n_tr_enc = {n_enc}")
print(f"  n_tr_prb = {n_prb}")
print(f"  n_te     = {n_te}")
print(f"  total    = {n_enc + n_prb + n_te}")

# --------------------------------- IDENTITY CHECK 2: each list vs its manifest
print("\n=== IDENTITY 2: tr_enc / tr_prb / te ID lists vs manifest lists ===")
enc_ids = [ids[i] for i in tr_enc_idx]
prb_ids = [ids[i] for i in tr_prb_idx]
te_ids  = [ids[i] for i in te_idx]
tr_ids  = [ids[i] for i in tr_idx]
gate("tr_enc digest", hids(enc_ids) == EXPECT["tr_enc"], f"{hids(enc_ids)} == {EXPECT['tr_enc']}")
gate("tr_prb digest", hids(prb_ids) == EXPECT["tr_prb"], f"{hids(prb_ids)} == {EXPECT['tr_prb']}")
gate("te digest",     hids(te_ids)  == EXPECT["te"],     f"{hids(te_ids)}  == {EXPECT['te']}")
s3c = json.load(open(S3C_SPLITS))["outer_folds"][0]
gate("te vs s3c outer_folds[0].test",   te_ids == [ids[i] for i in s3c["test"]],  f"n={len(s3c['test'])}")
gate("tr vs s3c outer_folds[0].train",  tr_ids == [ids[i] for i in s3c["train"]], f"n={len(s3c['train'])}")
gate("tr_enc | tr_prb partitions tr",   sorted(enc_ids + prb_ids) == sorted(tr_ids))
gate("tr_enc disjoint from tr_prb",     not (set(enc_ids) & set(prb_ids)))
gate("tr disjoint from te",             not (set(tr_ids) & set(te_ids)))
gate("enc+prb+te == full cohort",       sorted(enc_ids + prb_ids + te_ids) == sorted(ids))

# --------------------------------- IDENTITY CHECK 3: load BY ID, not by position
print("\n=== IDENTITY 3: subjects loaded BY ID ===")
id2row = {sid: i for i, sid in enumerate(ids)}
gate("id->row map complete", len(id2row) == 954)
enc_rows = np.array([id2row[s] for s in enc_ids], dtype=int)
gate("by-ID rows match index-derived rows", np.array_equal(enc_rows, tr_enc_idx))
R_enc = FC[enc_rows]                                  # selected BY ID
gate("selected block shape", R_enc.shape == (n_enc, 90, 90), str(R_enc.shape))

if FAILED:
    print("\nIDENTITY GATES FAILED:", FAILED); sys.exit(2)

# ------------------------------------------------------------------- compute M
print("\n=== M = mean Fisher-z over tr_enc ===")
Fz = np.arctanh(np.clip(R_enc.astype(np.float64), -0.999, 0.999))
M = Fz.mean(axis=0)
np.fill_diagonal(M, 0.0)
off = ~np.eye(90, dtype=bool)
asym = float(np.abs(M - M.T).max())
print(f"  off-diagonal  min {M[off].min():+.6f}  max {M[off].max():+.6f}  "
      f"mean {M[off].mean():+.6f}  sd {M[off].std():.6f}")
print(f"  max|M - M^T|  {asym:.3e}")
gate("M shape (90,90)",       M.shape == (90, 90))
gate("M all finite",          bool(np.isfinite(M).all()))
gate("M diagonal exactly 0",  bool((np.diag(M) == 0).all()))
gate("M symmetric < 1e-10",   asym < 1e-10, f"{asym:.3e}")

# ---------------------------------------------------------------- brainspace
print("\n=== BRAINSPACE embedding ===")
from brainspace.gradient import GradientMaps
from brainspace.gradient.kernels import compute_affinity
A = compute_affinity(M.copy(), kernel="normalized_angle", sparsity=None)
ncomp = int(__import__("scipy.sparse.csgraph", fromlist=["x"]).connected_components(
    (A > 0).astype(np.int8), directed=False, return_labels=False))
print(f"  affinity  min {A.min():.6f}  max {A.max():.6f}  mean {A.mean():.6f}  sd {A.std():.6f}")
print(f"  connected components (A>0): {ncomp}")
gate("affinity shape",        A.shape == (90, 90))
gate("affinity finite",       bool(np.isfinite(A).all()))
gate("affinity symmetric",    float(np.abs(A - A.T).max()) < 1e-10)
gate("affinity within [0,1]", bool(A.min() >= -1e-12 and A.max() <= 1 + 1e-12))
gate("affinity NOT all ~0.5 (tangent tell)", float(np.abs(A[off] - 0.5).max()) > 0.05,
     f"max|A-0.5| off-diag = {float(np.abs(A[off]-0.5).max()):.4f}")
gate("connected components == 1", ncomp == 1, str(ncomp))

gm = GradientMaps(n_components=4, approach="dm", kernel="normalized_angle",
                  random_state=1).fit(M.copy(), sparsity=None)
lam = np.asarray(gm.lambdas_, dtype=float)
G3  = np.asarray(gm.gradients_[:, :3], dtype=float)
print(f"  eigenvalues (4): {np.array2string(lam, precision=6)}")
print(f"  gradients shape: {G3.shape}   per-component sd: "
      f"{[round(float(G3[:,k].std()),6) for k in range(3)]}")
gate("4 eigenvalues", lam.shape == (4,), str(lam.shape))
gate("eigenvalues finite", bool(np.isfinite(lam).all()))
gate("eigenvalues descending", bool(np.all(np.diff(lam) <= 1e-12)))
gate("no repeated unit eigenvalues (fragmentation tell)",
     int(np.sum(np.abs(lam - 1.0) < 1e-6)) == 0, f"{int(np.sum(np.abs(lam-1.0)<1e-6))} at 1.0")
gate("gradients finite", bool(np.isfinite(G3).all()))
gate("gradient sd > 1e-8 each", bool(all(G3[:, k].std() > 1e-8 for k in range(3))))

# ------------------------------------------- independent numpy reimplementation
print("\n=== HAND REIMPLEMENTATION (numpy, eigh; independent of ARPACK) ===")
def hand_dm(Aff, n_components=4, alpha=0.5, diffusion_time=0):
    """Same settings as brainspace: normalized-angle affinity in, alpha=0.5
    anisotropic normalisation, diffusion operator P = D^-1 W, trivial first
    eigenpair removed, multi-scale scaling w/(1-w) when diffusion_time == 0.
    Solved via the symmetric conjugate S = Dw^-1/2 W Dw^-1/2 with numpy.eigh,
    rather than ARPACK eigsh on the non-symmetric P as brainspace does."""
    d  = Aff.sum(axis=1)
    da = d ** -alpha
    W  = da[:, None] * Aff * da[None, :]
    dw = W.sum(axis=1)
    s  = dw ** -0.5
    S  = s[:, None] * W * s[None, :]
    S  = (S + S.T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    o = np.argsort(vals)[::-1]
    vals, vecs = vals[o], vecs[:, o]
    psi = s[:, None] * vecs                    # right eigenvectors of P
    psi = psi / psi[:, [0]]                    # force trivial vector to ones
    w   = vals / vals[0]
    psi, w = psi[:, 1:n_components + 1], w[1:n_components + 1]
    w = w / (1 - w) if diffusion_time <= 0 else w ** diffusion_time
    return psi * w[None, :], w

H, hw = hand_dm(A.copy(), n_components=4)
H3 = H[:, :3]
print(f"  hand eigenvalues: {np.array2string(hw, precision=6)}")
print(f"  hand per-component sd: {[round(float(H3[:,k].std()),6) for k in range(3)]}")

C = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        C[i, j] = abs(np.corrcoef(G3[:, i], H3[:, j])[0, 1])
from scipy.optimize import linear_sum_assignment
ri, ci = linear_sum_assignment(-C)
print("  |corr| matrix (rows brainspace 1-3, cols hand 1-3):")
for i in range(3):
    print("    " + "  ".join(f"{C[i,j]:.6f}" for j in range(3)))
matching = [(int(i) + 1, int(j) + 1, float(C[i, j])) for i, j in zip(ri, ci)]
print("  Hungarian matching:")
for a, b, v in matching: print(f"    brainspace {a} <-> hand {b}   |r| = {v:.6f}")
gate("all matched |corr| > 0.95", all(v > 0.95 for _, _, v in matching),
     f"min matched |r| = {min(v for _,_,v in matching):.6f}")

if FAILED:
    print("\nGATES FAILED:", FAILED); sys.exit(3)

# ---------------------------------------------------------------------- write
np.save(NPY, G3.astype(np.float64))
meta = dict(
    step="branchB_step1_gradients", fold="lab0", split="tr_enc", source="raw signed Pearson FC",
    cache_file=ent["cache_file"], h_fc=ent["h_fc"],
    n_tr_enc=n_enc, n_tr_prb=n_prb, n_te=n_te, n_roi=90,
    id_digests=dict(all_ids=hids(ids), tr_enc=hids(enc_ids), tr_prb=hids(prb_ids), te=hids(te_ids)),
    transform="F = arctanh(clip(R, -0.999, 0.999)); M = mean over tr_enc; diag(M) = 0",
    M_offdiag=dict(min=float(M[off].min()), max=float(M[off].max()),
                   mean=float(M[off].mean()), sd=float(M[off].std())),
    M_asymmetry=asym,
    affinity=dict(kernel="normalized_angle", sparsity=None, min=float(A.min()), max=float(A.max()),
                  mean=float(A.mean()), sd=float(A.std()), connected_components=ncomp),
    embedding=dict(approach="dm", n_components=4, random_state=1, alpha=0.5, diffusion_time=0),
    eigenvalues=[float(v) for v in lam],
    gradients_shape=list(G3.shape),
    gradient_sd=[float(G3[:, k].std()) for k in range(3)],
    crosscheck=dict(abs_corr_matrix=[[float(C[i, j]) for j in range(3)] for i in range(3)],
                    hungarian=[dict(brainspace=a, hand=b, abs_corr=v) for a, b, v in matching],
                    min_matched_abs_corr=float(min(v for _, _, v in matching)),
                    hand_eigenvalues=[float(v) for v in hw]),
    versions={m.__name__: getattr(m, "__version__", "?") for m in
              (np, __import__("scipy"), __import__("sklearn"), __import__("brainspace"))},
    python=sys.version.split()[0],
    gates_failed=FAILED,
)
tmp = JSN + ".tmp"
with open(tmp, "w") as f: json.dump(meta, f, indent=2, sort_keys=True)
json.load(open(tmp)); os.replace(tmp, JSN)
print(f"\nWROTE {NPY}  ({os.path.getsize(NPY)} bytes)")
print(f"WROTE {JSN}  ({os.path.getsize(JSN)} bytes)")
print("\nALL GATES PASSED")
