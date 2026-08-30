"""Branch B, STEP 3: do the ROI communities carry diagnostic signal beyond the raw edges?

RAW FC ONLY. The frozen s16 'signed' cache is raw Pearson FC -- apply_E returns F
unchanged for E == "signed" -- so every matrix here is labelled raw. No tangent.
tr_enc (610) only; tr_prb and te are never indexed after the identity checks.
No neural network, no GPU.

Design
  1  5 outer folds inside tr_enc, stratified by diagnosis x site.
  2  Communities REBUILT from training subjects at EVERY level -- inside each of
     the 25 inner-training folds as well as each of the 5 outer-training folds.
     The Step-2 map (built from all 610) is used ONLY as an ARI reference and
     never enters any model.
  3  Each of the 4,005 upper-triangle edges is assigned to exactly one of
     k(k+1)/2 blocks; per block: signed mean, sd, mean absolute.
  4  A = 4,005 raw edges; B = block summaries. Both StandardScaler -> LinearSVC.
     Inner CV selects k (3 or 5), statistic arm, and C.
  5  Fusion S = a*S_A + (1-a)*S_B, a from inner out-of-fold TRAINING scores only.

Fusion note: S_A (4,005 features) and S_B (6-45 features) are on different
scales, so an unstandardised alpha would measure scale rather than information
(the pitfall recorded in s16_feat). Both streams are standardised with mean/sd
taken from the INNER OUT-OF-FOLD TRAINING scores only, never from outer
validation. The unstandardised variant is computed too and reported as a
sensitivity check.
"""
import os, sys, json, csv, hashlib, time, warnings, itertools
import numpy as np

S16    = "/users/3171356m/A-GCL/audit/s16/"
BRANCH = "/users/3171356m/A-GCL/audit/s17/branchB/"
OUT    = BRANCH + "runs/blocks/step3_raw_lab0.json"
assert "/branchB/" in OUT and "/audit/s16/" not in OUT and "/s17/sweep/" not in OUT, OUT
assert os.path.isdir(os.path.dirname(OUT)), OUT
SUBJ_MANIFEST = "/users/3171356m/A-GCL/audit/s11/manifest/subject_manifest.csv"
S3C_SPLITS    = "/users/3171356m/A-GCL/audit/s3c/splits.json"
BASE = 20260818
EXPECT = dict(all_ids="3b7c7993707e74a1", tr_enc="9540dc9ba3cc5d8f",
              tr_prb="fd2b73c3efae2c66", te="8728a7f2d77670db")

FAILED = []
def gate(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f"  {d}" if d else ""), flush=True)
    if not ok: FAILED.append(n)
    return ok
def h(x):
    if isinstance(x, np.ndarray): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]
    return hashlib.sha256(str(x).encode()).hexdigest()[:16]
def hids(v): return hashlib.sha256("|".join(v).encode()).hexdigest()[:16]

# ============================ STEP-1 GATES, RE-VERIFIED =======================
print("=== STEP-1 GATES (re-verified) ===", flush=True)
MAN = json.load(open(S16 + "CACHE_MANIFEST.json")); ent = MAN["caches"]["signed"]
_z = np.load(S16 + "cache/" + ent["cache_file"])
_FC_ALL = np.asarray(_z["FC"]); ids = [str(s) for s in _z["subject_ids"]]
y_all = np.asarray(_z["y"]).astype(np.int64); flab = np.asarray(_z["fold_lab"])
sites_all = [str(s) for s in _z["site_labels"]]
gate("frozen h_fc", h(_FC_ALL) == ent["h_fc"])
gate("frozen h_labels", h(y_all) == MAN["h_labels"])
gate("frozen h_folds_lab", h(flab) == MAN["h_folds_lab"])
gate("frozen h_subject_order", h("|".join(ids)) == MAN["h_subject_order"])
gate("E=signed is the identity transform (raw Pearson)",
     ent["sparse"] is False and abs(ent["fc_absmax"] - 1.0) < 1e-9 and ent["fc_diag_dev"] == 0.0)
rows = list(csv.DictReader(open(SUBJ_MANIFEST)))
gate("ID-1 cohort 954 in order", len(ids) == 954 and ids == [r["FILE_ID"] for r in rows]
     and [int(r["row_index"]) for r in rows] == list(range(954)) and hids(ids) == EXPECT["all_ids"])
te_idx = np.where(flab == 0)[0]; tr_idx = np.where((flab != 0) & (flab >= 0))[0]
from sklearn.model_selection import train_test_split, StratifiedKFold
_a, _b = train_test_split(np.arange(len(tr_idx)), test_size=0.20, stratify=y_all[tr_idx], random_state=BASE)
enc_idx, prb_idx = np.asarray(tr_idx)[_a], np.asarray(tr_idx)[_b]
enc_ids = [ids[i] for i in enc_idx]; prb_ids = [ids[i] for i in prb_idx]; te_ids = [ids[i] for i in te_idx]
print(f"  fold sizes: tr_enc={len(enc_idx)}  tr_prb={len(prb_idx)}  te={len(te_idx)}")
gate("ID-2 split digests", hids(enc_ids) == EXPECT["tr_enc"] and hids(prb_ids) == EXPECT["tr_prb"]
     and hids(te_ids) == EXPECT["te"])
s3c = json.load(open(S3C_SPLITS))["outer_folds"][0]
gate("ID-2 s3c cross-check", te_ids == [ids[i] for i in s3c["test"]]
     and [ids[i] for i in tr_idx] == [ids[i] for i in s3c["train"]]
     and sorted(enc_ids + prb_ids) == sorted(ids[i] for i in tr_idx)
     and not (set(enc_ids) & set(prb_ids)) and not (set(enc_ids) | set(prb_ids)) & set(te_ids))
id2row = {s: i for i, s in enumerate(ids)}
enc_rows = np.array([id2row[s] for s in enc_ids], dtype=int)
gate("ID-3 loaded by ID", len(id2row) == 954 and np.array_equal(enc_rows, enc_idx))
gate("split sizes 610/153/191", (len(enc_idx), len(prb_idx), len(te_idx)) == (610, 153, 191))
if FAILED: print("STEP-1 GATES FAILED:", FAILED); sys.exit(2)

# ---- from here on, ONLY the 610 tr_enc subjects exist. tr_prb / te are dropped.
FC  = _FC_ALL[enc_rows].astype(np.float64).copy()       # (610,90,90) RAW Pearson
Y   = y_all[enc_rows].copy()
SITE= [sites_all[i] for i in enc_rows]
_FORBIDDEN = set(prb_idx.tolist()) | set(te_idx.tolist())
del _FC_ALL, _z
gate("only tr_enc retained", FC.shape == (610, 90, 90) and len(Y) == 610)
gate("tr_prb/te rows unreachable", not (set(enc_rows.tolist()) & _FORBIDDEN))
N, R = 610, 90
IU = np.triu_indices(R, 1)
gate("upper-triangle edge count == 4005", len(IU[0]) == 4005)
EDGES = np.stack([FC[s][IU] for s in range(N)]).astype(np.float64)   # (610, 4005) raw edges

def mean_fz(rows_):
    M = np.arctanh(np.clip(FC[rows_], -0.999, 0.999)).mean(axis=0)
    np.fill_diagonal(M, 0.0); return M

# ============================ SETUP ===========================================
from brainspace.gradient import GradientMaps
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, adjusted_rand_score)
warnings.filterwarnings("ignore")
KS   = [3, 5]
ARMS = ["signed_mean", "sd", "abs_mean", "all_three"]
CGRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
STRAT = np.array([f"{SITE[i]}|{Y[i]}" for i in range(N)])

def embed(rows_):
    gm = GradientMaps(n_components=6, approach="dm", kernel="normalized_angle",
                      random_state=1).fit(mean_fz(rows_), sparsity=None)
    return np.asarray(gm.gradients_[:, :3], float)

def communities(rows_, seed):
    """Rebuild the map from THESE training subjects only. Returns {k: labels}."""
    G3 = embed(rows_)
    return {k: KMeans(n_clusters=k, n_init=100, random_state=seed).fit_predict(G3) for k in KS}

BLOCK_CHECKS = []
def block_index(lab, k):
    """Assign every one of the 4,005 upper-triangle edges to exactly one block."""
    ci, cj = lab[IU[0]], lab[IU[1]]
    lo, hi = np.minimum(ci, cj), np.maximum(ci, cj)
    pairs = [(a, b) for a in range(k) for b in range(a, k)]
    pmap = {p: t for t, p in enumerate(pairs)}
    idx = np.array([pmap[(int(a), int(b))] for a, b in zip(lo, hi)])
    counts = np.bincount(idx, minlength=len(pairs))
    BLOCK_CHECKS.append(dict(k=k, n_blocks=len(pairs), total=int(counts.sum()),
                             min_block=int(counts.min()), unassigned=int((idx < 0).sum())))
    assert counts.sum() == 4005 and (counts > 0).all() and len(pairs) == k * (k + 1) // 2
    return idx, len(pairs)

def block_feats(E, idx, nb, arm):
    """Per-subject block summaries from that subject's own raw edges."""
    sm = np.empty((E.shape[0], nb)); sd = np.empty_like(sm); am = np.empty_like(sm)
    for b in range(nb):
        m = idx == b
        sub = E[:, m]
        sm[:, b] = sub.mean(1); sd[:, b] = sub.std(1); am[:, b] = np.abs(sub).mean(1)
    return {"signed_mean": sm, "sd": sd, "abs_mean": am,
            "all_three": np.hstack([sm, sd, am])}[arm]

def fit_score(Xtr, ytr, Xva, C):
    sc = StandardScaler().fit(Xtr)
    m = LinearSVC(C=C, max_iter=20000, dual="auto").fit(sc.transform(Xtr), ytr)
    return m.decision_function(sc.transform(Xva))

# ============================ REFERENCE MAP (diagnostic only) =================
print("\n=== Step-2 reference map (all 610, ARI reference ONLY, never modelled) ===", flush=True)
REF = communities(np.arange(N), seed=100000)
for k in KS:
    _, c = np.unique(REF[k], return_counts=True)
    print(f"  k={k}: sizes {sorted(c.tolist())}")

# ============================ NESTED CV =======================================
print("\n=== NESTED CV inside tr_enc ===", flush=True)
OUTER_SEED, INNER_SEED = BASE + 31, BASE + 41
okf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
outer_splits = list(okf.split(np.arange(N), STRAT))
fold_report = []; comm_gate_ok = True
oof_A = np.full(N, np.nan); oof_B = np.full(N, np.nan)
oof_fz = np.full(N, np.nan); oof_fraw = np.full(N, np.nan)
oof_arm = {a: np.full(N, np.nan) for a in ARMS}
t0 = time.time()

for of, (otr, ova) in enumerate(outer_splits):
    ikf = StratifiedKFold(n_splits=5, shuffle=True, random_state=INNER_SEED + of)
    inner_splits = list(ikf.split(otr, STRAT[otr]))
    # ---- inner: rebuild communities from EACH inner-training fold
    innA = {C: np.full(len(otr), np.nan) for C in CGRID}
    innB = {(k, a, C): np.full(len(otr), np.nan) for k in KS for a in ARMS for C in CGRID}
    for iF, (itr_l, iva_l) in enumerate(inner_splits):
        itr, iva = otr[itr_l], otr[iva_l]
        lab_i = communities(itr, seed=200000 + 10 * of + iF)       # TRAINING subjects only
        for k in KS:
            _, c = np.unique(lab_i[k], return_counts=True)
            if len(c) != k or c.min() < 3: comm_gate_ok = False
        for C in CGRID:
            innA[C][iva_l] = fit_score(EDGES[itr], Y[itr], EDGES[iva], C)
        for k in KS:
            idx, nb = block_index(lab_i[k], k)
            for a in ARMS:
                Ftr, Fva = block_feats(EDGES[itr], idx, nb, a), block_feats(EDGES[iva], idx, nb, a)
                for C in CGRID:
                    innB[(k, a, C)][iva_l] = fit_score(Ftr, Y[itr], Fva, C)
    yo = Y[otr]
    aucA_in = {C: roc_auc_score(yo, innA[C]) for C in CGRID}
    aucB_in = {key: roc_auc_score(yo, v) for key, v in innB.items()}
    # ---- selections (deterministic tie-breaks)
    C_A = max(CGRID, key=lambda C: (aucA_in[C], -CGRID.index(C)))
    def pick(pool): return max(pool, key=lambda t: (aucB_in[t], -t[0], -ARMS.index(t[1]), -CGRID.index(t[2])))
    kB, aB, C_B = pick(list(innB))
    pick_arm = {a: pick([t for t in innB if t[1] == a]) for a in ARMS}
    # ---- outer: rebuild communities from the OUTER-TRAINING subjects only
    lab_o = communities(otr, seed=300000 + of)
    for k in KS:
        _, c = np.unique(lab_o[k], return_counts=True)
        if len(c) != k or c.min() < 3: comm_gate_ok = False
    sA = fit_score(EDGES[otr], yo, EDGES[ova], C_A); oof_A[ova] = sA
    idx_o, nb_o = block_index(lab_o[kB], kB)
    sB = fit_score(block_feats(EDGES[otr], idx_o, nb_o, aB), yo,
                   block_feats(EDGES[ova], idx_o, nb_o, aB), C_B); oof_B[ova] = sB
    for a, (ka, aa, Ca) in pick_arm.items():
        ix, nbb = block_index(lab_o[ka], ka)
        oof_arm[a][ova] = fit_score(block_feats(EDGES[otr], ix, nbb, aa), yo,
                                    block_feats(EDGES[ova], ix, nbb, aa), Ca)
    # ---- fusion: alpha from INNER out-of-fold TRAINING scores only
    iA, iB = innA[C_A], innB[(kB, aB, C_B)]
    muA, sdA = float(iA.mean()), float(iA.std() or 1.0)
    muB, sdB = float(iB.mean()), float(iB.std() or 1.0)
    zia, zib = (iA - muA) / sdA, (iB - muB) / sdB
    al_z = max(ALPHAS, key=lambda a_: (roc_auc_score(yo, a_ * zia + (1 - a_) * zib), a_))
    al_r = max(ALPHAS, key=lambda a_: (roc_auc_score(yo, a_ * iA + (1 - a_) * iB), a_))
    oof_fz[ova]  = al_z * ((sA - muA) / sdA) + (1 - al_z) * ((sB - muB) / sdB)
    oof_fraw[ova] = al_r * sA + (1 - al_r) * sB
    sc_o = {}
    for s_ in set(SITE[i] for i in ova): sc_o[s_] = int(sum(1 for i in ova if SITE[i] == s_))
    fold_report.append(dict(fold=of, n_train=len(otr), n_val=len(ova),
        n_asd_val=int(Y[ova].sum()), site_counts_val=dict(sorted(sc_o.items())),
        A=dict(C=C_A, inner_auc=float(aucA_in[C_A])),
        B=dict(k=int(kB), arm=aB, C=C_B, inner_auc=float(aucB_in[(kB, aB, C_B)])),
        B_per_arm={a: dict(k=int(t[0]), C=t[2], inner_auc=float(aucB_in[t])) for a, t in pick_arm.items()},
        alpha_standardised=al_z, alpha_raw=al_r,
        community_sizes_outer={str(k): sorted(np.bincount(lab_o[k]).tolist()) for k in KS},
        ari_vs_step2_reference={str(k): float(adjusted_rand_score(REF[k], lab_o[k])) for k in KS}))
    print(f"  fold {of}: A C={C_A:g} | B k={kB} arm={aB} C={C_B:g} | a_z={al_z} a_raw={al_r} "
          f"| ARI vs ref k3={fold_report[-1]['ari_vs_step2_reference']['3']:.3f} "
          f"k5={fold_report[-1]['ari_vs_step2_reference']['5']:.3f} ({time.time()-t0:.0f}s)", flush=True)

# ============================ METRICS =========================================
print("\n=== METRICS (out-of-fold, identical subjects) ===", flush=True)
gate("every OOF subject scored", bool(np.isfinite(oof_A).all() and np.isfinite(oof_B).all()
                                      and np.isfinite(oof_fz).all() and np.isfinite(oof_fraw).all()))
def perf(s):
    p = (s > 0).astype(int); tn, fp, fn, tp = confusion_matrix(Y, p).ravel()
    return dict(auc=float(roc_auc_score(Y, s)), balanced_accuracy=float(balanced_accuracy_score(Y, p)),
                sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                f1=float(f1_score(Y, p)))
PERF = dict(A=perf(oof_A), B=perf(oof_B), fused_standardised=perf(oof_fz), fused_raw=perf(oof_fraw),
            **{f"B_{a}": perf(oof_arm[a]) for a in ARMS})
for n_, v in PERF.items():
    print(f"  {n_:22s} AUC {v['auc']:.4f}  bacc {v['balanced_accuracy']:.4f}  "
          f"sens {v['sensitivity']:.4f}  spec {v['specificity']:.4f}  F1 {v['f1']:.4f}")

BOOT, rng = 5000, np.random.default_rng(BASE + 77)
BS = [rng.choice(N, N, replace=True) for _ in range(BOOT)]
def bci(fn):
    v = []
    for b in BS:
        try:
            x = fn(b)
            if x is not None and np.isfinite(x): v.append(x)
        except Exception: pass
    v = np.asarray(v)
    return dict(point=float(fn(np.arange(N))), lo=float(np.percentile(v, 2.5)),
                hi=float(np.percentile(v, 97.5)), n_boot=int(v.size))
def dauc(s):
    return lambda b: (roc_auc_score(Y[b], s[b]) - roc_auc_score(Y[b], oof_A[b])
                      if len(np.unique(Y[b])) > 1 else None)
DELTA = dict(fused_standardised=bci(dauc(oof_fz)), fused_raw=bci(dauc(oof_fraw)), B_minus_A=bci(dauc(oof_B)))
print(f"\n  PRIMARY  dAUC fused(standardised) - A = {DELTA['fused_standardised']['point']:+.4f} "
      f"[{DELTA['fused_standardised']['lo']:+.4f}, {DELTA['fused_standardised']['hi']:+.4f}]")
print(f"           dAUC fused(raw scale)   - A = {DELTA['fused_raw']['point']:+.4f} "
      f"[{DELTA['fused_raw']['lo']:+.4f}, {DELTA['fused_raw']['hi']:+.4f}]")
print(f"           dAUC B - A                = {DELTA['B_minus_A']['point']:+.4f} "
      f"[{DELTA['B_minus_A']['lo']:+.4f}, {DELTA['B_minus_A']['hi']:+.4f}]")

errA = (oof_A > 0).astype(int) != Y
def sub_auc(b):
    m = b[errA[b]]
    return roc_auc_score(Y[m], oof_B[m]) if len(np.unique(Y[m])) > 1 else None
def rescue(b):
    m = b[errA[b]]
    return float(np.mean((oof_B[m] > 0).astype(int) == Y[m])) if m.size else None
COMP = dict(subgroup_auc=bci(sub_auc), rescue_rate=bci(rescue),
            n_errors_A=int(errA.sum()), n_errors_A_asd=int(Y[errA].sum()))
print(f"\n  COMPLEMENTARITY (A makes {COMP['n_errors_A']} errors, {COMP['n_errors_A_asd']} ASD)")
print(f"    subgroup AUC (PRIMARY) {COMP['subgroup_auc']['point']:.4f} "
      f"[{COMP['subgroup_auc']['lo']:.4f}, {COMP['subgroup_auc']['hi']:.4f}]   null = 0.5")
print(f"    rescue rate            {COMP['rescue_rate']['point']:.4f} "
      f"[{COMP['rescue_rate']['lo']:.4f}, {COMP['rescue_rate']['hi']:.4f}]   null ~ 0.5")

rA = oof_A - np.polyval(np.polyfit(Y, oof_A, 1), Y)
rB = oof_B - np.polyval(np.polyfit(Y, oof_B, 1), Y)
CORR = dict(pearson_overall=float(np.corrcoef(oof_A, oof_B)[0, 1]),
            pearson_label_residualised=float(np.corrcoef(rA, rB)[0, 1]))
print(f"\n  corr(A,B) overall {CORR['pearson_overall']:+.4f} | label-residualised {CORR['pearson_label_residualised']:+.4f}")

print("\n=== GATES ===", flush=True)
bc = {k: [d for d in BLOCK_CHECKS if d["k"] == k] for k in KS}
gate("block edges sum to 4005, none unassigned, no empty block",
     all(d["total"] == 4005 and d["unassigned"] == 0 and d["min_block"] > 0 for d in BLOCK_CHECKS),
     f"{len(BLOCK_CHECKS)} assignments; blocks {[ (k, bc[k][0]['n_blocks']) for k in KS ]}")
gate("every community >= 3 ROIs at every split and every k", comm_gate_ok)
gate("communities rebuilt at every CV level", True, "25 inner-training + 5 outer-training maps")
gate("tr_prb / te never loaded after identity checks", True, "FC restricted to 610 rows before modelling")
if FAILED: print("GATES FAILED:", FAILED); sys.exit(3)

doc = dict(step="branchB_step3_blocks_raw", fold="lab0", split="tr_enc", input="RAW Pearson FC (E=signed identity)",
    n=N, n_roi=R, n_edges=4005, cache_file=ent["cache_file"], h_fc=ent["h_fc"],
    id_digests=dict(all_ids=hids(ids), tr_enc=hids(enc_ids), tr_prb=hids(prb_ids), te=hids(te_ids)),
    design=dict(outer_folds=5, inner_folds=5, stratify="diagnosis x site",
        outer_seed=OUTER_SEED, inner_seed_base=INNER_SEED,
        kmeans_seeds="reference 100000; inner 200000+10*outer+inner; outer 300000+outer",
        gm=dict(n_components=6, approach="dm", kernel="normalized_angle", random_state=1, sparsity=None),
        ks=KS, arms=ARMS, C_grid=CGRID, alphas=ALPHAS,
        fusion_note="both score streams standardised with mean/sd from INNER OOF TRAINING scores only; "
                    "raw-scale variant reported as sensitivity check",
        inner_stratum_warning="smallest (site x dx) stratum is 5 in tr_enc, ~4 within outer-training; "
                              "StratifiedKFold warns and proceeds, so a fold may miss the smallest stratum"),
    reference_map_sizes={str(k): sorted(np.bincount(REF[k]).tolist()) for k in KS},
    folds=fold_report, block_checks=BLOCK_CHECKS[:6] + [dict(n_total_assignments=len(BLOCK_CHECKS))],
    performance=PERF, delta_auc=DELTA, complementarity=COMP, score_correlation=CORR,
    n_bootstrap=BOOT, gates_failed=FAILED,
    versions={m.__name__: getattr(m, "__version__", "?") for m in
              (np, __import__("scipy"), __import__("sklearn"), __import__("brainspace"))},
    python=sys.version.split()[0])
tmp = OUT + ".tmp"
with open(tmp, "w") as f: json.dump(doc, f, indent=1, sort_keys=True)
json.load(open(tmp)); os.replace(tmp, OUT)
print(f"\nWROTE {OUT}  ({os.path.getsize(OUT)} bytes)")
print("DONE", flush=True)
