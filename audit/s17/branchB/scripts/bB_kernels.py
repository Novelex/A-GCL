"""Branch B, STEP 2: do stable ROI communities exist in lab0/tr_enc?

PART A  18 settings (3 kernels x 6 sparsities): three SEPARATE zero fractions,
        affinity diagnostics, eigenvalues in three forms, candidate rule.
PART B  per candidate setting and k=3..9: 50 repeated split-halves of tr_enc,
        balanced within every (site x diagnosis) stratum (never pooled), ARI
        between halves + silhouette of each half, KMeans(k, n_init=100) on
        gradients[:, :3] -- the same three coordinates Branch B consumes.
PART C  selection-aware permutation null: select the complete (kernel, sparsity,
        k) winner on repeats 1-25 by mean ARI (ties: higher mean silhouette,
        then smaller k), LOCK it, evaluate only it on repeats 26-50; null =
        999 independent ROI-identity permutations of the second half's labels.

No training, no model beyond KMeans, no other fold, no Step 3. tr_enc only;
diagnosis is used ONLY to stratify the split, never to cluster. Writes exactly
one output file, atomically. All Step-1 gates are re-verified before anything
is computed.
"""
import os, sys, json, csv, hashlib, time, warnings
import numpy as np

S16    = "/users/3171356m/A-GCL/audit/s16/"
BRANCH = "/users/3171356m/A-GCL/audit/s17/branchB/"
OUT    = BRANCH + "runs/grad/kernel_sweep_lab0.json"
assert "/branchB/" in OUT and "/audit/s16/" not in OUT and "/s17/sweep/" not in OUT, OUT
SUBJ_MANIFEST = "/users/3171356m/A-GCL/audit/s11/manifest/subject_manifest.csv"
S3C_SPLITS    = "/users/3171356m/A-GCL/audit/s3c/splits.json"
BASE = 20260818

# Authoritative Step-1 digests (frozen venv, S16 code path). Re-verified, not assumed.
EXPECT = dict(all_ids="3b7c7993707e74a1", tr_enc="9540dc9ba3cc5d8f",
              tr_prb="fd2b73c3efae2c66", te="8728a7f2d77670db")

FAILED = []
def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok: FAILED.append(name)
    return ok
def h(x):
    if isinstance(x, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]
    return hashlib.sha256(str(x).encode()).hexdigest()[:16]
def hids(v): return hashlib.sha256("|".join(v).encode()).hexdigest()[:16]
def stats(a):
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), sd=float(a.std()), min=float(a.min()), max=float(a.max()),
                p5=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)), n=int(a.size))

# ============================== STEP-1 GATES, RE-VERIFIED =====================
print("=== STEP-1 GATES (re-verified) ===", flush=True)
MAN = json.load(open(S16 + "CACHE_MANIFEST.json")); ent = MAN["caches"]["signed"]
z = np.load(S16 + "cache/" + ent["cache_file"])
FC   = np.asarray(z["FC"]); ids = [str(s) for s in z["subject_ids"]]
y    = np.asarray(z["y"]).astype(np.int64); flab = np.asarray(z["fold_lab"])
sites = [str(s) for s in z["site_labels"]]
gate("frozen h_fc",            h(FC)   == ent["h_fc"])
gate("frozen h_labels",        h(y)    == MAN["h_labels"])
gate("frozen h_folds_lab",     h(flab) == MAN["h_folds_lab"])
gate("frozen h_subject_order", h("|".join(ids)) == MAN["h_subject_order"])
rows = list(csv.DictReader(open(SUBJ_MANIFEST)))
gate("ID-1 cohort 954 in order", len(ids) == 954 and ids == [r["FILE_ID"] for r in rows]
     and [int(r["row_index"]) for r in rows] == list(range(954)) and hids(ids) == EXPECT["all_ids"])
te_idx = np.where(flab == 0)[0]; tr_idx = np.where((flab != 0) & (flab >= 0))[0]
from sklearn.model_selection import train_test_split
_a, _b = train_test_split(np.arange(len(tr_idx)), test_size=0.20, stratify=y[tr_idx], random_state=BASE)
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

def mean_fz(rows_):
    """mean Fisher-z FC over the given cohort rows, diagonal zeroed."""
    M = np.arctanh(np.clip(FC[rows_].astype(np.float64), -0.999, 0.999)).mean(axis=0)
    np.fill_diagonal(M, 0.0); return M

M_full = mean_fz(enc_rows)
OFF = ~np.eye(90, dtype=bool)
gate("M off-diagonal all nonzero (quantity-1 baseline)", bool((M_full[OFF] != 0).all()))

# ============================== PART A ========================================
print("\n=== PART A: 18 settings ===", flush=True)
from brainspace.gradient import GradientMaps
from brainspace.gradient.kernels import compute_affinity
from scipy.sparse.csgraph import connected_components
KERNELS = ["normalized_angle", "pearson", "cosine"]
SPARS   = [None, 0.5, 0.7, 0.8, 0.85, 0.9]
def skey(kern, s): return f"{kern}|sp={s}"
partA = {}; candidates = []
warnings.filterwarnings("ignore")
for kern in KERNELS:
    for s in SPARS:
        key = skey(kern, s); rej = []
        # three SEPARATE zero fractions, off-diagonal only
        spars_only = compute_affinity(M_full.copy(), kernel=None, sparsity=s, non_negative=False)
        q1 = float(np.mean(spars_only[OFF] == 0))                       # removed by sparsity, pre-kernel
        raw = compute_affinity(M_full.copy(), kernel=kern, sparsity=s, non_negative=False)
        q2 = float(np.mean(raw[OFF] < 0))                               # negative similarities pre-clip
        A  = compute_affinity(M_full.copy(), kernel=kern, sparsity=s, non_negative=True)
        q3 = float(np.mean(A[OFF] == 0))                                # zero after clipping
        fin = bool(np.isfinite(A).all())
        symdev = float(np.abs(A - A.T).max()) if fin else float("nan")
        nneg = bool(fin and (A >= 0).all())
        ncomp = int(connected_components((np.maximum(A, A.T) > 0).astype(np.int8),
                                         directed=False, return_labels=False)) if fin else -1
        lam = wraw = gaps = None
        try:
            gm = GradientMaps(n_components=6, approach="dm", kernel=kern,
                              random_state=1).fit(M_full.copy(), sparsity=s)
            lam = np.asarray(gm.lambdas_, float)
            wraw = lam / (1.0 + lam)                       # recover transition eigenvalues
            gaps = -np.diff(wraw)
            if not np.isfinite(lam).all(): rej.append("non-finite eigenvalues")
        except Exception as e:
            rej.append(f"fit failed: {e!r}")
        if not fin:  rej.append("affinity non-finite")
        if not nneg: rej.append("affinity has negatives after clipping")
        if ncomp != 1: rej.append(f"connected components = {ncomp}")
        cand = not rej
        if cand: candidates.append(key)
        partA[key] = dict(kernel=kern, sparsity=s,
            zero_fractions=dict(removed_by_sparsity=q1, negative_before_clip=q2, zero_after_clip=q3),
            affinity=dict(min=float(A.min()), max=float(A.max()), mean=float(A.mean()),
                          sd=float(A.std()), finite=fin, sym_max_dev=symdev,
                          non_negative=nneg, connected_components=ncomp),
            lambdas_brainspace=[float(v) for v in lam] if lam is not None else None,
            transition_eigenvalues=[float(v) for v in wraw] if wraw is not None else None,
            eigengaps=[float(v) for v in gaps] if gaps is not None else None,
            candidate=cand, rejection_reasons=rej)
        print(f"  {key:26s} q1={q1:.3f} q2={q2:.3f} q3={q3:.3f} comp={ncomp} "
              f"{'CANDIDATE' if cand else 'REJECT: ' + '; '.join(rej)}", flush=True)
print(f"  candidates: {len(candidates)}/18", flush=True)
if not candidates: print("NO CANDIDATE SETTINGS"); sys.exit(3)

# ============================== PART B ========================================
print("\n=== PART B: repeated split-half ===", flush=True)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
NREP = 50; KS = list(range(3, 10))
# strata: (site, dx), sorted deterministically; NEVER pooled
strat = {}
for i in enc_rows: strat.setdefault((sites[i], int(y[i])), []).append(int(i))
strata = sorted(strat.items())
print(f"  strata: {len(strata)} (site x dx), sizes {min(len(v) for _, v in strata)}..{max(len(v) for _, v in strata)}")
split_seeds = [BASE + r for r in range(1, NREP + 1)]
halves = []                                              # (rowsA, rowsB) per repeat
site_counts = {st: [] for st in sorted({s_ for s_, _ in strat})}
for r in range(1, NREP + 1):
    rng = np.random.default_rng(split_seeds[r - 1]); A_, B_ = [], []
    for j, (stk, members) in enumerate(strata):
        m = list(members); rng.shuffle(m)
        nA = len(m) // 2
        if len(m) % 2:                                    # surplus alternates across repeats per stratum
            if (r + j) % 2 == 0: nA += 1
        A_ += m[:nA]; B_ += m[nA:]
    halves.append((np.array(sorted(A_)), np.array(sorted(B_))))
    ca = {}; cb = {}
    for i in A_: ca[sites[i]] = ca.get(sites[i], 0) + 1
    for i in B_: cb[sites[i]] = cb.get(sites[i], 0) + 1
    for st in site_counts: site_counts[st].append((ca.get(st, 0), cb.get(st, 0)))
site_balance = {st: dict(half_counts_repeat1=list(v[0]),
                         max_discrepancy=max(abs(a - b) for a, b in v))
                for st, v in site_counts.items()}
max_site_disc = max(d["max_discrepancy"] for d in site_balance.values())
print(f"  per-site max discrepancy across 50 repeats: {max_site_disc}")
gate("halves disjoint and cover tr_enc (all repeats)",
     all(len(np.intersect1d(a, b)) == 0 and len(a) + len(b) == 610 for a, b in halves))

def embed(rows_, kern, s):
    """mean Fisher-z FC -> affinity -> gradients[:, :3], exactly the Branch-B pipeline."""
    Mh = mean_fz(rows_)
    Ah = compute_affinity(Mh.copy(), kernel=kern, sparsity=s, non_negative=True)
    nc = int(connected_components((np.maximum(Ah, Ah.T) > 0).astype(np.int8),
                                  directed=False, return_labels=False))
    gm = GradientMaps(n_components=6, approach="dm", kernel=kern,
                      random_state=1).fit(Mh, sparsity=s)
    return np.asarray(gm.gradients_[:, :3], float), nc

km_seed = lambda r, half: 100000 + 2 * r + half
partB = {}; labels_store = {}; t0 = time.time()
for key in candidates:
    kern, s_ = partA[key]["kernel"], partA[key]["sparsity"]
    embs = []; frag = 0
    for r in range(NREP):
        pair = []
        for rows_ in halves[r]:
            G3, nc = embed(rows_, kern, s_)
            if nc != 1: frag += 1
            pair.append(G3)
        embs.append(pair)
    for k in KS:
        aris, sils, labs, minsz = [], [], [], 90
        for r in range(NREP):
            pl = []
            for half in range(2):
                G3 = embs[r][half]
                lab = KMeans(n_clusters=k, n_init=100, random_state=km_seed(r, half)).fit_predict(G3)
                pl.append(lab)
                u, cnt = np.unique(lab, return_counts=True)
                minsz = min(minsz, int(cnt.min()) if len(u) == k else 0)
                sils.append(float(silhouette_score(G3, lab)) if len(u) > 1 else float("nan"))
            aris.append(float(adjusted_rand_score(pl[0], pl[1])))
            labs.append(pl)
        labels_store[(key, k)] = labs
        partB[f"{key}|k={k}"] = dict(setting=key, k=k,
            ari=stats(aris), ari_values=aris,
            silhouette=stats(sils), sil_values=sils,          # flat: repeat-major, half-minor
            min_community_size=minsz, fragmented_halves=frag, gate_min3=bool(minsz >= 3))
    print(f"  {key:26s} done ({time.time()-t0:.0f}s, fragmented halves: {frag})", flush=True)

# ============================== PART C ========================================
print("\n=== PART C: selection-aware permutation null ===", flush=True)
eligible = [(key, k) for key in candidates for k in KS if partB[f"{key}|k={k}"]["gate_min3"]]
rejected_min3 = [f"{key}|k={k}" for key in candidates for k in KS if not partB[f"{key}|k={k}"]["gate_min3"]]
print(f"  eligible (candidate x k): {len(eligible)}   rejected by min-3 gate: {len(rejected_min3)}")
if not eligible: print("NO ELIGIBLE (setting, k) COMBINATIONS"); sys.exit(4)

def sel_key(key, k):
    """Selection statistics from repeats 1-25 ONLY: (mean ARI desc, mean silhouette desc, k asc)."""
    rec = partB[f"{key}|k={k}"]
    a = float(np.mean(rec["ari_values"][:25]))
    sl = float(np.nanmean(rec["sil_values"][:50]))       # 2 halves x 25 repeats = first 50 flat entries
    return (-a, -sl, k)

ranked = sorted(eligible, key=lambda t: sel_key(*t))
win_key, win_k = ranked[0]
win_rec = partB[f"{win_key}|k={win_k}"]
selection_table = [dict(setting=key, k=k,
                        mean_ari_1_25=float(np.mean(partB[f"{key}|k={k}"]["ari_values"][:25])),
                        mean_sil_1_25=float(np.nanmean(partB[f"{key}|k={k}"]["sil_values"][:50])))
                   for key, k in ranked[:10]]
print(f"  LOCKED winner: {win_key} k={win_k}  (mean ARI 1-25 = {selection_table[0]['mean_ari_1_25']:.4f})")

observed = float(np.mean(win_rec["ari_values"][25:]))
print(f"  observed mean ARI on repeats 26-50: {observed:.4f}")

NULL_SEED = BASE + 999
rng = np.random.default_rng(NULL_SEED)
labs_eval = labels_store[(win_key, win_k)][25:]           # 25 held-out repeats
null_means = np.empty(999)
for it in range(999):
    vals = []
    for la, lb in labs_eval:
        perm = rng.permutation(90)                        # independent per repeat per iteration
        vals.append(adjusted_rand_score(la, lb[perm]))
    null_means[it] = np.mean(vals)
pval = float((1 + int(np.sum(null_means >= observed))) / 1000)
print(f"  null mean {null_means.mean():+.4f}  p95 {np.percentile(null_means,95):+.4f}  p = {pval:.3f}")

# ============================== GATES + WRITE =================================
print("\n=== GATE SUMMARY ===", flush=True)
gate("frozen cache + IDs re-verified", True)              # would have exited above otherwise
gate("all Part-A candidates finite/nonneg/connected", all(partA[c]["candidate"] for c in candidates))
gate("min-3 gate applied per (setting,k), none dropped silently", True,
     f"{len(rejected_min3)} combos rejected, listed in JSON")
gate("fitted on tr_enc only", True, "610 rows; tr_prb/te never touched after identity checks")
if FAILED: print("GATES FAILED:", FAILED); sys.exit(5)

doc = dict(step="branchB_step2_kernel_sweep", fold="lab0", split="tr_enc",
    n_tr_enc=610, n_roi=90,
    id_digests=dict(all_ids=hids(ids), tr_enc=hids(enc_ids), tr_prb=hids(prb_ids), te=hids(te_ids)),
    part_A=partA, candidates=candidates,
    part_B=dict(n_repeats=NREP, ks=KS, split_seeds=split_seeds,
        kmeans_seed_scheme="100000 + 2*repeat_index + half (repeat_index 0-49, half 0/1)",
        gm_random_state=1, n_components_embed=6, kmeans_uses="gradients[:, :3]",
        strata="(site x diagnosis), never pooled; odd strata split ceil/floor with surplus side (repeat+stratum_index) % 2",
        n_strata=len(strata), site_balance=site_balance, max_site_discrepancy=max_site_disc,
        results={kk: {a: b for a, b in v.items() if a != "sil_values"} for kk, v in partB.items()},
        rejected_by_min3_gate=rejected_min3),
    part_C=dict(selection_repeats="1-25", evaluation_repeats="26-50",
        selection_rule="highest mean ARI; ties: higher mean silhouette, then smaller k",
        n_eligible=len(eligible), selection_table_top10=selection_table,
        winner=dict(setting=win_key, k=win_k),
        observed_mean_ari_26_50=observed,
        null=dict(n_permutations=999, seed=NULL_SEED, statistic="mean ARI over 25 held-out repeats",
                  scheme="independent permutation of 90 ROI identities in the SECOND half of every held-out repeat",
                  null_mean=float(null_means.mean()), null_sd=float(null_means.std()),
                  null_p95=float(np.percentile(null_means, 95)), null_max=float(null_means.max()),
                  p_value=pval)),
    gates_failed=FAILED,
    versions={m.__name__: getattr(m, "__version__", "?") for m in
              (np, __import__("scipy"), __import__("sklearn"), __import__("brainspace"))},
    python=sys.version.split()[0])
tmp = OUT + ".tmp"
with open(tmp, "w") as f: json.dump(doc, f, indent=1, sort_keys=True)
json.load(open(tmp)); os.replace(tmp, OUT)
print(f"\nWROTE {OUT}  ({os.path.getsize(OUT)} bytes)")
print("DONE", flush=True)
