"""S15 canonical data builder + Gate-C. ONE build, delete-and-rebuild, hard
asserts, no try/except, no fallback. Gate-C also runs at the START of every job."""
import sys, os, json, time, socket, hashlib, numpy as np

sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11");           import s11_core as K
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s7");            import s7_core as C7
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a5/scripts"); import s12a5_core as M

S15 = "/users/3171356m/A-GCL/audit/s15/"
BASE = M.BASE                       # 20260818
BUILDER_VERSION = "s15_data v1.0"
GIT = "7970caa"
SMALL_SITE_MIN = 10

def h(x):
    if isinstance(x, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]
    return hashlib.sha256(str(x).encode()).hexdigest()[:16]

def source_manifest_sha():
    return K.sha(K.S11 + "manifest/subject_manifest.csv")

def cache_path():
    return f"{S15}cache/s15_data_{source_manifest_sha()[:16]}.npz"

def manifest_path():
    return f"{S15}CACHE_MANIFEST.json"

# ------------------------------------------------------------------ GATE-C
def gate_c(d, man, where="job"):
    """Every assertion from Section 2.3. Runs in the builder AND every job.
    Fails LOUD: prints expected vs actual, writes a FAILED record, exits non-zero."""
    def fail(msg, exp=None, act=None):
        rec = dict(where=where, error=msg, expected=str(exp), actual=str(act),
                   host=socket.gethostname(), time=time.strftime("%F %T"))
        os.makedirs(S15 + "out", exist_ok=True)
        p = f"{S15}out/GATE_C_FAILED_{where}_{os.getpid()}.json"
        json.dump(rec, open(p, "w"), indent=1)
        print(f"GATE-C FAIL [{where}]: {msg} | expected={exp} actual={act}", flush=True)
        sys.exit(3)
    FC, ALFF, y = d["FC"], d["ALFF"], d["y"]
    if len(y) != 954: fail("len != 954", 954, len(y))
    n_asd, n_nc = int((y == 1).sum()), int((y == 0).sum())
    if (n_asd, n_nc) != (455, 499): fail("ASD/NC != S11 manifest", (455, 499), (n_asd, n_nc))
    if FC.shape != (954, 90, 90): fail("FC.shape", (954, 90, 90), FC.shape)
    if ALFF.shape != (954, 90, 3): fail("ALFF.shape", (954, 90, 3), ALFF.shape)
    if y.shape != (954,): fail("y.shape", (954,), y.shape)
    if FC.dtype != np.float32: fail("FC.dtype", "float32", FC.dtype)
    if ALFF.dtype != np.float32: fail("ALFF.dtype", "float32", ALFF.dtype)
    sym = float(np.abs(FC - FC.transpose(0, 2, 1)).max())
    if not sym < 1e-6: fail("FC symmetry", "<1e-6", sym)
    dia = float(np.abs(FC[:, np.arange(90), np.arange(90)] - 1.0).max())
    if dia != 0.0: fail("diag(FC) != 1.0 exactly", 0.0, dia)
    mx = float(np.abs(FC).max())
    if not mx <= 1.0 + 1e-6: fail("|FC| > 1", "<=1+1e-6", mx)
    if not np.isfinite(FC).all(): fail("NaN/Inf in FC", "finite", "non-finite")
    if not np.isfinite(ALFF).all(): fail("NaN/Inf in ALFF", "finite", "non-finite")
    # hash identity against the FROZEN S11 sources (re-anchored on EVERY job)
    exp = expected_s11_hashes(full=(where == "builder"))
    for k, v in exp.items():
        if man[k] != v: fail(f"{k} != S11", v, man[k])
    # ALFF: anchored per job by the manifest value that the builder verified against
    # M1_B, plus verify_frozen_hashes() asserting the M1_B cache file SHA itself.
    if where != "builder" and man["h_alff_band_order"] != man["h_alff"]:
        fail("h_alff_band_order != h_alff", man["h_alff"], man["h_alff_band_order"])
    return True

def expected_s11_hashes(full=False):
    """Anchored to the FROZEN S11 sources. full=True re-derives ALFF from the 954
    .mat files (BUILD ONLY, ~minutes). full=False is the per-job path: it verifies
    the frozen source FILE SHAs plus the small frozen X_fc archive, which anchors
    subject order, labels, folds and FC edge order in ~1 s. ALFF provenance is
    anchored by the M1_B cache file SHA in verify_frozen_hashes(), not re-derived."""
    K.verify_frozen_hashes()          # M1_B_v1.pt, splits.json, ROI manifest SHAs
    X_fc, y, ids, meta = K.load_Xfc()  # asserts the frozen X_fc sha internally
    lab = np.full(954, -1, dtype=np.int16)
    for i, (tr, te) in enumerate(K.folds_ordinary()): lab[np.asarray(te)] = i
    out = dict(h_subject_order=h("|".join(ids)),
               h_labels=h(y.astype(np.int64)),
               h_folds_lab=h(lab),
               h_fc_edge_order=h(X_fc.astype(np.float64)))
    if full:
        df, _x, _y, _i, _g = A1.load_gate()
        _, ALFF, _s = A1.load_tensors(df)
        out["h_alff_band_order"] = h(ALFF.astype(np.float32))
    return out

# ------------------------------------------------------------------ folds
def build_folds(y, sites):
    """F-LAB loaded frozen from S11. F-SITE built once here. F-LOSO from S11."""
    from sklearn.model_selection import StratifiedKFold
    lab = np.full(954, -1, dtype=np.int16)
    for i, (tr, te) in enumerate(K.folds_ordinary()): lab[np.asarray(te)] = i
    assert (lab >= 0).all(), "F-LAB does not cover all subjects"
    # F-SITE: stratify on (label, site) with small sites pooled
    cnt = {}
    for s in sites: cnt[s] = cnt.get(s, 0) + 1
    pooled = sorted([s for s, c in cnt.items() if c < SMALL_SITE_MIN])
    site_eff = [("SMALL_SITE" if cnt[s] < SMALL_SITE_MIN else s) for s in sites]
    key = np.array([f"{int(yy)}|{ss}" for yy, ss in zip(y, site_eff)])
    site = np.full(954, -1, dtype=np.int16)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=BASE)
    for i, (tr, te) in enumerate(skf.split(np.zeros(954), key)): site[te] = i
    assert (site >= 0).all()
    # F-LOSO from S11 (only sites with both classes are evaluable)
    loso = np.full(954, -1, dtype=np.int16)
    for i, (tr, te) in enumerate(K.folds_loso(y)): loso[np.asarray(te)] = i
    n_loso = int(loso.max()) + 1
    return lab, site, loso, n_loso, pooled

# ------------------------------------------------------------------ build
def build():
    cp = cache_path()
    if os.path.exists(cp): os.remove(cp)
    df, X_fc, y, ids, gh = A1.load_gate()          # asserts every frozen S11 hash
    FC, ALFF, stats = A1.load_tensors(df)          # FC from canonical .mat, ALFF = M1_B
    import pandas as pd
    meta = pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv")
    assert list(meta.subject_id) == ids and np.array_equal(meta.y.values, y)
    sites = list(meta.site)
    FC = FC.astype(np.float32); ALFF = ALFF.astype(np.float32); y = y.astype(np.int8)
    # provenance identity: the .mat-derived FC must BE the frozen X_fc
    tri = FC.astype(np.float64)[:, K.IU[0], K.IU[1]]
    assert np.abs(tri - X_fc.astype(np.float64)).max() < 1e-6, "FC != frozen X_fc"
    assert stats["mism"] == 0 and stats["x_max"] == 0.0
    lab, site, loso, n_loso, pooled = build_folds(y.astype(np.int64), sites)
    man = dict(
        h_source_manifest=source_manifest_sha()[:16],
        h_fc=h(FC), h_alff=h(ALFF), h_labels=h(y.astype(np.int64)),
        h_subject_order=h("|".join(ids)),
        h_folds_lab=h(lab), h_folds_site=h(site), h_folds_loso=h(loso),
        h_fc_edge_order=h(X_fc.astype(np.float64)), h_alff_band_order=h(ALFF),
        git_sha=GIT, builder_version=BUILDER_VERSION,
        build_timestamp=time.strftime("%F %T"), cache_file=os.path.basename(cp),
        n_folds=dict(lab=int(lab.max()) + 1, site=int(site.max()) + 1, loso=n_loso),
        small_sites_pooled=pooled, host=socket.gethostname(),
        fc_sym_max=float(np.abs(FC - FC.transpose(0, 2, 1)).max()),
        fc_diag_dev=float(np.abs(FC[:, np.arange(90), np.arange(90)] - 1.0).max()),
        fc_absmax=float(np.abs(FC).max()))
    d = dict(FC=FC, ALFF=ALFF, y=y)
    gate_c(d, man, where="builder")
    tmp = cp + ".tmp.npz"
    np.savez_compressed(tmp[:-4], FC=FC, ALFF=ALFF, y=y,
                        subject_ids=np.array(ids), site_labels=np.array(sites),
                        fold_lab=lab, fold_site=site, fold_loso=loso)
    z = np.load(tmp); assert z["FC"].shape == (954, 90, 90); os.replace(tmp, cp)
    json.dump(man, open(manifest_path() + ".tmp", "w"), indent=1)
    os.replace(manifest_path() + ".tmp", manifest_path())
    print(f"BUILT {cp} ({os.path.getsize(cp)/1e6:.1f} MB)", flush=True)
    print(json.dumps(man, indent=1), flush=True)
    return cp, man

# ------------------------------------------------------------------ load
_CACHE = {}
def load(where="job"):
    """READ-ONLY (mmap) load + Gate-C. Called at the start of every job."""
    if "d" in _CACHE: return _CACHE["d"], _CACHE["m"]
    cp = cache_path()
    assert os.path.exists(cp), f"cache missing: {cp} — run s15_data.py first"
    man = json.load(open(manifest_path()))
    z = np.load(cp, mmap_mode="r")
    d = dict(FC=np.asarray(z["FC"]), ALFF=np.asarray(z["ALFF"]),
             y=np.asarray(z["y"]), ids=[str(s) for s in z["subject_ids"]],
             sites=[str(s) for s in z["site_labels"]],
             fold_lab=np.asarray(z["fold_lab"]), fold_site=np.asarray(z["fold_site"]),
             fold_loso=np.asarray(z["fold_loso"]))
    # cache-content identity, not just the manifest's word for it
    for k, hk in (("FC", "h_fc"), ("ALFF", "h_alff"), ("fold_lab", "h_folds_lab"),
                  ("fold_site", "h_folds_site"), ("fold_loso", "h_folds_loso")):
        got = h(d[k])
        if got != man[hk]:
            print(f"GATE-C FAIL [{where}]: {hk} expected={man[hk]} actual={got}", flush=True)
            sys.exit(3)
    gate_c(d, man, where=where)
    _CACHE["d"], _CACHE["m"] = d, man
    return d, man

def folds(d, protocol):
    """-> [(tag, train_idx, test_idx)] for 'lab' | 'site' | 'loso'."""
    a = d[f"fold_{protocol}"]; out = []
    y = d["y"]
    for i in range(int(a.max()) + 1):
        te = np.where(a == i)[0]; tr = np.where((a != i) & (a >= 0))[0]
        if len(te) == 0 or len(np.unique(y[te])) < 2: continue
        out.append((f"{protocol}{i}", tr, te))
    return out

def all_folds(d):
    return folds(d, "lab") + folds(d, "site") + folds(d, "loso")

if __name__ == "__main__":
    build()
    d, m = load(where="builder-verify")
    print(f"VERIFY OK — folds: lab {len(folds(d,'lab'))} site {len(folds(d,'site'))} "
          f"loso {len(folds(d,'loso'))} = {len(all_folds(d))} total", flush=True)
