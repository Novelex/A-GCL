"""S16 data: FOUR caches, one per edge/profile treatment E.
E in {signed, abs, pos_zero, shift}. Gate-C runs in the builder AND every job.
No try/except around loading, no fallback, no silent default. Fail loud."""
import sys, os, json, time, socket, hashlib, numpy as np

sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11");           import s11_core as K
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s7");            import s7_core as C7
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a5/scripts"); import s12a5_core as M

S16 = "/users/3171356m/A-GCL/audit/s16/"
BASE = M.BASE                      # 20260818
E_LEVELS = ["signed", "abs", "pos_zero", "shift"]
BUILDER_VERSION = "s16_data v1.0"
def _runtime_sha():
    """Runtime git SHA. A hard-coded constant here was stale (defect D11): it read
    d52798c while HEAD had moved on, misattributing every provenance record."""
    import subprocess
    try:
        return subprocess.run(["git","-C","/users/3171356m/A-GCL","rev-parse","HEAD"],
                              capture_output=True,text=True).stdout.strip()[:12] or "unknown"
    except Exception: return "unknown"
GIT = _runtime_sha()
SMALL_SITE_MIN = 10

def h(x):
    if isinstance(x, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]
    return hashlib.sha256(str(x).encode()).hexdigest()[:16]

def src_sha():  return K.sha(K.S11 + "manifest/subject_manifest.csv")
def cache_path(E): return f"{S16}cache/s16_{E}_{src_sha()[:16]}.npz"
def manifest_path(): return f"{S16}CACHE_MANIFEST.json"

# ------------------------------------------------------------------ E transform
def apply_E(FC, E):
    """Returns (FCt, sparse) — FCt is the transformed 954x90x90 matrix.
    sparse=True means the graph is subject-specific: edges are the NONZERO entries."""
    F = FC.astype(np.float32)
    if E == "signed":   return F, False
    if E == "abs":      return np.abs(F), False
    if E == "shift":    return ((F + 1.0) / 2.0).astype(np.float32), False
    if E == "pos_zero":
        Z = F.copy(); Z[Z < 0] = 0.0
        return Z, True
    raise ValueError(E)

def sparse_stats(FCt):
    """pos_zero diagnostics. Per-subject edge counts AND per-node degree, because
    diag(FC)=1.0 exactly guarantees >=90 edges per subject, making the per-subject
    flag vacuous — isolation is a PER-NODE property (see PROTOCOL_ADDENDA F2)."""
    nz = (FCt != 0)
    per_sub = nz.reshape(len(FCt), -1).sum(1)
    deg = nz.sum(2)                                   # [954,90] out-degree incl. self
    iso_nodes = (deg <= 1)                            # self-loop only -> isolated
    return dict(edges_min=int(per_sub.min()), edges_med=int(np.median(per_sub)),
                edges_max=int(per_sub.max()),
                pct_of_8100=float(100.0 * per_sub.mean() / 8100.0),
                subjects_under_90_edges=int((per_sub < 90).sum()),
                isolated_nodes_total=int(iso_nodes.sum()),
                subjects_with_isolated_node=int(iso_nodes.any(1).sum()),
                min_node_degree=int(deg.min()))

# ------------------------------------------------------------------ Gate-C
def gate_c(FCt, ALFF, y, man, E, where="job"):
    def fail(msg, exp=None, act=None):
        rec = dict(where=where, E=E, error=msg, expected=str(exp), actual=str(act),
                   host=socket.gethostname(), time=time.strftime("%F %T"))
        os.makedirs(S16 + "out", exist_ok=True)
        json.dump(rec, open(f"{S16}out/GATEC_FAILED_{E}_{where}_{os.getpid()}.json","w"), indent=1)
        print(f"GATE-C FAIL [{E}/{where}]: {msg} | expected={exp} actual={act}", flush=True)
        sys.exit(3)
    if len(y) != 954: fail("len != 954", 954, len(y))
    if (int((y==1).sum()), int((y==0).sum())) != (455,499):
        fail("ASD/NC != S11", (455,499), (int((y==1).sum()), int((y==0).sum())))
    if FCt.shape != (954,90,90): fail("FC.shape", (954,90,90), FCt.shape)
    if ALFF.shape != (954,90,3): fail("ALFF.shape", (954,90,3), ALFF.shape)
    if y.shape != (954,): fail("y.shape", (954,), y.shape)
    if FCt.dtype != np.float32: fail("FC.dtype", "float32", FCt.dtype)
    if ALFF.dtype != np.float32: fail("ALFF.dtype", "float32", ALFF.dtype)
    if not np.isfinite(FCt).all(): fail("NaN/Inf in FC", "finite", "non-finite")
    if not np.isfinite(ALFF).all(): fail("NaN/Inf in ALFF", "finite", "non-finite")
    sym = float(np.abs(FCt - FCt.transpose(0,2,1)).max())
    if not sym < 1e-6: fail("FC symmetry", "<1e-6", sym)
    # E-specific range/diag expectations, stated per level rather than assumed
    dia = FCt[:, np.arange(90), np.arange(90)]
    exp_diag = {"signed":1.0, "abs":1.0, "pos_zero":1.0, "shift":1.0}[E]
    if float(np.abs(dia - exp_diag).max()) != 0.0:
        fail(f"diag != {exp_diag} exactly", exp_diag, float(np.abs(dia-exp_diag).max()))
    lim = {"signed":1.0, "abs":1.0, "pos_zero":1.0, "shift":1.0}[E]
    if not float(np.abs(FCt).max()) <= lim + 1e-6: fail("|FC| out of range", lim, float(np.abs(FCt).max()))
    if E in ("abs","pos_zero","shift") and float(FCt.min()) < -1e-6:
        fail(f"{E} must be non-negative", ">=0", float(FCt.min()))
    exp = expected_s11(full=(where=="builder"))
    for k,v in exp.items():
        if man[k] != v: fail(f"{k} != S11", v, man[k])
    return True

def expected_s11(full=False):
    K.verify_frozen_hashes()
    X_fc, y, ids, meta = K.load_Xfc()
    lab = np.full(954, -1, dtype=np.int16)
    for i,(tr,te) in enumerate(K.folds_ordinary()): lab[np.asarray(te)] = i
    out = dict(h_subject_order=h("|".join(ids)), h_labels=h(y.astype(np.int64)),
               h_folds_lab=h(lab), h_fc_edge_order=h(X_fc.astype(np.float64)))
    if full:
        df,_x,_y,_i,_g = A1.load_gate(); _, ALFF, _s = A1.load_tensors(df)
        out["h_alff_band_order"] = h(ALFF.astype(np.float32))
    return out

# ------------------------------------------------------------------ folds
def build_folds(y, sites):
    from sklearn.model_selection import StratifiedKFold
    lab = np.full(954,-1,dtype=np.int16)
    for i,(tr,te) in enumerate(K.folds_ordinary()): lab[np.asarray(te)] = i
    assert (lab>=0).all()
    cnt = {}
    for s in sites: cnt[s] = cnt.get(s,0)+1
    pooled = sorted([s for s,c in cnt.items() if c < SMALL_SITE_MIN])
    eff = [("SMALL_SITE" if cnt[s] < SMALL_SITE_MIN else s) for s in sites]
    key = np.array([f"{int(a)}|{b}" for a,b in zip(y, eff)])
    site = np.full(954,-1,dtype=np.int16)
    for i,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=BASE
                               ).split(np.zeros(954), key)): site[te] = i
    loso = np.full(954,-1,dtype=np.int16)
    for i,(tr,te) in enumerate(K.folds_loso(y)): loso[np.asarray(te)] = i
    return lab, site, loso, pooled

# ------------------------------------------------------------------ build
def build_all():
    df, X_fc, y, ids, gh = A1.load_gate()
    FC0, ALFF, stats = A1.load_tensors(df)
    import pandas as pd
    meta = pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv")
    assert list(meta.subject_id)==ids and np.array_equal(meta.y.values, y)
    sites = list(meta.site)
    ALFF = ALFF.astype(np.float32); y8 = y.astype(np.int8)
    tri = FC0.astype(np.float64)[:, K.IU[0], K.IU[1]]
    assert np.abs(tri - X_fc.astype(np.float64)).max() < 1e-6, "FC != frozen X_fc"
    assert stats["mism"]==0 and stats["x_max"]==0.0
    lab, site, loso, pooled = build_folds(y.astype(np.int64), sites)
    MAN = dict(git_sha=GIT, builder_version=BUILDER_VERSION,
               build_timestamp=time.strftime("%F %T"), host=socket.gethostname(),
               h_source_manifest=src_sha()[:16], h_alff=h(ALFF),
               h_labels=h(y.astype(np.int64)), h_subject_order=h("|".join(ids)),
               h_folds_lab=h(lab), h_folds_site=h(site), h_folds_loso=h(loso),
               h_fc_edge_order=h(X_fc.astype(np.float64)), h_alff_band_order=h(ALFF),
               small_sites_pooled=pooled,
               n_folds=dict(lab=int(lab.max())+1, site=int(site.max())+1,
                            loso=int(loso.max())+1),
               first5_ids=ids[:5], last5_ids=ids[-5:],
               site_counts={s:int(sum(1 for x in sites if x==s)) for s in sorted(set(sites))},
               caches={})
    for E in E_LEVELS:
        cp = cache_path(E)
        if os.path.exists(cp): os.remove(cp)
        FCt, sparse = apply_E(FC0, E)
        ent = dict(E=E, sparse=bool(sparse), cache_file=os.path.basename(cp),
                   h_fc=h(FCt),
                   fc_sym_max=float(np.abs(FCt-FCt.transpose(0,2,1)).max()),
                   fc_diag_dev=float(np.abs(FCt[:,np.arange(90),np.arange(90)]-1.0).max()),
                   fc_absmax=float(np.abs(FCt).max()), fc_min=float(FCt.min()))
        if sparse: ent["sparse_stats"] = sparse_stats(FCt)
        MAN["caches"][E] = ent
        gate_c(FCt, ALFF, y.astype(np.int64), {**MAN, **ent}, E, where="builder")
        tmp = cp + ".tmp.npz"
        np.savez_compressed(tmp[:-4], FC=FCt, ALFF=ALFF, y=y8,
                            subject_ids=np.array(ids), site_labels=np.array(sites),
                            fold_lab=lab, fold_site=site, fold_loso=loso)
        z = np.load(tmp); assert z["FC"].shape==(954,90,90); os.replace(tmp, cp)
        ent["sha256_file"] = K.sha(cp)[:16]
        ent["size_mb"] = round(os.path.getsize(cp)/1e6, 1)
        print(f"BUILT {E:9s} {ent['cache_file']}  sha {ent['sha256_file']} "
              f"({ent['size_mb']} MB)  h_fc {ent['h_fc']}", flush=True)
        if sparse: print(f"          sparse: {ent['sparse_stats']}", flush=True)
    json.dump(MAN, open(manifest_path()+".tmp","w"), indent=1)
    os.replace(manifest_path()+".tmp", manifest_path())
    return MAN

# ------------------------------------------------------------------ load
_C = {}
def load(E, where="job"):
    if E in _C: return _C[E]
    cp = cache_path(E)
    assert os.path.exists(cp), f"cache missing: {cp} — run s16_data.py"
    MAN = json.load(open(manifest_path())); ent = MAN["caches"][E]
    z = np.load(cp, mmap_mode="r")
    d = dict(FC=np.asarray(z["FC"]), ALFF=np.asarray(z["ALFF"]), y=np.asarray(z["y"]),
             ids=[str(s) for s in z["subject_ids"]], sites=[str(s) for s in z["site_labels"]],
             fold_lab=np.asarray(z["fold_lab"]), fold_site=np.asarray(z["fold_site"]),
             fold_loso=np.asarray(z["fold_loso"]), E=E, sparse=ent["sparse"])
    for k,hk,src in (("FC","h_fc",ent),("ALFF","h_alff",MAN),
                     ("fold_lab","h_folds_lab",MAN),("fold_site","h_folds_site",MAN),
                     ("fold_loso","h_folds_loso",MAN)):
        got = h(d[k])
        if got != src[hk]:
            print(f"GATE-C FAIL [{E}/{where}]: {hk} expected={src[hk]} actual={got}", flush=True)
            sys.exit(3)
    gate_c(d["FC"], d["ALFF"], d["y"].astype(np.int64), {**MAN, **ent}, E, where=where)
    _C[E] = (d, MAN, ent)
    return _C[E]

def folds(d, proto):
    a = d[f"fold_{proto}"]; y = d["y"]; out=[]
    for i in range(int(a.max())+1):
        te = np.where(a==i)[0]; tr = np.where((a!=i)&(a>=0))[0]
        if len(te)==0 or len(np.unique(y[te]))<2: continue
        out.append((f"{proto}{i}", tr, te))
    return out

if __name__ == "__main__":
    MAN = build_all()
    for E in E_LEVELS:
        d,_,ent = load(E, where="builder-verify")
        print(f"VERIFY {E:9s} OK  folds lab {len(folds(d,'lab'))} site {len(folds(d,'site'))} "
              f"loso {len(folds(d,'loso'))}  sparse={ent['sparse']}")
