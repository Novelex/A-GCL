"""S12B core — forward-only encoder autopsy. One probe (Gate-1) for the whole audit.
Production WGINConv imported verbatim; TUEncoder replicated with a configurable norm
(parity-checked in Gate 2). No optimizer anywhere in this module."""
import sys, os, json, time, socket, hashlib, numpy as np, pandas as pd, torch
import torch.nn as nn
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s7");  import s7_core as C7
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
sys.path.insert(0, "/users/3171356m/A-GCL")
from unsupervised.convs.wgin_conv import WGINConv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix,
    brier_score_loss, r2_score)

S12B = "/users/3171356m/agcl_audit_s0/s12b/"
BASE = 20260818

# Determinism (Gate-2 finding, 2026-08-23): PyG propagate's CUDA scatter-add is
# atomicAdd-ordered and gave run-to-run drift ~6e-7 relative. Deterministic
# algorithms + the cuBLAS workspace pin make every stage tensor BITWISE
# reproducible on GPU (measured 0.0). Set at import so EVERY job inherits it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
SEEDS = [BASE + i for i in range(5)]
EMBS = [32, 64, 128]
NORMS = ["bn", "ln", "none"]
ARMS = ["A", "B", "C", "D", "Cshuf", "Brand"]
C_GRID = [1e-3, 1e-2, 1e-1, 1, 10, 100]
ALPHAS = np.logspace(-3, 3, 7)
GIT = "3047cf7"

# ---------------------------------------------------------------- Gate-0 data
def manifest_sha():
    return K.sha(K.S11 + "manifest/subject_manifest.csv")

def cache_path():
    return f"{S12B}cache/data_s12b_{manifest_sha()[:16]}.pt"

def build_cache():
    """Fresh namespace; DELETE-and-rebuild; hard asserts; no try/except, no fallback."""
    cp = cache_path()
    if os.path.exists(cp): os.remove(cp)
    df, X_fc, y, ids, gh = A1.load_gate()                 # S11 hashes asserted inside
    FC, M1B, stats = A1.load_tensors(df)                  # bitwise x==M1_B asserted inside
    Z = np.load("/users/3171356m/agcl_audit_s0/s3c/X_sources.npz", allow_pickle=True)
    xid = [str(s) for s in Z["ids"]]
    order = [xid.index(f) for f in ids]
    M1raw = Z["M1"].astype(np.float64)[order]             # raw ALFF, manifest order
    meta = pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv")
    assert list(meta.subject_id) == ids and np.array_equal(meta.y.values, y)
    F, _ = C7.splits()
    fold_sha = K.sha("/users/3171356m/agcl_audit_s0/s3c/splits.json")
    assert fold_sha == K.SPLITS_SHA
    assert len(ids) == 954 and int((y == 1).sum()) == 455 and int((y == 0).sum()) == 499
    assert FC.shape == (954, 90, 90) and M1B.shape == (954, 90, 3)
    # HARD ASSERTS (review R7): the .mat-derived FC must BE the frozen X_fc, and the
    # .mat-vs-graph-cache mismatch counter must be zero — previously only reported.
    fc_tri = FC[:, K.IU[0], K.IU[1]]
    assert np.abs(fc_tri - X_fc.astype(np.float64)).max() == 0.0, "FC(.mat) != frozen X_fc"
    assert stats["mism"] == 0, f"FC .mat vs S5 graph cache mismatches: {stats['mism']}"
    assert stats["sym"] < 1e-9 and stats["diag"] < 1e-9, "FC not symmetric / diag != 1"
    assert stats["x_max"] == 0.0, "node features != canonical M1_B"
    n_edges = 90 * 90
    obj = dict(FC=FC.astype(np.float32), M1B=M1B, M1raw=M1raw,
               X_fc=X_fc.astype(np.float64),        # f64 canonical (review R2/R4/R11)
               y=y.astype(np.int64), ids=ids,
               site=list(meta.site), age=meta.age.values.astype(np.float64),
               sex=meta.sex.values.astype(np.float64),
               fd=meta.func_mean_fd.values.astype(np.float64),
               nodes_per_graph=90, edges_per_graph=n_edges,
               manifest_sha=manifest_sha(), splits_sha=fold_sha,
               xfc_sha=gh["X_fc_sha256"], dataset_sha=K.DATASET_SHA,
               id_order_sha=hashlib.sha256("|".join(ids).encode()).hexdigest(),
               y_sha=hashlib.sha256(np.ascontiguousarray(y.astype(np.int64)).tobytes()).hexdigest(),
               fc_stats=stats)
    torch.save(obj, cp + ".tmp"); os.replace(cp + ".tmp", cp)
    return cp, obj

def load_all():
    """Every job calls this; asserts the Gate-0 contract before doing anything."""
    cp = cache_path()
    assert os.path.exists(cp), "Gate-0 cache missing — run gate0.py first"
    K.verify_frozen_hashes()
    d = torch.load(cp, weights_only=False)
    assert len(d["ids"]) == 954 and d["manifest_sha"] == manifest_sha()
    assert d["splits_sha"] == K.SPLITS_SHA
    # X_fc stored f64 == canonical: the sha is now genuinely ASSERTED (review R4)
    a = hashlib.sha256(np.ascontiguousarray(d["X_fc"]).tobytes()).hexdigest()
    assert a == d["xfc_sha"], "X_fc drift in s12b cache"
    Xf, y, ids, meta = K.load_Xfc()
    assert np.array_equal(Xf, d["X_fc"]), "cached X_fc != live frozen X_fc"
    assert list(ids) == list(d["ids"]) and np.array_equal(y, d["y"])
    fc_tri = d["FC"].astype(np.float64)[:, K.IU[0], K.IU[1]]
    assert np.abs(fc_tri - d["X_fc"]).max() < 1e-6, "FC tensor vs X_fc drift"
    return d

def folds_all(y):
    """[('o0',tr,te)...('l*',tr,te)] — frozen ordinary + LOSO."""
    F = [(f"o{i}", np.asarray(tr), np.asarray(te))
         for i, (tr, te) in enumerate(K.folds_ordinary())]
    F += [(f"l{i}", np.asarray(tr), np.asarray(te))
          for i, (tr, te) in enumerate(K.folds_loso(y))]
    return F

# ---------------------------------------------------------------- feature arms
def arm_features(arm, M1B, FC):
    """[954,90,d] float32. Edge weights are ALWAYS true FC; controls act on features."""
    n = len(M1B); I90 = np.eye(90, dtype=np.float32)
    A = M1B.astype(np.float32)
    if arm == "A": return A
    if arm == "B": return np.concatenate([A, np.repeat(I90[None], n, 0)], 2)
    if arm == "C": return np.concatenate([A, FC.astype(np.float32)], 2)
    if arm == "D":
        return np.concatenate([A, np.repeat(I90[None], n, 0), FC.astype(np.float32)], 2)
    if arm == "Cshuf":
        R = np.empty((n, 90, 90), dtype=np.float32)
        for s in range(n):
            perm = np.random.default_rng(BASE + s).permutation(90)
            R[s] = FC[s][:, perm]                       # one column perm per subject
        return np.concatenate([A, R], 2)
    if arm == "Brand":
        G = (np.random.default_rng(BASE).standard_normal((90, 90)) / np.sqrt(90)
             ).astype(np.float32)
        return np.concatenate([A, np.repeat(G[None], n, 0)], 2)
    raise ValueError(arm)

# ---------------------------------------------------------------- encoder
def edge_struct():
    n = torch.arange(90)
    return torch.stack([n.repeat_interleave(90), n.repeat(90)], 0)

def make_batch(X, FC, idxs, device):
    ei0 = edge_struct(); xs, eis, ews, bt = [], [], [], []
    for j, i in enumerate(idxs):
        xs.append(torch.from_numpy(X[i])); eis.append(ei0 + j * 90)
        ews.append(torch.from_numpy(FC[i].reshape(-1)))
        bt.append(torch.full((90,), j, dtype=torch.long))
    return (torch.cat(xs).to(device), torch.cat(eis, 1).to(device),
            torch.cat(ews).to(device), torch.cat(bt).to(device))

def _norm_layer(kind, emb):
    if kind == "bn":   return nn.BatchNorm1d(emb, momentum=None)  # cumulative stats
    if kind == "ln":   return nn.LayerNorm(emb)
    if kind == "none": return nn.Identity()
    raise ValueError(kind)

class S12BEncoder(nn.Module):
    """2-layer WGIN replica of production TUEncoder forward, norm configurable.
    Module creation order matches TUEncoder (conv0,norm0,conv1,norm1); norm layers
    draw no RNG, so Linear inits are identical across norm variants at a seed."""
    def __init__(self, d_in, emb, norm, mrelu, seed):
        super().__init__()
        torch.manual_seed(seed); np.random.seed(seed % 2**32)
        nn0 = nn.Sequential(nn.Linear(d_in, emb), nn.ReLU(), nn.Linear(emb, emb))
        self.conv0 = WGINConv(nn0, message_relu=mrelu)
        self.n0 = _norm_layer(norm, emb)
        nn1 = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(), nn.Linear(emb, emb))
        self.conv1 = WGINConv(nn1, message_relu=mrelu)
        self.n1 = _norm_layer(norm, emb)
    def stages(self, x, ei, ew):
        h1 = torch.relu(self.n0(self.conv0(x, ei, ew)))   # post_bn_relu=True (fixed)
        h2 = self.n1(self.conv1(h1, ei, ew))              # last layer: no relu (production)
        return h1, h2

def a1_stage(X, FC):
    """S1: A1 = X + FC^T @ X, param-free, message_relu=False (pre-registered),
    diagonal kept (self-loop fork). [954,90,d]."""
    return X + np.einsum("sij,sid->sjd", FC.astype(np.float64),
                         X.astype(np.float64)).astype(np.float32)

@torch.no_grad()
def extract_stages(enc, X, FC, device, bn_train_idx=None, bs=128):
    """Returns H1,H2 [954,90,emb]. If bn_train_idx is given (bn configs only):
    calibrate running stats with ONE train() pass over those subjects first."""
    enc = enc.to(device)
    if bn_train_idx is not None:
        enc.train()
        for lo in range(0, len(bn_train_idx), bs):
            x, ei, ew, _ = make_batch(X, FC, bn_train_idx[lo:lo + bs], device)
            enc.stages(x, ei, ew)
    enc.eval()
    H1, H2 = [], []
    for lo in range(0, len(X), bs):
        idxs = range(lo, min(lo + bs, len(X)))
        x, ei, ew, _ = make_batch(X, FC, idxs, device)
        h1, h2 = enc.stages(x, ei, ew)
        H1.append(h1.cpu().numpy().reshape(len(list(idxs)), 90, -1))
        H2.append(h2.cpu().numpy().reshape(len(list(idxs)), 90, -1))
    return np.concatenate(H1), np.concatenate(H2)

def derived_stages(H2, normalize):
    """S4/S5/S6 from H2 (numpy). normalize=True applies F.normalize(dim=1) per node."""
    Xn = H2
    if normalize:
        nrm = np.linalg.norm(H2, axis=2, keepdims=True)
        Xn = H2 / np.maximum(nrm, 1e-12)
    out = {}
    if normalize: out["S4"] = Xn
    out["S5"] = Xn.sum(1)                                  # global_add_pool
    out["S6"] = Xn.reshape(len(H2), -1)                    # ROI-aware flatten
    return out

# ---------------------------------------------------------------- metrics
def _clip_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def _boot_auc(y, p, B=2000, seed=BASE):
    """Vectorized bootstrap AUC (rank/Mann-Whitney identity, average ranks)."""
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    n = len(y); idx = rng.integers(0, n, (B, n))
    Y = y[idx].astype(np.float64); P = p[idx]
    npos = Y.sum(1); nneg = n - npos
    ok = (npos > 0) & (nneg > 0)
    r = rankdata(P, method="average", axis=1)
    auc = ((r * Y).sum(1) - npos * (npos + 1) / 2) / np.maximum(npos * nneg, 1)
    return auc[ok]

def metric_block(y, p, boot=2000, seed=BASE):
    """Full pre-registered suite from probabilities at threshold 0.5."""
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return dict(auc=float("nan"), n=int(len(y)))
    yh = (p > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    bs = _boot_auc(y, p, boot, seed)
    cal = LogisticRegression(C=1e10, max_iter=5000).fit(_clip_logit(p)[:, None], y)
    return dict(auc=float(roc_auc_score(y, p)),
        auc_ci_lo=float(np.percentile(bs, 2.5)), auc_ci_hi=float(np.percentile(bs, 97.5)),
        auprc=float(average_precision_score(y, p)),
        acc=float(accuracy_score(y, yh)), bal_acc=float(balanced_accuracy_score(y, yh)),
        sens=float(tp / max(tp + fn, 1)), spec=float(tn / max(tn + fp, 1)),
        ppv=float(tp / max(tp + fp, 1)), npv=float(tn / max(tn + fn, 1)),
        f1=float(f1_score(y, yh)), mcc=float(matthews_corrcoef(y, yh)),
        cm=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        brier=float(brier_score_loss(y, p)),
        calib_slope=float(cal.coef_[0, 0]), calib_intercept=float(cal.intercept_[0]),
        threshold=0.5, n=int(len(y)))

# ---------------------------------------------------------------- the probe
class StageEval:
    """Accumulates one stage across folds. I1 always; I2/I3 on ordinary folds."""
    def __init__(self, y, XFC=None, conf=None, keep_proj=True):
        self.y = y; self.XFC = XFC; self.conf = conf; self.keep_proj = keep_proj
        n = len(y)
        self.p_ord = np.full(n, np.nan); self.p_loso = np.full(n, np.nan)
        self.fold_metrics = {}
        self.fc_pred = np.full((n, XFC.shape[1]), np.nan, dtype=np.float32) \
            if XFC is not None else None
        self.cpred = {k: np.full(n, np.nan) for k in (conf or {})}
        self.site_pred = np.full(n, -1, dtype=int)
        self.proj = np.full((n, 200), np.nan, dtype=np.float32) if keep_proj else None
        self.dims = None; self.best_C = {}
        self.n_components = {}                    # per-fold PCA width (review R15)
        self.fc_r2_fold = {}                      # per-fold I2 R2 (review R18)

    def fold_fit(self, Xf, tr, te, tag, njobs=1, site_codes=None):
        y = self.y; self.dims = Xf.shape[1]
        sc = StandardScaler().fit(Xf[tr])
        Ztr = sc.transform(Xf[tr]).astype(np.float32)
        Zte = sc.transform(Xf[te]).astype(np.float32)
        nc = min(200, Xf.shape[1], len(tr) - 1)
        pca = PCA(n_components=nc, svd_solver="randomized", random_state=BASE,
                  iterated_power=4).fit(Ztr)
        Ptr, Pte = pca.transform(Ztr), pca.transform(Zte)
        gs = GridSearchCV(LogisticRegression(max_iter=5000),
                          {"C": C_GRID},
                          cv=StratifiedKFold(5, shuffle=True, random_state=BASE),
                          scoring="roc_auc", n_jobs=njobs)
        gs.fit(Ptr, y[tr])
        p = gs.predict_proba(Pte)[:, 1]
        self.best_C[tag] = float(gs.best_params_["C"])
        self.n_components[tag] = int(nc)
        (self.p_ord if tag.startswith("o") else self.p_loso)[te] = p
        self.fold_metrics[tag] = metric_block(y[te], p, boot=2000)
        if tag.startswith("o"):
            if self.keep_proj:
                self.proj[te, :Pte.shape[1]] = Pte
            if self.XFC is not None:
                # alpha_per_target: the per-edge R2 distribution is the instrument,
                # so each edge gets its own alpha (review R14)
                rr = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Ptr, self.XFC[tr])
                self.fc_pred[te] = rr.predict(Pte)
                self.fc_r2_fold[tag] = float(r2_score(
                    self.XFC[te], self.fc_pred[te], multioutput="variance_weighted"))
            if self.conf:
                for k, v in self.conf.items():
                    m = np.isfinite(v[tr])
                    r = RidgeCV(alphas=ALPHAS).fit(Ptr[m], v[tr][m])
                    self.cpred[k][te] = r.predict(Pte)
            if site_codes is not None:
                sl = LogisticRegression(C=1.0, max_iter=5000).fit(Ptr, site_codes[tr])
                self.site_pred[te] = sl.predict(Pte)

    def finalize(self, site_codes=None):
        y = self.y; out = {}
        co = np.isfinite(self.p_ord); cl = np.isfinite(self.p_loso)
        out["pooled_ordinary"] = metric_block(y[co], self.p_ord[co])
        if cl.any(): out["pooled_loso"] = metric_block(y[cl], self.p_loso[cl])
        out["per_fold"] = self.fold_metrics
        fa = [m["auc"] for t, m in self.fold_metrics.items()
              if t.startswith("o") and np.isfinite(m.get("auc", np.nan))]
        out["fold_auc_sd"] = float(np.std(fa, ddof=1)) if len(fa) > 1 else 0.0
        out["best_C"] = self.best_C; out["dim"] = int(self.dims or 0)
        out["pca_n_components"] = self.n_components
        if self.XFC is not None:
            P = self.fc_pred[co]; T = self.XFC[co]
            r2w = r2_score(T, P, multioutput="variance_weighted")
            r2e = r2_score(T, P, multioutput="raw_values")
            fr = list(self.fc_r2_fold.values())
            out["fc_recon"] = dict(r2=float(r2w),
                mse=float(np.mean((T - P) ** 2)),
                frac_edges_r2_gt_0p5=float((r2e > 0.5).mean()),
                per_edge_r2_pcts={q: float(np.percentile(r2e, q))
                                  for q in (5, 25, 50, 75, 95)},
                per_fold_r2=self.fc_r2_fold,          # review R18: pooled vs per-fold
                per_fold_r2_mean=float(np.mean(fr)) if fr else float("nan"))
        if self.conf:
            out["confounds"] = {k: float(r2_score(v[co & np.isfinite(v)],
                self.cpred[k][co & np.isfinite(v)])) for k, v in self.conf.items()}
        if site_codes is not None:
            m = co & (self.site_pred >= 0)
            out["site"] = dict(
                macro_f1=float(f1_score(site_codes[m], self.site_pred[m], average="macro")),
                acc=float(accuracy_score(site_codes[m], self.site_pred[m])))
        return out

def confound_dict(d):
    meanabs = np.abs(d["X_fc"]).mean(1).astype(np.float64)
    total = d["X_fc"].sum(1).astype(np.float64)
    return dict(age=d["age"], sex=d["sex"], motion=d["fd"],
                meanfc=meanabs, totalfc=total)

def site_codes(d):
    ss = sorted(set(d["site"]))
    return np.array([ss.index(s) for s in d["site"]]), ss

def atomic_json(obj, path):
    json.dump(obj, open(path + ".tmp", "w"), indent=1, default=str)
    json.load(open(path + ".tmp")); os.replace(path + ".tmp", path)

def atomic_text(text, path):
    """TEMP->validate->rename for reports and sentinels (review R3)."""
    open(path + ".tmp", "w").write(text)
    assert open(path + ".tmp").read() == text
    os.replace(path + ".tmp", path)

def fold_counts(y):
    F = folds_all(y)
    return (sum(1 for t, _, _ in F if t.startswith("o")),
            sum(1 for t, _, _ in F if t.startswith("l")))

def provenance(extra=None):
    p = dict(git=GIT, host=socket.gethostname(), time=time.strftime("%F %T"),
             python=sys.version.split()[0], torch=torch.__version__,
             cuda=torch.version.cuda,
             gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    if extra: p.update(extra)
    return p
