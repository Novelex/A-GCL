"""S13 core — Brain Network Transformer (Kan et al., NeurIPS 2022) on the frozen
ABIDE cohort. Model + OCREAD + gates + training. CPU, deterministic.
READ-ONLY imports: s11_core (K), s12a1_core (A1), s12a5_core (M). Nothing outside
audit/s13 is written."""
import sys, os, json, copy, time, socket, hashlib, subprocess
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11");            import s11_core as K
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s7");             import s7_core as C7
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts");  import s12a1_core as A1
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a5/scripts");  import s12a5_core as M

from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix,
    brier_score_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

S13  = "/users/3171356m/A-GCL/audit/s13/"
BASE = M.BASE                                   # 20260818, the frozen audit seed
SEEDS = [BASE, BASE + 1, BASE + 2]
GIT  = "9ea8e5f"

# ---- fixed model constants (NOT swept) ----
H_HID, N_LAYERS, N_HEADS = 128, 2, 4
D_K = H_HID // N_HEADS                          # 32
FFN_MULT = 2
P_ATTN = P_FFN = P_HEAD = 0.30
# ---- swept ----
K_GRID  = [2, 4, 8]
WD_GRID = [1e-4, 1e-3]
# ---- training ----
LR, BATCH, MAX_EPOCHS, PATIENCE, MIN_DELTA = 1e-4, 64, 200, 20, 1e-6
LABEL_SMOOTH, CLIP = 0.10, 1.0
PARAM_BUDGET = 1_030_000                        # EdgeMLP that overfit 96-100% of folds

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================== GATE 0 ===============================
def manifest_sha(): return K.sha(K.S11 + "manifest/subject_manifest.csv")
def cache_path():   return f"{S13}cache/data_s13_{manifest_sha()[:16]}.pt"

def build_cache():
    """Fresh namespace, DELETE-and-rebuild, hard asserts, no try/except, no fallback."""
    cp = cache_path()
    if os.path.exists(cp): os.remove(cp)
    df, X_fc, y, ids, gh = A1.load_gate()          # asserts every frozen S11 hash
    FC, ALFF, stats = A1.load_tensors(df)          # FC from canonical .mat; ALFF = M1_B
    assert len(ids) == 954, f"n={len(ids)} != 954"
    assert int((y == 1).sum()) == 455 and int((y == 0).sum()) == 499
    assert FC.shape == (954, 90, 90), FC.shape
    assert ALFF.shape == (954, 90, 3), ALFF.shape
    assert np.isfinite(FC).all() and np.isfinite(ALFF).all(), "NaN/Inf in FC or ALFF"
    sym = float(max(np.abs(FC[i] - FC[i].T).max() for i in range(954)))
    dia = float(max(np.abs(np.diag(FC[i]) - 1.0).max() for i in range(954)))
    assert sym < 1e-6, f"FC not symmetric: {sym}"
    assert dia == 0.0 or dia < 1e-12, f"diag(FC) != 1.0 exactly: {dia}"
    tri = FC[:, K.IU[0], K.IU[1]]
    assert np.abs(tri - X_fc.astype(np.float64)).max() == 0.0, "FC row-major order != S11 X_fc"
    assert stats["mism"] == 0 and stats["x_max"] == 0.0
    fold_sha = K.sha("/users/3171356m/agcl_audit_s0/s3c/splits.json")
    assert fold_sha == K.SPLITS_SHA, "fold authority drift"
    meta = pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv")
    assert list(meta.subject_id) == ids and np.array_equal(meta.y.values, y)
    obj = dict(
        FC=FC.astype(np.float32), ALFF=ALFF.astype(np.float32),
        X_fc=X_fc.astype(np.float64), y=y.astype(np.int64), ids=ids,
        site=list(meta.site), manifest_sha=manifest_sha(), splits_sha=fold_sha,
        xfc_sha=gh["X_fc_sha256"], dataset_sha=K.DATASET_SHA,
        id_order_sha=hashlib.sha256("|".join(ids).encode()).hexdigest(),
        y_sha=hashlib.sha256(np.ascontiguousarray(y.astype(np.int64)).tobytes()).hexdigest(),
        alff_sha=hashlib.sha256(np.ascontiguousarray(ALFF.astype(np.float32)).tobytes()).hexdigest(),
        fc_sym_max=sym, fc_diag_dev=dia, fc_stats=stats)
    torch.save(obj, cp + ".tmp"); os.replace(cp + ".tmp", cp)
    return cp, obj

def load_all():
    """Gate-0 re-verification. Called at the START OF EVERY JOB (Gate-2 test 9)."""
    cp = cache_path()
    assert os.path.exists(cp), "Gate-0 cache missing — run gate0.py"
    K.verify_frozen_hashes()
    d = torch.load(cp, weights_only=False)
    assert len(d["ids"]) == 954 and d["manifest_sha"] == manifest_sha()
    assert d["splits_sha"] == K.SPLITS_SHA
    a = hashlib.sha256(np.ascontiguousarray(d["X_fc"]).tobytes()).hexdigest()
    assert a == d["xfc_sha"], "X_fc drift"
    Xf, y, ids, _ = K.load_Xfc()
    assert np.array_equal(Xf, d["X_fc"]) and np.array_equal(y, d["y"])
    assert list(ids) == list(d["ids"])
    assert hashlib.sha256(np.ascontiguousarray(d["ALFF"]).tobytes()).hexdigest() == d["alff_sha"]
    return d

def folds_all(y):
    F = [(f"o{i}", np.asarray(tr), np.asarray(te))
         for i, (tr, te) in enumerate(K.folds_ordinary())]
    F += [(f"l{i}", np.asarray(tr), np.asarray(te))
          for i, (tr, te) in enumerate(K.folds_loso(y))]
    return F

# =============================== FEATURES ===============================
def alff_scaled(ALFF, tr, mode):
    """mode 'z': z-score ACROSS SUBJECTS per (ROI,band), mu/sd from TRAIN ONLY.
       mode 'minmax': per-subject per-band min-max (production v3 recipe)."""
    A = ALFF.astype(np.float64)
    if mode == "z":
        mu = A[tr].mean(0, keepdims=True)
        sd = A[tr].std(0, keepdims=True)
        return ((A - mu) / np.maximum(sd, 1e-6)).astype(np.float32)
    if mode == "minmax":
        mn = A.min(1, keepdims=True); mx = A.max(1, keepdims=True)
        span = mx - mn
        return np.where(span > 0, (A - mn) / np.where(span > 0, span, 1.0), A).astype(np.float32)
    raise ValueError(mode)

ARM_SPEC = {   # arm -> (use_alff, alff_mode, shuffle_fc_cols, permute_labels)
    "T1": (False, None,     False, False),
    "T2": (True,  "z",      False, False),
    "T4": (True,  "minmax", False, False),
    "T5": (True,  "z",      True,  False),
    "T6": (True,  "z",      False, True),
}

def arm_X(arm, FC, ALFF, tr):
    """[954,90,D] float32. FC row per node; controls act on the feature pathway."""
    use_alff, mode, shuf, _ = ARM_SPEC[arm]
    R = FC.astype(np.float32)
    if shuf:                                    # T5: one column perm per subject
        Rs = np.empty_like(R)
        for s in range(len(R)):
            p = np.random.default_rng(BASE + s).permutation(90)
            Rs[s] = R[s][:, p]
        R = Rs
    if not use_alff: return R
    return np.concatenate([R, alff_scaled(ALFF, tr, mode)], axis=2)

def arm_y(arm, y):
    if ARM_SPEC[arm][3]:                        # T6 leakage floor
        yp = np.random.default_rng(BASE).permutation(y)
        assert yp.sum() == y.sum() and (yp == y).mean() < 1.0
        return yp
    return y

# =============================== MODEL ===============================
def gram_schmidt(C):
    """Modified Gram-Schmidt; returns orthonormal rows (unit L2)."""
    Kc, Hh = C.shape
    E = torch.zeros_like(C)
    for i in range(Kc):
        v = C[i].clone()
        for j in range(i):
            v = v - torch.dot(E[j], v) * E[j]
        n = torch.linalg.norm(v)
        assert float(n) > 1e-8, "degenerate cluster centre"
        E[i] = v / n
    return E

class MHSA(nn.Module):
    def __init__(self, H, heads, p):
        super().__init__()
        self.h, self.dk = heads, H // heads
        self.q = nn.Linear(H, H, bias=False)
        self.k = nn.Linear(H, H, bias=False)
        self.v = nn.Linear(H, H, bias=False)
        self.o = nn.Linear(H, H)
        self.drop = nn.Dropout(p)
        self.last_attn = None
    def forward(self, Z, keep_attn=False):
        B, N, H = Z.shape
        sh = lambda t: t.view(B, N, self.h, self.dk).transpose(1, 2)   # [B,h,N,dk]
        q, k, v = sh(self.q(Z)), sh(self.k(Z)), sh(self.v(Z))
        att = torch.softmax(q @ k.transpose(-2, -1) / (self.dk ** 0.5), dim=-1)
        if keep_attn: self.last_attn = att.detach()
        out = (self.drop(att) @ v).transpose(1, 2).reshape(B, N, H)
        return self.o(out)

class Block(nn.Module):
    """PRE-NORM residual block."""
    def __init__(self, H, heads, p_attn, p_ffn):
        super().__init__()
        self.n1 = nn.LayerNorm(H); self.attn = MHSA(H, heads, p_attn)
        self.d1 = nn.Dropout(p_attn)
        self.n2 = nn.LayerNorm(H)
        self.ffn = nn.Sequential(nn.Linear(H, FFN_MULT * H), nn.GELU(),
                                 nn.Dropout(p_ffn), nn.Linear(FFN_MULT * H, H))
        self.d2 = nn.Dropout(p_ffn)
    def forward(self, Z, keep_attn=False):
        Z = Z + self.d1(self.attn(self.n1(Z), keep_attn))
        Z = Z + self.d2(self.ffn(self.n2(Z)))
        return Z

class BNTModel(nn.Module):
    """Brain Network Transformer with Orthonormal Clustering READout.
    FC edge weights NEVER enter the attention scores; FC enters only as features."""
    def __init__(self, arm, seed, K_clusters, D, H=H_HID, n_layers=N_LAYERS,
                 heads=N_HEADS, p_attn=P_ATTN, p_ffn=P_FFN, p_head=P_HEAD):
        super().__init__()
        assert H >= D, f"H({H}) must be >= D({D}): no compression allowed"
        torch.manual_seed(seed); np.random.seed(seed % 2**32)
        self.arm, self.K, self.D, self.H = arm, K_clusters, D, H
        self.inp = nn.Linear(D, H)
        self.blocks = nn.ModuleList([Block(H, heads, p_attn, p_ffn)
                                     for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(H)
        self.head = nn.Sequential(
            nn.LayerNorm(K_clusters * H), nn.Dropout(p_head),
            nn.Linear(K_clusters * H, 32), nn.LeakyReLU(0.1),
            nn.Dropout(p_head), nn.Linear(32, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        C = torch.empty(K_clusters, H); nn.init.xavier_uniform_(C)
        self.register_buffer("E", gram_schmidt(C))          # BUFFER, not Parameter
    def encode(self, X, keep_attn=False):
        Z = self.inp(X)
        for b in self.blocks: Z = b(Z, keep_attn)
        return self.norm_f(Z)                                # Z_L [B,90,H]
    def ocread(self, Z_L):
        P = torch.softmax(Z_L @ self.E.t(), dim=-1)          # [B,90,K] over K
        Z_G = P.transpose(1, 2) @ Z_L                        # [B,K,H]
        return Z_G, P
    def repr_of(self, batch, edge_vec=None):
        Z_G, _ = self.ocread(self.encode(batch.X))
        return Z_G.reshape(Z_G.shape[0], -1)
    def forward(self, batch, edge_vec=None):
        r = self.repr_of(batch, edge_vec)
        return r, self.head(r).squeeze(-1)

def n_params(m): return int(sum(p.numel() for p in m.parameters() if p.requires_grad))

class Batch:
    __slots__ = ("X",)
    def __init__(self, X): self.X = X

def make_batch(X, idxs, device="cpu"):
    return Batch(torch.tensor(X[np.asarray(idxs)], dtype=torch.float32, device=device))

# =============================== METRICS ===============================
GROUPS = {"inp": ("inp.",), "mhsa": ("blocks.",), "head": ("head.", "norm_f.")}

def group_grad_norms(model):
    out = {}
    for g, pref in GROUPS.items():
        s = 0.0
        for k, p in model.named_parameters():
            if p.grad is not None and any(k.startswith(x) for x in pref):
                s += float(p.grad.norm()) ** 2
        out[g] = s ** 0.5
    return out

def movement(init, model):
    out = {}
    for g, pref in GROUPS.items():
        di, df = [], []
        for k, v in model.state_dict().items():
            if not any(k.startswith(x) for x in pref): continue
            if not torch.is_floating_point(v): continue
            di.append(init[k].float().flatten()); df.append(v.cpu().float().flatten())
        if di:
            i0, f0 = torch.cat(di), torch.cat(df)
            out[g] = float((f0 - i0).norm() / i0.norm().clamp(min=1e-12))
    return out

def _boot_auc(y, s, B=2000, seed=BASE):
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed); n = len(y)
    idx = rng.integers(0, n, (B, n)); Y = y[idx].astype(np.float64); S = s[idx]
    npos = Y.sum(1); nneg = n - npos; ok = (npos > 0) & (nneg > 0)
    r = rankdata(S, method="average", axis=1)
    auc = ((r * Y).sum(1) - npos * (npos + 1) / 2) / np.maximum(npos * nneg, 1)
    return auc[ok]

def metric_block(y, score, boot=2000, seed=BASE):
    """score = logit (head) or LinearSVC decision value (probe). Threshold: score>0.
    Brier/calibration use sigmoid(score) — genuine for the head, INDICATIVE ONLY for
    the probe (LinearSVC scores are uncalibrated). Declared in PROTOCOL.md."""
    y = np.asarray(y); score = np.asarray(score, dtype=np.float64)
    if len(np.unique(y)) < 2: return dict(auc=float("nan"), n=int(len(y)))
    yh = (score > 0).astype(int)
    p = 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    bs = _boot_auc(y, score, boot, seed)
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    cal = LogisticRegression(C=1e10, max_iter=5000).fit(lg[:, None], y)
    return dict(auc=float(roc_auc_score(y, score)),
        auc_ci_lo=float(np.percentile(bs, 2.5)), auc_ci_hi=float(np.percentile(bs, 97.5)),
        auprc=float(average_precision_score(y, score)),
        acc=float(accuracy_score(y, yh)), bal_acc=float(balanced_accuracy_score(y, yh)),
        sens=float(tp / max(tp + fn, 1)), spec=float(tn / max(tn + fp, 1)),
        ppv=float(tp / max(tp + fp, 1)), npv=float(tn / max(tn + fn, 1)),
        f1=float(f1_score(y, yh)), mcc=float(matthews_corrcoef(y, yh)),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        brier=float(brier_score_loss(y, p)),
        calib_slope=float(cal.coef_[0, 0]), calib_intercept=float(cal.intercept_[0]),
        threshold="score>0 (sigmoid 0.5 for head; LinearSVC boundary for probe)",
        n=int(len(y)))

# =============================== TRAIN / EXTRACT ===============================
@torch.no_grad()
def extract(model, X, idxs, bs=128):
    model.eval(); R, S = [], []
    idxs = np.asarray(idxs)
    for lo in range(0, len(idxs), bs):
        b = make_batch(X, idxs[lo:lo + bs])
        r, lg = model(b, None)
        R.append(r.numpy()); S.append(lg.numpy())
    return np.concatenate(R), np.concatenate(S)

def train_fold(arm, K_clusters, wd, seed, tr, X, y, log=None, max_epochs=MAX_EPOCHS,
               smoothing=LABEL_SMOOTH):
    """Mirrors M.train_fold5's structure, parameterised. Early stop on VAL AUC only."""
    itr, iva = train_test_split(np.arange(len(tr)), test_size=0.20,
                                stratify=y[tr], random_state=BASE)
    itr, iva = np.asarray(tr)[itr], np.asarray(tr)[iva]
    torch.manual_seed(seed); np.random.seed(seed % 2**32)
    model = BNTModel(arm, seed, K_clusters, X.shape[2])
    init = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    yt = torch.tensor(y, dtype=torch.float32)
    tsm = yt * (1 - smoothing) + smoothing / 2                 # y*0.90 + 0.05
    curve, best, bad = [], (-1.0, None, 0), 0
    for ep in range(1, max_epochs + 1):
        t_ep = time.time(); model.train()
        perm = torch.randperm(len(itr), generator=gen).numpy()
        tl, gns, clips, nb = [], [], 0, 0
        for lo in range(0, len(perm), BATCH):
            ii = itr[perm[lo:lo + BATCH]]
            if len(ii) < 2: continue
            opt.zero_grad()
            _, lg = model(make_batch(X, ii), None)
            loss = lossf(lg, tsm[ii])
            loss.backward()
            assert torch.isfinite(loss), f"non-finite loss ep{ep}"
            gns.append(group_grad_norms(model))
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            assert torch.isfinite(gn), f"non-finite grad ep{ep}"
            if float(gn) > CLIP: clips += 1
            nb += 1; tl.append(float(loss)); opt.step()
        model.eval()
        with torch.no_grad():
            _, st = extract(model, X, itr); _, sv = extract(model, X, iva)
            vloss = float(lossf(torch.tensor(sv), tsm[iva]))
        row = dict(epoch=ep, lr=LR, train_loss=float(np.mean(tl)), val_loss=vloss,
                   train_auc=float(roc_auc_score(y[itr], st)),
                   val_auc=float(roc_auc_score(y[iva], sv)),
                   grad_inp=float(np.mean([g["inp"] for g in gns])),
                   grad_mhsa=float(np.mean([g["mhsa"] for g in gns])),
                   grad_head=float(np.mean([g["head"] for g in gns])),
                   clip_events=int(clips), n_batches=int(nb),
                   epoch_s=round(time.time() - t_ep, 2))
        curve.append(row)
        if row["val_auc"] > best[0] + MIN_DELTA:
            best = (row["val_auc"],
                    copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}), ep)
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE: break
        if log and ep % 20 == 0:
            print(f"[{log}] ep{ep} vl {vloss:.4f} va {row['val_auc']:.4f} "
                  f"clip {clips}/{nb}", flush=True)
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        _, sv2 = extract(model, X, iva)
    va2 = float(roc_auc_score(y[iva], sv2))
    assert va2 == best[0], f"checkpoint reload mismatch {va2} != {best[0]}"
    bl = curve[best[2] - 1]
    gap = bl["train_auc"] - bl["val_auc"]
    verdict = ("OVERFIT" if gap > 0.15 else
               "UNDERFIT" if bl["train_auc"] < 0.65 else "HEALTHY")
    integ = dict(loss_finite=True, grads_finite=True,
                 loss_decreased=bool(bl["train_loss"] < curve[0]["train_loss"]),
                 selection_val_only=True, ckpt_reload_bitwise=True)
    assert integ["loss_decreased"], "loss did not decrease to the selected epoch"
    info = dict(best_val_auc=best[0], best_epoch=best[2], epochs_run=len(curve),
                train_val_gap=float(gap), verdict=verdict, integrity=integ,
                movement=movement(init, model), n_params=n_params(model),
                n_train=len(itr), n_val=len(iva))
    return model, curve, info

# =============================== IO ===============================
def atomic_json(obj, path):
    json.dump(obj, open(path + ".tmp", "w"), indent=1, default=str)
    json.load(open(path + ".tmp")); os.replace(path + ".tmp", path)

def atomic_text(text, path):
    open(path + ".tmp", "w").write(text)
    assert open(path + ".tmp").read() == text
    os.replace(path + ".tmp", path)

def peak_rss_mb():
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)

def provenance(extra=None):
    p = dict(git=GIT, host=socket.gethostname(), time=time.strftime("%F %T"),
             python=sys.version.split()[0], torch=torch.__version__,
             numpy=np.__version__, cores=int(os.environ.get("SLURM_CPUS_PER_TASK",
             os.cpu_count())), omp=os.environ.get("OMP_NUM_THREADS", ""))
    import sklearn; p["sklearn"] = sklearn.__version__
    if extra: p.update(extra)
    return p
