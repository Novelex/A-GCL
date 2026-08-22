"""S12A2 core — READOUT ONLY. Inputs: S12A1 identity final_postnorm nodes [954,90,32].
No production file touched. All transforms inside training folds."""
import sys, os, json, hashlib, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts")
import s12a1_core as A1
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE = 20260818
OUT = "/users/3171356m/agcl_audit_s0/s12a2/out"

def load_nodes(seed_idx):
    z = np.load(f"/users/3171356m/agcl_audit_s0/s12a1/out/emb_id_s{seed_idx}.npz")
    N = z["final_postnorm"].astype(np.float64)          # [954,90,32]
    assert N.shape == (954, 90, 32) and np.isfinite(N).all()
    return N

class ReadoutNet(nn.Module):
    """ARM F: flatten -> Linear(2880,32) -> Linear(32,1) CE head. Readout only."""
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(2880, 32)
        self.head = nn.Linear(32, 1)
    def feats(self, x): return self.lin(x)
    def forward(self, x): return self.head(self.lin(x)).squeeze(-1)

def train_readout(Xtr, ytr, seed, max_epochs=200, patience=20, verbose=False):
    """Train per OUTER fold on outer-train only. Early stop on VALIDATION AUC (never accuracy)."""
    torch.manual_seed(seed); np.random.seed(seed)
    itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.2,
                                stratify=ytr, random_state=BASE)
    mu, sd = Xtr[itr].mean(0), Xtr[itr].std(0) + 1e-8    # scaler fit on inner-train only
    T = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32)
    Xt, Xv = T(Xtr[itr]), T(Xtr[iva])
    yt = torch.tensor(ytr[itr], dtype=torch.float32); yv = ytr[iva]
    net = ReadoutNet(seed); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    best_auc, best_state, bad, best_ep = -1.0, None, 0, 0
    for ep in range(max_epochs):
        net.train()
        for idx in torch.split(torch.randperm(len(yt), generator=g), 64):
            opt.zero_grad(); lossf(net(Xt[idx]), yt[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = roc_auc_score(yv, net(Xv).numpy())
        if va > best_auc + 1e-6:
            best_auc, bad, best_ep = va, 0, ep
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
        if verbose: print(f"  ep{ep} valAUC {va:.4f}")
    net.load_state_dict(best_state); net.eval()
    return net, (mu, sd), dict(best_val_auc=float(best_auc), best_epoch=best_ep, epochs_run=ep + 1)

def rep_for_arm(arm, N, y, folds, seed):
    """Pooled-OOF AUC via S11 harness (same metric as frozen 0.7565 baseline).
    F trains its readout per OUTER fold, then the S11 harness runs on that fold."""
    Xf = N.reshape(len(N), -1)                            # [n,2880]
    if arm == "P":
        d, _ = K.probe_pipe(N.sum(1), y, folds, []); return d, {}
    if arm == "X":
        d, _ = K.probe_pipe(Xf, y, folds, []); return d, {}
    if arm == "C":
        d, _ = K.probe_pipe(Xf, y, folds,
                            [("pca", PCA(n_components=32, random_state=BASE))]); return d, {}
    if arm == "F":
        aucs, head_aucs, meta = [], [], []
        oof = np.full(len(y), np.nan)
        for fi, (tr, te) in enumerate(folds):
            net, (mu, sd), info = train_readout(Xf[tr], y[tr], seed + fi)
            Z = np.zeros((len(y), 32))
            with torch.no_grad():
                Tt = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32)
                Z[tr] = net.feats(Tt(Xf[tr])).numpy()
                Z[te] = net.feats(Tt(Xf[te])).numpy()
                head_aucs.append(float(roc_auc_score(y[te], net(Tt(Xf[te])).numpy())))
            d1, o1 = K.probe_pipe(Z, y, [(tr, te)], [])   # S11 harness, this fold only
            oof[te] = o1[te]; aucs.append(d1["fold_auc"][0]); meta.append(info)
        cov = np.isfinite(oof)
        return dict(auc=float(roc_auc_score(y[cov], oof[cov])),
                    fold_auc=aucs, fold_mean=float(np.mean(aucs))), \
               dict(head_auc=head_aucs, head_mean=float(np.mean(head_aucs)), folds=meta)
    raise ValueError(arm)
