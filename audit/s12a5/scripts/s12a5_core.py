"""S12A5 core. Arms A/B/C (supervised CE + weight decay) and Wave-2 transductive A-GCL.
Reuses s12a4_core for the audited ROI stack and corrected-C step math."""
import sys, os, json, copy, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a4/scripts"); import s12a4_core as M4
sys.path.insert(0, "/users/3171356m/A-GCL")
from agcl_ABIDE_queue import MemoryBank_Q, setup_seed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE = 20260818
S12A5 = "/users/3171356m/agcl_audit_s0/s12a5/"
LR, BATCH, WD = 5e-4, 32, 1e-4
MAX_EPOCHS, PATIENCE = 200, 20

class EdgeMLP(nn.Module):
    """Edge branch: 4005 FC triangle -> 32-d embedding."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4005, 256), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(256, 32))
    def forward(self, e): return self.net(e)

class ArmModel(nn.Module):
    """arm 'A': WGIN ROI stack; 'B': WGIN + edge skip concat; 'C': edge MLP only.
    forward(batch_graph_or_None, edge_vec_or_None) -> (h_repr, logits)."""
    def __init__(self, arm, seed):
        super().__init__()
        setup_seed(seed)
        self.arm = arm
        if arm in ("A", "B"):
            self.roi = M4.ROIModel(seed, with_head=False)   # encoder + readout (h 32)
        if arm in ("B", "C"):
            self.edge = EdgeMLP()
        dim = dict(A=32, B=64, C=32)[arm]
        self.head = nn.Linear(dim, 1)
    def repr_of(self, b, ev):
        parts = []
        if self.arm in ("A", "B"):
            h, z, node = self.roi.encode(b.batch, b.x, b.edge_index, None, b.edge_weight)
            parts.append(h); self._node = node
        if self.arm in ("B", "C"):
            parts.append(self.edge(ev))
        return torch.cat(parts, 1)
    def forward(self, b, ev):
        r = self.repr_of(b, ev)
        return r, self.head(r).squeeze(-1)

def grad_group_norms(model):
    g = {"encoder": 0.0, "classifier": 0.0}
    for k, p in model.named_parameters():
        if p.grad is None: continue
        n2 = float(p.grad.norm()) ** 2
        if k.startswith("roi.gim.encoder."): g["encoder"] += n2
        elif k.startswith("head.") or k.startswith("roi.readout.") or k.startswith("edge."):
            g["classifier"] += n2
    return {k: v ** 0.5 for k, v in g.items()}

@torch.no_grad()
def extract5(model, Xid, FC, Xe, idxs, device, bs=64):
    model.eval(); R, S = [], []
    idxs = np.asarray(idxs)
    for lo in range(0, len(idxs), bs):
        ii = idxs[lo:lo + bs]
        b = M4.make_batch(Xid, FC, ii, device) if model.arm in ("A", "B") else None
        ev = torch.tensor(Xe[ii], dtype=torch.float32, device=device) \
            if model.arm in ("B", "C") else None
        r, lg = model(b, ev)
        R.append(r.cpu().numpy()); S.append(lg.cpu().numpy())
    return np.concatenate(R), np.concatenate(S)

def train_fold5(arm, seed, tr, Xid, FC, Xe, y, device, max_epochs=MAX_EPOCHS,
                log=None, perm_roi=None):
    """CE + weight decay. Per-epoch: train/val loss+AUC, grad norms. ES on val AUC.
    perm_roi: optional [954,90] per-subject ROI permutations (winner control)."""
    if perm_roi is not None:
        Xid = np.stack([Xid[i][perm_roi[i]] for i in range(len(Xid))])
        FC = np.stack([FC[i][perm_roi[i]][:, perm_roi[i]] for i in range(len(FC))])
        Xe = np.stack([FC[i][K.IU[0], K.IU[1]] for i in range(len(FC))])
    itr, iva = train_test_split(np.arange(len(tr)), test_size=0.2,
                                stratify=y[tr], random_state=BASE)
    itr, iva = np.asarray(tr)[itr], np.asarray(tr)[iva]
    torch.manual_seed(seed); np.random.seed(seed % 2**32)
    model = ArmModel(arm, seed).to(device)
    init_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(seed)
    lossf = nn.BCEWithLogitsLoss()
    curve = []; best = (-1.0, None, 0); bad = 0
    # epoch-0 embeddings (all 954)
    R0, S0 = extract5(model, Xid, FC, Xe, np.arange(954), device)
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(itr), generator=gen).numpy()
        tl, gns = [], []
        for lo in range(0, len(perm) - BATCH + 1, BATCH):
            ii = itr[perm[lo:lo + BATCH]]
            b = M4.make_batch(Xid, FC, ii, device) if arm in ("A", "B") else None
            ev = torch.tensor(Xe[ii], dtype=torch.float32, device=device) \
                if arm in ("B", "C") else None
            opt.zero_grad()
            _, lg = model(b, ev)
            loss = lossf(lg, torch.tensor(y[ii], dtype=torch.float32, device=device))
            loss.backward()
            gn = grad_group_norms(model)
            assert all(np.isfinite(v) for v in gn.values()), "non-finite grad"
            gns.append(gn); tl.append(float(loss)); opt.step()
        both = np.concatenate([itr, iva])
        Rb, Sb = extract5(model, Xid, FC, Xe, both, device)
        st, sv = Sb[:len(itr)], Sb[len(itr):]
        with torch.no_grad():
            vloss = float(lossf(torch.tensor(sv), torch.tensor(y[iva], dtype=torch.float32)))
        agg = dict(epoch=ep, train_loss=float(np.mean(tl)), val_loss=vloss,
                   train_auc=float(roc_auc_score(y[itr], st)),
                   val_auc=float(roc_auc_score(y[iva], sv)),
                   grad_encoder=float(np.mean([g["encoder"] for g in gns])),
                   grad_classifier=float(np.mean([g["classifier"] for g in gns])))
        curve.append(agg)
        if agg["val_auc"] > best[0] + 1e-6:
            best = (agg["val_auc"],
                    copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}), ep)
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE: break
        if log and ep % 20 == 0:
            print(f"[{log}] ep{ep} vl {agg['val_loss']:.4f} va {agg['val_auc']:.4f}", flush=True)
    model.load_state_dict(best[1]); model.eval()
    info = dict(best_val_auc=best[0], best_epoch=best[2], epochs_run=ep)
    return model, curve, info, init_state, (R0, S0)

# ---------- Wave 2: transductive A-GCL (corrected-C contract, S12A4 step math) ----------
def train_transductive(seed, Xid, FC, device, epochs=200, log=None):
    torch.manual_seed(seed); np.random.seed(seed % 2**32)
    model = M4.ROIModel(seed, with_head=False).to(device)
    view = M4.build_view(seed).to(device)
    mopt = torch.optim.Adam(model.parameters(), lr=M4.LR)
    vopt = torch.optim.Adam(view.parameters(), lr=M4.LR)
    bank = MemoryBank_Q(256, 32, device)
    gen = torch.Generator().manual_seed(seed)
    allidx = np.arange(954); curves = []
    for ep in range(1, epochs + 1):
        perm = torch.randperm(954, generator=gen).numpy()
        st = []
        for lo in range(0, 954 - M4.BATCH + 1, M4.BATCH):
            b = M4.make_batch(Xid, FC, allidx[perm[lo:lo + M4.BATCH]], device)
            s = M4.step_agcl(3, model, view, mopt, vopt, bank, b, device)
            # model-half grads still populated after step_agcl (review fix: log norms)
            s["grad_encoder"] = float(sum(float(p.grad.norm()) ** 2
                for k, p in model.named_parameters()
                if p.grad is not None and k.startswith("gim.encoder.")) ** 0.5)
            s["grad_readout"] = float(sum(float(p.grad.norm()) ** 2
                for k, p in model.named_parameters()
                if p.grad is not None and k.startswith("readout.")) ** 0.5)
            st.append(s)
        agg = {k: float(np.mean([s[k] for s in st])) for k in st[0] if k != "grad_finite"}
        agg["grad_finite"] = bool(all(s["grad_finite"] for s in st)); agg["epoch"] = ep
        assert agg["grad_finite"], f"non-finite grads ep{ep}"
        curves.append(agg)
        if log and ep % 20 == 0:
            print(f"[{log}] ep{ep} model {agg['model_loss']:.4f} keep_mu "
                  f"{agg['keep_mu']:.4f}", flush=True)
    return model, view, curves
