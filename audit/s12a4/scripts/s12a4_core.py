"""S12A4 core — controlled training wave. Fixed stack: identity input [M1_B|I90],
WGIN emb32 (norm=F per S12A3), ROI-aware readout Linear(2880->32). Only the objective
varies. Arm3 step math = verbatim S8 corrected-C contract; arm4 declared diff = hard
top-k mask (keep 0.8, straight-through), budget reg removed."""
import sys, os, json, copy, hashlib, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s8"); import s8_core as S8
sys.path.insert(0, "/users/3171356m/A-GCL")
from torch_scatter import scatter
from unsupervised.encoder.tu_encoder import TUEncoder
from unsupervised.learning.ginfominmax import GInfoMinMax
from unsupervised.view_learner import (ViewLearner, compute_reverse_index,
    symmetrize_edge_logits, sample_symmetric_logistic_noise)
from agcl_ABIDE_queue import MemoryBank_Q, calc_regloss, setup_seed
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE = 20260818
S12A4 = "/users/3171356m/agcl_audit_s0/s12a4/"
LR, BATCH, EMB = 5e-4, 32, 32
REG_LAMBDA, CR_LAMBDA = 2.0, 0.4
BATCH_T, MEM_T = 0.2, 0.1            # audited corrected-C temperatures
MAX_EPOCHS, PATIENCE_CE, EVAL_EVERY_UNSUP, PATIENCE_UNSUP = 200, 20, 5, 4

class ROIModel(nn.Module):
    """Encoder (norm=F stack) + ROI-aware readout. forward -> (z, node_emb) matching
    the GInfoMinMax contract used by the audited train_step; h/z are 32-d."""
    def __init__(self, seed, with_head):
        super().__init__()
        setup_seed(seed)
        enc = TUEncoder(num_dataset_features=93, emb_dim=EMB, num_gc_layers=2,
                        drop_ratio=0.3, pooling_type="standard", is_infograph=False,
                        normalize_nodes=False, message_relu=True, post_bn_relu=True)
        self.gim = GInfoMinMax(enc, EMB)             # positional (S8 lesson); proj_head 32->32
        self.readout = nn.Linear(90 * EMB, EMB)
        self.head = nn.Linear(EMB, 1) if with_head else None
    def encode(self, batch, x, edge_index, edge_attr=None, edge_weight=None):
        _, node = self.gim.encoder(batch, x, edge_index, edge_attr, edge_weight)
        B = int(batch.max()) + 1
        h = self.readout(node.reshape(B, 90 * EMB))
        z = self.gim.proj_head(h)
        return h, z, node
    def forward(self, batch, x, edge_index, edge_attr=None, edge_weight=None):
        h, z, node = self.encode(batch, x, edge_index, edge_attr, edge_weight)
        return z, node
    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        return GInfoMinMax.calc_loss(x, x_aug, temperature=temperature, sym=sym)

def build_view(seed):
    setup_seed(seed + 777)
    enc = TUEncoder(num_dataset_features=93, emb_dim=EMB, num_gc_layers=2,
                    drop_ratio=0.3, pooling_type="standard", is_infograph=False,
                    normalize_nodes=False, message_relu=True, post_bn_relu=True)
    return ViewLearner(enc, mlp_edge_model_dim=64)

# ---------- batching (S12A1-proven layout: 90 nodes/graph, 8100 directed edges) ----------
def make_batch(Xid, FC, idxs, device):
    ei0 = A1.edge_struct()
    xs, eis, ews, bt = [], [], [], []
    for j, i in enumerate(idxs):
        xs.append(torch.from_numpy(Xid[i])); eis.append(ei0 + j * 90)
        ews.append(torch.from_numpy(FC[i].reshape(-1).astype(np.float32)))
        bt.append(torch.full((90,), j, dtype=torch.long))
    class B: pass
    b = B()
    b.x = torch.cat(xs).to(device); b.edge_index = torch.cat(eis, 1).to(device)
    b.edge_weight = torch.cat(ews).to(device); b.batch = torch.cat(bt).to(device)
    b.subject_id = torch.tensor(list(idxs), dtype=torch.long, device=device)
    return b

def per_graph_mean(vals, b, nb):
    edge_batch = b.batch[b.edge_index[0]]
    sm = scatter(vals, edge_batch, reduce="sum", dim_size=nb)
    cnt = scatter(torch.ones_like(vals), edge_batch, reduce="sum", dim_size=nb)
    return (sm / cnt.clamp(min=1)).mean()

# ---------- masks ----------
def mask_symmetric(b, logits, device):
    """Audited corrected-C mask (S8 _mask 'symmetric' branch, verbatim math)."""
    rev = compute_reverse_index(b.edge_index)
    sym = symmetrize_edge_logits(b.edge_index, logits, rev)
    mu = torch.sigmoid(sym)
    noise = sample_symmetric_logistic_noise(b.edge_index, rev, bias=1e-4, device=device)
    return mu, torch.sigmoid((noise + sym) / 1.0)

def mask_topk(b, logits, nb, keep=0.80):
    """Arm4 declared diff: per-graph hard top-k on symmetrized logits, straight-through."""
    rev = compute_reverse_index(b.edge_index)
    sym = symmetrize_edge_logits(b.edge_index, logits, rev).squeeze(-1)
    soft = torch.sigmoid(sym)
    hard = torch.zeros_like(soft)
    edge_batch = b.batch[b.edge_index[0]]
    for g in range(nb):
        m = edge_batch == g
        kg = int(round(keep * int(m.sum())))
        idx = torch.nonzero(m).squeeze(-1)
        top = idx[torch.topk(sym[m], kg).indices]
        hard[top] = 1.0
    mask = hard + soft - soft.detach()            # straight-through
    return soft, mask

# ---------- arm training steps ----------
def step_supervised(arm, model, opt, b, yb, gen, device):
    model.train(); opt.zero_grad()
    h, z, node = model.encode(b.batch, b.x, b.edge_index, None, b.edge_weight)
    logits = model.head(h).squeeze(-1)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
    parts = dict(ce=float(loss))
    if arm == 2:
        keepm = (torch.rand(b.edge_weight.shape, generator=gen, device="cpu")
                 < 0.8).float().to(device)
        z_aug, _ = model(b.batch, b.x, b.edge_index, None, b.edge_weight * keepm)
        nce = ROIModel.calc_loss(z, z_aug, temperature=BATCH_T, sym=True)
        loss = loss + 1.0 * nce; parts["infonce"] = float(nce)
    loss.backward()
    gfin = all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)
    opt.step()
    parts.update(loss=float(loss), grad_finite=bool(gfin))
    return parts

def step_agcl(arm, model, view, mopt, vopt, bank, b, device):
    """S8 corrected-C train_step, verbatim math, on the ROI stack.
    Declared diffs for arm4 ONLY: mask_topk replaces mask_symmetric; budget reg removed."""
    nb = int(b.batch.max()) + 1
    view.train(); view.zero_grad(); model.eval()
    x, _ = model(b.batch, b.x, b.edge_index, None, b.edge_weight)
    logits = view(b.batch, b.x, b.edge_index, None, b.edge_weight)
    if arm == 3: mu, mask = mask_symmetric(b, logits, device)
    else:        mu, mask = mask_topk(b, logits, nb)
    mask = mask.squeeze()
    x_aug, _ = model(b.batch, b.x, b.edge_index, None, b.edge_weight * mask)
    vm, vids = bank.get_valid_memory()
    cr = x.sum() * 0.0 if vm.size(0) == 0 else calc_regloss(
        x, x_aug, vm, vids, b.subject_id, temperature=MEM_T)
    infonce_v = ROIModel.calc_loss(x, x_aug, temperature=BATCH_T, sym=True)
    reg = per_graph_mean(mu.squeeze(), b, nb)
    view_loss = infonce_v + (REG_LAMBDA * reg if arm == 3 else 0.0) + CR_LAMBDA * cr
    (-view_loss).backward(); vopt.step()
    model.train(); view.eval(); model.zero_grad()
    x, _ = model(b.batch, b.x, b.edge_index, None, b.edge_weight)
    with torch.no_grad():
        logits2 = view(b.batch, b.x, b.edge_index, None, b.edge_weight)
        if arm == 3: _, mask2 = mask_symmetric(b, logits2, device)
        else:        _, mask2 = mask_topk(b, logits2, nb)
        mask2 = mask2.squeeze()
    x_aug, _ = model(b.batch, b.x, b.edge_index, None, b.edge_weight * mask2)
    vm, vids = bank.get_valid_memory()
    cr2 = x.sum() * 0.0 if vm.size(0) == 0 else calc_regloss(
        x, x_aug, vm, vids, b.subject_id, temperature=MEM_T)
    infonce_m = ROIModel.calc_loss(x, x_aug, temperature=BATCH_T, sym=True)
    model_loss = infonce_m + CR_LAMBDA * cr2
    model_loss.backward()
    gfin = all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)
    mopt.step()
    bank.push(x_aug.detach(), b.subject_id)
    keep_hard = float((mask2 > 0.5).float().mean())
    return dict(view_loss=float(view_loss), model_loss=float(model_loss),
                infonce=float(infonce_m), cr=float(cr2), reg=float(reg),
                keep_mu=float(mu.mean()), keep_sampled=float(mask2.mean()),
                keep_hard=keep_hard, grad_finite=bool(gfin), loss=float(model_loss))

# ---------- feature extraction ----------
@torch.no_grad()
def extract_all(model, Xid, FC, device, bs=64):
    model.eval(); H, Z, ND = [], [], []
    for lo in range(0, len(Xid), bs):
        idxs = range(lo, min(lo + bs, len(Xid)))
        b = make_batch(Xid, FC, idxs, device)
        h, z, node = model.encode(b.batch, b.x, b.edge_index, None, b.edge_weight)
        H.append(h.cpu().numpy()); Z.append(z.cpu().numpy())
        ND.append(node.cpu().numpy().reshape(len(list(idxs)), -1))
    return np.concatenate(H), np.concatenate(Z), np.concatenate(ND)

def quick_val_auc(Htr, ytr, Hva, yva):
    sc = StandardScaler().fit(Htr)
    svm = LinearSVC(C=1.0, dual="auto", max_iter=20000, random_state=BASE)
    svm.fit(sc.transform(Htr), ytr)
    return float(roc_auc_score(yva, svm.decision_function(sc.transform(Hva))))

# ---------- one fold, one arm ----------
def train_fold(arm, seed, tr, Xid, FC, y, device, max_epochs=MAX_EPOCHS, log=None):
    """Returns (best_model_state, curve, info). tr = outer-train indices."""
    itr, iva = train_test_split(np.arange(len(tr)), test_size=0.2,
                                stratify=y[tr], random_state=BASE)
    itr, iva = tr[itr], tr[iva]
    torch.manual_seed(seed); np.random.seed(seed % 2**32)
    model = ROIModel(seed, with_head=(arm in (1, 2))).to(device)
    gen = torch.Generator().manual_seed(seed)
    curve = []; best = (-1.0, None, 0); bad = 0
    train_idx = itr if arm in (1, 2) else tr          # unsup trains on outer-train
    if arm in (3, 4):
        view = build_view(seed).to(device)
        mopt = torch.optim.Adam(model.parameters(), lr=LR)
        vopt = torch.optim.Adam(view.parameters(), lr=LR)
        bank = MemoryBank_Q(256, EMB, device)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, max_epochs + 1):
        perm = torch.randperm(len(train_idx), generator=gen).numpy()
        stats = []
        for lo in range(0, len(perm) - BATCH + 1, BATCH):       # drop_last=True
            bidx = train_idx[perm[lo:lo + BATCH]]
            b = make_batch(Xid, FC, bidx, device)
            if arm in (1, 2):
                yb = torch.tensor(y[bidx], dtype=torch.float32, device=device)
                stats.append(step_supervised(arm, model, opt, b, yb, gen, device))
            else:
                stats.append(step_agcl(arm, model, view, mopt, vopt, bank, b, device))
        agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]
               if k != "grad_finite"}
        agg["grad_finite"] = bool(all(s["grad_finite"] for s in stats))
        agg["epoch"] = ep
        # validation AUC (NEVER accuracy) --------------------------------------
        do_eval = (arm in (1, 2)) or (ep % EVAL_EVERY_UNSUP == 0)
        if do_eval:
            H, Z, ND = extract_all(model, Xid, FC, device)
            if arm in (1, 2):
                with torch.no_grad():
                    hv = torch.from_numpy(H[iva]).to(device)
                    sv = model.head(hv).squeeze(-1).cpu().numpy()
                    ht = torch.from_numpy(H[itr]).to(device)
                    st = model.head(ht).squeeze(-1).cpu().numpy()
                va = float(roc_auc_score(y[iva], sv))
                ta = float(roc_auc_score(y[itr], st))
                vacc = float(((sv > 0).astype(int) == y[iva]).mean())
            else:
                va = quick_val_auc(H[itr], y[itr], H[iva], y[iva])
                ta = quick_val_auc(H[itr], y[itr], H[itr], y[itr])
                vacc = float("nan")
            agg.update(val_auc=va, train_auc=ta, val_acc=vacc)
            patience = PATIENCE_CE if arm in (1, 2) else PATIENCE_UNSUP
            if va > best[0] + 1e-6:
                best = (va, copy.deepcopy({k: v.cpu() for k, v in
                                           model.state_dict().items()}), ep)
                bad = 0
            else:
                bad += 1
                if bad >= patience: curve.append(agg); break
        curve.append(agg)
        if log and ep % 10 == 0:
            print(f"[{log}] ep{ep} loss {agg.get('loss', 0):.4f} "
                  f"val {agg.get('val_auc', float('nan')):.4f}", flush=True)
    model.load_state_dict(best[1]); model.eval()
    info = dict(best_val_auc=best[0], best_epoch=best[2], epochs_run=ep,
                itr=len(itr), iva=len(iva),
                keep_rates=[c.get("keep_hard") for c in curve
                            if "keep_hard" in c] if arm == 4 else None)
    return model, curve, info
