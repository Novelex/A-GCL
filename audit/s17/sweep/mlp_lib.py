"""Stage B/C MLP + trainer. The OPTIMISER RECIPE is S16's frozen PROD policy
reproduced field-for-field (AdamW betas/eps, linear warmup 0.10 then cosine to
floor 0.05, label smoothing 0.05, batch 32, adaptive clip = p90 of the last 200
grad norms after 50 warm-up steps, early stopping min_epochs 80 / patience 50 /
max_epochs 400 on validation AUC of a stratified 20% split of the TRAINING block).
Only lr and wd are swept (the task's grid); everything else is the policy.
movement / clip_rate / best_epoch are computed exactly as s16_train does.
Single-threaded torch: several trainings run concurrently per 4-CPU task."""
import math, copy, collections, time
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys; sys.path.insert(0, "/users/3171356m/A-GCL/audit/s16/scripts")
import s16_policy as PL
POL = PL.PROD                                  # read-only; hash 798ed7790c1ddabc
BASE = 20260818

class SweepMLP(nn.Module):
    """Linear(D,W) GELU Dropout [Linear(W,W) GELU Dropout]{depth-1} Linear(W,32) -> head Linear(32,1).
    Representation width fixed at 32 (the probe width), everything else swept."""
    def __init__(self, D, width, depth, dropout, seed):
        super().__init__(); torch.manual_seed(seed); np.random.seed(seed % 2**32)
        layers = [nn.Linear(D, width), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(depth - 1): layers += [nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout)]
        self.enc = nn.Sequential(*layers, nn.Linear(width, 32)); self.repr_dim = 32
        self.head = nn.Linear(32, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, X): r = self.enc(X); return r, self.head(r).squeeze(-1)

GROUPS = {"inp": ("enc.0.",), "enc": ("enc.",), "head": ("head.",)}   # enc.0 is inside enc; reported separately
def _movement(init, model):
    out = {}
    for g, pref in (("inp", ("enc.0.",)), ("enc", ("enc.",)), ("head", ("head.",))):
        di, df = [], []
        for k, v in model.state_dict().items():
            if any(k.startswith(x) for x in pref) and torch.is_floating_point(v):
                di.append(init[k].flatten()); df.append(v.flatten())
        i0, f0 = torch.cat(di), torch.cat(df); out[g] = float((f0 - i0).norm() / i0.norm().clamp(min=1e-12))
    return out

def lr_at(step, total, base):
    w = max(1, int(POL.warmup_frac * total))
    if step < w: return base * (step + 1) / w
    p = (step - w) / max(1, total - w); cf = POL.cosine_floor
    return base * (cf + (1 - cf) * 0.5 * (1 + math.cos(math.pi * p)))

def loss_bce(lg, y, s=POL.label_smooth):
    return nn.functional.binary_cross_entropy_with_logits(lg, y * (1 - s) + s / 2)

def train_mlp(X, y, tr, cfg, seed=BASE, threads=1):
    """X: (954, D) float32 features ALREADY fitted/scaled on the caller's training block.
    Trains on tr with a stratified 80/20 val split for early stopping (as s16_train).
    Returns (model, info) with movement_max / clip_rate / best_epoch / valid."""
    torch.set_num_threads(threads)
    itr, iva = train_test_split(np.arange(len(tr)), test_size=0.20, stratify=y[tr], random_state=BASE)
    itr, iva = np.asarray(tr)[itr], np.asarray(tr)[iva]
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    model = SweepMLP(X.shape[1], cfg["width"], cfg["depth"], cfg["dropout"], seed)
    init = {k: v.clone() for k, v in model.state_dict().items()}
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg["wd"])
    gen = torch.Generator().manual_seed(seed)
    spe = max(1, math.ceil(len(itr) / POL.batch)); total = spe * POL.max_epochs
    gnorms = collections.deque(maxlen=POL.clip_window)
    best, bad, step, n_clip, n_steps, curve = (-1.0, None, 0), 0, 0, 0, 0, []
    t0 = time.time()
    for ep in range(1, POL.max_epochs + 1):
        model.train(); perm = torch.randperm(len(itr), generator=gen).numpy(); ep_clip = ep_steps = 0
        for lo in range(0, len(perm), POL.batch):
            ii = itr[perm[lo:lo + POL.batch]]
            if len(ii) < 2: continue
            for pg in opt.param_groups: pg["lr"] = lr_at(step, total, cfg["lr"])
            opt.zero_grad(); _, lg = model(Xt[ii]); loss = loss_bce(lg, yt[ii]); loss.backward()
            raw = float(torch.nn.utils.clip_grad_norm_(params, float("inf"))); gnorms.append(raw)
            if step >= POL.clip_warmup_steps and len(gnorms) >= 10:
                thr = float(np.percentile(gnorms, POL.clip_pctl))
                if raw > thr: torch.nn.utils.clip_grad_norm_(params, thr); ep_clip += 1; n_clip += 1
            opt.step(); step += 1; ep_steps += 1; n_steps += 1
        model.eval()
        with torch.no_grad(): _, sv = model(Xt[iva]); _, st = model(Xt[itr])
        va = float(roc_auc_score(y[iva], sv.numpy())); ta = float(roc_auc_score(y[itr], st.numpy()))
        curve.append((ep, ta, va, ep_clip / max(1, ep_steps)))
        if va > best[0] + POL.min_delta:
            best = (va, copy.deepcopy(model.state_dict()), ep); bad = 0
        else: bad += 1
        if ep >= POL.min_epochs and bad >= POL.patience: break
    model.load_state_dict(best[1]); model.eval()
    mv = _movement(init, model); clip_rate = n_clip / max(1, n_steps)
    info = dict(best_val_auc=best[0], best_epoch=best[2], epochs_run=len(curve), total_steps=n_steps,
                movement=mv, movement_max=float(max(mv.values())), clip_rate=float(clip_rate),
                valid=bool(max(mv.values()) > 0.10 and clip_rate < 0.30),
                train_auc_at_best=curve[best[2] - 1][1], secs=round(time.time() - t0, 1),
                n_params=int(sum(p.numel() for p in params)), policy_hash=POL.policy_hash())
    return model, info

@torch.no_grad()
def scores(model, X):
    model.eval(); _, lg = model(torch.tensor(X, dtype=torch.float32)); return lg.numpy().astype(np.float64)

def grid_B():
    G = []
    for width in (16, 32, 64, 128, 256, 512):
        for dropout in (0.0, 0.1, 0.2, 0.3, 0.5):
            for depth in (1, 2):
                for lr in (1e-4, 3e-4, 1e-3):
                    for wd in (1e-4, 1e-3, 1e-2):
                        G.append(dict(width=width, dropout=dropout, depth=depth, lr=lr, wd=wd))
    return G                                                   # 540
def cfg_id(c): return f"w{c['width']}_d{c['depth']}_p{c['dropout']}_lr{c['lr']:g}_wd{c['wd']:g}"
