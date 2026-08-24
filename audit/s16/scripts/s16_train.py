"""S15 repaired training loop. Every element targets a MEASURED S13 defect:
warmup (val AUC peaked at epoch 1), min_epochs (same), adaptive clipping (92% of
steps clipped at a hardcoded 1.0), batch 32 + 400 epochs (~500 steps -> ~4000),
lr {3e-4,1e-3} (1e-4 could not move the model), dropout 0.10 (0.30 was crushing),
EMA, and a pairwise-AUC loss option.
J1: NO training-dynamics observation may raise. All become recorded flags."""
import sys, os, math, time, copy, json, collections, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/users/3171356m/A-GCL/audit/s16/scripts")
import s16_models as MO
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE = 20260818
BATCH, MAX_EPOCHS, MIN_EPOCHS, PATIENCE, MIN_DELTA = 32, 400, 80, 50, 1e-5
WARMUP_FRAC, COSINE_FLOOR = 0.10, 0.05
LABEL_SMOOTH, EMA_DECAY = 0.05, 0.999
CLIP_WARMUP_STEPS, CLIP_WINDOW, CLIP_PCTL = 50, 200, 90
MOVEMENT_UNTRAINED = 0.10
GROUPS = {"BNT": {"inp": ("inp.",), "enc": ("blocks.", "norm_f."), "head": ("head.",)},
          "WGIN": {"inp": ("inp.",), "enc": ("convs.", "norms."), "head": ("head.",)},
          # EDGEMLP (A7) = s12a5_core EdgeMLP + ArmModel(arm='C'):
          #   net.0 Linear(4005,256) | net.1 ReLU | net.2 Dropout(0.3) | net.3 Linear(256,32)
          #   head  Linear(32,1)
          # Groups are NON-OVERLAPPING and exhaustive over trainable parameters;
          # net.1/net.2 hold none. Schema matches BNT/WGIN so the result columns
          # movement_inp / movement_enc / movement_head are populated identically.
          "EDGEMLP": {"inp": ("net.0.",), "enc": ("net.3.",), "head": ("head.",)}}

def assert_groups_cover(model, arch):
    """Every trainable parameter belongs to exactly ONE group. Raises on a gap or
    an overlap, so a missing/duplicated instrumentation key cannot pass silently."""
    pref = GROUPS[arch]
    owners = {}
    for k, p_ in model.named_parameters():
        if not p_.requires_grad: continue
        hit = [g for g, ps in pref.items() if any(k.startswith(x) for x in ps)]
        if len(hit) != 1:
            raise AssertionError(f"{arch}: parameter {k} maps to {len(hit)} groups {hit}")
        owners[k] = hit[0]
    for g in pref:
        if not any(v == g for v in owners.values()):
            raise AssertionError(f"{arch}: group '{g}' is empty")
    return owners

# ------------------------------------------------------------------ losses
def loss_bce(logits, y, smooth=LABEL_SMOOTH):
    t = y * (1 - smooth) + smooth / 2                 # y*0.95 + 0.025 at 0.05
    return nn.functional.binary_cross_entropy_with_logits(logits, t)

def loss_auc(logits, y):
    """Pairwise ranking surrogate: mean softplus(-(s_pos - s_neg)) over all pairs.
    You MEASURE AUC; this OPTIMISES it. Falls back to BCE if a batch is one-class."""
    pos, neg = logits[y > 0.5], logits[y <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0: return loss_bce(logits, y)
    return nn.functional.softplus(-(pos[:, None] - neg[None, :])).mean()

LOSSES = {"L-BCE": loss_bce, "L-AUC": loss_auc}

# ------------------------------------------------------------------ helpers
def lr_at(step, total, base_lr):
    """Linear warmup over the first 10% of steps, then cosine decay to 0.05*lr."""
    w = max(1, int(WARMUP_FRAC * total))
    if step < w: return base_lr * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return base_lr * (COSINE_FLOOR + (1 - COSINE_FLOOR) * 0.5 * (1 + math.cos(math.pi * p)))

def group_grad_norms(model, arch):
    out = {}
    for g, pref in GROUPS[arch].items():
        s = 0.0
        for k, p in model.named_parameters():
            if p.grad is not None and any(k.startswith(x) for x in pref):
                s += float(p.grad.norm()) ** 2
        out[g] = s ** 0.5
    return out

def movement(init, model, arch):
    out = {}
    for g, pref in GROUPS[arch].items():
        di, df = [], []
        for k, v in model.state_dict().items():
            if not any(k.startswith(x) for x in pref): continue
            if not torch.is_floating_point(v) or k.endswith(".E"): continue
            di.append(init[k].float().flatten()); df.append(v.cpu().float().flatten())
        if di:
            i0, f0 = torch.cat(di), torch.cat(df)
            out[g] = float((f0 - i0).norm() / i0.norm().clamp(min=1e-12))
    return out

@torch.no_grad()
def extract(model, X, FC, idxs, need_graph, bs=128, sparse=False):
    model.eval(); R, S = [], []
    idxs = np.asarray(idxs)
    for lo in range(0, len(idxs), bs):
        b = MO.make_batch(X, FC, idxs[lo:lo + bs], need_graph, sparse)
        r, lg = model(b, None)
        R.append(r.numpy()); S.append(lg.numpy())
    return np.concatenate(R), np.concatenate(S)

# ------------------------------------------------------------------ train
def train_fold(arch, X, FC, y, tr, cfg, seed, log=None, sparse=False):
    """cfg: dict(K_or_hidden, lr, wd, loss, freeze_encoder, readout, scaled_softmax,
    dropout, H, max_epochs, min_epochs). Returns (model, ema_sd, curve, info)."""
    need_graph = (arch == "WGIN")
    itr, iva = train_test_split(np.arange(len(tr)), test_size=0.20,
                                stratify=y[tr], random_state=BASE)
    itr, iva = np.asarray(tr)[itr], np.asarray(tr)[iva]
    torch.manual_seed(seed); np.random.seed(seed % 2**32)
    model = MO.build_model(arch, X.shape[-1], seed, cfg["K_or_hidden"],
                           freeze_encoder=cfg.get("freeze_encoder", False),
                           readout=cfg.get("readout", "roi"),
                           scaled_softmax=cfg.get("scaled_softmax", True),
                           p=cfg.get("dropout", 0.10), H=cfg.get("H", 128))
    init = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=cfg["wd"])
    ema = MO.EMA(model, EMA_DECAY)
    lossf = LOSSES[cfg["loss"]]
    gen = torch.Generator().manual_seed(seed)
    yt = torch.tensor(y, dtype=torch.float32)
    max_ep = cfg.get("max_epochs", MAX_EPOCHS); min_ep = cfg.get("min_epochs", MIN_EPOCHS)
    steps_per_epoch = max(1, math.ceil(len(itr) / BATCH))
    total_steps = steps_per_epoch * max_ep
    gnorms = collections.deque(maxlen=CLIP_WINDOW)
    curve, best, bad, step = [], (-1.0, None, None, 0), 0, 0
    n_clipped, n_steps = 0, 0
    for ep in range(1, max_ep + 1):
        t_ep = time.time(); model.train()
        perm = torch.randperm(len(itr), generator=gen).numpy()
        tl, gg, ep_clip, ep_steps, thr_last = [], [], 0, 0, float("nan")
        for lo in range(0, len(perm), BATCH):
            ii = itr[perm[lo:lo + BATCH]]
            if len(ii) < 2: continue
            for pg in opt.param_groups: pg["lr"] = lr_at(step, total_steps, cfg["lr"])
            opt.zero_grad()
            _, lg = model(MO.make_batch(X, FC, ii, need_graph, sparse), None)
            loss = lossf(lg, yt[ii])
            loss.backward()
            gg.append(group_grad_norms(model, arch))
            raw = float(torch.nn.utils.clip_grad_norm_(params, float("inf")))  # measure
            gnorms.append(raw)
            # ADAPTIVE CLIP: steps 1-50 record only; then threshold = p90 of last 200
            if step >= CLIP_WARMUP_STEPS and len(gnorms) >= 10:
                thr = float(np.percentile(gnorms, CLIP_PCTL)); thr_last = thr
                if raw > thr:
                    torch.nn.utils.clip_grad_norm_(params, thr)
                    ep_clip += 1; n_clipped += 1
            opt.step(); ema.update(model)
            tl.append(float(loss)); step += 1; ep_steps += 1; n_steps += 1
        model.eval()
        with torch.no_grad():
            _, st = extract(model, X, FC, itr, need_graph, sparse=sparse)
            _, sv = extract(model, X, FC, iva, need_graph, sparse=sparse)
            vloss = float(lossf(torch.tensor(sv), yt[iva]))
        row = dict(epoch=ep, lr=float(opt.param_groups[0]["lr"]),
                   train_loss=float(np.mean(tl)) if tl else float("nan"),
                   val_loss=vloss,
                   train_auc=float(roc_auc_score(y[itr], st)),
                   val_auc=float(roc_auc_score(y[iva], sv)),
                   grad_inp=float(np.mean([g["inp"] for g in gg])) if gg else 0.0,
                   grad_enc=float(np.mean([g["enc"] for g in gg])) if gg else 0.0,
                   grad_head=float(np.mean([g["head"] for g in gg])) if gg else 0.0,
                   adaptive_clip_threshold=thr_last, clip_events=int(ep_clip),
                   clip_rate=float(ep_clip / max(1, ep_steps)),
                   steps=int(ep_steps), epoch_s=round(time.time() - t_ep, 2))
        curve.append(row)
        if row["val_auc"] > best[0] + MIN_DELTA:
            best = (row["val_auc"],
                    copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                    copy.deepcopy(ema.state_dict(model)), ep)
            bad = 0
        else:
            bad += 1
        # EARLY STOPPING CANNOT FIRE BEFORE min_epochs (kills epoch-1 selection)
        if ep >= min_ep and bad >= PATIENCE: break
        if log and ep % 40 == 0:
            print(f"[{log}] ep{ep} vl {vloss:.4f} va {row['val_auc']:.4f} "
                  f"clip {row['clip_rate']:.2f} lr {row['lr']:.2e}", flush=True)
    raw_sd, ema_sd, best_ep = best[1], best[2], best[3]
    model.load_state_dict(raw_sd); model.eval()
    bl = curve[best_ep - 1]
    gap = bl["train_auc"] - bl["val_auc"]
    mv = movement(init, model, arch)
    mv_max = max(mv.values()) if mv else 0.0
    clip_rate = n_clipped / max(1, n_steps)
    # J1: every observation below is a RECORDED FLAG, never an exception
    info = dict(best_val_auc=best[0], best_epoch=best_ep, epochs_run=len(curve),
                total_steps=n_steps, train_val_gap=float(gap),
                verdict=("OVERFIT" if gap > 0.15 else
                         "UNDERFIT" if bl["train_auc"] < 0.65 else "HEALTHY"),
                movement=mv, movement_max=float(mv_max),
                flag_untrained=bool(mv_max < MOVEMENT_UNTRAINED),
                clip_rate=float(clip_rate),
                flag_clip_high=bool(clip_rate > 0.30),
                adaptive_clip_final=float(curve[-1]["adaptive_clip_threshold"]),
                flag_best_epoch_1=bool(best_ep == 1),
                integrity_loss_decreased=("n/a" if best_ep == 1 else
                    ("pass" if bl["train_loss"] < curve[0]["train_loss"] else "fail")),
                min_train_loss=float(min(c["train_loss"] for c in curve)),
                ocread_entropy=float(getattr(model, "_last_entropy", float("nan"))),
                n_params=MO.n_trainable(model), repr_dim=int(model.repr_dim),
                n_train=len(itr), n_val=len(iva))
    return model, ema_sd, curve, info
