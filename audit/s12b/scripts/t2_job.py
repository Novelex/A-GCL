"""S12B Track 2 — FC edge-model ceiling. One task = (hidden, weight_decay, seed).
AdamW (decoupled), per-epoch diagnostics, pre-registered verdict rule, integrity
asserts. Frozen ordinary + LOSO folds. Separate question from Track 1."""
import sys, os, json, time, copy, socket, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12b/scripts"); import s12b_core as B
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HID = [64, 256, 512]; WD = [1e-5, 1e-4, 1e-3, 1e-2]; NSEED = 3
TASKS = [(h, w, s) for h in HID for w in WD for s in range(NSEED)]   # 36
LR, BATCH, MAXEP, PAT = 5e-4, 32, 200, 20

class EdgeNet(nn.Module):
    def __init__(self, h, seed):
        super().__init__(); torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(4005, h), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(h, 32))
        self.head = nn.Linear(32, 1)
    def forward(self, e): return self.head(self.net(e)).squeeze(-1)

def gnorm(model, pref):
    s = 0.0
    for k, p in model.named_parameters():
        if p.grad is not None and k.startswith(pref): s += float(p.grad.norm()) ** 2
    return s ** 0.5

def movement(init, model):
    out = {}
    for grp in ("net", "head"):
        di, df = [], []
        for k, v in model.state_dict().items():
            if k.startswith(grp) and torch.is_floating_point(v):
                di.append(init[k].float().flatten()); df.append(v.cpu().float().flatten())
        i0, f0 = torch.cat(di), torch.cat(df)
        out[grp] = float((f0 - i0).norm() / i0.norm().clamp(min=1e-12))
    return out

def train_fold(h, wd, seed, tr, Xe, y, dev, tag):
    itr, iva = train_test_split(np.arange(len(tr)), test_size=0.2,
                                stratify=y[tr], random_state=B.BASE)
    itr, iva = np.asarray(tr)[itr], np.asarray(tr)[iva]
    torch.manual_seed(seed); np.random.seed(seed % 2**32)
    model = EdgeNet(h, seed).to(dev)
    init = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(); gen = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(Xe, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    curve = []; best = (-1.0, None, 0); bad = 0
    for ep in range(1, MAXEP + 1):
        model.train(); perm = torch.randperm(len(itr), generator=gen).numpy()
        tl, gn, gh = [], [], []
        for lo in range(0, len(perm) - BATCH + 1, BATCH):
            ii = itr[perm[lo:lo + BATCH]]
            opt.zero_grad()
            loss = lossf(model(Xt[ii]), yt[ii]); loss.backward()
            assert torch.isfinite(loss), "non-finite loss"
            gn.append(gnorm(model, "net")); gh.append(gnorm(model, "head"))
            assert np.isfinite(gn[-1]) and np.isfinite(gh[-1]), "non-finite grad"
            tl.append(float(loss)); opt.step()
        model.eval()
        with torch.no_grad():
            st = model(Xt[itr]).cpu().numpy(); sv = model(Xt[iva]).cpu().numpy()
            vloss = float(lossf(torch.tensor(sv), torch.tensor(y[iva], dtype=torch.float32)))
        mv = movement(init, model)                    # per-epoch, per-group (review R9)
        row = dict(epoch=ep, lr=LR, train_loss=float(np.mean(tl)), val_loss=vloss,
                   train_auc=float(roc_auc_score(y[itr], st)),
                   val_auc=float(roc_auc_score(y[iva], sv)),
                   grad_net=float(np.mean(gn)), grad_head=float(np.mean(gh)),
                   move_net=mv["net"], move_head=mv["head"])
        curve.append(row)
        if row["val_auc"] > best[0] + 1e-6:
            best = (row["val_auc"],
                    copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}), ep)
            bad = 0
        else:
            bad += 1
            if bad >= PAT: break
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        sv2 = model(Xt[iva]).cpu().numpy()
        s_all = model(Xt).cpu().numpy()
    va2 = float(roc_auc_score(y[iva], sv2))
    assert va2 == best[0], f"ckpt reload mismatch {va2} != {best[0]}"   # bitwise
    bl = curve[best[2] - 1]
    gap = bl["train_auc"] - bl["val_auc"]
    verdict = ("OVERFIT" if gap > 0.10 else
               "UNDERFIT" if bl["train_auc"] < 0.70 else "HEALTHY")
    integ = dict(loss_finite=True, grads_finite=True,
                 loss_decreased=bool(best[2] == 1 or
                                     curve[best[2] - 1]["train_loss"] < curve[0]["train_loss"]),
                 selection_on_val_only=True, ckpt_reload_bitwise=True)
    assert integ["loss_decreased"], "loss did not decrease to best epoch"
    info = dict(best_val_auc=best[0], best_epoch=best[2], epochs_run=len(curve),
                gap_at_best=float(gap), verdict=verdict, integrity=integ,
                movement=movement(init, model))
    return s_all, curve, info

def main(task):
    h, wd, sidx = TASKS[task]; seed = B.BASE + sidx
    tag = f"h{h}_wd{wd:g}_s{sidx}"
    jp = f"{B.S12B}jobs/t2_{tag}.json"
    if os.path.exists(jp): print("skip", tag); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = B.load_all(); y = d["y"]; Xe = d["X_fc"].astype(np.float32)
    folds = B.folds_all(y)
    t0 = time.time(); oof_o = np.full(954, np.nan); oof_l = np.full(954, np.nan)
    per_fold, diag_rows = {}, []
    for t, tr, te in folds:
        s_all, curve, info = train_fold(h, wd, seed, tr, Xe, y, dev, f"{tag}/{t}")
        p = 1 / (1 + np.exp(-s_all))
        (oof_o if t.startswith("o") else oof_l)[te] = p[te]
        per_fold[t] = dict(**info, fold_metrics=B.metric_block(y[te], p[te]))
        for r in curve: diag_rows.append(dict(unit=tag, fold=t, **r))
        print(f"{tag} {t} best_val {info['best_val_auc']:.4f} ep{info['best_epoch']}"
              f" {info['verdict']}", flush=True)
    co, cl = np.isfinite(oof_o), np.isfinite(oof_l)
    res = dict(unit=tag, hidden=h, weight_decay=wd, seed=seed,
               optimizer="AdamW(decoupled)", lr=LR, batch=BATCH,
               n_params=int(sum(p.numel() for p in EdgeNet(h, seed).parameters())),
               pooled_ordinary=B.metric_block(y[co], oof_o[co]),
               pooled_loso=B.metric_block(y[cl], oof_l[cl]),
               per_fold=per_fold,
               verdicts=[v["verdict"] for v in per_fold.values()],
               wall_s=round(time.time() - t0, 1), provenance=B.provenance())
    import csv
    dp = f"{B.S12B}jobs/t2_{tag}_diag.csv"
    with open(dp + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        w.writeheader(); w.writerows(diag_rows)
    os.replace(dp + ".tmp", dp)
    B.atomic_json(res, jp)
    print("DONE", tag, "ord", res["pooled_ordinary"]["auc"],
          "loso", res["pooled_loso"]["auc"], flush=True)

if __name__ == "__main__":
    main(int(sys.argv[1]))
