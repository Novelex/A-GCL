"""S12A4 consolidation: assemble pooled OOF per (arm,seed) from per-fold artifacts,
apply pre-registered decision bands. Hard-gates arm4 validity and grad-finiteness."""
import json, glob, numpy as np
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             accuracy_score, confusion_matrix)

S = "/users/3171356m/agcl_audit_s0/s12a4/"
ARM_NAMES = {1: "CE", 2: "CE+InfoNCE", 3: "A-GCL corrected", 4: "A-GCL hard-mask 0.8"}

def pooled(y, oof):
    cov = np.isfinite(oof); yc, oc = y[cov], oof[cov]
    yh = (oc > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(yc, yh, labels=[0, 1]).ravel()
    return dict(auc=float(roc_auc_score(yc, oc)), acc=float(accuracy_score(yc, yh)),
                bacc=float(balanced_accuracy_score(yc, yh)),
                sens=float(tp / (tp + fn)), spec=float(tn / (tn + fp)),
                n=int(cov.sum()))

def collect(arm, sidx, kind):
    pre = f"{S}feat/a{arm}_s{sidx}_{'o' if kind == 'ord' else 'l'}"
    files = sorted(glob.glob(pre + "*.npz"))
    if not files: return None
    z0 = np.load(files[0]); y = z0["y"]
    oof_h = np.full(len(y), np.nan); oof_z = np.full(len(y), np.nan)
    oof_head = np.full(len(y), np.nan)
    gaps, vals, epochs = [], [], []
    for f in files:
        z = np.load(f); te = z["te"]
        oof_h[te] = z["oof_h"][te]; oof_z[te] = z["oof_z"][te]
        if z["head"].shape[0] == len(y): oof_head[te] = z["head"][te]
        tag = f.split("/")[-1].replace(".npz", "")
        j = json.load(open(f"{S}out/fold_{tag}.json"))
        r = j["rec"]
        assert r["grad_finite_all"], f"non-finite grads in {tag}"
        if arm == 4: assert r["arm4_valid"], f"arm4 keep-rate INVALID in {tag}"
        vals.append(r["best_val_auc"]); epochs.append(r["best_epoch"])
        tr_auc = [c.get("train_auc") for c in j["curve"] if "train_auc" in c]
        va_auc = [c.get("val_auc") for c in j["curve"] if "val_auc" in c]
        if tr_auc and va_auc: gaps.append(tr_auc[-1] - va_auc[-1])
    return dict(y=y, h=pooled(y, oof_h), z=pooled(y, oof_z),
                head=pooled(y, oof_head) if np.isfinite(oof_head).any() else None,
                n_folds=len(files), mean_val=float(np.mean(vals)),
                mean_gap=float(np.mean(gaps)) if gaps else None,
                mean_best_epoch=float(np.mean(epochs)))

if __name__ == "__main__":
    R = {}
    for arm in (1, 2, 3, 4):
        for sidx in (0, 1, 2):
            for kind in ("ord", "loso"):
                c = collect(arm, sidx, kind)
                if c:
                    c.pop("y"); R[f"a{arm}_s{sidx}_{kind}"] = c
    # table
    print(f"{'arm':<22}{'metric':<7}{'s0':>8}{'s1':>8}{'s2':>8}{'mean':>8}  LOSO(h)")
    M = {}
    for arm in (1, 2, 3, 4):
        for met in (("head",) if arm in (1, 2) else ()) + ("h", "z"):
            v = [R[f"a{arm}_s{s}_ord"][met]["auc"] if R.get(f"a{arm}_s{s}_ord") and
                 R[f"a{arm}_s{s}_ord"][met] else np.nan for s in (0, 1, 2)]
            lo = [R[f"a{arm}_s{s}_loso"]["h"]["auc"] if R.get(f"a{arm}_s{s}_loso")
                  else np.nan for s in (0, 1, 2)]
            M[(arm, met)] = float(np.nanmean(v))
            print(f"{ARM_NAMES[arm]:<22}{met:<7}" + "".join(f"{x:8.4f}" for x in v)
                  + f"{np.nanmean(v):8.4f}  {np.nanmean(lo):.4f}"
                  if met == "h" else
                  f"{ARM_NAMES[arm]:<22}{met:<7}" + "".join(f"{x:8.4f}" for x in v)
                  + f"{np.nanmean(v):8.4f}")
    ce = M[(1, "head")]; cen = M[(2, "head")]
    a3, a4 = M[(3, "h")], M[(4, "h")]
    ce_h, cen_h = M[(1, "h")], M[(2, "h")]
    band = ("ARCHITECTURE CAN APPROACH FC BASELINE" if ce >= 0.70 else
            "ENCODER COMPRESSION LIMITS PERFORMANCE" if 0.60 <= ce <= 0.68 else
            "GNN DESIGN UNSUITABLE" if ce < 0.60 else "BORDERLINE 0.68-0.70")
    print(f"\nCE band (head OOF {ce:.4f}): {band}")
    print(f"cross-arm (SVM on h): CE {ce_h:.4f} | CE+NCE {cen_h:.4f} | "
          f"A-GCL {a3:.4f} | hard-mask {a4:.4f}")
    print(f"A-GCL>CE+NCE>CE: {a3 > cen_h > ce_h} | CE>A-GCL: {ce_h > a3} | "
          f"hard>normal: {a4 > a3}")
    json.dump(dict(results=R, means={f"{k[0]}_{k[1]}": v for k, v in M.items()},
                   ce_band=band), open(S + "out/CONSOLIDATED.json", "w"),
              indent=1, default=str)
