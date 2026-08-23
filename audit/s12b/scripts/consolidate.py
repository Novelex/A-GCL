"""S12B consolidation: information_audit.csv (full expanded factorial),
training_diagnostics.csv, the 6 mandated plots, best-config detection.
Pure assembly — no new measurements."""
import sys, os, json, glob, hashlib, numpy as np, pandas as pd
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12b/scripts"); import s12b_core as B
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CEIL = json.load(open(B.S12B + "out/GATE1.json"))["CEILING_PROBE"]

def ret(a): return (a - 0.5) / (CEIL - 0.5) if np.isfinite(a) else np.nan

def load_unit(tag):
    p = f"{B.S12B}jobs/t1_{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None

def row_from(res, extra):
    po = res.get("pooled_ordinary", {})
    pl = res.get("pooled_loso", {})
    fc = res.get("fc_recon", {})
    cf = res.get("confounds", {})
    st = res.get("site", {})
    r = dict(extra)
    r.update(dim=res.get("dim"), auc=po.get("auc"), auc_ci_lo=po.get("auc_ci_lo"),
        auc_ci_hi=po.get("auc_ci_hi"), auprc=po.get("auprc"), acc=po.get("acc"),
        bal_acc=po.get("bal_acc"), sens=po.get("sens"), spec=po.get("spec"),
        ppv=po.get("ppv"), npv=po.get("npv"), f1=po.get("f1"), mcc=po.get("mcc"),
        brier=po.get("brier"), calib_slope=po.get("calib_slope"),
        calib_intercept=po.get("calib_intercept"),
        retention_ratio=ret(po.get("auc", np.nan)),
        fc_recon_r2=fc.get("r2"), frac_edges_r2_gt_0p5=fc.get("frac_edges_r2_gt_0p5"),
        site_f1=st.get("macro_f1"), age_r2=cf.get("age"), sex_r2=cf.get("sex"),
        motion_r2=cf.get("motion"), meanfc_r2=cf.get("meanfc"),
        totalfc_r2=cf.get("totalfc"), auc_loso=pl.get("auc"),
        fold_auc_sd=res.get("fold_auc_sd"))
    return r

def build_csv():
    rows = []
    for arm in B.ARMS:
        arm_lvl = {}
        for st in ("S0", "S1"):
            u = load_unit(f"{arm}_{st}")
            if u: arm_lvl[st] = (u["results"][st], u)
        for emb in B.EMBS:
            for norm in B.NORMS:
                for mr in (True, False):
                    for s in range(len(B.SEEDS)):
                        u = load_unit(f"{arm}_e{emb}_{norm}_mr{int(mr)}_s{s}")
                        if u is None: continue
                        meta0 = dict(track="T1", arm=arm, seed=B.BASE + s, emb_dim=emb,
                                     norm_type=norm, message_relu=mr,
                                     runtime_s=u["wall_s"], gpu_mem_mb=u.get("gpu_mem_mb"),
                                     node=u["provenance"]["host"], git_sha=u["provenance"]["git"])
                        R = u["results"]
                        for nn in (True, False):
                            m = dict(meta0, normalize_nodes=nn)
                            for st, src, ro in (
                                ("S0", "arm", "node_flat"), ("S1", "arm", "node_flat"),
                                ("S2", "S2", "node_flat"), ("S3", "S3", "node_flat"),
                                ("S4", "S4", "node_flat"),
                                ("S5", f"S5_nn{'T' if nn else 'F'}", "global_add_pool"),
                                ("S6", "S4" if nn else "S3", "roi_flatten")):
                                if st == "S4" and not nn: continue
                                if src == "arm":
                                    if st not in arm_lvl: continue
                                    res, uu = arm_lvl[st]
                                    rows.append(row_from(res, dict(m, stage=st, readout=ro,
                                        runtime_s=uu["wall_s"], gpu_mem_mb=None)))
                                else:
                                    rows.append(row_from(R[src], dict(m, stage=st, readout=ro)))
    df = pd.DataFrame(rows)
    df.to_csv(B.S12B + "information_audit.csv", index=False)
    return df

def build_diag():
    fs = sorted(glob.glob(B.S12B + "jobs/t2_*_diag.csv"))
    if fs:
        pd.concat([pd.read_csv(f) for f in fs]).to_csv(
            B.S12B + "training_diagnostics.csv", index=False)
    return len(fs)

def best_config(df):
    d6 = df[(df.stage == "S6")].copy()
    g = d6.groupby(["arm", "emb_dim", "norm_type", "message_relu", "normalize_nodes"]
                   ).agg(auc=("auc", "mean"), r2=("fc_recon_r2", "mean"),
                         sd=("auc", "std")).reset_index().sort_values(
                         ["auc", "r2"], ascending=False)
    best = g.iloc[0].to_dict()
    B.atomic_json(dict(best=best, top10=g.head(10).to_dict("records"),
                       ceiling=CEIL), B.S12B + "out/BEST_CONFIG.json")
    return best, g

DEFAULT = dict(emb_dim=32, norm_type="bn", message_relu=True, normalize_nodes=True)

def plots(df):
    P = B.S12B + "plots/"
    stages = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
    # 1+2: retention curves at production-default config, mean+-sd over seeds
    for metric, fn, yl in (("auc", "p1_retention_auc.png", "pooled OOF AUC"),
                           ("fc_recon_r2", "p2_retention_fcr2.png", "FC recon R2")):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for arm in B.ARMS:
            q = df[(df.arm == arm) & (df.emb_dim == DEFAULT["emb_dim"]) &
                   (df.norm_type == DEFAULT["norm_type"]) &
                   (df.message_relu == DEFAULT["message_relu"]) &
                   (df.normalize_nodes == DEFAULT["normalize_nodes"])]
            mu = [q[q.stage == s][metric].mean() for s in stages]
            sd = [q[q.stage == s][metric].std() for s in stages]
            ax.errorbar(stages, mu, yerr=sd, marker="o", capsize=3, label=arm)
        if metric == "auc":
            ax.axhline(CEIL, ls="--", c="k", lw=1, label=f"ceiling {CEIL:.3f}")
            ax.axhline(0.5, ls=":", c="gray", lw=1)
        ax.set_ylabel(yl); ax.set_title(f"S12B stage {yl} — default config "
                                        "(emb32/bn/mrelu/nn), mean+-sd over 5 seeds")
        ax.legend(); fig.tight_layout(); fig.savefig(P + fn, dpi=140); plt.close(fig)
    # 3: heatmap emb x norm at H2 (S3), per arm, retention (mean over mrelu+seeds)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, arm in zip(axes, ["A", "B", "C", "D"]):
        q = df[(df.arm == arm) & (df.stage == "S3")]
        M = np.array([[q[(q.emb_dim == e) & (q.norm_type == n)].retention_ratio.mean()
                       for n in B.NORMS] for e in B.EMBS])
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(3), B.NORMS); ax.set_yticks(range(3), B.EMBS)
        ax.set_title(f"arm {arm} @H2"); fig.colorbar(im, ax=ax, fraction=0.046)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="w", fontsize=8)
    fig.suptitle("retention at H2: emb_dim x norm_type")
    fig.tight_layout(); fig.savefig(P + "p3_heat_emb_norm.png", dpi=140); plt.close(fig)
    # 4: heatmap normalize_nodes x readout (S5 vs S6), per arm
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, arm in zip(axes, ["A", "B", "C", "D"]):
        q = df[(df.arm == arm) & (df.stage.isin(["S5", "S6"]))]
        M = np.array([[q[(q.normalize_nodes == nn) & (q.stage == st)].retention_ratio.mean()
                       for st in ("S5", "S6")] for nn in (True, False)])
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(2), ["pool_sum(S5)", "ROI_flat(S6)"])
        ax.set_yticks(range(2), ["nn=T", "nn=F"])
        ax.set_title(f"arm {arm}"); fig.colorbar(im, ax=ax, fraction=0.046)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="w", fontsize=9)
    fig.suptitle("retention: normalize_nodes x readout")
    fig.tight_layout(); fig.savefig(P + "p4_heat_nn_readout.png", dpi=140); plt.close(fig)
    # 5: biology vs scanner scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    q = df.dropna(subset=["site_f1", "auc"])
    sc = ax.scatter(q.site_f1, q.auc, c=[stages.index(s) for s in q.stage],
                    cmap="plasma", s=14, alpha=0.65)
    fig.colorbar(sc, ax=ax, label="stage idx (S0..S6)")
    ax.set_xlabel("site macro-F1 (scanner)"); ax.set_ylabel("diagnosis AUC (biology)")
    ax.axhline(0.5, ls=":", c="gray"); ax.set_title("all configs x stages")
    fig.tight_layout(); fig.savefig(P + "p5_biology_vs_scanner.png", dpi=140); plt.close(fig)
    # 6: Track 2 curves per weight decay
    dg = B.S12B + "training_diagnostics.csv"
    if os.path.exists(dg):
        t = pd.read_csv(dg)
        t[["h", "wd", "s"]] = t.unit.str.extract(r"h(\d+)_wd([\deE.+-]+)_s(\d)")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
        for ax, h in zip(axes, ["64", "256", "512"]):
            for wd, qq in t[t.h == h].groupby("wd"):
                m = qq.groupby("epoch")[["train_auc", "val_auc"]].mean()
                ax.plot(m.index, m.train_auc, lw=1, alpha=0.9, label=f"wd={wd} train")
                ax.plot(m.index, m.val_auc, lw=1.6, ls="--", label=f"wd={wd} val")
            ax.set_title(f"hidden={h}"); ax.set_xlabel("epoch")
        axes[0].set_ylabel("AUC"); axes[-1].legend(fontsize=7)
        fig.suptitle("Track 2: train/val AUC by weight decay (mean over folds+seeds)")
        fig.tight_layout(); fig.savefig(P + "p6_t2_curves.png", dpi=140); plt.close(fig)

def main():
    df = build_csv(); n2 = build_diag()
    print("csv rows", len(df), "| t2 diag files", n2)
    best, g = best_config(df)
    print("BEST S6 config:", best)
    plots(df)
    man = {}
    for f in sorted(glob.glob(B.S12B + "jobs/*.json")) + \
             sorted(glob.glob(B.S12B + "out/*.json")) + \
             sorted(glob.glob(B.S12B + "plots/*.png")) + \
             [B.S12B + "information_audit.csv"]:
        if os.path.exists(f):
            man[f.replace(B.S12B, "")] = hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
    B.atomic_json(man, B.S12B + "manifest.json")
    du = sum(os.path.getsize(f) for f in glob.glob(B.S12B + "**/*", recursive=True)
             if os.path.isfile(f)) / 1e9
    print(f"disk used under s12b: {du:.2f} GB")

if __name__ == "__main__":
    main()
