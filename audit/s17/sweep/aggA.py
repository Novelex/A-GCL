"""Stage A aggregation. Selection on INNER scores only. Emits per protocol:
honest nested OOF AUC (per-fold inner winner applied once to its test fold), the
OPTIMISTIC best-single-config OOF AUC, top-20 configs, per-fold winners, boundary
pins, the 0.7565 gate, the shuffle arm, bootstrap of the winner vs flat+LinearSVC,
and the top-3 representations by honest nested AUC (consumed by Stage B)."""
import os, sys, json, glob, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_lib as L
from sklearn.metrics import roc_auc_score
OUT = L.ROOT + "A/"; D = L.data(); y = D["y"]
GRIDMIN = dict(linsvm=dict(C=min(L.C_WIDE)), logreg=dict(C=min(L.C_WIDE)), rbf=dict(C=min(L.RBF_C), gamma=min(L.RBF_G)), elasticnet=dict(C=min(L.EN_C), l1_ratio=min(L.EN_L1)))
GRIDMAX = dict(linsvm=dict(C=max(L.C_WIDE)), logreg=dict(C=max(L.C_WIDE)), rbf=dict(C=max(L.RBF_C), gamma=max(L.RBF_G)), elasticnet=dict(C=max(L.EN_C), l1_ratio=max(L.EN_L1)))
def parse(cid):
    kind, *kv = cid.split("|"); return kind, {k: float(v) for k, v in (x.split("=") for x in kv)}

REP_ORDER = [r.name() for r in L.rep_catalogue()]
CFG_ORDER = [L.cfg_id(k, h) for k, h in L.clf_catalogue()]

def load(p):
    """Loads sealed units only (json npz_sha must match the npz). Rows are keyed
    (fold, rep); iteration order everywhere downstream is CATALOGUE order, never
    filesystem order, so tie-breaks are deterministic."""
    folds = L.outer_folds(p); rows = {}; shuf = {}
    for k in range(len(folds)):
        for jf in sorted(glob.glob(f"{OUT}{p}/f{k}/*.json")):
            base = jf[:-5]; j = json.load(open(jf)); z = np.load(base + ".npz")
            assert j["status"] == "OK" and np.array_equal(z["test_idx"], folds[k][2]), jf
            assert j.get("npz_sha") == L.sha_file(base + ".npz"), f"unsealed unit {jf}"
            assert set(j["inner_mean"]) == set(CFG_ORDER), f"catalogue mismatch {jf}"
            (shuf if j["shuffled"] else rows)[(k, j["rep"])] = dict(inner=j["inner_mean"], te=folds[k][2],
                                                                   scores={c: z[c] for c in j["inner_mean"]}, secs=j["secs_total"])
    return folds, rows, shuf

def boot(oa, ob, yy, n=2000):
    rng = np.random.default_rng(L.SEED); d = []
    while len(d) < n:
        i = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[i])) < 2: continue
        d.append(roc_auc_score(yy[i], oa[i]) - roc_auc_score(yy[i], ob[i]))
    d = np.array(d); return dict(delta=float(roc_auc_score(yy, oa) - roc_auc_score(yy, ob)),
                                 lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))

def nested(folds, rows, rep_filter=None, cfg_filter=None):
    """Honest: per fold pick argmax inner over (rep,cfg) in the family; apply once."""
    oof = np.full(954, np.nan); chosen = []
    for k in range(len(folds)):
        best = (-1, None, None)
        for rep in REP_ORDER:                                   # catalogue order: first max wins
            if (k, rep) not in rows or (rep_filter and rep not in rep_filter): continue
            r = rows[(k, rep)]
            for c in CFG_ORDER:
                v = r["inner"][c]
                if cfg_filter and not cfg_filter(c): continue
                if v > best[0]: best = (v, rep, c)
        if best[1] is None: return None, None
        _, rep, c = best; oof[folds[k][2]] = rows[(k, rep)]["scores"][c]; chosen.append(dict(fold=folds[k][0], rep=rep, cfg=c, inner=best[0]))
    return (float(roc_auc_score(y, oof)) if np.isfinite(oof).all() else None), dict(oof=oof, chosen=chosen)

def main():
    report = {}
    for p in L.protocols():
        folds, rows, shuf = load(p)
        n_units = len(folds) * len(L.rep_catalogue()); n_shuf = len(folds)
        complete = (len(rows) == n_units) and (len(shuf) == n_shuf)      # the shuffle arm is part of completeness
        rep = dict(units_present=len(rows), units_expected=n_units, shuffle_present=len(shuf), shuffle_expected=n_shuf, complete=complete)
        if not complete:
            missing = [(k, r.name()) for k in range(len(folds)) for r in L.rep_catalogue() if (k, r.name()) not in rows]
            rep["missing"] = missing[:20]; report[p] = rep; print(f"{p}: INCOMPLETE {len(rows)}/{n_units} (+shuffle {len(shuf)}/{n_shuf})"); continue
        # ---- gate: flat + linsvm restricted to S5.5 grid must be 0.7565 on LAB
        s55 = {L.cfg_id("linsvm", dict(C=c)) for c in L.C_S55}
        g_auc, g = nested(folds, rows, rep_filter={"flat"}, cfg_filter=lambda c: c in s55)
        rep["gate_flat_linsvm_s55grid"] = g_auc
        if p == "lab":
            rep["gate_pass"] = bool(abs(g_auc - 0.7565) < 5e-4)
            if not rep["gate_pass"]: print(f"*** GATE FAILED on LAB: {g_auc:.4f} != 0.7565 — everything below is untrustworthy")
        # ---- shuffled-label arm
        so = np.full(954, np.nan)
        for k in range(len(folds)):
            r = shuf.get((k, "flat"))
            if r is None: so = None; break
            c, _ = L.select_inner({c_: v for c_, v in r["inner"].items() if c_ in s55}); so[r["te"]] = r["scores"][c]
        rng = np.random.default_rng(L.SEED); yp = y[rng.permutation(954)]
        rep["shuffle_arm_auc_vs_permuted_labels"] = float(roc_auc_score(yp, so)) if so is not None else None
        # ---- honest nested: whole family, per representation, per classifier kind
        h_all, ha = nested(folds, rows); rep["honest_nested_all"] = h_all; rep["per_fold_winners"] = ha["chosen"]
        rep["honest_nested_per_rep"] = {r.name(): nested(folds, rows, rep_filter={r.name()})[0] for r in L.rep_catalogue()}
        rep["honest_nested_per_clf"] = {kd: nested(folds, rows, cfg_filter=lambda c, kd=kd: c.startswith(kd + "|"))[0] for kd in ("linsvm", "logreg", "rbf", "elasticnet")}
        flat_ls, fl = nested(folds, rows, rep_filter={"flat"}, cfg_filter=lambda c: c.startswith("linsvm|"))
        rep["honest_flat_linsvm_widegrid"] = flat_ls
        # ---- optimistic: every (rep,cfg) pooled outer OOF; top-20; selection effect
        opt = []
        for rname in REP_ORDER:
            for c in CFG_ORDER:
                oof = np.full(954, np.nan)
                for k in range(len(folds)): oof[folds[k][2]] = rows[(k, rname)]["scores"][c]
                opt.append(dict(rep=rname, cfg=c, outer_oof_auc=float(roc_auc_score(y, oof)),
                                inner_mean=float(np.mean([rows[(k, rname)]["inner"][c] for k in range(len(folds))]))))
        opt.sort(key=lambda d: -d["outer_oof_auc"])
        rep["optimistic_best_single"] = opt[0]; rep["top20_optimistic"] = opt[:20]
        rep["selection_effect"] = float(opt[0]["outer_oof_auc"] - h_all)
        rep["n_configs_searched"] = len(opt)
        # ---- boundary pins of the per-fold winners
        pins = []
        for w in ha["chosen"]:
            kind, hp = parse(w["cfg"])
            for hk, hv in hp.items():
                if hv == GRIDMIN[kind][hk]: pins.append(f"{w['fold']}:{kind}.{hk}@MIN")
                if hv == GRIDMAX[kind][hk]: pins.append(f"{w['fold']}:{kind}.{hk}@MAX")
            if "tangent_shrink" in w["rep"] or "partial" in w["rep"]:
                a = float(w["rep"].split("_a")[1].split("_")[0])
                grid = (0.05, 0.8) if "shrink" in w["rep"] else (0.1, 0.6)
                if a in grid: pins.append(f"{w['fold']}:{w['rep']}@{'MIN' if a==grid[0] else 'MAX'}")
            if "tangent_trunc" in w["rep"]:
                kk = int(w["rep"].split("_k")[1].split("_")[0])
                if kk in (10, 90): pins.append(f"{w['fold']}:{w['rep']}@{'MIN' if kk==10 else 'MAX'}")
        rep["boundary_pins"] = pins
        rep["chosen_hparams_per_fold"] = [dict(fold=w["fold"], rep=w["rep"], clf=parse(w["cfg"])[0], **parse(w["cfg"])[1]) for w in ha["chosen"]]
        # ---- bootstrap: honest winner vs flat+LinearSVC (S5.5 grid)
        rep["boot_winner_vs_flat_linsvm_s55"] = boot(ha["oof"], g["oof"], y)
        # ---- top-3 representations for Stage B
        rep["top3_reps_by_honest_nested"] = sorted(REP_ORDER, key=lambda r_: -rep["honest_nested_per_rep"][r_])[:3]
        # Stage-B hand-off uses INNER information only: each rep's best inner score per fold, averaged over folds
        mean_inner = {r_: float(np.mean([max(rows[(k, r_)]["inner"].values()) for k in range(len(folds))])) for r_ in REP_ORDER}
        rep["rep_mean_inner_best"] = mean_inner
        rep["top3_reps_by_mean_inner"] = sorted(REP_ORDER, key=lambda r_: -mean_inner[r_])[:3]
        rep["mean_secs_per_unit"] = float(np.mean([r["secs"] for r in rows.values()]))
        report[p] = rep
        print(f"{p}: honest ALL {h_all:.4f} | optimistic best {opt[0]['outer_oof_auc']:.4f} ({opt[0]['rep']} {opt[0]['cfg']}) | "
              f"gate {g_auc:.4f} | shuffle {rep['shuffle_arm_auc_vs_permuted_labels']} | top3(inner) {rep['top3_reps_by_mean_inner']}")
        L.anpz(OUT + f"oof_{p}.npz", honest=ha["oof"], flat_s55=g["oof"], y=y)
    report["_meta"] = dict(reference_mean_implementations="log-Euclidean = expm(mean(logm)); geometric = Fletcher-2007 Karcher iteration "
                            "VENDORED from nilearn 0.14.0 _geometric_mean (nilearn not installed)", rep_order=REP_ORDER)
    L.aj(report, OUT + "AGG_A.json"); print("wrote", OUT + "AGG_A.json")
    if "lab" in report and report["lab"].get("complete") and not report["lab"].get("gate_pass"):
        print("*** LAB GATE FAILED: flat+LinearSVC did not reproduce 0.7565. Chain STOPS here (exit 5)."); return 5
    return 0 if all(report[p].get("complete") for p in report if not p.startswith("_")) else 4

if __name__ == "__main__": sys.exit(main())
