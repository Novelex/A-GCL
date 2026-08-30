"""Stage B + C aggregation and the ALFF blend. Selection on INNER scores only.
Neural folds failing movement > 0.10 or clip_rate < 0.30 are DISCARDED from
selection and from the honest pooled number (they are counted and listed)."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_lib as L, mlp_lib as M
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from aggA import boot, nested as nestedA, load as loadA, parse
D = L.data(); y = D["y"]
GRID_EDGES = dict(width=(16, 512), dropout=(0.0, 0.5), lr=(1e-4, 1e-3), wd=(1e-4, 1e-2))

CFG_B = [M.cfg_id(c) for c in M.grid_B()]

def expected_grid(stage, p):
    """The config set a row must carry to count as COMPLETE.

    DEVIATION_01: Stage-B LOSO is the confirmatory top-K wave, so its expected set
    is the frozen K=30 list in B/TOPK.json, not the full 540-config grid. Without
    this, load_stage() would pop every top-K LOSO row as 'incomplete' and the LOSO
    protocol would silently report as absent. Every other (stage, protocol) is
    unchanged and still requires the full grid."""
    if stage == "B" and p == "loso" and os.path.exists(L.ROOT + "B/TOPK.json"):
        d = json.load(open(L.ROOT + "B/TOPK.json"))
        assert d["deviation"] == "DEVIATION_01" and d["K"] == len(d["configs"]), "TOPK.json not the pre-registered wave. STOP."
        return set(d["configs"])
    if os.environ.get("SWEEP_SMOKE_GRID"): return set(CFG_B[:int(os.environ["SWEEP_SMOKE_GRID"])])
    return set(CFG_B)

def load_stage(stage, p):
    """Sealed chunks only; a (fold,input) row is COMPLETE only when every config of
    the grid is present - a missing chunk can never silently shrink the search."""
    folds = L.outer_folds(p); rows = {}
    for jf in sorted(glob.glob(f"{L.ROOT}{stage}/{p}/f*/**/chunk*.json", recursive=True)):
        j = json.load(open(jf)); z = np.load(jf[:-5] + ".npz"); k = j["k_fold"]
        assert j["status"] == "OK" and np.array_equal(z["test_idx"], folds[k][2]), jf
        assert j.get("npz_sha") == L.sha_file(jf[:-5] + ".npz"), f"unsealed chunk {jf}"
        key = (k, j.get("input_idx", 0))
        r = rows.setdefault(key, dict(inner={}, scores={}, outer_info={}, rep=j.get("rep", "mALFF90"), alff=j.get("alff", True), te=folds[k][2]))
        r["inner"].update(j["inner_mean"]); r["outer_info"].update(j["outer_info"])
        r.setdefault("inner_raw", {}).update(j["inner_auc"])
        r["scores"].update({c: z[c] for c in j["configs"]})
        r.setdefault("inner_info", {}).update(j["inner_info"])
    grid = expected_grid(stage, p)
    for key, r in list(rows.items()):
        if set(r["inner"]) != grid: rows.pop(key)              # incomplete row -> treated as absent
    return folds, rows

MIN_VALID_INNER = 3
def run_valid(r, c):
    """Health rule. The OUTER refit must satisfy movement>0.10 and clip<0.30, and at
    least MIN_VALID_INNER of the inner-split trainings must too. Inner runs failing
    health are EXCLUDED from the config's inner mean (see inner_score), so a stalled
    inner training can neither promote nor demote a config; it is reported."""
    return bool(r["outer_info"][c]["valid"] and sum(i_["valid"] for i_ in r["inner_info"][c]) >= MIN_VALID_INNER)

def inner_score(r, c, raw_aucs):
    """Mean inner AUC over the VALID inner runs only."""
    v = [a for a, i_ in zip(raw_aucs, r["inner_info"][c]) if i_["valid"]]
    return float(np.mean(v)) if v else -1.0

def nested_B(folds, rows, input_filter=None):
    oof = np.full(954, np.nan); chosen = []; discarded = []; unresolved = []
    for k in range(len(folds)):
        best = (-1, None, None)
        for (kk, i) in sorted(rows):                            # deterministic: fold, then input index
            r = rows[(kk, i)]
            if kk != k or (input_filter is not None and i not in input_filter): continue
            for c in CFG_B:
                if c not in r["inner"]: continue
                if not run_valid(r, c): discarded.append((folds[k][0], i, c)); continue
                v = inner_score(r, c, r["inner_raw"][c])
                if v > best[0]: best = (v, i, c)
        if best[1] is None:
            unresolved.append(folds[k][0]); continue
        _, i, c = best; oof[folds[k][2]] = rows[(k, i)]["scores"][c]
        oi = rows[(k, i)]["outer_info"][c]
        chosen.append(dict(fold=folds[k][0], input_idx=i, rep=rows[(k, i)]["rep"], alff=rows[(k, i)]["alff"], cfg=c, inner=best[0],
                           movement_max=oi["movement_max"], clip_rate=oi["clip_rate"], best_epoch=oi["best_epoch"]))
    info = dict(oof=oof, chosen=chosen, n_discarded=len(discarded), discarded=discarded[:50], unresolved_folds=unresolved)
    return (float(roc_auc_score(y, oof)) if not unresolved else None), info

def pins(chosen):
    out = []
    for w in chosen:
        hp = dict(width=int(w["cfg"].split("_")[0][1:]), dropout=float(w["cfg"].split("_p")[1].split("_")[0]),
                  lr=float(w["cfg"].split("_lr")[1].split("_")[0]), wd=float(w["cfg"].split("_wd")[1]))
        for k_, (lo, hi) in GRID_EDGES.items():
            if hp[k_] == lo: out.append(f"{w['fold']}:{k_}@MIN")
            if hp[k_] == hi: out.append(f"{w['fold']}:{k_}@MAX")
    return out

def main():
    aggA = json.load(open(L.ROOT + "A/AGG_A.json")); report = {}
    for p in L.protocols():
        rep = {}
        foldsB, rowsB = load_stage("B", p); foldsC, rowsC = load_stage("C", p)
        nin = len(json.load(open(L.ROOT + "B/inputs.json"))) if os.path.exists(L.ROOT + "B/inputs.json") else 6
        expB = len(foldsB) * nin; rep["B_units_present"] = len(rowsB); rep["B_expected"] = expB
        rep["C_folds_present"] = len(rowsC); rep["C_expected"] = len(foldsC)
        if len(rowsB) < expB or len(rowsC) < len(foldsC):
            rep["complete"] = False; report[p] = rep; print(f"{p}: INCOMPLETE B {len(rowsB)}/{expB} C {len(rowsC)}/{len(foldsC)}"); continue
        rep["complete"] = True
        if p == "loso" and os.path.exists(L.ROOT + "B/TOPK.json"):
            rep["B_selection"] = "DEVIATION_01_topk"
            rep["B_estimand"] = ("select-on-LAB+SITE, evaluate-on-LOSO: only the K=30 configs frozen in "
                                 "B/TOPK.json were evaluated here. This is NOT a 540-config LOSO search "
                                 "and must not be compared to LAB/SITE as if it were.")
            _tk = json.load(open(L.ROOT + "B/TOPK.json"))
            rep["B_topk_amendment"] = _tk.get("amendment")
            rep["B_topk_n_winners"] = _tk.get("n_winners")
            rep["B_topk_min_cells_for_padding"] = _tk.get("min_cells_for_padding")
            rep["B_topk_cutoff_mean_inner"] = _tk.get("cutoff_mean_inner")   # absent under rule A3
            rep["B_topk_rule"] = _tk.get("rule")
        # ---- Stage B honest nested (all inputs), per input, optimistic, top-20, pins, health
        hB, hb = nested_B(foldsB, rowsB); rep["B_honest_nested_all"] = hB; rep["B_per_fold_winners"] = hb["chosen"]
        rep["B_unresolved_folds"] = hb["unresolved_folds"]
        rep["B_n_runs_discarded_by_health"] = hb["n_discarded"]      # (fold, input, config) triples
        rep["B_discarded_examples"] = hb["discarded"]
        rep["B_honest_per_input"] = {str(i): nested_B(foldsB, rowsB, {i})[0] for i in sorted({i for _, i in rowsB})}
        opt = []
        for i in sorted({i_ for _, i_ in rowsB}):
            for c in CFG_B:
                if c not in rowsB[(0, i)]["inner"]: continue
                if not all(run_valid(rowsB[(k, i)], c) for k in range(len(foldsB))): continue
                oof = np.full(954, np.nan)
                for k in range(len(foldsB)): oof[foldsB[k][2]] = rowsB[(k, i)]["scores"][c]
                opt.append(dict(input_idx=i, rep=rowsB[(0, i)]["rep"], alff=rowsB[(0, i)]["alff"], cfg=c,
                                outer_oof_auc=float(roc_auc_score(y, oof)),
                                inner_mean=float(np.mean([inner_score(rowsB[(k, i)], c, rowsB[(k, i)]["inner_raw"][c]) for k in range(len(foldsB))])),
                                movement_med=float(np.median([rowsB[(k, i)]["outer_info"][c]["movement_max"] for k in range(len(foldsB))])),
                                best_epoch_med=float(np.median([rowsB[(k, i)]["outer_info"][c]["best_epoch"] for k in range(len(foldsB))]))))
        opt.sort(key=lambda d: -d["outer_oof_auc"])
        rep["B_optimistic_best_single"] = opt[0] if opt else None; rep["B_top20_optimistic"] = opt[:20]
        rep["B_selection_effect"] = (float(opt[0]["outer_oof_auc"] - hB) if opt else None)
        rep["B_boundary_pins"] = pins(hb["chosen"])
        rep["B_width_won"] = [int(w["cfg"].split("_")[0][1:]) for w in hb["chosen"]]
        rep["B_dropout_won"] = [float(w["cfg"].split("_p")[1].split("_")[0]) for w in hb["chosen"]]
        rep["B_alff_in_winners"] = int(sum(1 for w in hb["chosen"] if w["alff"]))
        rep["B_alff_in_top20"] = int(sum(1 for d in opt[:20] if d["alff"]))
        # ---- Stage C: ALFF-90 MLP alone
        hC, hc = nested_B(foldsC, rowsC); rep["C_alff_mlp_honest_nested"] = hC; rep["C_per_fold_winners"] = hc["chosen"]
        rep["C_unresolved_folds"] = hc["unresolved_folds"]; rep["C_n_runs_discarded_by_health"] = hc["n_discarded"]
        # ---- BLEND w1 z(FC) + w2 z(ALFF): FC score = Stage-A honest winner; inner cross-fit with the
        #      per-fold chosen configs so the weights see only out-of-sample scores of TRAIN subjects.
        foldsA, rowsA, _ = loadA(p); hA, ha = nestedA(foldsA, rowsA)
        if hA is None:
            rep["blend"] = "SKIPPED: Stage A incomplete for this protocol"; report[p] = rep; print(f"{p}: blend skipped"); continue
        flat_s55 = np.load(L.ROOT + f"A/oof_{p}.npz")["flat_s55"]
        if hB is not None: rep["B_boot_vs_flat_linsvm_s55"] = boot(hb["oof"], flat_s55, y)
        if hC is not None: rep["C_boot_vs_flat_linsvm_s55"] = boot(hc["oof"], flat_s55, y)
        if hC is None:
            rep["blend"] = f"SKIPPED: ALFF-MLP has no health-eligible config in folds {hc['unresolved_folds']}"
            report[p] = rep; print(f"{p}: B honest {hB} | C UNRESOLVED {hc['unresolved_folds']} -> blend skipped"); continue
        W2 = [round(x, 2) for x in np.arange(0, 1.0001, 0.1)]; fused = np.full(954, np.nan); w2s = []
        for k in range(len(foldsA)):
            tag, tr, te = foldsA[k]; wa = ha["chosen"][k]; wc = hc["chosen"][k]
            repA = next(r for r in L.rep_catalogue() if r.name() == wa["rep"]); kind, hp = parse(wa["cfg"])
            cfgC = next(cf for cf in M.grid_B() if M.cfg_id(cf) == wc["cfg"])
            in_fc = np.full(954, np.nan); in_al = np.full(954, np.nan)
            for a, b in L.inner_splits(tr, y):
                r = L.Rep(repA.kind, repA.alpha, repA.k, repA.mean).fit(D["FCsq"][a])
                Xa, Xb = r.transform(D["FCsq"][a]), r.transform(D["FCsq"][b]); sc = StandardScaler().fit(Xa)
                in_fc[b] = L.make_clf(kind, hp).fit(sc.transform(Xa), y[a]).decision_function(sc.transform(Xb))
                scA = StandardScaler().fit(D["ALFF"][a]); Xa2, Xb2 = scA.transform(D["ALFF"][a]).astype(np.float32), scA.transform(D["ALFF"][b]).astype(np.float32)
                m_, _ = M.train_mlp(np.vstack([Xa2, Xb2]), np.concatenate([y[a], np.zeros(len(b), int)]), np.arange(len(a)), cfgC, threads=1)
                in_al[b] = M.scores(m_, Xb2)
            mu_f, sd_f, mu_a, sd_a = in_fc[tr].mean(), in_fc[tr].std(), in_al[tr].mean(), in_al[tr].std()
            zf, za = (in_fc[tr] - mu_f) / sd_f, (in_al[tr] - mu_a) / sd_a
            w2 = max(W2, key=lambda w: (roc_auc_score(y[tr], (1 - w) * zf + w * za), -w)); w2s.append(w2)
            fused[te] = (1 - w2) * (rowsA[(k, wa["rep"])]["scores"][wa["cfg"]] - mu_f) / sd_f + w2 * (rowsC[(k, 0)]["scores"][wc["cfg"]] - mu_a) / sd_a
        rep["blend_fused_honest_auc"] = float(roc_auc_score(y, fused)); rep["blend_w2_per_fold"] = w2s
        rep["blend_w2_nonzero_folds"] = int(sum(w > 0 for w in w2s)); rep["blend_boot_vs_A_winner"] = boot(fused, ha["oof"], y)
        rep["blend_boot_vs_flat_linsvm_s55"] = boot(fused, flat_s55, y)
        report[p] = rep
        print(f"{p}: B honest {hB} | B optimistic {opt[0]['outer_oof_auc'] if opt else None} | C alff-mlp {hC} | blend {rep['blend_fused_honest_auc']:.4f} w2 {w2s}")
        L.anpz(L.ROOT + f"oof_BC_{p}.npz", B_honest=hb["oof"], C_honest=hc["oof"], blend=fused, y=y)
    L.aj(report, L.ROOT + "AGG_BC.json"); print("wrote AGG_BC.json")
    return 0 if all(report[p].get("complete") for p in report) else 4

if __name__ == "__main__": sys.exit(main())
