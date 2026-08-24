"""C2 FEASIBILITY REPORT — counting only. Fits NO probe, trains NOTHING, executes NO
refit. Determines whether the site x label matched estimator is even runnable."""
import sys, os, json, collections, numpy as np
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_data as DAT, s16_c2_bounded as CB
S16 = DAT.S16

def main():
    y, sites, ids = CB.cohort()
    d,_,_ = DAT.load("signed", where="feas")
    folds = DAT.folds(d, "lab")          # the ordinary folds C2 uses
    all_rows, per_fold = [], []
    for tag, tr, te in folds:
        seed_res = []
        for seed in CB.SEEDS:
            for hi,(te_a, te_b) in enumerate(CB.half_split(te, y, seed)):
                rows, ok = CB.feasibility(te_a, tr, y, sites)
                for r in rows:
                    r.update(fold=tag, seed=seed, half=hi)
                    all_rows.append(r)
                seed_res.append(ok)
        per_fold.append(dict(fold=tag, n_te=int(len(te)), n_tr=int(len(tr)),
                             halves_checked=len(seed_res),
                             halves_feasible=int(sum(seed_res)),
                             fold_feasible=bool(all(seed_res))))
    infeas = [r for r in all_rows if not r["feasible"]]
    overall = len(infeas) == 0
    # collapse to unique site x label cells for the human-readable table
    agg = collections.defaultdict(lambda: dict(req_max=0, avail_min=10**9, n_checks=0,
                                               n_infeasible=0))
    for r in all_rows:
        k = (r["fold"], r["site"], r["label"]); a = agg[k]
        a["req_max"] = max(a["req_max"], r["requested"])
        a["avail_min"] = min(a["avail_min"], r["available_in_tr"])
        a["n_checks"] += 1; a["n_infeasible"] += (0 if r["feasible"] else 1)
    md = ["# S16 C2 FEASIBILITY REPORT — counting only, NO refit executed", "",
      "This report fits no probe, trains no model and scores nothing. It answers one",
      "question: can the site x label matched biased comparator be drawn from `tr`",
      "WITHOUT REPLACEMENT for every fold, seed and half?", "",
      f"Folds: {len(folds)} ordinary. Seeds: {len(CB.SEEDS)} predefined "
      f"({CB.SEEDS[0]}..{CB.SEEDS[-1]}). Halves per seed: 2. "
      f"Cell checks performed: {len(all_rows)}.", "",
      "## Per-fold summary", "",
      "| fold | n_tr | n_te | halves checked | halves feasible | fold feasible |",
      "|---|---|---|---|---|---|"]
    for f in per_fold:
        md.append(f"| {f['fold']} | {f['n_tr']} | {f['n_te']} | {f['halves_checked']} | "
                  f"{f['halves_feasible']} | {'YES' if f['fold_feasible'] else '**NO**'} |")
    md += ["", "## Site x label cells — worst case across all seeds and halves", "",
      "`requested` is the maximum any half asked for; `available in tr` is the minimum",
      "the pool ever held. A cell is infeasible if requested > available at any point.", "",
      "| fold | site | label | requested (max) | available in tr (min) | feasible |",
      "|---|---|---|---|---|---|"]
    for (fold, site, lab), a in sorted(agg.items()):
        ok = a["n_infeasible"] == 0
        mark = "yes" if ok else f"**NO ({a['n_infeasible']}/{a['n_checks']} checks)**"
        md.append(f"| {fold} | {site} | {lab} | {a['req_max']} | {a['avail_min']} | {mark} |")
    md += ["", f"## VERDICT: {'FEASIBLE' if overall else 'INFEASIBLE'}", ""]
    if overall:
        md += ["Every site x label cell can be matched without replacement for every",
               "fold, seed and half. The bounded estimator is runnable. **It has NOT",
               "been run.**"]
    else:
        bad = sorted({(r['fold'], r['site'], r['label']) for r in infeas})
        md += [f"{len(infeas)} cell checks are infeasible across {len(bad)} distinct",
               "fold x site x label combinations. Per the policy: **no replacement, no",
               "pooling of sites, no weighting, no substitute estimator.** Retrospective",
               "pure bias is therefore **UNRESOLVED** for the affected folds.", "",
               "| fold | site | label |", "|---|---|---|"]
        md += [f"| {f} | {s} | {l} |" for f,s,l in bad[:40]]
    md += ["", "## Residual confounding NOT addressed in this pass",
      "Matching site x label removes site composition and class balance as explanations",
      "for the paired difference. The following are **deliberately NOT matched** and",
      "remain residual confounds: " + ", ".join(CB.RESIDUAL_CONFOUNDS) + ".", "",
      "## Predeclared calibration band",
      f"Random-encoder equivalence band **{list(CB.RANDOM_ENCODER_EQUIVALENCE_BAND)}**,",
      "declared before any estimate is produced. If the random encoder's mean paired",
      "difference falls outside it, ALL retrospective pure-bias estimates remain",
      "UNRESOLVED and no arm may be described as memorising."]
    open(S16+"C2_FEASIBILITY.md","w").write("\n".join(md)+"\n")
    json.dump(dict(per_fold=per_fold, cells=[dict(fold=k[0],site=k[1],label=k[2],**v)
              for k,v in sorted(agg.items())], overall_feasible=overall,
              n_checks=len(all_rows), n_infeasible=len(infeas), seeds=CB.SEEDS,
              band=list(CB.RANDOM_ENCODER_EQUIVALENCE_BAND),
              residual_confounds=CB.RESIDUAL_CONFOUNDS),
              open(S16+"out/C2_FEASIBILITY.json","w"), indent=1)
    print(f"cell checks {len(all_rows)} | infeasible {len(infeas)} | "
          f"VERDICT {'FEASIBLE' if overall else 'INFEASIBLE'}")
    for f in per_fold: print("  ", f)
    return 0 if overall else 4

if __name__ == "__main__": sys.exit(main())
