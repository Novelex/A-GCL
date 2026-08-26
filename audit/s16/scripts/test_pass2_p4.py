"""Pass 2, P4: the collector must use EACH unit's own E-level metadata.

Defect D35: audit() did `DAT.load("signed")` ONCE and validated every unit against
it, so h_fc and cache_file were structurally wrong for every abs / pos_zero / shift
cell — 810 of 1,431 (56.6%) would have been rejected as provenance failures, i.e.
every cell of the three non-signed E levels (3 of 4 levels).

This builds a REAL mixed-E fixture: all 159 units x 9 folds, complete 5-file sealed
bundles, in the isolated `test` namespace, and runs the REAL audit()."""
import sys, os, json, shutil, glob, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s16_prov as P, s16_policy as PL, s16_ledger as L, s16_grid as G, s16_collect as C
OK = []
def check(c, m): OK.append(bool(c)); print(f"  [{'PASS' if c else 'FAIL'}] {m}")

NS  = "test"
POL = PL.get(NS)
E_LEVELS = ("signed", "abs", "pos_zero", "shift")
# Synthetic per-E metadata: DISTINCT h_fc and cache_file per level, which is exactly
# what the real caches have and exactly what the single-load bug erased.
FAKE = {E: ({"h_alff": "h_alff", "h_labels": "h_lab", "h_subject_order": "h_ord",
             "h_folds_lab": "h_fl", "h_folds_site": "h_fs", "h_folds_loso": "h_fo"},
            {"h_fc": f"h_fc_{E}", "cache_file": f"fc_{E}.npz",
             "sparse": (E == "pos_zero")}) for E in E_LEVELS}

def build_fixture(root):
    for d in (P.jobs_dir(NS), P.feat_dir(NS), P.ckpt_dir(NS)): os.makedirs(d, exist_ok=True)
    cells, units, tags = L.expected_ledger()
    for uid, br, u in units:
        MAN, ent = FAKE[u["E"]]
        cfg = P.model_cfg(u); seed = G.SEEDS[u["seed_idx"]]
        jd = P.jobs_dir(NS) + uid; os.makedirs(jd, exist_ok=True)
        for tag in tags:
            fp  = P.feat_dir(NS) + f"{uid}__{tag}.npz"
            ckp = P.ckpt_dir(NS) + f"{uid}__{tag}.pt"
            prd = P.feat_dir(NS) + f"{uid}__{tag}.pred.json"
            res = jd + f"/fold_{tag}.json"
            np.savez_compressed(fp[:-4], repr=np.zeros((4, 4), np.float32))
            open(ckp, "wb").write(b"CKPT")
            rec = dict(status="OK",
                       unit=uid, fold=tag, namespace=NS, arm=u["arm"], E=u["E"],
                       arch=u["arch"], mode=u["mode"], seed=int(seed),
                       fold_protocol=tag.rstrip("0123456789"),
                       control=u.get("control"), alff_mode=u.get("alff_mode"),
                       kh=u["kh"], svm_tr_enc=0.75, svm_tr_full=0.75,
                       size_delta_paired=0.0, policy_hash=POL.policy_hash(),
                       policy_name=POL.name, ocread_entropy=float("nan"),
                       # complete evaluation set + the frozen evaluated_state, so the
                       # fixture satisfies the Pass-4 D55 per-cell contract exactly as
                       # a real worker record does
                       movement_max=0.5, clip_rate=0.05,
                       head=dict(auc=0.60), head_ema=dict(auc=0.59),
                       probe_honest=dict(auc=0.61), probe_old_full=dict(auc=0.62),
                       evaluated_state=("raw=validation-best checkpoint; EMA(0.999) "
                                        "reported alongside; selection by VALIDATION "
                                        "only (S15 PROTOCOL.md:186)"))
            # deterministic, self-consistent predictions so validate_fusion can
            # RECOMPUTE the fused AUC rather than trust the record
            import s16_feat as FT
            yv = np.array([0, 1, 0, 1]); sf = np.array([0.1, 0.9, 0.2, 0.8])
            from sklearn.metrics import roc_auc_score
            fauc = float(roc_auc_score(yv, sf))
            pred = {"unit": uid, "fold": tag, "label_used": yv.tolist()}
            if u["mode"] == "fused":
                pred["score_fused"] = sf.tolist()
                grid = [round(float(a), 4) for a in FT.ALPHA_GRID]
                inner = [{"alpha": a, "auc": 0.5} for a in grid]
                inner[-1]["auc"] = 0.9          # unique max -> tie-break reproducible
                rec["fusion"] = dict(
                    alpha_curve=[{"alpha": a, "auc": 0.5} for a in grid],
                    alpha_curve_inner=inner, alpha_selected=grid[-1],
                    fused_auc=fauc, alpha1_equals_svm_tr_enc=True,
                    alpha1_bitwise_equals_zsFC=True,
                    delta_vs_svm_tr_enc=fauc - rec["svm_tr_enc"],
                    delta_vs_svm_tr_full=fauc - rec["svm_tr_full"],
                    delta_is_unclamped=True)
            json.dump(pred, open(prd, "w"))
            json.dump({"rec": rec, "curve": []}, open(res, "w"))
            man = P.build_manifest(NS, u, cfg, uid, tag, seed,
                                   tag.rstrip("0123456789"), MAN, ent,
                                   P.expected_repr_dim(u["arch"], u["kh"], cfg["H"],
                                                       cfg["readout"]),
                                   fp, ckp, "OK", POL.train_consts(), policy=POL,
                                   result_path=res, pred_path=prd)
            man["worktree_clean"] = True
            P.atomic_json(man, fp + ".prov.json")
        json.dump(dict(unit=uid, namespace=NS, expected=len(tags), validated_reused=0,
                       newly_successful=len(tags), newly_attempted=len(tags),
                       failed=0, remaining=0), open(jd + "/TALLY.json", "w"))
        json.dump(dict(state="done"), open(jd + "/STATUS.json", "w"))
        open(jd + "/UNIT.done", "w").write("done")
    return cells, units, tags

root = P.root(NS)
if os.path.exists(root): shutil.rmtree(root)
try:
    cells, units, tags = build_fixture(root)
    byE = collections.Counter(u["E"] for _, _, u in units)
    print(f"fixture: {len(units)} units {dict(byE)} x {len(tags)} folds = {len(cells)} cells")

    print("\n=== 1. real audit() over the mixed-E fixture ===")
    prob, rows, tot = C.audit(ns=NS, data=FAKE)
    prov = {k: v for k, v in prob.items()
            if k in ("bundle_invalid", "wrong_namespace", "missing_unit",
                     "missing_fold", "identity_mismatch")}
    check(not prov, f"no provenance rejections (offending classes: "
                    f"{ {k: len(v) for k, v in prov.items()} })")
    check(len(rows) == len(cells), f"all {len(cells)} cells collected ({len(rows)} rows)")
    ns_rows = [r for r in rows if r[3]["E"] != "signed"]
    check(len(ns_rows) == 810, f"all 810 non-signed cells validated ({len(ns_rows)})")
    got = collections.Counter(r[3]["E"] for r in rows)
    check(all(got[E] == byE[E] * 9 for E in E_LEVELS), f"per-E counts exact: {dict(got)}")
    if prob: print("   other classes reported:", {k: len(v) for k, v in prob.items()})

    print("\n=== 2. the OLD single-signed behaviour must FAIL these same cells ===")
    # The buggy line bound the SIGNED metadata to every unit regardless of its E.
    # Reproduce that exactly — every level mapped to signed — so the comparison is
    # against the real old behaviour and never falls through to a real cache load.
    signed_for_all = {E: FAKE["signed"] for E in E_LEVELS}
    prob2, rows2, _ = C.audit(ns=NS, data=signed_for_all)
    bad = len(prob2.get("bundle_invalid", []))
    check(bad == 810, f"single-signed metadata rejects exactly 810 cells (got {bad}) "
                      f"— the defect is real and this fixture detects it")
    check(len(rows2) == 621, f"only the 621 signed cells survive (got {len(rows2)})")
    why = [w for w in prob2.get("bundle_invalid", []) if "h_fc" in w or "cache_file" in w]
    check(len(why) == 810, f"every rejection is an h_fc/cache_file mismatch ({len(why)})")
    if why: print(f"   e.g. {why[0][:130]}")
finally:
    if os.path.exists(root): shutil.rmtree(root)
    print(f"\nfixture removed: {root} exists={os.path.exists(root)}")

print(f"\n{sum(OK)}/{len(OK)} checks passed")
sys.exit(0 if all(OK) else 1)
