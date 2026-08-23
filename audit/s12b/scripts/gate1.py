"""GATE 1 — instrument calibration. R1 ceiling probe, R2 frozen SVC harness,
R3 label-permutation leak check, R4 ALFF floor. Pass criteria pre-registered."""
import sys, os, json, time, numpy as np
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12b/scripts"); import s12b_core as B
import s11_core as K

NJ = int(os.environ.get("S11_NJOBS", "4"))

def run_probe(X, y, folds, tag, loso=True):
    ev = B.StageEval(y, XFC=None, conf=None, keep_proj=False)
    for t, tr, te in folds:
        if not loso and not t.startswith("o"): continue
        ev.fold_fit(X, tr, te, t, njobs=NJ)
        print(f"[{tag}] fold {t} done", flush=True)
    return ev.finalize()

def main():
    t0 = time.time()
    d = B.load_all(); y = d["y"]; F = B.folds_all(y)
    Xfc, _y, _ids, _m = K.load_Xfc()                       # frozen f64 source (review R2)
    assert np.array_equal(Xfc, d["X_fc"]) and np.array_equal(_y.astype(np.int64), y)
    Xalff = d["M1B"].reshape(954, -1).astype(np.float64)
    n_ord, n_loso = B.fold_counts(y)

    r1 = run_probe(Xfc, y, F, "R1")
    # Amendment A1: R3 = mean over 10 permutation draws; max draw guards leakage side
    r3_aucs = []
    for k in range(10):
        r3y = np.random.default_rng(B.BASE + k).permutation(y)
        assert (r3y == y).mean() < 1.0 and r3y.sum() == y.sum()
        rk = run_probe(Xfc, r3y, F, f"R3.{k}", loso=False)
        r3_aucs.append(rk["pooled_ordinary"]["auc"])
        if k == 0: r3 = rk                                  # keep one full block for record
    r4 = run_probe(Xalff, y, F, "R4")
    os.environ["S11_NJOBS"] = str(NJ)
    dsvc, _ = K.probe_pipe(Xfc, y, [(tr, te) for t, tr, te in F if t.startswith("o")], [])
    r2 = dsvc

    R1, R2 = r1["pooled_ordinary"]["auc"], r2["auc"]
    R3 = float(np.mean(r3_aucs)); R3max = float(np.max(r3_aucs))
    ok_r2 = 0.74 <= R2 <= 0.77
    ok_r1 = R1 >= R2 - 0.03
    ok_r3 = (0.47 <= R3 <= 0.53) and (R3max <= 0.55)       # Amendment A1
    verdict = "PASS" if (ok_r1 and ok_r2 and ok_r3) else "FAIL — STOP"
    res = dict(R1_ceiling_probe=r1, R2_svc=r2, R3_perm_first_draw=r3,
               R3_draw_aucs=r3_aucs, R3_mean=R3, R3_max=R3max, R4_alff=r4,
               CEILING_PROBE=R1, checks=dict(
                   R2_in_range=ok_r2, R1_ge_R2_minus_003=ok_r1,
                   R3_mean_in_range_and_max_le_055=ok_r3),
               verdict=verdict, wall_s=round(time.time() - t0, 1),
               provenance=B.provenance())
    B.atomic_json(res, B.S12B + "out/GATE1.json")
    md = [
        f"# S12B GATE 1 — INSTRUMENT CALIBRATION: **{verdict}**",
        "Probe (frozen for the audit): per-fold Scaler -> PCA(min(200,d,n-1), randomized,"
        " rs=20260818) -> LogisticRegression(l2, C by inner 5-fold CV on"
        f" {B.C_GRID}); threshold 0.5; bootstrap 2000.",
        "",
        f"| ref | pooled OOF AUC | 95% CI | LOSO AUC | criterion | ok |",
        f"|---|---|---|---|---|---|",
        f"| R1 raw FC probe (CEILING_PROBE) | {R1:.4f} |"
        f" [{r1['pooled_ordinary']['auc_ci_lo']:.4f},{r1['pooled_ordinary']['auc_ci_hi']:.4f}]"
        f" | {r1.get('pooled_loso',{}).get('auc',float('nan')):.4f} | >= R2-0.03 | {ok_r1} |",
        f"| R2 raw FC LinearSVC (frozen S11 path) | {R2:.4f} |"
        f" [{r2['ci_lo']:.4f},{r2['ci_hi']:.4f}] | — | in [0.74,0.77] | {ok_r2} |",
        f"| R3 labels permuted (mean of 10 draws, A1) | {R3:.4f} |"
        f" draws [{min(r3_aucs):.3f},{R3max:.3f}]"
        f" | — | mean in [0.47,0.53] & max<=0.55 | {ok_r3} |",
        f"| R4 ALFF(joint-minmax, 270) probe | {r4['pooled_ordinary']['auc']:.4f} |"
        f" [{r4['pooled_ordinary']['auc_ci_lo']:.4f},{r4['pooled_ordinary']['auc_ci_hi']:.4f}]"
        f" | {r4.get('pooled_loso',{}).get('auc',float('nan')):.4f} | floor (report) | — |",
        "",
        f"RETENTION RATIO denominator: CEILING_PROBE = {R1:.4f} (above-chance {R1-0.5:.4f}).",
        f"Old FC benchmark 0.7565 is NOT compared to Arm D directly; R1 defines the new"
        " measurement ceiling (validation rule 12).",
        f"- wall {time.time()-t0:.1f}s | folds (computed from the frozen authority):"
        f" {n_ord} ordinary + {n_loso} LOSO",
    ]
    B.atomic_text("\n".join(md) + "\n", B.S12B + "GATE1_INSTRUMENT.md")
    print("GATE1", verdict, "R1", R1, "R2", R2, "R3", R3, flush=True)
    assert verdict == "PASS", "GATE 1 FAILED — STOPPING"

if __name__ == "__main__":
    main()
