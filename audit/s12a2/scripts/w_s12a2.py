"""S12A2 worker: one unit = (arm, seed_idx, fold_kind). Resumable, atomic writes."""
import sys, os, json, time, hashlib, numpy as np, torch
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a2/scripts"); import s12a2_core as M
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
import s11_core as K, s7_core as C7

UNITS = ([("plumb", 0, "ord")] +
         [(a, s, "ord") for a in ("P", "F", "C", "X") for s in (0, 1, 2)] +
         [(a, s, "loso") for a in ("P", "F") for s in (0, 1, 2)])

def run_unit(arm, sidx, kind, outdir):
    tag = f"{arm}_s{sidx}_{kind}"
    fp = os.path.join(outdir, f"unit_{tag}.json")
    if os.path.exists(fp): print("DONE(skip)", tag); return
    t0 = time.time()
    df, X_fc, y, ids, gh = A1.load_gate()
    if arm == "plumb":                                    # harness identity check
        d, _ = K.probe_pipe(X_fc, y, K.folds_ordinary(), [])
        res = dict(auc=d["auc"], expected=0.7565, delta=abs(d["auc"] - 0.7565))
        extra = {}
    else:
        N = M.load_nodes(sidx)
        folds = K.folds_ordinary() if kind == "ord" else K.folds_loso(y)
        res, extra = M.rep_for_arm(arm, N, y, folds, M.BASE + sidx)
    out = dict(unit=tag, arm=arm, seed=M.BASE + sidx if arm != "plumb" else None,
               kind=kind, result=res, extra=extra, wall_s=round(time.time() - t0, 1),
               provenance=C7.provenance({"unit": f"S12A2_{tag}"}))
    tmp = fp + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1, default=str)
    assert json.load(open(tmp))["result"], "validate"
    os.replace(tmp, fp)
    print("DONE", tag, json.dumps(res)[:200], f"{out['wall_s']}s")

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("S12A2_THREADS", "1")))
    outdir = os.environ.get("S12A2_OUT", "/users/3171356m/agcl_audit_s0/s12a2/out")
    os.makedirs(outdir, exist_ok=True)
    i = int(sys.argv[1])
    arm, sidx, kind = UNITS[i]
    run_unit(arm, sidx, kind, outdir)
