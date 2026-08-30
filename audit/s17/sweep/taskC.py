"""Stage C array task: the SAME neural grid on mALFF-90 ALONE. Units = (protocol,
fold, chunk); 29 x 18 = 522 tasks. Blending is done in aggC after A and C finish."""
import os, sys, json, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_lib as L, mlp_lib as M, taskB as B
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
OUT = L.ROOT + "C/"; CHUNK = 30; NPAR = int(os.environ.get("SWEEP_NPAR", "4"))

def cfgs_now(c):
    G = M.grid_B()
    if os.environ.get("SWEEP_SMOKE_GRID"): G = G[:int(os.environ["SWEEP_SMOKE_GRID"])]
    return G[c * CHUNK:(c + 1) * CHUNK]

def units():
    n = (len(M.grid_B()) + CHUNK - 1) // CHUNK
    return [(p, k, c) for p in ("lab", "site", "loso") for k in range(len(L.outer_folds(p))) for c in range(n)]

def _main():
    k = int(sys.argv[1]); p, kf, c = units()[k]; base = f"{OUT}{p}/f{kf}/chunk{c:02d}"
    D = L.data(); y = D["y"]; A = D["ALFF"]; tag, tr, te = L.outer_folds(p)[kf]
    if os.path.exists(base + ".json"):
        try:
            j = json.load(open(base + ".json")); z = np.load(base + ".npz")
            if (j["status"] == "OK" and np.array_equal(z["test_idx"], te) and j["configs"] == [M.cfg_id(cf) for cf in cfgs_now(c)]
                    and j.get("npz_sha") == L.sha_file(base + ".npz")): print("skip"); return 0
        except Exception: pass
    cfgs = cfgs_now(c)
    def build(a, b):
        sc = StandardScaler().fit(A[a]); return sc.transform(A[a]).astype(np.float32), sc.transform(A[b]).astype(np.float32)
    inner = [(a, b) + build(a, b) for a, b in L.inner_splits(tr, y)]
    jobs = [(si, ci) for si in range(5) for ci in range(len(cfgs))]
    t0 = time.time()
    res = Parallel(n_jobs=NPAR, backend="loky", max_nbytes=None)(delayed(B._train_eval)(inner[si][2], y[inner[si][0]], inner[si][3], cfgs[ci]) for si, ci in jobs)
    inner_auc = {M.cfg_id(cf): [] for cf in cfgs}; inner_info = {M.cfg_id(cf): [] for cf in cfgs}
    for (si, ci), (sb, info) in zip(jobs, res):
        inner_auc[M.cfg_id(cfgs[ci])].append(float(roc_auc_score(y[inner[si][1]], sb)))
        inner_info[M.cfg_id(cfgs[ci])].append(dict(best_epoch=info["best_epoch"], movement_max=info["movement_max"], clip_rate=info["clip_rate"], valid=info["valid"]))
    Xtr, Xte = build(tr, te)
    res_o = Parallel(n_jobs=NPAR, backend="loky", max_nbytes=None)(delayed(B._train_eval)(Xtr, y[tr], Xte, cf) for cf in cfgs)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    sha = L.anpz(base + ".npz", test_idx=te.astype(int), **{M.cfg_id(cf): so for cf, (so, _) in zip(cfgs, res_o)})
    L.aj(dict(status="OK", npz_sha=sha, protocol=p, fold=tag, k_fold=kf, chunk=c, input="mALFF90", D_in=90,
              configs=[M.cfg_id(cf) for cf in cfgs], inner_auc=inner_auc,
              inner_mean={k_: float(np.mean(v)) for k_, v in inner_auc.items()}, inner_info=inner_info,
              outer_info={M.cfg_id(cf): dict(best_epoch=i_["best_epoch"], movement_max=i_["movement_max"], clip_rate=i_["clip_rate"], valid=i_["valid"], secs=i_["secs"]) for cf, (_, i_) in zip(cfgs, res_o)},
              secs=round(time.time() - t0, 1), policy_hash=M.POL.policy_hash()), base + ".json")
    print(f"ok {p} f{kf} chunk{c} {time.time()-t0:.0f}s"); return 0

def main():
    try: return _main()
    except Exception as e:
        k = int(sys.argv[1]); p, kf, c = units()[k]; os.makedirs(OUT + "failed", exist_ok=True)
        open(OUT + f"failed/{p}_f{kf}_c{c}.txt", "w").write(traceback.format_exc()); print(f"FAILED {e!r}"); return 1

if __name__ == "__main__": sys.exit(main())
