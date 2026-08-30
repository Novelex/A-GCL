"""Stage B array task: neural sweep on the top-3 Stage-A representations, each
alone and with mALFF-90 appended (6 inputs). Units = (protocol, fold, input, chunk
of 30 configs); 29 x 6 x 18 = 3,132 units packed CONTIGUOUSLY into <=1000 tasks so
a task mostly owns chunks of the same (protocol, fold, input) and can hold that
fold's representation features in process memory. Features are NEVER written to
disk and never shared across folds or tasks. Four single-thread trainings run
concurrently per 4-CPU task."""
import os, sys, json, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_lib as L, mlp_lib as M
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

OUT = L.ROOT + "B/"; NTASK = int(os.environ.get("SWEEP_NTASK_B", "1000"))
CHUNK = 30; NPAR = int(os.environ.get("SWEEP_NPAR", "4"))
STAGE_B_SMOKE = os.environ.get("SWEEP_SMOKE_GRID")          # e.g. "2" -> first 2 configs only

def inputs():
    """Spec inputs {FC, tangent FC, FC+ALFF-90, tangent FC+ALFF-90} realised as:
    flat FC and each of the top-3 Stage-A representations, each alone and with
    mALFF-90 appended (deduplicated). Top-3 are chosen on LAB by mean INNER score,
    never by outer-fold AUC. The list is frozen in B/inputs.json at first use."""
    agg = json.load(open(L.ROOT + "A/AGG_A.json"))
    reps = ["flat"] + [r for r in agg["lab"]["top3_reps_by_mean_inner"] if r != "flat"]
    return [(r, alff) for r in reps for alff in (False, True)]

def units():
    G = M.grid_B(); nchunk = (len(G) + CHUNK - 1) // CHUNK; U = []
    for p in L.protocols():
        for k in range(len(L.outer_folds(p))):
            for i, (rname, alff) in enumerate(inputs()):
                for c in range(nchunk): U.append((p, k, i, c))
    return U

def path(p, k, i, c): return f"{OUT}{p}/f{k}/in{i}/chunk{c:02d}"

class FoldFeatures:
    """Representation + scaler fitted on the TRAIN side of each split, in process only."""
    def __init__(self, p, k, rname, alff):
        D = L.data(); self.y = D["y"]; mats = D["FCsq"]; A = D["ALFF"]
        tag, tr, te = L.outer_folds(p)[k]; rep = next(r for r in L.rep_catalogue() if r.name() == rname)
        def build(a, b):
            r = L.Rep(rep.kind, rep.alpha, rep.k, rep.mean).fit(mats[a])
            Xa, Xb = r.transform(mats[a]), r.transform(mats[b])
            if alff: Xa, Xb = np.hstack([Xa, A[a]]), np.hstack([Xb, A[b]])
            sc = StandardScaler().fit(Xa); return sc.transform(Xa).astype(np.float32), sc.transform(Xb).astype(np.float32)
        self.inner = [(a, b) + build(a, b) for a, b in L.inner_splits(tr, self.y)]
        self.outer = (tr, te) + build(tr, te); self.tag = tag

def _train_eval(Xa, ya, Xb, cfg):
    m, info = M.train_mlp(np.vstack([Xa, Xb]), np.concatenate([ya, np.zeros(len(Xb), int)]), np.arange(len(Xa)), cfg, threads=1)
    return M.scores(m, Xb), info

def run_unit(p, k, i, c, feats):
    base = path(p, k, i, c); rname, alff = inputs()[i]
    G = M.grid_B()
    if STAGE_B_SMOKE: G = G[:int(STAGE_B_SMOKE)]
    cfgs = G[c * CHUNK:(c + 1) * CHUNK]
    if not cfgs: return "empty"
    if os.path.exists(base + ".json"):                        # skip ONLY a sealed, identity-matched output
        try:
            j = json.load(open(base + ".json")); z = np.load(base + ".npz")
            if (j["status"] == "OK" and np.array_equal(z["test_idx"], L.outer_folds(p)[k][2])
                    and j["rep"] == rname and j["alff"] == alff and j["configs"] == [M.cfg_id(cf) for cf in cfgs]
                    and j.get("npz_sha") == L.sha_file(base + ".npz")): return "skip"
        except Exception: pass
    y = feats.y
    # inner: 5 splits x |cfgs| trainings, 4 concurrent single-thread processes
    jobs = [(si, ci) for si in range(len(feats.inner)) for ci in range(len(cfgs))]
    res = Parallel(n_jobs=NPAR, backend="loky", max_nbytes=None)(       # no /tmp memmaps
        delayed(_train_eval)(feats.inner[si][2], y[feats.inner[si][0]], feats.inner[si][3], cfgs[ci]) for si, ci in jobs)
    inner_auc = {M.cfg_id(cf): [] for cf in cfgs}; inner_info = {M.cfg_id(cf): [] for cf in cfgs}
    for (si, ci), (sb, info) in zip(jobs, res):
        b = feats.inner[si][1]; inner_auc[M.cfg_id(cfgs[ci])].append(float(roc_auc_score(y[b], sb)))
        inner_info[M.cfg_id(cfgs[ci])].append(dict(best_epoch=info["best_epoch"], movement_max=info["movement_max"], clip_rate=info["clip_rate"], valid=info["valid"]))
    tr, te, Xtr, Xte = feats.outer
    res_o = Parallel(n_jobs=NPAR, backend="loky", max_nbytes=None)(delayed(_train_eval)(Xtr, y[tr], Xte, cf) for cf in cfgs)
    test_scores = {M.cfg_id(cf): so for cf, (so, _) in zip(cfgs, res_o)}
    outer_info = {M.cfg_id(cf): dict(best_epoch=inf["best_epoch"], epochs_run=inf["epochs_run"], movement=inf["movement"],
                                     movement_max=inf["movement_max"], clip_rate=inf["clip_rate"], valid=inf["valid"],
                                     n_params=inf["n_params"], secs=inf["secs"]) for cf, (_, inf) in zip(cfgs, res_o)}
    os.makedirs(os.path.dirname(base), exist_ok=True)
    sha = L.anpz(base + ".npz", test_idx=te.astype(int), **test_scores)
    L.aj(dict(status="OK", npz_sha=sha, protocol=p, fold=feats.tag, k_fold=k, input_idx=i, rep=rname, alff=alff, chunk=c,
              D_in=int(Xtr.shape[1]), n_tr=int(len(tr)), n_te=int(len(te)), configs=[M.cfg_id(cf) for cf in cfgs],
              inner_auc=inner_auc, inner_mean={k_: float(np.mean(v)) for k_, v in inner_auc.items()},
              inner_info=inner_info, outer_info=outer_info, policy_hash=M.POL.policy_hash()), base + ".json")
    return "ok"

def main():
    k = int(sys.argv[1]); U = units(); per = (len(U) + NTASK - 1) // NTASK; mine = U[k * per:(k + 1) * per]
    cur = [dict(rep=r, alff=a) for r, a in inputs()]
    if not os.path.exists(OUT + "inputs.json"): os.makedirs(OUT, exist_ok=True); L.aj(cur, OUT + "inputs.json")
    assert json.load(open(OUT + "inputs.json")) == cur, "B/inputs.json disagrees with AGG_A.json - Stage A changed under Stage B. STOP."
    print(f"task {k}/{NTASK}: {len(mine)} units (of {len(U)})", flush=True); rc = 0; memo = {}
    for (p, kf, i, c) in mine:
        t0 = time.time()
        try:
            key = (p, kf, i)
            if key not in memo: memo.clear(); rname, alff = inputs()[i]; memo[key] = FoldFeatures(p, kf, rname, alff)
            msg = run_unit(p, kf, i, c, memo[key])
        except Exception as e:
            rc = 1; msg = f"FAILED {e!r}"; os.makedirs(OUT + "failed", exist_ok=True)
            open(OUT + f"failed/{p}_f{kf}_in{i}_c{c}.txt", "w").write(traceback.format_exc())
        print(f"  {p} f{kf} in{i} chunk{c}: {msg} ({time.time()-t0:.0f}s)", flush=True)
    return rc

if __name__ == "__main__": sys.exit(main())
