"""DEVIATION-01 Stage-D array task: confirmatory LOSO wave.

Evaluates ONLY the K=30 configs frozen in B/TOPK.json (selected on LAB+SITE
inner folds) across all 19 LOSO folds and all 8 frozen inputs. Units =
(fold, input); 19 x 8 = 152, one unit per array task.

Outputs are written into the SAME on-disk contract and the SAME tree as Stage B
(B/loso/f{k}/in{i}/chunkTK.*), so the aggregator consumes them as ordinary
Stage-B rows. Every record carries selection="DEVIATION_01_topk" so a LOSO row
can never be mistaken for a full-grid row.
"""
import os, sys, json, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_lib as L, mlp_lib as M
from taskB import FoldFeatures, _train_eval, inputs
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

OUT = L.ROOT + "B/"; NPAR = int(os.environ.get("SWEEP_NPAR", "4"))
PROTO = "loso"

def topk_cfgs():
    doc = json.load(open(OUT + "TOPK.json"))
    assert doc["deviation"] == "DEVIATION_01", "TOPK.json is not the pre-registered wave. STOP."
    ids = doc["configs"]; assert len(ids) == len(set(ids)) == doc["K"], "TOPK.json config list malformed. STOP."
    by = {M.cfg_id(c): c for c in M.grid_B()}
    missing = [i for i in ids if i not in by]
    assert not missing, f"TOPK config not in grid_B(): {missing}. STOP."
    return [by[i] for i in ids], ids

def units():
    return [(k, i) for k in range(len(L.outer_folds(PROTO))) for i in range(len(inputs()))]

def path(k, i): return f"{OUT}{PROTO}/f{k}/in{i}/chunkTK"

def run_unit(k, i, cfgs, ids):
    base = path(k, i); rname, alff = inputs()[i]
    if os.path.exists(base + ".json"):
        try:
            j = json.load(open(base + ".json")); z = np.load(base + ".npz")
            if (j["status"] == "OK" and np.array_equal(z["test_idx"], L.outer_folds(PROTO)[k][2])
                    and j["rep"] == rname and j["alff"] == alff and j["configs"] == ids
                    and j.get("selection") == "DEVIATION_01_topk"
                    and j.get("npz_sha") == L.sha_file(base + ".npz")): return "skip"
        except Exception: pass
    feats = FoldFeatures(PROTO, k, rname, alff); y = feats.y
    jobs = [(si, ci) for si in range(len(feats.inner)) for ci in range(len(cfgs))]
    res = Parallel(n_jobs=NPAR, backend="loky", max_nbytes=None)(
        delayed(_train_eval)(feats.inner[si][2], y[feats.inner[si][0]], feats.inner[si][3], cfgs[ci]) for si, ci in jobs)
    inner_auc = {c: [] for c in ids}; inner_info = {c: [] for c in ids}
    for (si, ci), (sb, info) in zip(jobs, res):
        b = feats.inner[si][1]; inner_auc[ids[ci]].append(float(roc_auc_score(y[b], sb)))
        inner_info[ids[ci]].append(dict(best_epoch=info["best_epoch"], movement_max=info["movement_max"],
                                        clip_rate=info["clip_rate"], valid=info["valid"]))
    tr, te, Xtr, Xte = feats.outer
    res_o = Parallel(n_jobs=NPAR, backend="loky", max_nbytes=None)(delayed(_train_eval)(Xtr, y[tr], Xte, cf) for cf in cfgs)
    test_scores = {c: so for c, (so, _) in zip(ids, res_o)}
    outer_info = {c: dict(best_epoch=inf["best_epoch"], epochs_run=inf["epochs_run"], movement=inf["movement"],
                          movement_max=inf["movement_max"], clip_rate=inf["clip_rate"], valid=inf["valid"],
                          n_params=inf["n_params"], secs=inf["secs"]) for c, (_, inf) in zip(ids, res_o)}
    os.makedirs(os.path.dirname(base), exist_ok=True)
    sha = L.anpz(base + ".npz", test_idx=te.astype(int), **test_scores)
    L.aj(dict(status="OK", npz_sha=sha, protocol=PROTO, fold=feats.tag, k_fold=k, input_idx=i, rep=rname, alff=alff,
              chunk="TK", selection="DEVIATION_01_topk",
              D_in=int(Xtr.shape[1]), n_tr=int(len(tr)), n_te=int(len(te)), configs=ids,
              inner_auc=inner_auc, inner_mean={c: float(np.mean(v)) for c, v in inner_auc.items()},
              inner_info=inner_info, outer_info=outer_info, policy_hash=M.POL.policy_hash()), base + ".json")
    return "ok"

def main():
    t = int(sys.argv[1]); U = units()
    if t >= len(U): print(f"task {t}: no unit (of {len(U)})", flush=True); return 0
    cfgs, ids = topk_cfgs(); k, i = U[t]
    print(f"task {t}/{len(U)}: loso f{k} in{i}, {len(cfgs)} configs", flush=True)
    t0 = time.time()
    try:
        msg = run_unit(k, i, cfgs, ids); rc = 0
    except Exception as e:
        rc = 1; msg = f"FAILED {e!r}"; os.makedirs(OUT + "failed", exist_ok=True)
        open(OUT + f"failed/{PROTO}_f{k}_in{i}_TK.txt", "w").write(traceback.format_exc())
    print(f"  loso f{k} in{i} TK: {msg} ({time.time()-t0:.0f}s)", flush=True)
    return rc

if __name__ == "__main__": sys.exit(main())
