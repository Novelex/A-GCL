"""S17 worker: executes ONE Wave-1 unit, writing ONLY under audit/s17/runs/.

S16's worker cannot be reused: s16_prov.root() resolves every namespace under
audit/s16/runs/, and S17 must never write there. This module therefore owns its own
namespace root and reuses S16 for everything scientific — data, folds, the honest
split, the frozen ExecPolicy, train_fold, extract, probe_honest and the metric
block are all imported, never reimplemented.
"""
import os, sys, json, time, socket, hashlib
import numpy as np, torch

S16S = "/users/3171356m/A-GCL/audit/s16/scripts"
S17  = "/users/3171356m/A-GCL/audit/s17/"
sys.path.insert(0, S16S); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import s16_data as DAT, s16_feat as FT, s16_train as TR, s16_models as MO
import s16_policy as PL, s16_prov as P
import s16_worker as W16                      # for the frozen metric block only
import s17_feat as F17                        # noqa: F401 (registers the specs)

NAMESPACES = ("e2e", "prod")

def root(ns):
    if ns not in NAMESPACES: raise ValueError(f"unknown S17 namespace {ns!r}")
    return f"{S17}runs/{ns}/"                 # NEVER audit/s16/runs/
def jobs_dir(ns): return root(ns) + "jobs/"
def feat_dir(ns): return root(ns) + "feat/"
def ensure(ns):
    for d in (jobs_dir(ns), feat_dir(ns)): os.makedirs(d, exist_ok=True)
    assert "/audit/s17/" in root(ns) and "/audit/s16/" not in root(ns)
    return root(ns)

def unit_id(u):
    c = f"_{u['control']}" if u.get("control") else ""
    return f"s17_{u['arm']}{c}_{u['E']}_{u['mode']}_s{u['seed_idx']}"

def aj(o, p):
    json.dump(o, open(p + ".tmp", "w"), indent=1, default=str)
    json.load(open(p + ".tmp")); os.replace(p + ".tmp", p)

def run_unit(u, ns="e2e", verbose=True):
    """Execute every fold of one S17 unit. Returns the list of fold records."""
    policy = PL.get(ns)                        # FROZEN S16 policy; nothing invented
    assert policy.namespace == ns
    ensure(ns)
    uid = unit_id(u); jd = jobs_dir(ns) + uid; os.makedirs(jd, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    d, MAN, ent = DAT.load(u["E"], where=uid)
    FC, ALFF, y_true = d["FC"], d["ALFF"], d["y"].astype(np.int64)
    folds = (DAT.folds(d, "lab")[:policy.n_lab] + DAT.folds(d, "site")[:policy.n_site]
             + DAT.folds(d, "loso")[:policy.n_loso])
    arch = u["arch"]; spec = FT.ARMS[u["arm"]][1]; ctrl = u.get("control")
    cfg = P.model_cfg(u)                       # the SAME cfg builder S16 uses
    recs = []; t_unit = time.time()

    for tag, tr, te in folds:
        t0 = time.time()
        y_use = y_true.copy()
        if ctrl == "C-PERM":                   # permute labels within the train side
            rng = np.random.default_rng(DAT.BASE + 91); p_ = np.asarray(tr).copy()
            y_use[p_] = y_use[rng.permutation(p_)]
        tr_enc, tr_prb = FT.honest_split(tr, y_use)
        X, FCu = FT.build_X(spec, FC, ALFF, tr_enc, control=ctrl,
                            alff_mode=u.get("alff_mode", "z"))
        m, ema_sd, curve, info = TR.train_fold(arch, X, FCu, y_use, tr_enc, cfg,
                                               u["seed"], policy=policy)
        need_graph = (arch == "WGIN")
        R, S = TR.extract(m, X, FCu, np.arange(len(y_use)), need_graph)
        _, ph = FT.probe_honest(R, y_use, tr_prb, te)
        rec = dict(status="OK", namespace=ns, unit=uid, fold=tag,
                   fold_protocol=tag.rstrip("0123456789"),
                   arm=u["arm"], arch=arch, E=u["E"], mode=u["mode"], control=ctrl,
                   alff_mode=u.get("alff_mode", "z"), kh=u["kh"], seed=int(u["seed"]),
                   spec=spec, D_in=int(X.shape[-1]), repr_dim_used=int(R.shape[1]),
                   head=W16.metrics(y_use[te], S[te]),
                   probe_honest=W16.metrics(y_use[te], ph[te]),
                   n_tr=int(len(tr)), n_tr_enc=int(len(tr_enc)), n_tr_probe=int(len(tr_prb)),
                   n_te=int(len(te)),
                   **{k: v for k, v in info.items() if k != "movement"},
                   movement=info["movement"],
                   h_fc=ent["h_fc"], cache_file=ent["cache_file"],
                   node=socket.gethostname(), wall_s=round(time.time() - t0, 1))
        aj(dict(rec=rec, curve=curve), f"{jd}/fold_{tag}.json")
        recs.append(rec)
        if verbose:
            print(f"  [{uid}/{tag}] head {rec['head']['auc']:.4f} "
                  f"probe {rec['probe_honest']['auc']:.4f} ep {info['best_epoch']} "
                  f"mv {info['movement_max']:.3f} {rec['wall_s']}s", flush=True)

    aj(dict(unit=uid, namespace=ns, expected=len(folds), newly_successful=len(recs),
            validated_reused=0, failed=0, remaining=0,
            wall_s=round(time.time() - t_unit, 1)), f"{jd}/TALLY.json")
    aj(dict(state="done", folds_done=len(recs), folds_total=len(folds)),
       f"{jd}/STATUS.json")
    open(f"{jd}/UNIT.done", "w").write("done")
    return recs
