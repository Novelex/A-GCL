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
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
from sklearn.metrics import roc_auc_score as _ras

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
    # W1b writes under a DISTINCT id so the Wave-1 records are never overwritten.
    pre = "s17b" if u.get("branch") == "s17w1b" else "s17"
    return f"{pre}_{u['arm']}{c}_{u['E']}_{u['mode']}_s{u['seed_idx']}"

def aj(o, p):
    json.dump(o, open(p + ".tmp", "w"), indent=1, default=str)
    json.load(open(p + ".tmp")); os.replace(p + ".tmp", p)

PROBE_WIDTHS = (None, 32, 64)          # None = the encoder's native repr_dim

def probe_at_widths(R, y, tr_enc, tr_prb, te, widths=PROBE_WIDTHS):
    """probe_honest at several representation widths.

    W1b measurement check. probe_honest fits on 153 tr_prb subjects, so a 2880-d
    RowMLP representation and a 32-d EdgeMLP representation are not being probed on
    equal terms: that is a 90x dimensionality asymmetry unrelated to representation
    quality. Reducing with PCA puts them on comparable footing.

    The PCA is fitted on tr_enc ONLY — the encoder has already seen those subjects,
    so it adds no information the encoder did not have, and tr_prb and te stay
    untouched. The probe still fits on tr_prb and scores te, both out-of-sample.

    Returns {label: metrics_or_None}. A width wider than the representation is
    recorded as None rather than silently falling back to the native width."""
    from sklearn.decomposition import PCA
    out = {}
    for w in widths:
        if w is None:
            _, oof = FT.probe_honest(R, y, tr_prb, te)
            out["native"] = dict(width=int(R.shape[1]),
                                 metrics=W16.metrics(y[te], oof[te]))
            continue
        if R.shape[1] <= w:
            out[f"pca{w}"] = None        # e.g. A7 at 32 dims has no PCA-64
            continue
        pca = PCA(n_components=w, random_state=0).fit(np.asarray(R, float)[tr_enc])
        Rw = pca.transform(np.asarray(R, float))
        _, oof = FT.probe_honest(Rw, y, tr_prb, te)
        out[f"pca{w}"] = dict(width=int(w),
                              evr=float(pca.explained_variance_ratio_.sum()),
                              metrics=W16.metrics(y[te], oof[te]))
    return out


def wave1b_units():
    """W1b: the SAME arms, seeds, folds and PROD policy as Wave 1. The ONLY change
    is that probe_honest is reported at three widths. No new arms, no gate change."""
    U = wave1_units()
    for u in U: u["branch"] = "s17w1b"
    return U


def wave1b_tasks(ns="prod"):
    pol = PL.get(ns)
    tags = ([f"lab{i}"  for i in range(pol.n_lab)]
            + [f"site{i}" for i in range(pol.n_site)]
            + [f"loso{i}" for i in range(pol.n_loso)])
    return [(u, t) for u in wave1b_units() for t in tags]


def wave1_units():
    """Wave 1 production grid: 4 arms x 3 seeds = 12 units, 9 folds each = 108 folds.

    A7 (EdgeMLP) is carried as the S16 REFERENCE arm so Branch R is judged against a
    contemporaneous baseline run under the identical policy, not against a number
    copied out of the C6 table."""
    import s16_grid as G
    U = []
    for arm in ("R1s", "R1a", "R1p", "A7"):
        for si, sd_ in enumerate(G.SEEDS):
            U.append(dict(branch="s17w1", arm=arm, E="signed", arch=G.ARCH[arm],
                          kh=G.KH[G.ARCH[arm]], mode="fused", seed_idx=si, seed=sd_,
                          alff_mode="z", control=None))
    return U


def wave1_tasks(ns="prod"):
    """Flatten to ONE TASK PER FOLD: 12 units x 9 folds = 108 array tasks."""
    pol = PL.get(ns)
    tags = ([f"lab{i}"  for i in range(pol.n_lab)]
            + [f"site{i}" for i in range(pol.n_site)]
            + [f"loso{i}" for i in range(pol.n_loso)])
    return [(u, t) for u in wave1_units() for t in tags]


def run_unit(u, ns="e2e", verbose=True, only_fold=None):
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
    Xfc, _y, _i, _m = K.load_Xfc()             # the frozen 4005-edge FC matrix
    arch = u["arch"]; spec = FT.ARMS[u["arm"]][1]; ctrl = u.get("control")
    cfg = P.model_cfg(u)                       # the SAME cfg builder S16 uses
    recs = []; t_unit = time.time()

    if only_fold is not None:
        folds = [f for f in folds if f[0] == only_fold]
        if not folds: raise ValueError(f"fold {only_fold!r} not in this policy")
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

        # ---- FC baseline and score-level fusion, using S16's frozen helpers.
        # W1_PLAN gate 3 ("adds value") is defined on the fused delta against THIS
        # fold's own svm_tr_enc, so it must be computed per fold, never pooled.
        s_fc, s_le = FT.scores_for_fusion(R, Xfc, y_use, tr_enc, tr_prb, te)
        svm_tr_enc = float(_ras(y_use[te], s_fc[te]))
        _df, _of = K.probe_pipe(Xfc.astype(np.float64), y_use,
                                [(np.asarray(tr), np.asarray(te))], [])
        svm_tr_full = float(_df["auc"])
        inner = [dict(alpha=float(a),
                      auc=float(_ras(y_use[tr_prb],
                                     FT.fuse_scores(s_fc, s_le, a, tr_prb)[tr_prb])))
                 for a in FT.ALPHA_GRID]
        # CONSERVATIVE TIE-BREAKING: highest inner AUC, and among ties the LARGEST
        # alpha, i.e. the most FC-favouring choice (S16 defect D7).
        a_sel = max(inner, key=lambda r: (r["auc"], r["alpha"]))["alpha"]
        f_sel = FT.fuse_scores(s_fc, s_le, a_sel, tr_prb)
        fused_auc = float(_ras(y_use[te], f_sel[te]))
        f1 = FT.fuse_scores(s_fc, s_le, 1.0, tr_prb)
        mu_, sd_ = FT.zfit(s_fc, tr_prb)
        fusion = dict(
            alpha_curve=[dict(alpha=float(a),
                              auc=float(_ras(y_use[te],
                                             FT.fuse_scores(s_fc, s_le, a, tr_prb)[te])))
                         for a in FT.ALPHA_GRID],
            alpha_curve_inner=inner, alpha_selected=float(a_sel), fused_auc=fused_auc,
            alpha1_auc=float(_ras(y_use[te], f1[te])),
            alpha1_equals_svm_tr_enc=bool(abs(float(_ras(y_use[te], f1[te]))
                                              - svm_tr_enc) < 1e-12),
            alpha1_bitwise_equals_zsFC=bool(np.array_equal(f1[te],
                                            FT.zapply(s_fc, mu_, sd_)[te])),
            delta_vs_svm_tr_enc=float(fused_auc - svm_tr_enc),
            delta_vs_svm_tr_full=float(fused_auc - svm_tr_full),
            delta_is_unclamped=True,
            endpoint_semantics=("alpha=1 is the FC FALLBACK ENDPOINT, not a "
                                "guaranteed floor; the outer-test delta may be "
                                "negative and is reported as measured"))
        rec = dict(status="OK", namespace=ns, unit=uid, fold=tag,
                   fold_protocol=tag.rstrip("0123456789"),
                   arm=u["arm"], arch=arch, E=u["E"], mode=u["mode"], control=ctrl,
                   alff_mode=u.get("alff_mode", "z"), kh=u["kh"], seed=int(u["seed"]),
                   spec=spec, D_in=int(X.shape[-1]), repr_dim_used=int(R.shape[1]),
                   head=W16.metrics(y_use[te], S[te]),
                   probe_honest=W16.metrics(y_use[te], ph[te]),
                   probe_widths=(probe_at_widths(R, y_use, tr_enc, tr_prb, te)
                                 if u.get("branch") == "s17w1b" else None),
                   n_tr=int(len(tr)), n_tr_enc=int(len(tr_enc)), n_tr_probe=int(len(tr_prb)),
                   n_te=int(len(te)),
                   svm_tr_enc=svm_tr_enc, svm_tr_full=svm_tr_full,
                   size_delta_paired=float(svm_tr_full - svm_tr_enc), fusion=fusion,
                   **{k: v for k, v in info.items() if k != "movement"},
                   movement=info["movement"],
                   h_fc=ent["h_fc"], cache_file=ent["cache_file"],
                   node=socket.gethostname(), wall_s=round(time.time() - t0, 1))
        aj(dict(rec=rec, curve=curve), f"{jd}/fold_{tag}.json")
        recs.append(rec)
        if verbose:
            print(f"  [{uid}/{tag}] head {rec['head']['auc']:.4f} "
                  f"probe {rec['probe_honest']['auc']:.4f} "
                  f"svm {svm_tr_enc:.4f} fused {fused_auc:.4f} "
                  f"d {fusion['delta_vs_svm_tr_enc']:+.4f} a={a_sel:.2f} "
                  f"ep {info['best_epoch']} mv {info['movement_max']:.3f} "
                  f"{rec['wall_s']}s", flush=True)

    if only_fold is None:
        # UNIT-level completion is claimed ONLY when the whole unit ran in this
        # process. In per-fold array mode each task owns one fold, so the unit
        # tally is written by the collector, never by a single task.
        aj(dict(unit=uid, namespace=ns, expected=len(folds),
                newly_successful=len(recs), validated_reused=0, failed=0,
                remaining=0, wall_s=round(time.time() - t_unit, 1)),
           f"{jd}/TALLY.json")
        aj(dict(state="done", folds_done=len(recs), folds_total=len(folds)),
           f"{jd}/STATUS.json")
        open(f"{jd}/UNIT.done", "w").write("done")
    return recs
