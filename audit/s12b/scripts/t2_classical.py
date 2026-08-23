"""S12B Track 2 classical arms on raw 4005 FC: LinearSVC (frozen S11 harness),
ridge-logistic (probe C-grid, no PCA), elastic-net logistic (saga)."""
import sys, os, time, numpy as np
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12b/scripts"); import s12b_core as B
import s11_core as K
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

NJ = int(os.environ.get("S11_NJOBS", "8"))

def lr_probe(X, y, folds, grid, tag):
    oof_o = np.full(954, np.nan); oof_l = np.full(954, np.nan); pf = {}
    for t, tr, te in folds:
        pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])
        gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=B.BASE),
                          scoring="roc_auc", n_jobs=NJ)
        gs.fit(X[tr], y[tr]); p = gs.predict_proba(X[te])[:, 1]
        (oof_o if t.startswith("o") else oof_l)[te] = p
        pf[t] = dict(best=gs.best_params_, **B.metric_block(y[te], p))
        print(tag, t, "done", flush=True)
    co, cl = np.isfinite(oof_o), np.isfinite(oof_l)
    return dict(pooled_ordinary=B.metric_block(y[co], oof_o[co]),
                pooled_loso=B.metric_block(y[cl], oof_l[cl]), per_fold=pf)

def main():
    jp = B.S12B + "jobs/t2_classical.json"
    if os.path.exists(jp): print("skip"); return
    d = B.load_all(); y = d["y"]; X = d["X_fc"].astype(np.float64)
    folds = B.folds_all(y); t0 = time.time()
    os.environ.setdefault("S11_NJOBS", str(NJ))
    svc, _ = K.probe_pipe(X, y, [(tr, te) for t, tr, te in folds if t.startswith("o")], [])
    svc_l, _ = K.probe_pipe(X, y, [(tr, te) for t, tr, te in folds if t.startswith("l")], [])
    ridge = lr_probe(X, y, folds, {"clf__C": B.C_GRID}, "ridgeLR")
    enet = lr_probe(X, y, folds,
                    {"clf__C": [1e-3, 1e-2, 1e-1, 1, 10],
                     "clf__l1_ratio": [0.2, 0.5, 0.8],
                     "clf__penalty": ["elasticnet"], "clf__solver": ["saga"]},
                    "enetLR")
    B.atomic_json(dict(linear_svc=dict(ordinary=svc, loso=svc_l), ridge_logistic=ridge,
                       elasticnet_logistic=enet, wall_s=round(time.time() - t0, 1),
                       provenance=B.provenance()), jp)
    print("DONE classical", svc["auc"], ridge["pooled_ordinary"]["auc"],
          enet["pooled_ordinary"]["auc"], flush=True)

if __name__ == "__main__":
    main()
