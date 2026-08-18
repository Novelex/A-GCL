"""S3C shared core. Frozen splits, leakage-safe normalization, nested CV."""
import numpy as np, json, hashlib
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, accuracy_score,
                             recall_score, f1_score, precision_score, confusion_matrix)
OUT="/users/3171356m/agcl_audit_s0/s3c/"
SEED=20260818
_S=json.load(open(OUT+"splits.json"))
SPEC=_S["spec"]; OUTER=[(np.array(f["train"]),np.array(f["test"])) for f in _S["outer_folds"]]
GROUPED=[(np.array(f["train"]),np.array(f["test"])) for f in _S["grouped_folds"]]
GRID={"clf__C":SPEC["grid"]["C"]}
BANDS={"slow5":[0],"slow4":[1],"classical":[2],"all3":[0,1,2]}

def load():
    d=np.load(OUT+"X_sources.npz",allow_pickle=True)
    return {k:d[k] for k in ("M1","M2","OLD")}, d["y"], [str(s) for s in d["ids"]]

# ---------- per-subject normalizations (leakage-free by construction) ----------
def norm_subject(X, kind):
    if kind=="A_RAW":  return X.copy()
    if kind=="B_SUBJ_JOINT_MINMAX":
        mn=X.min((1,2),keepdims=True); mx=X.max((1,2),keepdims=True); return (X-mn)/(mx-mn)
    if kind=="C_SUBJ_BAND_MINMAX":
        mn=X.min(1,keepdims=True); mx=X.max(1,keepdims=True); return (X-mn)/(mx-mn)
    if kind=="D_SUBJ_BAND_Z":
        return (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)
    if kind=="F_TRAINFOLD_BAND_Z": return X.copy()      # handled inside the pipeline
    raise ValueError(kind)

class PerBandStandardizer(BaseEstimator, TransformerMixin):
    """Per-band z-score with mean/std pooled over (train samples x ROIs) of that band.
    Lives INSIDE the pipeline so GridSearchCV refits it on every inner training fold."""
    def __init__(self, n_bands=3): self.n_bands=n_bands
    def fit(self, X, y=None):
        n,f=X.shape; b=self.n_bands; r=f//b
        Z=X.reshape(n,r,b); self.mu_=Z.mean((0,1)); self.sd_=Z.std((0,1)); self.sd_[self.sd_==0]=1.0
        return self
    def transform(self, X):
        n,f=X.shape; b=self.n_bands; r=f//b
        return ((X.reshape(n,r,b)-self.mu_)/self.sd_).reshape(n,f)

def make_pipe(clf_name, norm, n_bands):
    steps=[]
    if norm=="F_TRAINFOLD_BAND_Z": steps.append(("pbs",PerBandStandardizer(n_bands=n_bands)))
    clf = LinearSVC(dual=False,max_iter=20000,random_state=SEED) if clf_name=="linsvm" \
          else LogisticRegression(penalty="l2",solver="lbfgs",max_iter=5000,random_state=SEED)
    steps.append(("clf",clf))
    return Pipeline(steps)

def featurize(X3, band):
    idx=BANDS[band]; return X3[:,:,idx].reshape(X3.shape[0],-1), len(idx)

def metrics(y, s, yhat):
    tn,fp,fn,tp=confusion_matrix(y,yhat,labels=[0,1]).ravel()
    return dict(auc=roc_auc_score(y,s), bacc=balanced_accuracy_score(y,yhat),
        acc=accuracy_score(y,yhat), sens=recall_score(y,yhat,zero_division=0),
        spec=tn/(tn+fp) if (tn+fp) else np.nan, f1=f1_score(y,yhat,zero_division=0),
        prec=precision_score(y,yhat,zero_division=0), tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def nested_cv(X3, y, band, norm, clf_name, folds=None, n_jobs=1):
    """Returns per-fold metrics + out-of-fold decision scores. Test never touches any fit."""
    folds = OUTER if folds is None else folds
    Xn = norm_subject(X3, norm)
    X, nb = featurize(Xn, band)
    oof=np.full(len(y),np.nan); oofp=np.full(len(y),np.nan); rows=[]
    for k,(tr,te) in enumerate(folds):
        gs=GridSearchCV(make_pipe(clf_name,norm,nb), GRID,
                        cv=StratifiedKFold(5,shuffle=True,random_state=SEED),
                        scoring="roc_auc", n_jobs=n_jobs, refit=True)
        gs.fit(X[tr],y[tr])
        s=gs.decision_function(X[te]); yh=(s>0).astype(int)
        oof[te]=s; oofp[te]=yh
        m=metrics(y[te],s,yh); m.update(fold=k,best_C=gs.best_params_["clf__C"],
                                        inner_best_auc=float(gs.best_score_))
        rows.append(m)
    pooled=metrics(y,oof,oofp.astype(int))
    return rows, pooled, oof
