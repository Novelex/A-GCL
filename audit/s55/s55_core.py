"""S5.5 classical calibration core. Frozen S3C splits, leakage-safe pipelines."""
import numpy as np, pandas as pd, json, hashlib, os, scipy.io as sio
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, accuracy_score,
                             recall_score, f1_score, precision_score, confusion_matrix)
S3C="/users/3171356m/agcl_audit_s0/s3c/"; A="/users/3171356m/A-GCL/"
CACHE="/users/3171356m/agcl_audit_s0/s55/features.npz"
SEED=20260818
_S=json.load(open(S3C+"splits.json")); SPEC=_S["spec"]
OUTER=[(np.array(f["train"]),np.array(f["test"])) for f in _S["outer_folds"]]
GRID={"clf__C":[1e-4,1e-3,1e-2,1e-1,1,10,100]}

def build():
    if os.path.exists(CACHE):
        d=np.load(CACHE,allow_pickle=True); return {k:d[k] for k in d.files}
    coh=pd.read_csv(S3C.replace("s3c/","")+"s1_audit_table.csv")
    ids=coh.subject_id.tolist()
    h=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    assert h=="aca3d945f7d89ccb1cc6fb46ca01f6036ccf036d81d120d2b09fe4bd0d1d68c9","cohort drift"
    Z=np.load(S3C+"X_sources.npz",allow_pickle=True)
    assert [str(s) for s in Z["ids"]]==ids, "order drift vs S3C"
    RAW=Z["M1"].astype(np.float64); y=Z["y"]
    def nb(X,k):
        if k=="B": mn=X.min((1,2),keepdims=True); mx=X.max((1,2),keepdims=True); return (X-mn)/(mx-mn)
        if k=="C": mn=X.min(1,keepdims=True); mx=X.max(1,keepdims=True); return (X-mn)/(mx-mn)
        if k=="D": return (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)
    dx=dict(zip(coh.subject_id,coh.dx_storage))
    iu=np.triu_indices(90,k=1)                     # 4005 unique OFF-DIAGONAL edges
    off=~np.eye(90,dtype=bool)                     # 8010 full symmetric off-diagonal
    FCu=np.zeros((954,4005)); FCf=np.zeros((954,8010))
    for i,s in enumerate(ids):
        M=sio.loadmat(f"{A}data/raw/{'ASD' if dx[s]=='ASD' else 'NC'}_ADJ/{s}_adj.mat")["cropped_matrix"]
        FCu[i]=M[iu]; FCf[i]=M[off]
    out=dict(y=y, ids=np.array(ids), FC=FCu, FC_full=FCf)
    for k in "BCD": out[f"M1{k}"]=nb(RAW,k).reshape(954,-1)
    np.savez_compressed(CACHE,**out); return out

FEATSETS={
 "FC_only":        lambda F: F["FC"],
 "M1B_only":       lambda F: F["M1B"],
 "M1C_only":       lambda F: F["M1C"],
 "M1D_only":       lambda F: F["M1D"],
 "FC+M1B":         lambda F: np.hstack([F["FC"],F["M1B"]]),
 "FC+M1C":         lambda F: np.hstack([F["FC"],F["M1C"]]),
 "FC+M1D":         lambda F: np.hstack([F["FC"],F["M1D"]]),
 "CTRL_FCfull":    lambda F: F["FC_full"],          # control only, not primary
}
def make_pipe(clf):
    c = LinearSVC(dual="auto",max_iter=20000,random_state=SEED) if clf=="linsvm" else \
        LogisticRegression(penalty="l2",solver="lbfgs",max_iter=3000,random_state=SEED)
    # StandardScaler INSIDE the pipeline -> refit on every inner/outer training fold only
    return Pipeline([("sc",StandardScaler()),("clf",c)])

def metrics(y,s,yh):
    tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
    return dict(auc=roc_auc_score(y,s),bacc=balanced_accuracy_score(y,yh),acc=accuracy_score(y,yh),
        sens=recall_score(y,yh,zero_division=0),spec=tn/(tn+fp) if tn+fp else np.nan,
        f1=f1_score(y,yh,zero_division=0),prec=precision_score(y,yh,zero_division=0))

def nested(X,y,clf,folds=None,n_jobs=1):
    folds=OUTER if folds is None else folds
    oof=np.full(len(y),np.nan); rows=[]
    for k,(tr,te) in enumerate(folds):
        gs=GridSearchCV(make_pipe(clf),GRID,cv=StratifiedKFold(5,shuffle=True,random_state=SEED),
                        scoring="roc_auc",n_jobs=n_jobs,refit=True)
        gs.fit(X[tr],y[tr]); s=gs.decision_function(X[te]); oof[te]=s
        m=metrics(y[te],s,(s>0).astype(int)); m.update(fold=k,best_C=gs.best_params_["clf__C"])
        rows.append(m)
    return rows, metrics(y,oof,(oof>0).astype(int)), oof
