"""J3b - CPU leakage-safe linear probe over saved random-encoder embeddings."""
import sys, os, json, argparse, glob, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
ap=argparse.ArgumentParser()
ap.add_argument("--task",type=int,default=0); ap.add_argument("--ntask",type=int,default=1)
ap.add_argument("--smoke",action="store_true")
A=ap.parse_args()
TAG=os.environ.get("S7_SMOKE_TAG","smoke")
SRC=C.S7+(TAG+"/J3a" if A.smoke else "J3a"); OUT=C.S7+(TAG+"/J3b" if A.smoke else "J3b")
os.makedirs(OUT,exist_ok=True)
D=C.load_all(); FOLDS,_=C.splits(); meta=D["meta"]
GRID={"clf__C":[1e-4,1e-3,1e-2,1e-1,1,10,100]}
def probe(X,y,folds):
    oof=np.full(len(y),np.nan); fa=[]
    for tr,te in folds:
        nmin=int(np.bincount(y[tr]).min())
        gs=GridSearchCV(Pipeline([("sc",StandardScaler()),("clf",LinearSVC(dual="auto",max_iter=20000,random_state=C.BASE_SEED))]),
            GRID,cv=StratifiedKFold(min(5,max(2,nmin)),shuffle=True,random_state=C.BASE_SEED),scoring="roc_auc",n_jobs=1)
        gs.fit(X[tr],y[tr]); s=gs.decision_function(X[te]); oof[te]=s
        if len(np.unique(y[te]))>1: fa.append(roc_auc_score(y[te],s))
    return dict(auc=float(roc_auc_score(y,oof)) if len(np.unique(y))>1 else np.nan,
                bacc=float(balanced_accuracy_score(y,(oof>0).astype(int))),
                fold_sd=float(np.std(fa,ddof=1)) if len(fa)>1 else 0.0), oof
units=sorted(glob.glob(SRC+"/emb_*.DONE")); mine=units[A.task::A.ntask]
print(f"{len(units)} embeddings, this task {len(mine)}",flush=True)
rows=[]
for d in mine:
    base=os.path.basename(d)[:-5]; name="probe_"+base
    if C.is_done(OUT,name): print("skip",name,flush=True); continue
    z=np.load(SRC+"/"+base+".npz"); h=z["h"]; zz=z["z"]; y=z["y"].astype(int); si=z["subject_index"]
    _,pa,br=base.split("_")[1:4] if False else (None,base.split("_")[2],base.split("_")[3])
    seed=int(base.split("_")[1][1:])
    folds=FOLDS if len(y)==954 else [(np.setdiff1d(np.arange(len(y)),np.arange(i,len(y),3)),np.arange(i,len(y),3)) for i in range(3)]
    r={"seed":seed,"path":pa,"branch":br,"n":len(y)}
    for nm,M in (("h",h),("z",zz)):
        m,oof=probe(M,y,folds); r.update({f"{nm}_{k}":v for k,v in m.items()})
        if br=="B" and len(y)==954:                       # LOSO only for primary B
            site=meta.site.values[si]
            lo=[(np.where(site!=s)[0],np.where(site==s)[0]) for s in sorted(set(site))
                if len(np.unique(y[np.where(site==s)[0]]))>1]
            ml,_=probe(M,y,lo); r.update({f"{nm}_loso_{k}":v for k,v in ml.items()})
    rows.append(r); C.write_unit(OUT,name,payload_json=r); print("done",name,round(r["h_auc"],4),flush=True)
if rows: pd.DataFrame(rows).to_csv(OUT+f"/part_{A.task:03d}.csv",index=False)
print("J3b TASK COMPLETE",flush=True)
