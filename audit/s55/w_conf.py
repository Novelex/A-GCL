import sys, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s55")
import s55_core as C
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeCV
from sklearn.metrics import balanced_accuracy_score, accuracy_score, r2_score
F=C.build(); meta=pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv")
site=meta.site.values; rows=[]
for fs in ["FC_only","M1B_only","M1C_only","M1D_only","FC+M1D"]:
    X=C.FEATSETS[fs](F)
    pipe=Pipeline([("sc",StandardScaler()),("clf",LinearSVC(dual="auto",max_iter=20000,C=1.0,random_state=C.SEED))])
    pred=cross_val_predict(pipe,X,site,cv=StratifiedKFold(5,shuffle=True,random_state=C.SEED),n_jobs=4)
    r=dict(featset=fs,n_feat=X.shape[1],site_bacc=balanced_accuracy_score(site,pred),
           site_acc=accuracy_score(site,pred),site_chance=1/len(np.unique(site)))
    for tgt in ["TR","T","age","func_mean_fd"]:
        rp=Pipeline([("sc",StandardScaler()),("clf",RidgeCV(alphas=np.logspace(-2,5,15)))])
        p=cross_val_predict(rp,X,meta[tgt].values.astype(float),cv=KFold(5,shuffle=True,random_state=C.SEED),n_jobs=4)
        r[f"r2_{tgt}"]=r2_score(meta[tgt].values.astype(float),p)
    rows.append(r); print("done",fs,round(r["site_bacc"],4),flush=True)
pd.DataFrame(rows).to_csv("/users/3171356m/agcl_audit_s0/s55/s55_confound.csv",index=False)
print("COMPLETE")
