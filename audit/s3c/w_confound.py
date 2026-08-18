"""Level 4: how strongly does each representation encode SITE / TR / T / age / sex / motion?
No diagnosis label is used anywhere in this worker."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s3c")
import s3c_core as C
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeCV
from sklearn.metrics import balanced_accuracy_score, accuracy_score, r2_score
tid=int(sys.argv[1]); ntask=int(sys.argv[2])
S,y,ids=C.load(); meta=pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv")
COMBOS=[(s,b,n) for s in ["M1","M2","OLD"] for b in ["slow5","slow4","classical","all3"]
        for n in C.SPEC["normalizations"]]
mine=COMBOS[tid::ntask]; rows=[]
site=meta.site.values; sex=meta.sex.values
for (s,b,n) in mine:
    Xn=C.norm_subject(S[s],n); X,nb=C.featurize(Xn,b)
    r=dict(source=s,band=b,norm=n)
    # ---- SITE (19-class) ----
    pipe=Pipeline(([("pbs",C.PerBandStandardizer(nb))] if n=="F_TRAINFOLD_BAND_Z" else [])+
                  [("clf",LinearSVC(dual=False,max_iter=20000,random_state=C.SEED,C=1.0))])
    cv=StratifiedKFold(5,shuffle=True,random_state=C.SEED)
    pred=cross_val_predict(pipe,X,site,cv=cv,n_jobs=1)
    r["site_bacc"]=balanced_accuracy_score(site,pred); r["site_acc"]=accuracy_score(site,pred)
    r["site_bacc_chance"]=1.0/len(np.unique(site))
    # ---- SEX (binary) ----
    pred=cross_val_predict(pipe,X,sex,cv=cv,n_jobs=1)
    r["sex_bacc"]=balanced_accuracy_score(sex,pred)
    # ---- continuous confounds via RidgeCV (alpha chosen inside each training fold) ----
    for tgt in ["TR","T","age","func_mean_fd","func_dvars","func_perc_fd"]:
        t=meta[tgt].values.astype(float)
        rp=Pipeline(([("pbs",C.PerBandStandardizer(nb))] if n=="F_TRAINFOLD_BAND_Z" else [])+
                    [("clf",RidgeCV(alphas=np.logspace(-3,4,15)))])
        p=cross_val_predict(rp,X,t,cv=KFold(5,shuffle=True,random_state=C.SEED),n_jobs=1)
        r[f"r2_{tgt}"]=r2_score(t,p)
    rows.append(r); print("done",s,b,n,round(r["site_bacc"],4),flush=True)
pd.DataFrame(rows).to_csv(f"/users/3171356m/agcl_audit_s0/s3c/lvl4/conf_{tid:03d}.csv",index=False)
print("TASK COMPLETE",tid)
