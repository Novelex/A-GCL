import sys, json, numpy as np, pandas as pd, time
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s55")
import s55_core as C
tid=int(sys.argv[1]); nt=int(sys.argv[2]); NJ=int(sys.argv[3]); MODE=sys.argv[4]
F=C.build(); y=F["y"]; meta=pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv")
site=meta.site.values
COMB=[(fs,cl) for fs in C.FEATSETS for cl in ["linsvm","logreg"]]
mine=COMB[tid::nt]; rows=[]; oofs={}
loso=[(np.where(site!=s)[0],np.where(site==s)[0]) for s in C.SPEC["loso_sites"]]
for fs,cl in mine:
    X=C.FEATSETS[fs](F); t=time.time()
    folds = C.OUTER if MODE=="cv" else loso
    r,p,oof=C.nested(X,y,cl,folds=folds,n_jobs=NJ)
    oofs[f"{fs}|{cl}"]=oof
    rows.append(dict(featset=fs,clf=cl,n_feat=X.shape[1],mode=MODE,
        pooled_auc=p["auc"],auc_mean=float(np.mean([x["auc"] for x in r])),
        auc_sd=float(np.std([x["auc"] for x in r],ddof=1)),
        bacc=p["bacc"],acc=p["acc"],sens=p["sens"],spec=p["spec"],f1=p["f1"],prec=p["prec"],
        per_fold_auc=json.dumps([round(x["auc"],5) for x in r]),
        best_C=json.dumps([x["best_C"] for x in r]),secs=round(time.time()-t,1)))
    print("done",MODE,fs,cl,round(p["auc"],4),flush=True)
pd.DataFrame(rows).to_csv(f"/users/3171356m/agcl_audit_s0/s55/{MODE}/res_{tid:02d}.csv",index=False)
np.savez_compressed(f"/users/3171356m/agcl_audit_s0/s55/{MODE}/oof_{tid:02d}.npz",**oofs)
print("TASK COMPLETE",tid)
