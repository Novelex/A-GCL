"""Level 4B/4C: site-aware (grouped) and leave-one-site-out evaluation, same frozen design."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s3c")
import s3c_core as C
tid=int(sys.argv[1]); ntask=int(sys.argv[2]); NJ=int(sys.argv[3])
S,y,ids=C.load(); meta=pd.read_csv("/users/3171356m/agcl_audit_s0/s3c/meta.csv")
site=meta.site.values
COMBOS=[(s,b,n,c) for s in ["M1","M2","OLD"] for b in ["slow5","slow4","classical","all3"]
        for n in C.SPEC["normalizations"] for c in ["linsvm","logreg"]]
mine=COMBOS[tid::ntask]; rows=[]; oofs={}
loso=[(np.where(site!=st)[0],np.where(site==st)[0]) for st in C.SPEC["loso_sites"]]
for (s,b,n,c) in mine:
    r,p,oof=C.nested_cv(S[s],y,b,n,c,folds=C.GROUPED,n_jobs=NJ)
    lr,lp,loof=C.nested_cv(S[s],y,b,n,c,folds=loso,n_jobs=NJ)
    oofs[f"{s}|{b}|{n}|{c}|grouped"]=oof; oofs[f"{s}|{b}|{n}|{c}|loso"]=loof
    rows.append(dict(source=s,band=b,norm=n,clf=c,
      grouped_pooled_auc=p["auc"],grouped_auc_mean=float(np.mean([x["auc"] for x in r])),
      grouped_auc_sd=float(np.std([x["auc"] for x in r],ddof=1)),grouped_bacc=p["bacc"],
      loso_pooled_auc=lp["auc"],loso_auc_mean=float(np.mean([x["auc"] for x in lr])),
      loso_auc_sd=float(np.std([x["auc"] for x in lr],ddof=1)),loso_bacc=lp["bacc"],
      loso_per_site_auc=json.dumps({st:round(x["auc"],4) for st,x in zip(C.SPEC["loso_sites"],lr)})))
    print("done",s,b,n,c,round(p["auc"],4),round(lp["auc"],4),flush=True)
pd.DataFrame(rows).to_csv(f"/users/3171356m/agcl_audit_s0/s3c/lvl4/grp_{tid:03d}.csv",index=False)
np.savez_compressed(f"/users/3171356m/agcl_audit_s0/s3c/lvl4/oof_{tid:03d}.npz",**oofs)
print("TASK COMPLETE",tid)
