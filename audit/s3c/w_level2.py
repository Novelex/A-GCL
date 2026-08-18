import sys, json, itertools, time, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s3c")
import s3c_core as C
tid=int(sys.argv[1]); ntask=int(sys.argv[2]); NJ=int(sys.argv[3])
S,y,ids=C.load()
COMBOS=[(s,b,n,c) for s in ["M1","M2","OLD"] for b in ["slow5","slow4","classical","all3"]
        for n in C.SPEC["normalizations"] for c in ["linsvm","logreg"]]
mine=COMBOS[tid::ntask]
rows=[]; oofs={}
for (s,b,n,c) in mine:
    t=time.time(); r,p,oof=C.nested_cv(S[s],y,b,n,c,n_jobs=NJ)
    key=f"{s}|{b}|{n}|{c}"; oofs[key]=oof
    rows.append(dict(source=s,band=b,norm=n,clf=c,**{f"pooled_{k}":v for k,v in p.items()},
        auc_mean=float(np.mean([x["auc"] for x in r])), auc_sd=float(np.std([x["auc"] for x in r],ddof=1)),
        bacc_mean=float(np.mean([x["bacc"] for x in r])), bacc_sd=float(np.std([x["bacc"] for x in r],ddof=1)),
        acc_mean=float(np.mean([x["acc"] for x in r])), sens_mean=float(np.mean([x["sens"] for x in r])),
        spec_mean=float(np.mean([x["spec"] for x in r])), f1_mean=float(np.mean([x["f1"] for x in r])),
        prec_mean=float(np.mean([x["prec"] for x in r])),
        per_fold_auc=json.dumps([round(x["auc"],6) for x in r]),
        best_C=json.dumps([x["best_C"] for x in r]), secs=round(time.time()-t,1)))
    print("done",key,round(p["auc"],4),flush=True)
pd.DataFrame(rows).to_csv(f"/users/3171356m/agcl_audit_s0/s3c/lvl2/res_{tid:03d}.csv",index=False)
np.savez_compressed(f"/users/3171356m/agcl_audit_s0/s3c/lvl2/oof_{tid:03d}.npz", **oofs)
print("TASK COMPLETE",tid)
