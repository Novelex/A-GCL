"""Level 3 negative controls: label permutation + feature-column permutation.
Each permutation repeats the ENTIRE nested-CV logic (inner hyperparameter search included)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s3c")
import s3c_core as C
tid=int(sys.argv[1]); ntask=int(sys.argv[2]); NPERM=int(sys.argv[3]); NJ=int(sys.argv[4])
CONDS=json.load(open("/users/3171356m/agcl_audit_s0/s3c/perm_conditions.json"))
S,y,ids=C.load()
ci=tid % len(CONDS); chunk=tid // len(CONDS)
s,b,n,c = CONDS[ci]
nchunk = ntask // len(CONDS)
per = NPERM // nchunk
rows=[]
for j in range(per):
    seed = C.SEED + 100000*chunk + j
    rng=np.random.default_rng(seed)
    yp=rng.permutation(y)
    _,p,_=C.nested_cv(S[s],y=yp,band=b,norm=n,clf_name=c,n_jobs=NJ)
    # feature-column permutation control: shuffle each feature column independently
    # across subjects (keeps marginals, destroys subject-feature pairing), true labels kept
    X3=S[s].copy()
    for r_ in range(X3.shape[1]):
        for bb in range(X3.shape[2]):
            X3[:,r_,bb]=X3[rng.permutation(954),r_,bb]
    _,pf,_=C.nested_cv(X3,y=y,band=b,norm=n,clf_name=c,n_jobs=NJ)
    rows.append(dict(cond=f"{s}|{b}|{n}|{c}",seed=seed,perm_auc=p["auc"],perm_bacc=p["bacc"],
                     perm_acc=p["acc"],featperm_auc=pf["auc"],featperm_bacc=pf["bacc"]))
    if j%10==0: print("perm",j,round(p["auc"],4),round(pf["auc"],4),flush=True)
pd.DataFrame(rows).to_csv(f"/users/3171356m/agcl_audit_s0/s3c/lvl3/perm_{tid:04d}.csv",index=False)
print("TASK COMPLETE",tid)
