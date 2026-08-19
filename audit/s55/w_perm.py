import sys, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s55")
import s55_core as C
tid=int(sys.argv[1]); nt=int(sys.argv[2]); NPERM=int(sys.argv[3]); NJ=int(sys.argv[4])
CONDS=[("FC_only","linsvm"),("M1D_only","linsvm"),("FC+M1D","linsvm"),("FC_only","logreg")]
F=C.build(); y=F["y"]
ci=tid%len(CONDS); chunk=tid//len(CONDS); nchunk=nt//len(CONDS); per=NPERM//nchunk
fs,cl=CONDS[ci]; X=C.FEATSETS[fs](F)
rows=[]
for j in range(per):
    seed=C.SEED+100000*chunk+j; rng=np.random.default_rng(seed)
    _,p,_=C.nested(X,rng.permutation(y),cl,n_jobs=NJ)
    rows.append(dict(cond=f"{fs}|{cl}",seed=seed,perm_auc=p["auc"],perm_bacc=p["bacc"]))
    if j%5==0: print("perm",j,round(p["auc"],4),flush=True)
pd.DataFrame(rows).to_csv(f"/users/3171356m/agcl_audit_s0/s55/perm/perm_{tid:03d}.csv",index=False)
print("TASK COMPLETE",tid)
