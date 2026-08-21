"""S10 probes: one task = one (arm,seed,representation). Frozen S3C splits, S7.5 probe."""
import sys, os, json, argparse, numpy as np
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s75"); import s75_core as S75
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7");  import s7_core as C7
ap=argparse.ArgumentParser(); ap.add_argument("--task",type=int,required=True); A=ap.parse_args()
S10="/users/3171356m/agcl_audit_s0/s10/"; OUT=S10+"out"
RUNS=[(a,s) for a in "ABCD" for s in (0,1,2) if not (a=="A" and s==0)]   # A_s0 reused from S9
REPS=["post_norm_nodes","h","z"]
UNITS=[(a,s,r) for (a,s) in RUNS for r in REPS]
arm,seed,rep=UNITS[A.task]
name=f"probe_{arm}_s{seed}_{rep}"
if C7.is_done(OUT,name): print("skip"); sys.exit(0)
z=np.load(OUT+f"/emb_{arm}_s{seed}.npz")
y=z["labels"].astype(int); X=z[rep].reshape(len(y),-1)
FOLDS,_=C7.splits()
m,oof=S75.probe(X,y,FOLDS,boot=2000)
from sklearn.metrics import roc_auc_score, accuracy_score
fa=[float(roc_auc_score(y[te],oof[te])) for tr,te in FOLDS if len(np.unique(y[te]))>1]
m.update(arm=arm,seed=seed,representation=rep,n_feat=X.shape[1],fold_auc=fa,
         acc=float(accuracy_score(y,(oof>0).astype(int))))
C7.write_unit(OUT,name,payload_json=dict(**m,provenance=C7.provenance({"unit":"S10_probe"})))
print(f"{name}: AUC={m['auc']:.4f}",flush=True)
