"""S9 Step 4: frozen probes. One task = one representation x epoch (+ ROI-perm control)."""
import sys, os, json, argparse, numpy as np
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s75"); import s75_core as S75
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7");  import s7_core as C7
ap=argparse.ArgumentParser(); ap.add_argument("--task",type=int,required=True)
A=ap.parse_args()
S9="/users/3171356m/agcl_audit_s0/s9/"; OUT=S9+"out"
UNITS=[("pre_norm_nodes",0),("post_norm_nodes",0),("h",0),("z",0),
       ("pre_norm_nodes",200),("post_norm_nodes",200),("h",200),("z",200),
       ("pre_norm_nodes_ROIPERM",200)]
key,epoch=UNITS[A.task]
name=f"probe_{key}_ep{epoch:03d}"
if C7.is_done(OUT,name): print("skip"); sys.exit(0)
z=np.load(OUT+f"/embeddings_epoch{epoch:03d}.npz")
y=z["labels"].astype(int)
base=key.replace("_ROIPERM","")
M=z[base]
if key.endswith("ROIPERM"):
    rng=np.random.default_rng(20260818+9000)          # fixed, recorded permutation seed
    M=np.stack([M[i][rng.permutation(90)] for i in range(len(M))])
X=M.reshape(len(M),-1)
FOLDS,_=C7.splits()
m,oof=S75.probe(X,y,FOLDS,boot=2000)
fold_auc=[]
from sklearn.metrics import roc_auc_score
for tr,te in FOLDS:
    if len(np.unique(y[te]))>1: fold_auc.append(float(roc_auc_score(y[te],oof[te])))
m.update(representation=key,epoch=epoch,n_feat=X.shape[1],fold_auc=fold_auc,
         roiperm_seed=(20260818+9000 if key.endswith("ROIPERM") else None))
C7.write_unit(OUT,name,payload_json=dict(**m,provenance=C7.provenance({"unit":"S9_probe"})))
print(f"{name}: AUC={m['auc']:.4f} [{m['ci_lo']:.4f},{m['ci_hi']:.4f}] bacc={m['bacc']:.4f}",flush=True)
