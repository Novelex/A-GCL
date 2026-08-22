"""Ordinary-CV probe units + plumbing SVM + LOSO units. One unit per task."""
import sys, os, json, argparse, numpy as np
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
import s7_core as C7
ap=argparse.ArgumentParser(); ap.add_argument("--task",type=int,required=True); Aa=ap.parse_args()
OUT="/users/3171356m/agcl_audit_s0/s12a1/out"
PRIM=["H1_preBN","H1_BN","H1_to_layer2"]; SEC=["H2_BN","final_postnorm","h","z"]
UNITS=[("plumbing","REC",None,"ord")]
UNITS+=[(c,st,s,"ord") for s in (0,1,2) for c in ("old","id") for st in PRIM+SEC]
UNITS+=[(c,"H1_BN",s,"loso") for s in (0,1,2) for c in ("old","id")]
cond,stage,seed,ev=UNITS[Aa.task]
name=f"probe_{cond}_{stage}_s{seed}_{ev}" if seed is not None else "probe_plumbing_fc_ord"
if C7.is_done(OUT,name): print("skip"); sys.exit(0)
if cond=="plumbing":
    z=np.load(OUT+"/plumbing_recovered_fc.npz"); X=z["REC"]; y=z["y"].astype(int)
else:
    z=np.load(OUT+f"/emb_{cond}_s{seed}.npz"); y=z["y"].astype(int)
    M=z[stage]; X=M.reshape(954,-1)
folds=K.folds_ordinary() if ev=="ord" else K.folds_loso(y)
m,oof=K.probe_pipe(X,y,folds,[],boot=2000)
m.update(condition=cond,stage=stage,seed=seed,eval=ev,n_feat=X.shape[1])
C7.write_unit(OUT,name,payload_json=dict(**m,provenance=C7.provenance({"unit":"S12A1_probe"})))
print(f"{name}: AUC={m['auc']:.4f}",flush=True)
