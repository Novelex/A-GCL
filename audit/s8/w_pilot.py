"""S8 Step 4: pilot training — ONE config (arg), seed 20260818, 200 epochs, full 954."""
import sys, os, json, time, argparse, numpy as np, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s8"); import s8_core as S
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s75"); import s75_core as S75
import s7_core as C7
from sklearn.metrics import confusion_matrix
ap=argparse.ArgumentParser(); ap.add_argument("--cfg",required=True); ap.add_argument("--epochs",type=int,default=200)
A=ap.parse_args()
OUT=S.S8+"out"; os.makedirs(OUT,exist_ok=True)
name=f"pilot_{A.cfg}"
if C7.is_done(OUT,name): print("already done"); sys.exit(0)
t0=time.time(); dl=S.load_dataset()
model,view,curves=S.train(A.cfg,seed=0,epochs=A.epochs,dl=dl,collect_every=10)
h,z,y=S.extract(model,dl,weighted=True)
huw,zuw,_=S.extract(model,dl,weighted=False)   # 08339b7 eval slip: edge weights silently dropped
FOLDS,_=C7.splits()
res={"provenance":C7.provenance({"unit":"S8_pilot","cfg":A.cfg,"epochs":A.epochs,
     "seed":S.BASE,"train_secs":round(time.time()-t0,1)}),"curves":curves}
for nm,M in (("h",h),("z",z),("h_unweighted_origslip",huw),("z_unweighted_origslip",zuw)):
    m,oof=S75.probe(M,y,FOLDS,boot=2000)
    yh=(oof>0).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
    m.update(acc=float((tp+tn)/len(y)),sens=float(tp/(tp+fn)),spec=float(tn/(tn+fp)))
    res[nm]=m
    print(f"{A.cfg} {nm}: AUC={m['auc']:.4f} [{m['ci_lo']:.4f},{m['ci_hi']:.4f}] "
          f"bacc={m['bacc']:.4f} sens={m['sens']:.4f} spec={m['spec']:.4f}",flush=True)
np.savez_compressed(OUT+f"/{name}_emb.npz",h=h,z=z,y=y)
C7.write_unit(OUT,name,payload_json=res)
print("PILOT COMPLETE",A.cfg,round(time.time()-t0,1),"s",flush=True)
