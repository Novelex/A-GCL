"""Winner control: per-subject ROI permutation, arm C seed 20260818, ordinary folds.
Permutation applied GLOBALLY at load (training AND extraction see identical arrays)."""
import sys, os, json, numpy as np, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s12a5/scripts"); import s12a5_core as M
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
import s11_core as K, s7_core as C7
from sklearn.metrics import roc_auc_score
dev="cuda" if torch.cuda.is_available() else "cpu"
df,X_fc,y,ids,gh=A1.load_gate(); FC,Xold,st=A1.load_tensors(df)
I90=np.eye(90,dtype=np.float32)
Xid=np.concatenate([Xold,np.repeat(I90[None],954,0)],axis=2)
rng=np.random.default_rng(M.BASE)
P=np.stack([rng.permutation(90) for _ in range(954)])
FCp=np.stack([FC[i][P[i]][:,P[i]] for i in range(954)])
Xidp=np.stack([Xid[i][P[i]] for i in range(954)])
Xep=np.stack([FCp[i][K.IU[0],K.IU[1]] for i in range(954)]).astype(np.float32)
oof=np.full(954,np.nan)
for i,(tr,te) in enumerate(K.folds_ordinary()):
    model,curve,info,init,_=M.train_fold5("C",M.BASE,np.asarray(tr),Xidp,FCp,Xep,y,dev,log=f"permroi/o{i}")
    R,Sc=M.extract5(model,Xidp,FCp,Xep,np.arange(954),dev)
    oof[te]=Sc[te]; print("fold",i,"done",flush=True)
auc=float(roc_auc_score(y,oof))
res=dict(winner="C",perm_roi_auc_head=auc,PASS=bool(auc<=0.55),
  note="per-subject ROI permutation of FC rows+cols; edge vector rebuilt from permuted FC",
  provenance=C7.provenance({"unit":"S12A5_permroi"}))
json.dump(res,open("/users/3171356m/agcl_audit_s0/s12a5/out/PERM_ROI.json","w"),indent=1,default=str)
print("PERM_ROI:",auc,"PASS" if res["PASS"] else "HIGH -- investigate")
