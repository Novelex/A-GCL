"""Step 4: epoch-0 (init) flatten probe, arm1 seeds. Init reconstructed deterministically
(same seeded constructor as S12A4; no training performed — audit only)."""
import sys, json, numpy as np, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s12a4/scripts"); import s12a4_core as M
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
import s11_core as K
sid=int(sys.argv[1])
df,X_fc,y,ids,gh=A1.load_gate(); FC,Xold,st=A1.load_tensors(df)
I90=np.eye(90,dtype=np.float32)
Xid=np.concatenate([Xold,np.repeat(I90[None],954,0)],axis=2)
torch.set_num_threads(2)
m=M.ROIModel(20260818+sid, with_head=True); m.eval()
H,Z,ND=M.extract_all(m,Xid,FC,"cpu")
assert np.isfinite(ND).all()
d,_=K.probe_pipe(ND.astype(np.float64),y,K.folds_ordinary(),[])
json.dump(dict(seed=20260818+sid,init_flat_auc=d["auc"],fold_auc=d["fold_auc"]),
  open(f"/users/3171356m/agcl_audit_s0/s12a4b/out/step4_s{sid}.json","w"),indent=1)
print("init flat probe seed",sid,d["auc"])
