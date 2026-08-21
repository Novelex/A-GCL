"""S11 Steps 5-8: one task = one (method, dim, rp_seed, eval_mode). Transform INSIDE inner CV."""
import sys, os, json, argparse, hashlib, numpy as np
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7");  import s7_core as C7
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from sklearn.preprocessing import StandardScaler
ap=argparse.ArgumentParser(); ap.add_argument("--task",type=int,required=True)
ap.add_argument("--pls128",action="store_true"); A=ap.parse_args()
UNITS=[("RAW",None,None)]
UNITS+=[("PCA",d,None) for d in (32,64,128,256)]
UNITS+=[("RP",d,s) for d in (32,64,128,256) for s in (0,1,2)]
UNITS+=[("PLS",d,None) for d in (8,16,32,64)]
if A.pls128: UNITS=[("PLS",128,None)]
UNITS=[(m,d,s,ev) for (m,d,s) in UNITS for ev in ("ord","loso")]
method,dim,rseed,ev=UNITS[A.task]
name=f"{method}{'' if dim is None else dim}{'' if rseed is None else f'_r{rseed}'}_{ev}"
OUT=K.S11+"out"
if C7.is_done(OUT,name): print("skip"); sys.exit(0)
X,y,ids,meta=K.load_Xfc()
folds=K.folds_ordinary() if ev=="ord" else K.folds_loso(y)
if method=="RAW": mid=[]
elif method=="PCA": mid=[("pca",PCA(n_components=dim,random_state=K.BASE))]
elif method=="RP":  mid=[("rp",GaussianRandomProjection(n_components=dim,random_state=K.BASE+rseed))]
else:               mid=[("pls",K.PLSScores(n_components=dim))]
m,oof=K.probe_pipe(X,y,folds,mid)
extra={}
if method=="PCA" and ev=="ord":       # retained variance per OUTER train fold (reporting only)
    rv=[]
    for tr,_ in folds:
        p=PCA(n_components=dim,random_state=K.BASE).fit(StandardScaler().fit_transform(X[tr]))
        rv.append(float(p.explained_variance_ratio_.sum()))
    extra["retained_variance_mean"]=float(np.mean(rv))
m.update(method=method,dim=dim,rp_seed=rseed,eval=ev,n_folds=len(folds),**extra,
    manifest_sha=meta["manifest_sha256"][:16],X_fc_sha=meta["X_fc_sha256"][:16],
    pair_map_sha=meta["pair_map_sha256"][:16],splits_sha=K.SPLITS_SHA[:16])
np.save(OUT+f"/oof_{name}.npy",oof)
C7.write_unit(OUT,name,payload_json=dict(**m,provenance=C7.provenance({"unit":"S11"})))
print(f"{name}: AUC={m['auc']:.4f} [{m['ci_lo']:.4f},{m['ci_hi']:.4f}]",flush=True)
