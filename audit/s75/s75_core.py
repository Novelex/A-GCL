"""S7.5 core. CPU-only. NO TRAINING. Reuses frozen S3C splits and S7 model contract."""
import sys, os, json, numpy as np, pandas as pd, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C7
sys.path.insert(0,"/users/3171356m/A-GCL")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
S75="/users/3171356m/agcl_audit_s0/s75/"
GRID={"clf__C":[1e-4,1e-3,1e-2,1e-1,1,10,100]}
BASE=C7.BASE_SEED

def probe(X,y,folds,boot=1000,seed=BASE):
    """Leakage-safe: scaler + C chosen inside training folds only."""
    oof=np.full(len(y),np.nan); fa=[]
    for tr,te in folds:
        nmin=int(np.bincount(y[tr]).min())
        gs=GridSearchCV(Pipeline([("sc",StandardScaler()),
                                  ("clf",LinearSVC(dual="auto",max_iter=20000,random_state=seed))]),
            GRID,cv=StratifiedKFold(min(5,max(2,nmin)),shuffle=True,random_state=seed),
            scoring="roc_auc",n_jobs=int(os.environ.get("S75_NJOBS","1")))
        gs.fit(X[tr],y[tr]); s=gs.decision_function(X[te]); oof[te]=s
        if len(np.unique(y[te]))>1: fa.append(roc_auc_score(y[te],s))
    auc=float(roc_auc_score(y,oof)) if len(np.unique(y))>1 else np.nan
    rng=np.random.default_rng(seed); bs=[]
    for _ in range(boot):
        j=rng.integers(0,len(y),len(y))
        if len(np.unique(y[j]))>1: bs.append(roc_auc_score(y[j],oof[j]))
    return dict(auc=auc,bacc=float(balanced_accuracy_score(y,(oof>0).astype(int))),
                fold_sd=float(np.std(fa,ddof=1)) if len(fa)>1 else 0.0,
                ci_lo=float(np.percentile(bs,2.5)) if bs else np.nan,
                ci_hi=float(np.percentile(bs,97.5)) if bs else np.nan,n_feat=X.shape[1]), oof

def loso_folds(y,site):
    return [(np.where(site!=s)[0],np.where(site==s)[0]) for s in sorted(set(site))
            if len(np.unique(y[np.where(site==s)[0]]))>1]

def paired_delta(oof_a,oof_b,y,boot=2000,seed=BASE):
    rng=np.random.default_rng(seed); d=[]
    for _ in range(boot):
        j=rng.integers(0,len(y),len(y))
        if len(np.unique(y[j]))<2: continue
        d.append(roc_auc_score(y[j],oof_a[j])-roc_auc_score(y[j],oof_b[j]))
    d=np.array(d)
    return dict(mean=float(d.mean()),ci_lo=float(np.percentile(d,2.5)),
                ci_hi=float(np.percentile(d,97.5)),
                obs=float(roc_auc_score(y,oof_a)-roc_auc_score(y,oof_b)))

# ---- stage extraction with forward hooks (production code untouched) ----
def stages(path,seed,branch,idx,bs=64):
    """Returns dict of per-subject flattened stage representations."""
    m=C7.build_model(path,seed); m.eval(); enc=m.encoder
    acts={}; hks=[]
    def mk(nm):
        def f(mod,i,o): acts.setdefault(nm,[]).append(o.detach().float().cpu())
        return f
    for i,cv in enumerate(enc.convs): hks.append(cv.register_forward_hook(mk(f"wgin{i+1}")))
    for i,bn in enumerate(enc.bns):   hks.append(bn.register_forward_hook(mk(f"bn{i+1}")))
    H=[];Z=[];NODE=[]
    with torch.no_grad():
        for s in range(0,len(idx),bs):
            ch=idx[s:s+bs]; x,ei,ew,bt=C7.batch_graphs(ch,branch)
            h,z,ne=m.encode(bt,x,ei,None,ew)
            H.append(h.cpu()); Z.append(z.cpu()); NODE.append(ne.cpu())
    for k in hks: k.remove()
    n=len(idx)
    out={}
    for k,v in acts.items(): out[k]=torch.cat(v).numpy().reshape(n,-1)
    out["final_nodes_postnorm"]=torch.cat(NODE).numpy().reshape(n,-1)   # after optional L2 norm
    out["h"]=torch.cat(H).numpy(); out["z"]=torch.cat(Z).numpy()
    # post-BN1 activation actually handed to WGIN2 (replicates encoder logic, eval -> no dropout)
    b1=torch.from_numpy(out["bn1"].reshape(n*90,-1))
    out["postbn1"]=(torch.relu(b1) if enc.post_bn_relu else b1).numpy().reshape(n,-1)
    return out
