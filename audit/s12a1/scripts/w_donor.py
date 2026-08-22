"""Phase J: PRIMARY DONOR CONTROL, fold-safe. One task = one (seed, eval_mode).
Inner C selection uses INNER-TRAIN mean FC donors; final fit uses OUTER-TRAIN mean FC.
Reuses the EXACT saved 93-input encoder state of the matching real-FC condition."""
import sys, os, json, argparse, numpy as np, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
import s7_core as C7
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True)
ap.add_argument("--eval",choices=["ord","loso"],required=True); Aa=ap.parse_args()
OUT=A.A1+"out"; name=f"donor_H1_BN_s{Aa.seed}_{Aa.eval}"
if C7.is_done(OUT,name): print("skip"); sys.exit(0)
z=np.load(OUT+"/gate_tensors.npz"); FC=z["FC"].astype(np.float64); Xold=z["Xold"]; y=z["y"].astype(int)
I90=np.eye(90,dtype=np.float32)
Xid=np.concatenate([Xold,np.repeat(I90[None],954,0)],axis=2)
m=A.build_encoder(93,A.BASE+Aa.seed)
m.load_state_dict(torch.load(OUT+f"/enc93_s{Aa.seed}.pt",weights_only=False))   # EXACT real-FC state
GRID=[1e-4,1e-3,1e-2,1e-1,1,10,100]
def donor_H1(subj_idx, mean_fc):
    """H1_BN for given subjects with the SAME donor mean-FC graph for everyone."""
    E=[]
    mf=mean_fc.astype(np.float32)
    for s0 in range(0,len(subj_idx),64):
        ch=subj_idx[s0:s0+64]
        S=A.stages_for_subject_batch(m,Xid[ch],np.repeat(mf[None],len(ch),0),want_z=False)
        E.append(S["H1_BN"].reshape(len(ch),-1))
    return np.concatenate(E)
folds=K.folds_ordinary() if Aa.eval=="ord" else K.folds_loso(y)
oof=np.full(954,np.nan); fold_auc=[]
for tr,te in folds:
    # ---- inner C selection: inner-train mean FC only ----
    nmin=int(np.bincount(y[tr]).min())
    inner=StratifiedKFold(min(5,max(2,nmin)),shuffle=True,random_state=A.BASE)
    scores={C:[] for C in GRID}
    for itr_r,iva_r in inner.split(np.zeros(len(tr)),y[tr]):
        itr,iva=tr[itr_r],tr[iva_r]
        mfc=FC[itr].mean(0)                              # INNER-TRAIN mean only
        Xi=donor_H1(itr,mfc); Xv=donor_H1(iva,mfc)
        sc=StandardScaler().fit(Xi); Xis=sc.transform(Xi); Xvs=sc.transform(Xv)
        # identical fits, executed in parallel across C values (math unchanged; perf only)
        from joblib import Parallel, delayed
        def fitC(C):
            clf=LinearSVC(dual="auto",max_iter=20000,random_state=A.BASE,C=C).fit(Xis,y[itr])
            sv=clf.decision_function(Xvs)
            return roc_auc_score(y[iva],sv) if len(np.unique(y[iva]))>1 else 0.5
        for C,v in zip(GRID,Parallel(n_jobs=int(os.environ.get("S11_NJOBS","4")))(delayed(fitC)(C) for C in GRID)):
            scores[C].append(v)
    Cbest=max(GRID,key=lambda C:np.mean(scores[C]))
    # ---- final: OUTER-TRAIN mean FC ----
    mfc=FC[tr].mean(0)
    Xtr=donor_H1(tr,mfc); Xte=donor_H1(te,mfc)
    sc=StandardScaler().fit(Xtr)
    clf=LinearSVC(dual="auto",max_iter=20000,random_state=A.BASE,C=Cbest).fit(sc.transform(Xtr),y[tr])
    s=clf.decision_function(sc.transform(Xte)); oof[te]=s
    if len(np.unique(y[te]))>1: fold_auc.append(float(roc_auc_score(y[te],s)))
    print(f"fold done C={Cbest} auc={fold_auc[-1] if fold_auc else float('nan'):.4f}",flush=True)
cov=np.isfinite(oof); yc,oc=y[cov],oof[cov]
yh=(oc>0).astype(int); tn,fp,fn,tp=confusion_matrix(yc,yh,labels=[0,1]).ravel()
rng=np.random.default_rng(A.BASE); bs=[]
for _ in range(2000):
    j=rng.integers(0,len(yc),len(yc))
    if len(np.unique(yc[j]))>1: bs.append(roc_auc_score(yc[j],oc[j]))
res=dict(condition="id_donor",stage="H1_BN",seed=Aa.seed,eval=Aa.eval,n_feat=2880,
    auc=float(roc_auc_score(yc,oc)),bacc=float(balanced_accuracy_score(yc,yh)),
    acc=float((tp+tn)/len(yc)),sens=float(tp/(tp+fn)),spec=float(tn/(tn+fp)),
    fold_auc=fold_auc,fold_sd=float(np.std(fold_auc,ddof=1)),
    ci_lo=float(np.percentile(bs,2.5)),ci_hi=float(np.percentile(bs,97.5)),
    leakage_protocol="inner-train mean FC for C selection; outer-train mean FC for final; never global")
C7.write_unit(OUT,name,payload_json=dict(**res,provenance=C7.provenance({"unit":"S12A1_donor"})))
print(f"{name}: AUC={res['auc']:.4f}",flush=True)
