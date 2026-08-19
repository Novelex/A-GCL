"""J1 - CPU analytic: J1A geometry, J1B band algebra, J1C Q1 probe, J1D controls, J1E ledger."""
import sys, os, json, re, argparse, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C
from scipy.stats import pearsonr, spearmanr
from scipy.signal import detrend
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--n",type=int,default=8)
A=ap.parse_args()
TAG=os.environ.get("S7_SMOKE_TAG","smoke")
OUT=C.S7+(TAG+"/J1" if A.smoke else "J1"); os.makedirs(OUT,exist_ok=True)
D=C.load_all(); ids=D["ids"]; y=D["y"]; RAW=D["RAW"]; FC=D["FC"]; NODE=D["NODE"]
FOLDS,SPEC=C.splits()
if A.smoke:
    sel=np.array([0,1,2,477,478,479,900,901][:A.n])
    ids=[ids[i] for i in sel]; y=y[sel]; RAW=RAW[sel]; FC=FC[sel]
    NODE={k:v[sel] for k,v in NODE.items()}
    n=len(sel); k=min(3,n//2)
    FOLDS=[(np.setdiff1d(np.arange(n),np.arange(i,n,3)),np.arange(i,n,3)) for i in range(3)]
res={"provenance":C.provenance({"unit":"J1","smoke":A.smoke,"n_subjects":len(ids)})}

# ---------------- J1A three-band geometry ----------------
j1a=[]
for nm,Xs in [("RAW",RAW),("M1_B",NODE["B"]),("M1_C",NODE["C"]),("M1_D",NODE["D"])]:
    flat=Xs.reshape(-1,3)
    g=C.geom(flat); s=np.linalg.svd(flat-flat.mean(0),compute_uv=False); v=s**2/(s**2).sum()
    P=np.corrcoef(flat.T); Sp=np.array([[spearmanr(flat[:,i],flat[:,j])[0] for j in range(3)] for i in range(3)])
    er_sub=[C.geom(Xs[i])["eff_rank"] for i in range(len(Xs))]
    j1a.append(dict(source=nm, pc1=float(v[0]),pc2=float(v[1]),pc3=float(v[2]),
        eff_rank=g["eff_rank"], pearson=P.tolist(), spearman=Sp.tolist(),
        subj_eff_rank_mean=float(np.mean(er_sub)), subj_eff_rank_sd=float(np.std(er_sub)),
        subj_eff_rank_min=float(np.min(er_sub)), subj_eff_rank_max=float(np.max(er_sub))))
res["J1A"]=j1a; print("J1A done",flush=True)

# ---------------- J1B exact band algebra (formula FROZEN by S3A: mean amplitude) ----------------
BANDS=[(0.010,0.027),(0.027,0.073),(0.010,0.080)]
tr_tab=pd.read_csv(C.REPO+"/data/subject_tr.csv").set_index("FILE_ID")
rows=[]
for i,s in enumerate(ids):
    p=f"{C.REPO}/data/ALFF_need/rois_aal/{s}_rois_aal.1D"
    hdr=open(p).readline(); lab=np.array([int(re.sub(r"^#","",t)) for t in hdr.split()])
    ts=np.loadtxt(p,skiprows=1)[:,lab<9001]; T=ts.shape[0]; tr=float(tr_tab.TR_seconds[s])
    nfft=2**int(np.ceil(np.log2(T)))
    amp=2*np.abs(np.fft.rfft(detrend(ts,axis=0),n=nfft,axis=0))/T
    f=np.fft.rfftfreq(nfft,d=tr)
    S=[set(np.where((f>=lo)&(f<=hi))[0]) for lo,hi in BANDS]
    S5,S4,CL=S
    inter=S5&S4; union=S5|S4; tail=CL-union; extra=union-CL
    # exact identity: N_CL * mean_CL == sum over CL bins == sum(S5)+sum(S4)-sum(inter)+sum(tail)
    sm=lambda idx: amp[sorted(idx)].sum(0) if idx else np.zeros(90)
    lhs=len(CL)*amp[sorted(CL)].mean(0)
    rhs=sm(S5)+sm(S4)-sm(inter)+sm(tail)
    e=np.abs(lhs-rhs); rel=e/np.maximum(np.abs(rhs),1e-300)
    # bins exactly at nominal endpoints?
    onedge=sum(int(np.any(np.isclose(f,b,atol=1e-12))) for lo,hi in BANDS for b in (lo,hi))
    rows.append(dict(file_id=s,T=T,tr=tr,nfft=nfft,n_s5=len(S5),n_s4=len(S4),n_cl=len(CL),
        n_overlap=len(inter),n_tail=len(tail),n_extra=len(extra),bins_on_nominal_edges=onedge,
        max_abs_err=float(e.max()),max_rel_err=float(rel.max()),
        mismatch_1em9=int((e>1e-9).sum()),
        tail_frac_of_cl=float(sm(tail).sum()/max(amp[sorted(CL)].sum(),1e-300))))
b=pd.DataFrame(rows); b.to_csv(OUT+"/J1B_band_algebra.csv",index=False)
res["J1B"]=dict(n=len(b),max_abs_err=float(b.max_abs_err.max()),max_rel_err=float(b.max_rel_err.max()),
    total_mismatch=int(b.mismatch_1em9.sum()),
    n_s5=[int(b.n_s5.min()),int(b.n_s5.max())],n_s4=[int(b.n_s4.min()),int(b.n_s4.max())],
    n_cl=[int(b.n_cl.min()),int(b.n_cl.max())],n_tail=[int(b.n_tail.min()),int(b.n_tail.max())],
    total_overlap_bins=int(b.n_overlap.sum()),total_extra_bins=int(b.n_extra.sum()),
    bins_on_nominal_edges=int(b.bins_on_nominal_edges.sum()),
    tail_frac_mean=float(b.tail_frac_of_cl.mean()))
print("J1B done",flush=True)

# ---------------- probe helper (leakage-safe) ----------------
GRID={"clf__C":[1e-4,1e-3,1e-2,1e-1,1,10,100]}
def probe(Xf,yv,folds,clf="linsvm"):
    oof=np.full(len(yv),np.nan); fa=[]
    for tr_,te in folds:
        c=LinearSVC(dual="auto",max_iter=20000,random_state=C.BASE_SEED) if clf=="linsvm" else \
          LogisticRegression(penalty="l2",solver="lbfgs",max_iter=3000,random_state=C.BASE_SEED)
        gs=GridSearchCV(Pipeline([("sc",StandardScaler()),("clf",c)]),GRID,
            cv=StratifiedKFold(min(5,max(2,int(np.bincount(yv[tr_]).min()))),shuffle=True,random_state=C.BASE_SEED),
            scoring="roc_auc",n_jobs=1)
        gs.fit(Xf[tr_],yv[tr_]); sc=gs.decision_function(Xf[te]); oof[te]=sc
        if len(np.unique(yv[te]))>1: fa.append(roc_auc_score(yv[te],sc))
    m=dict(auc=float(roc_auc_score(yv,oof)) if len(np.unique(yv))>1 else np.nan,
           bacc=float(balanced_accuracy_score(yv,(oof>0).astype(int))),
           fold_auc_sd=float(np.std(fa,ddof=1)) if len(fa)>1 else 0.0, n_folds=len(fa))
    rng=np.random.default_rng(C.BASE_SEED); bs=[]
    for _ in range(1000):
        j=rng.integers(0,len(yv),len(yv))
        if len(np.unique(yv[j]))>1: bs.append(roc_auc_score(yv[j],oof[j]))
    m["ci95"]=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))] if bs else [np.nan,np.nan]
    return m,oof

# ---------------- J1C parameter-free Q1 probe ----------------
j1c=[]
for br in C.BRANCHES:
    X=NODE[br]
    Q=np.stack([(np.eye(90)+FC[i].T)@X[i] for i in range(len(X))]).reshape(len(X),-1)
    Xf=X.reshape(len(X),-1)
    for clf in ["linsvm","logreg"]:
        m,_=probe(Q,y,FOLDS,clf); m.update(branch=br,repr="Q1",clf=clf); j1c.append(m)
        m2,_=probe(Xf,y,FOLDS,clf); m2.update(branch=br,repr="X",clf=clf); j1c.append(m2)
res["J1C"]=j1c; print("J1C done",flush=True)

# ---------------- J1D fold-safe mechanistic controls ----------------
j1d=[]
for br in ["B"]:
    X=NODE[br]; nS=len(X)
    for clf in ["linsvm"]:
        # CONTROL 1: shuffled-FC donors drawn ONLY from the training fold
        for perm_seed in range(3 if not A.smoke else 1):
            oof=np.full(nS,np.nan)
            for fi,(tr_,te) in enumerate(FOLDS):
                rng=np.random.default_rng(C.BASE_SEED+1000*perm_seed+fi)
                don_tr=tr_[rng.permutation(len(tr_))]           # train<->train permutation
                don_te=tr_[rng.integers(0,len(tr_),len(te))]    # test donors FROM TRAIN ONLY
                Qtr=np.stack([(np.eye(90)+FC[d].T)@X[s] for s,d in zip(tr_,don_tr)]).reshape(len(tr_),-1)
                Qte=np.stack([(np.eye(90)+FC[d].T)@X[s] for s,d in zip(te,don_te)]).reshape(len(te),-1)
                gs=GridSearchCV(Pipeline([("sc",StandardScaler()),("clf",LinearSVC(dual="auto",max_iter=20000,random_state=C.BASE_SEED))]),
                    GRID,cv=StratifiedKFold(min(5,max(2,int(np.bincount(y[tr_]).min()))),shuffle=True,random_state=C.BASE_SEED),
                    scoring="roc_auc",n_jobs=1)
                gs.fit(Qtr,y[tr_]); oof[te]=gs.decision_function(Qte)
            j1d.append(dict(branch=br,clf=clf,control="shuffled_FC_trainonly",perm_seed=perm_seed,
                auc=float(roc_auc_score(y,oof)),bacc=float(balanced_accuracy_score(y,(oof>0).astype(int)))))
        # CONTROL 2: training-mean FC (never all-954)
        oof=np.full(nS,np.nan)
        for tr_,te in FOLDS:
            Em=FC[tr_].mean(0); M=(np.eye(90)+Em.T)
            Qtr=np.stack([M@X[s] for s in tr_]).reshape(len(tr_),-1)
            Qte=np.stack([M@X[s] for s in te]).reshape(len(te),-1)
            gs=GridSearchCV(Pipeline([("sc",StandardScaler()),("clf",LinearSVC(dual="auto",max_iter=20000,random_state=C.BASE_SEED))]),
                GRID,cv=StratifiedKFold(min(5,max(2,int(np.bincount(y[tr_]).min()))),shuffle=True,random_state=C.BASE_SEED),
                scoring="roc_auc",n_jobs=1)
            gs.fit(Qtr,y[tr_]); oof[te]=gs.decision_function(Qte)
        j1d.append(dict(branch=br,clf=clf,control="training_mean_FC",perm_seed=None,
            auc=float(roc_auc_score(y,oof)),bacc=float(balanced_accuracy_score(y,(oof>0).astype(int)))))
res["J1D"]=j1d; print("J1D done",flush=True)

# ---------------- J1E ledger ----------------
res["J1E"]=dict(arch=C.ARCH,
  h_dim=32, z_dim=32,
  AUDIT_CONSTRUCTOR_ERROR_DETECTED=(
    "A first S7 run (J1 job 1869274, J2 1869275, J3a 1869276) recorded z_dim=300. That was an "
    "AUDIT-SIDE constructor error: s7_core.build_model used GInfoMinMax's STANDALONE class "
    "default proj_hidden_dim=300 instead of instantiating as the real A-GCL callers do. "
    "Corrected 2026-08-19; the affected J2/J3a/J3b outputs were discarded and recomputed."),
  proj_dim_ledger=dict(
    class_default="GInfoMinMax.__init__(self, encoder, proj_hidden_dim=300)  <- standalone default, "
                  "never used by any A-GCL training script",
    actual_call_site_current="agcl_ABIDE.py:98-102 and agcl_ABIDE_queue.py:229-233 -> "
                  "GInfoMinMax(TUEncoder(...), args.emb_dim), --emb_dim default 32",
    actual_call_site_original="bed5441:A-GCL/adgcl_edge.py:50-53 -> "
                  "GInfoMinMax(TUEncoder(...), args.emb_dim), --emb_dim default 32",
    standalone_default_z_dim=300, actual_agcl_experimental_z_dim=32,
    proj_head="Linear(32,32) -> ReLU -> Linear(32,32)"),
  eval_repr=dict(P="z (paper_exact profile sets eval_representation='z')",
                 O="h (original passes model.encoder to the evaluator -> pooled encoder output, PRE-projection)",
                 C_corrected="z (argparse default)", C_paper_exact="z (profile override)",
                 paper="AMBIGUOUS - paper says only that an SVM classifies 'the extracted features'"),
  pooling=dict(default="standard", overridden_anywhere=False,
               note="argparse default 'standard'; no call site sets layerwise in the production mains"),
  arch_equivalence="O and C have IDENTICAL architecture flags; they differ only in drop_ratio "
                   "(inactive under eval()) and in the h-vs-z readout.")
pd.DataFrame(j1a).to_csv(OUT+"/J1A_geometry.csv",index=False)
pd.DataFrame(j1c).to_csv(OUT+"/J1C_probe.csv",index=False)
pd.DataFrame(j1d).to_csv(OUT+"/J1D_controls.csv",index=False)
C.write_unit(OUT,"J1_RESULTS",payload_json=res)
open(OUT+"/J1_EVIDENCE.txt","w").write(json.dumps(res,indent=1,default=str))
open(OUT+"/J1_DONE","w").write(C.git_head()+"\n")
print("J1 COMPLETE",flush=True)
