"""S7.5 close-out. CPU only. NO TRAINING, NO S8, NO 116-ROI BRANCH."""
import sys, os, json, argparse, numpy as np, pandas as pd, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s75"); import s75_core as S
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C7
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true")
ap.add_argument("--seeds",type=int,default=10); ap.add_argument("--permseeds",type=int,default=20)
A=ap.parse_args()
OUT=S.S75+("out_smoke" if A.smoke else "out"); os.makedirs(OUT,exist_ok=True)
D=C7.load_all(); y=D["y"]; FC=D["FC"]; NODE=D["NODE"]; meta=D["meta"]; ids=D["ids"]
FOLDS,_=C7.splits(); site=meta.site.values
idx=np.arange(954)
if A.smoke:
    idx=np.array([0,1,2,477,478,479,900,901]); y=y[idx]; FC=FC[idx]
    NODE={k:v[idx] for k,v in NODE.items()}; site=site[idx]
    n=len(idx); FOLDS=[(np.setdiff1d(np.arange(n),np.arange(i,n,3)),np.arange(i,n,3)) for i in range(3)]
    idx=np.arange(len(idx))*0+np.array([0,1,2,477,478,479,900,901])
NS=2 if A.smoke else A.seeds; NP=2 if A.smoke else A.permseeds; BOOT=200 if A.smoke else 1000
res={"provenance":C7.provenance({"unit":"S7.5","smoke":A.smoke,"n":len(y),"seeds":NS})}
roi=pd.read_csv("/users/3171356m/agcl_audit_s0/s1_roi_alignment_90.csv")
NAMES=roi.aal_name.tolist()
def sub(M): return M[np.arange(len(y))] if len(M)==len(y) else M

# ---------------- 1. STAGE-WISE PROBE ----------------
X=NODE["B"].reshape(len(y),-1)
Q1=np.stack([NODE["B"][i]+FC[i].T@NODE["B"][i] for i in range(len(y))]).reshape(len(y),-1)
stage_rows=[]; oofs={}
m,o=S.probe(X,y,FOLDS,BOOT);  m.update(stage="X",path="-",seed=-1); stage_rows.append(m); oofs["X"]=o
m,o=S.probe(Q1,y,FOLDS,BOOT); m.update(stage="Q1",path="-",seed=-1); stage_rows.append(m); oofs["Q1"]=o
STAGES=[("wgin1","WGIN1_preBN"),("bn1","BN1"),("postbn1","postBN1_toWGIN2"),
        ("wgin2","WGIN2_preBN"),("bn2","BN2_final_nodes_preNorm"),
        ("final_nodes_postnorm","final_nodes_postNorm"),("h","h"),("z","z")]
per_seed={}
for path in ["P","O"]:
    for sd in range(NS):
        st=S.stages(path,sd,"B",idx)
        for key,lab in STAGES:
            mm,oo=S.probe(st[key],y,FOLDS,BOOT)
            mm.update(stage=lab,path=path,seed=sd); stage_rows.append(mm)
            per_seed.setdefault((path,sd),{})[lab]=oo
        print("stages done",path,sd,flush=True)
sr=pd.DataFrame(stage_rows); sr.to_csv(OUT+"/S75_stagewise.csv",index=False)
agg=sr[sr.seed>=0].groupby(["path","stage"]).agg(auc_mean=("auc","mean"),auc_sd=("auc","std"),
    bacc=("bacc","mean"),n_feat=("n_feat","first")).reset_index()
res["stagewise"]=dict(single=sr[sr.seed<0].to_dict("records"),random=agg.to_dict("records"))

# ---------------- 2. PRE-POOLING vs POOLING ----------------
pp=[]
for (path,sd),dd in per_seed.items():
    pp.append(dict(path=path,seed=sd,
        nodes_minus_h=S.paired_delta(dd["final_nodes_postNorm"],dd["h"],y,BOOT)["obs"],
        h_minus_z=S.paired_delta(dd["h"],dd["z"],y,BOOT)["obs"],
        Q1_minus_nodes=float(np.nan)))
ppd=pd.DataFrame(pp); ppd.to_csv(OUT+"/S75_prepool_vs_pool.csv",index=False)
def seedci(v):
    v=np.asarray(v); rng=np.random.default_rng(S.BASE)
    b=[np.mean(v[rng.integers(0,len(v),len(v))]) for _ in range(2000)]
    return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)) if len(v)>1 else 0.0,
                ci_lo=float(np.percentile(b,2.5)),ci_hi=float(np.percentile(b,97.5)))
res["prepool_vs_pool"]={p:{"nodes_minus_h":seedci(ppd[ppd.path==p].nodes_minus_h),
                           "h_minus_z":seedci(ppd[ppd.path==p].h_minus_z)} for p in ppd.path.unique()}

# ---------------- 3. REGIONAL FC-STRENGTH BASELINES ----------------
off=~np.eye(90,dtype=bool)
sv=np.stack([np.where(off,FC[i],0).sum(1) for i in range(len(y))])
av=np.stack([np.abs(np.where(off,FC[i],0)).sum(1) for i in range(len(y))])
pv=np.stack([np.maximum(np.where(off,FC[i],0),0).sum(1) for i in range(len(y))])
nv=np.stack([np.maximum(-np.where(off,FC[i],0),0).sum(1) for i in range(len(y))])
FEATS={"signed_s":sv,"absolute_a":av,"positive_p":pv,"negative_n":nv,
       "concat_sapn":np.hstack([sv,av,pv,nv])}
iu=np.triu_indices(90,1); FCU=np.stack([FC[i][iu] for i in range(len(y))])
FEATS["FULL_FC_4005"]=FCU
rows=[]; roofs={}
LF=S.loso_folds(y,site)
for k,M in FEATS.items():
    mm,oo=S.probe(M,y,FOLDS,BOOT); roofs[k]=oo
    ml,_=S.probe(M,y,LF,BOOT) if len(LF)>1 else ({},None)
    mm.update(feature=k,loso_auc=ml.get("auc",np.nan),loso_bacc=ml.get("bacc",np.nan)); rows.append(mm)
    print("strength",k,round(mm["auc"],4),flush=True)
rs=pd.DataFrame(rows); rs.to_csv(OUT+"/S75_strength_baselines.csv",index=False)
res["strength"]=rows
res["strength_vs_fullFC"]={k:S.paired_delta(roofs[k],roofs["FULL_FC_4005"],y,BOOT)
                           for k in FEATS if k!="FULL_FC_4005"}

# ---------------- 4. PERMUTATION-INVARIANT BASELINES ----------------
Q3=np.stack([NODE["B"][i]+FC[i].T@NODE["B"][i] for i in range(len(y))])
INV={"sorted_signed_strength_90":np.sort(sv,axis=1),
     "sorted_absolute_strength_90":np.sort(av,axis=1),
     "FC_eigenvalues_90":np.stack([np.sort(np.linalg.eigvalsh(FC[i])) for i in range(len(y))]),
     "FC_eig_zerodiag_90":np.stack([np.sort(np.linalg.eigvalsh(np.where(off,FC[i],0))) for i in range(len(y))]),
     "sorted_Q1_perchannel_270":np.hstack([np.sort(Q3[:,:,k],axis=1) for k in range(3)]),
     "sorted_Q1_nodenorm_90":np.sort(np.linalg.norm(Q3,axis=2),axis=1),
     "Q1_singular_values_3":np.stack([np.linalg.svd(Q3[i],compute_uv=False) for i in range(len(y))])}
INV["COMBINED_invariant"]=np.hstack([INV["sorted_signed_strength_90"],INV["sorted_absolute_strength_90"],
                                     INV["FC_eigenvalues_90"],INV["sorted_Q1_perchannel_270"]])
irows=[]
for k,M in INV.items():
    mm,oo=S.probe(M,y,FOLDS,BOOT)
    ml,_=S.probe(M,y,LF,BOOT) if (len(LF)>1 and k in ("sorted_signed_strength_90","FC_eigenvalues_90","COMBINED_invariant")) else ({},None)
    mm.update(feature=k,loso_auc=ml.get("auc",np.nan)); irows.append(mm)
    print("invariant",k,M.shape[1],round(mm["auc"],4),flush=True)
pd.DataFrame(irows).to_csv(OUT+"/S75_invariant_baselines.csv",index=False)
res["invariant"]=irows

# ---------------- 5. ROI-ALIGNMENT DEPENDENCE CONTROL ----------------
pr=[]
for ps in range(NP):
    rng=np.random.default_rng(S.BASE+7000+ps)
    Xp=np.zeros((len(y),4005))
    for i in range(len(y)):
        P=rng.permutation(90); Xp[i]=FC[i][np.ix_(P,P)][iu]
    mm,_=S.probe(Xp,y,FOLDS,boot=0); pr.append(dict(perm_seed=ps,auc=mm["auc"],bacc=mm["bacc"]))
    print("roiperm",ps,round(mm["auc"],4),flush=True)
prd=pd.DataFrame(pr); prd.to_csv(OUT+"/S75_roi_permutation.csv",index=False)
rngc=np.random.default_rng(S.BASE+99); Pc=rngc.permutation(90)
Xc=np.stack([FC[i][np.ix_(Pc,Pc)][iu] for i in range(len(y))])
mc,_=S.probe(Xc,y,FOLDS,boot=0)
res["roi_alignment"]=dict(independent_perm_mean=float(prd.auc.mean()),sd=float(prd.auc.std(ddof=1)),
    lo=float(prd.auc.min()),hi=float(prd.auc.max()),n_seeds=len(prd),
    aligned_reference=float(rs[rs.feature=="FULL_FC_4005"].auc.iloc[0]),
    common_perm_sanity_auc=float(mc["auc"]),
    common_perm_note="one common relabeling must reproduce aligned performance (sanity only)")

# ---------------- 6. PERMUTATION EQUIVARIANCE / INVARIANCE PROOF ----------------
eq=[]
for path in ["P","O"]:
    m=C7.build_model(path,0); m.eval()
    rng=np.random.default_rng(S.BASE+1); P=rng.permutation(90)
    for i in idx[:3]:
        x=torch.tensor(NODE["B"][list(idx).index(i)] if A.smoke else NODE["B"][i],dtype=torch.float32)
        E=torch.tensor(FC[list(idx).index(i)] if A.smoke else FC[i],dtype=torch.float32)
        ei=C7.edge_index(); bt=torch.zeros(90,dtype=torch.long)
        with torch.no_grad():
            h0,z0,n0=m.encode(bt,x,ei,None,E.reshape(-1))
            xp=x[P]; Ep=E[np.ix_(P,P)]
            h1,z1,n1=m.encode(bt,xp,ei,None,Ep.reshape(-1))
        inv=np.argsort(P)
        eq.append(dict(path=path,subject=int(i),
            h_max_abs=float((h0-h1).abs().max()),
            h_max_rel=float(((h0-h1).abs()/h0.abs().clamp(min=1e-8)).max()),
            z_max_abs=float((z0-z1).abs().max()),
            node_equivar_max_abs=float((n0-n1[inv]).abs().max())))
pd.DataFrame(eq).to_csv(OUT+"/S75_perm_invariance.csv",index=False)
res["perm_invariance"]=eq

# ---------------- 7. FULL 90-ROI FC RECOVERABILITY (no labels) ----------------
rec=[]
for path in ["P","O"]:
    st=S.stages(path,0,"B",idx)
    for rep in ["h","z"]:
        for tname,T in (("signed",sv),("absolute",av)):
            r2=[]
            for c in range(90):
                p=cross_val_predict(RidgeCV(alphas=np.logspace(-3,4,12)),st[rep],T[:,c],
                                    cv=KFold(5,shuffle=True,random_state=S.BASE))
                r2.append(r2_score(T[:,c],p))
            r2=np.array(r2)
            rec.append(dict(path=path,repr=rep,target=tname,mean=float(r2.mean()),median=float(np.median(r2)),
                sd=float(r2.std()),min=float(r2.min()),q1=float(np.percentile(r2,25)),
                q3=float(np.percentile(r2,75)),max=float(r2.max()),
                n_gt0=int((r2>0).sum()),n_ge25=int((r2>=.25).sum()),n_ge50=int((r2>=.50).sum()),
                n_ge75=int((r2>=.75).sum()),
                top10=[(NAMES[j],round(float(r2[j]),4)) for j in np.argsort(-r2)[:10]],
                bottom10=[(NAMES[j],round(float(r2[j]),4)) for j in np.argsort(r2)[:10]],
                per_roi=[float(v) for v in r2]))
            print("recov",path,rep,tname,round(r2.mean(),4),flush=True)
pd.DataFrame([{k:v for k,v in r.items() if k not in("top10","bottom10","per_roi")} for r in rec]).to_csv(OUT+"/S75_fc_recoverability.csv",index=False)
res["fc_recoverability"]=rec

# ---------------- 8. FOLD-SAFE ROI COEFFICIENT DISTRIBUTION ----------------
from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC; from sklearn.model_selection import GridSearchCV, StratifiedKFold
coefs=[]
for tr,te in FOLDS:
    gs=GridSearchCV(Pipeline([("sc",StandardScaler()),("clf",LinearSVC(dual="auto",max_iter=20000,random_state=S.BASE))]),
        S.GRID,cv=StratifiedKFold(min(5,max(2,int(np.bincount(y[tr]).min()))),shuffle=True,random_state=S.BASE),
        scoring="roc_auc",n_jobs=1)
    gs.fit(sv[tr],y[tr]); coefs.append(np.abs(gs.best_estimator_.named_steps["clf"].coef_.ravel()))
cf=np.mean(coefs,axis=0); cfn=cf/cf.sum()
srt=np.sort(cfn)[::-1]
res["roi_importance"]=dict(top10=[(NAMES[j],round(float(cfn[j]),4)) for j in np.argsort(-cfn)[:10]],
    frac_top5=float(srt[:5].sum()),frac_top10=float(srt[:10].sum()),frac_top20=float(srt[:20].sum()),
    gini=float(1-2*np.trapz(np.cumsum(np.sort(cfn))/cfn.sum(),dx=1/90)),
    n_needed_for_50pct=int(np.searchsorted(np.cumsum(srt),0.5)+1),
    note="coefficients fitted INSIDE training folds only, then averaged; no full-cohort selection")
pd.DataFrame(dict(roi=NAMES,mean_abs_coef=cf,norm=cfn)).to_csv(OUT+"/S75_roi_importance.csv",index=False)

C7.write_unit(OUT,"S75_RESULTS",payload_json=res)
open(OUT+"/S75_DONE","w").write(C7.git_head()+"\n")
print("S7.5 COMPLETE",flush=True)
