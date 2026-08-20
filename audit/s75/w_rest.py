"""S7.5 sections 3-8. Each section is its own resumable unit with its own DONE."""
import sys, os, argparse, numpy as np, pandas as pd, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s75"); import s75_core as S
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C7
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import r2_score
ap=argparse.ArgumentParser(); ap.add_argument("--sec",required=True); ap.add_argument("--task",type=int,default=0)
ap.add_argument("--ntask",type=int,default=1); A=ap.parse_args()
OUT=S.S75+"out"; os.makedirs(OUT,exist_ok=True)
NJ=int(os.environ.get("S75_NJOBS","1"))
D=C7.load_all(); y=D["y"]; FC=D["FC"]; NODE=D["NODE"]; meta=D["meta"]
FOLDS,_=C7.splits(); site=meta.site.values; idx=np.arange(954)
NAMES=pd.read_csv("/users/3171356m/agcl_audit_s0/s1_roi_alignment_90.csv").aal_name.tolist()
off=~np.eye(90,dtype=bool); iu=np.triu_indices(90,1)
sv=np.stack([np.where(off,FC[i],0).sum(1) for i in range(954)])
av=np.stack([np.abs(np.where(off,FC[i],0)).sum(1) for i in range(954)])
pv=np.stack([np.maximum(np.where(off,FC[i],0),0).sum(1) for i in range(954)])
nv=np.stack([np.maximum(-np.where(off,FC[i],0),0).sum(1) for i in range(954)])
Q3=np.stack([NODE["B"][i]+FC[i].T@NODE["B"][i] for i in range(954)])
LF=S.loso_folds(y,site)
if A.sec=="strength":
    F={"signed_s":sv,"absolute_a":av,"positive_p":pv,"negative_n":nv,
       "concat_sapn":np.hstack([sv,av,pv,nv]),
       "FULL_FC_4005":np.stack([FC[i][iu] for i in range(954)])}
    rows=[]; oo={}
    for k,M in F.items():
        m,o=S.probe(M,y,FOLDS,1000); oo[k]=o
        ml,_=S.probe(M,y,LF,0)
        m.update(feature=k,loso_auc=ml["auc"],loso_bacc=ml["bacc"]); rows.append(m)
        print(k,round(m["auc"],4),"loso",round(ml["auc"],4),flush=True)
    C7.write_unit(OUT,"sec3_strength",payload_json=dict(rows=rows,
        vs_fullFC={k:S.paired_delta(oo[k],oo["FULL_FC_4005"],y,2000) for k in F if k!="FULL_FC_4005"}))
elif A.sec=="invariant":
    I={"sorted_signed_strength_90":np.sort(sv,1),"sorted_absolute_strength_90":np.sort(av,1),
       "FC_eigenvalues_90":np.stack([np.sort(np.linalg.eigvalsh(FC[i])) for i in range(954)]),
       "FC_eig_zerodiag_90":np.stack([np.sort(np.linalg.eigvalsh(np.where(off,FC[i],0))) for i in range(954)]),
       "sorted_Q1_perchannel_270":np.hstack([np.sort(Q3[:,:,k],1) for k in range(3)]),
       "sorted_Q1_nodenorm_90":np.sort(np.linalg.norm(Q3,axis=2),1),
       "Q1_singular_values_3":np.stack([np.linalg.svd(Q3[i],compute_uv=False) for i in range(954)])}
    I["COMBINED_invariant"]=np.hstack([I["sorted_signed_strength_90"],I["sorted_absolute_strength_90"],
                                       I["FC_eigenvalues_90"],I["sorted_Q1_perchannel_270"]])
    rows=[]
    for k,M in I.items():
        m,_=S.probe(M,y,FOLDS,1000)
        ml=S.probe(M,y,LF,0)[0] if k in ("sorted_signed_strength_90","FC_eigenvalues_90","COMBINED_invariant") else {"auc":float("nan")}
        m.update(feature=k,loso_auc=ml["auc"]); rows.append(m)
        print(k,M.shape[1],round(m["auc"],4),flush=True)
    C7.write_unit(OUT,"sec4_invariant",payload_json=dict(rows=rows))
elif A.sec=="roiperm":
    pr=[]
    for ps in list(range(20))[A.task::A.ntask]:
        rng=np.random.default_rng(S.BASE+7000+ps)
        Xp=np.zeros((954,4005))
        for i in range(954):
            P=rng.permutation(90); Xp[i]=FC[i][np.ix_(P,P)][iu]
        m,_=S.probe(Xp,y,FOLDS,0); pr.append(dict(perm_seed=ps,**m)); print("perm",ps,round(m["auc"],4),flush=True)
    mc=None
    if A.task==0:
        rngc=np.random.default_rng(S.BASE+99); Pc=rngc.permutation(90)
        mc,_=S.probe(np.stack([FC[i][np.ix_(Pc,Pc)][iu] for i in range(954)]),y,FOLDS,0)
    C7.write_unit(OUT,f"sec5_roiperm_{A.task:02d}",payload_json=dict(rows=pr,common_perm_sanity=mc))
elif A.sec=="recov":
    out=[]
    for path in ["P","O"]:
        st=S.stages(path,0,"B",idx)
        for rep in ["h","z"]:
            for tn,T in (("signed",sv),("absolute",av)):
                r2=np.array([r2_score(T[:,c],cross_val_predict(RidgeCV(alphas=np.logspace(-3,4,12)),
                    st[rep],T[:,c],cv=KFold(5,shuffle=True,random_state=S.BASE),n_jobs=NJ)) for c in range(90)])
                out.append(dict(path=path,repr=rep,target=tn,mean=float(r2.mean()),median=float(np.median(r2)),
                    sd=float(r2.std()),min=float(r2.min()),q1=float(np.percentile(r2,25)),
                    q3=float(np.percentile(r2,75)),max=float(r2.max()),
                    n_gt0=int((r2>0).sum()),n_ge25=int((r2>=.25).sum()),n_ge50=int((r2>=.5).sum()),n_ge75=int((r2>=.75).sum()),
                    top10=[(NAMES[j],round(float(r2[j]),4)) for j in np.argsort(-r2)[:10]],
                    bottom10=[(NAMES[j],round(float(r2[j]),4)) for j in np.argsort(r2)[:10]],
                    per_roi=[float(v) for v in r2]))
                print(path,rep,tn,round(r2.mean(),4),flush=True)
    C7.write_unit(OUT,"sec7_recoverability",payload_json=dict(rows=out))
elif A.sec=="misc":
    eq=[]
    for path in ["P","O"]:
        m=C7.build_model(path,0); m.eval()
        rng=np.random.default_rng(S.BASE+1); P=rng.permutation(90); inv=np.argsort(P)
        ei=C7.edge_index(); bt=torch.zeros(90,dtype=torch.long)
        for i in [0,1,2]:
            x=torch.tensor(NODE["B"][i],dtype=torch.float32); E=torch.tensor(FC[i],dtype=torch.float32)
            with torch.no_grad():
                h0,z0,n0=m.encode(bt,x,ei,None,E.reshape(-1))
                h1,z1,n1=m.encode(bt,x[P],ei,None,E[np.ix_(P,P)].reshape(-1))
            eq.append(dict(path=path,subject=int(i),h_max_abs=float((h0-h1).abs().max()),
                h_max_rel=float(((h0-h1).abs()/h0.abs().clamp(min=1e-8)).max()),
                z_max_abs=float((z0-z1).abs().max()),
                node_equivar_max_abs=float((n0-n1[inv]).abs().max())))
    cf=[]
    for tr,te in FOLDS:
        gs=GridSearchCV(Pipeline([("sc",StandardScaler()),("clf",LinearSVC(dual="auto",max_iter=20000,random_state=S.BASE))]),
            S.GRID,cv=StratifiedKFold(5,shuffle=True,random_state=S.BASE),scoring="roc_auc",n_jobs=NJ)
        gs.fit(sv[tr],y[tr]); cf.append(np.abs(gs.best_estimator_.named_steps["clf"].coef_.ravel()))
    c=np.mean(cf,0); cn=c/c.sum(); srt=np.sort(cn)[::-1]
    C7.write_unit(OUT,"sec6_8_misc",payload_json=dict(perm_invariance=eq,
        roi_importance=dict(top10=[(NAMES[j],round(float(cn[j]),4)) for j in np.argsort(-cn)[:10]],
          frac_top5=float(srt[:5].sum()),frac_top10=float(srt[:10].sum()),frac_top20=float(srt[:20].sum()),
          n_for_50pct=int(np.searchsorted(np.cumsum(srt),0.5)+1),
          per_roi=dict(zip(NAMES,[float(v) for v in cn])))))
print("SECTION COMPLETE",A.sec,flush=True)
