"""Level 1: signal audit + univariate diagnosis statistics (DESCRIPTIVE ONLY -
these statistics are never used to select features for the Level 2 predictors)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s3c")
import s3c_core as C
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (same as statsmodels fdr_bh)."""
    p=np.asarray(p,float); n=p.size; o=np.argsort(p); ranked=p[o]
    q=ranked*n/np.arange(1,n+1)
    q=np.minimum.accumulate(q[::-1])[::-1]
    out=np.empty(n); out[o]=np.minimum(q,1.0); return out

S,y,ids=C.load(); OUT="/users/3171356m/agcl_audit_s0/s3c/"
BN=["slow5","slow4","classical"]; rng=np.random.default_rng(C.SEED)

# ---- distributions / variance / outliers / stability ----
rows=[]
for src,X in S.items():
    for b,bn in enumerate(BN):
        x=X[:,:,b]; q1,q3=np.percentile(x,[25,75]); iqr=q3-q1
        subj_sd=x.std(1); roi_sd=x.std(0)
        rows.append(dict(source=src,band=bn,min=x.min(),max=x.max(),mean=x.mean(),
            median=np.median(x),sd=x.std(),iqr=iqr,
            outliers_3iqr=int(((x<q1-3*iqr)|(x>q3+3*iqr)).sum()),
            n_nan=int((~np.isfinite(x)).sum()),
            subject_sd_cv=float(subj_sd.std()/subj_sd.mean()),
            roi_sd_cv=float(roi_sd.std()/roi_sd.mean()),
            var_between_subject=float(np.var(x.mean(1))), var_between_roi=float(np.var(x.mean(0)))))
pd.DataFrame(rows).to_csv(OUT+"lvl1_distributions.csv",index=False)

# ---- bootstrap reliability of the ROI mean profile ----
rel=[]
for src,X in S.items():
    for b,bn in enumerate(BN):
        x=X[:,:,b]; base=x.mean(0)
        rs=[]
        for _ in range(500):
            i=rng.integers(0,954,954); rs.append(pearsonr(base,x[i].mean(0))[0])
        rel.append(dict(source=src,band=bn,boot_profile_r_mean=float(np.mean(rs)),
                        boot_profile_r_p2p5=float(np.percentile(rs,2.5)),
                        boot_profile_r_p97p5=float(np.percentile(rs,97.5))))
pd.DataFrame(rel).to_csv(OUT+"lvl1_bootstrap_reliability.csv",index=False)

# ---- cross-source agreement ----
ag=[]
for a,bsrc in (("M1","M2"),("M1","OLD"),("M2","OLD")):
    for b,bn in enumerate(BN):
        u,v=S[a][:,:,b].ravel(),S[bsrc][:,:,b].ravel()
        ag.append(dict(pair=f"{a}_vs_{bsrc}",band=bn,pearson=pearsonr(u,v)[0],spearman=spearmanr(u,v)[0]))
pd.DataFrame(ag).to_csv(OUT+"lvl1_agreement.csv",index=False)

# ---- univariate diagnosis statistics per ROI x band ----
uni=[]
for src,X in S.items():
    for b,bn in enumerate(BN):
        x=X[:,:,b]; a=x[y==1]; c=x[y==0]
        na,nc=len(a),len(c)
        sp=np.sqrt(((na-1)*a.var(0,ddof=1)+(nc-1)*c.var(0,ddof=1))/(na+nc-2))
        d=(a.mean(0)-c.mean(0))/sp
        g=d*(1-3/(4*(na+nc)-9))                      # Hedges correction
        p=np.array([mannwhitneyu(a[:,k],c[:,k],alternative="two-sided")[1] for k in range(90)])
        auc=np.array([roc_auc_score(y,x[:,k]) for k in range(90)])
        mi=mutual_info_classif(x,y,random_state=C.SEED)
        q=bh_fdr(p); rej=q<0.05
        for k in range(90):
            uni.append(dict(source=src,band=bn,roi=k,cohen_d=d[k],hedges_g=g[k],mwu_p=p[k],
                            fdr_q=q[k],fdr_sig=bool(rej[k]),univar_auc=auc[k],
                            univar_auc_dev=abs(auc[k]-0.5),mutual_info=mi[k]))
u=pd.DataFrame(uni); u.to_csv(OUT+"lvl1_univariate.csv",index=False)
s=u.groupby(["source","band"]).agg(n_fdr_sig=("fdr_sig","sum"),max_absg=("hedges_g",lambda z: np.abs(z).max()),
    mean_absg=("hedges_g",lambda z: np.abs(z).mean()),max_auc_dev=("univar_auc_dev","max"),
    mean_auc_dev=("univar_auc_dev","mean"),mean_mi=("mutual_info","mean"),min_q=("fdr_q","min")).reset_index()
s.to_csv(OUT+"lvl1_univariate_summary.csv",index=False)
print(s.to_string(index=False))
print("LEVEL1 COMPLETE")
