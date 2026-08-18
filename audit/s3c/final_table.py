import pandas as pd, glob, numpy as np, json
from sklearn.metrics import roc_auc_score
pd.set_option("display.width",250)
m=pd.read_csv("lvl2_lvl4_merged.csv"); conf=pd.read_csv("lvl4_confound_all.csv")
S=np.load("X_sources.npz",allow_pickle=True); y=S["y"]
p=pd.concat([pd.read_csv(f) for f in sorted(glob.glob("lvl3/perm_*.csv"))],ignore_index=True)
p.to_csv("lvl3_perm_all.csv",index=False)
print("="*100); print("LEVEL 3 — NEGATIVE CONTROLS"); print("="*100)
print(f"total permutation runs: {len(p)}  (each = a FULL nested-CV re-run incl. inner grid search)")
rows=[]
for cond,g in p.groupby("cond"):
    s,b,n,c=cond.split("|")
    obs=m[(m.source==s)&(m.band==b)&(m.norm==n)&(m.clf==c)].pooled_auc.iloc[0]
    nperm=len(g); pv=(1+ (g.perm_auc>=obs).sum())/(1+nperm)
    rows.append(dict(cond=cond,n_perm=nperm,observed_auc=obs,
        perm_auc_mean=g.perm_auc.mean(),perm_auc_sd=g.perm_auc.std(ddof=1),
        perm_auc_p2p5=g.perm_auc.quantile(.025),perm_auc_p97p5=g.perm_auc.quantile(.975),
        perm_auc_max=g.perm_auc.max(),perm_p=pv,
        perm_bacc_mean=g.perm_bacc.mean(),
        featperm_auc_mean=g.featperm_auc.mean(),featperm_auc_sd=g.featperm_auc.std(ddof=1),
        featperm_auc_max=g.featperm_auc.max()))
pr=pd.DataFrame(rows).sort_values("observed_auc",ascending=False)
pr.to_csv("lvl3_perm_summary.csv",index=False)
print(pr.to_string(index=False,float_format=lambda x:f"{x:9.4f}"))
print("\n  label-permutation AUC should centre on 0.500; feature-column permutation likewise.")
print(f"  overall label-perm mean = {p.perm_auc.mean():.4f} (sd {p.perm_auc.std(ddof=1):.4f})")
print(f"  overall feat-perm  mean = {p.featperm_auc.mean():.4f} (sd {p.featperm_auc.std(ddof=1):.4f})")

print("\n"+"="*100); print("FINAL S3C DECISION TABLE"); print("="*100)
out=[]
for s in ["M1","M2","OLD"]:
    ob=m[m.source==s].nlargest(1,"pooled_auc").iloc[0]        # best on ordinary CV
    lb=m[m.source==s].nlargest(1,"loso_pooled_auc").iloc[0]   # best under site-held-out
    cb=conf[(conf.source==s)]
    pc=pr[pr.cond.str.startswith(s+"|")]
    strongest=pc.nlargest(1,"observed_auc").iloc[0] if len(pc) else None
    out.append(dict(source=s,
      best_band=ob.band, best_norm=ob.norm, best_clf=ob.clf,
      best_nested_auc=ob.pooled_auc, auc_sd=ob.auc_sd,
      bacc=ob.bacc_mean, sens=ob.sens_mean, spec=ob.spec_mean, f1=ob.f1_mean, prec=ob.prec_mean,
      perm_p=(strongest.perm_p if strongest is not None else np.nan),
      perm_mean_auc=(strongest.perm_auc_mean if strongest is not None else np.nan),
      site_bacc_mean=cb.site_bacc.mean(), site_bacc_max=cb.site_bacc.max(),
      r2_TR=cb.r2_TR.mean(), r2_T=cb.r2_T.mean(), r2_age=cb.r2_age.mean(), r2_fd=cb.r2_func_mean_fd.mean(),
      grouped_auc=ob.grouped_pooled_auc, loso_auc_sameCond=ob.loso_pooled_auc,
      best_loso_auc=lb.loso_pooled_auc, best_loso_cond=f"{lb.band}/{lb.norm}/{lb.clf}",
      mean_drop_loso=m[m.source==s].drop_loso.mean(),
      fold_auc_sd=ob.auc_sd))
f=pd.DataFrame(out); f.to_csv("S3C_DECISION_TABLE.csv",index=False)
print(f.to_string(index=False,float_format=lambda x:f"{x:9.4f}"))
