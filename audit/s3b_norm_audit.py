import numpy as np, pandas as pd, json
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
np.seterr(all="ignore")
A="/users/3171356m/A-GCL/"; OUT="/users/3171356m/agcl_audit_s0/"
coh=pd.read_csv(OUT+"s1_audit_table.csv")
ids=coh.subject_id.tolist(); site=coh.site_tr.tolist()
tr_tab=pd.read_csv(A+"data/subject_tr.csv").set_index("FILE_ID")
TR=np.array([float(tr_tab.TR_seconds[i]) for i in ids])
NV=np.array([int(tr_tab.N_VOLUMES[i]) for i in ids])
z1=np.load(A+"ALFF_func_proc/method1/alff_roi_first.npz")
z2=np.load(A+"ALFF_func_proc/method2/alff_voxel_first.npz")
zo=np.load(A+"data/ALFF_need/alff_new.npz"); oi={str(s):i for i,s in enumerate(zo["file_ids"])}
SRC={"M1":z1["alff"].astype(np.float64),"M2":z2["alff"].astype(np.float64),
     "OLD":zo["alff"][np.array([oi[s] for s in ids])].astype(np.float64)}
BN=["slow5","slow4","classical"]; EPS=0.0   # no epsilon: expose zero-variance honestly

# ---------------- transforms ----------------
def A_raw(X, tr_idx=None):  return X.copy()
def B_joint(X, tr_idx=None):
    mn=X.min(axis=(1,2),keepdims=True); mx=X.max(axis=(1,2),keepdims=True)
    return (X-mn)/(mx-mn)
def C_pb_mm(X, tr_idx=None):
    mn=X.min(axis=1,keepdims=True); mx=X.max(axis=1,keepdims=True)
    return (X-mn)/(mx-mn)
def D_pb_z(X, tr_idx=None):
    mu=X.mean(axis=1,keepdims=True); sd=X.std(axis=1,keepdims=True,ddof=0)
    return (X-mu)/sd
def E_fold_mm(X, tr_idx):
    """min/max per band from TRAIN subjects only; applied to everyone."""
    T=X[tr_idx]; mn=T.min(axis=(0,1)); mx=T.max(axis=(0,1))   # per band -> shape (3,)
    return (X-mn)/(mx-mn), mn, mx
def F_fold_z(X, tr_idx):
    T=X[tr_idx]; mu=T.mean(axis=(0,1)); sd=T.std(axis=(0,1),ddof=0)
    return (X-mu)/sd, mu, sd

# ---------------- property helpers ----------------
def var_decomp(X):
    return dict(subj=float(np.var(X.mean(axis=(1,2)))),
                roi =float(np.var(X.mean(axis=(0,2)))),
                band=float(np.var(X.mean(axis=(0,1)))),
                tot =float(np.var(X)))
def rank_pres(R,Y):
    wb=[];  wa=[]
    for i in range(0,954,25):
        for b in range(3):
            s=spearmanr(R[i,:,b],Y[i,:,b])[0]; wb.append(1.0 if np.isnan(s) else s)
        s=spearmanr(R[i].ravel(),Y[i].ravel())[0]; wa.append(1.0 if np.isnan(s) else s)
    return float(np.nanmin(wb)), float(np.nanmin(wa))
def scale_probe(X):
    lvl=X.mean(axis=(1,2))
    r_tr = pearsonr(lvl,TR)[0] if np.std(lvl)>0 else np.nan
    r_nv = pearsonr(lvl,NV)[0] if np.std(lvl)>0 else np.nan
    df=pd.DataFrame({"lvl":lvl,"site":site})
    gm=df.lvl.mean(); ss_t=((df.lvl-gm)**2).sum()
    ss_b=df.groupby("site").apply(lambda g: len(g)*(g.lvl.mean()-gm)**2, include_groups=False).sum()
    eta = float(ss_b/ss_t) if ss_t>0 else np.nan
    return r_tr, r_nv, eta

rows=[]; leak=[]
kf=KFold(n_splits=5, shuffle=True, random_state=0)     # NO labels used
folds=list(kf.split(np.arange(954)))
tr_idx, te_idx = folds[0]

for sname,X in SRC.items():
    R=X.copy(); vr=var_decomp(R)
    for tname,fn,formula in [
        ("A_RAW", A_raw, "Y = X  (identity)"),
        ("B_SUBJ_JOINT_MINMAX", B_joint, "Y[s,r,b] = (X[s,r,b]-min_{r,b}X[s])/(max_{r,b}X[s]-min_{r,b}X[s])"),
        ("C_SUBJ_BAND_MINMAX", C_pb_mm, "Y[s,r,b] = (X[s,r,b]-min_r X[s,:,b])/(max_r X[s,:,b]-min_r X[s,:,b])"),
        ("D_SUBJ_BAND_ZSCORE", D_pb_z, "Y[s,r,b] = (X[s,r,b]-mean_r X[s,:,b])/std_r X[s,:,b]"),
        ("E_TRAINFOLD_BAND_MINMAX", None, "Y[:,:,b] = (X[:,:,b]-min_{s in TRAIN,r})/(max_{s in TRAIN,r}-min_{s in TRAIN,r})"),
        ("F_TRAINFOLD_BAND_ZSCORE", None, "Y[:,:,b] = (X[:,:,b]-mean_{s in TRAIN,r})/std_{s in TRAIN,r}"),
    ]:
        if tname.startswith("E"): Y,p1,p2=E_fold_mm(X,tr_idx)
        elif tname.startswith("F"): Y,p1,p2=F_fold_z(X,tr_idx)
        else: Y=fn(X)
        v=var_decomp(Y)
        zb=int((X.max(axis=1)-X.min(axis=1)==0).sum()); zs=int((X.std(axis=1,ddof=0)==0).sum())
        zj=int((X.max(axis=(1,2))-X.min(axis=(1,2))==0).sum())
        rb,ra=rank_pres(R,Y)
        rtr,rnv,eta=scale_probe(Y)
        good=np.isfinite(Y)
        pr=pearsonr(R[good].ravel(),Y[good].ravel())[0]
        sp=spearmanr(R[good].ravel(),Y[good].ravel())[0]
        rows.append(dict(source=sname, transform=tname, formula=formula,
            out_min=float(np.nanmin(Y)), out_max=float(np.nanmax(Y)),
            out_mean=float(np.nanmean(Y)), out_std=float(np.nanstd(Y)),
            n_nan=int((~np.isfinite(Y)).sum()), zerovar_subjband=zb, zerostd_subjband=zs,
            zerovar_subj_joint=zj,
            rank_pres_within_band=rb, rank_pres_within_subject_all270=ra,
            var_subj_ret=v["subj"]/vr["subj"], var_band_ret=v["band"]/vr["band"],
            var_roi_ret=v["roi"]/vr["roi"], var_tot_ret=v["tot"]/vr["tot"],
            pearson_vs_raw=pr, spearman_vs_raw=sp,
            corr_level_TR=rtr, corr_level_NVOL=rnv, site_eta2_level=eta))

pd.DataFrame(rows).to_csv(OUT+"s3b_norm_matrix.csv", index=False)

# ---------------- 11. LEAKAGE PROOF ----------------
print("="*84); print("11. TRAIN-ONLY FITTING / LEAKAGE PROOF (KFold(5, shuffle, seed=0), NO labels used)")
print("="*84)
X=SRC["M1"]
print(f"  fold 0: n_train={len(tr_idx)}  n_test={len(te_idx)}  disjoint={len(set(tr_idx)&set(te_idx))==0}")
for f,(tri,tei) in enumerate(folds):
    Yt,mn_t,mx_t=E_fold_mm(X,tri)
    _,mn_g,mx_g=E_fold_mm(X,np.arange(954))          # global fit = LEAKY reference
    te=Yt[tei]
    out_lo=int((te<0).sum()); out_hi=int((te>1).sum())
    Yg=(X-mn_g)/(mx_g-mn_g); teg=Yg[tei]
    print(f"  fold{f}: train min/max per band = {np.round(mn_t,4)} / {np.round(mx_t,4)}")
    print(f"         global  min/max per band = {np.round(mn_g,4)} / {np.round(mx_g,4)}   "
          f"param delta max = {float(np.abs(mx_t-mx_g).max()):.4f}")
    print(f"         TEST values outside [0,1] : train-only fit = {out_lo+out_hi:5d}   "
          f"global(leaky) fit = {int((teg<0).sum()+(teg>1).sum()):5d}")
    leak.append(dict(fold=f, n_train=len(tri), n_test=len(tei),
        train_max=mx_t.tolist(), global_max=mx_g.tolist(),
        test_outside_trainfit=out_lo+out_hi, test_outside_globalfit=int((teg<0).sum()+(teg>1).sum())))
print("\n  INTERPRETATION: under train-only fitting some TEST values necessarily fall outside")
print("  [0,1] because the test extremes were never seen. Under global fitting that count is")
print("  0 by construction -> the 0 is the observable SIGNATURE of test-set leakage.")
json.dump(leak, open(OUT+"s3b_leakage.json","w"), indent=1)

# fold-parameter stability
print("\n  fold-to-fold variation of fitted params (E, per band):")
mxs=np.array([l["train_max"] for l in leak]); print(f"    max  across folds: {np.round(mxs,3).tolist()}")
print(f"    spread (max-min) per band: {np.round(mxs.max(0)-mxs.min(0),4).tolist()}")
print("\nwrote s3b_norm_matrix.csv, s3b_leakage.json")
