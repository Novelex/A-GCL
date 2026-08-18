"""S3B.5 end-to-end reconstruction of norm_matrix from the surviving DPARSF mALFF maps."""
import nibabel as nib, numpy as np, pandas as pd, scipy.io as sio, glob, os
from nibabel.processing import resample_from_to
from scipy.stats import pearsonr, spearmanr
W="/mnt/scratch/users/3171356m/muhammad-GraSTIACL/data/dparsf_work/"
ATL="/mnt/scratch/users/3171356m/muhammad-GraSTIACL/data/software/atlases/aal_mask_pad.nii.gz"
A="/users/3171356m/A-GCL/"
BANDS=["slow5","slow4","classical"]
atlas_img=nib.load(ATL); atlas=np.round(atlas_img.get_fdata()).astype(int)
labels=sorted(int(l) for l in np.unique(atlas) if l!=0 and l<9001); assert len(labels)==90
masks=[atlas==l for l in labels]

# subject -> chunk per band, exactly as the notebook built it
def s2c(band):
    m={}
    for i in range(1,11):
        p=f"{W}full_{band}_chunk{i:02d}/SubjectList.txt"
        if os.path.exists(p):
            for sid in open(p).read().splitlines():
                if sid.strip(): m[sid.strip()]=f"{i:02d}"
    return m
S2C={b:s2c(b) for b in BANDS}
print({b:len(S2C[b]) for b in BANDS}, flush=True)

coh=pd.read_csv("/users/3171356m/agcl_audit_s0/s1_audit_table.csv")
sample=coh.subject_id.tolist()[::64][:15]
NM={r.subject_id: sio.loadmat(f"{A}data/raw/{'ASD' if r.dx_storage=='ASD' else 'NC'}_NF/{r.subject_id}_nf.mat")["norm_matrix"]
    for r in coh.itertuples() if r.subject_id in set(sample)}

rows=[]
for fid in sample:
    M=np.full((90,3),np.nan)
    ok=True
    for bi,b in enumerate(BANDS):
        ch=S2C[b].get(fid)
        if ch is None: ok=False; break
        p=f"{W}full_{b}_chunk{ch}/Results/ALFF_FunImgD/mALFFMap_{fid}.nii"
        if not os.path.exists(p): ok=False; break
        img=nib.load(p)
        r=resample_from_to(img,(atlas_img.shape,atlas_img.affine),order=1)  # trilinear ~ FLIRT
        d=r.get_fdata()
        for ri,mk in enumerate(masks): M[ri,bi]=d[mk].mean()
    if not ok: print("MISSING",fid,flush=True); continue
    mine=(M-M.mean(0,keepdims=True))/M.std(0,keepdims=True)   # per-subject per-band z-score
    st=NM[fid]
    rows.append(dict(file_id=fid, max_abs=float(np.abs(mine-st).max()),
        pearson=float(pearsonr(mine.ravel(),st.ravel())[0]),
        spearman=float(spearmanr(mine.ravel(),st.ravel())[0]),
        raw_min=float(M.min()), raw_max=float(M.max())))
    print("done",fid,rows[-1]["pearson"],flush=True)
d=pd.DataFrame(rows); d.to_csv("/users/3171356m/agcl_audit_s0/s3b5_reconstruct.csv",index=False)
print(d.to_string(index=False))
print(f"\nSUMMARY n={len(d)}  pearson min={d.pearson.min():.6f} median={d.pearson.median():.6f}")
print(f"         max_abs min={d.max_abs.min():.4f} max={d.max_abs.max():.4f}")
