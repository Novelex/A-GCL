"""Full-cohort independent ALFF recomputation, one chunk per SLURM array task."""
import numpy as np, pandas as pd, json, re, os, sys
import nibabel as nib
from scipy.signal import detrend
A="/users/3171356m/A-GCL/"; G="/mnt/scratch/users/3171356m/muhammad-GraSTIACL/data/raw/"
ATL="/mnt/scratch/users/3171356m/muhammad-GraSTIACL/data/software/DPABI/Templates/AAL_61x73x61_YCG.nii"
OUT="/users/3171356m/agcl_audit_s0/full/"
BANDS=[(0.010,0.027),(0.027,0.073),(0.010,0.080)]
cid=int(sys.argv[1]); nch=int(sys.argv[2])
os.makedirs(OUT, exist_ok=True)
coh=pd.read_csv("/users/3171356m/agcl_audit_s0/s1_audit_table.csv")["subject_id"].tolist()
mine=coh[cid::nch]
tr_tab={r.FILE_ID:float(r.TR_seconds) for r in pd.read_csv(A+"data/subject_tr.csv").itertuples()}
atlas=np.round(nib.load(ATL).get_fdata()).astype(int)
z1=np.load(A+"ALFF_func_proc/method1/alff_roi_first.npz"); z2=np.load(A+"ALFF_func_proc/method2/alff_voxel_first.npz")
zo=np.load(A+"data/ALFF_need/alff_new.npz")
i1={str(s):i for i,s in enumerate(z1["file_ids"])}; i2={str(s):i for i,s in enumerate(z2["file_ids"])}
io={str(s):i for i,s in enumerate(zo["file_ids"])}
def band(ts,tr,T):
    nfft=2**int(np.ceil(np.log2(T)))
    amp=2*np.abs(np.fft.rfft(ts,n=nfft,axis=0))/T; f=np.fft.rfftfreq(nfft,d=tr)
    o=np.zeros((ts.shape[1],3))
    for b,(lo,hi) in enumerate(BANDS): o[:,b]=amp[(f>=lo)&(f<=hi)].mean(axis=0)
    return o
def load1d(fid):
    h=open(A+f"data/ALFF_need/rois_aal/{fid}_rois_aal.1D").readline()
    lab=np.array([int(re.sub(r"^#","",t)) for t in h.split()])
    return np.loadtxt(A+f"data/ALFF_need/rois_aal/{fid}_rois_aal.1D")[:,lab<9001]
rows=[]
for fid in mine:
    tr=tr_tab[fid]
    ts=load1d(fid); old=band(detrend(ts,axis=0),tr,ts.shape[0])
    d=nib.load(G+f"func_preproc/{fid}_func_preproc.nii.gz").get_fdata(); T=d.shape[3]
    valid=np.all(np.isfinite(d),3)&np.any(d!=0,3)
    m1=np.zeros((90,3)); m2=np.zeros((90,3)); mv=10**9
    for l in range(1,91):
        D=detrend(d[(atlas==l)&valid].T,axis=0); mv=min(mv,D.shape[1])
        m1[l-1]=band(D.mean(axis=1,keepdims=True),tr,T)[0]
        m2[l-1]=band(D,tr,T).mean(axis=0)
    r={"file_id":fid,"T":T,"tr":tr,"min_valid_vox":int(mv)}
    for tag,mn,st in (("OLD",old,zo["alff"][io[fid]]),("M1",m1,z1["alff"][i1[fid]]),("M2",m2,z2["alff"][i2[fid]])):
        a=np.abs(mn-st); rel=a/np.maximum(np.abs(st),1e-300)
        r[f"{tag}_max_abs"]=float(a.max()); r[f"{tag}_max_rel"]=float(rel.max()); r[f"{tag}_mean_abs"]=float(a.mean())
    rows.append(r); print("done",fid,flush=True)
    del d
pd.DataFrame(rows).to_csv(OUT+f"chunk_{cid:03d}.csv",index=False)
print("CHUNK COMPLETE",cid)
