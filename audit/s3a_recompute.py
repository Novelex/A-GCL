"""S3A independent ALFF recomputation - written from the documented spec,
does NOT import either generator. Read-only."""
import numpy as np, pandas as pd, json, re, os
import nibabel as nib
from scipy.signal import detrend

A="/users/3171356m/A-GCL/"
G="/mnt/scratch/users/3171356m/muhammad-GraSTIACL/data/raw/"
ATL="/mnt/scratch/users/3171356m/muhammad-GraSTIACL/data/software/DPABI/Templates/AAL_61x73x61_YCG.nii"
OUT="/users/3171356m/agcl_audit_s0/"
BANDS=[(0.010,0.027),(0.027,0.073),(0.010,0.080)]
NB=["slow5","slow4","classical"]

sample=json.load(open(OUT+"s3a_sample.json"))
tr_tab={r.FILE_ID:float(r.TR_seconds) for r in pd.read_csv(A+"data/subject_tr.csv").itertuples()}
atlas=np.round(nib.load(ATL).get_fdata()).astype(int)

z1=np.load(A+"ALFF_func_proc/method1/alff_roi_first.npz")
z2=np.load(A+"ALFF_func_proc/method2/alff_voxel_first.npz")
zo=np.load(A+"data/ALFF_need/alff_new.npz")
i1={str(s):i for i,s in enumerate(z1["file_ids"])}
i2={str(s):i for i,s in enumerate(z2["file_ids"])}
io={str(s):i for i,s in enumerate(zo["file_ids"])}

def amp_spec(ts, tr, T):
    """2|rfft(ts, nfft)|/T  with nfft = 2^ceil(log2 T); returns (amp, freqs)."""
    nfft = 2**int(np.ceil(np.log2(T)))
    amp = 2*np.abs(np.fft.rfft(ts, n=nfft, axis=0))/T
    freqs = np.fft.rfftfreq(nfft, d=tr)
    return amp, freqs, nfft

def band_alff(ts, tr, T, upper_inclusive=True):
    amp, freqs, nfft = amp_spec(ts, tr, T)
    out=np.zeros((ts.shape[1], 3)); bins=[]
    for b,(lo,hi) in enumerate(BANDS):
        m = (freqs>=lo) & ((freqs<=hi) if upper_inclusive else (freqs<hi))
        bins.append(int(m.sum()))
        out[:,b]=amp[m].mean(axis=0)
    return out, bins, nfft, freqs

def load_1d90(fid):
    hdr=open(A+f"data/ALFF_need/rois_aal/{fid}_rois_aal.1D").readline()
    labels=np.array([int(re.sub(r"^#","",t)) for t in hdr.split()])
    raw=np.loadtxt(A+f"data/ALFF_need/rois_aal/{fid}_rois_aal.1D")
    return raw[:, labels<9001]

rows=[]
for fid in sample:
    tr=tr_tab[fid]; rec={"file_id":fid,"tr":tr}
    # ---------- OLD ----------
    ts=load_1d90(fid); T=ts.shape[0]
    old_mine,bins,nfft,freqs = band_alff(detrend(ts,axis=0), tr, T)
    old_stored=zo["alff"][io[fid]]
    # ---------- M1 / M2 ----------
    img=nib.load(G+f"func_preproc/{fid}_func_preproc.nii.gz"); d=img.get_fdata()
    Tf=d.shape[3]
    valid=np.all(np.isfinite(d),axis=3) & np.any(d!=0,axis=3)
    m1=np.zeros((90,3)); m2=np.zeros((90,3)); vox=[]
    for l in range(1,91):
        D=detrend(d[(atlas==l)&valid].T, axis=0)     # [T, V]
        vox.append(D.shape[1])
        m1[l-1]=band_alff(D.mean(axis=1,keepdims=True), tr, Tf)[0][0]
        m2[l-1]=band_alff(D, tr, Tf)[0].mean(axis=0)
    m1_stored=z1["alff"][i1[fid]]; m2_stored=z2["alff"][i2[fid]]
    def err(mine,stored):
        a=np.abs(mine-stored); rel=a/np.maximum(np.abs(stored),1e-300)
        return float(a.max()), float(a.mean()), float(rel.max()), float(rel.mean())
    rec.update(T_1d=T, T_nii=Tf, nfft=nfft, df=float(freqs[1]-freqs[0]),
               fs=1.0/tr, nyq=1.0/(2*tr), bins_slow5=bins[0], bins_slow4=bins[1],
               bins_classical=bins[2], min_valid_vox=int(min(vox)), max_valid_vox=int(max(vox)))
    for tag,mine,stored in (("OLD",old_mine,old_stored),("M1",m1,m1_stored),("M2",m2,m2_stored)):
        ma,me,rma,rme=err(mine,stored)
        rec[f"{tag}_max_abs"]=ma; rec[f"{tag}_mean_abs"]=me
        rec[f"{tag}_max_rel"]=rma; rec[f"{tag}_mean_rel"]=rme
    # boundary-convention sensitivity (exclusive upper edge)
    old_excl,bins_x,_,_ = band_alff(detrend(ts,axis=0), tr, T, upper_inclusive=False)
    rec["bins_excl"]=str(bins_x)
    rec["excl_vs_incl_max_abs"]=float(np.abs(old_excl-old_mine).max())
    rows.append(rec); print("done", fid, flush=True)

df=pd.DataFrame(rows); df.to_csv(OUT+"s3a_recompute.csv", index=False)
print(df.to_string(index=False))
