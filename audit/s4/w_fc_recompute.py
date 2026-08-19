"""S4: independently recompute Pearson FC from the EXACT filt_noglobal rois_aal
source the generator used (GraSTIACL data/raw/rois_aal), compare to stored cropped_matrix."""
import sys, re, os, numpy as np, pandas as pd, scipy.io as sio
tid=int(sys.argv[1]); ntask=int(sys.argv[2])
A="/users/3171356m/A-GCL/"; G="/mnt/scratch/users/3171356m/muhammad-GraSTIACL/data/raw/rois_aal/"
CUT=9001; EPS=1e-8
coh=pd.read_csv("/users/3171356m/agcl_audit_s0/s1_audit_table.csv")
mine=coh.iloc[tid::ntask]
rows=[]
for r in mine.itertuples():
    fid=r.subject_id; cls="ASD" if r.dx_storage=="ASD" else "NC"
    p=G+f"{fid}_rois_aal.1D"
    hdr=open(p).readline()
    lab=np.array([int(re.sub(r"^#","",t)) for t in hdr.split()])
    ts=np.loadtxt(p, skiprows=1)[:, lab<CUT]          # notebook: keep_mask = labels < 9001
    assert ts.shape[1]==90
    W=np.corrcoef(ts, rowvar=False)                   # notebook: signed global PCC
    S=sio.loadmat(A+f"data/raw/{cls}_ADJ/{fid}_adj.mat")["cropped_matrix"]
    d=np.abs(W-S)
    roi_std=ts.std(axis=0)
    rows.append(dict(file_id=fid, T=ts.shape[0],
        max_abs_err=float(d.max()), mean_abs_err=float(d.mean()),
        corr=float(np.corrcoef(W.ravel(),S.ravel())[0,1]),
        n_mismatch_1em8=int((d>1e-8).sum()), n_mismatch_1em10=int((d>1e-10).sum()),
        n_dead_roi=int((roi_std<=EPS).sum())))
pd.DataFrame(rows).to_csv(f"/users/3171356m/agcl_audit_s0/s4/rec_{tid:02d}.csv",index=False)
print("TASK COMPLETE",tid,len(rows))
