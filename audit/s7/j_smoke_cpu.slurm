#!/bin/bash
#SBATCH --job-name=s7-smokecpu
#SBATCH --partition=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/3171356m/agcl_audit_s0/s7/logs/smokecpu_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s7/logs/smokecpu_%j.err
set -euo pipefail
S7=/users/3171356m/agcl_audit_s0/s7
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export S7_SMOKE_TAG=smoke_cpu
mkdir -p $S7/smoke_cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/users/3171356m/A-GCL/.venv/bin/python
cd $S7
echo "host=$(hostname) job=$SLURM_JOB_ID git=$(git -C /users/3171356m/A-GCL rev-parse HEAD) date=$(date -Is)"
rm -rf smoke_cpu/J1 smoke_cpu/J2 smoke_cpu/J3a smoke_cpu/J3b
$PY assert_contract.py
$PY w_j1.py  --smoke
$PY w_j2.py  --smoke --device cpu
$PY w_j3a.py --smoke --device cpu
$PY w_j3b.py --smoke
echo "--- determinism: identical rerun of one J3a unit ---"
$PY - <<'PY'
import sys,hashlib,numpy as np,os,shutil
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C
p=C.S7+"smoke_cpu/J3a/emb_s000_P_B.npz"; a=hashlib.sha256(open(p,'rb').read()).hexdigest()
z=np.load(p); h0=z['h'].copy()
os.remove(p); os.remove(p.replace('.npz','.DONE')); os.remove(p.replace('.npz','.json'))
os.system(f"{sys.executable} /users/3171356m/agcl_audit_s0/s7/w_j3a.py --smoke --device cpu >/dev/null 2>&1")
h1=np.load(p)['h']
print("determinism identical:", bool(np.array_equal(h0,h1)))
assert np.array_equal(h0,h1), "NON-DETERMINISTIC RERUN"
PY
echo "--- benchmark: ONE FULL 954-subject random-encoder unit on CPU ---"
$PY - <<'PY'
import sys,time,numpy as np,torch,json
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C
idx=np.arange(954); m=C.build_model("O",0,device="cpu"); m.eval()
t0=time.time(); H=[];Z=[]
with torch.no_grad():
    for s in range(0,954,64):
        ch=idx[s:s+64]; x,ei,ew,bt=C.batch_graphs(ch,"B",device="cpu")
        h,z,_=m.encode(bt,x,ei,None,ew); H.append(h.cpu()); Z.append(z.cpu())
fwd=time.time()-t0
h=torch.cat(H).numpy().astype(np.float32); z=torch.cat(Z).numpy().astype(np.float32)
t1=time.time(); np.savez_compressed("/tmp/_bench_cpu.npz",h=h,z=z); ser=time.time()-t1
from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold; from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
D=C.load_all(); y=D["y"]; F,_=C.splits()
t2=time.time(); oof=np.full(954,np.nan)
for tr,te in F:
    gs=GridSearchCV(Pipeline([("sc",StandardScaler()),("clf",LinearSVC(dual="auto",max_iter=20000))]),
        {"clf__C":[1e-3,1e-2,1e-1,1,10]},cv=StratifiedKFold(5,shuffle=True,random_state=20260818),
        scoring="roc_auc",n_jobs=1); gs.fit(h[tr],y[tr]); oof[te]=gs.decision_function(h[te])
prb=time.time()-t2
r=dict(device="cpu",forward_s=round(fwd,2),serialize_s=round(ser,2),probe_s=round(prb,2),
       total_s=round(fwd+ser+prb,2),h_shape=list(h.shape),z_shape=list(z.shape))
json.dump(r,open("/users/3171356m/agcl_audit_s0/s7/bench/bench_cpu.json","w"),indent=1)
np.savez("/users/3171356m/agcl_audit_s0/s7/bench/ref_cpu.npz",h=h,z=z)
print("BENCH_CPU",json.dumps(r))
PY
touch $S7/smoke_cpu/SMOKE_CPU_PASS
echo "SMOKE_CPU_PASS"
