#!/bin/bash
#SBATCH --job-name=s7-smokegpu
#SBATCH --partition=gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/3171356m/agcl_audit_s0/s7/logs/smokegpu_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s7/logs/smokegpu_%j.err
set -euo pipefail
S7=/users/3171356m/agcl_audit_s0/s7
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export S7_SMOKE_TAG=smoke_gpu
mkdir -p $S7/smoke_gpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/users/3171356m/A-GCL/.venv/bin/python
cd $S7
echo "host=$(hostname) job=$SLURM_JOB_ID git=$(git -C /users/3171356m/A-GCL rev-parse HEAD) date=$(date -Is)"
rm -rf smoke_gpu/J1 smoke_gpu/J2 smoke_gpu/J3a smoke_gpu/J3b
$PY w_j1.py  --smoke
$PY w_j2.py  --smoke --device cuda
$PY w_j3a.py --smoke --device cuda
$PY w_j3b.py --smoke
echo "--- determinism: identical rerun of one J3a unit ---"
$PY - <<'PY'
import sys,hashlib,numpy as np,os,shutil
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C
p=C.S7+"smoke_gpu/J3a/emb_s000_P_B.npz"; a=hashlib.sha256(open(p,'rb').read()).hexdigest()
z=np.load(p); h0=z['h'].copy()
os.remove(p); os.remove(p.replace('.npz','.DONE')); os.remove(p.replace('.npz','.json'))
os.system(f"{sys.executable} /users/3171356m/agcl_audit_s0/s7/w_j3a.py --smoke --device cuda >/dev/null 2>&1")
h1=np.load(p)['h']
d=float(np.abs(h0-h1).max()); rel=float((np.abs(h0-h1)/np.maximum(np.abs(h1),1e-12)).max())
print("GPU rerun bitwise identical:", bool(np.array_equal(h0,h1)))
print("GPU rerun max_abs_diff:", d, "max_rel_diff:", rel)
import json; json.dump(dict(bitwise=bool(np.array_equal(h0,h1)),max_abs=d,max_rel=rel),
    open("/users/3171356m/agcl_audit_s0/s7/bench/gpu_determinism.json","w"),indent=1)
# GPU need only be NUMERICALLY reproducible (CUDA scatter_add uses atomics); CPU must be bitwise.
print("NOTE: CUDA scatter_add uses atomics -> run-to-run non-determinism is EXPECTED and is\n      recorded as evidence. CPU remains the bitwise-reproducible reference and is the\n      resource chosen for all real S7 units.")
assert d < 0.05, f"GPU rerun differs implausibly: {d}"
PY
echo "--- benchmark: ONE FULL 954-subject random-encoder unit on CPU ---"
$PY - <<'PY'
import sys,time,numpy as np,torch,json
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C
idx=np.arange(954); m=C.build_model("O",0,device="cuda"); m.eval()
t0=time.time(); H=[];Z=[]
with torch.no_grad():
    for s in range(0,954,64):
        ch=idx[s:s+64]; x,ei,ew,bt=C.batch_graphs(ch,"B",device="cuda")
        h,z,_=m.encode(bt,x,ei,None,ew); H.append(h.cpu()); Z.append(z.cpu())
fwd=time.time()-t0
h=torch.cat(H).numpy().astype(np.float32); z=torch.cat(Z).numpy().astype(np.float32)
t1=time.time(); np.savez_compressed("/tmp/_bench_gpu.npz",h=h,z=z); ser=time.time()-t1
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
r=dict(device="cuda",forward_s=round(fwd,2),serialize_s=round(ser,2),probe_s=round(prb,2),
       total_s=round(fwd+ser+prb,2),h_shape=list(h.shape),z_shape=list(z.shape))
json.dump(r,open("/users/3171356m/agcl_audit_s0/s7/bench/bench_gpu.json","w"),indent=1)
np.savez("/users/3171356m/agcl_audit_s0/s7/bench/ref_gpu.npz",h=h,z=z)
print("BENCH_GPU",json.dumps(r))
PY
echo "--- CPU vs GPU numerical agreement (section 16) ---"
$PY - <<'PY2'
import sys,numpy as np,torch,json
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C
ref=np.load("/users/3171356m/agcl_audit_s0/s7/bench/ref_cpu.npz")
out={}
idx=np.arange(954)
for dev in ["cpu","cuda"]:
    m=C.build_model("O",0,device=dev); m.eval(); H=[];Z=[]
    with torch.no_grad():
        for s0 in range(0,954,64):
            ch=idx[s0:s0+64]; x,ei,ew,bt=C.batch_graphs(ch,"B",device=dev)
            h,z,_=m.encode(bt,x,ei,None,ew); H.append(h.float().cpu()); Z.append(z.float().cpu())
    out[dev]=(torch.cat(H).numpy(),torch.cat(Z).numpy())
r={}
for k,i in (("h",0),("z",1)):
    d=np.abs(out["cpu"][i]-out["cuda"][i])
    r[k]=dict(max_abs=float(d.max()),max_rel=float((d/np.maximum(np.abs(out["cpu"][i]),1e-12)).max()))
    d2=np.abs(ref[k]-out["cuda"][i]); r[k+"_vs_refcpu"]=dict(max_abs=float(d2.max()))
json.dump(r,open("/users/3171356m/agcl_audit_s0/s7/bench/cpu_gpu_numerical.json","w"),indent=1)
print("CPU_GPU_NUMERICAL",json.dumps(r))
PY2
touch $S7/smoke_gpu/SMOKE_GPU_PASS
echo "SMOKE_GPU_PASS"
