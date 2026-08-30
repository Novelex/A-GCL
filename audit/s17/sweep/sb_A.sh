#!/bin/bash
#SBATCH --job-name=s17swA
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --array=0-639%60
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/users/3171356m/A-GCL/audit/s17/logs/sweep/A_%A_%a.out
#SBATCH --error=/users/3171356m/A-GCL/audit/s17/logs/sweep/A_%A_%a.err
# Stage A: 1,276 units (29 folds x 43 reps + 29 shuffled-label) packed 2-per-task. CPU only, no --gres.
set -euo pipefail
unset SWEEP_SMOKE_REPS SWEEP_SMOKE_CLFS SWEEP_SMOKE_PROTOS SWEEP_SMOKE_GRID SWEEP_ROOT SWEEP_NTASK SWEEP_NTASK_B SWEEP_NPAR
find /users/3171356m/A-GCL/audit/s17/sweep -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache PYTHONUNBUFFERED=1
export JOBLIB_TEMP_FOLDER=/tmp
S=/users/3171356m/A-GCL/audit/s17/sweep
PY=/users/3171356m/A-GCL/.venv/bin/python
# grid + data + policy guards: any drift STOPS the task before it computes anything
$PY -c "
import sys; sys.path.insert(0,\"$S\"); import sweep_lib as L, mlp_lib as M
assert len(L.rep_catalogue())==43 and len(L.clf_catalogue())==75 and len(M.grid_B())==540, \"GRID CHANGED. STOP.\"
assert M.POL.policy_hash()==\"798ed7790c1ddabc\", \"PROD POLICY CHANGED. STOP.\"
assert L.ROOT==\"/users/3171356m/A-GCL/audit/s17/runs/sweep/\", L.ROOT
assert L.protocols()==(\"lab\",\"site\",\"loso\"), L.protocols()
L.data()   # re-verifies raw M1, frozen FC chain, LAB==S5.5 folds
"
export SWEEP_NTASK=640
$PY -u $S/taskA.py $SLURM_ARRAY_TASK_ID
