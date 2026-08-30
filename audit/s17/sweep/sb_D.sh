#!/bin/bash
#SBATCH --job-name=s17swD
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --array=0-151%60
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/users/3171356m/A-GCL/audit/s17/logs/sweep/D_%A_%a.out
#SBATCH --error=/users/3171356m/A-GCL/audit/s17/logs/sweep/D_%A_%a.err
# DEVIATION_01 Stage D: confirmatory LOSO wave. 19 folds x 8 inputs = 152 units,
# one unit per task, K=30 configs frozen in B/TOPK.json. 4 single-thread trainings per task.
set -euo pipefail
unset SWEEP_SMOKE_REPS SWEEP_SMOKE_CLFS SWEEP_SMOKE_PROTOS SWEEP_SMOKE_GRID SWEEP_ROOT SWEEP_NTASK SWEEP_NTASK_B SWEEP_NPAR
find /users/3171356m/A-GCL/audit/s17/sweep -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache PYTHONUNBUFFERED=1
export JOBLIB_TEMP_FOLDER=/tmp
S=/users/3171356m/A-GCL/audit/s17/sweep
PY=/users/3171356m/A-GCL/.venv/bin/python
R=/users/3171356m/A-GCL/audit/s17/runs/sweep
# grid + data + policy guards: any drift STOPS the task before it computes anything
$PY -c "
import sys; sys.path.insert(0,\"$S\"); import sweep_lib as L, mlp_lib as M
assert len(L.rep_catalogue())==43 and len(L.clf_catalogue())==75 and len(M.grid_B())==540, \"GRID CHANGED. STOP.\"
assert M.POL.policy_hash()==\"798ed7790c1ddabc\", \"PROD POLICY CHANGED. STOP.\"
assert L.ROOT==\"/users/3171356m/A-GCL/audit/s17/runs/sweep/\", L.ROOT
assert L.protocols()==(\"lab\",\"site\",\"loso\"), L.protocols()
L.data()   # re-verifies raw M1, frozen FC chain, LAB==S5.5 folds
"
test -f $R/A/AGG_A.json || { echo "AGG_A.json absent. STOP."; exit 3; }
test -f $R/B/TOPK.json  || { echo "B/TOPK.json absent - selK.py has not frozen the top-K. STOP."; exit 3; }
test -f $R/B/inputs.json || { echo "B/inputs.json absent. STOP."; exit 3; }
$PY -c "
import json
a=json.load(open('$R/A/AGG_A.json'))
assert a['lab'].get('gate_pass') is True, 'LAB GATE FAILED IN STAGE A. STOP.'
assert all(a[p].get('complete') for p in ('lab','site','loso')), 'STAGE A INCOMPLETE. STOP.'
d=json.load(open('$R/B/TOPK.json'))
assert d['deviation']=='DEVIATION_01' and d['K']==30, 'TOPK.json is not the pre-registered wave. STOP.'
assert len(d['configs'])==30==len(set(d['configs'])), 'TOPK config list malformed. STOP.'
assert len(json.load(open('$R/B/inputs.json')))==8, 'inputs.json is not the 8 frozen inputs. STOP.'
"
export SWEEP_NPAR=4
$PY -u $S/taskD.py $SLURM_ARRAY_TASK_ID
