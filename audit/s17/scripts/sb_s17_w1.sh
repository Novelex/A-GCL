#!/bin/bash
#SBATCH --job-name=s17w1
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --array=0-107%30
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/users/3171356m/A-GCL/audit/s17/logs/w1_%a.out
#SBATCH --error=/users/3171356m/A-GCL/audit/s17/logs/w1_%a.err
# SELF-CONTAINED: partition, array range, throttle, CPUs, memory, wall time, requeue,
# signal and thread caps are ALL declared here. NO --gres: this cluster has only GPU
# partitions, so CPU-only means simply not requesting one. Resource overrides on the
# sbatch command line are neither required nor relied upon.
#
# 108 tasks = 4 arms (R1s, R1a, R1p, A7 reference) x 3 seeds x 9 folds.
# Throttled to 30 concurrent = 120 CPUs.
set -euo pipefail
export S16_NS=prod
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 S11_NJOBS=4
# PYTHONPYCACHEPREFIX points OUTSIDE the repository: the repo has tracked .pyc files.
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache PYTHONUNBUFFERED=1
S=/users/3171356m/A-GCL/audit/s17/scripts
mkdir -p /users/3171356m/A-GCL/audit/s17/logs

# STOP if the Wave-1 grid has drifted from the pre-registered shape.
/users/3171356m/A-GCL/.venv/bin/python -u -c "
import sys; sys.path.insert(0,'$S'); sys.path.insert(0,'/users/3171356m/A-GCL/audit/s16/scripts')
import s17_worker as W, s16_policy as PL
t=W.wave1_tasks('prod'); assert len(t)==108, f'FROZEN GRID CHANGED: {len(t)} tasks, expected 108. STOP.'
p=PL.get('prod'); assert p.policy_hash()=='798ed7790c1ddabc', 'PROD POLICY CHANGED. STOP.'
assert p.max_epochs==400, 'max_epochs CHANGED - cosine schedule would differ. STOP.'
assert '/audit/s17/' in W.root('prod') and '/audit/s16/' not in W.root('prod')
"

# SIGNAL FORWARDING: --signal=B:USR1 delivers to the BATCH SHELL, not the python
# child, so the signal is forwarded explicitly and the child's exit status propagated.
_fwd() { [ -n "${PY_PID:-}" ] && kill -USR1 "$PY_PID" 2>/dev/null || true; }
trap _fwd USR1
/users/3171356m/A-GCL/.venv/bin/python -u $S/s17_w1_run.py $SLURM_ARRAY_TASK_ID &
PY_PID=$!
while :; do
  if wait "$PY_PID"; then RC=0; else RC=$?; fi
  kill -0 "$PY_PID" 2>/dev/null || break
done
exit "$RC"
