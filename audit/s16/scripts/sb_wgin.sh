#!/bin/bash
#SBATCH --job-name=s16wgin
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --array=0-62%12
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/users/3171356m/A-GCL/audit/s16/logs/wgin_%a.out
#SBATCH --error=/users/3171356m/A-GCL/audit/s16/logs/wgin_%a.err
# SELF-CONTAINED: partition, array range, concurrency cap, CPUs, memory, wall time,
# requeue, signal, namespace and thread caps are ALL declared here. No GPU is
# requested (this cluster has only GPU partitions; CPU-only == no --gres).
# Resource overrides on the sbatch command line are NOT required and NOT relied on.
set -euo pipefail
export S16_NS=prod
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 JOBLIB_START_METHOD=loky S11_NJOBS=4
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache PYTHONUNBUFFERED=1
S=/users/3171356m/A-GCL/audit/s16/scripts
# STOP if the frozen grid changed
/users/3171356m/A-GCL/.venv/bin/python -u -c "
import sys; sys.path.insert(0,'$S'); import s16_grid as G
n={'bnt':len(G.BNTU),'wgin':len(G.WGINU),'ctrlu':len(G.CTRLU)}['wgin']
exp=63
assert n==exp, f'FROZEN GRID CHANGED: wgin has {n} units, expected {exp}. STOP.'
"
# SIGNAL FORWARDING (defect D45). --signal=B:USR1 delivers to the BATCH SHELL,
# not to the python child, so the worker's SIGUSR1 handler could never fire and the
# graceful-stop path was dead code. Run python in the background, forward the signal
# to it explicitly, and propagate its real exit status.
_fwd() { [ -n "${PY_PID:-}" ] && kill -USR1 "$PY_PID" 2>/dev/null || true; }
trap _fwd USR1
/users/3171356m/A-GCL/.venv/bin/python -u $S/s16_worker.py wgin $SLURM_ARRAY_TASK_ID prod &
PY_PID=$!
# `wait` returns early when a trap fires; keep waiting until the child is really gone.
while :; do
  if wait "$PY_PID"; then RC=0; else RC=$?; fi
  kill -0 "$PY_PID" 2>/dev/null || break
done
exit "$RC"
