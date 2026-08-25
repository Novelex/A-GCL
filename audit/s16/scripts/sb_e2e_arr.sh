#!/bin/bash
#SBATCH --job-name=s16e2a
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --array=0-28%29
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/users/3171356m/A-GCL/audit/s16/logs/e2ea_%a.out
#SBATCH --error=/users/3171356m/A-GCL/audit/s16/logs/e2ea_%a.err
set -euo pipefail
export S16_NS=e2e
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 S11_NJOBS=4
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache PYTHONUNBUFFERED=1
S=/users/3171356m/A-GCL/audit/s16/scripts
/users/3171356m/A-GCL/.venv/bin/python -u -c "
import sys; sys.path.insert(0,'$S'); import _e2e_run as R
n=len(R.targets()); assert n==29, f'E2E target list changed: {n} != 29. STOP.'"
/users/3171356m/A-GCL/.venv/bin/python -u $S/_e2e_run.py $SLURM_ARRAY_TASK_ID
