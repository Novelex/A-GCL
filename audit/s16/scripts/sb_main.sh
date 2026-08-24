#!/bin/bash
#SBATCH --job-name=s16main
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --output=/users/3171356m/A-GCL/audit/s16/logs/main_%a.out
#SBATCH --error=/users/3171356m/A-GCL/audit/s16/logs/main_%a.err
#SBATCH --signal=B:USR1@300
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 S11_NJOBS=4
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/A-GCL/audit/s16/scripts/s16_worker.py main $SLURM_ARRAY_TASK_ID
