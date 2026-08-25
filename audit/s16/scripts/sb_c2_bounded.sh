#!/bin/bash
#SBATCH --job-name=s16c2b
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/users/3171356m/A-GCL/audit/s16/logs/c2_bounded.out
#SBATCH --error=/users/3171356m/A-GCL/audit/s16/logs/c2_bounded.err
set -euo pipefail
export S16_NS=prod
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 S11_NJOBS=8
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache PYTHONUNBUFFERED=1
/users/3171356m/A-GCL/.venv/bin/python -u \
  /users/3171356m/A-GCL/audit/s16/scripts/s16_c2_bounded_run.py
