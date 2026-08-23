#!/bin/bash
#SBATCH --job-name=s15smoke
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --output=/users/3171356m/A-GCL/audit/s15/logs/smoke.out
#SBATCH --error=/users/3171356m/A-GCL/audit/s15/logs/smoke.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 S11_NJOBS=8
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/A-GCL/audit/s15/scripts/s15_smoke.py
