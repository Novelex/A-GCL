#!/bin/bash
#SBATCH --job-name=s16e2c
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/users/3171356m/A-GCL/audit/s16/logs/e2e_check.out
#SBATCH --error=/users/3171356m/A-GCL/audit/s16/logs/e2e_check.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache
export OMP_NUM_THREADS=4 S11_NJOBS=4
/users/3171356m/A-GCL/.venv/bin/python -u /users/3171356m/A-GCL/audit/s16/scripts/_e2e_check.py
