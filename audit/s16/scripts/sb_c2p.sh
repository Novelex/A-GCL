#!/bin/bash
#SBATCH --job-name=s16c2p
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/users/3171356m/A-GCL/audit/s16/logs/c2_precision.out
#SBATCH --error=/users/3171356m/A-GCL/audit/s16/logs/c2_precision.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 S11_NJOBS=8
/users/3171356m/A-GCL/.venv/bin/python -u /users/3171356m/A-GCL/audit/s16/scripts/s16_c2_precision.py
