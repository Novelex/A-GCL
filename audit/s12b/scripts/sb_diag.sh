#!/bin/bash
#SBATCH --job-name=s12bdiag
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12b/logs/diag.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12b/logs/diag.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache OMP_NUM_THREADS=4
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12b/scripts/diag_cuda.py
