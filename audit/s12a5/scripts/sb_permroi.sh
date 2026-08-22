#!/bin/bash
#SBATCH --job-name=s12a5pr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12a5/logs/pr.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12a5/logs/pr.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache OMP_NUM_THREADS=2
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12a5/scripts/w_permroi.py
