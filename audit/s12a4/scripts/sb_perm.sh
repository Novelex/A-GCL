#!/bin/bash
#SBATCH --job-name=s12a4perm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12a4/logs/perm.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12a4/logs/perm.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache S11_NJOBS=2 OMP_NUM_THREADS=2
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12a4/scripts/w_perm.py
