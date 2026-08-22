#!/bin/bash
#SBATCH --job-name=s12a4flat
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12a4/logs/flat.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12a4/logs/flat.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache S11_NJOBS=16 OMP_NUM_THREADS=2
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12a4/scripts/w_flat.py
