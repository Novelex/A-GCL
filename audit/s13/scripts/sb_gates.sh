#!/bin/bash
#SBATCH --job-name=s13gate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/users/3171356m/A-GCL/audit/s13/logs/gates.out
#SBATCH --error=/users/3171356m/A-GCL/audit/s13/logs/gates.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 S11_NJOBS=4
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/A-GCL/audit/s13/scripts/gates.py all
