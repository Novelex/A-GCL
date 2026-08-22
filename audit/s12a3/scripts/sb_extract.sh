#!/bin/bash
#SBATCH --job-name=s12a3x
#SBATCH --array=0-17
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12a3/logs/x_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12a3/logs/x_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12a3/scripts/w_extract3.py $SLURM_ARRAY_TASK_ID
