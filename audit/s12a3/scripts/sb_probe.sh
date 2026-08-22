#!/bin/bash
#SBATCH --job-name=s12a3p
#SBATCH --array=0-36
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12a3/logs/p_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12a3/logs/p_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache
export S11_NJOBS=10 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12a3/scripts/w_probe3.py $SLURM_ARRAY_TASK_ID
