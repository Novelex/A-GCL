#!/bin/bash
#SBATCH --job-name=s13bnt
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --array=0-29%30
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/users/3171356m/A-GCL/audit/s13/logs/u_%a.out
#SBATCH --error=/users/3171356m/A-GCL/audit/s13/logs/u_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 S11_NJOBS=4
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/A-GCL/audit/s13/scripts/w_s13.py $SLURM_ARRAY_TASK_ID
