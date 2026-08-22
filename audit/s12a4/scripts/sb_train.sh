#!/bin/bash
#SBATCH --job-name=s12a4t
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12a4/logs/t_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12a4/logs/t_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache
export S11_NJOBS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12a4/scripts/w_train4.py $SLURM_ARRAY_TASK_ID
