#!/bin/bash
#SBATCH --job-name=s75rp
#SBATCH --partition=gpu-h100
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/3171356m/agcl_audit_s0/s75/logs/rp_%A_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s75/logs/rp_%A_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export S75_NJOBS=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /users/3171356m/agcl_audit_s0/s75
/users/3171356m/A-GCL/.venv/bin/python w_rest.py --sec roiperm --task ${SLURM_ARRAY_TASK_ID} --ntask 10
