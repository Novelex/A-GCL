#!/bin/bash
#SBATCH --job-name=s75-rp
#SBATCH --array=0-4%5
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s75/logs/rp_%A_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s75/logs/rp_%A_%a.err
#SBATCH --partition=gpu-h100
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export S75_NJOBS=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /users/3171356m/agcl_audit_s0/s75
echo "host=$(hostname) job=${SLURM_JOB_ID} arr=${SLURM_ARRAY_TASK_ID:-NA} date=$(date -Is)"
/users/3171356m/A-GCL/.venv/bin/python w_rest.py --sec roiperm --task ${SLURM_ARRAY_TASK_ID} --ntask 5
