#!/bin/bash
#SBATCH --job-name=s11
#SBATCH --partition=gpu-h100
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/3171356m/agcl_audit_s0/s11/logs/u_%A_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s11/logs/u_%A_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 S11_NJOBS=4 MPLBACKEND=Agg
cd /users/3171356m/agcl_audit_s0/s11
echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID git=$(git -C /users/3171356m/A-GCL rev-parse HEAD) date=$(date -Is)"
/users/3171356m/A-GCL/.venv/bin/python w_unit.py --task ${SLURM_ARRAY_TASK_ID} --pls128
