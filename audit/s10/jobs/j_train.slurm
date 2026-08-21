#!/bin/bash
#SBATCH --job-name=s10
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/3171356m/agcl_audit_s0/s10/logs/%x_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s10/logs/%x_%j.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
cd /users/3171356m/agcl_audit_s0/s10
echo "host=$(hostname) job=$SLURM_JOB_ID arm=$1 seed=$2 git=$(git -C /users/3171356m/A-GCL rev-parse HEAD) date=$(date -Is)"
/users/3171356m/A-GCL/.venv/bin/python w_s10.py --arm $1 --seed $2 --device cuda --epochs 200
