#!/bin/bash
#SBATCH --job-name=s9-train
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/3171356m/agcl_audit_s0/s9/logs/train_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s9/logs/train_%j.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
cd /users/3171356m/agcl_audit_s0/s9
echo "host=$(hostname) job=$SLURM_JOB_ID git=$(git -C /users/3171356m/A-GCL rev-parse HEAD) seed=20260818 date=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
/users/3171356m/A-GCL/.venv/bin/python w_s9.py --device cuda --epochs 200
