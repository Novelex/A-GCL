#!/bin/bash
#SBATCH --job-name=r0-full
#SBATCH --partition=gpu-h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/3171356m/agcl_audit_s0/r0/logs/full_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/r0/logs/full_%j.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export R0_NJOBS=16 R0_TORCH_THREADS=1
cd /users/3171356m/agcl_audit_s0/r0
echo "host=$(hostname) job=$SLURM_JOB_ID git=$(git -C /users/3171356m/A-GCL rev-parse HEAD) seed=123 date=$(date -Is)"
/users/3171356m/A-GCL/.venv/bin/python w_r0_run.py
