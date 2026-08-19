#!/bin/bash
#SBATCH --job-name=s7-J3b
#SBATCH --array=0-15%16
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s7/logs/J3b_%A_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s7/logs/J3b_%A_%a.err
#SBATCH --partition=gpu-h100
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
S7=/users/3171356m/agcl_audit_s0/s7
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/users/3171356m/A-GCL/.venv/bin/python
cd $S7
echo "host=$(hostname) job=${SLURM_JOB_ID} arr=${SLURM_ARRAY_TASK_ID:-NA} git=$(git -C /users/3171356m/A-GCL rev-parse HEAD) date=$(date -Is)"
$PY w_j3b.py --task ${SLURM_ARRAY_TASK_ID} --ntask 16
