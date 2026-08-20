#!/bin/bash
#SBATCH --job-name=s75
#SBATCH --partition=gpu-h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/3171356m/agcl_audit_s0/s75/logs/s75_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s75/logs/s75_%j.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd /users/3171356m/agcl_audit_s0/s75
echo "host=$(hostname) job=$SLURM_JOB_ID git=$(git -C /users/3171356m/A-GCL rev-parse HEAD) date=$(date -Is)"
/users/3171356m/A-GCL/.venv/bin/python w_s75.py --seeds 10 --permseeds 20
