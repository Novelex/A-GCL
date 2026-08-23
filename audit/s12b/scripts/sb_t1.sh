#!/bin/bash
#SBATCH --job-name=s12bt1
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --array=5,9,10,11,17%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12b/logs/t1_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12b/logs/t1_%a.err
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache
export S12B_WORKERS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12b/scripts/t1_job.py $SLURM_ARRAY_TASK_ID
