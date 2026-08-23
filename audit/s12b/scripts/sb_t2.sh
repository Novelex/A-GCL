#!/bin/bash
#SBATCH --job-name=s12bt2
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --array=0-35%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12b/logs/t2_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12b/logs/t2_%a.err
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache OMP_NUM_THREADS=2
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12b/scripts/t2_job.py $SLURM_ARRAY_TASK_ID
