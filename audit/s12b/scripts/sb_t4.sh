#!/bin/bash
#SBATCH --job-name=s12bt4
#SBATCH --partition=gpu-l40s
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12b/logs/t4_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12b/logs/t4_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache OMP_NUM_THREADS=2
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12b/scripts/t4_job.py $SLURM_ARRAY_TASK_ID
