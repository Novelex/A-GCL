#!/bin/bash
#SBATCH --job-name=s12bt1
#SBATCH --partition=gpu-l40s
#SBATCH --array=0-17%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12b/logs/t1_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12b/logs/t1_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache
export S12B_WORKERS=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12b/scripts/t1_job.py $SLURM_ARRAY_TASK_ID
