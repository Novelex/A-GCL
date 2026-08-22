#!/bin/bash
#SBATCH --job-name=s12a5w2
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12a5/logs/w2_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12a5/logs/w2_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache S11_NJOBS=8 OMP_NUM_THREADS=2
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12a5/scripts/w_wave2.py $SLURM_ARRAY_TASK_ID
