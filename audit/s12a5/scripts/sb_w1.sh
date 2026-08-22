#!/bin/bash
#SBATCH --job-name=s12a5w1
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12a5/logs/w1_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12a5/logs/w1_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache S11_NJOBS=2 OMP_NUM_THREADS=2
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12a5/scripts/w_wave1.py $SLURM_ARRAY_TASK_ID
