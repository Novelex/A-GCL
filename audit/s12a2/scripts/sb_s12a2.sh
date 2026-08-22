#!/bin/bash
#SBATCH --job-name=s12a2
#SBATCH --array=0-18
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12a2/logs/slurm_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12a2/logs/slurm_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache
export S11_NJOBS=8 S12A2_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12a2/scripts/w_s12a2.py $SLURM_ARRAY_TASK_ID
