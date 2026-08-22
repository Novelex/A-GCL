#!/bin/bash
#SBATCH --job-name=s12a4b4
#SBATCH --array=0-2
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12a4b/logs/s4_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12a4b/logs/s4_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache S11_NJOBS=12 OMP_NUM_THREADS=2
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s12a4b/scripts/w_step4.py $SLURM_ARRAY_TASK_ID
