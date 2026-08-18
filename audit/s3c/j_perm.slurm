#!/bin/bash
#SBATCH --job-name=s3c-perm
#SBATCH --partition=gpu-h100
#SBATCH --array=0-119%24
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=08:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s3c/lvl3/slurm_%A_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s3c/lvl3/slurm_%A_%a.err
mkdir -p /users/3171356m/agcl_audit_s0/s3c/lvl3
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s3c/w_perm.py ${SLURM_ARRAY_TASK_ID} 120 1000 4
