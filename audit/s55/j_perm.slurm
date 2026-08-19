#!/bin/bash
#SBATCH --job-name=s55-perm
#SBATCH --partition=gpu-h100
#SBATCH --array=0-47%24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s55/perm/slurm_%A_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s55/perm/slurm_%A_%a.err
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s55/w_perm.py ${SLURM_ARRAY_TASK_ID} 48 480 4
