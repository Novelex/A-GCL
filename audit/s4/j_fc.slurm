#!/bin/bash
#SBATCH --job-name=s4-fc
#SBATCH --partition=gpu-h100
#SBATCH --array=0-15%16
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s4/slurm_%A_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s4/slurm_%A_%a.err
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s4/w_fc_recompute.py ${SLURM_ARRAY_TASK_ID} 16
