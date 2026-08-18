#!/bin/bash
#SBATCH --job-name=s3a-full
#SBATCH --partition=gpu-l40s
#SBATCH --array=0-31
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/full/slurm_%A_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/full/slurm_%A_%a.err
mkdir -p /users/3171356m/agcl_audit_s0/full
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
/users/3171356m/miniconda3/bin/python /users/3171356m/agcl_audit_s0/s3a_full_chunk.py ${SLURM_ARRAY_TASK_ID} 32
