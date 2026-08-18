#!/bin/bash
#SBATCH --job-name=s3c-lvl1
#SBATCH --partition=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s3c/lvl1_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s3c/lvl1_%j.err
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s3c/w_level1.py
