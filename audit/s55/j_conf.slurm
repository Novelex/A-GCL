#!/bin/bash
#SBATCH --job-name=s55-conf
#SBATCH --partition=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s55/conf_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s55/conf_%j.err
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
/users/3171356m/A-GCL/.venv/bin/python /users/3171356m/agcl_audit_s0/s55/w_conf.py
