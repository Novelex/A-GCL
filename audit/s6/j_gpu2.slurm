#!/bin/bash
#SBATCH --job-name=s6-gpu
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s6/gpu_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s6/gpu_%j.err
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cd /users/3171356m/agcl_audit_s0/s6
/users/3171356m/A-GCL/.venv/bin/python tiny_test.py cuda
