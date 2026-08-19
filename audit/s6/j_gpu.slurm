#!/bin/bash
#SBATCH --job-name=s6-gpu
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s6/gpu_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s6/gpu_%j.err
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd /users/3171356m/agcl_audit_s0/s6
/users/3171356m/A-GCL/.venv/bin/python tiny_test.py cuda
