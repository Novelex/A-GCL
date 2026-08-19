#!/bin/bash
#SBATCH --job-name=s6-cpunode
#SBATCH --partition=gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s6/cpunode_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s6/cpunode_%j.err
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
cd /users/3171356m/agcl_audit_s0/s6
/users/3171356m/A-GCL/.venv/bin/python tiny_test.py cpu
