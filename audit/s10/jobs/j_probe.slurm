#!/bin/bash
#SBATCH --job-name=s10-probe
#SBATCH --partition=gpu-h100
#SBATCH --array=0-32%33
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/3171356m/agcl_audit_s0/s10/logs/probe_%A_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s10/logs/probe_%A_%a.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 S75_NJOBS=4 MPLBACKEND=Agg
cd /users/3171356m/agcl_audit_s0/s10
/users/3171356m/A-GCL/.venv/bin/python w_probe.py --task ${SLURM_ARRAY_TASK_ID}
