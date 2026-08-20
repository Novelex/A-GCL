#!/bin/bash
#SBATCH --job-name=s8-pilot-P
#SBATCH --partition=gpu-h100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/3171356m/agcl_audit_s0/s8/logs/pilot_P_%j.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s8/logs/pilot_P_%j.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
cd /users/3171356m/agcl_audit_s0/s8
echo "host=$(hostname) job=$SLURM_JOB_ID git=$(git -C /users/3171356m/A-GCL rev-parse HEAD) seed=20260818 date=$(date -Is)"
/users/3171356m/A-GCL/.venv/bin/python w_pilot.py --cfg P --epochs 200
