#!/bin/bash
#SBATCH --job-name=a1-donor
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12a1/logs/d_%A_%a.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12a1/logs/d_%A_%a.err
#SBATCH --partition=gpu-h100
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache_slurm
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 S11_NJOBS=4 MPLBACKEND=Agg
cd /users/3171356m/agcl_audit_s0/s12a1
echo "host=$(hostname) job=${SLURM_JOB_ID} git=$(git -C /users/3171356m/A-GCL rev-parse HEAD) date=$(date -Is)"
S=$((SLURM_ARRAY_TASK_ID % 3)); E=$([ $((SLURM_ARRAY_TASK_ID / 3)) -eq 0 ] && echo ord || echo loso)
/users/3171356m/A-GCL/.venv/bin/python scripts/w_donor.py --seed $S --eval $E
