#!/bin/bash
#SBATCH --job-name=s12bt3
#SBATCH --partition=gpu-l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12b/logs/t3.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12b/logs/t3.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache S11_NJOBS=8 OMP_NUM_THREADS=8
PY=/users/3171356m/A-GCL/.venv/bin/python
S=/users/3171356m/agcl_audit_s0/s12b/scripts
$PY $S/t3_job.py
$PY $S/t2_classical.py
