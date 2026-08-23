#!/bin/bash
#SBATCH --job-name=s12bfin
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12b/logs/final.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12b/logs/final.err
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache S11_NJOBS=4 OMP_NUM_THREADS=4
PY=/users/3171356m/A-GCL/.venv/bin/python
S=/users/3171356m/agcl_audit_s0/s12b/scripts
$PY $S/consolidate.py          # builds csv + BEST_CONFIG.json + plots
$PY $S/t1_controls.py          # P-lab / P-roi on winner + top-5 tensor dumps
$PY $S/consolidate.py          # refresh manifest with control artifacts
echo S12B_ALL_COMPUTE_DONE
