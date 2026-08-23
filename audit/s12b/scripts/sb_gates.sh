#!/bin/bash
#SBATCH --job-name=s12bgate
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/users/3171356m/agcl_audit_s0/s12b/logs/gates.out
#SBATCH --error=/users/3171356m/agcl_audit_s0/s12b/logs/gates.err
set -euo pipefail
export PYTHONPYCACHEPREFIX=/users/3171356m/agcl_audit_s0/pycache S11_NJOBS=10 OMP_NUM_THREADS=10
PY=/users/3171356m/A-GCL/.venv/bin/python
S=/users/3171356m/agcl_audit_s0/s12b/scripts
$PY $S/gate0.py
$PY $S/gate1.py
$PY $S/gate2.py
echo ALL_GATES_PASS
