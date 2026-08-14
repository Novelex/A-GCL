#!/bin/bash
#SBATCH --job-name=agcl-compute-alff
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%j.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%j.err

set -euo pipefail

cd /users/3171356m/A-GCL

/users/3171356m/A-GCL/.venv/bin/python scripts/compute_alff.py \
  --rois-dir data/ALFF_need/rois_aal \
  --subject-tr data/subject_tr.csv \
  --output data/ALFF_need/alff_new.npz
