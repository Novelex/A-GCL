#!/bin/bash
#SBATCH --job-name=agcl-download-alff-rois
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%j.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%j.err

set -euo pipefail

cd /users/3171356m/A-GCL

/users/3171356m/A-GCL/.venv/bin/python scripts/download_alff_rois.py \
  --raw-dir data/raw \
  --output-dir data/ALFF_need \
  --workers 4 \
  --retries 5
