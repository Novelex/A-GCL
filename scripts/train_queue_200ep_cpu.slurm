#!/bin/bash
#SBATCH --job-name=agcl-queue-200ep-cpu
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%j.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%j.err

set -euo pipefail

cd /users/3171356m/A-GCL

/users/3171356m/A-GCL/.venv/bin/python agcl_ABIDE_queue.py \
  --path data --name ABIDE \
  --epochs 200 --eval_interval 5 --batch_size 32 \
  --cr_lambda 0.4 --memory_type queue --feature_type instance --max_length 256
