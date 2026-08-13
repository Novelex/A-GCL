#!/bin/bash
#SBATCH --job-name=agcl-smoke-queue-cpu1ep
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%j.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%j.err

set -euo pipefail

cd /users/3171356m/A-GCL

/users/3171356m/A-GCL/.venv/bin/python agcl_ABIDE_queue.py \
  --path data --name ABIDE \
  --epochs 1 --eval_interval 1 --batch_size 32
