#!/bin/bash
#SBATCH --job-name=agcl-smoke
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%j.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%j.err

set -euo pipefail

cd /users/3171356m/A-GCL

echo "=== nvidia-smi ==="
nvidia-smi

/users/3171356m/A-GCL/.venv/bin/python agcl_ABIDE.py --path data --name ABIDE --epochs 1 --eval_interval 1 --batch_size 32
