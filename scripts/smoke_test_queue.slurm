#!/bin/bash
#SBATCH --job-name=agcl-smoke-queue
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --exclude=node07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%j.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%j.err

set -euo pipefail

cd /users/3171356m/A-GCL

/users/3171356m/A-GCL/.venv/bin/python agcl_ABIDE_queue.py --path data --name ABIDE --epochs 2 --eval_interval 2 --batch_size 32
