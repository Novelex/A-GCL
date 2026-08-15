#!/bin/bash
#SBATCH --job-name=agcl-layer6-loss
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%j.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%j.err

set -euo pipefail

cd /users/3171356m/A-GCL
export PYTHONPATH="/users/3171356m/A-GCL:${PYTHONPATH:-}"

/users/3171356m/A-GCL/.venv/bin/python layertesting/layer6/test_layer6_loss_functions.py --epochs 30
