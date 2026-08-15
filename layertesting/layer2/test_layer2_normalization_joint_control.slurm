#!/bin/bash
#SBATCH --job-name=agcl-layer2-jointctrl
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%j.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%j.err

set -euo pipefail

cd /users/3171356m/A-GCL

/users/3171356m/A-GCL/.venv/bin/python layertesting/layer2/test_layer2_normalization_joint_control.py
