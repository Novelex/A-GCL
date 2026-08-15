#!/bin/bash
#SBATCH --job-name=agcl-signal-localization
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-1
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%A_%a.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%A_%a.err

set -euo pipefail

cd /users/3171356m/A-GCL
export PYTHONPATH="/users/3171356m/A-GCL:${PYTHONPATH:-}"

SOURCES=(old new)
SOURCE=${SOURCES[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: source=$SOURCE"

/users/3171356m/A-GCL/.venv/bin/python layertesting/layer3/test_signal_localization.py --source "$SOURCE"
