#!/bin/bash
#SBATCH --job-name=agcl-pooling-ablation
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-5
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%A_%a.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%A_%a.err

set -euo pipefail

cd /users/3171356m/A-GCL
export PYTHONPATH="/users/3171356m/A-GCL:${PYTHONPATH:-}"

# array index -> (variant, source) combination
VARIANTS=(mean_pool mean_pool l2norm l2norm 1layer 1layer)
SOURCES=(old new old new old new)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SOURCE=${SOURCES[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: variant=$VARIANT source=$SOURCE"

/users/3171356m/A-GCL/.venv/bin/python layertesting/layer3/test_pooling_ablation.py \
  --variant "$VARIANT" \
  --source "$SOURCE"
