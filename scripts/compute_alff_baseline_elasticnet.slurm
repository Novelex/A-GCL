#!/bin/bash
#SBATCH --job-name=agcl-alff-elasticnet
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%A_%a.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%A_%a.err

set -euo pipefail

cd /users/3171356m/A-GCL

# array index -> feature-set (ElasticNet-LR only, both already-finished
# LinearSVC results from job 1867572 are untouched)
FEATURE_SETS=(old new)
FEATURE_SET=${FEATURE_SETS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: feature-set=$FEATURE_SET classifier=ElasticNet-LR"

/users/3171356m/A-GCL/.venv/bin/python scripts/compute_alff_baseline.py \
  --feature-set "$FEATURE_SET" \
  --classifier ElasticNet-LR \
  --n-jobs 8
