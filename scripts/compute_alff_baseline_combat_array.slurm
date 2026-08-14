#!/bin/bash
#SBATCH --job-name=agcl-alff-combat
#SBATCH --partition=gpu-l40s,gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/users/3171356m/A-GCL/logs/%x_%A_%a.out
#SBATCH --error=/users/3171356m/A-GCL/logs/%x_%A_%a.err

set -euo pipefail

cd /users/3171356m/A-GCL

# array index -> (feature-set, classifier) combination
FEATURE_SETS=(old old new new)
CLASSIFIERS=(LinearSVC ElasticNet-LR LinearSVC ElasticNet-LR)

FEATURE_SET=${FEATURE_SETS[$SLURM_ARRAY_TASK_ID]}
CLASSIFIER=${CLASSIFIERS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: feature-set=$FEATURE_SET classifier=$CLASSIFIER (+ ComBat)"

/users/3171356m/A-GCL/.venv/bin/python scripts/compute_alff_baseline_combat.py \
  --feature-set "$FEATURE_SET" \
  --classifier "$CLASSIFIER" \
  --n-jobs 8
