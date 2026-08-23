#!/bin/bash
# S15 — THE SINGLE COMMAND. Everything submitted at once; only the smoke gate is a
# dependency, so a correctness failure cancels the wave instead of burning a night.
# CPU only: NO --gres. Main/controls/transductive are fully parallel with each other.
set -euo pipefail
S=/users/3171356m/A-GCL/audit/s15
cd "$S"
SMOKE=$(sbatch --parsable --cpus-per-task=8 --mem=16G --time=02:00:00 scripts/sb_smoke.sh)
MAIN=$(sbatch  --parsable --dependency=afterok:$SMOKE --array=0-287%29 \
        --cpus-per-task=4 --mem=8G --time=48:00:00 --requeue scripts/sb_main.sh)
CTRL=$(sbatch  --parsable --dependency=afterok:$SMOKE --array=0-23%29 \
        --cpus-per-task=4 --mem=8G --time=48:00:00 --requeue scripts/sb_ctrl.sh)
TRAN=$(sbatch  --parsable --dependency=afterok:$SMOKE --array=0-17%29 \
        --cpus-per-task=4 --mem=8G --time=48:00:00 --requeue scripts/sb_tran.sh)
printf "SMOKE=%s\nMAIN=%s\nCTRL=%s\nTRAN=%s\n" "$SMOKE" "$MAIN" "$CTRL" "$TRAN" \
  | tee "$S/JOBIDS.txt"
echo "submitted: smoke gates everything; main/ctrl/tran run fully parallel."
echo "morning: python $S/scripts/collect.py"
