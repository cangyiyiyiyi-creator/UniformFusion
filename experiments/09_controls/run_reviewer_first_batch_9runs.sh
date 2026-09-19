#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
RUN_ID="${RUN_ID:-run_20260905_reviewer_first_batch_3seeds}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT/runs_reviewer_validation/$RUN_ID}"
mkdir -p "$SAVE_ROOT"
SEEDS=(930163947 1786430941 553800223)
METHODS=(UF_NoSingleLoss Plain_BCE_SingleLoss UF_DenseSelector)
{
 echo "protocol=pre-registered reviewer first-batch ablations"
 echo "methods=${METHODS[*]}"; echo "seeds=${SEEDS[*]}"
 echo "selection=Val checkpoint; Test only after all nine validation runs"
 echo "dense_selector=topk exceeds C4/C5 token counts, therefore exact full-spatial softmax"
 echo "plain_single=final correction fixed exactly zero; only paired BCE plus single-view auxiliary BCE"
} > "$SAVE_ROOT/protocol.txt"
for phase in val test; do
  if [[ "$phase" == test ]]; then touch "$SAVE_ROOT/validation_phase_complete.marker"; fi
  repeat=0
  for seed in "${SEEDS[@]}"; do
    repeat=$((repeat+1))
    for method in "${METHODS[@]}"; do
      SAVE_ROOT="$SAVE_ROOT" METHOD="$method" SEED="$seed" REPEAT="$repeat" PHASE="$phase" \
       bash run_reviewer_first_batch_one.sh
    done
  done
done
"${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}" tools/summarize_reviewer_first_batch.py --root "$SAVE_ROOT"
touch "$SAVE_ROOT/suite_complete.marker"
echo "DONE: $SAVE_ROOT"
