#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

SAVE_ROOT="${SAVE_ROOT:-./runs_final_noanchor_confirmation/run_20260830_final_noanchor_resnet50_n5_locked}"
SUMMARY_CSV="${SUMMARY_CSV:-${SAVE_ROOT}/validation_selection_results.csv}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATIENCE="${PATIENCE:-25}"
EPOCHS="${EPOCHS:-180}"
DRY_RUN="${DRY_RUN:-false}"

SEEDS=(207027553 1716854429)
REPEATS=(4 5)

for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  repeat="${REPEATS[$index]}"
  output_dir="${SAVE_ROOT}/resnet50/seed_${seed}/repeat_${repeat}/Final_NoAnchor"

  if [[ -s "${output_dir}/checkpoint_best.pth" && \
        -s "${output_dir}/val_metrics.csv" && \
        -s "${output_dir}/test_metrics.csv" ]]; then
    echo "[SKIP RESNET FINAL] complete seed=${seed} repeat=${repeat}"
    continue
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "Incomplete Final-NoAnchor output requires manual review: ${output_dir}"
    exit 1
  fi

  BACKBONE=resnet50 METHOD=Final_NoAnchor \
  SEED="${seed}" REPEAT="${repeat}" \
  SAVE_ROOT="${SAVE_ROOT}" SUMMARY_CSV="${SUMMARY_CSV}" \
  GPU_ID="${GPU_ID}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
  PATIENCE="${PATIENCE}" EPOCHS="${EPOCHS}" RESUME_PARTIAL=false \
  DRY_RUN="${DRY_RUN}" TRAIN_LIST=annotations/DvXray_train.txt \
  VAL_LIST=annotations/DvXray_val.txt TEST_LIST=annotations/DvXray_test.txt \
    bash run_final_noanchor_locked_pair_one.sh
done

echo "[$(date '+%F %T')] Missing ResNet50 Final-NoAnchor seeds are complete"
