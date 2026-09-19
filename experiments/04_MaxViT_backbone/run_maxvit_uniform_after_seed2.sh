#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

SAVE_ROOT="${SAVE_ROOT:-./runs_maxvit_uniform/run_20260902_maxvit_uniform_3seeds_locked}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-180}"
PATIENCE="${PATIENCE:-25}"

seed2_dir="${SAVE_ROOT}/maxvit_tiny/seed_1786430941/repeat_2/Uniform_Fusion"
echo "[$(date '+%F %T')] Waiting for seed 1786430941 Uniform Fusion validation."
until [[ -f "${seed2_dir}/training_complete.marker" && \
         -s "${seed2_dir}/val_metrics.json" && \
         -s "${seed2_dir}/val_metrics.csv" ]]; do
  sleep 60
done

echo "[$(date '+%F %T')] Seed 2 complete; starting seed 553800223."
for method in Plain_BCE Uniform_Fusion; do
  METHOD="${method}" PHASE=val SEED=553800223 REPEAT=3 \
  SAVE_ROOT="${SAVE_ROOT}" GPU_ID="${GPU_ID}" \
  BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
  EPOCHS="${EPOCHS}" PATIENCE="${PATIENCE}" RESUME_PARTIAL=true \
    bash run_maxvit_uniform_one.sh
done

touch "${SAVE_ROOT}/validation_phase_complete.marker"
seeds=(930163947 1786430941 553800223)
methods=(Plain_BCE Uniform_Fusion)
for index in "${!seeds[@]}"; do
  for method in "${methods[@]}"; do
    METHOD="${method}" PHASE=test SEED="${seeds[$index]}" \
    REPEAT="$((index + 1))" SAVE_ROOT="${SAVE_ROOT}" GPU_ID="${GPU_ID}" \
    BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
    EPOCHS="${EPOCHS}" PATIENCE="${PATIENCE}" \
      bash run_maxvit_uniform_one.sh
  done
done

touch "${SAVE_ROOT}/suite_complete.marker"
echo "[$(date '+%F %T')] MaxViT three-seed validation and test sequence complete."
