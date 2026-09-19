#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
RUN_ID="${RUN_ID:-run_20260830_final_noanchor_convnextv2_3seeds_locked}"
SAVE_ROOT="${SAVE_ROOT:-./runs_final_noanchor_generalization/${RUN_ID}}"
SUMMARY_CSV="${SUMMARY_CSV:-${SAVE_ROOT}/validation_selection_results.csv}"
MASTER_LOG="${MASTER_LOG:-${SAVE_ROOT}/master.log}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATIENCE="${PATIENCE:-25}"
EPOCHS="${EPOCHS:-180}"
RESUME_PARTIAL="${RESUME_PARTIAL:-true}"
DRY_RUN="${DRY_RUN:-false}"
TRAIN_LIST="${TRAIN_LIST:-annotations/DvXray_train.txt}"
VAL_LIST="${VAL_LIST:-annotations/DvXray_val.txt}"
TEST_LIST="${TEST_LIST:-annotations/DvXray_test.txt}"
FINETUNE="${FINETUNE:-./student_weights_switch_no_head/convnextv2_tiny.mapped_to_backbone.safetensors}"

EXPECTED_TRAIN_SHA256="f0a5c6f810a5725e3336b28df184542343f99e4d9afd5c866e860b4052254dcf"
EXPECTED_VAL_SHA256="a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67"
EXPECTED_TEST_SHA256="6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6"

SEEDS=(930163947 1786430941 553800223)
REPEATS=(1 2 3)
METHODS=(Plain_BCE Final_NoAnchor)

mkdir -p "${SAVE_ROOT}"
if [[ "${DRY_RUN,,}" != "true" ]] && pgrep -f '[m]ain_finetune.py' >/dev/null; then
  echo "Another main_finetune.py process is active."
  exit 1
fi
[[ -f "${FINETUNE}" ]] || { echo "Missing pretrained weights: ${FINETUNE}"; exit 1; }

check_hash() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "Split checksum mismatch for ${path}: ${actual}"; exit 1;
  }
}
check_hash "${TRAIN_LIST}" "${EXPECTED_TRAIN_SHA256}"
check_hash "${VAL_LIST}" "${EXPECTED_VAL_SHA256}"
check_hash "${TEST_LIST}" "${EXPECTED_TEST_SHA256}"

sha256sum \
  main_finetune.py engine_finetune.py datasets.py models/convnextv2_dual.py \
  models/modules/plain_bce_innovations.py "${FINETUNE}" \
  run_final_noanchor_locked_pair_one.sh \
  run_final_noanchor_convnextv2_3seeds.sh \
  tools/evaluate_project_checkpoint.py tools/verify_checkpoint_protocol.py \
  tools/summarize_valtest_grid.py tools/paired_method_statistics.py \
  "${TRAIN_LIST}" "${VAL_LIST}" "${TEST_LIST}" \
  > "${SAVE_ROOT}/snapshot_sha256.txt"
{
  echo "protocol=train on train; select best checkpoint on val; evaluate test once"
  echo "status=Final_NoAnchor frozen before cross-backbone evaluation"
  echo "main_method=Final_NoAnchor"
  echo "baseline=Plain_BCE"
  echo "backbone=convnextv2_tiny"
  echo "pretrained=${FINETUNE}"
  echo "seeds=${SEEDS[*]}"
  echo "training_groups=6"
  echo "train_list=${TRAIN_LIST}"
  echo "val_list=${VAL_LIST}"
  echo "test_list=${TEST_LIST}"
} > "${SAVE_ROOT}/protocol.txt"

{
  echo "[$(date '+%F %T')] ConvNeXtV2-Tiny locked generalization started"
  for index in "${!SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      BACKBONE=convnextv2_tiny METHOD="${method}" \
      SEED="${SEEDS[$index]}" REPEAT="${REPEATS[$index]}" \
      SAVE_ROOT="${SAVE_ROOT}" SUMMARY_CSV="${SUMMARY_CSV}" \
      GPU_ID="${GPU_ID}" BATCH_SIZE="${BATCH_SIZE}" \
      NUM_WORKERS="${NUM_WORKERS}" PATIENCE="${PATIENCE}" EPOCHS="${EPOCHS}" \
      RESUME_PARTIAL="${RESUME_PARTIAL}" DRY_RUN="${DRY_RUN}" \
      TRAIN_LIST="${TRAIN_LIST}" VAL_LIST="${VAL_LIST}" TEST_LIST="${TEST_LIST}" \
      FINETUNE="${FINETUNE}" \
        bash run_final_noanchor_locked_pair_one.sh
    done
  done
  if [[ "${DRY_RUN,,}" != "true" ]]; then
    "${PYTHON_BIN}" tools/summarize_valtest_grid.py \
      --root "${SAVE_ROOT}" --prefix final_noanchor_convnextv2_3seeds \
      --baseline-method Plain_BCE --expected-seeds 3
    "${PYTHON_BIN}" tools/paired_method_statistics.py \
      --detailed-csv "${SAVE_ROOT}/final_noanchor_convnextv2_3seeds_detailed.csv" \
      --baseline Plain_BCE --method Final_NoAnchor \
      --output-csv "${SAVE_ROOT}/paired_statistics.csv" \
      --output-json "${SAVE_ROOT}/paired_statistics.json"
    touch "${SAVE_ROOT}/generalization_complete.marker"
  fi
  echo "[$(date '+%F %T')] ConvNeXtV2-Tiny locked generalization finished"
} 2>&1 | tee -a "${MASTER_LOG}"
