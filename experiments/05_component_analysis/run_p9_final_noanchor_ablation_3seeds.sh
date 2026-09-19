#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
RUN_ID="${RUN_ID:-run_20260829_p9_final_noanchor_ablation_3seeds}"
SAVE_ROOT="${SAVE_ROOT:-./runs_p9_final_noanchor_ablation/${RUN_ID}}"
PLAIN_ROOT="${PLAIN_ROOT:-./runs_p9_plain_valtest_2seeds/run_20260828_p9_plain_valtest_2seeds_rerun/resnet50}"
FINAL_ROOT="${FINAL_ROOT:-./runs_p9_component_valtest/run_20260828_p9_component_valtest_3seeds_final/resnet50}"
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
EXPECTED_TRAIN_SHA256="${EXPECTED_TRAIN_SHA256:-f0a5c6f810a5725e3336b28df184542343f99e4d9afd5c866e860b4052254dcf}"
EXPECTED_VAL_SHA256="${EXPECTED_VAL_SHA256:-a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67}"
EXPECTED_TEST_SHA256="${EXPECTED_TEST_SHA256:-6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6}"
VALIDATION_PHASE_MARKER="${SAVE_ROOT}/validation_phase_complete.marker"

SEEDS=(930163947 1786430941 553800223)
REPEATS=(1 2 3)
METHODS=(
  Plain_BCE
  Final_NoAnchor
  Final_NoAnchor_NoCounterfactualExperts
  Final_NoAnchor_NoRouter
  Final_NoAnchor_NoGuardLoss
  Final_NoAnchor_SingleScaleC5
)

mkdir -p "${SAVE_ROOT}"
if [[ "${DRY_RUN,,}" != "true" ]] && pgrep -f '[m]ain_finetune.py' >/dev/null; then
  echo "Another main_finetune.py process is active."
  exit 1
fi
[[ "$(realpath "${VAL_LIST}")" != "$(realpath "${TEST_LIST}")" ]] || {
  echo "VAL_LIST and TEST_LIST must differ"; exit 2;
}
[[ "$(sha256sum "${TRAIN_LIST}" | awk '{print $1}')" == "${EXPECTED_TRAIN_SHA256}" ]] || {
  echo "Train split checksum mismatch"; exit 2;
}
[[ "$(sha256sum "${VAL_LIST}" | awk '{print $1}')" == "${EXPECTED_VAL_SHA256}" ]] || {
  echo "Validation split checksum mismatch"; exit 2;
}
[[ "$(sha256sum "${TEST_LIST}" | awk '{print $1}')" == "${EXPECTED_TEST_SHA256}" ]] || {
  echo "Test split checksum mismatch"; exit 2;
}

bash -n run_p9_final_noanchor_ablation_one.sh run_p9_final_noanchor_ablation_3seeds.sh
"${PYTHON_BIN}" -m py_compile \
  main_finetune.py models/convnextv2_dual.py \
  models/modules/plain_bce_innovations.py \
  tools/evaluate_project_checkpoint.py tools/verify_checkpoint_protocol.py \
  tools/smoke_test_p9_final_noanchor_configs.py \
  tools/summarize_component_valtest.py
"${PYTHON_BIN}" tools/smoke_test_p9_final_noanchor_configs.py \
  > "${SAVE_ROOT}/preflight.log" 2>&1
grep -Fq P9_FINAL_NOANCHOR_CONFIGS_SMOKE_OK "${SAVE_ROOT}/preflight.log"

sha256sum \
  main_finetune.py models/convnextv2_dual.py \
  models/modules/plain_bce_innovations.py \
  tools/evaluate_project_checkpoint.py tools/verify_checkpoint_protocol.py \
  tools/smoke_test_p9_final_noanchor_configs.py \
  tools/summarize_component_valtest.py \
  run_p9_final_noanchor_ablation_one.sh \
  run_p9_final_noanchor_ablation_3seeds.sh \
  "${TRAIN_LIST}" "${VAL_LIST}" "${TEST_LIST}" \
  > "${SAVE_ROOT}/snapshot_sha256.txt"
{
  echo "protocol=train on train; select checkpoints on val; lock all variants; report test once"
  echo "final_method=Final_NoAnchor; fixed before this ablation"
  echo "purpose=final-centered component attribution; no further structure search"
  echo "plain_checkpoints=${PLAIN_ROOT}"
  echo "final_checkpoints=${FINAL_ROOT}"
  echo "train_list=${TRAIN_LIST}"
  echo "val_list=${VAL_LIST}"
  echo "test_list=${TEST_LIST}"
  echo "seeds=${SEEDS[*]}"
  echo "methods=${METHODS[*]}"
  echo "new_training_groups=12"
} > "${SAVE_ROOT}/protocol.txt"

run_phase () {
  local phase="$1"
  for index in "${!SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      METHOD="${method}" PHASE="${phase}" SEED="${SEEDS[$index]}" \
      REPEAT="${REPEATS[$index]}" RUN_ID="${RUN_ID}" SAVE_ROOT="${SAVE_ROOT}" \
      PLAIN_ROOT="${PLAIN_ROOT}" FINAL_ROOT="${FINAL_ROOT}" \
      SUMMARY_CSV="${SUMMARY_CSV}" GPU_ID="${GPU_ID}" \
      BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
      PATIENCE="${PATIENCE}" EPOCHS="${EPOCHS}" \
      RESUME_PARTIAL="${RESUME_PARTIAL}" DRY_RUN="${DRY_RUN}" \
      TRAIN_LIST="${TRAIN_LIST}" VAL_LIST="${VAL_LIST}" TEST_LIST="${TEST_LIST}" \
      VALIDATION_PHASE_MARKER="${VALIDATION_PHASE_MARKER}" \
        bash run_p9_final_noanchor_ablation_one.sh
    done
  done
}

{
  echo "[$(date '+%F %T')] Final NoAnchor ablation validation phase started"
  run_phase val
  if [[ "${DRY_RUN,,}" != "true" ]]; then
    "${PYTHON_BIN}" tools/summarize_component_valtest.py \
      --root "${SAVE_ROOT}" --phase val --expected-seeds 3 \
      --expected-methods "${METHODS[@]}"
    touch "${VALIDATION_PHASE_MARKER}"
  fi
  echo "[$(date '+%F %T')] Final NoAnchor ablation validation phase locked"

  echo "[$(date '+%F %T')] Final NoAnchor ablation test phase started"
  run_phase test
  if [[ "${DRY_RUN,,}" != "true" ]]; then
    "${PYTHON_BIN}" tools/summarize_component_valtest.py \
      --root "${SAVE_ROOT}" --phase final --expected-seeds 3 \
      --expected-methods "${METHODS[@]}"
    touch "${SAVE_ROOT}/structure_search_stopped.marker"
  fi
  echo "[$(date '+%F %T')] Final NoAnchor ablation test phase finished"
} 2>&1 | tee -a "${MASTER_LOG}"
