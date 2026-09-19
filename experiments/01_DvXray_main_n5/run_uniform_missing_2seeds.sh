#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
RUN_ID="${RUN_ID:-run_20260831_uniform_missing2_locked}"
SAVE_ROOT="${SAVE_ROOT:-./runs_uniform_fusion_n5_confirmation/${RUN_ID}}"
MASTER_LOG="${MASTER_LOG:-${SAVE_ROOT}/master.log}"
SUMMARY_CSV="${SUMMARY_CSV:-${SAVE_ROOT}/training_results.csv}"
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
EXPECTED_TRAIN_SHA256="f0a5c6f810a5725e3336b28df184542343f99e4d9afd5c866e860b4052254dcf"
EXPECTED_VAL_SHA256="a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67"
EXPECTED_TEST_SHA256="6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6"
EXISTING_ABLATION_CSV="${EXISTING_ABLATION_CSV:-./runs_p9_final_noanchor_ablation/run_20260829_p9_final_noanchor_ablation_3seeds_final/component_ablation_detailed.csv}"
MAIN_DETAILED_CSV="${MAIN_DETAILED_CSV:-./runs_final_noanchor_evidence/run_20260830_final_noanchor_all_evidence_locked/resnet_detailed.csv}"
VALIDATION_PHASE_MARKER="${SAVE_ROOT}/validation_phase_complete.marker"
MAIN_FINETUNE_LOCK_FILE="${SAVE_ROOT}/main_finetune_script.lock"
LOCKED_MAIN_FINETUNE_SCRIPT="main_finetune.py"

SEEDS=(207027553 1716854429)
REPEATS=(4 5)
METHOD="Final_NoAnchor_NoRouter"

mkdir -p "${SAVE_ROOT}"
if [[ -f "${MAIN_FINETUNE_LOCK_FILE}" ]]; then
  IFS= read -r LOCKED_MAIN_FINETUNE_SCRIPT < "${MAIN_FINETUNE_LOCK_FILE}"
fi
[[ -f "${LOCKED_MAIN_FINETUNE_SCRIPT}" ]] || {
  echo "Missing locked main_finetune script: ${LOCKED_MAIN_FINETUNE_SCRIPT}"; exit 2;
}
if [[ "${DRY_RUN,,}" != "true" ]] && pgrep -f '[m]ain_finetune.py' >/dev/null; then
  echo "Another main_finetune.py process is active."
  exit 1
fi
[[ "$(sha256sum "${TRAIN_LIST}" | awk '{print $1}')" == "${EXPECTED_TRAIN_SHA256}" ]] || {
  echo "Train split checksum mismatch"; exit 2;
}
[[ "$(sha256sum "${VAL_LIST}" | awk '{print $1}')" == "${EXPECTED_VAL_SHA256}" ]] || {
  echo "Validation split checksum mismatch"; exit 2;
}
[[ "$(sha256sum "${TEST_LIST}" | awk '{print $1}')" == "${EXPECTED_TEST_SHA256}" ]] || {
  echo "Test split checksum mismatch"; exit 2;
}
[[ -s "${EXISTING_ABLATION_CSV}" ]] || { echo "Missing existing n=3 ablation CSV"; exit 2; }
[[ -s "${MAIN_DETAILED_CSV}" ]] || { echo "Missing locked ResNet50 n=5 CSV"; exit 2; }

bash -n run_p9_final_noanchor_ablation_one.sh run_uniform_missing_2seeds.sh
"${PYTHON_BIN}" -m py_compile \
  "${LOCKED_MAIN_FINETUNE_SCRIPT}" models/convnextv2_dual.py \
  models/modules/plain_bce_innovations.py \
  tools/evaluate_project_checkpoint.py tools/verify_checkpoint_protocol.py \
  tools/smoke_test_p9_final_noanchor_configs.py \
  tools/summarize_uniform_n5_confirmation.py tools/paired_method_statistics.py
"${PYTHON_BIN}" tools/smoke_test_p9_final_noanchor_configs.py \
  > "${SAVE_ROOT}/preflight.log" 2>&1
grep -Fq P9_FINAL_NOANCHOR_CONFIGS_SMOKE_OK "${SAVE_ROOT}/preflight.log"

sha256sum \
  "${LOCKED_MAIN_FINETUNE_SCRIPT}" models/convnextv2_dual.py \
  models/modules/plain_bce_innovations.py \
  tools/evaluate_project_checkpoint.py tools/verify_checkpoint_protocol.py \
  tools/smoke_test_p9_final_noanchor_configs.py \
  tools/summarize_uniform_n5_confirmation.py tools/paired_method_statistics.py \
  run_p9_final_noanchor_ablation_one.sh run_uniform_missing_2seeds.sh \
  "${TRAIN_LIST}" "${VAL_LIST}" "${TEST_LIST}" \
  > "${SAVE_ROOT}/snapshot_sha256.txt"
{
  echo "protocol=train on train; select checkpoint on val; report locked checkpoint on test once"
  echo "purpose=complete Uniform Fusion from existing fixed n=3 to fixed n=5"
  echo "method=${METHOD}"
  echo "existing_seeds=930163947 1786430941 553800223"
  echo "new_seeds=${SEEDS[*]}"
  echo "repeats=${REPEATS[*]}"
  echo "batch_size=${BATCH_SIZE}"
  echo "patience=${PATIENCE}"
  echo "epochs=${EPOCHS}"
  echo "existing_ablation_csv=${EXISTING_ABLATION_CSV}"
  echo "main_detailed_csv=${MAIN_DETAILED_CSV}"
  echo "locked_main_finetune_script=${LOCKED_MAIN_FINETUNE_SCRIPT}"
  echo "locked_main_finetune_sha256=$(sha256sum "${LOCKED_MAIN_FINETUNE_SCRIPT}" | awk '{print $1}')"
} > "${SAVE_ROOT}/protocol.txt"

run_phase() {
  local phase="$1"
  for index in "${!SEEDS[@]}"; do
    METHOD="${METHOD}" PHASE="${phase}" SEED="${SEEDS[$index]}" \
    REPEAT="${REPEATS[$index]}" RUN_ID="${RUN_ID}" SAVE_ROOT="${SAVE_ROOT}" \
    SUMMARY_CSV="${SUMMARY_CSV}" GPU_ID="${GPU_ID}" \
    BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
    PATIENCE="${PATIENCE}" EPOCHS="${EPOCHS}" \
    RESUME_PARTIAL="${RESUME_PARTIAL}" DRY_RUN="${DRY_RUN}" \
    TRAIN_LIST="${TRAIN_LIST}" VAL_LIST="${VAL_LIST}" TEST_LIST="${TEST_LIST}" \
    MAIN_FINETUNE_SCRIPT="${LOCKED_MAIN_FINETUNE_SCRIPT}" \
    VALIDATION_PHASE_MARKER="${VALIDATION_PHASE_MARKER}" \
      bash run_p9_final_noanchor_ablation_one.sh
  done
}

{
  echo "[$(date '+%F %T')] Uniform Fusion missing-two validation phase started"
  run_phase val
  if [[ "${DRY_RUN,,}" == "true" ]]; then
    run_phase test
    echo "DRY_RUN_OK Uniform Fusion missing-two"
    exit 0
  fi
  touch "${VALIDATION_PHASE_MARKER}"
  echo "[$(date '+%F %T')] Validation phase locked; test phase started"
  run_phase test

  "${PYTHON_BIN}" tools/summarize_uniform_n5_confirmation.py \
    --existing-ablation-csv "${EXISTING_ABLATION_CSV}" \
    --new-root "${SAVE_ROOT}" --main-detailed-csv "${MAIN_DETAILED_CSV}" \
    --output-root "${SAVE_ROOT}"
  "${PYTHON_BIN}" tools/paired_method_statistics.py \
    --detailed-csv "${SAVE_ROOT}/uniform_n5_detailed.csv" \
    --baseline Final_NoAnchor --method Uniform_Fusion \
    --output-csv "${SAVE_ROOT}/uniform_vs_final_paired_statistics.csv" \
    --output-json "${SAVE_ROOT}/uniform_vs_final_paired_statistics.json"
  "${PYTHON_BIN}" tools/paired_method_statistics.py \
    --detailed-csv "${SAVE_ROOT}/uniform_n5_detailed.csv" \
    --baseline Plain_BCE --method Uniform_Fusion \
    --output-csv "${SAVE_ROOT}/uniform_vs_plain_paired_statistics.csv" \
    --output-json "${SAVE_ROOT}/uniform_vs_plain_paired_statistics.json"
  touch "${SAVE_ROOT}/suite_complete.marker"
  echo "[$(date '+%F %T')] Uniform Fusion fixed n=5 confirmation finished"
} 2>&1 | tee -a "${MASTER_LOG}"
