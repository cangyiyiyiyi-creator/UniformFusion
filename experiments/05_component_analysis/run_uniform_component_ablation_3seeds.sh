#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
RUN_ID="${RUN_ID:-run_20260831_uniform_component_ablation_3seeds_locked}"
SAVE_ROOT="${SAVE_ROOT:-./runs_uniform_component_ablation/${RUN_ID}}"
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
CLASSES_FILE="${CLASSES_FILE:-annotations/classes.txt}"
UNIFORM_MAIN_CSV="${UNIFORM_MAIN_CSV:-./runs_uniform_fusion_n5_confirmation/run_20260831_uniform_missing2_locked_final/uniform_n5_detailed.csv}"
LOCKED_CODE_DIR="${LOCKED_CODE_DIR:-${ROOT_DIR}/论文最终归档_20260830/05_运行配置与代码}"
LOCKED_MAIN_FINETUNE_SCRIPT="${LOCKED_MAIN_FINETUNE_SCRIPT:-${LOCKED_CODE_DIR}/main_finetune.py}"
LOCKED_ENGINE_SCRIPT="${LOCKED_ENGINE_SCRIPT:-${LOCKED_CODE_DIR}/engine_finetune.py}"
VALIDATION_PHASE_MARKER="${SAVE_ROOT}/validation_phase_complete.marker"

EXPECTED_MAIN_SHA256="d1ec51b1e44152db52def599d2185d02318786de2db1177bff3158b26ce1782d"
EXPECTED_ENGINE_SHA256="dbe37f44aee890644a3016012b1b240a18ddd933e09fadb18d0ea38e7c1140ff"
EXPECTED_DATASETS_SHA256="48ae2214937417a4df730ede8efb52165cfd9b351401d3e472b2094bdaacdf39"
EXPECTED_MODULE_SHA256="0c7960199810874a3925b4a240bcd431afe565b05dd1cf6df399e91274a85b24"
EXPECTED_EVALUATOR_SHA256="bda55ba1232119144abee5d6b67751f2e5209a29f695c432ba9b6fbe64b8ab77"
EXPECTED_VERIFIER_SHA256="65e701c234e9f66f161f7ca43ecc5e0f1e91048dc75a8a63d89b3844608ae4d0"
EXPECTED_TRAIN_SHA256="f0a5c6f810a5725e3336b28df184542343f99e4d9afd5c866e860b4052254dcf"
EXPECTED_VAL_SHA256="a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67"
EXPECTED_TEST_SHA256="6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6"

SEEDS=(930163947 1786430941 553800223)
REPEATS=(1 2 3)
METHODS=(Uniform_NoCounterfactual Uniform_NoGuard Uniform_C5Only)

mkdir -p "${SAVE_ROOT}"
if [[ "${DRY_RUN,,}" != "true" ]] && pgrep -f '[m]ain_finetune.py' >/dev/null; then
  echo "Another main_finetune.py process is active."; exit 1
fi

check_hash() {
  local path="$1"
  local expected="$2"
  [[ -f "${path}" ]] || { echo "Missing locked input: ${path}"; exit 2; }
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "SHA256 mismatch for ${path}: expected=${expected} actual=${actual}"; exit 2;
  }
}

check_hash "${LOCKED_MAIN_FINETUNE_SCRIPT}" "${EXPECTED_MAIN_SHA256}"
check_hash "${LOCKED_ENGINE_SCRIPT}" "${EXPECTED_ENGINE_SHA256}"
check_hash datasets.py "${EXPECTED_DATASETS_SHA256}"
check_hash models/modules/plain_bce_innovations.py "${EXPECTED_MODULE_SHA256}"
check_hash tools/evaluate_project_checkpoint.py "${EXPECTED_EVALUATOR_SHA256}"
check_hash tools/verify_checkpoint_protocol.py "${EXPECTED_VERIFIER_SHA256}"
check_hash "${TRAIN_LIST}" "${EXPECTED_TRAIN_SHA256}"
check_hash "${VAL_LIST}" "${EXPECTED_VAL_SHA256}"
check_hash "${TEST_LIST}" "${EXPECTED_TEST_SHA256}"
[[ -s "${UNIFORM_MAIN_CSV}" ]] || { echo "Missing Uniform Fusion n=5 evidence"; exit 2; }

bash -n run_uniform_component_ablation_one.sh \
  run_uniform_component_ablation_3seeds.sh
"${PYTHON_BIN}" -m py_compile \
  tools/smoke_test_uniform_component_ablation.py \
  tools/smoke_test_uniform_component_summary.py \
  tools/summarize_uniform_component_ablation.py \
  tools/evaluate_project_checkpoint.py tools/verify_checkpoint_protocol.py \
  tools/paired_method_statistics.py
"${PYTHON_BIN}" tools/smoke_test_uniform_component_ablation.py \
  > "${SAVE_ROOT}/preflight.log" 2>&1
grep -Fq UNIFORM_COMPONENT_CONFIGS_SMOKE_OK "${SAVE_ROOT}/preflight.log"
"${PYTHON_BIN}" tools/smoke_test_uniform_component_summary.py \
  > "${SAVE_ROOT}/result_pipeline_preflight.log" 2>&1
grep -Fq UNIFORM_COMPONENT_SUMMARY_SMOKE_OK \
  "${SAVE_ROOT}/result_pipeline_preflight.log"

sha256sum \
  "${LOCKED_MAIN_FINETUNE_SCRIPT}" "${LOCKED_ENGINE_SCRIPT}" datasets.py \
  models/modules/plain_bce_innovations.py \
  tools/evaluate_project_checkpoint.py tools/verify_checkpoint_protocol.py \
  tools/smoke_test_uniform_component_ablation.py \
  tools/smoke_test_uniform_component_summary.py \
  tools/summarize_uniform_component_ablation.py tools/paired_method_statistics.py \
  run_uniform_component_ablation_one.sh run_uniform_component_ablation_3seeds.sh \
  "${TRAIN_LIST}" "${VAL_LIST}" "${TEST_LIST}" "${UNIFORM_MAIN_CSV}" \
  > "${SAVE_ROOT}/snapshot_sha256.txt"

{
  echo "protocol=train on train; lock all checkpoints on val; evaluate test only after validation phase completes"
  echo "purpose=strict single-factor component ablation of Uniform Fusion"
  echo "backbone=resnet50"
  echo "main_method=Uniform_Fusion reused from locked n=5 evidence"
  echo "ablation_methods=${METHODS[*]}"
  echo "common_constraint=plain_innovation_use_learned_router=false and route_weight=0"
  echo "seeds=${SEEDS[*]}"
  echo "repeats=${REPEATS[*]}"
  echo "new_training_groups=9"
  echo "batch_size=${BATCH_SIZE}"
  echo "patience=${PATIENCE}"
  echo "epochs=${EPOCHS}"
  echo "locked_main_sha256=${EXPECTED_MAIN_SHA256}"
  echo "test_list_sha256=${EXPECTED_TEST_SHA256}"
} > "${SAVE_ROOT}/protocol.txt"

run_phase() {
  local phase="$1"
  for index in "${!SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      METHOD="${method}" PHASE="${phase}" SEED="${SEEDS[$index]}" \
      REPEAT="${REPEATS[$index]}" SAVE_ROOT="${SAVE_ROOT}" \
      SUMMARY_CSV="${SUMMARY_CSV}" GPU_ID="${GPU_ID}" \
      BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
      PATIENCE="${PATIENCE}" EPOCHS="${EPOCHS}" \
      RESUME_PARTIAL="${RESUME_PARTIAL}" DRY_RUN="${DRY_RUN}" \
      TRAIN_LIST="${TRAIN_LIST}" VAL_LIST="${VAL_LIST}" TEST_LIST="${TEST_LIST}" \
      CLASSES_FILE="${CLASSES_FILE}" \
      LOCKED_MAIN_FINETUNE_SCRIPT="${LOCKED_MAIN_FINETUNE_SCRIPT}" \
      VALIDATION_PHASE_MARKER="${VALIDATION_PHASE_MARKER}" \
        bash run_uniform_component_ablation_one.sh
    done
  done
}

{
  echo "[$(date '+%F %T')] Uniform component ablation fixed n=3 started"
  run_phase val
  if [[ "${DRY_RUN,,}" == "true" ]]; then
    run_phase test
    echo "DRY_RUN_OK Uniform component ablation groups=9"
    exit 0
  fi
  touch "${VALIDATION_PHASE_MARKER}"
  echo "[$(date '+%F %T')] All nine Val selections locked; Test phase started"
  run_phase test

  "${PYTHON_BIN}" tools/summarize_uniform_component_ablation.py \
    --uniform-main-csv "${UNIFORM_MAIN_CSV}" --ablation-root "${SAVE_ROOT}" \
    --classes-file "${CLASSES_FILE}" --output-root "${SAVE_ROOT}"
  for method in "${METHODS[@]}"; do
    "${PYTHON_BIN}" tools/paired_method_statistics.py \
      --detailed-csv "${SAVE_ROOT}/uniform_component_ablation_3seeds_detailed.csv" \
      --baseline "${method}" --method Uniform_Fusion \
      --output-csv "${SAVE_ROOT}/Uniform_Fusion_vs_${method}_statistics.csv" \
      --output-json "${SAVE_ROOT}/Uniform_Fusion_vs_${method}_statistics.json"
  done
  touch "${SAVE_ROOT}/suite_complete.marker"
  echo "[$(date '+%F %T')] Uniform component ablation fixed n=3 finished"
} 2>&1 | tee -a "${MASTER_LOG}"
