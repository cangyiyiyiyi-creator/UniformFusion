#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SAVE_ROOT="${SAVE_ROOT:?Set SAVE_ROOT}"
SEED="${SEED:?Set SEED}"
REPEAT="${REPEAT:?Set REPEAT}"
GPU_ID="${GPU_ID:-0}"
RESUME_PARTIAL="${RESUME_PARTIAL:-true}"
DRY_RUN="${DRY_RUN:-false}"
SOURCE_DIR="${SOURCE_DIR:-third_party/DvXray_official}"
TRAIN_LIST="${TRAIN_LIST:-annotations/DvXray_train.txt}"
VAL_LIST="${VAL_LIST:-annotations/DvXray_val.txt}"
TEST_LIST="${TEST_LIST:-annotations/DvXray_test.txt}"
CLASSES_FILE="${CLASSES_FILE:-annotations/classes.txt}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ahcr_official_mpl_${UID}}"
mkdir -p "${MPLCONFIGDIR}"

OUT_DIR="${SAVE_ROOT}/resnet50/seed_${SEED}/repeat_${REPEAT}/AHCR_Official_Strict"
LOG_FILE="${OUT_DIR}/train_eval.log"
COMPLETE_MARKER="${OUT_DIR}/suite_complete.marker"
mkdir -p "${OUT_DIR}"

if [[ -f "${COMPLETE_MARKER}" && -s "${OUT_DIR}/val_metrics.json" && -s "${OUT_DIR}/test_metrics.json" ]]; then
  echo "[SKIP COMPLETE] AHCR-Official seed=${SEED} repeat=${REPEAT}"
  exit 0
fi

command=(
  env "CUDA_VISIBLE_DEVICES=${GPU_ID}" "${PYTHON_BIN}" -u
  tools/run_ahcr_official_strict.py
  --source-dir "${SOURCE_DIR}"
  --train-list "${TRAIN_LIST}"
  --val-list "${VAL_LIST}"
  --test-list "${TEST_LIST}"
  --classes-file "${CLASSES_FILE}"
  --output-dir "${OUT_DIR}"
  --seed "${SEED}"
  --repeat "${REPEAT}"
  --device cuda
)
if [[ "${RESUME_PARTIAL,,}" == "true" ]]; then
  command+=(--resume)
fi

if [[ "${DRY_RUN,,}" == "true" ]]; then
  printf '[DRY RUN AHCR-OFFICIAL] '
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

echo "[$(date '+%F %T')] START AHCR-Official seed=${SEED} repeat=${REPEAT}" | tee -a "${LOG_FILE}"
"${command[@]}" 2>&1 | tee -a "${LOG_FILE}"

[[ -f "${COMPLETE_MARKER}" ]] || { echo "Missing completion marker: ${OUT_DIR}"; exit 1; }
[[ -s "${OUT_DIR}/checkpoint_final_epoch30.pth.tar" ]] || { echo "Missing final checkpoint"; exit 1; }
[[ -s "${OUT_DIR}/val_metrics.json" ]] || { echo "Missing validation metrics"; exit 1; }
[[ -s "${OUT_DIR}/test_metrics.json" ]] || { echo "Missing test metrics"; exit 1; }
echo "[$(date '+%F %T')] FINISH AHCR-Official seed=${SEED} repeat=${REPEAT}" | tee -a "${LOG_FILE}"

