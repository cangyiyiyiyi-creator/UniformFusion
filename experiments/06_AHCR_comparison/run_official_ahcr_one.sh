#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SAVE_ROOT="${SAVE_ROOT:?Set SAVE_ROOT}"
SUMMARY_CSV="${SUMMARY_CSV:-${SAVE_ROOT}/training_results.csv}"
PHASE="${PHASE:?Set PHASE to val or test}"
SEED="${SEED:?Set SEED}"
REPEAT="${REPEAT:?Set REPEAT}"
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
SOURCE_DIR="${SOURCE_DIR:-third_party/DvXray_official}"
SOURCE_COMMIT="${SOURCE_COMMIT:-a6bfc1b1299d28e8226c106a94967287a8e30927}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-IMAGENET1K_V2}"
VALIDATION_PHASE_MARKER="${VALIDATION_PHASE_MARKER:-${SAVE_ROOT}/validation_phase_complete.marker}"

case "${PHASE}" in val|test) ;; *) echo "PHASE must be val or test"; exit 2 ;; esac
OUT_DIR="${SAVE_ROOT}/resnet50/seed_${SEED}/repeat_${REPEAT}/AHCR_Uniform"
mkdir -p "${OUT_DIR}"
checkpoint="${OUT_DIR}/checkpoint_best.pth"
training_marker="${OUT_DIR}/training_complete.marker"

command=(
  "${PYTHON_BIN}" -u main_finetune.py
  --aug_mode conditional --patience "${PATIENCE}"
  --model official_ahcr_resnet50 --model_prefix ""
  --official_ahcr_source_dir "${SOURCE_DIR}"
  --official_ahcr_source_commit "${SOURCE_COMMIT}"
  --official_ahcr_pretrained_weights "${PRETRAINED_WEIGHTS}"
  --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" --lr 1e-4
  --weight_decay 0.05 --warmup_epochs 5 --drop_path 0.2
  --input_size 224 --dual_view true --view_mode paired --teacher_mode false
  --num_workers "${NUM_WORKERS}" --seed "${SEED}" --device cuda
  --deterministic true --reseed_before_training true
  --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}"
  --classes_file annotations/classes.txt --num_classes 15
  --gspf_lambda_consistency 0.0 --gspf_lambda_ortho 0.0
  --cv_lambda_sem 0.0 --cv_lambda_geo 0.0
  --base_loss bce --use_semantic_branch false --return_intermediate false
  --summary_csv "${SUMMARY_CSV}" --output_dir "${OUT_DIR}"
)
if [[ "${RESUME_PARTIAL,,}" == "true" && -f "${OUT_DIR}/checkpoint_last.pth" && ! -f "${training_marker}" ]]; then
  command+=(--resume "${OUT_DIR}")
fi

run_eval() {
  local split="$1"
  local list_path="$2"
  local output_json="${OUT_DIR}/${split}_metrics.json"
  local output_csv="${OUT_DIR}/${split}_metrics.csv"
  if [[ "${DRY_RUN,,}" == "true" ]]; then
    printf '[DRY RUN %s] ' "${split^^}"
    printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
      tools/evaluate_project_checkpoint.py --checkpoint "${checkpoint}" \
      --list "${list_path}" --classes-file annotations/classes.txt \
      --view-mode paired --output-json "${output_json}" \
      --output-csv "${output_csv}" --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" --device cuda
    printf '\n'
    return
  fi
  if [[ -s "${output_json}" && -s "${output_csv}" ]]; then
    echo "[SKIP ${split^^}] Official_AHCR seed=${SEED}"
    return
  fi
  if [[ -e "${output_json}" || -e "${output_csv}" ]]; then
    echo "Incomplete ${split} output: ${OUT_DIR}"; exit 1
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
    tools/evaluate_project_checkpoint.py --checkpoint "${checkpoint}" \
    --list "${list_path}" --classes-file annotations/classes.txt \
    --view-mode paired --output-json "${output_json}" \
    --output-csv "${output_csv}" --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" --device cuda
  echo "[$(date '+%F %T')] ${split^^} DONE Official_AHCR seed=${SEED}"
}

if [[ "${DRY_RUN,,}" == "true" ]]; then
  if [[ "${PHASE}" == "val" ]]; then
    printf '[DRY RUN TRAIN] '
    printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}"
    printf '\n'
    run_eval val "${VAL_LIST}"
  else
    run_eval test "${TEST_LIST}"
  fi
  exit 0
fi

if [[ "${PHASE}" == "val" && ! -f "${training_marker}" ]]; then
  echo "[$(date '+%F %T')] TRAIN START Official_AHCR seed=${SEED}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}" 2>&1 | tee "${OUT_DIR}/train.log"
  [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint after training"; exit 1; }
  touch "${training_marker}"
  echo "[$(date '+%F %T')] TRAIN DONE Official_AHCR seed=${SEED}"
fi
[[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}"; exit 1; }
"${PYTHON_BIN}" tools/verify_checkpoint_protocol.py \
  --checkpoint "${checkpoint}" --expected-val-list "${VAL_LIST}"

if [[ "${PHASE}" == "val" ]]; then
  run_eval val "${VAL_LIST}"
else
  [[ -f "${VALIDATION_PHASE_MARKER}" ]] || {
    echo "Validation phase is not globally locked"; exit 2;
  }
  run_eval test "${TEST_LIST}"
fi
