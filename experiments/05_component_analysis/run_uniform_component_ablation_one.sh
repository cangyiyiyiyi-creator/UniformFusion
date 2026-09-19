#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SAVE_ROOT="${SAVE_ROOT:?Set SAVE_ROOT}"
SUMMARY_CSV="${SUMMARY_CSV:-${SAVE_ROOT}/training_results.csv}"
METHOD="${METHOD:?Set Uniform component ablation METHOD}"
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
CLASSES_FILE="${CLASSES_FILE:-annotations/classes.txt}"
LOCKED_MAIN_FINETUNE_SCRIPT="${LOCKED_MAIN_FINETUNE_SCRIPT:-${ROOT_DIR}/paper_archive_20260830/05_run_config_and_code/main_finetune.py}"
VALIDATION_PHASE_MARKER="${VALIDATION_PHASE_MARKER:-${SAVE_ROOT}/validation_phase_complete.marker}"

case "${PHASE}" in val|test) ;; *) echo "PHASE must be val or test"; exit 2 ;; esac
case "${METHOD}" in
  Uniform_NoCounterfactual|Uniform_NoGuard|Uniform_C5Only) ;;
  *) echo "Unknown Uniform component ablation: ${METHOD}"; exit 2 ;;
esac
[[ -f "${LOCKED_MAIN_FINETUNE_SCRIPT}" ]] || {
  echo "Missing locked main_finetune.py: ${LOCKED_MAIN_FINETUNE_SCRIPT}"; exit 2;
}
[[ "$(realpath "${VAL_LIST}")" != "$(realpath "${TEST_LIST}")" ]] || {
  echo "VAL_LIST and TEST_LIST must differ"; exit 2;
}

levels=(C4 C5)
use_counterfactual=true
guard_weight=0.10
single_weight=0.02
case "${METHOD}" in
  Uniform_NoCounterfactual)
    use_counterfactual=false
    single_weight=0.0
    ;;
  Uniform_NoGuard)
    guard_weight=0.0
    ;;
  Uniform_C5Only)
    levels=(C5)
    ;;
esac

OUT_DIR="${SAVE_ROOT}/resnet50/seed_${SEED}/repeat_${REPEAT}/${METHOD}"
mkdir -p "${OUT_DIR}"
training_marker="${OUT_DIR}/training_complete.marker"
checkpoint="${OUT_DIR}/checkpoint_best.pth"

command=(
  "${PYTHON_BIN}" -u "${LOCKED_MAIN_FINETUNE_SCRIPT}"
  --aug_mode conditional --patience "${PATIENCE}"
  --model resnet50 --model_prefix "" --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}" --lr 1e-4 --weight_decay 0.05
  --warmup_epochs 5 --drop_path 0.2 --input_size 224
  --dual_view true --view_mode paired --teacher_mode false
  --num_workers "${NUM_WORKERS}" --seed "${SEED}" --device cuda
  --deterministic true --reseed_before_training true
  --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}"
  --classes_file "${CLASSES_FILE}" --num_classes 15
  --fpn_out_channels 256 --gspf_lambda_consistency 0.0
  --gspf_lambda_ortho 0.0 --head_type c5 --fuse_mode add
  --base_loss bce --use_semantic_branch false --return_intermediate true
  --use_p9_caprs true --plain_innovation_levels "${levels[@]}"
  --plain_innovation_projection_dim 64 --plain_innovation_topk 8
  --plain_innovation_temperature 0.2 --plain_innovation_dropout 0.1
  --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05
  --plain_innovation_base_floor 0.0
  --plain_innovation_use_counterfactual_experts "${use_counterfactual}"
  --plain_innovation_use_learned_router false
  --plain_innovation_warmup_epochs 15 --plain_innovation_ramp_epochs 10
  --plain_innovation_aux_weight 0.03 --plain_innovation_route_weight 0.0
  --plain_innovation_guard_weight "${guard_weight}"
  --plain_innovation_single_weight "${single_weight}"
  --summary_csv "${SUMMARY_CSV}" --output_dir "${OUT_DIR}"
)
if [[ "${RESUME_PARTIAL,,}" == "true" && -f "${OUT_DIR}/checkpoint_last.pth" && ! -f "${training_marker}" ]]; then
  command+=(--resume "${OUT_DIR}")
fi

print_evaluation() {
  local split="$1"
  local list="$2"
  printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
    tools/evaluate_project_checkpoint.py --checkpoint "${checkpoint}" \
    --list "${list}" --classes-file "${CLASSES_FILE}" --view-mode paired \
    --output-json "${OUT_DIR}/${split}_metrics.json" \
    --output-csv "${OUT_DIR}/${split}_metrics.csv" \
    --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --device cuda
  printf '\n'
}

if [[ "${DRY_RUN,,}" == "true" ]]; then
  if [[ "${PHASE}" == "val" ]]; then
    printf '[DRY RUN TRAIN] '
    printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}"
    printf '\n'
  fi
  printf '[DRY RUN %s] ' "${PHASE^^}"
  if [[ "${PHASE}" == "val" ]]; then
    print_evaluation val "${VAL_LIST}"
  else
    print_evaluation test "${TEST_LIST}"
  fi
  exit 0
fi

if [[ "${PHASE}" == "val" && ! -f "${training_marker}" ]]; then
  echo "[$(date '+%F %T')] TRAIN START ${METHOD} seed=${SEED}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}" 2>&1 | tee "${OUT_DIR}/train.log"
  [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint after training: ${checkpoint}"; exit 1; }
  touch "${training_marker}"
  echo "[$(date '+%F %T')] TRAIN DONE ${METHOD} seed=${SEED}"
elif [[ "${PHASE}" == "val" ]]; then
  echo "[SKIP TRAIN] ${METHOD} seed=${SEED}"
fi

[[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}"; exit 1; }
"${PYTHON_BIN}" tools/verify_checkpoint_protocol.py \
  --checkpoint "${checkpoint}" --expected-val-list "${VAL_LIST}"
if [[ "${PHASE}" == "test" && ! -f "${VALIDATION_PHASE_MARKER}" ]]; then
  echo "Validation phase is not globally locked"; exit 2
fi

list="${VAL_LIST}"
[[ "${PHASE}" == "test" ]] && list="${TEST_LIST}"
output_json="${OUT_DIR}/${PHASE}_metrics.json"
output_csv="${OUT_DIR}/${PHASE}_metrics.csv"
if [[ -s "${output_json}" && -s "${output_csv}" ]]; then
  echo "[SKIP ${PHASE^^}] ${METHOD} seed=${SEED}"
  exit 0
fi
if [[ -e "${output_json}" || -e "${output_csv}" ]]; then
  echo "Incomplete ${PHASE} output for ${METHOD} seed=${SEED}"; exit 1
fi
echo "[$(date '+%F %T')] ${PHASE^^} START ${METHOD} seed=${SEED}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
  tools/evaluate_project_checkpoint.py --checkpoint "${checkpoint}" \
  --list "${list}" --classes-file "${CLASSES_FILE}" --view-mode paired \
  --output-json "${output_json}" --output-csv "${output_csv}" \
  --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --device cuda
echo "[$(date '+%F %T')] ${PHASE^^} DONE ${METHOD} seed=${SEED}"
