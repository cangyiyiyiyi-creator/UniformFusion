#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SAVE_ROOT="${SAVE_ROOT:?Set SAVE_ROOT}"
SUMMARY_CSV="${SUMMARY_CSV:-${SAVE_ROOT}/validation_selection_results.csv}"
BACKBONE="${BACKBONE:?Set BACKBONE to resnet50 or convnextv2_tiny}"
METHOD="${METHOD:?Set METHOD to Plain_BCE or Final_NoAnchor}"
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
FINETUNE="${FINETUNE:-./student_weights_switch_no_head/convnextv2_tiny.mapped_to_backbone.safetensors}"

case "${BACKBONE}" in
  resnet50|convnextv2_tiny) ;;
  *) echo "BACKBONE must be resnet50 or convnextv2_tiny"; exit 2 ;;
esac
case "${METHOD}" in
  Plain_BCE|Final_NoAnchor) ;;
  *) echo "METHOD must be Plain_BCE or Final_NoAnchor"; exit 2 ;;
esac
[[ "$(realpath "${VAL_LIST}")" != "$(realpath "${TEST_LIST}")" ]] || {
  echo "VAL_LIST and TEST_LIST must differ"; exit 2;
}
if [[ "${BACKBONE}" == "convnextv2_tiny" ]]; then
  [[ -f "${FINETUNE}" ]] || { echo "Missing pretrained weights: ${FINETUNE}"; exit 1; }
fi

OUT_DIR="${SAVE_ROOT}/${BACKBONE}/seed_${SEED}/repeat_${REPEAT}/${METHOD}"
mkdir -p "${OUT_DIR}"
training_marker="${OUT_DIR}/training_complete.marker"
checkpoint="${OUT_DIR}/checkpoint_best.pth"

command=(
  "${PYTHON_BIN}" -u main_finetune.py
  --aug_mode conditional --patience "${PATIENCE}"
  --model "${BACKBONE}" --model_prefix ""
  --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" --lr 1e-4
  --weight_decay 0.05 --warmup_epochs 5 --drop_path 0.2
  --input_size 224 --dual_view true --view_mode paired --teacher_mode false
  --num_workers "${NUM_WORKERS}" --seed "${SEED}" --device cuda
  --deterministic true --reseed_before_training true
  --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}"
  --classes_file annotations/classes.txt --num_classes 15
  --fpn_out_channels 256 --gspf_lambda_consistency 0.0
  --gspf_lambda_ortho 0.0 --head_type c5 --fuse_mode add
  --base_loss bce --use_semantic_branch false
  --summary_csv "${SUMMARY_CSV}" --output_dir "${OUT_DIR}"
)
if [[ "${BACKBONE}" == "convnextv2_tiny" ]]; then
  command+=(--finetune "${FINETUNE}")
fi
if [[ "${METHOD}" == "Final_NoAnchor" ]]; then
  command+=(
    --return_intermediate true --use_p9_caprs true
    --plain_innovation_levels C4 C5
    --plain_innovation_projection_dim 64 --plain_innovation_topk 8
    --plain_innovation_temperature 0.2 --plain_innovation_dropout 0.1
    --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05
    --plain_innovation_base_floor 0.0
    --plain_innovation_use_counterfactual_experts true
    --plain_innovation_use_learned_router true
    --plain_innovation_warmup_epochs 15 --plain_innovation_ramp_epochs 10
    --plain_innovation_aux_weight 0.03 --plain_innovation_route_weight 0.05
    --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02
  )
fi
if [[ "${RESUME_PARTIAL,,}" == "true" && -f "${OUT_DIR}/checkpoint_last.pth" && ! -f "${training_marker}" ]]; then
  command+=(--resume "${OUT_DIR}")
fi

eval_command() {
  local split_name="$1"
  local list_path="$2"
  local view_mode="$3"
  local output_json="${OUT_DIR}/${split_name}_metrics.json"
  local output_csv="${OUT_DIR}/${split_name}_metrics.csv"
  printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
    tools/evaluate_project_checkpoint.py \
    --checkpoint "${checkpoint}" --list "${list_path}" \
    --classes-file annotations/classes.txt --view-mode "${view_mode}" \
    --output-json "${output_json}" --output-csv "${output_csv}" \
    --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --device cuda
  printf '\n'
}

if [[ "${DRY_RUN,,}" == "true" ]]; then
  printf '[DRY RUN TRAIN] '
  printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}"
  printf '\n[DRY RUN VAL] '
  eval_command val "${VAL_LIST}" paired
  printf '[DRY RUN TEST] '
  eval_command test "${TEST_LIST}" paired
  exit 0
fi

if [[ ! -f "${training_marker}" ]]; then
  echo "[$(date '+%F %T')] TRAIN START ${BACKBONE} ${METHOD} seed=${SEED}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}" 2>&1 | tee "${OUT_DIR}/train.log"
  [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint after training: ${checkpoint}"; exit 1; }
  touch "${training_marker}"
  echo "[$(date '+%F %T')] TRAIN DONE ${BACKBONE} ${METHOD} seed=${SEED}"
else
  echo "[SKIP TRAIN] ${BACKBONE} ${METHOD} seed=${SEED}"
fi

[[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}"; exit 1; }
"${PYTHON_BIN}" tools/verify_checkpoint_protocol.py \
  --checkpoint "${checkpoint}" --expected-val-list "${VAL_LIST}"

run_evaluation() {
  local split_name="$1"
  local list_path="$2"
  local output_json="${OUT_DIR}/${split_name}_metrics.json"
  local output_csv="${OUT_DIR}/${split_name}_metrics.csv"
  if [[ -s "${output_json}" && -s "${output_csv}" ]]; then
    echo "[SKIP ${split_name^^}] ${BACKBONE} ${METHOD} seed=${SEED}"
    return
  fi
  if [[ -e "${output_json}" || -e "${output_csv}" ]]; then
    echo "Incomplete ${split_name} output: ${OUT_DIR}"
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
    tools/evaluate_project_checkpoint.py \
    --checkpoint "${checkpoint}" --list "${list_path}" \
    --classes-file annotations/classes.txt --view-mode paired \
    --output-json "${output_json}" --output-csv "${output_csv}" \
    --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --device cuda
  echo "[$(date '+%F %T')] ${split_name^^} DONE ${BACKBONE} ${METHOD} seed=${SEED}"
}

run_evaluation val "${VAL_LIST}"
run_evaluation test "${TEST_LIST}"
