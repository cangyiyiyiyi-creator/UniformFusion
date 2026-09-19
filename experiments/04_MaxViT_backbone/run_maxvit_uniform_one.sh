#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
METHOD="${METHOD:?Set METHOD to Plain_BCE or Uniform_Fusion}"
PHASE="${PHASE:?Set PHASE to val or test}"
SEED="${SEED:?Set SEED}"
REPEAT="${REPEAT:?Set REPEAT}"
SAVE_ROOT="${SAVE_ROOT:?Set SAVE_ROOT}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-180}"
PATIENCE="${PATIENCE:-25}"
RESUME_PARTIAL="${RESUME_PARTIAL:-true}"
TRAIN_LIST="${TRAIN_LIST:-annotations/DvXray_train.txt}"
VAL_LIST="${VAL_LIST:-annotations/DvXray_val.txt}"
TEST_LIST="${TEST_LIST:-annotations/DvXray_test.txt}"
CLASSES_FILE="${CLASSES_FILE:-annotations/classes.txt}"
VALIDATION_MARKER="${SAVE_ROOT}/validation_phase_complete.marker"

case "${METHOD}" in Plain_BCE|Uniform_Fusion) ;; *) echo "Unknown METHOD=${METHOD}"; exit 2 ;; esac
case "${PHASE}" in val|test) ;; *) echo "Unknown PHASE=${PHASE}"; exit 2 ;; esac
[[ "$(realpath "${VAL_LIST}")" != "$(realpath "${TEST_LIST}")" ]] || {
  echo "VAL_LIST and TEST_LIST must differ"; exit 2;
}

OUT_DIR="${SAVE_ROOT}/maxvit_tiny/seed_${SEED}/repeat_${REPEAT}/${METHOD}"
mkdir -p "${OUT_DIR}"
checkpoint="${OUT_DIR}/checkpoint_best.pth"
training_marker="${OUT_DIR}/training_complete.marker"

command=(
  "${PYTHON_BIN}" -u main_finetune.py
  --aug_mode conditional --patience "${PATIENCE}"
  --model maxvit_tiny --model_prefix "" --batch_size "${BATCH_SIZE}"
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
  --summary_csv "${SAVE_ROOT}/training_results.csv" --output_dir "${OUT_DIR}"
)
if [[ "${METHOD}" == "Uniform_Fusion" ]]; then
  command+=(
    --use_p9_caprs true --plain_innovation_levels C4 C5
    --plain_innovation_projection_dim 64 --plain_innovation_topk 8
    --plain_innovation_temperature 0.2 --plain_innovation_dropout 0.1
    --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05
    --plain_innovation_base_floor 0.0
    --plain_innovation_use_counterfactual_experts true
    --plain_innovation_use_learned_router false
    --plain_innovation_warmup_epochs 15 --plain_innovation_ramp_epochs 10
    --plain_innovation_aux_weight 0.03 --plain_innovation_route_weight 0.0
    --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02
  )
else
  command+=(--use_p9_caprs false)
fi
if [[ "${RESUME_PARTIAL,,}" == "true" && -f "${OUT_DIR}/checkpoint_last.pth" && ! -f "${training_marker}" ]]; then
  command+=(--resume "${OUT_DIR}")
fi

if [[ "${PHASE}" == "val" && ! -f "${training_marker}" ]]; then
  echo "[$(date '+%F %T')] TRAIN START maxvit_tiny ${METHOD} seed=${SEED}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}" 2>&1 | tee "${OUT_DIR}/train.log"
  [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint after training"; exit 1; }
  touch "${training_marker}"
elif [[ "${PHASE}" == "val" ]]; then
  echo "[SKIP TRAIN] maxvit_tiny ${METHOD} seed=${SEED}"
fi

[[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}"; exit 1; }
"${PYTHON_BIN}" tools/verify_checkpoint_protocol.py \
  --checkpoint "${checkpoint}" --expected-val-list "${VAL_LIST}"
if [[ "${PHASE}" == "test" && ! -f "${VALIDATION_MARKER}" ]]; then
  echo "Test phase is not unlocked"; exit 2
fi

list="${VAL_LIST}"; [[ "${PHASE}" == "test" ]] && list="${TEST_LIST}"
output_json="${OUT_DIR}/${PHASE}_metrics.json"
output_csv="${OUT_DIR}/${PHASE}_metrics.csv"
if [[ -s "${output_json}" && -s "${output_csv}" ]]; then
  echo "[SKIP ${PHASE^^}] maxvit_tiny ${METHOD} seed=${SEED}"; exit 0
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u tools/evaluate_project_checkpoint.py \
  --checkpoint "${checkpoint}" --list "${list}" --classes-file "${CLASSES_FILE}" \
  --view-mode paired --output-json "${output_json}" --output-csv "${output_csv}" \
  --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --device cuda
echo "[$(date '+%F %T')] ${PHASE^^} DONE maxvit_tiny ${METHOD} seed=${SEED}"
