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
MAIN_FINETUNE_SCRIPT="${MAIN_FINETUNE_SCRIPT:-main_finetune.py}"
MAIN_FINETUNE_LOCK_FILE="${SAVE_ROOT}/main_finetune_script.lock"
if [[ -f "${MAIN_FINETUNE_LOCK_FILE}" ]]; then
  IFS= read -r MAIN_FINETUNE_SCRIPT < "${MAIN_FINETUNE_LOCK_FILE}"
fi
[[ -f "${MAIN_FINETUNE_SCRIPT}" ]] || {
  echo "Missing locked main_finetune script: ${MAIN_FINETUNE_SCRIPT}"; exit 2;
}
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PLAIN_ROOT="${PLAIN_ROOT:-./runs_p9_plain_valtest_2seeds/run_20260828_p9_plain_valtest_2seeds_rerun/resnet50}"
FINAL_ROOT="${FINAL_ROOT:-./runs_p9_component_valtest/run_20260828_p9_component_valtest_3seeds_final/resnet50}"
SUMMARY_CSV="${SUMMARY_CSV:-${SAVE_ROOT}/validation_selection_results.csv}"
METHOD="${METHOD:?Set final NoAnchor ablation METHOD}"
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
VALIDATION_PHASE_MARKER="${VALIDATION_PHASE_MARKER:-${SAVE_ROOT}/validation_phase_complete.marker}"

case "${PHASE}" in val|test) ;; *) echo "PHASE must be val or test"; exit 2 ;; esac
case "${METHOD}" in
  Plain_BCE|Final_NoAnchor|Final_NoAnchor_NoCounterfactualExperts|Final_NoAnchor_NoRouter|Final_NoAnchor_NoGuardLoss|Final_NoAnchor_SingleScaleC5) ;;
  *) echo "Unknown final NoAnchor ablation: ${METHOD}"; exit 2 ;;
esac
[[ "$(realpath "${VAL_LIST}")" != "$(realpath "${TEST_LIST}")" ]] || {
  echo "VAL_LIST and TEST_LIST must differ"; exit 2;
}

OUT_DIR="${SAVE_ROOT}/resnet50/seed_${SEED}/repeat_${REPEAT}/${METHOD}"
mkdir -p "${OUT_DIR}"
training_complete_marker="${OUT_DIR}/training_complete.marker"

checkpoint=""
case "${METHOD}" in
  Plain_BCE)
    checkpoint="${PLAIN_ROOT}/seed_${SEED}/repeat_${REPEAT}/Plain_BCE/checkpoint_best.pth"
    ;;
  Final_NoAnchor)
    checkpoint="${FINAL_ROOT}/seed_${SEED}/repeat_${REPEAT}/P9_NoAnchorFloor/checkpoint_best.pth"
    ;;
esac

if [[ "${PHASE}" == "val" && -z "${checkpoint}" ]]; then
  levels=(C4 C5)
  guard_weight=0.10
  route_weight=0.05
  single_weight=0.02
  use_counterfactual=true
  use_learned_router=true
  case "${METHOD}" in
    Final_NoAnchor_NoCounterfactualExperts)
      use_counterfactual=false
      single_weight=0.0
      ;;
    Final_NoAnchor_NoRouter)
      use_learned_router=false
      route_weight=0.0
      ;;
    Final_NoAnchor_NoGuardLoss)
      guard_weight=0.0
      ;;
    Final_NoAnchor_SingleScaleC5)
      levels=(C5)
      ;;
  esac
  command=(
    "${PYTHON_BIN}" -u "${MAIN_FINETUNE_SCRIPT}"
    --aug_mode conditional --patience "${PATIENCE}"
    --model resnet50 --model_prefix "" --batch_size "${BATCH_SIZE}"
    --epochs "${EPOCHS}" --lr 1e-4 --weight_decay 0.05
    --warmup_epochs 5 --drop_path 0.2 --input_size 224
    --dual_view true --view_mode paired --teacher_mode false
    --num_workers "${NUM_WORKERS}" --seed "${SEED}" --device cuda
    --deterministic true --reseed_before_training true
    --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}"
    --classes_file annotations/classes.txt --num_classes 15
    --fpn_out_channels 256 --gspf_lambda_consistency 0.0
    --gspf_lambda_ortho 0.0 --head_type c5 --fuse_mode add
    --base_loss bce --use_semantic_branch false --return_intermediate true
    --use_p9_caprs true --plain_innovation_levels "${levels[@]}"
    --plain_innovation_projection_dim 64 --plain_innovation_topk 8
    --plain_innovation_temperature 0.2 --plain_innovation_dropout 0.1
    --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05
    --plain_innovation_base_floor 0.0
    --plain_innovation_use_counterfactual_experts "${use_counterfactual}"
    --plain_innovation_use_learned_router "${use_learned_router}"
    --plain_innovation_warmup_epochs 15 --plain_innovation_ramp_epochs 10
    --plain_innovation_aux_weight 0.03
    --plain_innovation_route_weight "${route_weight}"
    --plain_innovation_guard_weight "${guard_weight}"
    --plain_innovation_single_weight "${single_weight}"
    --summary_csv "${SUMMARY_CSV}" --output_dir "${OUT_DIR}"
  )
  if [[ "${RESUME_PARTIAL,,}" == "true" && -f "${OUT_DIR}/checkpoint_last.pth" ]]; then
    command+=(--resume "${OUT_DIR}")
  fi
  if [[ "${DRY_RUN,,}" == "true" ]]; then
    printf '[DRY RUN TRAIN] '
    printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}"
    printf '\n'
  elif [[ ! -f "${training_complete_marker}" ]]; then
    echo "[$(date '+%F %T')] TRAIN START ${METHOD} seed=${SEED} selection=val"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${command[@]}" 2>&1 | tee "${OUT_DIR}/train.log"
    touch "${training_complete_marker}"
    echo "[$(date '+%F %T')] TRAIN DONE ${METHOD} seed=${SEED}"
  else
    echo "[SKIP TRAIN] ${METHOD} seed=${SEED}"
  fi
  checkpoint="${OUT_DIR}/checkpoint_best.pth"
elif [[ -z "${checkpoint}" ]]; then
  checkpoint="${OUT_DIR}/checkpoint_best.pth"
fi

if [[ "${DRY_RUN,,}" != "true" ]]; then
  [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}"; exit 1; }
  "${PYTHON_BIN}" tools/verify_checkpoint_protocol.py \
    --checkpoint "${checkpoint}" --expected-val-list "${VAL_LIST}"
fi

if [[ "${PHASE}" == "test" && "${DRY_RUN,,}" != "true" ]]; then
  [[ -f "${VALIDATION_PHASE_MARKER}" ]] || {
    echo "Validation phase is not globally locked"; exit 2;
  }
fi

list="${VAL_LIST}"
prefix=val_metrics
[[ "${PHASE}" == "test" ]] && { list="${TEST_LIST}"; prefix=test_metrics; }
output_json="${OUT_DIR}/${prefix}.json"
output_csv="${OUT_DIR}/${prefix}.csv"
if [[ "${DRY_RUN,,}" == "true" ]]; then
  printf '[DRY RUN %s EVAL] ' "${PHASE^^}"
  printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
    tools/evaluate_project_checkpoint.py --checkpoint "${checkpoint}" \
    --list "${list}" --classes-file annotations/classes.txt --view-mode paired \
    --output-json "${output_json}" --output-csv "${output_csv}" \
    --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --device cuda
  printf '\n'
  exit 0
fi
if [[ -s "${output_json}" && -s "${output_csv}" ]]; then
  echo "[SKIP ${PHASE^^} EVAL] ${METHOD} seed=${SEED}"
  exit 0
fi
if [[ -e "${output_json}" || -e "${output_csv}" ]]; then
  echo "Incomplete ${PHASE} output for ${METHOD} seed=${SEED}"
  exit 1
fi
echo "[$(date '+%F %T')] ${PHASE^^} EVAL START ${METHOD} seed=${SEED}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
  tools/evaluate_project_checkpoint.py \
  --checkpoint "${checkpoint}" --list "${list}" \
  --classes-file annotations/classes.txt --view-mode paired \
  --output-json "${output_json}" --output-csv "${output_csv}" \
  --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --device cuda
echo "[$(date '+%F %T')] ${PHASE^^} EVAL DONE ${METHOD} seed=${SEED}"
