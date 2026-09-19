#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
METHOD="${METHOD:?Plain_BCE or Uniform_Fusion}"
PHASE="${PHASE:?val or test}"
SEED="${SEED:?Set SEED}"
REPEAT="${REPEAT:?Set REPEAT}"
SAVE_ROOT="${SAVE_ROOT:?Set SAVE_ROOT}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-15}"
INNOV_WARMUP_EPOCHS="${INNOV_WARMUP_EPOCHS:-15}"
INNOV_RAMP_EPOCHS="${INNOV_RAMP_EPOCHS:-10}"
INNOV_GAMMA_INIT="${INNOV_GAMMA_INIT:-0.005}"
INNOV_GAMMA_MAX="${INNOV_GAMMA_MAX:-0.05}"
INNOV_GUARD_WEIGHT="${INNOV_GUARD_WEIGHT:-0.10}"
TRAIN_LIST="${TRAIN_LIST:-annotations/ldxray/LDXray_train.txt}"
VAL_LIST="${VAL_LIST:-annotations/ldxray/LDXray_val.txt}"
TEST_LIST="${TEST_LIST:-annotations/ldxray/LDXray_test.txt}"
CLASSES_FILE="${CLASSES_FILE:-annotations/ldxray/ldxray_classes.txt}"
VALIDATION_MARKER="${SAVE_ROOT}/validation_phase_complete.marker"
OUT_DIR="${SAVE_ROOT}/resnet50/seed_${SEED}/repeat_${REPEAT}/${METHOD}"
mkdir -p "${OUT_DIR}"

# Optional one-time protocol gate for a user-approved first-seed Test preview.
# The gate is encountered by the fresh repeat-2 Plain invocation, after both
# repeat-1 validation runs have completed and before repeat-2 training starts.
early_test_request="${SAVE_ROOT}/early_test_after_seed1.request"
early_test_complete="${SAVE_ROOT}/early_test_after_seed1.complete"
if [[ "${PHASE}" == "val" && "${REPEAT}" == "2" && "${METHOD}" == "Plain_BCE" \
      && -f "${early_test_request}" && ! -f "${early_test_complete}" ]]; then
  echo "[$(date '+%F %T')] EARLY TEST GATE seed=930163947 started"
  touch "${VALIDATION_MARKER}"
  for seed1_method in Plain_BCE Uniform_Fusion_StepMatched; do
    METHOD="${seed1_method}" PHASE=test SEED=930163947 REPEAT=1 \
    SAVE_ROOT="${SAVE_ROOT}" GPU_ID="${GPU_ID}" BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" EPOCHS="${EPOCHS}" PATIENCE="${PATIENCE}" \
      bash run_ldxray_uniform_one.sh
  done
  touch "${early_test_complete}"
  echo "[$(date '+%F %T')] EARLY TEST GATE seed=930163947 completed; repeat 2 unlocked"
fi

common=(
  "${PYTHON_BIN}" -u main_finetune.py --aug_mode standard --patience "${PATIENCE}"
  --model resnet50 --model_prefix "" --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}"
  --lr 1e-4 --weight_decay 0.05 --warmup_epochs 5 --drop_path 0.2 --input_size 224
  --dual_view true --view_mode paired --teacher_mode false --num_workers "${NUM_WORKERS}"
  --seed "${SEED}" --device cuda --deterministic true --reseed_before_training true
  --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" --classes_file "${CLASSES_FILE}"
  --num_classes 12 --fpn_out_channels 256 --head_type c5 --fuse_mode add --base_loss bce
  --use_semantic_branch false --return_intermediate true --summary_csv "${SAVE_ROOT}/training_results.csv"
  --output_dir "${OUT_DIR}"
)
if [[ "${METHOD}" == Uniform_Fusion* ]]; then
  common+=(--use_p9_caprs true --plain_innovation_levels C4 C5
    --plain_innovation_projection_dim 64 --plain_innovation_topk 8
    --plain_innovation_temperature 0.2 --plain_innovation_dropout 0.1
    --plain_innovation_gamma_init "${INNOV_GAMMA_INIT}" --plain_innovation_gamma_max "${INNOV_GAMMA_MAX}"
    --plain_innovation_base_floor 0.0 --plain_innovation_use_counterfactual_experts true
    --plain_innovation_use_learned_router false --plain_innovation_warmup_epochs "${INNOV_WARMUP_EPOCHS}"
    --plain_innovation_ramp_epochs "${INNOV_RAMP_EPOCHS}" --plain_innovation_aux_weight 0.03
    --plain_innovation_route_weight 0.0 --plain_innovation_guard_weight "${INNOV_GUARD_WEIGHT}"
    --plain_innovation_single_weight 0.02)
else
  common+=(--use_p9_caprs false)
fi

checkpoint="${OUT_DIR}/checkpoint_best.pth"
if [[ "${PHASE}" == "val" && ! -f "${OUT_DIR}/training_complete.marker" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${common[@]}" 2>&1 | tee "${OUT_DIR}/train.log"
  touch "${OUT_DIR}/training_complete.marker"
fi
[[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}"; exit 1; }
[[ "${PHASE}" != "test" || -f "${VALIDATION_MARKER}" ]] || { echo "Test is not unlocked"; exit 2; }
list="${VAL_LIST}"; [[ "${PHASE}" == "test" ]] && list="${TEST_LIST}"
metrics_json="${OUT_DIR}/${PHASE}_metrics.json"
metrics_csv="${OUT_DIR}/${PHASE}_metrics.csv"
if [[ -s "${metrics_json}" && -s "${metrics_csv}" ]]; then
  echo "Existing ${PHASE} result found; skipping: ${metrics_json}"
  exit 0
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u tools/evaluate_project_checkpoint.py \
  --checkpoint "${checkpoint}" --list "${list}" --classes-file "${CLASSES_FILE}" \
  --view-mode paired --output-json "${metrics_json}" \
  --output-csv "${metrics_csv}" --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" --device cuda
