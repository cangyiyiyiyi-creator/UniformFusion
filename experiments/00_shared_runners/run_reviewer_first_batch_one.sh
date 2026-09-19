#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
PY="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
: "${SAVE_ROOT:?}" "${METHOD:?}" "${SEED:?}" "${REPEAT:?}" "${PHASE:?val or test}"
GPU_ID="${GPU_ID:-0}"; BATCH_SIZE="${BATCH_SIZE:-32}"; NUM_WORKERS="${NUM_WORKERS:-8}"
OUT="$SAVE_ROOT/resnet50/seed_${SEED}/repeat_${REPEAT}/${METHOD}"; mkdir -p "$OUT"
SUMMARY="${SUMMARY_CSV:-$SAVE_ROOT/results.csv}"
COMMON=(--aug_mode conditional --patience 25 --model resnet50 --model_prefix "" --batch_size "$BATCH_SIZE"
 --epochs 180 --lr 1e-4 --weight_decay 0.05 --warmup_epochs 5 --drop_path 0.2 --input_size 224
 --dual_view true --view_mode paired --teacher_mode false --num_workers "$NUM_WORKERS" --seed "$SEED"
 --device cuda --deterministic true --reseed_before_training true --train_list annotations/DvXray_train.txt
 --val_list annotations/DvXray_val.txt --classes_file annotations/classes.txt --num_classes 15
 --fpn_out_channels 256 --gspf_lambda_consistency 0 --gspf_lambda_ortho 0 --head_type c5 --fuse_mode add
 --base_loss bce --use_semantic_branch false --return_intermediate true --use_p9_caprs true
 --plain_innovation_levels C4 C5 --plain_innovation_projection_dim 64 --plain_innovation_temperature 0.2
 --plain_innovation_dropout 0.1 --plain_innovation_base_floor 0 --plain_innovation_use_counterfactual_experts true
 --plain_innovation_use_learned_router false --plain_innovation_warmup_epochs 15 --plain_innovation_ramp_epochs 10
 --plain_innovation_route_weight 0 --plain_innovation_consistency_weight 0 --plain_innovation_regret_weight 0
 --plain_innovation_evidence_weight 0 --plain_innovation_match_weight 0 --plain_innovation_rank_weight 0
 --plain_innovation_budget_weight 0 --summary_csv "$SUMMARY" --output_dir "$OUT")
case "$METHOD" in
 UF_NoSingleLoss) EXTRA=(--plain_innovation_topk 8 --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05 --plain_innovation_gamma_trainable true --plain_innovation_aux_weight 0.03 --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0) ;;
 Plain_BCE_SingleLoss) EXTRA=(--plain_innovation_topk 8 --plain_innovation_gamma_init 0 --plain_innovation_gamma_max 0.05 --plain_innovation_gamma_trainable false --plain_innovation_aux_weight 0 --plain_innovation_guard_weight 0 --plain_innovation_single_weight 0.02) ;;
 UF_DenseSelector) EXTRA=(--plain_innovation_topk 100000 --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05 --plain_innovation_gamma_trainable true --plain_innovation_aux_weight 0.03 --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02) ;;
 UF_GAP) EXTRA=(--plain_innovation_topk 8 --plain_innovation_region_pooling gap --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05 --plain_innovation_gamma_trainable true --plain_innovation_aux_weight 0.03 --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02) ;;
 UF_NoAux) EXTRA=(--plain_innovation_topk 8 --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05 --plain_innovation_gamma_trainable true --plain_innovation_aux_weight 0 --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02) ;;
 UF_K4) EXTRA=(--plain_innovation_topk 4 --plain_innovation_temperature 0.2 --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05 --plain_innovation_gamma_trainable true --plain_innovation_aux_weight 0.03 --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02) ;;
 UF_K16) EXTRA=(--plain_innovation_topk 16 --plain_innovation_temperature 0.2 --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05 --plain_innovation_gamma_trainable true --plain_innovation_aux_weight 0.03 --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02) ;;
 UF_Temp010) EXTRA=(--plain_innovation_topk 8 --plain_innovation_temperature 0.1 --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05 --plain_innovation_gamma_trainable true --plain_innovation_aux_weight 0.03 --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02) ;;
 UF_Temp040) EXTRA=(--plain_innovation_topk 8 --plain_innovation_temperature 0.4 --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05 --plain_innovation_gamma_trainable true --plain_innovation_aux_weight 0.03 --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02) ;;
 UF_GammaMax002) EXTRA=(--plain_innovation_topk 8 --plain_innovation_temperature 0.2 --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.02 --plain_innovation_gamma_trainable true --plain_innovation_aux_weight 0.03 --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02) ;;
 UF_GammaMax010) EXTRA=(--plain_innovation_topk 8 --plain_innovation_temperature 0.2 --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.10 --plain_innovation_gamma_trainable true --plain_innovation_aux_weight 0.03 --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02) ;;
 UF_NoCorrection) EXTRA=(--plain_innovation_topk 8 --plain_innovation_temperature 0.2 --plain_innovation_gamma_init 0 --plain_innovation_gamma_max 0.05 --plain_innovation_gamma_trainable false --plain_innovation_aux_weight 0.03 --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02) ;;
 UF_NoRamp) EXTRA=(--plain_innovation_topk 8 --plain_innovation_temperature 0.2 --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05 --plain_innovation_gamma_trainable true --plain_innovation_warmup_epochs 0 --plain_innovation_ramp_epochs 0 --plain_innovation_aux_weight 0.03 --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02) ;;
 *) echo "Unknown METHOD=$METHOD"; exit 2 ;;
esac
CKPT="$OUT/checkpoint_best.pth"
if [[ "$PHASE" == val ]]; then
  if [[ ! -f "$OUT/training_complete.marker" ]]; then
    RESUME=()
    [[ -f "$OUT/checkpoint_last.pth" ]] && RESUME=(--resume "$OUT")
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u main_finetune.py "${COMMON[@]}" "${EXTRA[@]}" "${RESUME[@]}" 2>&1 | tee "$OUT/train.log"
    [[ -f "$CKPT" ]]; touch "$OUT/training_complete.marker"
  fi
  LIST=annotations/DvXray_val.txt
elif [[ "$PHASE" == test ]]; then
  [[ -f "$SAVE_ROOT/validation_phase_complete.marker" ]] || { echo "Validation phase is not locked"; exit 2; }
  [[ -f "$CKPT" ]] || { echo "Missing checkpoint: $CKPT"; exit 2; }
  LIST=annotations/DvXray_test.txt
else echo "PHASE must be val or test"; exit 2; fi
"$PY" tools/verify_checkpoint_protocol.py --checkpoint "$CKPT" --expected-val-list annotations/DvXray_val.txt
if [[ ! -s "$OUT/${PHASE}_metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" tools/evaluate_project_checkpoint.py --checkpoint "$CKPT" --list "$LIST" \
   --classes-file annotations/classes.txt --view-mode paired --output-json "$OUT/${PHASE}_metrics.json" \
   --output-csv "$OUT/${PHASE}_metrics.csv" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --device cuda
fi
