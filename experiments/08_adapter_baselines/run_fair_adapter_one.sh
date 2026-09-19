#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
PY="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
: "${SAVE_ROOT:?}" "${METHOD:?}" "${SEED:?}" "${REPEAT:?}" "${PHASE:?}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"

case "$METHOD" in
  DAGNet_OfficialArchitecture)
    MODEL=dagnet_official_adapter
    INPUT_SIZE=256
    BATCH_SIZE="${DAGNET_BATCH_SIZE:-16}"
    ;;
  MLDecoder_DualView)
    MODEL=resnet50_ml_decoder_adapter
    INPUT_SIZE=224
    BATCH_SIZE="${MLDECODER_BATCH_SIZE:-32}"
    ;;
  *) echo "unknown METHOD=$METHOD"; exit 2 ;;
esac

OUT="$SAVE_ROOT/$MODEL/seed_${SEED}/repeat_${REPEAT}/$METHOD"
mkdir -p "$OUT"
CKPT="$OUT/checkpoint_best.pth"
COMMON=(
  --model "$MODEL" --model_prefix "" --input_size "$INPUT_SIZE"
  --batch_size "$BATCH_SIZE" --accum_iter "${ACCUM_ITER:-1}" --epochs 180 --lr 1e-4 --weight_decay 0.05
  --warmup_epochs 5 --drop_path 0.2 --patience 25
  --dual_view true --view_mode paired --teacher_mode false
  --train_list annotations/DvXray_train.txt --val_list annotations/DvXray_val.txt
  --classes_file annotations/classes.txt --num_classes 15
  --base_loss bce --fuse_mode add --head_type c5 --return_intermediate false
  --use_semantic_branch false --use_p9_caprs false
  --num_workers "$NUM_WORKERS" --seed "$SEED" --device cuda
  --deterministic true --reseed_before_training true
  --aug_mode conditional --summary_csv "$SAVE_ROOT/results.csv" --output_dir "$OUT"
)

if [[ "$PHASE" == val ]]; then
  if [[ ! -f "$OUT/training_complete.marker" ]]; then
    RESUME=()
    [[ -f "$OUT/checkpoint_last.pth" ]] && RESUME=(--resume "$OUT")
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u main_finetune.py \
      "${COMMON[@]}" "${RESUME[@]}" 2>&1 | tee "$OUT/train.log"
    [[ -f "$CKPT" ]]
    touch "$OUT/training_complete.marker"
  fi
  LIST=annotations/DvXray_val.txt
elif [[ "$PHASE" == test ]]; then
  [[ -f "$SAVE_ROOT/validation_phase_complete.marker" ]] || {
    echo "validation phase is not locked"; exit 2;
  }
  [[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT"; exit 2; }
  LIST=annotations/DvXray_test.txt
else
  echo "PHASE must be val or test"; exit 2
fi

"$PY" tools/verify_checkpoint_protocol.py --checkpoint "$CKPT" \
  --expected-val-list annotations/DvXray_val.txt
if [[ ! -s "$OUT/${PHASE}_metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" tools/evaluate_project_checkpoint.py \
    --checkpoint "$CKPT" --list "$LIST" --classes-file annotations/classes.txt \
    --view-mode paired --output-json "$OUT/${PHASE}_metrics.json" \
    --output-csv "$OUT/${PHASE}_metrics.csv" --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" --device cuda
fi
