#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-./runs_fair_fusion_baselines/run_20260903_fair_fusion_3seeds}"
SEEDS=(930163947 1786430941 553800223)
METHODS=(Mean_Fusion Max_Fusion Concat_Fusion CrossAttention_Fusion)

mkdir -p "${SAVE_ROOT}"
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"; repeat="$((i+1))"
  for method in "${METHODS[@]}"; do
    case "${method}" in
      Mean_Fusion) mode=mean ;;
      Max_Fusion) mode=max ;;
      Concat_Fusion) mode=concat ;;
      CrossAttention_Fusion) mode=xattn ;;
    esac
    out="${SAVE_ROOT}/resnet50/seed_${seed}/repeat_${repeat}/${method}"
    [[ -f "${out}/training_complete.marker" ]] && { echo "[SKIP] ${method} seed=${seed}"; continue; }
    mkdir -p "${out}"
    CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" "${PYTHON_BIN}" -u main_finetune.py \
      --model resnet50 --batch_size 32 --epochs 180 --patience 25 --lr 1e-4 \
      --weight_decay 0.05 --warmup_epochs 5 --drop_path 0.2 --input_size 224 \
      --dual_view true --view_mode paired --teacher_mode false --num_workers 8 \
      --seed "${seed}" --device cuda --deterministic true --reseed_before_training true \
      --train_list annotations/DvXray_train.txt --val_list annotations/DvXray_val.txt \
      --classes_file annotations/classes.txt --num_classes 15 --aug_mode conditional \
      --head_type c5 --fuse_mode "${mode}" --base_loss bce \
      --use_semantic_branch false --use_p9_caprs false \
      --summary_csv "${SAVE_ROOT}/training_results.csv" --output_dir "${out}" \
      2>&1 | tee "${out}/train.log"
    touch "${out}/training_complete.marker"
  done
done

echo "Validation training complete. Evaluate only the locked best checkpoints on Test."
