#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SUPPLEMENT="$ROOT/supplementary_verification_20260907"
PRED_ROOT="$SUPPLEMENT/07_repro_source_and_sample_predictions/04_sample_predictions/DvXray"
LIST="annotations/DvXray_test.txt"
CLASSES="annotations/classes.txt"

export_one() {
  local method="$1"
  local seed="$2"
  local run_dir="$3"
  local batch_size="$4"
  local output_dir="$PRED_ROOT/$method/seed_$seed"
  local checkpoint="$run_dir/checkpoint_best.pth"
  local metrics="$run_dir/test_metrics.json"

  if [[ ! -f "$checkpoint" || ! -f "$metrics" ]]; then
    printf 'MISSING_LOCKED_INPUT method=%s seed=%s dir=%s\n' "$method" "$seed" "$run_dir" >&2
    return 1
  fi
  if [[ -d "$output_dir" ]] && rmdir "$output_dir" 2>/dev/null; then
    printf 'REMOVE_EMPTY_OUTPUT method=%s seed=%s\n' "$method" "$seed"
  fi
  if [[ -f "$output_dir/predictions.npz" && -f "$output_dir/prediction_manifest.json" && -f "$output_dir/sample_manifest.csv" ]]; then
    printf 'SKIP_COMPLETE method=%s seed=%s\n' "$method" "$seed"
  elif [[ -e "$output_dir" ]]; then
    printf 'REFUSE_PARTIAL_OUTPUT method=%s seed=%s dir=%s\n' "$method" "$seed" "$output_dir" >&2
    return 1
  else
    "$PYTHON" tools/export_locked_predictions.py \
      --checkpoint "$checkpoint" \
      --list "$LIST" \
      --classes-file "$CLASSES" \
      --output-dir "$output_dir" \
      --expected-metrics-json "$metrics" \
      --dataset-name DvXray \
      --method-name "$method" \
      --seed "$seed" \
      --batch-size "$batch_size" \
      --num-workers 8
  fi
  if [[ ! -f "$output_dir/sample_predictions.csv" ]]; then
    "$PYTHON" tools/materialize_sample_predictions_csv.py \
      --prediction-dir "$output_dir"
  fi
}

"$PYTHON" tools/archive_supplement_repro_materials.py --python "$PYTHON"

seeds=(930163947 1786430941 553800223)
repeats=(1 2 3)
for index in "${!seeds[@]}"; do
  seed="${seeds[$index]}"
  repeat="${repeats[$index]}"

  export_one \
    UF_NoAux "$seed" \
    "$SUPPLEMENT/01_uf_noaux_3seeds/run_20260907_uf_noaux_3seeds/resnet50/seed_$seed/repeat_$repeat/UF_NoAux" \
    32

  export_one \
    DAGNet_OfficialArchitecture_EffectiveBatch32 "$seed" \
    "$SUPPLEMENT/02B_dagnet_batch32_recheck/run_20260907_dagnet_batch32_3seeds/dagnet_official_adapter/seed_$seed/repeat_$repeat/DAGNet_OfficialArchitecture" \
    16

  export_one \
    MLDecoder_DualView "$seed" \
    "$SUPPLEMENT/02_fair_adapter_baselines/run_20260907_fair_adapters_3seeds/resnet50_ml_decoder_adapter/seed_$seed/repeat_$repeat/MLDecoder_DualView" \
    32

  export_one \
    Plain_BCE_OL_Only "$seed" \
    "$SUPPLEMENT/04_single_view_training/run_20260907_single_view_plain/resnet50/seed_$seed/repeat_$repeat/Plain_BCE_OL_Only" \
    32

  export_one \
    Plain_BCE_SD_Only "$seed" \
    "$SUPPLEMENT/04_single_view_training/run_20260907_single_view_plain/resnet50/seed_$seed/repeat_$repeat/Plain_BCE_SD_Only" \
    32

  export_one \
    UF_NoCorrection "$seed" \
    "$SUPPLEMENT/05_residual_ablation/run_20260907_residual_ablation/resnet50/seed_$seed/repeat_$repeat/UF_NoCorrection" \
    32

  export_one \
    UF_NoRamp "$seed" \
    "$SUPPLEMENT/05_residual_ablation/run_20260907_residual_ablation/resnet50/seed_$seed/repeat_$repeat/UF_NoRamp" \
    32
done

"$PYTHON" tools/archive_supplement_repro_materials.py --python "$PYTHON"
printf 'ALL_SUPPLEMENT_PREDICTIONS_COMPLETE root=%s\n' "$PRED_ROOT"
