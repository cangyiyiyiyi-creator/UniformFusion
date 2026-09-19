#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
OUT="${OUT:-$ROOT/论文补充验证_20260907/06_效率5Session}"
RAW="$OUT/raw"
PIPELINE="$OUT/pipeline_raw"
mkdir -p "$RAW" "$PIPELINE"
FAIR="$ROOT/论文补充验证_20260907/02_公平Adapter基线/run_20260907_fair_adapters_3seeds"
PLAIN="$ROOT/论文最终归档_20260830/03_基线与消融模型/Plain_BCE__ResNet50__seed_930163947/checkpoint_best.pth"
UNIFORM="$ROOT/论文最终归档_20260830/02_UniformFusion主方法模型/UniformFusion__ResNet50__seed_930163947/checkpoint_best.pth"
DAG=$(find "$FAIR/dagnet_official_adapter/seed_930163947" -name checkpoint_best.pth -print -quit)
ML=$(find "$FAIR/resnet50_ml_decoder_adapter/seed_930163947" -name checkpoint_best.pth -print -quit)
for item in "Plain_BCE|$PLAIN" "Uniform_Fusion|$UNIFORM" "DAGNet|$DAG" "ML_Decoder|$ML"; do
  IFS='|' read -r label ckpt <<< "$item"
  [[ -f "$ckpt" ]] || { echo "missing $label checkpoint"; exit 2; }
  for batch in 1 16; do
    for session in 1 2 3 4 5; do
      file="$RAW/${label}_batch${batch}_session${session}.json"
      if [[ ! -f "$file" ]]; then
        CUDA_VISIBLE_DEVICES=0 "$PY" tools/profile_project_checkpoint.py --checkpoint "$ckpt" --method-label "$label" --batch-size "$batch" --warmup 30 --repeats 100 --device cuda --output-json "$file"
      fi
      pipeline_file="$PIPELINE/${label}_batch${batch}_session${session}.json"
      if [[ ! -f "$pipeline_file" ]]; then
        CUDA_VISIBLE_DEVICES=0 "$PY" tools/profile_end_to_end_checkpoint.py --checkpoint "$ckpt" --method-label "$label" --batch-size "$batch" --warmup 10 --repeats 100 --device cuda --output-json "$pipeline_file"
      fi
    done
  done
done
"$PY" tools/summarize_latency_sessions.py --input-dir "$RAW" --output-csv "$OUT/效率5Session汇总.csv"
"$PY" tools/summarize_latency_sessions.py --input-dir "$PIPELINE" --output-csv "$OUT/完整流水线效率5Session汇总.csv"
printf 'scopes=model_forward_amp,dataloader_h2d_model; sessions=5; repeats=100; batches=1,16\n' > "$OUT/protocol.txt"
touch "$OUT/suite_complete.marker"
