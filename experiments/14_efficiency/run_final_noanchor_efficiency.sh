#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-./runs_final_noanchor_confirmation/run_20260830_final_noanchor_resnet50_n5_locked}"
RUN_ID="${RUN_ID:-run_20260830_final_noanchor_efficiency_locked}"
SAVE_ROOT="${SAVE_ROOT:-./runs_final_noanchor_efficiency/${RUN_ID}}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-false}"
SEED=930163947
REPEAT=1
METHODS=(Plain_BCE Final_NoAnchor)
BATCH_SIZES=(1 32)

mkdir -p "${SAVE_ROOT}/profiles"
for method in "${METHODS[@]}"; do
  checkpoint="${SOURCE_ROOT}/resnet50/seed_${SEED}/repeat_${REPEAT}/${method}/checkpoint_best.pth"
  if [[ "${DRY_RUN,,}" != "true" && ! -f "${checkpoint}" ]]; then
    echo "Missing efficiency checkpoint: ${checkpoint}"
    exit 1
  fi
  for batch_size in "${BATCH_SIZES[@]}"; do
    repeats=100
    if [[ "${batch_size}" -gt 1 ]]; then
      repeats=30
    fi
    output_json="${SAVE_ROOT}/profiles/${method}_batch${batch_size}.json"
    command=(
      env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u
      tools/profile_project_checkpoint.py
      --checkpoint "${checkpoint}" --method-label "${method}"
      --output-json "${output_json}" --device cuda
      --batch-size "${batch_size}" --warmup 20 --repeats "${repeats}"
    )
    if [[ "${DRY_RUN,,}" == "true" ]]; then
      printf '[DRY RUN PROFILE] '
      printf '%q ' "${command[@]}"
      printf '\n'
    elif [[ -s "${output_json}" ]]; then
      echo "[SKIP PROFILE] ${method} batch=${batch_size}"
    elif [[ -e "${output_json}" ]]; then
      echo "Incomplete efficiency profile: ${output_json}"
      exit 1
    else
      "${command[@]}"
    fi
  done
done

if [[ "${DRY_RUN,,}" != "true" ]]; then
  "${PYTHON_BIN}" tools/summarize_efficiency_profiles.py \
    --root "${SAVE_ROOT}" --expected 4 \
    --output "${SAVE_ROOT}/efficiency_summary.csv"
  {
    echo "protocol=actual locked checkpoints; paired synthetic inputs; AMP FP16"
    echo "source_root=${SOURCE_ROOT}"
    echo "seed=${SEED}"
    echo "methods=${METHODS[*]}"
    echo "batch_sizes=${BATCH_SIZES[*]}"
    echo "note=torch profiler FLOPs are measured estimates and may undercount unsupported ops"
  } > "${SAVE_ROOT}/protocol.txt"
  touch "${SAVE_ROOT}/efficiency_complete.marker"
fi
