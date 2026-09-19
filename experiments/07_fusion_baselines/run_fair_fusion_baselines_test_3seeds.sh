#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-./runs_fair_fusion_baselines/run_20260903_fair_fusion_3seeds}"
TEST_LIST="${TEST_LIST:-annotations/DvXray_test.txt}"
CLASSES_FILE="${CLASSES_FILE:-annotations/classes.txt}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEEDS=(930163947 1786430941 553800223)
METHODS=(Mean_Fusion Max_Fusion Concat_Fusion CrossAttention_Fusion)

[[ -f "${TEST_LIST}" ]] || { echo "Missing Test list: ${TEST_LIST}"; exit 2; }
[[ -f "${CLASSES_FILE}" ]] || { echo "Missing classes file: ${CLASSES_FILE}"; exit 2; }

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  repeat="$((i + 1))"
  for method in "${METHODS[@]}"; do
    out="${SAVE_ROOT}/resnet50/seed_${seed}/repeat_${repeat}/${method}"
    checkpoint="${out}/checkpoint_best.pth"
    output_json="${out}/test_metrics.json"
    output_csv="${out}/test_metrics.csv"

    [[ -f "${out}/training_complete.marker" ]] || {
      echo "Missing training completion marker: ${out}"; exit 2;
    }
    [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}"; exit 2; }
    if [[ -s "${output_json}" && -s "${output_csv}" ]]; then
      echo "[SKIP TEST] ${method} seed=${seed}"
      continue
    fi
    if [[ -e "${output_json}" || -e "${output_csv}" ]]; then
      echo "Incomplete Test output exists: ${out}"; exit 2
    fi

    "${PYTHON_BIN}" tools/verify_checkpoint_protocol.py \
      --checkpoint "${checkpoint}" \
      --expected-val-list annotations/DvXray_val.txt

    echo "[$(date '+%F %T')] TEST START ${method} seed=${seed}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
      tools/evaluate_project_checkpoint.py \
      --checkpoint "${checkpoint}" \
      --list "${TEST_LIST}" \
      --classes-file "${CLASSES_FILE}" \
      --view-mode paired \
      --output-json "${output_json}" \
      --output-csv "${output_csv}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --device cuda \
      2>&1 | tee "${out}/test.log"
    echo "[$(date '+%F %T')] TEST DONE ${method} seed=${seed}"
  done
done

echo "All 12 locked fair-fusion checkpoints have been evaluated on Test."
