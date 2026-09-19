#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-./runs_final_noanchor_confirmation/run_20260830_final_noanchor_resnet50_n5_locked}"
RUN_ID="${RUN_ID:-run_20260830_final_noanchor_view_ablation_test_n5_locked}"
SAVE_ROOT="${SAVE_ROOT:-./runs_final_noanchor_view_ablation/${RUN_ID}}"
MASTER_LOG="${MASTER_LOG:-${SAVE_ROOT}/master.log}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DRY_RUN="${DRY_RUN:-false}"
VAL_LIST="${VAL_LIST:-annotations/DvXray_val.txt}"
TEST_LIST="${TEST_LIST:-annotations/DvXray_test.txt}"
EXPECTED_TEST_SHA256="6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6"

SEEDS=(930163947 1786430941 553800223 207027553 1716854429)
REPEATS=(1 2 3 4 5)
METHODS=(Plain_BCE Final_NoAnchor)
MODES=(paired ol_only sd_only mismatched)

mkdir -p "${SAVE_ROOT}"
if [[ "${DRY_RUN,,}" != "true" ]]; then
  [[ -f "${SOURCE_ROOT}/confirmation_complete.marker" ]] || {
    echo "ResNet50 n=5 confirmation is not complete: ${SOURCE_ROOT}"; exit 1;
  }
fi

# The seed-4/5 Final-NoAnchor checkpoints were intentionally removed for
# replication. Rebuild only those missing locked runs before test ablations.
SAVE_ROOT="${SOURCE_ROOT}" GPU_ID="${GPU_ID}" BATCH_SIZE="${BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" DRY_RUN="${DRY_RUN}" \
  bash run_final_noanchor_resnet50_missing_final_2seeds.sh

actual_test_sha="$(sha256sum "${TEST_LIST}" | awk '{print $1}')"
[[ "${actual_test_sha}" == "${EXPECTED_TEST_SHA256}" ]] || {
  echo "Test split checksum mismatch: ${actual_test_sha}"; exit 1;
}

checkpoints=()
for index in "${!SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    checkpoint="${SOURCE_ROOT}/resnet50/seed_${SEEDS[$index]}/repeat_${REPEATS[$index]}/${method}/checkpoint_best.pth"
    checkpoints+=("${checkpoint}")
    if [[ "${DRY_RUN,,}" != "true" && ! -f "${checkpoint}" ]]; then
      echo "Missing checkpoint: ${checkpoint}"
      exit 1
    fi
  done
done

if [[ "${DRY_RUN,,}" != "true" ]]; then
  "${PYTHON_BIN}" tools/verify_checkpoint_protocol.py \
    --checkpoint "${checkpoints[@]}" --expected-val-list "${VAL_LIST}"
fi

sha256sum \
  tools/evaluate_project_checkpoint.py tools/verify_checkpoint_protocol.py \
  tools/summarize_final_view_ablation.py \
  run_final_noanchor_view_ablation_test_n5.sh \
  "${VAL_LIST}" "${TEST_LIST}" \
  > "${SAVE_ROOT}/snapshot_sha256.txt"
{
  echo "protocol=frozen checkpoints selected on val; test inference only"
  echo "purpose=dual-view evidence ablation, not model selection"
  echo "source_root=${SOURCE_ROOT}"
  echo "methods=${METHODS[*]}"
  echo "modes=${MODES[*]}"
  echo "seeds=${SEEDS[*]}"
  echo "evaluations=40"
  echo "test_list=${TEST_LIST}"
  echo "test_sha256=${EXPECTED_TEST_SHA256}"
} > "${SAVE_ROOT}/protocol.txt"

{
  echo "[$(date '+%F %T')] Final-NoAnchor Test view ablation started"
  for mode in "${MODES[@]}"; do
    for index in "${!SEEDS[@]}"; do
      seed="${SEEDS[$index]}"
      repeat="${REPEATS[$index]}"
      for method in "${METHODS[@]}"; do
        checkpoint="${SOURCE_ROOT}/resnet50/seed_${seed}/repeat_${repeat}/${method}/checkpoint_best.pth"
        output_dir="${SAVE_ROOT}/${mode}/seed_${seed}"
        output_json="${output_dir}/${method}.json"
        output_csv="${output_dir}/${method}.csv"
        mkdir -p "${output_dir}"
        if [[ "${DRY_RUN,,}" == "true" ]]; then
          printf '[DRY RUN VIEW] '
          printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
            tools/evaluate_project_checkpoint.py \
            --checkpoint "${checkpoint}" --list "${TEST_LIST}" \
            --classes-file annotations/classes.txt --view-mode "${mode}" \
            --output-json "${output_json}" --output-csv "${output_csv}" \
            --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --device cuda
          printf '\n'
          continue
        fi
        if [[ -s "${output_json}" && -s "${output_csv}" ]]; then
          echo "[SKIP VIEW] mode=${mode} method=${method} seed=${seed}"
          continue
        fi
        if [[ -e "${output_json}" || -e "${output_csv}" ]]; then
          echo "Incomplete view output: mode=${mode} method=${method} seed=${seed}"
          exit 1
        fi
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
          tools/evaluate_project_checkpoint.py \
          --checkpoint "${checkpoint}" --list "${TEST_LIST}" \
          --classes-file annotations/classes.txt --view-mode "${mode}" \
          --output-json "${output_json}" --output-csv "${output_csv}" \
          --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --device cuda
        echo "[$(date '+%F %T')] VIEW DONE mode=${mode} method=${method} seed=${seed}"
      done
    done
  done
  if [[ "${DRY_RUN,,}" != "true" ]]; then
    "${PYTHON_BIN}" tools/summarize_final_view_ablation.py \
      --root "${SAVE_ROOT}" --expected-seeds 5 \
      --expected-list-sha256 "${EXPECTED_TEST_SHA256}"
    touch "${SAVE_ROOT}/evaluation_complete.marker"
  fi
  echo "[$(date '+%F %T')] Final-NoAnchor Test view ablation finished"
} 2>&1 | tee -a "${MASTER_LOG}"
