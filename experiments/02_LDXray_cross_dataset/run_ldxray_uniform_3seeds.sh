#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
RUN_ID="${RUN_ID:-run_20260901_ldxray_uniform_3seeds}"
SAVE_ROOT="${SAVE_ROOT:-./runs_ldxray_uniform/${RUN_ID}}"
DATASET_ROOT="${DATASET_ROOT:-./data/LDXray}"
SEEDS=(930163947 1786430941 553800223)
METHODS=(Plain_BCE Uniform_Fusion)
mkdir -p "${SAVE_ROOT}" annotations/ldxray

"${PYTHON_BIN}" tools/prepare_ldxray_multilabel.py --dataset-root "${DATASET_ROOT}" \
  --output-dir annotations/ldxray --val-ratio 0.1 --split-seed 20260901
bash -n run_ldxray_uniform_one.sh run_ldxray_uniform_3seeds.sh
"${PYTHON_BIN}" -m py_compile tools/prepare_ldxray_multilabel.py main_finetune.py
for phase in val test; do
  [[ "${phase}" == "test" ]] && touch "${SAVE_ROOT}/validation_phase_complete.marker"
  for i in "${!SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      METHOD="${method}" PHASE="${phase}" SEED="${SEEDS[$i]}" REPEAT="$((i+1))" \
        SAVE_ROOT="${SAVE_ROOT}" bash run_ldxray_uniform_one.sh
    done
  done
done 2>&1 | tee -a "${SAVE_ROOT}/queue.log"
touch "${SAVE_ROOT}/suite_complete.marker"
