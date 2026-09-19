#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
RUN_ID="${RUN_ID:-run_20260902_ldxray_uniform_stepmatched_3seeds}"
SAVE_ROOT="${SAVE_ROOT:-./runs_ldxray_uniform_stepmatched/${RUN_ID}}"
DATASET_ROOT="${DATASET_ROOT:-/home/hfuu/桌面/LDXRAY-20260901/dataset_clean}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-15}"
SEEDS=(930163947 1786430941 553800223)
METHODS=(Plain_BCE Uniform_Fusion_StepMatched)

mkdir -p "${SAVE_ROOT}" annotations/ldxray
if pgrep -f '[m]ain_finetune.py' >/dev/null; then
  echo "Another main_finetune.py process is active; refusing concurrent training."
  exit 1
fi
"${PYTHON_BIN}" tools/prepare_ldxray_multilabel.py \
  --dataset-root "${DATASET_ROOT}" --output-dir annotations/ldxray \
  --val-ratio 0.1 --split-seed 20260901
bash -n run_ldxray_uniform_one.sh run_ldxray_uniform_stepmatched_3seeds.sh
"${PYTHON_BIN}" -m py_compile tools/prepare_ldxray_multilabel.py \
  main_finetune.py models/convnextv2_dual.py \
  models/modules/plain_bce_innovations.py
sha256sum main_finetune.py models/convnextv2_dual.py \
  models/modules/plain_bce_innovations.py run_ldxray_uniform_one.sh \
  run_ldxray_uniform_stepmatched_3seeds.sh \
  annotations/ldxray/LDXray_train.txt annotations/ldxray/LDXray_val.txt \
  annotations/ldxray/LDXray_test.txt > "${SAVE_ROOT}/snapshot_sha256.txt"
{
  echo "protocol=train on train; select best checkpoint on val; test once after all validation runs"
  echo "purpose=LDXray step-matched Uniform Fusion confirmation"
  echo "methods=${METHODS[*]}"
  echo "seeds=${SEEDS[*]}"
  echo "innovation_warmup_epochs=3"
  echo "innovation_ramp_epochs=5"
  echo "innovation_gamma_init=0.003"
  echo "innovation_gamma_max=0.02"
  echo "innovation_guard_weight=0.15"
  echo "innovation_aux_weight=0.03"
} > "${SAVE_ROOT}/protocol.txt"

for phase in val test; do
  [[ "${phase}" == "test" ]] && touch "${SAVE_ROOT}/validation_phase_complete.marker"
  for index in "${!SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      METHOD="${method}" PHASE="${phase}" SEED="${SEEDS[$index]}" \
      REPEAT="$((index + 1))" SAVE_ROOT="${SAVE_ROOT}" GPU_ID="${GPU_ID}" \
      BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
      EPOCHS="${EPOCHS}" PATIENCE="${PATIENCE}" \
      INNOV_WARMUP_EPOCHS=3 INNOV_RAMP_EPOCHS=5 \
      INNOV_GAMMA_INIT=0.003 INNOV_GAMMA_MAX=0.02 \
      INNOV_GUARD_WEIGHT=0.15 bash run_ldxray_uniform_one.sh
    done
  done
done 2>&1 | tee -a "${SAVE_ROOT}/queue.log"
touch "${SAVE_ROOT}/suite_complete.marker"
echo "LDXray step-matched Uniform Fusion fixed 3-seed suite finished."
