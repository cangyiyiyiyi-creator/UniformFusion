#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
RUN_ID="${RUN_ID:-run_20260902_maxvit_uniform_3seeds_locked}"
SAVE_ROOT="${SAVE_ROOT:-./runs_maxvit_uniform/${RUN_ID}}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-180}"
PATIENCE="${PATIENCE:-25}"
SEEDS=(930163947 1786430941 553800223)
METHODS=(Plain_BCE Uniform_Fusion)

mkdir -p "${SAVE_ROOT}"
if pgrep -f '[m]ain_finetune.py' >/dev/null; then
  echo "Another training is active; MaxViT queue was not started."
  exit 1
fi
bash -n run_maxvit_uniform_one.sh run_maxvit_uniform_3seeds.sh
"${PYTHON_BIN}" -m py_compile main_finetune.py models/timm_backbones.py \
  models/convnextv2_dual.py models/modules/plain_bce_innovations.py \
  tools/evaluate_project_checkpoint.py tools/verify_checkpoint_protocol.py
HF_HUB_OFFLINE=1 "${PYTHON_BIN}" -c \
  'from models.timm_backbones import maxvit_tiny; m=maxvit_tiny(num_classes=0); c=list(m.feature_info.channels()); assert len(c) >= 4; print(f"MaxViT-Tiny offline preflight OK: channels={c}, params={sum(p.numel() for p in m.parameters())}")'
sha256sum main_finetune.py models/timm_backbones.py models/convnextv2_dual.py \
  models/modules/plain_bce_innovations.py run_maxvit_uniform_one.sh \
  run_maxvit_uniform_3seeds.sh annotations/DvXray_train.txt \
  annotations/DvXray_val.txt annotations/DvXray_test.txt \
  > "${SAVE_ROOT}/snapshot_sha256.txt"
{
  echo "protocol=train on train; select checkpoint on val; report test once after all validation runs"
  echo "backbone=maxvit_tiny_rw_224.sw_in1k"
  echo "methods=${METHODS[*]}"
  echo "seeds=${SEEDS[*]}"
  echo "batch_size=${BATCH_SIZE}"
  echo "epochs=${EPOCHS}"
  echo "patience=${PATIENCE}"
} > "${SAVE_ROOT}/protocol.txt"

for phase in val test; do
  [[ "${phase}" == "test" ]] && touch "${SAVE_ROOT}/validation_phase_complete.marker"
  for index in "${!SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      METHOD="${method}" PHASE="${phase}" SEED="${SEEDS[$index]}" \
      REPEAT="$((index + 1))" SAVE_ROOT="${SAVE_ROOT}" GPU_ID="${GPU_ID}" \
      BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
      EPOCHS="${EPOCHS}" PATIENCE="${PATIENCE}" \
        bash run_maxvit_uniform_one.sh
    done
  done
done 2>&1 | tee -a "${SAVE_ROOT}/queue.log"
touch "${SAVE_ROOT}/suite_complete.marker"
echo "MaxViT-Tiny Plain-BCE vs Uniform Fusion fixed 3-seed suite finished."
