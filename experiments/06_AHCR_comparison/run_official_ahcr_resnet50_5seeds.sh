#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
RUN_ID="${RUN_ID:-run_20260831_ahcr_uniform_resnet50_n5_locked}"
SAVE_ROOT="${SAVE_ROOT:-./runs_ahcr_uniform_n5/${RUN_ID}}"
MASTER_LOG="${MASTER_LOG:-${SAVE_ROOT}/master.log}"
SUMMARY_CSV="${SUMMARY_CSV:-${SAVE_ROOT}/training_results.csv}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATIENCE="${PATIENCE:-25}"
EPOCHS="${EPOCHS:-180}"
RESUME_PARTIAL="${RESUME_PARTIAL:-true}"
DRY_RUN="${DRY_RUN:-false}"
TRAIN_LIST="${TRAIN_LIST:-annotations/DvXray_train.txt}"
VAL_LIST="${VAL_LIST:-annotations/DvXray_val.txt}"
TEST_LIST="${TEST_LIST:-annotations/DvXray_test.txt}"
SOURCE_DIR="${SOURCE_DIR:-third_party/DvXray_official}"
SOURCE_COMMIT="a6bfc1b1299d28e8226c106a94967287a8e30927"
SOURCE_MODEL_SHA256="bbc135c43b908cb33278d3f1baee510a4074e0c9b1a252d36cf23e8a973eca85"
MAIN_DETAILED_CSV="${MAIN_DETAILED_CSV:-./runs_final_noanchor_evidence/run_20260830_final_noanchor_all_evidence_locked/resnet_detailed.csv}"
VALIDATION_PHASE_MARKER="${SAVE_ROOT}/validation_phase_complete.marker"
COMPONENT_GATE_RUN_ID="${COMPONENT_GATE_RUN_ID:-run_20260831_uniform_component_ablation_3seeds_locked}"
COMPONENT_GATE_ROOT="${COMPONENT_GATE_ROOT:-${ROOT_DIR}/runs_pre_ahcr_uniform_component_gate/${COMPONENT_GATE_RUN_ID}}"
COMPONENT_GATE_REQUEST="${COMPONENT_GATE_ROOT}/queue_requested.marker"
COMPONENT_GATE_COMPLETE="${COMPONENT_GATE_ROOT}/suite_complete.marker"
COMPONENT_GATE_BLOCKED="${COMPONENT_GATE_ROOT}/queue_blocked.marker"

EXPECTED_TRAIN_SHA256="f0a5c6f810a5725e3336b28df184542343f99e4d9afd5c866e860b4052254dcf"
EXPECTED_VAL_SHA256="a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67"
EXPECTED_TEST_SHA256="6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6"
SEEDS=(930163947 1786430941 553800223 207027553 1716854429)
REPEATS=(1 2 3 4 5)

if [[ "${DRY_RUN,,}" != "true" && -f "${COMPONENT_GATE_REQUEST}" && ! -f "${COMPONENT_GATE_COMPLETE}" ]]; then
  echo "[$(date '+%F %T')] Waiting for requested Uniform component ablation before AHCR-Uniform."
  while [[ ! -f "${COMPONENT_GATE_COMPLETE}" ]]; do
    if [[ -f "${COMPONENT_GATE_BLOCKED}" ]]; then
      echo "Uniform component ablation queue is blocked; AHCR-Uniform will not start."
      exit 2
    fi
    sleep 60
  done
  echo "[$(date '+%F %T')] Uniform component ablation complete; AHCR-Uniform released."
fi

mkdir -p "${SAVE_ROOT}"
if [[ "${DRY_RUN,,}" != "true" ]] && pgrep -f '[m]ain_finetune.py' >/dev/null; then
  echo "Another main_finetune.py process is active."
  exit 1
fi
[[ "$(sha256sum "${TRAIN_LIST}" | awk '{print $1}')" == "${EXPECTED_TRAIN_SHA256}" ]] || {
  echo "Train split checksum mismatch"; exit 2;
}
[[ "$(sha256sum "${VAL_LIST}" | awk '{print $1}')" == "${EXPECTED_VAL_SHA256}" ]] || {
  echo "Validation split checksum mismatch"; exit 2;
}
[[ "$(sha256sum "${TEST_LIST}" | awk '{print $1}')" == "${EXPECTED_TEST_SHA256}" ]] || {
  echo "Test split checksum mismatch"; exit 2;
}
[[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" == "${SOURCE_COMMIT}" ]] || {
  echo "Official AHCR commit mismatch"; exit 2;
}
[[ -z "$(git -C "${SOURCE_DIR}" status --porcelain)" ]] || {
  echo "Official AHCR checkout is not clean"; exit 2;
}
[[ "$(sha256sum "${SOURCE_DIR}/model_ResNet.py" | awk '{print $1}')" == "${SOURCE_MODEL_SHA256}" ]] || {
  echo "Official AHCR model source hash mismatch"; exit 2;
}
[[ -s "${MAIN_DETAILED_CSV}" ]] || { echo "Missing locked ResNet50 n=5 CSV"; exit 2; }

bash -n run_official_ahcr_one.sh run_official_ahcr_resnet50_5seeds.sh
"${PYTHON_BIN}" -m py_compile \
  models/official_ahcr_adapter.py main_finetune.py engine_finetune.py \
  tools/evaluate_project_checkpoint.py tools/verify_checkpoint_protocol.py \
  tools/smoke_test_official_ahcr_adapter.py \
  tools/summarize_official_ahcr_n5.py tools/paired_method_statistics.py
"${PYTHON_BIN}" tools/smoke_test_official_ahcr_adapter.py \
  > "${SAVE_ROOT}/preflight.log" 2>&1
grep -Fq OFFICIAL_AHCR_ADAPTER_SMOKE_OK "${SAVE_ROOT}/preflight.log"

sha256sum \
  models/official_ahcr_adapter.py main_finetune.py engine_finetune.py \
  tools/evaluate_project_checkpoint.py tools/verify_checkpoint_protocol.py \
  tools/smoke_test_official_ahcr_adapter.py \
  tools/summarize_official_ahcr_n5.py tools/paired_method_statistics.py \
  run_official_ahcr_one.sh run_official_ahcr_resnet50_5seeds.sh \
  third_party/DvXray_official_LOCK.json "${SOURCE_DIR}/model_ResNet.py" \
  "${TRAIN_LIST}" "${VAL_LIST}" "${TEST_LIST}" \
  > "${SAVE_ROOT}/snapshot_sha256.txt"
{
  echo "method=AHCR-Uniform"
  echo "purpose=official AHCR architecture under our locked fair protocol"
  echo "upstream=https://github.com/Mbwslib/DvXray.git"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "source_model_sha256=${SOURCE_MODEL_SHA256}"
  echo "official_source_modified=false"
  echo "architecture=upstream model_ResNet.py AHCR"
  echo "fusion=batch-invariant confidence-weighted probability fusion; B=1 equivalent to upstream"
  echo "supervision=mean of OL and SD BCEWithLogitsLoss"
  echo "pretrained_weights=IMAGENET1K_V2 to match local Plain-BCE ResNet50"
  echo "optimizer=AdamW lr=1e-4 weight_decay=0.05"
  echo "schedule=5 epoch warmup; cosine minimum lr=1e-6; max_epochs=${EPOCHS}; patience=${PATIENCE}"
  echo "selection=train on train; select checkpoint on val; lock all five; report test once"
  echo "seeds=${SEEDS[*]}"
  echo "batch_size=${BATCH_SIZE}"
  echo "official_upstream_training_note=upstream script uses V1, SGD, 30 epochs and no val selection; not used for fair main table"
} > "${SAVE_ROOT}/protocol.txt"

run_phase() {
  local phase="$1"
  for index in "${!SEEDS[@]}"; do
    PHASE="${phase}" SEED="${SEEDS[$index]}" REPEAT="${REPEATS[$index]}" \
    SAVE_ROOT="${SAVE_ROOT}" SUMMARY_CSV="${SUMMARY_CSV}" GPU_ID="${GPU_ID}" \
    BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
    PATIENCE="${PATIENCE}" EPOCHS="${EPOCHS}" \
    RESUME_PARTIAL="${RESUME_PARTIAL}" DRY_RUN="${DRY_RUN}" \
    TRAIN_LIST="${TRAIN_LIST}" VAL_LIST="${VAL_LIST}" TEST_LIST="${TEST_LIST}" \
    SOURCE_DIR="${SOURCE_DIR}" SOURCE_COMMIT="${SOURCE_COMMIT}" \
    PRETRAINED_WEIGHTS=IMAGENET1K_V2 \
    VALIDATION_PHASE_MARKER="${VALIDATION_PHASE_MARKER}" \
      bash run_official_ahcr_one.sh
  done
}

{
  echo "[$(date '+%F %T')] AHCR-Uniform fixed n=5 validation phase started"
  run_phase val
  if [[ "${DRY_RUN,,}" == "true" ]]; then
    run_phase test
    echo "DRY_RUN_OK AHCR-Uniform fixed n=5"
    exit 0
  fi
  touch "${VALIDATION_PHASE_MARKER}"
  echo "[$(date '+%F %T')] All five validation checkpoints locked"
  run_phase test

  "${PYTHON_BIN}" tools/summarize_official_ahcr_n5.py \
    --run-root "${SAVE_ROOT}" --main-detailed-csv "${MAIN_DETAILED_CSV}" \
    --output-root "${SAVE_ROOT}"
  "${PYTHON_BIN}" tools/paired_method_statistics.py \
    --detailed-csv "${SAVE_ROOT}/ahcr_uniform_n5_detailed.csv" \
    --baseline Plain_BCE --method AHCR_Uniform \
    --output-csv "${SAVE_ROOT}/ahcr_uniform_vs_plain_statistics.csv" \
    --output-json "${SAVE_ROOT}/ahcr_uniform_vs_plain_statistics.json"
  "${PYTHON_BIN}" tools/paired_method_statistics.py \
    --detailed-csv "${SAVE_ROOT}/ahcr_uniform_n5_detailed.csv" \
    --baseline AHCR_Uniform --method Final_NoAnchor \
    --output-csv "${SAVE_ROOT}/final_vs_ahcr_uniform_statistics.csv" \
    --output-json "${SAVE_ROOT}/final_vs_ahcr_uniform_statistics.json"
  touch "${SAVE_ROOT}/suite_complete.marker"
  echo "[$(date '+%F %T')] AHCR-Uniform fixed n=5 finished"
} 2>&1 | tee -a "${MASTER_LOG}"
