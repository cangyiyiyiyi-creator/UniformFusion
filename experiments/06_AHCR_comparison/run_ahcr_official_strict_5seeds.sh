#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ahcr_official_mpl_${UID}}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
RUN_ID="${RUN_ID:-run_20260831_ahcr_official_strict_resnet50_n5_locked}"
SAVE_ROOT="${SAVE_ROOT:-./runs_ahcr_official_strict_n5/${RUN_ID}}"
MASTER_LOG="${MASTER_LOG:-${SAVE_ROOT}/master.log}"
GPU_ID="${GPU_ID:-0}"
RESUME_PARTIAL="${RESUME_PARTIAL:-true}"
DRY_RUN="${DRY_RUN:-false}"
SOURCE_DIR="${SOURCE_DIR:-third_party/DvXray_official}"
TRAIN_LIST="${TRAIN_LIST:-annotations/DvXray_train.txt}"
VAL_LIST="${VAL_LIST:-annotations/DvXray_val.txt}"
TEST_LIST="${TEST_LIST:-annotations/DvXray_test.txt}"
CLASSES_FILE="${CLASSES_FILE:-annotations/classes.txt}"
MAIN_DETAILED_CSV="${MAIN_DETAILED_CSV:-./runs_final_noanchor_evidence/run_20260830_final_noanchor_all_evidence_locked/resnet_detailed.csv}"

SOURCE_COMMIT="a6bfc1b1299d28e8226c106a94967287a8e30927"
SOURCE_MODEL_SHA256="bbc135c43b908cb33278d3f1baee510a4074e0c9b1a252d36cf23e8a973eca85"
V1_WEIGHTS="/home/hfuu/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
V1_WEIGHTS_SHA256="0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
EXPECTED_TRAIN_SHA256="f0a5c6f810a5725e3336b28df184542343f99e4d9afd5c866e860b4052254dcf"
EXPECTED_VAL_SHA256="a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67"
EXPECTED_TEST_SHA256="6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6"
SEEDS=(930163947 1786430941 553800223 207027553 1716854429)
REPEATS=(1 2 3 4 5)

mkdir -p "${SAVE_ROOT}" "${MPLCONFIGDIR}"
if [[ "${DRY_RUN,,}" != "true" ]] && pgrep -f '[m]ain_finetune.py' >/dev/null; then
  echo "Another main_finetune.py process is active."
  exit 1
fi
if [[ "${DRY_RUN,,}" != "true" ]] && pgrep -f '[r]un_ahcr_official_strict.py' >/dev/null; then
  echo "Another strict AHCR process is active."
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
[[ -s "${V1_WEIGHTS}" ]] || { echo "Missing cached official ResNet50 V1 weights"; exit 2; }
[[ "$(sha256sum "${V1_WEIGHTS}" | awk '{print $1}')" == "${V1_WEIGHTS_SHA256}" ]] || {
  echo "ResNet50 V1 weight checksum mismatch"; exit 2;
}
[[ -s "${MAIN_DETAILED_CSV}" ]] || { echo "Missing locked ResNet50 n=5 CSV"; exit 2; }

bash -n run_ahcr_official_strict_one.sh run_ahcr_official_strict_5seeds.sh
"${PYTHON_BIN}" -m py_compile \
  tools/run_ahcr_official_strict.py tools/summarize_ahcr_official_strict_n5.py
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${PYTHON_BIN}" \
  tools/run_ahcr_official_strict.py --preflight-only --device cpu \
  > "${SAVE_ROOT}/preflight.log" 2>&1
grep -Fq AHCR_OFFICIAL_STRICT_PREFLIGHT_OK "${SAVE_ROOT}/preflight.log"

sha256sum \
  tools/run_ahcr_official_strict.py tools/summarize_ahcr_official_strict_n5.py \
  run_ahcr_official_strict_one.sh run_ahcr_official_strict_5seeds.sh \
  third_party/DvXray_official_LOCK.json "${SOURCE_DIR}/model_ResNet.py" \
  "${SOURCE_DIR}/dataset.py" "${SOURCE_DIR}/loss_func.py" \
  "${SOURCE_DIR}/train.py" "${SOURCE_DIR}/utils.py" "${SOURCE_DIR}/get_ap.py" \
  "${V1_WEIGHTS}" "${TRAIN_LIST}" "${VAL_LIST}" "${TEST_LIST}" \
  > "${SAVE_ROOT}/snapshot_sha256.txt"

{
  echo "method=AHCR-Official"
  echo "purpose=reproduce released architecture and training recipe"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "official_source_modified=false"
  echo "architecture=verbatim upstream model_ResNet.py AHCR"
  echo "training_dataset=verbatim upstream dataset.py data_loader"
  echo "training_loop=verbatim upstream train.py train function"
  echo "loss=verbatim upstream BCELoss reduction=sum"
  echo "weights=ResNet50_Weights.IMAGENET1K_V1"
  echo "optimizer=SGD lr=0.01 momentum=0 weight_decay=0"
  echo "schedule=30 epochs; lr multiplied by 0.1 before epochs 11 and 21"
  echo "batch_size=32 drop_last=true workers=0 gradient_element_clip=5"
  echo "selection=final epoch 30; validation not used for selection"
  echo "evaluation_extension=deterministic released preprocessing without training HSV jitter"
  echo "evaluation_fusion=verbatim upstream function at batch_size=1"
  echo "seeds=${SEEDS[*]}"
  echo "seed_note=upstream does not publish seeds; fixed n=5 added for repeatability"
} > "${SAVE_ROOT}/protocol.txt"

{
  echo "[$(date '+%F %T')] AHCR-Official strict fixed n=5 started"
  for index in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$index]}" REPEAT="${REPEATS[$index]}" \
    SAVE_ROOT="${SAVE_ROOT}" GPU_ID="${GPU_ID}" \
    RESUME_PARTIAL="${RESUME_PARTIAL}" DRY_RUN="${DRY_RUN}" \
    SOURCE_DIR="${SOURCE_DIR}" TRAIN_LIST="${TRAIN_LIST}" \
    VAL_LIST="${VAL_LIST}" TEST_LIST="${TEST_LIST}" \
    CLASSES_FILE="${CLASSES_FILE}" \
      bash run_ahcr_official_strict_one.sh
  done

  if [[ "${DRY_RUN,,}" == "true" ]]; then
    echo "DRY_RUN_OK AHCR-Official strict fixed n=5"
    exit 0
  fi

  "${PYTHON_BIN}" tools/summarize_ahcr_official_strict_n5.py \
    --run-root "${SAVE_ROOT}" --main-detailed-csv "${MAIN_DETAILED_CSV}" \
    --classes-file "${CLASSES_FILE}" --output-root "${SAVE_ROOT}"
  "${PYTHON_BIN}" tools/paired_method_statistics.py \
    --detailed-csv "${SAVE_ROOT}/ahcr_official_strict_n5_detailed.csv" \
    --baseline Plain_BCE --method AHCR_Official_Strict \
    --output-csv "${SAVE_ROOT}/strict_vs_plain_descriptive_statistics.csv" \
    --output-json "${SAVE_ROOT}/strict_vs_plain_descriptive_statistics.json"
  touch "${SAVE_ROOT}/suite_complete.marker"
  echo "[$(date '+%F %T')] AHCR-Official strict fixed n=5 finished"
} 2>&1 | tee -a "${MASTER_LOG}"

