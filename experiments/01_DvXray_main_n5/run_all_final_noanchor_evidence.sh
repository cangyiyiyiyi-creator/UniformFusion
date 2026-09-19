#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATIENCE="${PATIENCE:-25}"
EPOCHS="${EPOCHS:-180}"
RESUME_PARTIAL="${RESUME_PARTIAL:-true}"
DRY_RUN="${DRY_RUN:-false}"

RESNET_RUN_ID="${RESNET_RUN_ID:-run_20260830_final_noanchor_resnet50_n5_locked}"
RESNET_ROOT="${RESNET_ROOT:-./runs_final_noanchor_confirmation/${RESNET_RUN_ID}}"
CONVNEXT_RUN_ID="${CONVNEXT_RUN_ID:-run_20260830_final_noanchor_convnextv2_3seeds_locked}"
CONVNEXT_ROOT="${CONVNEXT_ROOT:-./runs_final_noanchor_generalization/${CONVNEXT_RUN_ID}}"
VIEW_RUN_ID="${VIEW_RUN_ID:-run_20260830_final_noanchor_view_ablation_test_n5_locked}"
VIEW_ROOT="${VIEW_ROOT:-./runs_final_noanchor_view_ablation/${VIEW_RUN_ID}}"
EFFICIENCY_RUN_ID="${EFFICIENCY_RUN_ID:-run_20260830_final_noanchor_efficiency_locked}"
EFFICIENCY_ROOT="${EFFICIENCY_ROOT:-./runs_final_noanchor_efficiency/${EFFICIENCY_RUN_ID}}"
BUNDLE_ROOT="${BUNDLE_ROOT:-./runs_final_noanchor_evidence/run_20260830_final_noanchor_all_evidence_locked}"
MASTER_ROOT="${MASTER_ROOT:-./runs_final_noanchor_evidence}"
MASTER_LOG="${MASTER_LOG:-${MASTER_ROOT}/run_20260830_all_evidence_master.log}"

mkdir -p "${MASTER_ROOT}"
ORCHESTRATOR_PID_FILE="${MASTER_ROOT}/run_20260830_all_evidence.pid"
ORCHESTRATOR_STATUS_FILE="${MASTER_ROOT}/run_20260830_all_evidence.status"
printf '%s\n' "$$" > "${ORCHESTRATOR_PID_FILE}"
printf '%s\n' "running" > "${ORCHESTRATOR_STATUS_FILE}"
record_exit_status() {
  local status=$?
  if [[ "${status}" -eq 0 ]]; then
    printf '%s\n' "complete" > "${ORCHESTRATOR_STATUS_FILE}"
  else
    printf 'failed exit=%s\n' "${status}" > "${ORCHESTRATOR_STATUS_FILE}"
  fi
}
trap record_exit_status EXIT
if [[ "${DRY_RUN,,}" != "true" ]] && pgrep -f '[m]ain_finetune.py' >/dev/null; then
  echo "Another main_finetune.py process is active."
  exit 1
fi

{
  echo "[$(date '+%F %T')] ALL FINAL-NOANCHOR EVIDENCE START"
  echo "order=ResNet50_n5 ConvNeXtV2_3seeds ViewAblation_Test_n5 Efficiency Bundle"
  echo "main_method=Final_NoAnchor"
  echo "router_search=frozen"

  "${PYTHON_BIN}" tools/smoke_test_final_noanchor_backbones.py

  RUN_ID="${RESNET_RUN_ID}" SAVE_ROOT="${RESNET_ROOT}" \
  MASTER_LOG="${RESNET_ROOT}/master.log" \
  GPU_ID="${GPU_ID}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
  PATIENCE="${PATIENCE}" EPOCHS="${EPOCHS}" \
  RESUME_PARTIAL="${RESUME_PARTIAL}" DRY_RUN="${DRY_RUN}" \
    bash run_final_noanchor_resnet50_n5.sh

  RUN_ID="${CONVNEXT_RUN_ID}" SAVE_ROOT="${CONVNEXT_ROOT}" \
  MASTER_LOG="${CONVNEXT_ROOT}/master.log" \
  GPU_ID="${GPU_ID}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
  PATIENCE="${PATIENCE}" EPOCHS="${EPOCHS}" \
  RESUME_PARTIAL="${RESUME_PARTIAL}" DRY_RUN="${DRY_RUN}" \
    bash run_final_noanchor_convnextv2_3seeds.sh

  SOURCE_ROOT="${RESNET_ROOT}" RUN_ID="${VIEW_RUN_ID}" SAVE_ROOT="${VIEW_ROOT}" \
  MASTER_LOG="${VIEW_ROOT}/master.log" \
  GPU_ID="${GPU_ID}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
  DRY_RUN="${DRY_RUN}" \
    bash run_final_noanchor_view_ablation_test_n5.sh

  SOURCE_ROOT="${RESNET_ROOT}" RUN_ID="${EFFICIENCY_RUN_ID}" \
  SAVE_ROOT="${EFFICIENCY_ROOT}" GPU_ID="${GPU_ID}" DRY_RUN="${DRY_RUN}" \
    bash run_final_noanchor_efficiency.sh

  if [[ "${DRY_RUN,,}" != "true" ]]; then
    "${PYTHON_BIN}" tools/build_final_evidence_bundle.py \
      --resnet-root "${RESNET_ROOT}" --convnext-root "${CONVNEXT_ROOT}" \
      --view-root "${VIEW_ROOT}" --efficiency-root "${EFFICIENCY_ROOT}" \
      --output-root "${BUNDLE_ROOT}"
  fi
  echo "[$(date '+%F %T')] ALL FINAL-NOANCHOR EVIDENCE FINISHED"
} 2>&1 | tee -a "${MASTER_LOG}"
