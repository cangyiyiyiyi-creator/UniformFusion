#!/usr/bin/env bash
# =============================================================================
# UniformFusion — one-stop entry script (main.sh)
#
# Paper: Class-Conditioned Multi-Scale Regional Learning for Dual-View
#        X-Ray Multi-Label Recognition
#
# This script chains "prepare annotations -> preflight -> train the main method ->
# evaluate the test set once" into a single reproducible pipeline and enforces the
# protocol of section 4.1.3 of the paper:
#   * train on the training split -> select the best checkpoint on validation ->
#     after locking, evaluate the test set exactly once
#   * fixed random seeds and deterministic algorithms (CUBLAS_WORKSPACE_CONFIG=:4096:8)
#   * SHA256 verification of the locked sources and split manifests
#
# Usage:
#   bash main.sh                 # same as: bash main.sh all
#   bash main.sh help            # list every subcommand
#   bash main.sh setup           # only expand data_splits/ into core/annotations/
#   bash main.sh preflight       # only run the environment and integrity checks
#   bash main.sh train           # only train (checkpoint selection on validation)
#   bash main.sh eval            # only evaluate the test set, once (needs checkpoint_best.pth)
#   bash main.sh status          # show the artefacts and metrics of this run
#
# Common environment variables (all overridable):
#   PYTHON_BIN=...   GPU_ID=0        SEED=930163947
#   BATCH_SIZE=32    EPOCHS=180      PATIENCE=25      NUM_WORKERS=8
#   RUN_ID=...       OUT_ROOT=...    SUMMARY_CSV=...  MASTER_LOG=...
#   SKIP_CHECKSUM=true   FORCE_RETEST=true   DRY_RUN=true
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-${REPO_DIR}/core}"
ANNOT_DIR="${ANNOT_DIR:-${CODE_DIR}/annotations}"
SPLITS_DIR="${SPLITS_DIR:-${REPO_DIR}/data_splits}"
TOOLS_DIR="${CODE_DIR}/tools"

# ---------------------------------------------------------------------------
# 2. Overridable run configuration (defaults match the locked scripts in experiments/)
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-930163947}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-180}"
PATIENCE="${PATIENCE:-25}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
DROP_PATH="${DROP_PATH:-0.2}"
INPUT_SIZE="${INPUT_SIZE:-224}"
NUM_CLASSES="${NUM_CLASSES:-15}"

RUN_ID="${RUN_ID:-run_$(date +%Y%m%d)_uniform_fusion_main}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/runs_uniform_fusion/${RUN_ID}}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/resnet50/seed_${SEED}}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_ROOT}/training_results.csv}"
MASTER_LOG="${MASTER_LOG:-${OUT_ROOT}/main.log}"
CHECKPOINT="${CHECKPOINT:-${OUT_DIR}/checkpoint_best.pth}"
TEST_LOCK="${TEST_LOCK:-${OUT_DIR}/test_eval_done.marker}"
PROTOCOL_FILE="${PROTOCOL_FILE:-${OUT_ROOT}/protocol.txt}"

CLASSES_FILE="${CLASSES_FILE:-annotations/classes.txt}"
TRAIN_LIST="${TRAIN_LIST:-annotations/DvXray_train.txt}"
VAL_LIST="${VAL_LIST:-annotations/DvXray_val.txt}"
TEST_LIST="${TEST_LIST:-annotations/DvXray_test.txt}"

MAIN_FINETUNE_SCRIPT="${MAIN_FINETUNE_SCRIPT:-main_finetune.py}"
RESUME_PARTIAL="${RESUME_PARTIAL:-true}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_CHECKSUM="${SKIP_CHECKSUM:-false}"
SKIP_DATA_CHECK="${SKIP_DATA_CHECK:-false}"
FORCE_RETEST="${FORCE_RETEST:-false}"
FORCE_COPY="${FORCE_COPY:-false}"
RELINK_FROM="${RELINK_FROM:-}"
RELINK_TO="${RELINK_TO:-}"

# ---------------------------------------------------------------------------
# 3. Hashes locked by the paper (see README section 4 / HASH_MAPPING.txt)
#    These are the hashes of the English edition (v1.1-paper). The original
#    Chinese snapshot of the paper archive is preserved by the v1.0-paper tag;
#    HASH_MAPPING.txt maps every original hash to its current value.
# ---------------------------------------------------------------------------
SHA_MAIN_FINETUNE="e157ce6f264e9b2e8fa841bdc5958863a506966964850ceae961c735f963c2d0"
SHA_CONVNEXTV2_DUAL="2ee664f54bc6cd7d6db0e09a40be70f6143072a9eeeb59fc311ff79091426a99"
SHA_ENGINE_FINETUNE="3c41cc4b264b19ad26449ce30519451c6e8f4c7c6e54a81c735c188176a86cf7"
SHA_TRAIN_SPLIT="f0a5c6f810a5725e3336b28df184542343f99e4d9afd5c866e860b4052254dcf"
SHA_VAL_SPLIT="a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67"
SHA_TEST_SPLIT="6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6"

# ---------------------------------------------------------------------------
# 4. Small helper functions
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
info() { printf '  - %s\n' "$*"; }
warn() { printf '  ! %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

is_dry() { [[ "${DRY_RUN,,}" == "true" ]]; }

# Print or execute a command (with DRY_RUN=true it is only printed)
run_cmd() {
  if is_dry; then
    printf '[DRY RUN] '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

require_file() {
  [[ -s "$1" ]] || die "missing file: $1"
}

# Relative paths are resolved against the code root (= core/)
resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *)  printf '%s\n' "${CODE_DIR}/$1" ;;
  esac
}

# Record the path and SHA256 of a list file (NA when it does not exist)
record_hash() {
  local label="$1" path
  path="$(resolve_path "$2")"
  if [[ -s "${path}" ]]; then
    echo "${label}=${2} sha256=$(sha256sum "${path}" | awk '{print $1}')"
  else
    echo "${label}=${2} sha256=NA"
  fi
}

verify_sha256() {
  local file="$1" expected="$2" label="$3" actual
  require_file "${file}"
  actual="$(sha256sum "${file}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    die "${label} checksum mismatch: ${file}
  expected ${expected}
  actual   ${actual}
  (make sure the locked sources/split manifests were not modified; set SKIP_CHECKSUM=true to bypass)"
  fi
  info "${label} SHA256 OK  ${file}"
}

# Copy files into a target directory; existing files are skipped unless FORCE_COPY=true
copy_into() {
  local dest_dir="$1"; shift
  local src base
  for src in "$@"; do
    base="$(basename "${src}")"
    if [[ ! -e "${src}" ]]; then
      die "missing source file: ${src}"
    fi
    if [[ "${FORCE_COPY,,}" == "true" || ! -e "${dest_dir}/${base}" ]]; then
      run_cmd cp -f "${src}" "${dest_dir}/"
    else
      info "already present, skipped: ${base}"
    fi
  done
}

# ---------------------------------------------------------------------------
# 5. setup: expand data_splits/ into the core/annotations/ layout the scripts expect
# ---------------------------------------------------------------------------
cmd_setup() {
  log "preparing annotation directory: ${ANNOT_DIR}"
  run_cmd mkdir -p "${ANNOT_DIR}/ldxray"
  # DvXray: classes.txt + DvXray_{train,val,test}.txt (used by the main experiments)
  copy_into "${ANNOT_DIR}" \
    "${SPLITS_DIR}/DvXray/classes.txt" \
    "${SPLITS_DIR}/DvXray/DvXray_train.txt" \
    "${SPLITS_DIR}/DvXray/DvXray_val.txt" \
    "${SPLITS_DIR}/DvXray/DvXray_test.txt"
  # LDXray: used by 02_LDXray_cross_dataset (annotations/ldxray/)
  copy_into "${ANNOT_DIR}/ldxray" \
    "${SPLITS_DIR}/LDXray/LDXray_train.txt" \
    "${SPLITS_DIR}/LDXray/LDXray_val.txt" \
    "${SPLITS_DIR}/LDXray/LDXray_test.txt" \
    "${SPLITS_DIR}/LDXray/ldxray_classes.txt" \
    "${SPLITS_DIR}/LDXray/LDXray_split_manifest.json"

  if ! is_dry; then
    info "DvXray splits: $(wc -l < "${ANNOT_DIR}/DvXray_train.txt") / $(wc -l < "${ANNOT_DIR}/DvXray_val.txt") / $(wc -l < "${ANNOT_DIR}/DvXray_test.txt")  (train/val/test)"
  fi
}

ensure_annotations() {
  if is_dry; then
    return 0   # nothing is written in dry-run mode, avoids repeated setup output
  fi
  if [[ ! -s "${ANNOT_DIR}/classes.txt" || ! -s "${ANNOT_DIR}/DvXray_train.txt" ]]; then
    cmd_setup
  fi
}

# ---------------------------------------------------------------------------
# 6. preflight: interpreter, dependencies, CUDA, locked files and split verification
# ---------------------------------------------------------------------------
cmd_preflight() {
  log "running environment and protocol checks"
  ensure_annotations

  [[ -x "${PYTHON_BIN}" ]] || die "Python interpreter not found: ${PYTHON_BIN}
  set PYTHON_BIN (e.g. PYTHON_BIN=\$(which python))"
  info "PYTHON_BIN = ${PYTHON_BIN}"

  if is_dry; then
    info "[DRY RUN] skipping dependency and CUDA probe"
  else
    "${PYTHON_BIN}" - <<'PY'
import importlib, sys
print(f"  - python   : {sys.version.split()[0]}")
for mod in ("torch", "torchvision", "timm", "numpy", "pandas"):
    try:
        m = importlib.import_module(mod)
        print(f"  - {mod:<10}: {getattr(m, '__version__', 'unknown')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  - {mod:<10}: MISSING ({exc})")
try:
    import torch
    print(f"  - cuda     : available={torch.cuda.is_available()} "
          f"runtime={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  - device0  : {torch.cuda.get_device_name(0)}")
except Exception:  # noqa: BLE001
    pass
PY
  fi

  if [[ "${SKIP_CHECKSUM,,}" == "true" ]]; then
    warn "SKIP_CHECKSUM=true: skipping every SHA256 check (results lose protocol validity)"
  else
    verify_sha256 "${CODE_DIR}/${MAIN_FINETUNE_SCRIPT}" "${SHA_MAIN_FINETUNE}" "locked source main_finetune.py"
    verify_sha256 "${CODE_DIR}/models/convnextv2_dual.py" "${SHA_CONVNEXTV2_DUAL}" "locked source convnextv2_dual.py"
    verify_sha256 "${CODE_DIR}/engine_finetune.py" "${SHA_ENGINE_FINETUNE}" "locked source engine_finetune.py"
    verify_sha256 "${SPLITS_DIR}/DvXray/DvXray_train.txt" "${SHA_TRAIN_SPLIT}" "DvXray train split"
    verify_sha256 "${SPLITS_DIR}/DvXray/DvXray_val.txt" "${SHA_VAL_SPLIT}" "DvXray val split"
    verify_sha256 "${SPLITS_DIR}/DvXray/DvXray_test.txt" "${SHA_TEST_SPLIT}" "DvXray test split"
  fi

  if is_dry; then
    info "[DRY RUN] skipping syntax check"
    return 0
  fi
  ( cd "${CODE_DIR}" && PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" - <<'PY'
import pathlib, sys

files = [
    "main_finetune.py",
    "engine_finetune.py",
    "models/convnextv2_dual.py",
    "models/modules/plain_bce_innovations.py",
    "tools/evaluate_project_checkpoint.py",
    "tools/verify_checkpoint_protocol.py",
]
bad = []
for name in files:
    path = pathlib.Path(name)
    if not path.is_file():
        bad.append(f"{name}: file not found")
        continue
    try:
        compile(path.read_text(encoding="utf-8"), name, "exec")
    except SyntaxError as exc:  # noqa: PERF203
        bad.append(f"{name}: {exc}")
if bad:
    print("\n".join(bad))
    sys.exit(1)
print(f"  - syntax check passed ({len(files)} key files)")
PY
  )

  local running
  running="$(pgrep -fc '[m]ain_finetune.py' || true)"
  if [[ "${running:-0}" -gt 0 ]]; then
    warn "${running} main_finetune.py process(es) already running; starting another run may exhaust GPU memory"
  fi

  check_dataset_access
  log "preflight passed"
}

# Sample the annotation lists to check that the images are reachable on this machine
# (the split files store the original machine's absolute paths)
check_dataset_access() {
  [[ "${SKIP_DATA_CHECK,,}" == "true" ]] && { warn "SKIP_DATA_CHECK=true: skipping the dataset accessibility check"; return 0; }
  ACCESS_LISTS="${TRAIN_LIST}|${VAL_LIST}|${TEST_LIST}" \
  ACCESS_CODE_DIR="${CODE_DIR}" \
    "${PYTHON_BIN}" - <<'PY'
import os, pathlib, sys

code_dir = pathlib.Path(os.environ["ACCESS_CODE_DIR"])
problems = []


def resolve(entry: str) -> pathlib.Path:
    path = pathlib.Path(entry)
    return path if path.is_absolute() else code_dir / path


for entry in os.environ["ACCESS_LISTS"].split("|"):
    path = resolve(entry)
    if not path.is_file():
        problems.append(f"{entry}: list file not found")
        continue
    lines = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()][:20]
    missing = 0
    for toks in lines:
        for img in toks[:2]:
            if not resolve(img).is_file():
                missing += 1
    if missing:
        problems.append(f"{path.name}: {missing} of {len(lines)} sampled rows have missing image files")
    else:
        print(f"  - {path.name}: sampled images are reachable")

if problems:
    print("  ! dataset path check failed:")
    for item in problems:
        print(f"    - {item}")
    print("    -> the split files store the original machine's absolute paths; point them at your own data root first:")
    print("       RELINK_FROM=<old prefix> RELINK_TO=<your path> bash main.sh relink")
    print("       (set SKIP_DATA_CHECK=true to bypass this check once the paths are known to be fine)")
    sys.exit(1)
PY
}

# ---------------------------------------------------------------------------
# 7. train: the Uniform Fusion main method (selects on validation, never touches test)
# ---------------------------------------------------------------------------
cmd_train() {
  log "training Uniform Fusion (main method, internal config name Final_NoAnchor_NoRouter)"
  ensure_annotations

  run_cmd mkdir -p "${OUT_DIR}"

  if [[ ! -f "${PROTOCOL_FILE}" ]] && ! is_dry; then
    {
      echo "protocol=train on train; select checkpoint on val; report locked checkpoint on test once"
      echo "method=Uniform_Fusion (Final_NoAnchor_NoRouter)"
      echo "paper=Table 1 DvXray ResNet-50 n=5 (single seed run)"
      echo "seed=${SEED}"
      echo "batch_size=${BATCH_SIZE}"
      echo "epochs=${EPOCHS}"
      echo "patience=${PATIENCE}"
      echo "deterministic=true"
      echo "cublas_workspace_config=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
      record_hash "train_list" "${TRAIN_LIST}"
      record_hash "val_list" "${VAL_LIST}"
      record_hash "test_list" "${TEST_LIST}"
      record_hash "classes_file" "${CLASSES_FILE}"
    } > "${PROTOCOL_FILE}"
  fi

  local train_cmd=(
    "${PYTHON_BIN}" -u "${MAIN_FINETUNE_SCRIPT}"
    --aug_mode conditional --patience "${PATIENCE}"
    --model resnet50 --model_prefix ""
    --batch_size "${BATCH_SIZE}" --epochs "${EPOCHS}" --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}" --warmup_epochs "${WARMUP_EPOCHS}"
    --drop_path "${DROP_PATH}" --input_size "${INPUT_SIZE}"
    --dual_view true --view_mode paired --teacher_mode false
    --num_workers "${NUM_WORKERS}" --seed "${SEED}" --device cuda
    --deterministic true --reseed_before_training true
    --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}"
    --classes_file "${CLASSES_FILE}" --num_classes "${NUM_CLASSES}"
    --fpn_out_channels 256 --gspf_lambda_consistency 0.0 --gspf_lambda_ortho 0.0
    --head_type c5 --fuse_mode add
    --base_loss bce --use_semantic_branch false
    --summary_csv "${SUMMARY_CSV}" --output_dir "${OUT_DIR}"
    --return_intermediate true --use_p9_caprs true
    --plain_innovation_levels C4 C5
    --plain_innovation_projection_dim 64 --plain_innovation_topk 8
    --plain_innovation_temperature 0.2 --plain_innovation_dropout 0.1
    --plain_innovation_gamma_init 0.005 --plain_innovation_gamma_max 0.05
    --plain_innovation_base_floor 0.0
    --plain_innovation_use_counterfactual_experts true
    --plain_innovation_use_learned_router false
    --plain_innovation_warmup_epochs 15 --plain_innovation_ramp_epochs 10
    --plain_innovation_aux_weight 0.03 --plain_innovation_route_weight 0.0
    --plain_innovation_guard_weight 0.10 --plain_innovation_single_weight 0.02
  )
  if [[ "${RESUME_PARTIAL,,}" == "true" && -f "${OUT_DIR}/checkpoint_last.pth" ]]; then
    info "found checkpoint_last.pth, resuming training"
    train_cmd+=(--resume "${OUT_DIR}")
  fi
  if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
    read -r -a __extra <<< "${EXTRA_TRAIN_ARGS}"
    train_cmd+=("${__extra[@]}")
  fi

  if is_dry; then
    printf '[DRY RUN TRAIN] '
    printf '%q ' "${train_cmd[@]}"
    printf '\n'
    return 0
  fi

  log "training output directory: ${OUT_DIR}"
  ( cd "${CODE_DIR}" && "${train_cmd[@]}" ) 2>&1 | tee -a "${MASTER_LOG}"
  [[ -f "${CHECKPOINT}" ]] || die "training finished but no best checkpoint was found: ${CHECKPOINT}"
  log "training finished, best checkpoint: ${CHECKPOINT}"
}

# ---------------------------------------------------------------------------
# 8. eval: validate the checkpoint protocol, then evaluate the test set exactly once
# ---------------------------------------------------------------------------
cmd_eval() {
  log "locked test evaluation (once per configuration)"
  ensure_annotations

  if ! is_dry; then
    require_file "${CHECKPOINT}"
    if [[ -f "${TEST_LOCK}" && "${FORCE_RETEST,,}" != "true" ]]; then
      die "this checkpoint has already been evaluated on the test set (${TEST_LOCK}).
  Re-evaluating would violate the pre-declared protocol; set FORCE_RETEST=true if you must re-run it."
    fi
  fi

  local eval_dir="${OUT_DIR}"
  local json_out="${eval_dir}/test_metrics.json"
  local csv_out="${eval_dir}/test_metrics.csv"

  local eval_cmd=(
    "${PYTHON_BIN}" tools/evaluate_project_checkpoint.py
    --checkpoint "${CHECKPOINT}"
    --list "${TEST_LIST}"
    --classes-file "${CLASSES_FILE}"
    --view-mode paired
    --output-json "${json_out}"
    --output-csv "${csv_out}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --device cuda
  )

  if is_dry; then
    printf '[DRY RUN VERIFY] '
    printf '%q ' "${PYTHON_BIN}" tools/verify_checkpoint_protocol.py \
      --checkpoint "${CHECKPOINT}" --expected-val-list "${VAL_LIST}"
    printf '\n'
    printf '[DRY RUN EVAL] '
    printf '%q ' "${eval_cmd[@]}"
    printf '\n'
    info "[DRY RUN] skipping the test-evaluation lock file"
    return 0
  fi

  ( cd "${CODE_DIR}" && "${PYTHON_BIN}" tools/verify_checkpoint_protocol.py \
      --checkpoint "${CHECKPOINT}" \
      --expected-val-list "${VAL_LIST}" ) 2>&1 | tee -a "${MASTER_LOG}"

  ( cd "${CODE_DIR}" && "${eval_cmd[@]}" ) 2>&1 | tee -a "${MASTER_LOG}"

  {
    echo "evaluated_at=$(date '+%F %T')"
    echo "checkpoint=${CHECKPOINT}"
    echo "checkpoint_sha256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
    record_hash "test_list" "${TEST_LIST}"
    echo "output_json=${json_out}"
  } > "${TEST_LOCK}"
  log "test evaluation finished: ${json_out} / ${csv_out}"
}

# ---------------------------------------------------------------------------
# 9. relink: rewrite the original machine's absolute path prefix in the annotation lists
# ---------------------------------------------------------------------------
cmd_relink() {
  local from="${RELINK_FROM:-}" to="${RELINK_TO:-}"
  if [[ -z "${from}" || -z "${to}" ]]; then
    die "relink needs both the original prefix and the local path
  e.g. RELINK_FROM=/home/hfuu/桌面/convnextv2/data RELINK_TO=/data/DvXray bash main.sh relink"
  fi
  log "rewriting annotation path prefix: ${from} -> ${to}"
  if is_dry; then
    info "[DRY RUN] would rewrite the prefix in the .txt lists under ${ANNOT_DIR}"
    return 0
  fi
  [[ -d "${ANNOT_DIR}" ]] || die "annotation directory does not exist; run 'bash main.sh setup' first"

  RELINK_ANNOT_DIR="${ANNOT_DIR}" RELINK_FROM="${from}" RELINK_TO="${to}" \
    "${PYTHON_BIN}" - <<'PY'
import os, pathlib

root = pathlib.Path(os.environ["RELINK_ANNOT_DIR"])
src = os.environ["RELINK_FROM"]
dst = os.environ["RELINK_TO"]
targets = sorted([*root.glob("*.txt"), *root.glob("ldxray/*.txt")])
if not targets:
    raise SystemExit(f"no .txt split list found under {root}")

total = 0
for path in targets:
    text = path.read_text(encoding="utf-8")
    hits = text.count(src)
    if hits == 0:
        print(f"  - {path.name}: nothing to replace")
        continue
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(src, dst), encoding="utf-8")
    total += hits
    print(f"  - {path.name}: {hits} replacement(s) (original backed up as {backup.name})")
print(f"  {total} path prefix occurrence(s) rewritten")
PY
  info "run 'bash main.sh preflight' again to confirm the new paths"
}

# ---------------------------------------------------------------------------
# 10. status: show artefacts and metrics
# ---------------------------------------------------------------------------
cmd_status() {
  log "run status"
  info "REPO_DIR     = ${REPO_DIR}"
  info "CODE_DIR     = ${CODE_DIR}"
  info "OUT_DIR      = ${OUT_DIR}"
  info "MASTER_LOG   = ${MASTER_LOG}"
  for f in "${CHECKPOINT}" "${SUMMARY_CSV}" "${PROTOCOL_FILE}" "${OUT_DIR}/test_metrics.json" "${TEST_LOCK}"; do
    if [[ -s "${f}" ]]; then info "[present] ${f}"; else info "[missing] ${f}"; fi
  done
  if [[ -s "${OUT_DIR}/test_metrics.json" ]]; then
    printf '\nTest metrics (test_metrics.json):\n'
    cat "${OUT_DIR}/test_metrics.json"
    printf '\n'
  fi
  if [[ -s "${SUMMARY_CSV}" ]]; then
    printf '\nValidation selection detail (tail of training_results.csv):\n'
    tail -n 5 "${SUMMARY_CSV}"
  fi
}

# ---------------------------------------------------------------------------
# 11. help and dispatch
# ---------------------------------------------------------------------------
cmd_help() {
  cat <<'EOF'
UniformFusion — one-stop entry script (main.sh)

Usage: bash main.sh [subcommand]

Subcommands:
  all         setup -> preflight -> train -> eval (default)
  setup       expand data_splits/ into core/annotations/ (DvXray + annotations/ldxray/)
  relink      rewrite the original machine's absolute path prefix in the annotation lists
              (requires RELINK_FROM=<old prefix> RELINK_TO=<local path>)
  preflight   interpreter/dependency/CUDA probe + locked-source and split SHA256 checks
              + dataset accessibility check
  train       train the Uniform Fusion main method (select the best checkpoint on
              validation, never evaluate the test set)
  eval        evaluate the test set exactly once (refused if already evaluated,
              unless FORCE_RETEST=true)
  status      show checkpoints, metrics and protocol files
  help        show this help

Examples:
  DRY_RUN=true bash main.sh                 # only print the commands
  GPU_ID=1 SEED=553800223 bash main.sh all  # different GPU and seed
  RUN_ID=my_repro bash main.sh all          # custom run identifier
  RELINK_FROM=/old/data RELINK_TO=/new/data bash main.sh relink

Key environment variables:
  PYTHON_BIN (default /home/hfuu/miniforge3/envs/v2b384_env/bin/python)
  GPU_ID, SEED, BATCH_SIZE, EPOCHS, PATIENCE, NUM_WORKERS
  RUN_ID, OUT_ROOT, OUT_DIR, SUMMARY_CSV, MASTER_LOG, CHECKPOINT
  TRAIN_LIST, VAL_LIST, TEST_LIST, CLASSES_FILE
  SKIP_CHECKSUM=true, SKIP_DATA_CHECK=true, FORCE_RETEST=true, FORCE_COPY=true,
  RESUME_PARTIAL=true, EXTRA_TRAIN_ARGS="--flag value", DRY_RUN=true

Protocol reminder: the paper requires "select on validation -> evaluate the test set
only once"; do not repeatedly evaluate the test set for the same configuration.
EOF
}

cmd_all() {
  cmd_setup
  cmd_preflight
  cmd_train
  cmd_eval
  cmd_status
}

main() {
  # global environment required for deterministic training (same as the scripts in experiments/)
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
  export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

  local command="${1:-all}"
  [[ $# -gt 0 ]] && shift

  case "${command}" in
    all)              cmd_all ;;
    setup)            cmd_setup ;;
    relink)           cmd_relink ;;
    preflight|check)  cmd_preflight ;;
    train)            cmd_preflight; cmd_train ;;
    eval|test)        cmd_preflight; cmd_eval ;;
    status)           cmd_status ;;
    help|-h|--help)   cmd_help ;;
    *) printf 'ERROR: unknown subcommand %s\n\n' "${command}" >&2; cmd_help >&2; exit 2 ;;
  esac
}

main "$@"
