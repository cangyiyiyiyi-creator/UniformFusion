#!/usr/bin/env bash
# =============================================================================
# Uniform Fusion — 总入口脚本 (main.sh)
#
# 对应稿件：Class-Conditioned Multi-Scale Regional Learning for Dual-View
#          X-Ray Multi-Label Recognition
#
# 本脚本把「准备标注 → 环境自检 → 训练主方法 → 锁定 Test 评估一次」
# 串成一条可复现链路，内置论文 4.1.3 节协议要求：
#   * 训练集训练 → Val 选最佳 checkpoint → 锁定后 Test 只评估一次
#   * 固定随机种子、确定性算法（CUBLAS_WORKSPACE_CONFIG=:4096:8）
#   * 锁定源码与划分清单的 SHA256 校验
#
# 用法：
#   bash main.sh                 # 等价于 bash main.sh all
#   bash main.sh help            # 查看全部子命令
#   bash main.sh setup           # 仅把 data_splits/ 展开到 core/annotations/
#   bash main.sh preflight       # 仅做环境与校验自检
#   bash main.sh train           # 仅训练（Val 选点）
#   bash main.sh eval            # 仅锁定 Test 评估（需已有 checkpoint_best.pth）
#   bash main.sh status          # 查看本次运行产物与指标
#
# 常用环境变量（全部可覆盖）：
#   PYTHON_BIN=...   GPU_ID=0        SEED=930163947
#   BATCH_SIZE=32    EPOCHS=180      PATIENCE=25      NUM_WORKERS=8
#   RUN_ID=...       OUT_ROOT=...    SUMMARY_CSV=...  MASTER_LOG=...
#   SKIP_CHECKSUM=true   FORCE_RETEST=true   DRY_RUN=true
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. 路径
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-${REPO_DIR}/core}"
ANNOT_DIR="${ANNOT_DIR:-${CODE_DIR}/annotations}"
SPLITS_DIR="${SPLITS_DIR:-${REPO_DIR}/data_splits}"
TOOLS_DIR="${CODE_DIR}/tools"

# ---------------------------------------------------------------------------
# 2. 可覆盖运行配置（默认值与 experiments/ 下锁定脚本一致）
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
# 3. 论文锁定校验值（见 README 第 4 节 / protocol_uniformfusion_n5.txt）
# ---------------------------------------------------------------------------
SHA_MAIN_FINETUNE="d1ec51b1e44152db52def599d2185d02318786de2db1177bff3158b26ce1782d"
SHA_CONVNEXTV2_DUAL="f39a507d877dd8350fe862fdfc013032f7c872e6ca869371f9bbdb64fa81abcb"
SHA_ENGINE_FINETUNE="dbe37f44aee890644a3016012b1b240a18ddd933e09fadb18d0ea38e7c1140ff"
SHA_TRAIN_SPLIT="f0a5c6f810a5725e3336b28df184542343f99e4d9afd5c866e860b4052254dcf"
SHA_VAL_SPLIT="a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67"
SHA_TEST_SPLIT="6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6"

# ---------------------------------------------------------------------------
# 4. 小工具函数
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
info() { printf '  - %s\n' "$*"; }
warn() { printf '  ! %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

is_dry() { [[ "${DRY_RUN,,}" == "true" ]]; }

# 打印或执行命令（DRY_RUN=true 时只打印）
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
  [[ -s "$1" ]] || die "缺少文件：$1"
}

# 相对路径按「代码根目录 = core/」解析（脚本内的相对路径都是相对 core/ 的）
resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *)  printf '%s\n' "${CODE_DIR}/$1" ;;
  esac
}

# 记录某个列表文件的路径与 SHA256（不存在时写 NA）
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
    die "${label} 校验失败：${file}
  期望 ${expected}
  实际 ${actual}
  （确认未改动锁定源码/划分清单；确需跳过请设置 SKIP_CHECKSUM=true）"
  fi
  info "${label} SHA256 OK  ${file}"
}

# 复制文件到目标目录；已存在则跳过（FORCE_COPY=true 时强制覆盖）
copy_into() {
  local dest_dir="$1"; shift
  local src base
  for src in "$@"; do
    base="$(basename "${src}")"
    if [[ ! -e "${src}" ]]; then
      die "缺少源文件：${src}"
    fi
    if [[ "${FORCE_COPY,,}" == "true" || ! -e "${dest_dir}/${base}" ]]; then
      run_cmd cp -f "${src}" "${dest_dir}/"
    else
      info "已存在，跳过：${base}"
    fi
  done
}

# ---------------------------------------------------------------------------
# 5. setup：把 data_splits/ 展开为脚本期望的 core/annotations/ 布局
# ---------------------------------------------------------------------------
cmd_setup() {
  log "准备标注目录：${ANNOT_DIR}"
  run_cmd mkdir -p "${ANNOT_DIR}/ldxray"
  # DvXray：classes.txt + DvXray_{train,val,test}.txt（主实验使用）
  copy_into "${ANNOT_DIR}" \
    "${SPLITS_DIR}/DvXray/classes.txt" \
    "${SPLITS_DIR}/DvXray/DvXray_train.txt" \
    "${SPLITS_DIR}/DvXray/DvXray_val.txt" \
    "${SPLITS_DIR}/DvXray/DvXray_test.txt"
  # LDXray：供 02_LDXray_cross_dataset 使用（annotations/ldxray/）
  copy_into "${ANNOT_DIR}/ldxray" \
    "${SPLITS_DIR}/LDXray/LDXray_train.txt" \
    "${SPLITS_DIR}/LDXray/LDXray_val.txt" \
    "${SPLITS_DIR}/LDXray/LDXray_test.txt" \
    "${SPLITS_DIR}/LDXray/ldxray_classes.txt" \
    "${SPLITS_DIR}/LDXray/LDXray_split_manifest.json"

  if ! is_dry; then
    info "DvXray 划分：$(wc -l < "${ANNOT_DIR}/DvXray_train.txt") / $(wc -l < "${ANNOT_DIR}/DvXray_val.txt") / $(wc -l < "${ANNOT_DIR}/DvXray_test.txt")  (train/val/test)"
  fi
}

ensure_annotations() {
  if is_dry; then
    return 0   # 干跑不落盘，避免重复打印 setup 步骤
  fi
  if [[ ! -s "${ANNOT_DIR}/classes.txt" || ! -s "${ANNOT_DIR}/DvXray_train.txt" ]]; then
    cmd_setup
  fi
}

# ---------------------------------------------------------------------------
# 6. preflight：解释器、依赖、CUDA、锁定文件与划分校验
# ---------------------------------------------------------------------------
cmd_preflight() {
  log "环境与协议自检"
  ensure_annotations

  [[ -x "${PYTHON_BIN}" ]] || die "找不到 Python 解释器：${PYTHON_BIN}
  请设置 PYTHON_BIN（如 PYTHON_BIN=\$(which python)）"
  info "PYTHON_BIN = ${PYTHON_BIN}"

  if is_dry; then
    info "[DRY RUN] 跳过依赖与 CUDA 探测"
  else
    "${PYTHON_BIN}" - <<'PY'
import importlib, sys
print(f"  - python   : {sys.version.split()[0]}")
for mod in ("torch", "torchvision", "timm", "numpy", "pandas"):
    try:
        m = importlib.import_module(mod)
        print(f"  - {mod:<10}: {getattr(m, '__version__', 'unknown')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  - {mod:<10}: 缺失 ({exc})")
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
    warn "SKIP_CHECKSUM=true：跳过所有 SHA256 校验（结果不具协议效力）"
  else
    verify_sha256 "${CODE_DIR}/${MAIN_FINETUNE_SCRIPT}" "${SHA_MAIN_FINETUNE}" "锁定源码 main_finetune.py"
    verify_sha256 "${CODE_DIR}/models/convnextv2_dual.py" "${SHA_CONVNEXTV2_DUAL}" "锁定源码 convnextv2_dual.py"
    verify_sha256 "${CODE_DIR}/engine_finetune.py" "${SHA_ENGINE_FINETUNE}" "锁定源码 engine_finetune.py"
    verify_sha256 "${SPLITS_DIR}/DvXray/DvXray_train.txt" "${SHA_TRAIN_SPLIT}" "DvXray train 划分"
    verify_sha256 "${SPLITS_DIR}/DvXray/DvXray_val.txt" "${SHA_VAL_SPLIT}" "DvXray val 划分"
    verify_sha256 "${SPLITS_DIR}/DvXray/DvXray_test.txt" "${SHA_TEST_SPLIT}" "DvXray test 划分"
  fi

  if is_dry; then
    info "[DRY RUN] 跳过语法检查"
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
        bad.append(f"{name}: 文件不存在")
        continue
    try:
        compile(path.read_text(encoding="utf-8"), name, "exec")
    except SyntaxError as exc:  # noqa: PERF203
        bad.append(f"{name}: {exc}")
if bad:
    print("\n".join(bad))
    sys.exit(1)
print(f"  - 语法检查通过（{len(files)} 个关键文件）")
PY
  )

  local running
  running="$(pgrep -fc '[m]ain_finetune.py' || true)"
  if [[ "${running:-0}" -gt 0 ]]; then
    warn "检测到 ${running} 个 main_finetune.py 进程正在运行；此时开新训练可能显存不足"
  fi

  check_dataset_access
  log "自检通过"
}

# 抽样检查标注列表里的图像在本机是否可访问（划分文件保存的是原始机器绝对路径）
check_dataset_access() {
  [[ "${SKIP_DATA_CHECK,,}" == "true" ]] && { warn "SKIP_DATA_CHECK=true：跳过数据集可访问性检查"; return 0; }
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
        problems.append(f"{entry}：列表文件不存在")
        continue
    lines = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()][:20]
    missing = 0
    for toks in lines:
        for img in toks[:2]:
            if not resolve(img).is_file():
                missing += 1
    if missing:
        problems.append(f"{path.name}：抽样 {len(lines)} 行中有 {missing} 个图像文件不存在")
    else:
        print(f"  - {path.name}：抽样图像可访问")

if problems:
    print("  ! 数据集路径检查未通过：")
    for item in problems:
        print(f"    - {item}")
    print("    → 划分文件保存的是原始机器的绝对路径，请先指向本机数据根：")
    print("      RELINK_FROM=<原前缀> RELINK_TO=<本机路径> bash main.sh relink")
    print("      （确认路径无误后可用 SKIP_DATA_CHECK=true 跳过本检查）")
    sys.exit(1)
PY
}

# ---------------------------------------------------------------------------
# 7. train：Uniform Fusion 主方法（Val 选点，不触碰 Test）
# ---------------------------------------------------------------------------
cmd_train() {
  log "训练 Uniform Fusion（论文主方法，内部配置名 Final_NoAnchor_NoRouter）"
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
    info "发现 checkpoint_last.pth，续训"
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

  log "训练输出目录：${OUT_DIR}"
  ( cd "${CODE_DIR}" && "${train_cmd[@]}" ) 2>&1 | tee -a "${MASTER_LOG}"
  [[ -f "${CHECKPOINT}" ]] || die "训练结束但未找到最佳 checkpoint：${CHECKPOINT}"
  log "训练完成，最佳 checkpoint：${CHECKPOINT}"
}

# ---------------------------------------------------------------------------
# 8. eval：Val 选点校验 + Test 锁定评估一次
# ---------------------------------------------------------------------------
cmd_eval() {
  log "锁定 Test 评估（每配置仅执行一次）"
  ensure_annotations

  if ! is_dry; then
    require_file "${CHECKPOINT}"
    if [[ -f "${TEST_LOCK}" && "${FORCE_RETEST,,}" != "true" ]]; then
      die "该 checkpoint 已评估过 Test（${TEST_LOCK}）。
  为遵守预先声明的协议，不允许重复评估；确需重跑请设置 FORCE_RETEST=true"
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
    info "[DRY RUN] 跳过 Test 锁定标记写入"
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
  log "Test 评估完成：${json_out} / ${csv_out}"
}

# ---------------------------------------------------------------------------
# 9. relink：把标注列表中的原始机器绝对路径前缀替换为本机数据根
# ---------------------------------------------------------------------------
cmd_relink() {
  local from="${RELINK_FROM:-}" to="${RELINK_TO:-}"
  if [[ -z "${from}" || -z "${to}" ]]; then
    die "relink 需要同时设置原前缀与本机路径
  例：RELINK_FROM=/home/hfuu/桌面/convnextv2/data RELINK_TO=/data/DvXray bash main.sh relink"
  fi
  log "重写标注路径前缀：${from} -> ${to}"
  if is_dry; then
    info "[DRY RUN] 将对 ${ANNOT_DIR} 下的 .txt 列表做前缀替换"
    return 0
  fi
  [[ -d "${ANNOT_DIR}" ]] || die "标注目录不存在，请先运行 bash main.sh setup"

  RELINK_ANNOT_DIR="${ANNOT_DIR}" RELINK_FROM="${from}" RELINK_TO="${to}" \
    "${PYTHON_BIN}" - <<'PY'
import os, pathlib

root = pathlib.Path(os.environ["RELINK_ANNOT_DIR"])
src = os.environ["RELINK_FROM"]
dst = os.environ["RELINK_TO"]
targets = sorted([*root.glob("*.txt"), *root.glob("ldxray/*.txt")])
if not targets:
    raise SystemExit(f"在 {root} 下没有找到任何 .txt 划分列表")

total = 0
for path in targets:
    text = path.read_text(encoding="utf-8")
    hits = text.count(src)
    if hits == 0:
        print(f"  - {path.name}：无需替换")
        continue
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(src, dst), encoding="utf-8")
    total += hits
    print(f"  - {path.name}：替换 {hits} 处（原文件备份为 {backup.name}）")
print(f"  共替换 {total} 处路径前缀")
PY
  info "确认替换结果后，再运行 bash main.sh preflight 复核"
}

# ---------------------------------------------------------------------------
# 10. status：查看产物与指标
# ---------------------------------------------------------------------------
cmd_status() {
  log "运行状态"
  info "REPO_DIR     = ${REPO_DIR}"
  info "CODE_DIR     = ${CODE_DIR}"
  info "OUT_DIR      = ${OUT_DIR}"
  info "MASTER_LOG   = ${MASTER_LOG}"
  for f in "${CHECKPOINT}" "${SUMMARY_CSV}" "${PROTOCOL_FILE}" "${OUT_DIR}/test_metrics.json" "${TEST_LOCK}"; do
    if [[ -s "${f}" ]]; then info "[有] ${f}"; else info "[无] ${f}"; fi
  done
  if [[ -s "${OUT_DIR}/test_metrics.json" ]]; then
    printf '\nTest 指标（test_metrics.json）：\n'
    cat "${OUT_DIR}/test_metrics.json"
    printf '\n'
  fi
  if [[ -s "${SUMMARY_CSV}" ]]; then
    printf '\nVal 选点明细（training_results.csv 末尾）：\n'
    tail -n 5 "${SUMMARY_CSV}"
  fi
}

# ---------------------------------------------------------------------------
# 11. help 与分发
# ---------------------------------------------------------------------------
cmd_help() {
  cat <<'EOF'
Uniform Fusion 总入口 (main.sh)

用法： bash main.sh [子命令]

子命令：
  all         setup → preflight → train → eval（默认）
  setup       把 data_splits/ 展开到 core/annotations/（DvXray + annotations/ldxray/）
  relink      把标注列表里原始机器的绝对路径前缀换成本机路径
              （需同时设置 RELINK_FROM=<原前缀> RELINK_TO=<本机路径>）
  preflight   解释器/依赖/CUDA 探测 + 锁定源码与划分 SHA256 校验 + 数据集可访问性检查
  train       Uniform Fusion 主方法训练（Val 选最佳 checkpoint，不评估 Test）
  eval        Test 锁定评估一次（已评估过则拒绝重跑，除非 FORCE_RETEST=true）
  status      查看 checkpoint、指标与协议文件
  help        显示本帮助

示例：
  DRY_RUN=true bash main.sh                 # 只打印将执行的命令
  GPU_ID=1 SEED=553800223 bash main.sh all  # 换卡换种子跑一遍
  RUN_ID=my_repro bash main.sh all          # 指定运行标识
  RELINK_FROM=/old/data RELINK_TO=/new/data bash main.sh relink

关键环境变量：
  PYTHON_BIN (默认 /home/hfuu/miniforge3/envs/v2b384_env/bin/python)
  GPU_ID, SEED, BATCH_SIZE, EPOCHS, PATIENCE, NUM_WORKERS
  RUN_ID, OUT_ROOT, OUT_DIR, SUMMARY_CSV, MASTER_LOG, CHECKPOINT
  TRAIN_LIST, VAL_LIST, TEST_LIST, CLASSES_FILE
  SKIP_CHECKSUM=true, SKIP_DATA_CHECK=true, FORCE_RETEST=true, FORCE_COPY=true,
  RESUME_PARTIAL=true, EXTRA_TRAIN_ARGS="--flag value", DRY_RUN=true

协议提醒：论文要求「Val 选点 → Test 只评估一次」，请勿在同一配置上反复评估 Test。
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
  # 确定性训练所需的全局环境（与 experiments/ 下脚本一致）
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
    *) printf 'ERROR: 未知子命令 %s\n\n' "${command}" >&2; cmd_help >&2; exit 2 ;;
  esac
}

main "$@"
