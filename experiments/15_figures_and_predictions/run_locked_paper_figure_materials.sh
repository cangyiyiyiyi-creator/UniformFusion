#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/tools${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ARCHIVE="${ROOT_DIR}/论文最终归档_20260830"
OUT="${ARCHIVE}/08_真实可视化与PR曲线_20260902"
PRED="${OUT}/01_样本级预测"
mkdir -p "${PRED}" "${OUT}/05_生成代码"

if pgrep -f '[m]ain_finetune.py' >/dev/null; then
  echo "Another training process is active; refusing concurrent figure inference."
  exit 1
fi

"${PYTHON_BIN}" -m py_compile \
  tools/locked_checkpoint_utils.py \
  tools/export_locked_predictions.py \
  tools/build_pr_and_case_materials.py \
  tools/export_uniform_region_heatmaps.py \
  tools/finalize_figure_material_archive.py

export_one() {
  local dataset="$1" method="$2" seed="$3" checkpoint_dir="$4" list_file="$5" classes_file="$6"
  local output_dir="${PRED}/${dataset}/${method}/seed_${seed}"
  if [[ -s "${output_dir}/predictions.npz" && -s "${output_dir}/prediction_manifest.json" && -s "${output_dir}/sample_manifest.csv" ]]; then
    echo "SKIP existing prediction export: ${dataset}/${method}/seed_${seed}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u tools/export_locked_predictions.py \
    --checkpoint "${checkpoint_dir}/checkpoint_best.pth" \
    --expected-metrics-json "${checkpoint_dir}/test_metrics.json" \
    --list "${list_file}" --classes-file "${classes_file}" \
    --output-dir "${output_dir}" --dataset-name "${dataset}" \
    --method-name "${method}" --seed "${seed}" \
    --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --device cuda
}

DV_SEEDS=(930163947 1786430941 553800223 207027553 1716854429)
for seed in "${DV_SEEDS[@]}"; do
  export_one DvXray Plain_BCE "${seed}" \
    "${ARCHIVE}/03_基线与消融模型/Plain_BCE__ResNet50__seed_${seed}" \
    annotations/DvXray_test.txt annotations/classes.txt
  export_one DvXray Uniform_Fusion "${seed}" \
    "${ARCHIVE}/02_UniformFusion主方法模型/UniformFusion__ResNet50__seed_${seed}" \
    annotations/DvXray_test.txt annotations/classes.txt
done

LDX_SEEDS=(930163947 1786430941 553800223)
for seed in "${LDX_SEEDS[@]}"; do
  export_one LDXray Plain_BCE "${seed}" \
    "${ARCHIVE}/07_LDXray跨数据集泛化_20260902/03_最佳模型/LDXray_Plain_BCE__ResNet50__seed_${seed}" \
    annotations/ldxray/LDXray_test.txt annotations/ldxray/ldxray_classes.txt
  export_one LDXray Uniform_Fusion "${seed}" \
    "${ARCHIVE}/07_LDXray跨数据集泛化_20260902/03_最佳模型/LDXray_UniformFusion_StepMatched__ResNet50__seed_${seed}" \
    annotations/ldxray/LDXray_test.txt annotations/ldxray/ldxray_classes.txt
done

if [[ ! -s "${OUT}/03_成功失败案例/DvXray/selection_manifest.csv" || ! -s "${OUT}/03_成功失败案例/LDXray/selection_manifest.csv" ]]; then
  "${PYTHON_BIN}" -u tools/build_pr_and_case_materials.py \
    --prediction-root "${PRED}" --output-root "${OUT}" \
    --representative-seed 930163947
fi

DV_HEAT="${OUT}/04_区域热图/DvXray"
if [[ ! -s "${DV_HEAT}/heatmap_manifest.csv" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u tools/export_uniform_region_heatmaps.py \
    --checkpoint "${ARCHIVE}/02_UniformFusion主方法模型/UniformFusion__ResNet50__seed_930163947/checkpoint_best.pth" \
    --list annotations/DvXray_test.txt --classes-file annotations/classes.txt \
    --selection-manifest "${OUT}/03_成功失败案例/DvXray/selection_manifest.csv" \
    --plain-predictions "${PRED}/DvXray/Plain_BCE/seed_930163947/predictions.npz" \
    --uniform-predictions "${PRED}/DvXray/Uniform_Fusion/seed_930163947/predictions.npz" \
    --output-dir "${DV_HEAT}" --dataset-name DvXray --device cuda
fi

LDX_HEAT="${OUT}/04_区域热图/LDXray"
if [[ ! -s "${LDX_HEAT}/heatmap_manifest.csv" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u tools/export_uniform_region_heatmaps.py \
    --checkpoint "${ARCHIVE}/07_LDXray跨数据集泛化_20260902/03_最佳模型/LDXray_UniformFusion_StepMatched__ResNet50__seed_930163947/checkpoint_best.pth" \
    --list annotations/ldxray/LDXray_test.txt --classes-file annotations/ldxray/ldxray_classes.txt \
    --selection-manifest "${OUT}/03_成功失败案例/LDXray/selection_manifest.csv" \
    --plain-predictions "${PRED}/LDXray/Plain_BCE/seed_930163947/predictions.npz" \
    --uniform-predictions "${PRED}/LDXray/Uniform_Fusion/seed_930163947/predictions.npz" \
    --output-dir "${LDX_HEAT}" --dataset-name LDXray --device cuda
fi

cp -a tools/locked_checkpoint_utils.py \
  tools/export_locked_predictions.py \
  tools/build_pr_and_case_materials.py \
  tools/export_uniform_region_heatmaps.py \
  tools/finalize_figure_material_archive.py \
  run_locked_paper_figure_materials.sh \
  "${OUT}/05_生成代码/"

"${PYTHON_BIN}" -u tools/finalize_figure_material_archive.py --root "${OUT}"
echo "LOCKED_PAPER_FIGURE_MATERIALS_COMPLETE ${OUT}"
