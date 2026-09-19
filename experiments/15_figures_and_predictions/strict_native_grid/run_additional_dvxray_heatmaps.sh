#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
FIG_ROOT="$ROOT/paper_archive_20260830/08_visualisation_and_pr_curves_20260902"
PRED_ROOT="$FIG_ROOT/01_sample_predictions/DvXray"
ORIGINAL_CASES="$FIG_ROOT/03_success_failure_cases/DvXray/selection_manifest.csv"
OUT="$FIG_ROOT/07_dvxray_additional_strict_native_grid_heatmaps_20260910"
SELECTION="$OUT/01_selection_manifests"
EVIDENCE="$OUT/02_raw_region_evidence_export"
EXACT="$OUT/03_strict_native_grid_heatmaps"
SOURCE="$OUT/04_generation_code"
CHECKPOINT="$ROOT/paper_archive_20260830/02_uniform_fusion_main_models/UniformFusion__ResNet50__seed_930163947/checkpoint_best.pth"

if [[ -e "$OUT" ]]; then
  printf 'Refusing to overwrite existing output: %s\n' "$OUT" >&2
  exit 1
fi

"$PYTHON" tools/build_additional_dvxray_heatmap_selection.py \
  --plain-prediction-dir "$PRED_ROOT/Plain_BCE/seed_930163947" \
  --uniform-prediction-dir "$PRED_ROOT/Uniform_Fusion/seed_930163947" \
  --existing-selection-manifest "$ORIGINAL_CASES" \
  --output-dir "$SELECTION" \
  --representative-seed 930163947

"$PYTHON" tools/export_uniform_region_heatmaps.py \
  --checkpoint "$CHECKPOINT" \
  --list annotations/DvXray_test.txt \
  --classes-file annotations/classes.txt \
  --selection-manifest "$SELECTION/selection_manifest.csv" \
  --plain-predictions "$PRED_ROOT/Plain_BCE/seed_930163947/predictions.npz" \
  --uniform-predictions "$PRED_ROOT/Uniform_Fusion/seed_930163947/predictions.npz" \
  --output-dir "$EVIDENCE" \
  --dataset-name DvXray

mkdir -p "$EXACT" "$SOURCE"
while IFS=, read -r case_number case_id rest; do
  [[ "$case_number" == $'\357\273\277case_number' || "$case_number" == "case_number" ]] && continue
  "$PYTHON" tools/render_exact_region_evidence.py \
    --evidence "$EVIDENCE/${case_id}_evidence.npz" \
    --metadata "$EVIDENCE/${case_id}_metadata.json" \
    --output "$EXACT/${case_id}_exact.png"
done < "$EVIDENCE/heatmap_manifest.csv"

cp tools/build_additional_dvxray_heatmap_selection.py "$SOURCE/"
cp tools/export_uniform_region_heatmaps.py "$SOURCE/"
cp tools/render_exact_region_evidence.py "$SOURCE/"
cp run_additional_dvxray_heatmaps.sh "$SOURCE/"

"$PYTHON" -c "from pathlib import Path; import hashlib, json; root=Path(r'$OUT'); files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'); payload={'cases':8,'files_excluding_manifest':len(files),'sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}; (root/'artifact_manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')"

printf 'ADDITIONAL_DVXRAY_HEATMAPS_COMPLETE output=%s\n' "$OUT"
