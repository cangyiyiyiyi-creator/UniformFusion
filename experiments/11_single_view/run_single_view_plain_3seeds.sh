#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
PY="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT/论文补充验证_20260907/04_独立单视角训练/run_20260907_single_view_plain}"
SEEDS=(930163947 1786430941 553800223); METHODS=(Plain_BCE_OL_Only Plain_BCE_SD_Only)
mkdir -p "$SAVE_ROOT"
printf 'methods=%s\nmeaning=independent_training_with_selected_view_replicated_to_shared_dual_branches\nselection=all_Val_then_locked_Test\n' "${METHODS[*]}" > "$SAVE_ROOT/protocol.txt"
run_one() {
 local method=$1 seed=$2 repeat=$3 phase=$4 mode
 [[ "$method" == Plain_BCE_OL_Only ]] && mode=a_only || mode=b_only
 local out="$SAVE_ROOT/resnet50/seed_${seed}/repeat_${repeat}/$method" ckpt="$SAVE_ROOT/resnet50/seed_${seed}/repeat_${repeat}/$method/checkpoint_best.pth"
 mkdir -p "$out"
 local common=(--model resnet50 --model_prefix "" --input_size 224 --batch_size 32 --epochs 180 --lr 1e-4 --weight_decay 0.05 --warmup_epochs 5 --drop_path 0.2 --patience 25 --dual_view true --view_mode "$mode" --teacher_mode false --train_list annotations/DvXray_train.txt --val_list annotations/DvXray_val.txt --classes_file annotations/classes.txt --num_classes 15 --base_loss bce --fuse_mode add --head_type c5 --return_intermediate false --use_semantic_branch false --use_p9_caprs false --num_workers 8 --seed "$seed" --device cuda --deterministic true --reseed_before_training true --aug_mode conditional --summary_csv "$SAVE_ROOT/results.csv" --output_dir "$out")
 if [[ "$phase" == val ]]; then
  if [[ ! -f "$out/training_complete.marker" ]]; then local resume=(); [[ -f "$out/checkpoint_last.pth" ]] && resume=(--resume "$out"); CUDA_VISIBLE_DEVICES=0 "$PY" -u main_finetune.py "${common[@]}" "${resume[@]}" 2>&1 | tee "$out/train.log"; [[ -f "$ckpt" ]]; touch "$out/training_complete.marker"; fi
  local list=annotations/DvXray_val.txt
 else
  [[ -f "$SAVE_ROOT/validation_phase_complete.marker" ]]; local list=annotations/DvXray_test.txt
 fi
 "$PY" tools/verify_checkpoint_protocol.py --checkpoint "$ckpt" --expected-val-list annotations/DvXray_val.txt
 if [[ ! -s "$out/${phase}_metrics.json" ]]; then CUDA_VISIBLE_DEVICES=0 "$PY" tools/evaluate_project_checkpoint.py --checkpoint "$ckpt" --list "$list" --classes-file annotations/classes.txt --view-mode checkpoint --output-json "$out/${phase}_metrics.json" --output-csv "$out/${phase}_metrics.csv" --batch-size 32 --num-workers 8 --device cuda; fi
}
for phase in val test; do
 [[ "$phase" == test ]] && touch "$SAVE_ROOT/validation_phase_complete.marker"
 for method in "${METHODS[@]}"; do repeat=0; for seed in "${SEEDS[@]}"; do repeat=$((repeat+1)); run_one "$method" "$seed" "$repeat" "$phase"; done; done
done
"$PY" - "$SAVE_ROOT" <<'PY'
import csv,json,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]);rows=[]
for p in sorted(r.rglob('test_metrics.json')):
 t=json.loads(p.read_text());v=json.loads((p.parent/'val_metrics.json').read_text());rows.append({'method':p.parent.name,'seed':int(p.parents[2].name.split('_')[-1]),'val_mAP':v['stats']['mAP'],'test_mAP':t['stats']['mAP']})
if len(rows)!=6:raise RuntimeError(f'expected 6 results, got {len(rows)}')
with (r/'独立单视角_逐seed.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
s=[]
for m in sorted({x['method'] for x in rows}):
 a=[x['test_mAP'] for x in rows if x['method']==m];s.append({'method':m,'n':3,'mean':statistics.mean(a),'std':statistics.stdev(a)})
with (r/'独立单视角_汇总.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=s[0]);w.writeheader();w.writerows(s)
PY
touch "$SAVE_ROOT/suite_complete.marker"
