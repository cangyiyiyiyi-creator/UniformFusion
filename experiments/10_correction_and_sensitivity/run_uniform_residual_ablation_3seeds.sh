#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PY="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT/论文补充验证_20260907/05_残差机制对照/run_20260907_residual_ablation}"
SEEDS=(930163947 1786430941 553800223); METHODS=(UF_NoCorrection UF_NoRamp)
mkdir -p "$SAVE_ROOT"
printf 'methods=%s\nselection=all_Val_then_locked_Test\n' "${METHODS[*]}" > "$SAVE_ROOT/protocol.txt"
for phase in val test; do
  [[ "$phase" == test ]] && touch "$SAVE_ROOT/validation_phase_complete.marker"
  for method in "${METHODS[@]}"; do
    repeat=0
    for seed in "${SEEDS[@]}"; do
      repeat=$((repeat+1))
      SAVE_ROOT="$SAVE_ROOT" METHOD="$method" SEED="$seed" REPEAT="$repeat" PHASE="$phase" bash run_reviewer_first_batch_one.sh
    done
  done
done
"$PY" - "$SAVE_ROOT" <<'PY'
import csv,json,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]); rows=[]
for p in sorted(r.rglob('test_metrics.json')):
 t=json.loads(p.read_text());v=json.loads((p.parent/'val_metrics.json').read_text());rows.append({'method':p.parent.name,'seed':int(p.parents[2].name.split('_')[-1]),'best_epoch':v['checkpoint_epoch'],'val_mAP':v['stats']['mAP'],'test_mAP':t['stats']['mAP']})
if len(rows)!=6:raise RuntimeError(f'expected 6 results, got {len(rows)}')
with (r/'residual_ablation_per_seed.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
s=[]
for m in sorted({x['method'] for x in rows}):
 a=[x['test_mAP'] for x in rows if x['method']==m];s.append({'method':m,'n':3,'mean':statistics.mean(a),'std':statistics.stdev(a)})
with (r/'residual_ablation_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=s[0]);w.writeheader();w.writerows(s)
PY
touch "$SAVE_ROOT/suite_complete.marker"
