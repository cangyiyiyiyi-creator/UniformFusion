#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PY="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT/supplementary_verification_20260907/02B_dagnet_batch32_recheck/run_20260907_dagnet_batch32_3seeds}"
SEEDS=(930163947 1786430941 553800223); mkdir -p "$SAVE_ROOT"
printf 'method=DAGNet_OfficialArchitecture\nphysical_batch_size=16\ngradient_accumulation=2\neffective_batch_size=32\ninput_size=256\nselection=all_Val_then_locked_Test\nreplaces_batch16_only_if_complete=true\n' > "$SAVE_ROOT/protocol.txt"
for phase in val test; do
 [[ "$phase" == test ]] && touch "$SAVE_ROOT/validation_phase_complete.marker"
 repeat=0
 for seed in "${SEEDS[@]}"; do repeat=$((repeat+1)); DAGNET_BATCH_SIZE=16 ACCUM_ITER=2 SAVE_ROOT="$SAVE_ROOT" METHOD=DAGNet_OfficialArchitecture SEED="$seed" REPEAT="$repeat" PHASE="$phase" bash run_fair_adapter_one.sh; done
done
"$PY" - "$SAVE_ROOT" <<'PY'
import csv,json,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]);rows=[]
for p in sorted(r.rglob('test_metrics.json')):
 t=json.loads(p.read_text());v=json.loads((p.parent/'val_metrics.json').read_text());rows.append({'seed':int(p.parents[2].name.split('_')[-1]),'best_epoch':v['checkpoint_epoch'],'val_mAP':v['stats']['mAP'],'test_mAP':t['stats']['mAP']})
if len(rows)!=3:raise RuntimeError(f'expected 3 results, got {len(rows)}')
with (r/'DAGNet_batch32_per_seed.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
a=[x['test_mAP'] for x in rows];(r/'DAGNet_batch32_summary.txt').write_text(f'n=3\ntest_mAP={statistics.mean(a):.6f} ± {statistics.stdev(a):.6f}\n')
PY
touch "$SAVE_ROOT/suite_complete.marker"
