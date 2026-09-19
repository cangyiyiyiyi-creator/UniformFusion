#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PY="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
PREREQ_ROOT="${PREREQ_ROOT:-$ROOT/论文补充验证_20260905/06_首批9组训练结果/run_20260905_first_batch}"
[[ -f "$PREREQ_ROOT/suite_complete.marker" ]] || { echo "First-batch 9-run suite is not complete: $PREREQ_ROOT"; exit 2; }
SAVE_ROOT="${SAVE_ROOT:-$ROOT/论文补充验证_20260905/07_UF_GAP三种子/run_20260906_uf_gap_3seeds}"
mkdir -p "$SAVE_ROOT"
SEEDS=(930163947 1786430941 553800223)
{
 echo "method=UF_GAP"; echo "seeds=${SEEDS[*]}"
 echo "only_change=class-conditioned Top-K pooling replaced by uniform C4/C5 global average pooling"
 echo "selection=finish all Val runs, lock, then Test once"
 echo "prerequisite=$PREREQ_ROOT/suite_complete.marker"
} > "$SAVE_ROOT/protocol.txt"
for phase in val test; do
 [[ "$phase" == test ]] && touch "$SAVE_ROOT/validation_phase_complete.marker"
 repeat=0
 for seed in "${SEEDS[@]}"; do
  repeat=$((repeat+1))
  SAVE_ROOT="$SAVE_ROOT" METHOD=UF_GAP SEED="$seed" REPEAT="$repeat" PHASE="$phase" \
   bash run_reviewer_first_batch_one.sh
 done
done
"$PY" - "$SAVE_ROOT" <<'PY'
import csv,json,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]); rows=[]
for p in sorted(r.rglob('test_metrics.json')):
 d=json.loads(p.read_text()); v=json.loads((p.parent/'val_metrics.json').read_text())
 rows.append({'seed':int(p.parents[2].name.split('_')[-1]),'val_mAP':v['stats']['mAP'],'test_mAP':d['stats']['mAP'],'checkpoint':str(p.parent/'checkpoint_best.pth')})
if len(rows)!=3: raise RuntimeError(f'expected 3 results, got {len(rows)}')
with (r/'UF_GAP_逐seed.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
vals=[x['test_mAP'] for x in rows]
(r/'UF_GAP_汇总.txt').write_text(f'n=3\ntest_mAP={statistics.mean(vals):.6f}±{statistics.stdev(vals):.6f}\n')
PY
touch "$SAVE_ROOT/suite_complete.marker"
echo "UF-GAP DONE: $SAVE_ROOT"

