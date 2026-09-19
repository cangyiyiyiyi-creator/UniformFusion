#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PY="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT/supplementary_verification_20260907/03_valonly_sensitivity/run_20260907_val_sensitivity}"
SEED=930163947
METHODS=(UF_K4 UF_K16 UF_Temp010 UF_Temp040 UF_GammaMax002 UF_GammaMax010)
mkdir -p "$SAVE_ROOT"
printf 'split=validation_only\nseed=%s\nmethods=%s\ndefault=K8_temperature0.2_gamma_max0.05_reused_from_locked_Uniform\nTest_is_never_evaluated=true\n' "$SEED" "${METHODS[*]}" > "$SAVE_ROOT/protocol.txt"
repeat=0
for method in "${METHODS[@]}"; do
  repeat=$((repeat+1))
  SAVE_ROOT="$SAVE_ROOT" METHOD="$method" SEED="$SEED" REPEAT="$repeat" PHASE=val bash run_reviewer_first_batch_one.sh
done
"$PY" - "$SAVE_ROOT" <<'PY'
import csv,json,sys
from pathlib import Path
r=Path(sys.argv[1]); rows=[]
for p in sorted(r.rglob('val_metrics.json')):
 d=json.loads(p.read_text()); rows.append({'method':p.parent.name,'seed':930163947,'best_epoch':d['checkpoint_epoch'],'val_mAP':d['stats']['mAP'],'test_accessed':False})
if len(rows)!=6: raise RuntimeError(f'expected 6 Val results, got {len(rows)}')
with (r/'val_only_sensitivity.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
PY
touch "$SAVE_ROOT/suite_complete.marker"
