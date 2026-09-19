#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT/论文补充验证_20260907/02_公平Adapter基线/run_20260907_fair_adapters_3seeds}"
SEEDS=(930163947 1786430941 553800223)
METHODS=(DAGNet_OfficialArchitecture MLDecoder_DualView)
mkdir -p "$SAVE_ROOT"

{
  echo "seeds=${SEEDS[*]}"
  echo "methods=${METHODS[*]}"
  echo "selection=all Val runs finish and lock before any Test evaluation"
  echo "shared_protocol=AdamW,BCE,180 epochs,patience 25,conditional augmentation"
  echo "DAGNet=official architecture at required 256 input; project data and optimization protocol"
  echo "MLDecoder=official head; shared ResNet50 C5 additive dual-view fusion at 224 input"
} > "$SAVE_ROOT/protocol.txt"

for phase in val test; do
  [[ "$phase" == test ]] && touch "$SAVE_ROOT/validation_phase_complete.marker"
  for method in "${METHODS[@]}"; do
    repeat=0
    for seed in "${SEEDS[@]}"; do
      repeat=$((repeat + 1))
      SAVE_ROOT="$SAVE_ROOT" METHOD="$method" SEED="$seed" REPEAT="$repeat" PHASE="$phase" \
        bash run_fair_adapter_one.sh
    done
  done
done

"$PY" - "$SAVE_ROOT" <<'PY'
import csv, json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1]); rows = []
for path in sorted(root.rglob("test_metrics.json")):
    test = json.loads(path.read_text()); val = json.loads((path.parent / "val_metrics.json").read_text())
    rows.append({"method": path.parent.name, "seed": int(path.parents[2].name.split("_")[-1]),
                 "best_epoch": val["checkpoint_epoch"], "val_mAP": val["stats"]["mAP"],
                 "test_mAP": test["stats"]["mAP"], "checkpoint": str(path.parent / "checkpoint_best.pth")})
if len(rows) != 6: raise RuntimeError(f"expected 6 results, got {len(rows)}")
with (root / "公平Adapter_逐seed.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
summary=[]
for method in sorted({r["method"] for r in rows}):
    values=[r["test_mAP"] for r in rows if r["method"]==method]
    summary.append({"method":method,"n":len(values),"test_mAP_mean":statistics.mean(values),
                    "test_mAP_std_sample":statistics.stdev(values)})
with (root / "公平Adapter_汇总.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
PY
touch "$SAVE_ROOT/suite_complete.marker"
echo "FAIR ADAPTER SUITE DONE: $SAVE_ROOT"
