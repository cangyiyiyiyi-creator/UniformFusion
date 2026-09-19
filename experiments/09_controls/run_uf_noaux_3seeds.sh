#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PYTHON_BIN:-/home/hfuu/miniforge3/envs/v2b384_env/bin/python}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT/论文补充验证_20260907/01_UF_NoAux三种子/run_20260907_uf_noaux_3seeds}"
SEEDS=(930163947 1786430941 553800223)
mkdir -p "$SAVE_ROOT"

{
  echo "method=UF_NoAux"
  echo "seeds=${SEEDS[*]}"
  echo "only_change=plain_innovation_aux_weight: 0.03 -> 0"
  echo "selection=finish all Val runs, lock configuration, then evaluate Test once"
  echo "reference=Uniform Fusion with Top-K=8, uniform expert fusion, Guard=0.10, SingleLoss=0.02"
} > "$SAVE_ROOT/protocol.txt"

for phase in val test; do
  [[ "$phase" == test ]] && touch "$SAVE_ROOT/validation_phase_complete.marker"
  repeat=0
  for seed in "${SEEDS[@]}"; do
    repeat=$((repeat + 1))
    SAVE_ROOT="$SAVE_ROOT" METHOD=UF_NoAux SEED="$seed" REPEAT="$repeat" PHASE="$phase" \
      bash run_reviewer_first_batch_one.sh
  done
done

"$PY" - "$SAVE_ROOT" <<'PY'
import csv, json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.rglob("test_metrics.json")):
    test = json.loads(path.read_text())
    val = json.loads((path.parent / "val_metrics.json").read_text())
    rows.append({
        "seed": int(path.parents[2].name.split("_")[-1]),
        "best_epoch": val["checkpoint_epoch"],
        "val_mAP": val["stats"]["mAP"],
        "test_mAP": test["stats"]["mAP"],
        "checkpoint": str(path.parent / "checkpoint_best.pth"),
    })
if len(rows) != 3:
    raise RuntimeError(f"expected 3 results, got {len(rows)}")
with (root / "UF_NoAux_逐seed.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
values = [row["test_mAP"] for row in rows]
(root / "UF_NoAux_汇总.txt").write_text(
    f"n=3\ntest_mAP={statistics.mean(values):.6f}±{statistics.stdev(values):.6f}\n",
    encoding="utf-8",
)
PY

touch "$SAVE_ROOT/suite_complete.marker"
echo "UF-NoAux DONE: $SAVE_ROOT"
