#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
FIRST_ROOT="${FIRST_ROOT:-$ROOT/论文补充验证_20260905/06_首批9组训练结果/run_20260905_first_batch}"
SAVE_ROOT="$FIRST_ROOT" bash run_reviewer_first_batch_9runs.sh
PREREQ_ROOT="$FIRST_ROOT" bash run_uf_gap_3seeds.sh
