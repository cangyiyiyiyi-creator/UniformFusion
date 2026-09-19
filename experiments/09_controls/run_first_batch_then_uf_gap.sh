#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
FIRST_ROOT="${FIRST_ROOT:-$ROOT/supplementary_verification_20260905/06_first_batch_9runs_results/run_20260905_first_batch}"
SAVE_ROOT="$FIRST_ROOT" bash run_reviewer_first_batch_9runs.sh
PREREQ_ROOT="$FIRST_ROOT" bash run_uf_gap_3seeds.sh
