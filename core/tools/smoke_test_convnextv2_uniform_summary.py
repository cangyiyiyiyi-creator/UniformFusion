#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT
    / "runs_final_noanchor_generalization"
    / "run_20260830_final_noanchor_convnextv2_3seeds_locked"
)
BASELINE_CSV = (
    ROOT
    / "runs_final_noanchor_evidence"
    / "run_20260830_final_noanchor_all_evidence_locked"
    / "convnext_detailed.csv"
)
SEEDS = ((930163947, 1), (1786430941, 2), (553800223, 3))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if not BASELINE_CSV.is_file():
        raise FileNotFoundError(BASELINE_CSV)
    with tempfile.TemporaryDirectory(prefix="convnext_uniform_summary_") as temp:
        root = Path(temp)
        uniform_root = root / "uniform"
        output_root = root / "output"
        for seed, repeat in SEEDS:
            source = (
                SOURCE_ROOT
                / "convnextv2_tiny"
                / f"seed_{seed}"
                / f"repeat_{repeat}"
                / "Final_NoAnchor"
            )
            target = (
                uniform_root
                / "convnextv2_tiny"
                / f"seed_{seed}"
                / f"repeat_{repeat}"
                / "Uniform_Fusion"
            )
            target.mkdir(parents=True, exist_ok=True)
            for name in ("val_metrics.json", "test_metrics.json"):
                shutil.copy2(source / name, target / name)

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "summarize_convnextv2_uniform_fusion.py"),
                "--baseline-detailed-csv",
                str(BASELINE_CSV),
                "--uniform-root",
                str(uniform_root),
                "--classes-file",
                str(ROOT / "annotations" / "classes.txt"),
                "--output-root",
                str(output_root),
            ],
            check=True,
            cwd=ROOT,
        )
        detailed = read_csv(
            output_root / "convnextv2_uniform_fusion_3seeds_detailed.csv"
        )
        summary = read_csv(
            output_root / "convnextv2_uniform_fusion_3seeds_summary.csv"
        )
        if len(detailed) != 6 or len(summary) != 2:
            raise AssertionError(
                f"unexpected result sizes: detailed={len(detailed)} summary={len(summary)}"
            )
        if {row["method"] for row in detailed} != {"Plain_BCE", "Uniform_Fusion"}:
            raise AssertionError("unexpected methods in detailed smoke result")
    print("CONVNEXTV2_UNIFORM_SUMMARY_SMOKE_OK rows=6 methods=2")


if __name__ == "__main__":
    main()
