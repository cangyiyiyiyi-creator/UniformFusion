#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "runs_p9_final_noanchor_ablation"
    / "run_20260829_p9_final_noanchor_ablation_3seeds_final" / "resnet50"
)
MAIN_CSV = (
    ROOT / "runs_uniform_fusion_n5_confirmation"
    / "run_20260831_uniform_missing2_locked_final" / "uniform_n5_detailed.csv"
)
SEEDS = ((930163947, 1), (1786430941, 2), (553800223, 3))
METHODS = {
    "Final_NoAnchor_NoCounterfactualExperts": "Uniform_NoCounterfactual",
    "Final_NoAnchor_NoGuardLoss": "Uniform_NoGuard",
    "Final_NoAnchor_SingleScaleC5": "Uniform_C5Only",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="uniform_component_summary_") as temp:
        root = Path(temp)
        data = root / "data"
        output = root / "output"
        for seed, repeat in SEEDS:
            for source_method, target_method in METHODS.items():
                source = SOURCE / f"seed_{seed}" / f"repeat_{repeat}" / source_method
                target = data / "resnet50" / f"seed_{seed}" / f"repeat_{repeat}" / target_method
                target.mkdir(parents=True, exist_ok=True)
                for name in ("val_metrics.json", "test_metrics.json"):
                    shutil.copy2(source / name, target / name)
        subprocess.run(
            [
                sys.executable, str(ROOT / "tools/summarize_uniform_component_ablation.py"),
                "--uniform-main-csv", str(MAIN_CSV),
                "--ablation-root", str(data),
                "--classes-file", str(ROOT / "annotations/classes.txt"),
                "--output-root", str(output),
            ],
            cwd=ROOT,
            check=True,
        )
        detailed = rows(output / "uniform_component_ablation_3seeds_detailed.csv")
        summary = rows(output / "uniform_component_ablation_3seeds_summary.csv")
        if len(detailed) != 12 or len(summary) != 4:
            raise AssertionError(
                f"unexpected summary sizes: detailed={len(detailed)} summary={len(summary)}"
            )
    print("UNIFORM_COMPONENT_SUMMARY_SMOKE_OK rows=12 methods=4")


if __name__ == "__main__":
    main()
