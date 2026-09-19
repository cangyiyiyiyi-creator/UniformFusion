#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


SEEDS = {930163947, 1786430941, 553800223}
METHODS = (
    "Uniform_Fusion",
    "Uniform_NoCounterfactual",
    "Uniform_NoGuard",
    "Uniform_C5Only",
)
VAL_SHA = "a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67"
TEST_SHA = "6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6"
WEAK_CLASSES = {"Knife", "Scissors", "Lighter", "Razor_blade"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform-main-csv", required=True)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--classes-file", default="annotations/classes.txt")
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main_rows(path: Path) -> list[dict]:
    rows = []
    for row in read_csv(path):
        if row["method"] != "Uniform_Fusion" or int(row["seed"]) not in SEEDS:
            continue
        item = dict(row)
        item["seed"] = int(row["seed"])
        item["repeat"] = int(row["repeat"])
        rows.append(item)
    return rows


def ablation_rows(root: Path, classes: list[str]) -> list[dict]:
    rows = []
    pattern = "resnet50/seed_*/repeat_*/Uniform_*/val_metrics.json"
    for val_path in sorted(root.glob(pattern)):
        method = val_path.parent.name
        if method not in METHODS[1:]:
            continue
        test_path = val_path.with_name("test_metrics.json")
        if not test_path.is_file():
            raise FileNotFoundError(f"missing Test result: {test_path}")
        val_payload = read_json(val_path)
        test_payload = read_json(test_path)
        if val_payload.get("evaluation_list_sha256") != VAL_SHA:
            raise ValueError(f"Val split mismatch: {val_path}")
        if test_payload.get("evaluation_list_sha256") != TEST_SHA:
            raise ValueError(f"Test split mismatch: {test_path}")
        parts = val_path.relative_to(root).parts
        seed = int(parts[1].removeprefix("seed_"))
        repeat = int(parts[2].removeprefix("repeat_"))
        per_class = [float(value) for value in test_payload["stats"]["per_class_ap"]]
        weak = statistics.fmean(
            value for name, value in zip(classes, per_class) if name in WEAK_CLASSES
        )
        row = {
            "backbone": "resnet50",
            "seed": seed,
            "repeat": repeat,
            "method": method,
            "best_val_epoch": val_payload.get("checkpoint_epoch", ""),
            "best_val_mAP": float(val_payload["stats"]["mAP"]),
            "test_mAP": float(test_payload["stats"]["mAP"]),
            "test_weak_class_mAP": weak,
            "checkpoint": val_payload["checkpoint"],
            "test_list_sha256": test_payload["evaluation_list_sha256"],
        }
        row.update(
            {f"test_AP_{name}": value for name, value in zip(classes, per_class)}
        )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    classes = [
        line.strip()
        for line in Path(args.classes_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = main_rows(Path(args.uniform_main_csv))
    rows += ablation_rows(Path(args.ablation_root), classes)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    if set(grouped) != set(METHODS):
        raise ValueError(f"unexpected methods: {sorted(grouped)}")
    for method in METHODS:
        selected = grouped[method]
        seeds = {int(row["seed"]) for row in selected}
        if len(selected) != 3 or seeds != SEEDS:
            raise ValueError(
                f"{method}: expected n=3 seeds={sorted(SEEDS)}, got n={len(selected)} seeds={sorted(seeds)}"
            )
        if any(row["test_list_sha256"] != TEST_SHA for row in selected):
            raise ValueError(f"{method}: Test split mismatch")

    fields = [
        "backbone", "seed", "repeat", "method", "best_val_epoch",
        "best_val_mAP", "test_mAP", "test_weak_class_mAP", "checkpoint",
        "test_list_sha256", *[f"test_AP_{name}" for name in classes],
    ]
    rows.sort(key=lambda row: (int(row["seed"]), METHODS.index(row["method"])))
    detailed = output_root / "uniform_component_ablation_3seeds_detailed.csv"
    with detailed.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    main_test = {
        int(row["seed"]): float(row["test_mAP"])
        for row in grouped["Uniform_Fusion"]
    }
    summary_rows = []
    for method in METHODS:
        selected = grouped[method]
        val_values = [float(row["best_val_mAP"]) for row in selected]
        test_values = [float(row["test_mAP"]) for row in selected]
        weak_values = [float(row["test_weak_class_mAP"]) for row in selected]
        deltas = [float(row["test_mAP"]) - main_test[int(row["seed"])] for row in selected]
        summary_rows.append(
            {
                "method": method,
                "n": 3,
                "best_val_mAP_mean": statistics.fmean(val_values),
                "best_val_mAP_std_sample": sample_std(val_values),
                "test_mAP_mean": statistics.fmean(test_values),
                "test_mAP_std_sample": sample_std(test_values),
                "delta_vs_Uniform_Fusion": statistics.fmean(deltas),
                "wins_vs_Uniform_Fusion": sum(delta > 0 for delta in deltas),
                "test_weak_class_mAP_mean": statistics.fmean(weak_values),
                "test_weak_class_mAP_std_sample": sample_std(weak_values),
            }
        )
    summary = output_root / "uniform_component_ablation_3seeds_summary.csv"
    with summary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"UNIFORM_COMPONENT_SUMMARY_OK rows={len(rows)} methods=4 seeds=3 output={output_root}")


if __name__ == "__main__":
    main()
