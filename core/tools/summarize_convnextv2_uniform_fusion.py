#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_SEEDS = {930163947, 1786430941, 553800223}
EXPECTED_VAL_SHA = "a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67"
EXPECTED_TEST_SHA = "6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6"
WEAK_CLASSES = {"Knife", "Scissors", "Lighter", "Razor_blade"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-detailed-csv", required=True)
    parser.add_argument("--uniform-root", required=True)
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


def baseline_rows(path: Path) -> list[dict]:
    rows = []
    for row in read_csv(path):
        if row["backbone"] != "convnextv2_tiny" or row["method"] != "Plain_BCE":
            continue
        normalized = dict(row)
        normalized["seed"] = int(row["seed"])
        normalized["repeat"] = int(row["repeat"])
        rows.append(normalized)
    return rows


def uniform_rows(root: Path, classes: list[str]) -> list[dict]:
    rows = []
    pattern = "convnextv2_tiny/seed_*/repeat_*/Uniform_Fusion/val_metrics.json"
    for val_path in sorted(root.glob(pattern)):
        test_path = val_path.with_name("test_metrics.json")
        if not test_path.is_file():
            raise FileNotFoundError(f"missing Test result: {test_path}")
        val_payload = read_json(val_path)
        test_payload = read_json(test_path)
        if val_payload.get("evaluation_list_sha256") != EXPECTED_VAL_SHA:
            raise ValueError(f"validation hash mismatch: {val_path}")
        if test_payload.get("evaluation_list_sha256") != EXPECTED_TEST_SHA:
            raise ValueError(f"Test hash mismatch: {test_path}")

        relative = val_path.relative_to(root).parts
        seed = int(relative[1].removeprefix("seed_"))
        repeat = int(relative[2].removeprefix("repeat_"))
        per_class = [float(value) for value in test_payload["stats"]["per_class_ap"]]
        weak = statistics.fmean(
            value for name, value in zip(classes, per_class) if name in WEAK_CLASSES
        )
        row = {
            "backbone": "convnextv2_tiny",
            "seed": seed,
            "repeat": repeat,
            "method": "Uniform_Fusion",
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


def validate(method: str, rows: list[dict]) -> None:
    seeds = {int(row["seed"]) for row in rows}
    if seeds != EXPECTED_SEEDS or len(rows) != 3:
        raise ValueError(
            f"{method}: expected fixed n=3, got rows={len(rows)} seeds={sorted(seeds)}"
        )
    if any(row["test_list_sha256"] != EXPECTED_TEST_SHA for row in rows):
        raise ValueError(f"{method}: Test split hash mismatch")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    classes = [
        line.strip()
        for line in Path(args.classes_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = baseline_rows(Path(args.baseline_detailed_csv))
    rows += uniform_rows(Path(args.uniform_root), classes)

    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    if set(by_method) != {"Plain_BCE", "Uniform_Fusion"}:
        raise ValueError(f"unexpected methods: {sorted(by_method)}")
    for method, selected in by_method.items():
        validate(method, selected)

    fields = [
        "backbone", "seed", "repeat", "method", "best_val_epoch",
        "best_val_mAP", "test_mAP", "test_weak_class_mAP", "checkpoint",
        "test_list_sha256", *[f"test_AP_{name}" for name in classes],
    ]
    rows.sort(key=lambda row: (int(row["seed"]), row["method"]))
    detailed_path = output_root / "convnextv2_uniform_fusion_3seeds_detailed.csv"
    with detailed_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    plain = {
        int(row["seed"]): float(row["test_mAP"])
        for row in by_method["Plain_BCE"]
    }
    summary_rows = []
    for method in ("Plain_BCE", "Uniform_Fusion"):
        selected = by_method[method]
        val_values = [float(row["best_val_mAP"]) for row in selected]
        test_values = [float(row["test_mAP"]) for row in selected]
        weak_values = [float(row["test_weak_class_mAP"]) for row in selected]
        deltas = [float(row["test_mAP"]) - plain[int(row["seed"])] for row in selected]
        summary_rows.append(
            {
                "backbone": "convnextv2_tiny",
                "method": method,
                "n": len(selected),
                "best_val_mAP_mean": statistics.fmean(val_values),
                "best_val_mAP_std_sample": sample_std(val_values),
                "test_mAP_mean": statistics.fmean(test_values),
                "test_mAP_std_sample": sample_std(test_values),
                "delta_vs_Plain": statistics.fmean(deltas),
                "wins_vs_Plain": sum(delta > 0 for delta in deltas),
                "test_weak_class_mAP_mean": statistics.fmean(weak_values),
                "test_weak_class_mAP_std_sample": sample_std(weak_values),
            }
        )
    summary_path = output_root / "convnextv2_uniform_fusion_3seeds_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"CONVNEXTV2_UNIFORM_SUMMARY_OK rows={len(rows)} seeds=3 output={output_root}")


if __name__ == "__main__":
    main()
