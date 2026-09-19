#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_SEEDS = {930163947, 1786430941, 553800223, 207027553, 1716854429}
EXPECTED_VAL_SHA = "a795ccfb147de3d16836b74d5640ed2c4f6ee4b3d78f7ad6c21e14a3fd4f1a67"
EXPECTED_TEST_SHA = "6c50e83f34a499243c3c103137584981f66ac258d9a7abe09c8131ec484276c6"
WEAK_CLASSES = {"Knife", "Scissors", "Lighter", "Razor_blade"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-ablation-csv", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--main-detailed-csv", required=True)
    parser.add_argument("--classes-file", default="annotations/classes.txt")
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def normalized_main_rows(path: Path) -> list[dict]:
    rows = read_csv(path)
    selected = [
        row for row in rows if row["method"] in {"Plain_BCE", "Final_NoAnchor"}
    ]
    for row in selected:
        row["seed"] = int(row["seed"])
        row["repeat"] = int(row["repeat"])
    return selected


def normalized_existing_uniform(path: Path) -> list[dict]:
    output = []
    for row in read_csv(path):
        if row["method"] != "Final_NoAnchor_NoRouter":
            continue
        normalized = {
            "backbone": row["backbone"],
            "seed": int(row["seed"]),
            "repeat": int(row["repeat"]),
            "method": "Uniform_Fusion",
            "best_val_epoch": row["checkpoint_epoch"],
            "best_val_mAP": row["val_mAP"],
            "test_mAP": row["test_mAP"],
            "test_weak_class_mAP": row["test_weak_class_mAP"],
            "checkpoint": row["checkpoint"],
            "test_list_sha256": row["test_list_sha256"],
        }
        normalized.update(
            {key: value for key, value in row.items() if key.startswith("test_AP_")}
        )
        output.append(normalized)
    return output


def normalized_new_uniform(root: Path, classes: list[str]) -> list[dict]:
    output = []
    pattern = "resnet50/seed_*/repeat_*/Final_NoAnchor_NoRouter/val_metrics.json"
    for val_path in sorted(root.glob(pattern)):
        test_path = val_path.with_name("test_metrics.json")
        if not test_path.is_file():
            raise FileNotFoundError(f"missing test result: {test_path}")
        val_payload = read_metrics(val_path)
        test_payload = read_metrics(test_path)
        if val_payload.get("evaluation_list_sha256") != EXPECTED_VAL_SHA:
            raise ValueError(f"validation hash mismatch: {val_path}")
        if test_payload.get("evaluation_list_sha256") != EXPECTED_TEST_SHA:
            raise ValueError(f"test hash mismatch: {test_path}")
        relative = val_path.relative_to(root).parts
        seed = int(relative[1].removeprefix("seed_"))
        repeat = int(relative[2].removeprefix("repeat_"))
        per_class = [float(value) for value in test_payload["stats"]["per_class_ap"]]
        weak = statistics.fmean(
            value for name, value in zip(classes, per_class) if name in WEAK_CLASSES
        )
        row = {
            "backbone": "resnet50",
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
        output.append(row)
    return output


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    classes = [
        line.strip()
        for line in Path(args.classes_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = normalized_main_rows(Path(args.main_detailed_csv))
    rows += normalized_existing_uniform(Path(args.existing_ablation_csv))
    rows += normalized_new_uniform(Path(args.new_root), classes)

    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    expected_methods = {"Plain_BCE", "Final_NoAnchor", "Uniform_Fusion"}
    if set(by_method) != expected_methods:
        raise ValueError(f"unexpected methods: {sorted(by_method)}")
    for method, method_rows in by_method.items():
        seeds = {int(row["seed"]) for row in method_rows}
        if seeds != EXPECTED_SEEDS or len(method_rows) != 5:
            raise ValueError(
                f"{method}: expected fixed n=5 seeds, got rows={len(method_rows)} "
                f"seeds={sorted(seeds)}"
            )

    fields = [
        "backbone", "seed", "repeat", "method", "best_val_epoch",
        "best_val_mAP", "test_mAP", "test_weak_class_mAP", "checkpoint",
        "test_list_sha256", *[f"test_AP_{name}" for name in classes],
    ]
    rows.sort(key=lambda row: (int(row["seed"]), row["method"]))
    detailed = output_root / "uniform_n5_detailed.csv"
    with detailed.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    plain = {int(row["seed"]): float(row["test_mAP"]) for row in by_method["Plain_BCE"]}
    final = {
        int(row["seed"]): float(row["test_mAP"])
        for row in by_method["Final_NoAnchor"]
    }
    summary_rows = []
    for method in ("Plain_BCE", "Final_NoAnchor", "Uniform_Fusion"):
        selected = by_method[method]
        values = [float(row["test_mAP"]) for row in selected]
        weak = [float(row["test_weak_class_mAP"]) for row in selected]
        deltas_plain = [float(row["test_mAP"]) - plain[int(row["seed"])] for row in selected]
        deltas_final = [float(row["test_mAP"]) - final[int(row["seed"])] for row in selected]
        summary_rows.append(
            {
                "method": method,
                "n": len(selected),
                "test_mAP_mean": statistics.fmean(values),
                "test_mAP_std_sample": sample_std(values),
                "delta_vs_Plain": statistics.fmean(deltas_plain),
                "wins_vs_Plain": sum(delta > 0 for delta in deltas_plain),
                "delta_vs_Final": statistics.fmean(deltas_final),
                "wins_vs_Final": sum(delta > 0 for delta in deltas_final),
                "test_weak_class_mAP_mean": statistics.fmean(weak),
            }
        )
    summary = output_root / "uniform_n5_summary.csv"
    with summary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"UNIFORM_N5_SUMMARY_OK rows={len(rows)} seeds=5 output={output_root}")


if __name__ == "__main__":
    main()
