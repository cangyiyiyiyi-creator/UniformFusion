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
STRICT_METHOD = "AHCR_Official_Strict"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--main-detailed-csv", required=True)
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


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    classes = [
        line.strip()
        for line in Path(args.classes_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows: list[dict] = []
    for row in read_csv(Path(args.main_detailed_csv)):
        if row["method"] not in {"Plain_BCE", "Final_NoAnchor"}:
            continue
        row["seed"] = int(row["seed"])
        row["repeat"] = int(row["repeat"])
        row["selection_rule"] = "val_best"
        rows.append(row)

    pattern = "resnet50/seed_*/repeat_*/AHCR_Official_Strict/val_metrics.json"
    for val_path in sorted(run_root.glob(pattern)):
        test_path = val_path.with_name("test_metrics.json")
        if not test_path.is_file():
            raise FileNotFoundError(f"missing strict test metrics: {test_path}")
        val_payload = read_json(val_path)
        test_payload = read_json(test_path)
        if val_payload.get("evaluation_list_sha256") != EXPECTED_VAL_SHA:
            raise ValueError(f"validation hash mismatch: {val_path}")
        if test_payload.get("evaluation_list_sha256") != EXPECTED_TEST_SHA:
            raise ValueError(f"test hash mismatch: {test_path}")
        if val_payload.get("selection_rule") != "final_epoch_no_validation_selection":
            raise ValueError(f"invalid strict selection rule: {val_path}")
        parts = val_path.relative_to(run_root).parts
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
            "method": STRICT_METHOD,
            "best_val_epoch": 29,
            "best_val_mAP": float(val_payload["stats"]["mAP"]),
            "test_mAP": float(test_payload["stats"]["mAP"]),
            "test_weak_class_mAP": weak,
            "checkpoint": val_payload["checkpoint"],
            "test_list_sha256": test_payload["evaluation_list_sha256"],
            "selection_rule": "final_epoch_no_validation_selection",
        }
        row.update(
            {f"test_AP_{name}": value for name, value in zip(classes, per_class)}
        )
        rows.append(row)

    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    expected_methods = {"Plain_BCE", "Final_NoAnchor", STRICT_METHOD}
    if set(by_method) != expected_methods:
        raise ValueError(f"unexpected methods: {sorted(by_method)}")
    for method, method_rows in by_method.items():
        seeds = {int(row["seed"]) for row in method_rows}
        if len(method_rows) != 5 or seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"{method}: expected fixed n=5, got rows={len(method_rows)} "
                f"seeds={sorted(seeds)}"
            )

    fields = [
        "backbone", "seed", "repeat", "method", "best_val_epoch",
        "best_val_mAP", "test_mAP", "test_weak_class_mAP", "checkpoint",
        "test_list_sha256", "selection_rule",
        *[f"test_AP_{name}" for name in classes],
    ]
    rows.sort(key=lambda row: (int(row["seed"]), row["method"]))
    detailed_path = output_root / "ahcr_official_strict_n5_detailed.csv"
    with detailed_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    plain = {
        int(row["seed"]): float(row["test_mAP"])
        for row in by_method["Plain_BCE"]
    }
    final = {
        int(row["seed"]): float(row["test_mAP"])
        for row in by_method["Final_NoAnchor"]
    }
    summary_rows = []
    for method in ("Plain_BCE", STRICT_METHOD, "Final_NoAnchor"):
        selected = by_method[method]
        values = [float(row["test_mAP"]) for row in selected]
        weak = [float(row["test_weak_class_mAP"]) for row in selected]
        delta_plain = [
            float(row["test_mAP"]) - plain[int(row["seed"])] for row in selected
        ]
        delta_final = [
            float(row["test_mAP"]) - final[int(row["seed"])] for row in selected
        ]
        summary_rows.append(
            {
                "method": method,
                "n": len(selected),
                "test_mAP_mean": statistics.fmean(values),
                "test_mAP_std_sample": sample_std(values),
                "delta_vs_Plain": statistics.fmean(delta_plain),
                "wins_vs_Plain": sum(delta > 0 for delta in delta_plain),
                "delta_vs_Final": statistics.fmean(delta_final),
                "wins_vs_Final": sum(delta > 0 for delta in delta_final),
                "test_weak_class_mAP_mean": statistics.fmean(weak),
                "comparison_note": (
                    "descriptive_only_different_training_protocol"
                    if method == STRICT_METHOD
                    else "same_locked_project_protocol"
                ),
            }
        )
    summary_path = output_root / "ahcr_official_strict_n5_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"AHCR_OFFICIAL_STRICT_N5_SUMMARY_OK rows={len(rows)} output={output_root}")


if __name__ == "__main__":
    main()
