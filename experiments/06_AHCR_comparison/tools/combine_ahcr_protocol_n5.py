#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_SEEDS = {930163947, 1786430941, 553800223, 207027553, 1716854429}
METHOD_ORDER = (
    "Plain_BCE",
    "AHCR_Official_Strict",
    "AHCR_Uniform",
    "Final_NoAnchor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-detailed", required=True)
    parser.add_argument("--uniform-detailed", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    strict_rows = read_rows(Path(args.strict_detailed))
    uniform_rows = read_rows(Path(args.uniform_detailed))
    selected: dict[tuple[str, int], dict[str, str]] = {}
    for row in strict_rows + uniform_rows:
        method = row["method"]
        if method not in METHOD_ORDER:
            continue
        key = (method, int(row["seed"]))
        if key in selected:
            prior = selected[key]
            for field in ("repeat", "best_val_mAP", "test_mAP", "test_list_sha256"):
                if prior[field] != row[field]:
                    raise ValueError(f"duplicate baseline mismatch for {key}: {field}")
            continue
        selected[key] = row

    rows = list(selected.values())
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    if set(by_method) != set(METHOD_ORDER):
        raise ValueError(f"missing combined AHCR methods: {sorted(by_method)}")
    for method in METHOD_ORDER:
        seeds = {int(row["seed"]) for row in by_method[method]}
        if len(by_method[method]) != 5 or seeds != EXPECTED_SEEDS:
            raise ValueError(f"{method} is not fixed n=5: {sorted(seeds)}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(
        list(uniform_rows[0].keys()) + list(strict_rows[0].keys())
    ))
    rows.sort(key=lambda row: (int(row["seed"]), METHOD_ORDER.index(row["method"])))
    detailed_path = output_root / "ahcr_official_and_uniform_n5_detailed.csv"
    with detailed_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    plain = {
        int(row["seed"]): float(row["test_mAP"])
        for row in by_method["Plain_BCE"]
    }
    summary = []
    for method in METHOD_ORDER:
        method_rows = by_method[method]
        test_values = [float(row["test_mAP"]) for row in method_rows]
        val_values = [float(row["best_val_mAP"]) for row in method_rows]
        weak_values = [float(row["test_weak_class_mAP"]) for row in method_rows]
        deltas = [
            float(row["test_mAP"]) - plain[int(row["seed"])]
            for row in method_rows
        ]
        summary.append(
            {
                "method": method,
                "n": 5,
                "val_mAP_mean": statistics.fmean(val_values),
                "val_mAP_std_sample": sample_std(val_values),
                "test_mAP_mean": statistics.fmean(test_values),
                "test_mAP_std_sample": sample_std(test_values),
                "test_weak_class_mAP_mean": statistics.fmean(weak_values),
                "test_delta_vs_Plain_mean": statistics.fmean(deltas),
                "test_wins_vs_Plain": sum(delta > 0 for delta in deltas),
                "protocol_relation_to_Plain": (
                    "different_recipe_descriptive_only"
                    if method == "AHCR_Official_Strict"
                    else "same_locked_protocol"
                ),
            }
        )
    summary_path = output_root / "ahcr_official_and_uniform_n5_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"AHCR_PROTOCOL_COMBINE_OK rows={len(rows)} output={output_root}")


if __name__ == "__main__":
    main()
