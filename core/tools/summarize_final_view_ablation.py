#!/usr/bin/env python3
"""Summarize locked Final-NoAnchor view ablations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


MODES = ("paired", "ol_only", "sd_only", "mismatched")
METHODS = ("Plain_BCE", "Final_NoAnchor")
WEAK_CLASSES = {"Knife", "Scissors", "Lighter", "Razor_blade"}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-seeds", type=int, required=True)
    parser.add_argument("--expected-list-sha256", required=True)
    parser.add_argument("--classes-file", default="annotations/classes.txt")
    args = parser.parse_args()

    root = Path(args.root)
    classes = [
        line.strip()
        for line in Path(args.classes_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    weak_indices = [
        index for index, name in enumerate(classes) if name in WEAK_CLASSES
    ]
    rows = []
    for mode in MODES:
        for payload_path in sorted((root / mode).glob("seed_*/*.json")):
            method = payload_path.stem
            if method not in METHODS:
                continue
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            if payload.get("requested_view_mode") != mode:
                raise ValueError(f"view-mode mismatch in {payload_path}")
            split_hash = payload.get("evaluation_list_sha256", "")
            if split_hash != args.expected_list_sha256:
                raise ValueError(f"evaluation-list hash mismatch in {payload_path}")
            stats = payload["stats"]
            per_class = [float(value) for value in stats["per_class_ap"]]
            seed = int(payload_path.parent.name.removeprefix("seed_"))
            row = {
                "mode": mode,
                "seed": seed,
                "method": method,
                "mAP": float(stats["mAP"]),
                "weak_class_mAP": mean([per_class[i] for i in weak_indices]),
                "evaluation_list_sha256": split_hash,
            }
            row.update(
                {f"AP_{name}": per_class[index] for index, name in enumerate(classes)}
            )
            rows.append(row)

    expected_rows = len(MODES) * len(METHODS) * args.expected_seeds
    if len(rows) != expected_rows:
        raise SystemExit(
            f"expected {expected_rows} evaluations, found {len(rows)} under {root}"
        )

    detailed_path = root / "final_view_ablation_detailed.csv"
    with detailed_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lookup = {
        (row["mode"], row["method"], row["seed"]): row for row in rows
    }
    summary_rows = []
    for mode in MODES:
        for method in METHODS:
            selected = [
                row for row in rows
                if row["mode"] == mode and row["method"] == method
            ]
            maps = [float(row["mAP"]) for row in selected]
            weak_maps = [float(row["weak_class_mAP"]) for row in selected]
            paired_deltas = [
                float(row["mAP"])
                - float(lookup[("paired", method, row["seed"])]["mAP"])
                for row in selected
            ]
            method_deltas = [
                float(lookup[(mode, "Final_NoAnchor", row["seed"])]["mAP"])
                - float(lookup[(mode, "Plain_BCE", row["seed"])]["mAP"])
                for row in selected
            ]
            summary_rows.append(
                {
                    "mode": mode,
                    "method": method,
                    "n": len(selected),
                    "mAP_mean": mean(maps),
                    "mAP_std_sample": sample_std(maps),
                    "delta_vs_same_method_paired": mean(paired_deltas),
                    "Final_minus_Plain_BCE": mean(method_deltas),
                    "Final_wins_vs_Plain_BCE": sum(
                        value > 0 for value in method_deltas
                    ),
                    "weak_class_mAP_mean": mean(weak_maps),
                    "weak_class_mAP_std_sample": sample_std(weak_maps),
                }
            )

    summary_path = root / "final_view_ablation_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(
        f"FINAL_VIEW_ABLATION_SUMMARY_OK rows={len(rows)} "
        f"seeds={args.expected_seeds}"
    )


if __name__ == "__main__":
    main()
