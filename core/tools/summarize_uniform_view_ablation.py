#!/usr/bin/env python3
"""Summarize locked Uniform Fusion view-mode inference."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


MODES = ("paired", "ol_only", "sd_only", "mismatched")
WEAK_CLASSES = {"Knife", "Scissors", "Lighter", "Razor_blade"}


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
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
    weak_indices = [i for i, name in enumerate(classes) if name in WEAK_CLASSES]
    rows = []
    for mode in MODES:
        for path in sorted((root / "01_视角消融" / mode).glob("seed_*/Uniform_Fusion.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("requested_view_mode") != mode:
                raise RuntimeError(f"View-mode mismatch: {path}")
            if payload.get("evaluation_list_sha256") != args.expected_list_sha256:
                raise RuntimeError(f"Test-list hash mismatch: {path}")
            values = [float(value) for value in payload["stats"]["per_class_ap"]]
            row = {
                "mode": mode,
                "seed": int(path.parent.name.removeprefix("seed_")),
                "mAP": float(payload["stats"]["mAP"]),
                "weak_class_mAP": mean([values[i] for i in weak_indices]),
                "checkpoint_epoch": int(payload["checkpoint_epoch"]),
                "evaluation_list_sha256": payload["evaluation_list_sha256"],
            }
            row.update({f"AP_{name}": values[i] for i, name in enumerate(classes)})
            rows.append(row)

    expected = len(MODES) * args.expected_seeds
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} evaluations, found {len(rows)}")
    detailed = root / "03_汇总" / "UniformFusion_双视角消融_逐seed.csv"
    detailed.parent.mkdir(parents=True, exist_ok=True)
    with detailed.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lookup = {(row["mode"], row["seed"]): row for row in rows}
    summary = []
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        maps = [row["mAP"] for row in selected]
        weak = [row["weak_class_mAP"] for row in selected]
        deltas = [
            row["mAP"] - lookup[("paired", row["seed"])]["mAP"]
            for row in selected
        ]
        summary.append(
            {
                "mode": mode,
                "n": len(selected),
                "mAP_mean": mean(maps),
                "mAP_std_sample": sample_std(maps),
                "delta_vs_paired_mean": mean(deltas),
                "delta_vs_paired_std_sample": sample_std(deltas),
                "wins_vs_paired": sum(delta > 0 for delta in deltas),
                "weak_class_mAP_mean": mean(weak),
                "weak_class_mAP_std_sample": sample_std(weak),
            }
        )
    summary_path = root / "03_汇总" / "UniformFusion_双视角消融_汇总.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"UNIFORM_VIEW_ABLATION_SUMMARY_OK rows={len(rows)}")


if __name__ == "__main__":
    main()
