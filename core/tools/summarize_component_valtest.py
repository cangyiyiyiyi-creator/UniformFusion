import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


WEAK_CLASSES = {"Knife", "Scissors", "Lighter", "Razor_blade"}
DEFAULT_METHODS = (
    "Plain_BCE",
    "P9_Full",
    "P9_NoCounterfactualExperts",
    "P9_NoAnchorFloor",
    "P9_NoGuardLoss",
    "P9_SingleScaleC5",
)


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_metrics(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload["stats"]
    return payload, float(stats["mAP"]), [float(v) for v in stats["per_class_ap"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--phase", choices=("val", "final"), required=True)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--expected-methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--baseline-method", default="Plain_BCE")
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
    for val_path in sorted(root.glob("*/seed_*/repeat_*/*/val_metrics.json")):
        backbone, seed_part, repeat_part, method, _ = val_path.relative_to(root).parts
        if method not in args.expected_methods:
            continue
        val_payload, val_map, val_per_class = load_metrics(val_path)
        row = {
            "backbone": backbone,
            "seed": int(seed_part.removeprefix("seed_")),
            "repeat": int(repeat_part.removeprefix("repeat_")),
            "method": method,
            "checkpoint": val_payload["checkpoint"],
            "checkpoint_epoch": val_payload.get("checkpoint_epoch", ""),
            "val_mAP": val_map,
            "val_weak_class_mAP": mean([val_per_class[i] for i in weak_indices]),
            "val_list_sha256": val_payload.get("evaluation_list_sha256", ""),
        }
        if args.phase == "final":
            test_path = val_path.with_name("test_metrics.json")
            if not test_path.is_file():
                raise SystemExit(f"missing final test result: {test_path}")
            test_payload, test_map, test_per_class = load_metrics(test_path)
            row.update({
                "test_mAP": test_map,
                "test_weak_class_mAP": mean(
                    [test_per_class[i] for i in weak_indices]
                ),
                "test_list_sha256": test_payload.get("evaluation_list_sha256", ""),
            })
            row.update(
                {f"test_AP_{name}": test_per_class[i] for i, name in enumerate(classes)}
            )
        rows.append(row)
    if not rows:
        raise SystemExit(f"no validation metrics under {root}")

    groups = defaultdict(list)
    for row in rows:
        groups[(row["backbone"], row["method"])].append(row)
    found_methods = {method for _, method in groups}
    missing_methods = set(args.expected_methods) - found_methods
    if missing_methods:
        raise SystemExit(f"missing methods: {sorted(missing_methods)}")
    for key, selected in groups.items():
        if len(selected) != args.expected_seeds:
            raise SystemExit(
                f"{key}: expected {args.expected_seeds} seeds, found {len(selected)}"
            )

    prefix = "component_ablation" if args.phase == "final" else "component_ablation_val"
    detailed_path = root / f"{prefix}_detailed.csv"
    with detailed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline = {
        (row["backbone"], row["seed"]): row
        for row in rows
        if row["method"] == args.baseline_method
    }
    summary_rows = []
    for (backbone, method), selected in sorted(groups.items()):
        val_maps = [row["val_mAP"] for row in selected]
        val_weak = [row["val_weak_class_mAP"] for row in selected]
        val_deltas = [
            row["val_mAP"] - baseline[(backbone, row["seed"])]["val_mAP"]
            for row in selected
        ]
        summary = {
            "backbone": backbone,
            "method": method,
            "n": len(selected),
            "val_mAP_mean": mean(val_maps),
            "val_mAP_std_sample": sample_std(val_maps),
            "val_delta_vs_baseline": mean(val_deltas),
            "val_wins_vs_baseline": sum(delta > 0 for delta in val_deltas),
            "val_weak_class_mAP_mean": mean(val_weak),
        }
        if args.phase == "final":
            test_maps = [row["test_mAP"] for row in selected]
            test_weak = [row["test_weak_class_mAP"] for row in selected]
            test_deltas = [
                row["test_mAP"] - baseline[(backbone, row["seed"])]["test_mAP"]
                for row in selected
            ]
            summary.update({
                "test_mAP_mean": mean(test_maps),
                "test_mAP_std_sample": sample_std(test_maps),
                "test_delta_vs_baseline": mean(test_deltas),
                "test_wins_vs_baseline": sum(delta > 0 for delta in test_deltas),
                "test_weak_class_mAP_mean": mean(test_weak),
            })
        summary_rows.append(summary)

    summary_path = root / f"{prefix}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(
        f"COMPONENT_VALTEST_SUMMARY_OK phase={args.phase} "
        f"rows={len(rows)} groups={len(summary_rows)}"
    )
    print(summary_path)


if __name__ == "__main__":
    main()
