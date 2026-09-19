import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


WEAK_CLASSES = {"Knife", "Scissors", "Lighter", "Razor_blade"}


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def read_best_validation_metric(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty training log: {path}")
    best = max(rows, key=lambda row: float(row["val_metric"]))
    return int(best["epoch"]), float(best["val_metric"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--prefix", default="p9_plain_valtest")
    parser.add_argument("--baseline-method", default="Plain_BCE")
    parser.add_argument("--expected-seeds", type=int, required=True)
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
    for path in sorted(root.glob("*/seed_*/repeat_*/*/test_metrics.json")):
        backbone, seed_part, repeat_part, method, _ = path.relative_to(root).parts
        payload = json.loads(path.read_text(encoding="utf-8"))
        stats = payload["stats"]
        per_class = [float(value) for value in stats["per_class_ap"]]
        best_epoch, best_val_map = read_best_validation_metric(
            path.parent / "training_log.csv"
        )
        row = {
            "backbone": backbone,
            "seed": int(seed_part.removeprefix("seed_")),
            "repeat": int(repeat_part.removeprefix("repeat_")),
            "method": method,
            "best_val_epoch": best_epoch,
            "best_val_mAP": best_val_map,
            "test_mAP": float(stats["mAP"]),
            "test_weak_class_mAP": mean([per_class[i] for i in weak_indices]),
            "checkpoint": payload["checkpoint"],
            "test_list_sha256": payload.get("evaluation_list_sha256", ""),
        }
        row.update({f"test_AP_{name}": per_class[i] for i, name in enumerate(classes)})
        rows.append(row)
    if not rows:
        raise SystemExit(f"no completed test results under {root}")

    detailed_path = root / f"{args.prefix}_detailed.csv"
    with detailed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["backbone"], row["method"])].append(row)
    baseline = {
        (row["backbone"], row["seed"]): row
        for row in rows
        if row["method"] == args.baseline_method
    }
    summary_rows = []
    for (backbone, method), selected in sorted(groups.items()):
        if len(selected) != args.expected_seeds:
            raise SystemExit(
                f"{backbone}/{method}: expected {args.expected_seeds} seeds, "
                f"found {len(selected)}"
            )
        val_maps = [row["best_val_mAP"] for row in selected]
        test_maps = [row["test_mAP"] for row in selected]
        weak_maps = [row["test_weak_class_mAP"] for row in selected]
        deltas = [
            row["test_mAP"] - baseline[(backbone, row["seed"])]["test_mAP"]
            for row in selected
        ]
        summary_rows.append({
            "backbone": backbone,
            "method": method,
            "n": len(selected),
            "best_val_mAP_mean": mean(val_maps),
            "best_val_mAP_std_sample": sample_std(val_maps),
            "test_mAP_mean": mean(test_maps),
            "test_mAP_std_sample": sample_std(test_maps),
            "test_delta_vs_baseline": mean(deltas),
            "test_wins_vs_baseline": sum(value > 0 for value in deltas),
            "test_weak_class_mAP_mean": mean(weak_maps),
            "test_weak_class_mAP_std_sample": sample_std(weak_maps),
        })

    summary_path = root / f"{args.prefix}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"VALTEST_GRID_SUMMARY_OK rows={len(rows)} groups={len(summary_rows)}")
    print(summary_path)


if __name__ == "__main__":
    main()
