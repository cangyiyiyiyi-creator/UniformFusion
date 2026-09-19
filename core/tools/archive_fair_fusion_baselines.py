#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs_fair_fusion_baselines/run_20260903_fair_fusion_3seeds"
ARCHIVE = ROOT / "论文最终归档_20260830"
TARGET = ARCHIVE / "15_公平融合基线_20260904"
TABLE_DIR = ARCHIVE / "06_最终论文资料_20260831/04_论文表格"
SEEDS = (930163947, 1786430941, 553800223)
REPEATS = {930163947: 1, 1786430941: 2, 553800223: 3}
METHODS = {
    "Mean_Fusion": "Mean Fusion",
    "Max_Fusion": "Max Fusion",
    "Concat_Fusion": "Concat Fusion",
    "CrossAttention_Fusion": "Cross-Attention Fusion",
}
WEAK_INDICES = (1, 4, 5, 8)


def load_metrics(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload["stats"]
    aps = [float(value) for value in stats["per_class_ap"]]
    return {
        "mAP": float(stats["mAP"]),
        "weak_mAP": statistics.mean(aps[index] for index in WEAK_INDICES),
        "params_M": float(payload["total_params"]) / 1e6,
        "checkpoint_epoch": int(payload["checkpoint_epoch"]),
        "samples": int(payload["samples"]),
        "test_sha256": payload["evaluation_list_sha256"],
        "per_class_ap": aps,
    }


def reference_metrics(kind, seed):
    if kind == "Plain-BCE":
        directory = ARCHIVE / f"03_基线与消融模型/Plain_BCE__ResNet50__seed_{seed}"
    else:
        directory = ARCHIVE / f"02_UniformFusion主方法模型/UniformFusion__ResNet50__seed_{seed}"
    return load_metrics(directory / "test_metrics.json")


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    detailed = []
    by_method = {}
    expected_hash = None
    for method_dir, method in METHODS.items():
        rows = []
        for seed in SEEDS:
            repeat = REPEATS[seed]
            directory = SOURCE / f"resnet50/seed_{seed}/repeat_{repeat}/{method_dir}"
            metrics = load_metrics(directory / "test_metrics.json")
            if metrics["samples"] != 1600:
                raise ValueError(f"unexpected Test sample count: {directory}")
            if expected_hash is None:
                expected_hash = metrics["test_sha256"]
            if metrics["test_sha256"] != expected_hash:
                raise ValueError(f"Test split hash mismatch: {directory}")
            row = {
                "method": method,
                "backbone": "ResNet50",
                "seed": seed,
                "repeat": repeat,
                "best_val_epoch": metrics["checkpoint_epoch"],
                "test_mAP": f'{metrics["mAP"]:.10f}',
                "test_weak_class_mAP": f'{metrics["weak_mAP"]:.10f}',
                "params_M": f'{metrics["params_M"]:.6f}',
                "test_samples": metrics["samples"],
                "checkpoint": str(directory / "checkpoint_best.pth"),
            }
            detailed.append(row)
            rows.append(metrics)
            evidence_dir = TARGET / "01_原始证据" / f"seed_{seed}" / method_dir
            for name in (
                "checkpoint_best.pth",
                "training_log.csv",
                "test_metrics.json",
                "test_metrics.csv",
                "test.log",
                "training_complete.marker",
            ):
                source_file = directory / name
                if source_file.exists():
                    link_or_copy(source_file, evidence_dir / name)
        by_method[method] = rows

    summary = []
    paired = []
    references = {
        name: [reference_metrics(name, seed) for seed in SEEDS]
        for name in ("Plain-BCE", "Uniform Fusion")
    }
    all_groups = {**references, **by_method}
    for method, rows in all_groups.items():
        maps = [row["mAP"] for row in rows]
        weak = [row["weak_mAP"] for row in rows]
        mean, std = mean_std(maps)
        summary.append({
            "method": method,
            "backbone": "ResNet50",
            "n": len(rows),
            "test_mAP_mean": f"{mean:.10f}",
            "test_mAP_std": f"{std:.10f}",
            "test_weak_class_mAP": f"{statistics.mean(weak):.10f}",
            "params_M": f'{rows[0]["params_M"]:.6f}',
        })

    uniform = [row["mAP"] for row in references["Uniform Fusion"]]
    for method, rows in {"Plain-BCE": references["Plain-BCE"], **by_method}.items():
        values = [row["mAP"] for row in rows]
        deltas = [u - value for u, value in zip(uniform, values)]
        delta_mean, delta_std = mean_std(deltas)
        paired.append({
            "comparison": f"Uniform Fusion - {method}",
            "n_common_seeds": len(SEEDS),
            "mean_delta_mAP": f"{delta_mean:.10f}",
            "std_delta_mAP": f"{delta_std:.10f}",
            "uniform_wins": f"{sum(delta > 0 for delta in deltas)}/{len(deltas)}",
            "seed_deltas": ";".join(f"{seed}:{delta:.10f}" for seed, delta in zip(SEEDS, deltas)),
        })

    write_csv(TARGET / "fair_fusion_3seed_detailed.csv", list(detailed[0]), detailed)
    write_csv(TARGET / "fair_fusion_3seed_summary.csv", list(summary[0]), summary)
    write_csv(TARGET / "uniform_vs_fair_fusion_paired_3seed.csv", list(paired[0]), paired)
    (TARGET / "protocol.txt").write_text(
        "Dataset=DvXray\nBackbone=ResNet50\nTrain=train split\n"
        "Checkpoint selection=best validation mAP\nFinal evaluation=locked Test split once\n"
        f"Test samples=1600\nTest list SHA256={expected_hash}\n"
        f"Seeds={','.join(map(str, SEEDS))}\nWeak classes=Knife,Scissors,Lighter,Razor_blade\n",
        encoding="utf-8",
    )

    table8 = TABLE_DIR / "表8_DvXray公平主对比_已有结果.csv"
    with table8.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = handle.readline() if False else list(rows[0])
    lookup = {row["method"]: row for row in summary}
    plain3 = [row["mAP"] for row in references["Plain-BCE"]]
    for row in rows:
        if row["method"] not in lookup or row["method"] in ("Plain-BCE", "Uniform Fusion"):
            continue
        result = lookup[row["method"]]
        method_values = [item["mAP"] for item in by_method[row["method"]]]
        deltas = [value - plain for value, plain in zip(method_values, plain3)]
        row.update({
            "n": result["n"],
            "test_mAP_mean": result["test_mAP_mean"],
            "test_mAP_std": result["test_mAP_std"],
            "delta_vs_Plain": f'{statistics.mean(deltas):.10f}',
            "wins_vs_Plain": f'{sum(delta > 0 for delta in deltas)}/{len(deltas)}',
            "test_weak_class_mAP": result["test_weak_class_mAP"],
            "params_M": result["params_M"],
            "status": "complete_locked_test",
        })
    write_csv(table8, fields, rows)

    table6 = TABLE_DIR / "表6_公开SOTA与公平基线_协议分层对比.csv"
    with table6.open(newline="", encoding="utf-8") as handle:
        rows6 = list(csv.DictReader(handle))
        fields6 = list(rows6[0])
    existing = {row["method"] for row in rows6}
    for result in summary:
        if result["method"] in existing or result["method"] in ("Plain-BCE", "Uniform Fusion"):
            continue
        rows6.append({
            "evidence_layer": "locked_fair_fusion_baseline",
            "method": result["method"],
            "backbone_or_publication": "ResNet50",
            "dataset": "DvXray",
            "metric": "Test mAP",
            "result": f'{float(result["test_mAP_mean"]):.4f} ± {float(result["test_mAP_std"]):.4f}',
            "n": result["n"],
            "directly_comparable_to_latest_test": "True",
            "source_or_note": "same locked Train/Val/Test protocol; three fixed seeds",
        })
    write_csv(table6, fields6, rows6)

    code_dir = TARGET / "02_协议与代码"
    for source_file in (
        ROOT / "run_fair_fusion_baselines_3seeds.sh",
        ROOT / "run_fair_fusion_baselines_test_3seeds.sh",
        ROOT / "tools/evaluate_project_checkpoint.py",
        ROOT / "tools/verify_checkpoint_protocol.py",
        ROOT / "tools/archive_fair_fusion_baselines.py",
        ROOT / "main_finetune.py",
        ROOT / "models/convnextv2_dual.py",
    ):
        target_file = code_dir / source_file.relative_to(ROOT)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)

    manifest_rows = []
    for path in sorted(TARGET.rglob("*")):
        if path.is_file() and path.name != "文件清单与SHA256.csv":
            manifest_rows.append({
                "file": str(path.relative_to(TARGET)),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    write_csv(
        TARGET / "文件清单与SHA256.csv",
        ("file", "size_bytes", "sha256"),
        manifest_rows,
    )
    print(f"ARCHIVE_OK {TARGET}")


if __name__ == "__main__":
    main()
