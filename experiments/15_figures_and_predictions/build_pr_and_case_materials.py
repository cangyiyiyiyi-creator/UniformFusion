#!/usr/bin/env python3
"""Build reproducible PR data and deterministic qualitative case selections."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


CONFIGS = {
    "DvXray": {
        "seeds": [930163947, 1786430941, 553800223, 207027553, 1716854429],
        "classes": ["Knife", "Scissors", "Lighter", "Razor_blade"],
    },
    "LDXray": {
        "seeds": [930163947, 1786430941, 553800223],
        "classes": [
            "Green_Liquid",
            "Cylindrical_Orange_Liquid",
            "Cylindrical_Green_Liquid",
            "Nonmetallic_Lighter",
        ],
    },
}
METHODS = ["Plain_BCE", "Uniform_Fusion"]
COLORS = {"Plain_BCE": "#D55E00", "Uniform_Fusion": "#0072B2"}
LABELS = {"Plain_BCE": "Plain-BCE", "Uniform_Fusion": "Uniform Fusion"}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def read_sample_manifest(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_export(root: Path, dataset: str, method: str, seed: int):
    directory = root / dataset / method / f"seed_{seed}"
    manifest = json.loads(
        (directory / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    arrays = np.load(directory / "predictions.npz")
    return directory, manifest, arrays


def interpolate_pr(targets, scores, recall_grid):
    precision, recall, _ = precision_recall_curve(targets, scores)
    order = np.argsort(recall)
    recall_sorted = recall[order]
    precision_sorted = precision[order]
    unique_recall, unique_indices = np.unique(recall_sorted, return_index=True)
    unique_precision = precision_sorted[unique_indices]
    return np.interp(recall_grid, unique_recall, unique_precision)


def choose_case(targets, delta, plain, uniform, used, mode):
    positive = np.flatnonzero(targets > 0)
    positive = np.asarray([index for index in positive if index not in used])
    if positive.size == 0:
        return None, "no_unused_positive_sample"
    if mode == "gain":
        crossing = positive[(plain[positive] < 0.5) & (uniform[positive] >= 0.5)]
        if crossing.size:
            index = int(crossing[np.argmax(delta[crossing])])
            return index, "positive_label_plain_below_0.5_uniform_at_or_above_0.5_max_delta"
        helpful = positive[delta[positive] > 0]
        if helpful.size:
            index = int(helpful[np.argmax(delta[helpful])])
            return index, "positive_label_largest_positive_probability_gain"
        return None, "no_positive_gain_for_positive_label"
    crossing = positive[(plain[positive] >= 0.5) & (uniform[positive] < 0.5)]
    if crossing.size:
        index = int(crossing[np.argmin(delta[crossing])])
        return index, "positive_label_plain_at_or_above_0.5_uniform_below_0.5_min_delta"
    harmful = positive[delta[positive] < 0]
    if harmful.size:
        index = int(harmful[np.argmin(delta[harmful])])
        return index, "positive_label_largest_probability_drop"
    return None, "no_probability_drop_for_positive_label"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--representative-seed", type=int, default=930163947)
    args = parser.parse_args()
    prediction_root = Path(args.prediction_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    global_summary = []
    for dataset, config in CONFIGS.items():
        dataset_pr = output_root / "02_pr_curves" / dataset
        dataset_cases = output_root / "03_success_failure_cases" / dataset
        dataset_pr.mkdir(parents=True, exist_ok=True)
        dataset_cases.mkdir(parents=True, exist_ok=True)

        exports = {method: {} for method in METHODS}
        reference_targets = None
        reference_samples = None
        class_names = None
        ap_rows = []
        for method in METHODS:
            for seed in config["seeds"]:
                directory, manifest, arrays = load_export(
                    prediction_root, dataset, method, seed
                )
                targets = arrays["targets"]
                probabilities = arrays["probabilities"]
                if reference_targets is None:
                    reference_targets = targets.copy()
                    reference_samples = read_sample_manifest(
                        directory / "sample_manifest.csv"
                    )
                    class_names = manifest["class_names"]
                elif not np.array_equal(targets, reference_targets):
                    raise RuntimeError(f"Target order mismatch: {dataset}/{method}/{seed}")
                if manifest["class_names"] != class_names:
                    raise RuntimeError(f"Class order mismatch: {dataset}/{method}/{seed}")
                exports[method][seed] = {
                    "targets": targets,
                    "probabilities": probabilities,
                    "manifest": manifest,
                }
                for class_index, class_name in enumerate(class_names):
                    ap_rows.append(
                        {
                            "dataset": dataset,
                            "method": LABELS[method],
                            "seed": seed,
                            "class_index": class_index,
                            "class_name": class_name,
                            "AP_sklearn": f"{average_precision_score(targets[:, class_index], probabilities[:, class_index]):.10f}",
                            "AP_locked_project": f"{manifest['per_class_ap'][class_index]:.10f}",
                        }
                    )
        write_csv(dataset_pr / "per_seed_per_class_AP.csv", ap_rows)

        recall_grid = np.linspace(0.0, 1.0, 201)
        curve_rows = []
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), constrained_layout=True)
        for axis, target_class in zip(axes.flat, config["classes"]):
            class_index = class_names.index(target_class)
            for method in METHODS:
                seed_curves = []
                aps = []
                for seed in config["seeds"]:
                    item = exports[method][seed]
                    target = item["targets"][:, class_index]
                    score = item["probabilities"][:, class_index]
                    curve = interpolate_pr(target, score, recall_grid)
                    seed_curves.append(curve)
                    aps.append(average_precision_score(target, score))
                seed_curves = np.asarray(seed_curves)
                mean_curve = seed_curves.mean(axis=0)
                std_curve = seed_curves.std(axis=0, ddof=1)
                axis.plot(
                    recall_grid,
                    mean_curve,
                    color=COLORS[method],
                    linewidth=2.0,
                    label=f"{LABELS[method]} (AP={np.mean(aps):.3f})",
                )
                axis.fill_between(
                    recall_grid,
                    np.clip(mean_curve - std_curve, 0, 1),
                    np.clip(mean_curve + std_curve, 0, 1),
                    color=COLORS[method],
                    alpha=0.14,
                    linewidth=0,
                )
                for recall, mean, std in zip(recall_grid, mean_curve, std_curve):
                    curve_rows.append(
                        {
                            "dataset": dataset,
                            "class_name": target_class,
                            "method": LABELS[method],
                            "recall": f"{recall:.6f}",
                            "precision_mean": f"{mean:.10f}",
                            "precision_std_sample": f"{std:.10f}",
                            "AP_mean_sklearn": f"{np.mean(aps):.10f}",
                            "AP_std_sample_sklearn": f"{np.std(aps, ddof=1):.10f}",
                            "n_seeds": len(aps),
                        }
                    )
            axis.set_title(target_class.replace("_", " "), fontsize=10)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1.02)
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
            axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
            axis.legend(frameon=False, fontsize=8, loc="lower left")
        fig.suptitle(f"{dataset}: precision-recall curves across locked seeds", fontsize=12)
        for suffix, kwargs in (
            ("png", {"dpi": 600}),
            ("pdf", {}),
            ("svg", {}),
        ):
            fig.savefig(dataset_pr / f"{dataset}_weak_or_rare_class_PR.{suffix}", **kwargs)
        plt.close(fig)
        write_csv(dataset_pr / "PR_curve_plot_data.csv", curve_rows)

        # Deterministic cases use the first pre-registered shared seed only.
        seed = args.representative_seed
        plain = exports["Plain_BCE"][seed]["probabilities"]
        uniform = exports["Uniform_Fusion"][seed]["probabilities"]
        targets = exports["Plain_BCE"][seed]["targets"]
        used = set()
        selection_rows = []
        raw_dir = dataset_cases / "raw_pairs"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for mode in ("gain", "drop"):
            for target_class in config["classes"]:
                class_index = class_names.index(target_class)
                delta = uniform[:, class_index] - plain[:, class_index]
                sample_index, rule = choose_case(
                    targets[:, class_index],
                    delta,
                    plain[:, class_index],
                    uniform[:, class_index],
                    used,
                    mode,
                )
                if sample_index is None:
                    continue
                used.add(sample_index)
                sample = reference_samples[sample_index]
                case_type = "success" if mode == "gain" else "failure"
                case_id = (
                    f"{dataset}_{case_type}_{safe_name(target_class)}_"
                    f"idx{sample_index:06d}"
                )
                source_a = Path(sample["path_a"])
                source_b = Path(sample["path_b"])
                extension_a = source_a.suffix.lower() or ".png"
                extension_b = source_b.suffix.lower() or ".png"
                archived_a = raw_dir / f"{case_id}_OL{extension_a}"
                archived_b = raw_dir / f"{case_id}_SD{extension_b}"
                shutil.copy2(source_a, archived_a)
                shutil.copy2(source_b, archived_b)
                selection_rows.append(
                    {
                        "case_id": case_id,
                        "dataset": dataset,
                        "case_type": case_type,
                        "selection_rule": rule,
                        "representative_seed": seed,
                        "sample_index": sample_index,
                        "target_class_index": class_index,
                        "target_class": target_class,
                        "target": int(targets[sample_index, class_index]),
                        "plain_probability": f"{plain[sample_index, class_index]:.10f}",
                        "uniform_probability": f"{uniform[sample_index, class_index]:.10f}",
                        "probability_delta": f"{delta[sample_index]:+.10f}",
                        "source_path_a": sample["path_a"],
                        "source_path_b": sample["path_b"],
                        "archived_path_a": str(archived_a.resolve()),
                        "archived_path_b": str(archived_b.resolve()),
                    }
                )
        write_csv(dataset_cases / "selection_manifest.csv", selection_rows)
        policy = f"""# {dataset} qualitative-case selection policy

- Predictions: locked Plain-BCE and Uniform Fusion checkpoints.
- Representative seed fixed before selection: `{seed}`.
- Candidate pool: every positive target instance in the complete locked Test split.
- Success preference: Plain probability < 0.5 and Uniform probability >= 0.5; otherwise the largest positive probability gain.
- Failure preference: Plain probability >= 0.5 and Uniform probability < 0.5; otherwise the largest probability drop.
- One unique sample is selected per predefined target class and direction whenever available.
- No image or model response is generated, edited, or manually substituted.
- A probability-gain case is not automatically a detection claim; the exact rule and scores remain in `selection_manifest.csv`.
"""
        (dataset_cases / "selection_policy.md").write_text(policy, encoding="utf-8")

        dataset_summary = {
            "dataset": dataset,
            "seeds": config["seeds"],
            "classes_visualized": config["classes"],
            "prediction_exports": len(config["seeds"]) * len(METHODS),
            "selected_cases": len(selection_rows),
            "representative_seed": seed,
        }
        (output_root / f"{dataset}_figure_material_summary.json").write_text(
            json.dumps(dataset_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        global_summary.append(dataset_summary)

    (output_root / "figure_material_summary.json").write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PR_AND_CASE_MATERIALS_OK", json.dumps(global_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
