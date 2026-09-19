#!/usr/bin/env python3
"""Select the next unused DvXray cases for auditable regional heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np


TARGET_CLASSES = ("Knife", "Scissors", "Lighter", "Razor_blade")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No additional cases were selected")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def ranked_candidates(
    targets: np.ndarray,
    plain: np.ndarray,
    uniform: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, str]:
    positive = np.flatnonzero(targets > 0)
    delta = uniform - plain
    if mode == "gain":
        crossing = positive[(plain[positive] < 0.5) & (uniform[positive] >= 0.5)]
        if crossing.size:
            return crossing[np.argsort(-delta[crossing])], (
                "positive_label_plain_below_0.5_uniform_at_or_above_0.5_"
                "next_unused_delta"
            )
        helpful = positive[delta[positive] > 0]
        return helpful[np.argsort(-delta[helpful])], (
            "positive_label_next_unused_positive_probability_gain"
        )
    crossing = positive[(plain[positive] >= 0.5) & (uniform[positive] < 0.5)]
    if crossing.size:
        return crossing[np.argsort(delta[crossing])], (
            "positive_label_plain_at_or_above_0.5_uniform_below_0.5_"
            "next_unused_delta"
        )
    harmful = positive[delta[positive] < 0]
    return harmful[np.argsort(delta[harmful])], (
        "positive_label_next_unused_probability_drop"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain-prediction-dir", required=True)
    parser.add_argument("--uniform-prediction-dir", required=True)
    parser.add_argument("--existing-selection-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--representative-seed", type=int, default=930163947)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    raw_dir = output_dir / "raw_pairs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    plain_dir = Path(args.plain_prediction_dir)
    uniform_dir = Path(args.uniform_prediction_dir)
    plain_manifest = json.loads(
        (plain_dir / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    uniform_manifest = json.loads(
        (uniform_dir / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    if plain_manifest["class_names"] != uniform_manifest["class_names"]:
        raise RuntimeError("Plain/Uniform class order mismatch")
    if plain_manifest["evaluation_list_sha256"] != uniform_manifest[
        "evaluation_list_sha256"
    ]:
        raise RuntimeError("Plain/Uniform evaluation-list hash mismatch")
    class_names = list(plain_manifest["class_names"])

    with np.load(plain_dir / "predictions.npz", allow_pickle=False) as stored:
        plain_targets = np.asarray(stored["targets"])
        plain_probabilities = np.asarray(stored["probabilities"])
    with np.load(uniform_dir / "predictions.npz", allow_pickle=False) as stored:
        uniform_targets = np.asarray(stored["targets"])
        uniform_probabilities = np.asarray(stored["probabilities"])
    if not np.array_equal(plain_targets, uniform_targets):
        raise RuntimeError("Plain/Uniform target order mismatch")

    samples = read_csv(plain_dir / "sample_manifest.csv")
    if len(samples) != plain_targets.shape[0]:
        raise RuntimeError("Sample manifest length mismatch")
    existing = read_csv(Path(args.existing_selection_manifest))
    used = {int(row["sample_index"]) for row in existing}
    rows = []

    for mode in ("gain", "drop"):
        for target_class in TARGET_CLASSES:
            class_index = class_names.index(target_class)
            ranked, rule = ranked_candidates(
                plain_targets[:, class_index],
                plain_probabilities[:, class_index],
                uniform_probabilities[:, class_index],
                mode,
            )
            sample_index = next(
                (int(index) for index in ranked if int(index) not in used), None
            )
            if sample_index is None:
                raise RuntimeError(
                    f"No unused {mode} case for class {target_class}"
                )
            used.add(sample_index)
            sample = samples[sample_index]
            case_type = "success" if mode == "gain" else "failure"
            case_id = (
                f"DvXray_{case_type}_{safe_name(target_class)}_"
                f"idx{sample_index:06d}_additional"
            )
            source_a = Path(sample["path_a"])
            source_b = Path(sample["path_b"])
            if not source_a.is_file() or not source_b.is_file():
                raise FileNotFoundError(f"Missing source pair: {source_a}, {source_b}")
            archived_a = raw_dir / f"{case_id}_OL{source_a.suffix.lower() or '.png'}"
            archived_b = raw_dir / f"{case_id}_SD{source_b.suffix.lower() or '.png'}"
            shutil.copy2(source_a, archived_a)
            shutil.copy2(source_b, archived_b)
            delta = (
                uniform_probabilities[sample_index, class_index]
                - plain_probabilities[sample_index, class_index]
            )
            rows.append(
                {
                    "case_id": case_id,
                    "dataset": "DvXray",
                    "case_type": case_type,
                    "selection_rule": rule,
                    "representative_seed": args.representative_seed,
                    "sample_index": sample_index,
                    "target_class_index": class_index,
                    "target_class": target_class,
                    "target": int(plain_targets[sample_index, class_index]),
                    "plain_probability": f"{plain_probabilities[sample_index, class_index]:.10f}",
                    "uniform_probability": f"{uniform_probabilities[sample_index, class_index]:.10f}",
                    "probability_delta": f"{delta:+.10f}",
                    "source_path_a": str(source_a),
                    "source_path_b": str(source_b),
                    "archived_path_a": str(archived_a.resolve()),
                    "archived_path_b": str(archived_b.resolve()),
                }
            )

    write_csv(output_dir / "selection_manifest.csv", rows)
    policy = f"""# Additional DvXray qualitative-case selection policy

- Representative seed: `{args.representative_seed}`.
- Candidate pool: every positive target instance in the complete locked Test split.
- Classes: Knife, Scissors, Lighter, Razor_blade.
- One additional success and one additional failure are selected per class.
- All sample indices already present in the original selection manifest are excluded.
- Ranking follows the original threshold-crossing rule, then probability delta.
- No image or model response is generated, edited, or manually substituted.
"""
    (output_dir / "selection_policy.md").write_text(policy, encoding="utf-8")
    print(
        f"ADDITIONAL_DVXRAY_SELECTION_OK cases={len(rows)} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
