#!/usr/bin/env python3
"""Materialize a human-readable sample prediction table from a locked NPZ export."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _format_number(value: float) -> str:
    return format(float(value), ".10g")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir)
    npz_path = prediction_dir / "predictions.npz"
    manifest_path = prediction_dir / "prediction_manifest.json"
    samples_path = prediction_dir / "sample_manifest.csv"
    output_path = (
        Path(args.output)
        if args.output
        else prediction_dir / "sample_predictions.csv"
    )
    for path in (npz_path, manifest_path, samples_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_names = list(manifest["class_names"])
    with np.load(npz_path, allow_pickle=False) as stored:
        targets = np.asarray(stored["targets"])
        logits = np.asarray(stored["logits"])
        probabilities = np.asarray(stored["probabilities"])

    expected_shape = (int(manifest["samples"]), len(class_names))
    for name, array in (
        ("targets", targets),
        ("logits", logits),
        ("probabilities", probabilities),
    ):
        if array.shape != expected_shape:
            raise RuntimeError(f"{name} shape {array.shape} != {expected_shape}")

    with samples_path.open("r", encoding="utf-8-sig", newline="") as handle:
        samples = list(csv.DictReader(handle))
    if len(samples) != expected_shape[0]:
        raise RuntimeError(
            f"sample manifest rows {len(samples)} != {expected_shape[0]}"
        )

    fields = ["sample_index", "path_a", "path_b"]
    fields += [f"target::{name}" for name in class_names]
    fields += [f"probability::{name}" for name in class_names]
    fields += [f"logit::{name}" for name in class_names]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(samples):
            row = {
                "sample_index": sample["sample_index"],
                "path_a": sample["path_a"],
                "path_b": sample["path_b"],
            }
            for class_index, class_name in enumerate(class_names):
                target = int(targets[index, class_index])
                if target != int(float(sample[class_name])):
                    raise RuntimeError(
                        f"label mismatch at sample={index}, class={class_name}"
                    )
                row[f"target::{class_name}"] = target
                row[f"probability::{class_name}"] = _format_number(
                    probabilities[index, class_index]
                )
                row[f"logit::{class_name}"] = _format_number(
                    logits[index, class_index]
                )
            writer.writerow(row)

    print(
        f"SAMPLE_PREDICTION_TABLE_OK path={output_path} "
        f"samples={expected_shape[0]} classes={expected_shape[1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
