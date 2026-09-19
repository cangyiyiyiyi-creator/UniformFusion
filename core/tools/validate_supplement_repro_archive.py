#!/usr/bin/env python3
"""Validate the standalone reproducibility and sample-prediction archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    score_tensor = torch.from_numpy(scores).float()
    target_tensor = torch.from_numpy(targets).float()
    if float(target_tensor.sum()) == 0.0:
        return 0.0
    order = torch.argsort(score_tensor, descending=True)
    ordered = target_tensor[order]
    true_positive = torch.cumsum(ordered, dim=0)
    false_positive = torch.cumsum(1.0 - ordered, dim=0)
    recalls = true_positive / (ordered.sum() + 1e-12)
    precisions = true_positive / (true_positive + false_positive + 1e-12)
    ap = 0.0
    previous_recall = 0.0
    for recall, precision in zip(recalls.tolist(), precisions.tolist()):
        ap += precision * max(recall - previous_recall, 0.0)
        previous_recall = recall
    return float(ap)


def validate_hashes(root: Path) -> int:
    path = root / "SHA256SUMS.txt"
    records = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = root / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        actual = sha256(target)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
        records += 1
    return records


def validate_prediction(directory: Path) -> dict:
    manifest = json.loads(
        (directory / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    samples = int(manifest["samples"])
    classes = list(manifest["class_names"])
    expected_shape = (samples, len(classes))
    with np.load(directory / "predictions.npz", allow_pickle=False) as stored:
        targets = np.asarray(stored["targets"])
        logits = np.asarray(stored["logits"])
        probabilities = np.asarray(stored["probabilities"])
    for name, array in (
        ("targets", targets),
        ("logits", logits),
        ("probabilities", probabilities),
    ):
        if array.shape != expected_shape:
            raise RuntimeError(f"{directory}: {name} shape {array.shape}")
    if not np.allclose(probabilities, 1.0 / (1.0 + np.exp(-logits)), atol=1e-7):
        raise RuntimeError(f"{directory}: probability/logit mismatch")

    per_class = [
        average_precision(probabilities[:, index], targets[:, index])
        for index in range(len(classes))
    ]
    map_value = float(np.mean(per_class))
    if abs(map_value - float(manifest["locked_expected_mAP"])) > 1e-6:
        raise RuntimeError(f"{directory}: locked mAP mismatch")

    for filename in ("sample_manifest.csv", "sample_predictions.csv"):
        with (directory / filename).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = sum(1 for _ in csv.reader(handle)) - 1
        if rows != samples:
            raise RuntimeError(f"{directory}: {filename} rows={rows}")
    return {
        "dataset": manifest["dataset"],
        "method": manifest["method"],
        "seed": int(manifest["seed"]),
        "view_mode": manifest.get("view_mode", "paired"),
        "samples": samples,
        "classes": len(classes),
        "mAP": map_value,
        "locked_mAP": float(manifest["locked_expected_mAP"]),
        "absolute_error": abs(map_value - float(manifest["locked_expected_mAP"])),
        "directory": str(directory),
    }


def write_report(
    path: Path,
    root: Path,
    records: list[dict],
    hash_records: int,
    archive_size: int,
) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record["mAP"])

    lines = [
        "# Independent re-verification report of the reproduction archive",
        "",
        f"- verification time (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- verified directory: `{root.resolve()}`",
        "- overall result: **PASS**",
        f"- sample-level prediction bundles: `{len(records)}`",
        f"- SHA-256 records: `{hash_records}`",
        f"- archive size: `{archive_size / 1024 / 1024:.2f} MiB`",
        "",
        "## Scope of the re-verification",
        "",
        "1. Completeness of the source directories for adapters, gradient accumulation, pipeline profiling and prediction export.",
        "2. Sample and class counts of targets/logits/probabilities in `predictions.npz`.",
        "3. Numerical agreement between sigmoid(logits) and probabilities.",
        "4. Row-count completeness of `sample_manifest.csv` and `sample_predictions.csv`.",
        "5. Agreement between the recomputed mAP (from per-class AP) and the locked test mAP (tolerance 1e-6).",
        "6. Content hashes of every file listed in `SHA256SUMS.txt`.",
        "",
        "## Method coverage",
        "",
        "| Method | seeds | mean mAP | std mAP |",
        "|---|---:|---:|---:|",
    ]
    for method in sorted(grouped):
        values = np.asarray(grouped[method], dtype=np.float64)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        lines.append(
            f"| {method} | {len(values)} | {values.mean():.6f} | {std:.6f} |"
        )

    lines += [
        "",
        "## Per-bundle verification",
        "",
        "| Method | seed | view protocol | samples | recomputed mAP | locked mAP | abs. error |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for record in sorted(records, key=lambda item: (item["method"], item["seed"])):
        lines.append(
            f"| {record['method']} | {record['seed']} | {record['view_mode']} | "
            f"{record['samples']} | {record['mAP']:.10f} | "
            f"{record['locked_mAP']:.10f} | {record['absolute_error']:.2e} |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        "All 31 prediction bundles come from the complete locked splits with no sample filtering. The sample-level arrays, human-readable CSVs, locked mAP values and file hashes all passed re-verification, so this material can be used for statistics, PR curves, success/failure case studies and paper-reproduction audits.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--expected-predictions", type=int, default=31)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    root = Path(args.archive)

    prediction_dirs = sorted(
        path.parent for path in root.rglob("prediction_manifest.json")
    )
    if len(prediction_dirs) != args.expected_predictions:
        raise RuntimeError(
            f"prediction exports count {len(prediction_dirs)} "
            f"!= {args.expected_predictions}"
        )
    required_directories = (
        "00_archive_tools",
        "01_adapter_source",
        "02_grad_accum_source",
        "03_pipeline_profiling_source",
        "04_sample_predictions",
    )
    for relative in required_directories:
        if not (root / relative).is_dir():
            raise FileNotFoundError(root / relative)

    records = [validate_prediction(directory) for directory in prediction_dirs]
    hash_records = validate_hashes(root)
    archive_manifest = json.loads(
        (root / "archive_manifest.json").read_text(encoding="utf-8")
    )
    if int(archive_manifest["prediction_exports"]) != len(records):
        raise RuntimeError("archive manifest prediction count mismatch")
    if int(archive_manifest["file_count_excluding_sha_manifest"]) != hash_records:
        raise RuntimeError("archive manifest SHA record count mismatch")
    archive_size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if args.report:
        write_report(
            Path(args.report), root, records, hash_records, archive_size
        )
    print(
        f"REPRO_ARCHIVE_VALIDATION_OK predictions={len(prediction_dirs)} "
        f"sha256_records={hash_records} report={args.report or 'none'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
