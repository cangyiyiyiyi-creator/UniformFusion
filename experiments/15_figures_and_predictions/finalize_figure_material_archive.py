#!/usr/bin/env python3
"""Validate and index the non-fabricated paper-figure evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    prediction_manifests = sorted(root.glob("01_sample_predictions/*/*/seed_*/prediction_manifest.json"))
    if len(prediction_manifests) != 16:
        raise RuntimeError(f"Expected 16 prediction exports, found {len(prediction_manifests)}")
    manifest_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in prediction_manifests]
    max_map_error = max(item["mAP_absolute_error"] for item in manifest_payloads)
    if max_map_error > 1e-6:
        raise RuntimeError(f"Locked mAP verification failed: max error={max_map_error}")

    case_counts = {}
    heatmap_counts = {}
    for dataset in ("DvXray", "LDXray"):
        selections = read_csv(root / "03_success_failure_cases" / dataset / "selection_manifest.csv")
        heatmaps = read_csv(root / "04_region_heatmaps" / dataset / "heatmap_manifest.csv")
        if len(selections) != len(heatmaps):
            raise RuntimeError(
                f"Selection/heatmap count mismatch for {dataset}: "
                f"{len(selections)} != {len(heatmaps)}"
            )
        case_counts[dataset] = len(selections)
        heatmap_counts[dataset] = len(heatmaps)
        for row in selections:
            if not Path(row["archived_path_a"]).is_file() or not Path(row["archived_path_b"]).is_file():
                raise FileNotFoundError(f"Missing archived raw pair for {row['case_id']}")

    readme = f"""# Real visualisation and PR-curve paper materials

This directory was produced by pure inference on the locked test split with the locked `checkpoint_best.pth`; no retraining and no model modification took place.

## Contents

- `01_sample_predictions/`: complete test targets, logits, probabilities and expert evidence for Plain/Uniform on the 5 DvXray seeds and the 3 LDXray seeds.
- `02_pr_curves/`: multi-seed mean PR curves for weak/rare classes, standard-deviation bands, SVG/PDF/PNG and the full plotting CSVs.
- `03_success_failure_cases/`: real paired source images selected by pre-declared fixed rules, together with selection scores and provenance paths.
- `04_region_heatmaps/`: real C4/C5 class-query Top-K selection weights, four-expert evidence, raw NPZ files and typeset-ready figures.
- `05_generation_code/`: all standalone inference and plotting code used for this material bundle.

## Integrity

- locked prediction exports: 16/16.
- DvXray objective case selection / heat maps: {case_counts['DvXray']}/{heatmap_counts['DvXray']}.
- LDXray objective case selection / heat maps: {case_counts['LDXray']}/{heatmap_counts['LDXray']}.
- maximum absolute error between the exported mAP and the original locked test metric: `{max_map_error:.3e}`.

## Scope of the claims

1. The heat maps show class-query Top-K selection evidence; they are neither Grad-CAM nor object bounding boxes.
2. `success` and `failure` follow a fixed probability rule in the selection manifest and must not be read independently of the targets and scores.
3. The shaded area of the PR curves is the standard deviation across fixed seeds, not a confidence interval.
4. The source images were copied from the official/locked local data paths and were neither AI-generated nor edited.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    manifest_path = root / "file_manifest_and_sha256.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest_path:
            rows.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"FIGURE_MATERIAL_ARCHIVE_OK predictions={len(prediction_manifests)} "
        f"cases={case_counts} files={len(rows) + 1} max_mAP_error={max_map_error:.3e}"
    )


if __name__ == "__main__":
    main()
