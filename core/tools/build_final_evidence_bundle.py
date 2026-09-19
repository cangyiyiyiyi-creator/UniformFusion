#!/usr/bin/env python3
"""Collect completed locked experiment summaries into one evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_marker(root: Path, marker: str) -> None:
    if not (root / marker).is_file():
        raise FileNotFoundError(f"missing completion marker: {root / marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet-root", required=True)
    parser.add_argument("--convnext-root", required=True)
    parser.add_argument("--view-root", required=True)
    parser.add_argument("--efficiency-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    roots = {
        "resnet_n5": Path(args.resnet_root),
        "convnext_3seeds": Path(args.convnext_root),
        "view_ablation_test_n5": Path(args.view_root),
        "efficiency": Path(args.efficiency_root),
    }
    require_marker(roots["resnet_n5"], "confirmation_complete.marker")
    require_marker(roots["convnext_3seeds"], "generalization_complete.marker")
    require_marker(roots["view_ablation_test_n5"], "evaluation_complete.marker")
    require_marker(roots["efficiency"], "efficiency_complete.marker")

    files = {
        "resnet_summary.csv": roots["resnet_n5"]
        / "final_noanchor_resnet50_n5_summary.csv",
        "resnet_detailed.csv": roots["resnet_n5"]
        / "final_noanchor_resnet50_n5_detailed.csv",
        "resnet_paired_statistics.csv": roots["resnet_n5"]
        / "paired_statistics.csv",
        "resnet_paired_statistics.json": roots["resnet_n5"]
        / "paired_statistics.json",
        "convnext_summary.csv": roots["convnext_3seeds"]
        / "final_noanchor_convnextv2_3seeds_summary.csv",
        "convnext_detailed.csv": roots["convnext_3seeds"]
        / "final_noanchor_convnextv2_3seeds_detailed.csv",
        "convnext_paired_statistics.csv": roots["convnext_3seeds"]
        / "paired_statistics.csv",
        "convnext_paired_statistics.json": roots["convnext_3seeds"]
        / "paired_statistics.json",
        "view_summary.csv": roots["view_ablation_test_n5"]
        / "final_view_ablation_summary.csv",
        "view_detailed.csv": roots["view_ablation_test_n5"]
        / "final_view_ablation_detailed.csv",
        "efficiency_summary.csv": roots["efficiency"] / "efficiency_summary.csv",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing final evidence file: {path}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for archive_name, source in files.items():
        destination = output_root / archive_name
        shutil.copy2(source, destination)
        manifest.append(
            {
                "file": archive_name,
                "source": str(source),
                "sha256": sha256(destination),
            }
        )

    combined_rows = []
    for evidence_name in ("resnet_summary.csv", "convnext_summary.csv"):
        with (output_root / evidence_name).open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            for row in csv.DictReader(handle):
                combined_rows.append({"evidence": evidence_name, **row})
    with (output_root / "all_backbone_performance.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)

    (output_root / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "locked method; val selection; test reporting",
                "main_method": "Final_NoAnchor",
                "baseline": "Plain_BCE",
                "artifacts": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "suite_complete.marker").touch()
    print(f"FINAL_EVIDENCE_BUNDLE_OK files={len(files)} root={output_root}")


if __name__ == "__main__":
    main()
