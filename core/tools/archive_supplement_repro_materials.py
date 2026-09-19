#!/usr/bin/env python3
"""Collect reproducibility source and existing locked predictions in one archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / "supplementary_verification_20260907"
ARCHIVE = SUPPLEMENT / "07_repro_source_and_sample_predictions"

SOURCE_FILES = {
    "00_archive_tools/tools/archive_supplement_repro_materials.py": ROOT
    / "tools/archive_supplement_repro_materials.py",
    "00_archive_tools/tools/validate_supplement_repro_archive.py": ROOT
    / "tools/validate_supplement_repro_archive.py",
    "01_adapter_source/models/fair_baseline_adapters.py": ROOT
    / "models/fair_baseline_adapters.py",
    "01_adapter_source/main_finetune.py": ROOT / "main_finetune.py",
    "01_adapter_source/run_fair_adapter_one.sh": ROOT
    / "run_fair_adapter_one.sh",
    "01_adapter_source/run_fair_adapters_3seeds.sh": ROOT
    / "run_fair_adapters_3seeds.sh",
    "01_adapter_source/run_dagnet_batch32_recheck_3seeds.sh": ROOT
    / "run_dagnet_batch32_recheck_3seeds.sh",
    "01_adapter_source/third_party/DAGNet/model/model_v2.py": ROOT
    / "third_party/DAGNet_official/model/model_v2.py",
    "01_adapter_source/third_party/DAGNet/module/CAFM.py": ROOT
    / "third_party/DAGNet_official/module/CAFM.py",
    "01_adapter_source/third_party/DAGNet/module/CBAM.py": ROOT
    / "third_party/DAGNet_official/module/CBAM.py",
    "01_adapter_source/third_party/DAGNet/module/ConvNormLayer.py": ROOT
    / "third_party/DAGNet_official/module/ConvNormLayer.py",
    "01_adapter_source/third_party/DAGNet/module/FDIM.py": ROOT
    / "third_party/DAGNet_official/module/FDIM.py",
    "01_adapter_source/third_party/DAGNet/module/MSCFE.py": ROOT
    / "third_party/DAGNet_official/module/MSCFE.py",
    "01_adapter_source/third_party/DAGNet/module/utils.py": ROOT
    / "third_party/DAGNet_official/module/utils.py",
    "01_adapter_source/third_party/ML_Decoder/ml_decoder.py": ROOT
    / "third_party/ML_Decoder_official/src_files/ml_decoder/ml_decoder.py",
    "02_grad_accum_source/main_finetune.py": ROOT / "main_finetune.py",
    "02_grad_accum_source/engine_finetune.py": ROOT / "engine_finetune.py",
    "02_grad_accum_source/run_dagnet_batch32_recheck_3seeds.sh": ROOT
    / "run_dagnet_batch32_recheck_3seeds.sh",
    "03_pipeline_profiling_source/tools/profile_project_checkpoint.py": ROOT
    / "tools/profile_project_checkpoint.py",
    "03_pipeline_profiling_source/tools/profile_end_to_end_checkpoint.py": ROOT
    / "tools/profile_end_to_end_checkpoint.py",
    "03_pipeline_profiling_source/tools/summarize_latency_sessions.py": ROOT
    / "tools/summarize_latency_sessions.py",
    "03_pipeline_profiling_source/run_final_efficiency_5sessions.sh": ROOT
    / "run_final_efficiency_5sessions.sh",
    "04_sample_predictions/tools/export_locked_predictions.py": ROOT
    / "tools/export_locked_predictions.py",
    "04_sample_predictions/tools/locked_checkpoint_utils.py": ROOT
    / "tools/locked_checkpoint_utils.py",
    "04_sample_predictions/tools/materialize_sample_predictions_csv.py": ROOT
    / "tools/materialize_sample_predictions_csv.py",
    "04_sample_predictions/datasets.py": ROOT / "datasets.py",
    "04_sample_predictions/run_export_supplement_predictions.sh": ROOT
    / "run_export_supplement_predictions.sh",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def copy_sources() -> None:
    for relative, source in SOURCE_FILES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = ARCHIVE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    commits = {
        "DAGNet_official": {
            "commit": git_commit(ROOT / "third_party/DAGNet_official"),
            "adapter_expected_commit": "ab4e3ff202af5328eedb61c8953c4069e0bf8fee",
        },
        "ML_Decoder_official": {
            "commit": git_commit(ROOT / "third_party/ML_Decoder_official"),
            "adapter_expected_commit": "8a9e984f671c9c30c98d2c45dfcaf4383381c254",
        },
    }
    path = ARCHIVE / "01_adapter_source/third_party_locked_commits.json"
    path.write_text(
        json.dumps(commits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def copy_existing_predictions() -> None:
    source = (
        ROOT
        / "paper_archive_20260830"
        / "08_visualisation_and_pr_curves_20260902"
        / "01_sample_predictions"
        / "DvXray"
    )
    destination = ARCHIVE / "04_sample_predictions/DvXray"
    for method in ("Plain_BCE", "Uniform_Fusion"):
        method_source = source / method
        if not method_source.is_dir():
            raise FileNotFoundError(method_source)
        shutil.copytree(method_source, destination / method, dirs_exist_ok=True)


def materialize_tables(python: str) -> None:
    tool = ROOT / "tools/materialize_sample_predictions_csv.py"
    prediction_root = ARCHIVE / "04_sample_predictions"
    for npz_path in sorted(prediction_root.rglob("predictions.npz")):
        output = npz_path.parent / "sample_predictions.csv"
        if output.is_file():
            continue
        subprocess.run(
            [python, str(tool), "--prediction-dir", str(npz_path.parent)],
            check=True,
        )


def write_prediction_index() -> None:
    output = ARCHIVE / "04_sample_predictions/prediction_index.csv"
    fields = [
        "dataset",
        "method",
        "seed",
        "view_mode",
        "samples",
        "mAP",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "evaluation_list_sha256",
        "relative_directory",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for manifest_path in sorted(ARCHIVE.rglob("prediction_manifest.json")):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            writer.writerow(
                {
                    "dataset": payload["dataset"],
                    "method": payload["method"],
                    "seed": payload["seed"],
                    "view_mode": payload.get("view_mode", "paired"),
                    "samples": payload["samples"],
                    "mAP": format(float(payload["mAP"]), ".10f"),
                    "checkpoint_epoch": payload["checkpoint_epoch"],
                    "checkpoint_sha256": payload["checkpoint_sha256"],
                    "evaluation_list_sha256": payload["evaluation_list_sha256"],
                    "relative_directory": manifest_path.parent.relative_to(
                        ARCHIVE
                    ).as_posix(),
                }
            )


def write_manifest() -> None:
    sums = ARCHIVE / "SHA256SUMS.txt"
    manifest_path = ARCHIVE / "archive_manifest.json"
    material_files = [
        path
        for path in sorted(ARCHIVE.rglob("*"))
        if path.is_file() and path not in (sums, manifest_path)
    ]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(ARCHIVE),
        "file_count_excluding_sha_manifest": len(material_files) + 1,
        "prediction_exports": len(list(ARCHIVE.rglob("predictions.npz"))),
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    records = []
    for path in sorted(ARCHIVE.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            records.append((sha256(path), path.relative_to(ARCHIVE).as_posix()))
    sums.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python")
    args = parser.parse_args()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    copy_sources()
    copy_existing_predictions()
    materialize_tables(args.python)
    write_prediction_index()
    write_manifest()
    print(f"REPRO_ARCHIVE_OK path={ARCHIVE}", flush=True)


if __name__ == "__main__":
    main()
