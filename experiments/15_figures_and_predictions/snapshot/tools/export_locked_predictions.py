#!/usr/bin/env python3
"""Export sample-level probabilities from a locked best checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from locked_checkpoint_utils import (
    extract_logits,
    file_sha256,
    load_locked_model,
    read_list_entries,
)

import datasets as D


def append_optional(storage, key, value):
    if value is not None:
        storage[key].append(value.detach().float().cpu().numpy())


def project_average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    """Match engine_finetune._average_precision_score exactly."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--classes-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-metrics-json", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--view-mode",
        default="checkpoint",
        choices=("checkpoint", "paired", "a_only", "b_only", "mismatched"),
        help="Use the locked metrics/checkpoint view mode by default.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "predictions.npz"
    manifest_path = output_dir / "prediction_manifest.json"
    samples_path = output_dir / "sample_manifest.csv"
    if predictions_path.exists() or manifest_path.exists() or samples_path.exists():
        raise FileExistsError(f"Refusing to overwrite prediction export: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, checkpoint, model_args = load_locked_model(
        args.checkpoint, device, force_intermediate=True
    )
    expected = json.loads(Path(args.expected_metrics_json).read_text(encoding="utf-8"))
    locked_view_mode = expected.get(
        "effective_view_mode", getattr(model_args, "view_mode", "paired")
    )
    view_mode = locked_view_mode if args.view_mode == "checkpoint" else args.view_mode
    if view_mode != locked_view_mode:
        raise RuntimeError(
            f"Requested view mode {view_mode!r} does not match locked metrics "
            f"{locked_view_mode!r}"
        )
    num_classes = int(getattr(model_args, "num_classes", 15))
    input_size = int(getattr(model_args, "input_size", 224))
    class_names = D._read_class_names(args.classes_file, num_classes)
    entries = read_list_entries(args.list, num_classes)
    dataset = D.DualViewTxtDataset(
        args.list,
        input_size,
        num_classes,
        train=False,
        class_names=class_names,
        view_mode=view_mode,
    )
    if len(dataset) != len(entries):
        raise RuntimeError("Dataset/list length mismatch")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    collected = {
        key: []
        for key in (
            "targets",
            "logits",
            "base_logits",
            "expert_logits",
            "correction",
            "router_weights",
            "aux_logits",
            "single_logits_a",
            "single_logits_b",
        )
    }
    try:
        bf16_ok = torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    with torch.inference_mode():
        for batch_index, ((view_a, view_b), targets) in enumerate(loader):
            view_a = view_a.to(device, non_blocking=True)
            view_b = view_b.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                output = model(view_a, view_b)
            logits = extract_logits(output)
            collected["targets"].append(targets.float().numpy())
            collected["logits"].append(logits.detach().float().cpu().numpy())
            if isinstance(output, dict):
                base_logits = output.get("logits_base")
                append_optional(collected, "base_logits", base_logits)
                aux = output.get("plain_innovation_aux")
                if isinstance(aux, dict):
                    for source_key, target_key in (
                        ("expert_logits", "expert_logits"),
                        ("correction", "correction"),
                        ("router_weights", "router_weights"),
                        ("aux_logits", "aux_logits"),
                        ("single_logits_a", "single_logits_a"),
                        ("single_logits_b", "single_logits_b"),
                    ):
                        append_optional(collected, target_key, aux.get(source_key))
            if batch_index % 100 == 0 or batch_index + 1 == len(loader):
                print(
                    f"EXPORT_PROGRESS batch={batch_index + 1}/{len(loader)} "
                    f"samples={min((batch_index + 1) * args.batch_size, len(dataset))}",
                    flush=True,
                )

    arrays = {}
    for key, items in collected.items():
        if items:
            arrays[key] = np.concatenate(items, axis=0)
    arrays["probabilities"] = 1.0 / (1.0 + np.exp(-arrays["logits"]))
    arrays["targets"] = arrays["targets"].astype(np.uint8)
    if arrays["targets"].shape != (len(entries), num_classes):
        raise RuntimeError(f"Unexpected target shape: {arrays['targets'].shape}")

    per_class_ap = [
        project_average_precision(
            arrays["probabilities"][:, index], arrays["targets"][:, index]
        )
        for index in range(num_classes)
    ]
    map_value = float(np.mean(per_class_ap))
    expected_map = float(expected["stats"]["mAP"])
    expected_list_hash = expected["evaluation_list_sha256"]
    actual_list_hash = file_sha256(args.list)
    if actual_list_hash != expected_list_hash:
        raise RuntimeError(
            f"Evaluation-list hash mismatch: {actual_list_hash} != {expected_list_hash}"
        )
    # CPU/NumPy and the original Torch evaluator can differ by a few ulps.
    # This tolerance is still four orders below the paper's mAP precision.
    if abs(map_value - expected_map) > 1e-6:
        raise RuntimeError(
            f"Exported mAP does not match locked metrics: {map_value} != {expected_map}"
        )

    np.savez_compressed(predictions_path, **arrays)
    with samples_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["sample_index", "path_a", "path_b"] + class_names
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = {
                "sample_index": entry["sample_index"],
                "path_a": entry["path_a"],
                "path_b": entry["path_b"],
            }
            row.update(dict(zip(class_names, entry["labels"])))
            writer.writerow(row)

    payload = {
        "dataset": args.dataset_name,
        "method": args.method_name,
        "seed": args.seed,
        "view_mode": view_mode,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "evaluation_list": str(Path(args.list).resolve()),
        "evaluation_list_sha256": actual_list_hash,
        "classes_file": str(Path(args.classes_file).resolve()),
        "classes_file_sha256": file_sha256(args.classes_file),
        "samples": len(entries),
        "class_names": class_names,
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
        "mAP": map_value,
        "per_class_ap": per_class_ap,
        "locked_expected_mAP": expected_map,
        "mAP_absolute_error": abs(map_value - expected_map),
        "selection_policy": "none; complete locked test split exported in original order",
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"LOCKED_PREDICTION_EXPORT_OK dataset={args.dataset_name} "
        f"method={args.method_name} seed={args.seed} samples={len(entries)} "
        f"mAP={map_value:.10f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
