#!/usr/bin/env python3
"""Export real class-query Top-K region evidence for preselected test cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from locked_checkpoint_utils import file_sha256, load_locked_model

import datasets as D


EXPERT_NAMES = ["Paired base", "OL-only", "SD-only", "Regional pair"]
EXPERT_COLORS = ["#666666", "#009E73", "#56B4E9", "#CC79A7"]


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def topk_evidence_map(encoder, features, level, class_index):
    projected = encoder.projections[level](features[level])
    _, _, height, width = projected.shape
    tokens = projected.flatten(2).transpose(1, 2)
    normalized_tokens = F.normalize(tokens, dim=-1)
    query = F.normalize(encoder.class_queries[class_index], dim=-1)
    scores = torch.einsum("bnd,d->bn", normalized_tokens, query)
    scores = scores / max(float(encoder.temperature), 1e-6)
    keep = min(int(encoder.topk), scores.shape[-1])
    values, indices = scores.topk(keep, dim=-1)
    weights = values.float().softmax(dim=-1)
    evidence = torch.zeros_like(scores, dtype=torch.float32)
    evidence.scatter_(1, indices, weights)
    return {
        "evidence": evidence[0].reshape(height, width).detach().cpu().numpy(),
        "similarity": scores[0].reshape(height, width).detach().float().cpu().numpy(),
        "topk_indices": indices[0].detach().cpu().numpy(),
        "topk_weights": weights[0].detach().cpu().numpy(),
        "feature_height": height,
        "feature_width": width,
    }


def resize_heatmap(heatmap, size):
    if float(heatmap.max()) <= 0:
        return np.zeros((size[1], size[0]), dtype=np.float32)
    normalized = heatmap / float(heatmap.max())
    image = Image.fromarray(np.uint8(np.clip(normalized, 0, 1) * 255), mode="L")
    image = image.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def show_overlay(axis, image, heatmap, title):
    image_array = np.asarray(image)
    resized = resize_heatmap(heatmap, image.size)
    axis.imshow(image_array)
    masked = np.ma.masked_where(resized <= 0.015, resized)
    axis.imshow(masked, cmap="magma", vmin=0, vmax=1, alpha=np.clip(resized * 0.72, 0, 0.72))
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--classes-file", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--plain-predictions", required=True)
    parser.add_argument("--uniform-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite heatmap directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, checkpoint, model_args = load_locked_model(
        args.checkpoint, device, force_intermediate=True
    )
    head = getattr(model, "plain_innovation_head", None)
    if head is None or not hasattr(head, "encoder"):
        raise RuntimeError("Uniform checkpoint does not expose the P9 region encoder")
    encoder = head.encoder
    levels = tuple(encoder.levels)
    if not {"C4", "C5"}.issubset(levels):
        raise RuntimeError(f"Expected C4/C5 region levels, got {levels}")

    num_classes = int(getattr(model_args, "num_classes", 15))
    input_size = int(getattr(model_args, "input_size", 224))
    class_names = D._read_class_names(args.classes_file, num_classes)
    dataset = D.DualViewTxtDataset(
        args.list,
        input_size,
        num_classes,
        train=False,
        class_names=class_names,
        view_mode="paired",
    )
    selections = read_csv(Path(args.selection_manifest))
    plain_export = np.load(args.plain_predictions)
    uniform_export = np.load(args.uniform_predictions)
    if not np.array_equal(plain_export["targets"], uniform_export["targets"]):
        raise RuntimeError("Plain/Uniform prediction target order mismatch")

    try:
        bf16_ok = torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    manifest_rows = []

    for case_number, selection in enumerate(selections, start=1):
        sample_index = int(selection["sample_index"])
        class_index = int(selection["target_class_index"])
        class_name = selection["target_class"]
        case_id = selection["case_id"]
        (view_a, view_b), target = dataset[sample_index]
        view_a_batch = view_a.unsqueeze(0).to(device)
        view_b_batch = view_b.unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            output = model(view_a_batch, view_b_batch)
            if not isinstance(output, dict) or "feats" not in output:
                raise RuntimeError("Model did not return intermediate feature dictionaries")
            aux = output.get("plain_innovation_aux")
            if not isinstance(aux, dict) or aux.get("expert_logits") is None:
                raise RuntimeError("Model did not return Uniform expert evidence")

            maps = {}
            for view_key in ("A", "B"):
                for level in ("C4", "C5"):
                    maps[f"{view_key}_{level}"] = topk_evidence_map(
                        encoder, output["feats"][view_key], level, class_index
                    )

        path_a = Path(selection["source_path_a"])
        path_b = Path(selection["source_path_b"])
        raw_a = Image.open(path_a).convert("RGB").resize(
            (input_size, input_size), Image.Resampling.BILINEAR
        )
        raw_b = Image.open(path_b).convert("RGB").resize(
            (input_size, input_size), Image.Resampling.BILINEAR
        )
        expert_logits = (
            aux["expert_logits"][0, class_index].detach().float().cpu().numpy()
        )
        expert_probs = sigmoid(expert_logits)
        weights = aux["router_weights"][0, class_index].detach().float().cpu().numpy()
        correction = float(aux["correction"][0, class_index].detach().float().cpu())
        base_logit = float(output["logits_base"][0, class_index].detach().float().cpu())
        final_logit = float(output["logits"][0, class_index].detach().float().cpu())
        plain_probability = float(plain_export["probabilities"][sample_index, class_index])
        locked_uniform_probability = float(
            uniform_export["probabilities"][sample_index, class_index]
        )
        batch1_uniform_probability = float(sigmoid(final_logit))
        probability_error = abs(batch1_uniform_probability - locked_uniform_probability)

        fig = plt.figure(figsize=(10.2, 7.3), constrained_layout=True)
        grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.62])
        axes = [[fig.add_subplot(grid[row, column]) for column in range(3)] for row in range(2)]
        axes[0][0].imshow(raw_a)
        axes[0][0].set_title("OL view", fontsize=9)
        axes[0][0].axis("off")
        show_overlay(axes[0][1], raw_a, maps["A_C4"]["evidence"], "OL: C4 Top-K evidence")
        show_overlay(axes[0][2], raw_a, maps["A_C5"]["evidence"], "OL: C5 Top-K evidence")
        axes[1][0].imshow(raw_b)
        axes[1][0].set_title("SD view", fontsize=9)
        axes[1][0].axis("off")
        show_overlay(axes[1][1], raw_b, maps["B_C4"]["evidence"], "SD: C4 Top-K evidence")
        show_overlay(axes[1][2], raw_b, maps["B_C5"]["evidence"], "SD: C5 Top-K evidence")
        expert_axis = fig.add_subplot(grid[2, :])
        positions = np.arange(len(EXPERT_NAMES))
        bars = expert_axis.bar(
            positions,
            expert_probs,
            color=EXPERT_COLORS,
            width=0.68,
            edgecolor="white",
        )
        expert_axis.set_xticks(positions, EXPERT_NAMES, fontsize=8)
        expert_axis.set_ylim(0, 1.08)
        expert_axis.set_ylabel("Expert probability", fontsize=8)
        expert_axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        for bar, probability, weight in zip(bars, expert_probs, weights):
            expert_axis.text(
                bar.get_x() + bar.get_width() / 2,
                probability + 0.025,
                f"p={probability:.3f}\nw={weight:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        fig.suptitle(
            f"{args.dataset_name} | {selection['case_type']} | {class_name}\n"
            f"Plain={plain_probability:.3f}, Uniform={locked_uniform_probability:.3f}, "
            f"delta={locked_uniform_probability - plain_probability:+.3f}, "
            f"bounded correction={correction:+.3f}",
            fontsize=11,
        )
        for suffix, kwargs in (
            ("png", {"dpi": 400}),
            ("pdf", {}),
            ("svg", {}),
        ):
            fig.savefig(output_dir / f"{case_id}_region_evidence.{suffix}", **kwargs)
        plt.close(fig)

        evidence_payload = {
            "sample_index": np.asarray(sample_index),
            "class_index": np.asarray(class_index),
            "target": np.asarray(int(target[class_index].item())),
            "plain_probability": np.asarray(plain_probability),
            "locked_uniform_probability": np.asarray(locked_uniform_probability),
            "batch1_uniform_probability": np.asarray(batch1_uniform_probability),
            "base_logit": np.asarray(base_logit),
            "final_logit": np.asarray(final_logit),
            "correction": np.asarray(correction),
            "expert_logits": expert_logits,
            "expert_probabilities": expert_probs,
            "expert_weights": weights,
        }
        for map_name, item in maps.items():
            evidence_payload[f"{map_name}_topk_evidence"] = item["evidence"]
            evidence_payload[f"{map_name}_full_similarity"] = item["similarity"]
            evidence_payload[f"{map_name}_topk_indices"] = item["topk_indices"]
            evidence_payload[f"{map_name}_topk_weights"] = item["topk_weights"]
        np.savez_compressed(output_dir / f"{case_id}_evidence.npz", **evidence_payload)

        metadata = {
            **selection,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "evaluation_list": str(Path(args.list).resolve()),
            "evaluation_list_sha256": file_sha256(args.list),
            "heatmap_definition": "class-query Top-K softmax weights scattered onto the native feature grid",
            "levels": ["C4", "C5"],
            "topk": int(encoder.topk),
            "temperature": float(encoder.temperature),
            "expert_names": EXPERT_NAMES,
            "expert_probabilities": expert_probs.tolist(),
            "expert_weights": weights.tolist(),
            "base_logit": base_logit,
            "final_logit": final_logit,
            "correction": correction,
            "plain_probability": plain_probability,
            "locked_uniform_probability": locked_uniform_probability,
            "batch1_uniform_probability": batch1_uniform_probability,
            "batch1_vs_locked_probability_abs_error": probability_error,
        }
        (output_dir / f"{case_id}_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "case_number": case_number,
                "case_id": case_id,
                "case_type": selection["case_type"],
                "sample_index": sample_index,
                "target_class": class_name,
                "plain_probability": f"{plain_probability:.10f}",
                "locked_uniform_probability": f"{locked_uniform_probability:.10f}",
                "delta": f"{locked_uniform_probability - plain_probability:+.10f}",
                "batch1_vs_locked_abs_error": f"{probability_error:.10f}",
                "png": f"{case_id}_region_evidence.png",
                "evidence_npz": f"{case_id}_evidence.npz",
                "metadata_json": f"{case_id}_metadata.json",
            }
        )
        print(
            f"HEATMAP_PROGRESS {case_number}/{len(selections)} case={case_id} "
            f"delta={locked_uniform_probability - plain_probability:+.4f}",
            flush=True,
        )

    write_csv(output_dir / "heatmap_manifest.csv", manifest_rows)
    readme = f"""# {args.dataset_name} real region-evidence exports

- Every image is copied from the locked Test split; no image content was generated.
- Every heatmap is the actual class-query Top-K softmax evidence used by the Uniform Fusion region encoder.
- C4 and C5 maps are shown separately for OL and SD views.
- Expert bars are actual sigmoid probabilities for the four fixed-weight experts.
- The raw numeric evidence is stored beside every figure in an NPZ file.
- `batch1_vs_locked_abs_error` records numerical differences between individual visualization inference and the locked batch export.
- These maps are selector evidence maps, not Grad-CAM and not bounding-box annotations.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(
        f"UNIFORM_REGION_HEATMAP_EXPORT_OK dataset={args.dataset_name} "
        f"cases={len(manifest_rows)} epoch={checkpoint.get('epoch', -1)}"
    )


if __name__ == "__main__":
    main()
