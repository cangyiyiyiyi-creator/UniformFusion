#!/usr/bin/env python3
"""Render auditable native-grid Top-K regional evidence without smoothing."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


MAPS = (
    ("A_C4", "OL: C4 regional evidence"),
    ("A_C5", "OL: C5 regional evidence"),
    ("B_C4", "SD: C4 regional evidence"),
    ("B_C5", "SD: C5 regional evidence"),
)
EXPERT_NAMES = ("Paired base", "OL-only", "SD-only", "Regional pair")
EXPERT_COLORS = ("#666666", "#009E73", "#56B4E9", "#CC79A7")


def render_map(axis, raw_image, evidence, title, vmax):
    image = np.asarray(raw_image)
    height, width = image.shape[:2]
    grid_h, grid_w = evidence.shape
    axis.imshow(image, extent=(0, width, height, 0))

    masked = np.ma.masked_where(evidence <= 0, evidence)
    axis.imshow(
        masked,
        cmap="magma",
        norm=colors.Normalize(vmin=0, vmax=vmax),
        interpolation="nearest",
        extent=(0, width, height, 0),
        alpha=0.68,
    )

    selected = np.argwhere(evidence > 0)
    selected = sorted(selected, key=lambda item: evidence[tuple(item)], reverse=True)
    cell_w, cell_h = width / grid_w, height / grid_h
    for rank, (row, column) in enumerate(selected, start=1):
        value = float(evidence[row, column])
        x, y = column * cell_w, row * cell_h
        axis.add_patch(Rectangle((x, y), cell_w, cell_h, fill=False, edgecolor="white", linewidth=0.75))
        if rank <= 3:
            axis.text(
                x + cell_w / 2,
                y + cell_h / 2,
                f"{rank}\n{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white",
                weight="bold",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 0.8, "edgecolor": "none"},
            )
    axis.set_title(f"{title}\n{grid_h}x{grid_w}, Top-{len(selected)}", fontsize=8.5)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evidence_path = Path(args.evidence)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    payload = np.load(evidence_path)
    raw_a = Image.open(metadata["archived_path_a"]).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    raw_b = Image.open(metadata["archived_path_b"]).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arrays = {name: payload[f"{name}_topk_evidence"] for name, _ in MAPS}
    vmax = max(float(array.max()) for array in arrays.values())

    fig = plt.figure(figsize=(11.8, 8.6))
    grid = fig.add_gridspec(
        3,
        4,
        height_ratios=(1.0, 1.0, 0.62),
        width_ratios=(1.0, 1.0, 1.0, 0.045),
        left=0.045,
        right=0.94,
        bottom=0.09,
        top=0.88,
        wspace=0.12,
        hspace=0.20,
    )
    raw_axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]))
    for axis, image, title in zip(raw_axes, (raw_a, raw_b), ("OL view (model input)", "SD view (model input)")):
        axis.imshow(image)
        axis.set_title(title, fontsize=9)
        axis.axis("off")

    map_axes = (
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    )
    for axis, (name, title) in zip(map_axes, MAPS):
        render_map(axis, raw_a if name.startswith("A_") else raw_b, arrays[name], title, vmax)

    scalar = plt.cm.ScalarMappable(norm=colors.Normalize(0, vmax), cmap="magma")
    colorbar_axis = fig.add_subplot(grid[:2, 3])
    colorbar = fig.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label("Actual Top-K softmax weight", fontsize=8)

    expert_axis = fig.add_subplot(grid[2, :3])
    probabilities = payload["expert_probabilities"]
    weights = payload["expert_weights"]
    bars = expert_axis.bar(np.arange(4), probabilities, color=EXPERT_COLORS, width=0.64)
    expert_axis.set_xticks(np.arange(4), EXPERT_NAMES, fontsize=8)
    expert_axis.set_ylim(0, 1.10)
    expert_axis.set_ylabel("Expert probability", fontsize=8)
    expert_axis.grid(axis="y", linewidth=0.5, color="#dddddd")
    for bar, probability, weight in zip(bars, probabilities, weights):
        expert_axis.text(
            bar.get_x() + bar.get_width() / 2,
            float(probability) + 0.025,
            f"p={float(probability):.3f}, w={float(weight):.2f}",
            ha="center",
            fontsize=7,
        )

    fig.suptitle(
        f"{metadata['target_class']} | {metadata['case_type']} | sample {metadata['sample_index']}\n"
        f"Plain={metadata['plain_probability']:.3f}, Uniform={metadata['locked_uniform_probability']:.3f}, "
        f"bounded correction={metadata['correction']:+.3f}",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.025,
        "Native-grid class-query Top-K selector weights; no Grad-CAM and no interpolation smoothing.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=350)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)
    print(f"EXACT_REGION_EVIDENCE_OK output={output} shared_vmax={vmax:.10f}")


if __name__ == "__main__":
    main()
