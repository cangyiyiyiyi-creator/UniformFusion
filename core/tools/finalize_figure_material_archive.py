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
    prediction_manifests = sorted(root.glob("01_样本级预测/*/*/seed_*/prediction_manifest.json"))
    if len(prediction_manifests) != 16:
        raise RuntimeError(f"Expected 16 prediction exports, found {len(prediction_manifests)}")
    manifest_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in prediction_manifests]
    max_map_error = max(item["mAP_absolute_error"] for item in manifest_payloads)
    if max_map_error > 1e-6:
        raise RuntimeError(f"Locked mAP verification failed: max error={max_map_error}")

    case_counts = {}
    heatmap_counts = {}
    for dataset in ("DvXray", "LDXray"):
        selections = read_csv(root / "03_成功失败案例" / dataset / "selection_manifest.csv")
        heatmaps = read_csv(root / "04_区域热图" / dataset / "heatmap_manifest.csv")
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

    readme = f"""# 真实可视化与 PR 曲线论文材料

本目录由锁定 Test 划分和锁定 `checkpoint_best.pth` 纯推理生成，没有重新训练或修改模型。

## 内容

- `01_样本级预测/`：DvXray 5 seeds 与 LDXray 3 seeds 的 Plain/Uniform 完整 Test targets、logits、probabilities 和专家证据。
- `02_PR曲线/`：弱类/稀有类的多种子均值 PR 曲线、标准差带、SVG/PDF/PNG 和完整绘图 CSV。
- `03_成功失败案例/`：按预先固定规则选出的真实成对原图、选样分数和来源路径。
- `04_区域热图/`：真实 C4/C5 class-query Top-K 选择权重、四专家证据、原始 NPZ 与可排版图。
- `05_生成代码/`：本材料包使用的全部独立推理与绘图代码。

## 完整性

- 锁定预测导出：16/16。
- DvXray 客观选样/热图：{case_counts['DvXray']}/{heatmap_counts['DvXray']}。
- LDXray 客观选样/热图：{case_counts['LDXray']}/{heatmap_counts['LDXray']}。
- 导出 mAP 与原锁定 Test 指标的最大绝对误差：`{max_map_error:.3e}`。

## 论文表述边界

1. 热图是 class-query Top-K 选择证据，不是 Grad-CAM，不是物体标注框。
2. `success` 与 `failure` 是按选样清单中的固定概率规则命名，不能脱离 target 和分数解读。
3. PR 曲线的阴影是固定种子间标准差，不是置信区间。
4. 原始图像仅从本机官方/锁定数据路径复制，未经 AI 生成或编辑。
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    manifest_path = root / "文件清单与SHA256.csv"
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
