#!/usr/bin/env python3
"""Audit UF base/final corrections and tie-invariant AP from locked NPZ files."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def grouped_ap(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ys, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]) + 1, len(ss)]
    tp = np.cumsum(ys)[ends - 1]
    precision = tp / ends
    previous = np.r_[0, tp[:-1]]
    return float(np.sum(precision * (tp - previous)) / positives)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_names(root, dataset):
    rel = "annotations/classes.txt" if dataset == "DvXray" else "annotations/ldxray/ldxray_classes.txt"
    return [line.strip() for line in (root / rel).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="论文最终归档_20260830")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive, output = root / args.archive, Path(args.output)
    if not output.is_absolute():
        output = root / output
    pred_root = archive / "08_真实可视化与PR曲线_20260902/01_样本级预测"
    run_rows, class_rows, tie_rows = [], [], []
    rng = np.random.default_rng(20260905)

    for dataset in ("DvXray", "LDXray"):
        names = class_names(root, dataset)
        for path in sorted((pred_root / dataset / "Uniform_Fusion").glob("seed_*/predictions.npz")):
            seed = int(path.parent.name.removeprefix("seed_"))
            data = np.load(path)
            target = data["targets"].astype(np.int64)
            base, final = data["base_logits"].astype(np.float64), data["logits"].astype(np.float64)
            correction = data["correction"].astype(np.float64)
            if not np.allclose(final, base + correction, atol=2e-5, rtol=0):
                raise AssertionError(f"logit identity failed: {path}")
            base_prob = 1 / (1 + np.exp(-base))
            final_prob = 1 / (1 + np.exp(-final))
            base_ap = average_precision_score(target, base_prob, average=None)
            final_ap = average_precision_score(target, final_prob, average=None)
            q = np.quantile(correction, [0, .01, .05, .25, .5, .75, .95, .99, 1])
            flips = (base_prob >= .5) != (final_prob >= .5)
            rank_changed = 0
            for c in range(target.shape[1]):
                rb = np.empty(target.shape[0], int); rb[np.argsort(-base[:, c], kind="mergesort")] = np.arange(target.shape[0])
                rf = np.empty(target.shape[0], int); rf[np.argsort(-final[:, c], kind="mergesort")] = np.arange(target.shape[0])
                rank_changed += int(np.count_nonzero(rb != rf))
                gap = float(final_ap[c] - base_ap[c])
                cc = correction[:, c]
                class_rows.append({
                    "dataset": dataset, "seed": seed, "class_index": c, "class_name": names[c],
                    "base_AP": base_ap[c], "final_AP": final_ap[c], "delta_AP": gap,
                    "correction_mean": cc.mean(), "correction_std": cc.std(ddof=1),
                    "correction_positive_ratio": (cc > 0).mean(), "correction_negative_ratio": (cc < 0).mean(),
                    "threshold_flips": int(flips[:, c].sum()),
                })
                gap_grouped = grouped_ap(target[:, c], final_prob[:, c])
                shuffled = []
                for _ in range(args.shuffle_repeats):
                    idx = rng.permutation(target.shape[0])
                    shuffled.append(grouped_ap(target[idx, c], final_prob[idx, c]))
                tie_rows.append({
                    "dataset": dataset, "seed": seed, "class_index": c, "class_name": names[c],
                    "sklearn_AP": final_ap[c], "grouped_tie_AP": gap_grouped,
                    "max_shuffle_abs_difference": max(abs(x-gap_grouped) for x in shuffled),
                })
            run_rows.append({
                "dataset": dataset, "seed": seed, "samples": target.shape[0], "classes": target.shape[1],
                "base_mAP": base_ap.mean(), "final_mAP": final_ap.mean(), "delta_mAP": (final_ap-base_ap).mean(),
                "correction_mean": correction.mean(), "correction_std": correction.std(ddof=1),
                "correction_min": q[0], "correction_p01": q[1], "correction_p05": q[2],
                "correction_p25": q[3], "correction_median": q[4], "correction_p75": q[5],
                "correction_p95": q[6], "correction_p99": q[7], "correction_max": q[8],
                "positive_ratio": (correction > 0).mean(), "negative_ratio": (correction < 0).mean(),
                "zero_ratio": (correction == 0).mean(), "threshold_flips": int(flips.sum()),
                "rank_positions_changed": rank_changed,
            })

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "UF内部base_vs_final_逐seed.csv", run_rows)
    write_csv(output / "UF内部base_vs_final_逐类.csv", class_rows)
    write_csv(output / "GroupedTieAP_顺序不敏感审计.csv", tie_rows)
    summary = []
    for dataset in ("DvXray", "LDXray"):
        rows = [r for r in run_rows if r["dataset"] == dataset]
        summary.append({"dataset": dataset, "n": len(rows), **{
            f"{key}_mean": float(np.mean([r[key] for r in rows]))
            for key in ("base_mAP", "final_mAP", "delta_mAP", "correction_mean", "correction_std", "positive_ratio", "negative_ratio", "threshold_flips", "rank_positions_changed")
        }})
    write_csv(output / "UF内部base_vs_final_汇总.csv", summary)
    max_shuffle = max(float(r["max_shuffle_abs_difference"]) for r in tie_rows)
    report = {
        "status": "PASS" if max_shuffle <= 1e-12 else "FAIL",
        "npz_runs": len(run_rows), "class_rows": len(class_rows),
        "shuffle_repeats_per_class": args.shuffle_repeats,
        "max_grouped_ap_shuffle_difference": max_shuffle,
        "summary": summary,
    }
    (output / "审计摘要.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"UF_INTERNAL_AUDIT_{report['status']} runs={len(run_rows)} class_rows={len(class_rows)} max_shuffle_diff={max_shuffle:.3g}")


if __name__ == "__main__":
    main()
