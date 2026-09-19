#!/usr/bin/env python3
"""Compute locked, two-sided paired statistics for two methods."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = ("best_val_mAP", "test_mAP", "test_weak_class_mAP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed-csv", required=True)
    parser.add_argument("--baseline", default="Plain_BCE")
    parser.add_argument("--method", default="Final_NoAnchor")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty detailed results: {path}")
    return rows


def paired_values(
    rows: list[dict[str, str]], baseline: str, method: str, metric: str
) -> tuple[list[int], np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["method"] in {baseline, method}]
    by_key: dict[tuple[str, int], float] = {}
    for row in selected:
        key = (row["method"], int(row["seed"]))
        if key in by_key:
            raise ValueError(f"duplicate method/seed row: {key}")
        by_key[key] = float(row[metric])
    baseline_seeds = {seed for name, seed in by_key if name == baseline}
    method_seeds = {seed for name, seed in by_key if name == method}
    if baseline_seeds != method_seeds:
        raise ValueError(
            f"unpaired seeds: baseline={sorted(baseline_seeds)}, "
            f"method={sorted(method_seeds)}"
        )
    seeds = sorted(baseline_seeds)
    if len(seeds) < 3:
        raise ValueError(f"at least three paired seeds are required, found {len(seeds)}")
    baseline_values = np.asarray([by_key[(baseline, seed)] for seed in seeds])
    method_values = np.asarray([by_key[(method, seed)] for seed in seeds])
    return seeds, baseline_values, method_values


def statistic_row(
    metric: str,
    baseline: str,
    method: str,
    seeds: list[int],
    baseline_values: np.ndarray,
    method_values: np.ndarray,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, float | int | str], dict[str, object]]:
    differences = method_values - baseline_values
    n = len(differences)
    mean_difference = float(differences.mean())
    sd_difference = float(differences.std(ddof=1))
    standard_error = sd_difference / math.sqrt(n)
    t_critical = float(stats.t.ppf(0.975, df=n - 1))
    t_ci = (
        mean_difference - t_critical * standard_error,
        mean_difference + t_critical * standard_error,
    )

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, n, size=(bootstrap_samples, n))
    bootstrap_means = differences[indices].mean(axis=1)
    bootstrap_ci = tuple(np.quantile(bootstrap_means, [0.025, 0.975]))

    paired_t = stats.ttest_rel(method_values, baseline_values)
    if np.allclose(differences, 0.0):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
    else:
        wilcoxon = stats.wilcoxon(
            method_values,
            baseline_values,
            alternative="two-sided",
            zero_method="wilcox",
            method="exact",
        )
        wilcoxon_statistic = float(wilcoxon.statistic)
        wilcoxon_p = float(wilcoxon.pvalue)

    cohen_dz = mean_difference / sd_difference if sd_difference > 0 else 0.0
    row: dict[str, float | int | str] = {
        "metric": metric,
        "alternative": "two-sided",
        "n": n,
        "baseline": baseline,
        "method": method,
        "baseline_mean": float(baseline_values.mean()),
        "method_mean": float(method_values.mean()),
        "mean_paired_difference": mean_difference,
        "sd_paired_difference": sd_difference,
        "t_ci95_low": float(t_ci[0]),
        "t_ci95_high": float(t_ci[1]),
        "bootstrap_ci95_low": float(bootstrap_ci[0]),
        "bootstrap_ci95_high": float(bootstrap_ci[1]),
        "paired_t_statistic": float(paired_t.statistic),
        "paired_t_pvalue": float(paired_t.pvalue),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_p,
        "cohen_dz": cohen_dz,
        "method_wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    detail = {
        "metric": metric,
        "seeds": seeds,
        "baseline_values": baseline_values.tolist(),
        "method_values": method_values.tolist(),
        "paired_differences": differences.tolist(),
        "statistics": row,
    }
    return row, detail


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.detailed_csv))
    output_rows = []
    details = []
    for metric_index, metric in enumerate(METRICS):
        seeds, baseline_values, method_values = paired_values(
            rows, args.baseline, args.method, metric
        )
        row, detail = statistic_row(
            metric,
            args.baseline,
            args.method,
            seeds,
            baseline_values,
            method_values,
            args.bootstrap_samples,
            args.bootstrap_seed + metric_index,
        )
        output_rows.append(row)
        details.append(detail)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    output_json.write_text(
        json.dumps(
            {
                "source": str(Path(args.detailed_csv)),
                "alternative": "two-sided",
                "comparisons": details,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PAIRED_STATISTICS_OK metrics={len(output_rows)} n={output_rows[0]['n']}")


if __name__ == "__main__":
    main()
