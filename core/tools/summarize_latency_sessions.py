#!/usr/bin/env python3
"""Summarize independent checkpoint profiling sessions."""

import argparse
import csv
import json
import statistics
from pathlib import Path


FIELDS = (
    "latency_ms_per_batch",
    "latency_ms_per_sample",
    "throughput_samples_per_s",
    "peak_memory_MiB",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(args.input_dir).glob("*.json"))
    ]
    if not payloads:
        raise ValueError("no profile JSON files found")

    groups = {}
    for payload in payloads:
        key = (payload["method"], int(payload["batch_size"]))
        groups.setdefault(key, []).append(payload)

    rows = []
    for (method, batch_size), items in sorted(groups.items()):
        row = {
            "method": method,
            "batch_size": batch_size,
            "sessions": len(items),
            "warmup": items[0]["warmup"],
            "repeats_per_session": items[0]["repeats"],
            "precision": items[0]["precision"],
            "device": items[0]["device"],
        }
        for field in FIELDS:
            values = [float(item[field]) for item in items]
            row[f"{field}_mean"] = statistics.mean(values)
            row[f"{field}_std_sample"] = statistics.stdev(values)
        rows.append(row)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"LATENCY_SESSION_SUMMARY_OK groups={len(rows)} profiles={len(payloads)}")


if __name__ == "__main__":
    main()
