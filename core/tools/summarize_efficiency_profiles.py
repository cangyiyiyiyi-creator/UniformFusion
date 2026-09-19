#!/usr/bin/env python3
"""Merge checkpoint profiling JSON files into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("profiles/*.json"))
    ]
    if len(rows) != args.expected:
        raise SystemExit(f"expected {args.expected} profiles, found {len(rows)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"EFFICIENCY_PROFILE_SUMMARY_OK rows={len(rows)}")


if __name__ == "__main__":
    main()
