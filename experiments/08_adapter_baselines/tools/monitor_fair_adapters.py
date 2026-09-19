#!/usr/bin/env python3
import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(
    "论文补充验证_20260907/02_公平Adapter基线/"
    "run_20260907_fair_adapters_3seeds"
)
METHODS = (
    ("DAGNet", "dagnet_official_adapter", "DAGNet_OfficialArchitecture"),
    ("ML-Decoder", "resnet50_ml_decoder_adapter", "MLDecoder_DualView"),
)
SEEDS = ("930163947", "1786430941", "553800223")
PATIENCE = 25


def metric(path: Path):
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return float(payload.get("stats", {}).get("mAP"))


def fmt(value):
    return "-" if value is None else f"{value:.4f}"


print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("Fair Adapter Baselines: Val selection -> lock -> Test")
print(
    f"Training {len(list(ROOT.rglob('training_complete.marker')))}/6  |  "
    f"Val {len(list(ROOT.rglob('val_metrics.json')))}/6  |  "
    f"Test {len(list(ROOT.rglob('test_metrics.json')))}/6"
)
print("-" * 113)
print(
    f"{'Method':<12} {'Seed':<11} {'Status':<9} {'Epoch':>6} "
    f"{'Current':>9} {'Best':>9} {'BestEp':>7} {'EarlyStop':>10} "
    f"{'ETA(h)':>8} {'Test':>9}"
)
print("-" * 113)

for label, model, method in METHODS:
    for seed in SEEDS:
        matches = list(ROOT.glob(f"{model}/seed_{seed}/repeat_*/{method}"))
        if not matches:
            print(f"{label:<12} {seed:<11} {'WAITING':<9}")
            continue

        directory = matches[0]
        log = directory / "training_log.csv"
        rows = list(csv.DictReader(log.open())) if log.is_file() else []
        done = (directory / "training_complete.marker").is_file()
        val = metric(directory / "val_metrics.json")
        test = metric(directory / "test_metrics.json")

        if not rows:
            print(f"{label:<12} {seed:<11} {'STARTING':<9}")
            continue

        latest = rows[-1]
        best_row = max(rows, key=lambda row: float(row["val_metric"]))
        epoch = int(latest["epoch"]) + 1
        best_epoch = int(best_row["epoch"]) + 1
        stale = max(0, epoch - best_epoch)
        current = float(latest["val_metric"])
        best = float(best_row["val_metric"])
        eta = float(latest.get("estimated_remaining_hours") or 0)
        status = "DONE" if done else "RUNNING"
        early = "-" if done else f"{stale}/{PATIENCE}"
        shown_best = val if val is not None else best

        print(
            f"{label:<12} {seed:<11} {status:<9} {epoch:>6} "
            f"{current:>9.4f} {shown_best:>9.4f} {best_epoch:>7} "
            f"{early:>10} {eta:>8.2f} {fmt(test):>9}"
        )

print("-" * 113)
print("Current/Best are Val mAP. DONE rows use the locked best checkpoint.")
