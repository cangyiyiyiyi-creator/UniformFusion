#!/usr/bin/env python3
"""Shared helpers for read-only inference from locked project checkpoints."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_finetune import build_model, set_seed


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_list_entries(path: str | Path, num_classes: int):
    entries = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            tokens = line.strip().split()
            if not tokens:
                continue
            if len(tokens) < 3:
                raise ValueError(f"Malformed sample at line {index + 1}: {line!r}")
            labels = tokens[2].replace(",", " ").split()
            if len(labels) != num_classes:
                raise ValueError(
                    f"Expected {num_classes} labels at line {index + 1}, "
                    f"got {len(labels)}"
                )
            entries.append(
                {
                    "sample_index": len(entries),
                    "path_a": tokens[0],
                    "path_b": tokens[1],
                    "labels": [int(value) for value in labels],
                }
            )
    return entries


def load_locked_model(
    checkpoint_path: str | Path,
    device: torch.device,
    force_intermediate: bool = False,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain a model state dictionary")
    stored_args = dict(checkpoint.get("args", {}))
    if not stored_args:
        raise ValueError("Checkpoint does not contain reconstruction arguments")
    if force_intermediate:
        stored_args["return_intermediate"] = True
    model_args = SimpleNamespace(**stored_args)
    set_seed(
        int(getattr(model_args, "seed", 0)),
        deterministic=bool(getattr(model_args, "deterministic", True)),
    )
    model = build_model(model_args)
    state = {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in checkpoint["model"].items()
    }
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint reconstruction mismatch: "
            f"missing={load_result.missing_keys[:10]}, "
            f"unexpected={load_result.unexpected_keys[:10]}"
        )
    model.to(device)
    model.eval()
    return model, checkpoint, model_args


def extract_logits(model_output):
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    if isinstance(model_output, dict):
        return model_output["logits"]
    return model_output

