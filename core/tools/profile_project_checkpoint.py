#!/usr/bin/env python3
"""Profile an actual project checkpoint with paired synthetic inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main_finetune import build_model, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stored_args = payload.get("args", {})
    if not isinstance(stored_args, dict):
        stored_args = vars(stored_args)
    if not stored_args:
        raise ValueError(f"checkpoint has no reconstruction arguments: {checkpoint_path}")
    model_args = SimpleNamespace(**stored_args)
    model = build_model(model_args)
    state = {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in payload["model"].items()
    }
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "checkpoint reconstruction mismatch: "
            f"missing={result.missing_keys[:10]}, "
            f"unexpected={result.unexpected_keys[:10]}"
        )
    return model.to(device).eval(), payload, model_args


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_flops(model, view_a, view_b, device: torch.device) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
        with autocast_context(device):
            model(view_a, view_b)
    synchronize(device)
    return int(sum(event.flops or 0 for event in profiler.key_averages()))


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output_json)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite profile: {output_path}")
    if args.batch_size < 1 or args.warmup < 1 or args.repeats < 1:
        raise ValueError("batch-size, warmup and repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")

    device = torch.device(args.device)
    set_seed(args.seed, deterministic=True)
    model, checkpoint, model_args = load_model(checkpoint_path, device)
    input_size = int(getattr(model_args, "input_size", 224))
    view_a = torch.randn(
        args.batch_size, 3, input_size, input_size, device=device
    )
    view_b = torch.randn_like(view_a)

    with torch.inference_mode():
        for _ in range(args.warmup):
            with autocast_context(device):
                model(view_a, view_b)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(args.repeats):
                with autocast_context(device):
                    model(view_a, view_b)
            end_event.record()
            synchronize(device)
            elapsed_seconds = start_event.elapsed_time(end_event) / 1000.0
            peak_memory = torch.cuda.max_memory_allocated(device)
        else:
            start = time.perf_counter()
            for _ in range(args.repeats):
                model(view_a, view_b)
            elapsed_seconds = time.perf_counter() - start
            peak_memory = 0
        flops_per_batch = profile_flops(model, view_a, view_b, device)

    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    latency_ms_per_batch = 1000.0 * elapsed_seconds / args.repeats
    result = {
        "method": args.method_label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "backbone": str(getattr(model_args, "model", "unknown")),
        "device": str(device),
        "precision": "amp_fp16" if device.type == "cuda" else "fp32",
        "batch_size": args.batch_size,
        "input_size": input_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "params": params,
        "params_M": params / 1e6,
        "trainable_params": trainable,
        "trainable_params_M": trainable / 1e6,
        "checkpoint_MiB": checkpoint_path.stat().st_size / (2**20),
        "profiler_GFLOPs_per_batch": flops_per_batch / 1e9,
        "profiler_GFLOPs_per_sample": flops_per_batch / args.batch_size / 1e9,
        "latency_ms_per_batch": latency_ms_per_batch,
        "latency_ms_per_sample": latency_ms_per_batch / args.batch_size,
        "throughput_samples_per_s": args.batch_size * args.repeats / elapsed_seconds,
        "peak_memory_MiB": peak_memory / (2**20),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        "PROJECT_CHECKPOINT_PROFILE_OK "
        f"method={args.method_label} batch={args.batch_size} "
        f"latency_ms={latency_ms_per_batch:.6f}"
    )


if __name__ == "__main__":
    main()
