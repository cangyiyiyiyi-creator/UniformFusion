import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import datasets as D
from engine_finetune import evaluate
from main_finetune import build_model, set_seed


class RouterStatsCollector:
    def __init__(self):
        self.sample_count = 0
        self.element_count = 0
        self.expert_sum = None
        self.per_class_sum = None
        self.entropy_sum = 0.0
        self.gate_sum = 0.0
        self.correction_abs_sum = 0.0
        self.use_learned_router = None

    def __call__(self, _module, _inputs, output):
        if not isinstance(output, dict) or output.get("router_weights") is None:
            return
        weights = output["router_weights"].detach().float().cpu()
        if weights.ndim != 3:
            raise ValueError(f"expected [B,C,E] router weights, got {weights.shape}")
        batch_size, num_classes, num_experts = weights.shape
        if self.expert_sum is None:
            self.expert_sum = weights.new_zeros(num_experts)
            self.per_class_sum = weights.new_zeros(num_classes, num_experts)
        if tuple(self.per_class_sum.shape) != (num_classes, num_experts):
            raise ValueError("router shape changed between evaluation batches")
        self.sample_count += batch_size
        self.element_count += batch_size * num_classes
        self.expert_sum += weights.sum(dim=(0, 1))
        self.per_class_sum += weights.sum(dim=0)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
        if num_experts > 1:
            entropy = entropy / math.log(num_experts)
        self.entropy_sum += float(entropy.sum())
        gate = output.get("gate")
        if gate is not None:
            self.gate_sum += float(gate.detach().float().cpu().sum())
        correction = output.get("correction")
        if correction is not None:
            self.correction_abs_sum += float(
                correction.detach().float().cpu().abs().sum()
            )
        self.use_learned_router = bool(output.get("use_learned_router", True))

    def finalize(self, class_names):
        if self.element_count == 0 or self.expert_sum is None:
            raise RuntimeError("router statistics were requested but not collected")
        expert_mean = self.expert_sum / self.element_count
        class_mean = self.per_class_sum / self.sample_count
        return {
            "use_learned_router": self.use_learned_router,
            "samples": self.sample_count,
            "num_experts": int(expert_mean.numel()),
            "expert_weight_mean": [float(value) for value in expert_mean],
            "base_weight_mean": float(expert_mean[0]),
            "non_base_mass_mean": float(1.0 - expert_mean[0]),
            "normalized_entropy_mean": self.entropy_sum / self.element_count,
            "gate_mean": self.gate_sum / self.element_count,
            "correction_abs_mean": self.correction_abs_sum / self.element_count,
            "per_class_expert_weight_mean": {
                name: [float(value) for value in class_mean[index]]
                for index, name in enumerate(class_names)
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--classes-file", default="annotations/classes.txt")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--collect-router-stats",
        action="store_true",
        help="Collect P9 expert-weight and correction diagnostics during evaluation.",
    )
    parser.add_argument(
        "--view-mode",
        default="checkpoint",
        choices=("checkpoint", "paired", "ol_only", "sd_only", "mismatched"),
        help="Evaluation-only view override; checkpoint preserves prior behavior.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint must contain model and args dictionaries")
    stored_args = dict(checkpoint.get("args", {}))
    if not stored_args:
        raise ValueError("checkpoint does not contain reconstruction arguments")
    model_args = SimpleNamespace(**stored_args)
    set_seed(
        int(getattr(model_args, "seed", 0)),
        deterministic=bool(getattr(model_args, "deterministic", True)),
    )
    device = torch.device(args.device)
    model = build_model(model_args)
    state = {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in checkpoint["model"].items()
    }
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "checkpoint reconstruction mismatch: "
            f"missing={load_result.missing_keys[:10]}, "
            f"unexpected={load_result.unexpected_keys[:10]}"
        )
    model.to(device)
    model.eval()

    num_classes = int(getattr(model_args, "num_classes", 15))
    input_size = int(getattr(model_args, "input_size", 224))
    checkpoint_view_mode = str(getattr(model_args, "view_mode", "paired"))
    requested_view_mode = str(args.view_mode).lower()
    view_mode = {
        "checkpoint": checkpoint_view_mode,
        "paired": "paired",
        "ol_only": "a_only",
        "sd_only": "b_only",
        "mismatched": "mismatched",
    }[requested_view_mode]
    class_names = D._read_class_names(args.classes_file, num_classes)
    dataset = D.DualViewTxtDataset(
        args.list,
        input_size,
        num_classes,
        train=False,
        class_names=class_names,
        view_mode=view_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_json.exists() or output_csv.exists():
        raise FileExistsError("refusing to append to an existing evaluation output")

    router_collector = None
    router_hook = None
    if args.collect_router_stats:
        innovation_head = getattr(model, "plain_innovation_head", None)
        if innovation_head is None:
            raise ValueError("router statistics require a Plain-BCE innovation head")
        router_collector = RouterStatsCollector()
        router_hook = innovation_head.register_forward_hook(router_collector)
    try:
        stats = evaluate(
            data_loader=loader,
            model=model,
            device=device,
            criterion=None,
            amp=True,
            threshold=args.threshold,
            class_names=class_names,
            csv_path=str(output_csv),
            epoch=int(checkpoint.get("epoch", -1)),
        )
    finally:
        if router_hook is not None:
            router_hook.remove()
    payload = {
        "checkpoint": str(Path(args.checkpoint)),
        "evaluation_list": str(Path(args.list)),
        "evaluation_list_sha256": hashlib.sha256(
            Path(args.list).read_bytes()
        ).hexdigest(),
        "requested_view_mode": requested_view_mode,
        "effective_view_mode": view_mode,
        "mismatch_policy": (
            "cyclic_half_split_b_view"
            if view_mode == "mismatched"
            else None
        ),
        "samples": len(dataset),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_metric": checkpoint.get("metric", {}),
        "total_params": sum(parameter.numel() for parameter in model.parameters()),
        "stats": stats,
    }
    if router_collector is not None:
        payload["router_stats"] = router_collector.finalize(class_names)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PROJECT_CHECKPOINT_EVAL_OK "
        f"samples={len(dataset)} mAP={float(stats['mAP']):.10f}"
    )


if __name__ == "__main__":
    main()
