#!/usr/bin/env python3
from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
LOCKED_CODE = ROOT / "paper_archive_20260830" / "05_run_config_and_code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LOCKED_CODE))

from engine_finetune import _plain_bce_innovation_loss
from main_finetune import build_model, get_args_parser


CONFIGS = {
    "Uniform_NoCounterfactual": {
        "counterfactual": False,
        "single_weight": 0.0,
    },
    "Uniform_NoGuard": {"guard_weight": 0.0},
    "Uniform_C5Only": {"levels": ["C5"]},
}


def check(name: str, config: dict) -> None:
    values = [
        "--model", "resnet50", "--num_classes", "15",
        "--teacher_mode", "false", "--fuse_mode", "add",
        "--head_type", "c5", "--base_loss", "bce",
        "--return_intermediate", "true", "--use_p9_caprs", "true",
        "--plain_innovation_levels", *config.get("levels", ["C4", "C5"]),
        "--plain_innovation_projection_dim", "32",
        "--plain_innovation_topk", "4",
        "--plain_innovation_warmup_epochs", "0",
        "--plain_innovation_ramp_epochs", "0",
        "--plain_innovation_base_floor", "0.0",
        "--plain_innovation_guard_weight", str(config.get("guard_weight", 0.10)),
        "--plain_innovation_route_weight", "0.0",
        "--plain_innovation_single_weight", str(config.get("single_weight", 0.02)),
        "--plain_innovation_use_counterfactual_experts",
        str(config.get("counterfactual", True)).lower(),
        "--plain_innovation_use_learned_router", "false",
    ]
    model = build_model(get_args_parser().parse_args(values))
    model.set_plain_innovation_epoch(1)
    model.train()
    view_a = torch.randn(1, 3, 64, 64)
    view_b = torch.randn_like(view_a)
    target = torch.arange(15).remainder(2).float().unsqueeze(0)
    output = model(view_a, view_b)
    objective = _plain_bce_innovation_loss(output, target)
    loss = F.binary_cross_entropy_with_logits(output["logits"], target)
    loss = loss + 0.03 * objective["aux"]
    loss = loss + config.get("guard_weight", 0.10) * objective["guard"]
    loss = loss + config.get("single_weight", 0.02) * objective["single"]
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError(f"non-finite loss for {name}")
    aux = output["plain_innovation_aux"]
    experts = 2 if not config.get("counterfactual", True) else 4
    if aux["expert_logits"].shape != (1, 15, experts):
        raise AssertionError(f"unexpected expert shape for {name}")
    if aux["router_logits"] is not None:
        raise AssertionError(f"{name} unexpectedly produced router logits")
    expected = torch.full_like(aux["router_weights"], 1.0 / experts)
    if not torch.allclose(aux["router_weights"], expected):
        raise AssertionError(f"{name} is not uniformly fused")
    if float(objective["route"].detach()) != 0.0:
        raise AssertionError(f"{name} route loss is not zero")
    print(f"UNIFORM_COMPONENT_CONFIG_OK method={name} loss={float(loss.detach()):.6f}")
    del model, output, objective, loss
    gc.collect()


def main() -> None:
    torch.manual_seed(20260831)
    torch.set_num_threads(2)
    for name, config in CONFIGS.items():
        check(name, config)
    print("UNIFORM_COMPONENT_CONFIGS_SMOKE_OK methods=3")


if __name__ == "__main__":
    main()
