#!/usr/bin/env python3
"""Forward/backward smoke test for the locked method on both backbones."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine_finetune import _plain_bce_innovation_loss
from main_finetune import build_model, get_args_parser


def build(backbone: str):
    values = [
        "--model", backbone,
        "--num_classes", "15",
        "--teacher_mode", "false",
        "--fuse_mode", "add",
        "--head_type", "c5",
        "--base_loss", "bce",
        "--return_intermediate", "true",
        "--use_p9_caprs", "true",
        "--plain_innovation_levels", "C4", "C5",
        "--plain_innovation_projection_dim", "64",
        "--plain_innovation_topk", "8",
        "--plain_innovation_temperature", "0.2",
        "--plain_innovation_dropout", "0.1",
        "--plain_innovation_gamma_init", "0.005",
        "--plain_innovation_gamma_max", "0.05",
        "--plain_innovation_base_floor", "0.0",
        "--plain_innovation_use_counterfactual_experts", "true",
        "--plain_innovation_use_learned_router", "true",
        "--plain_innovation_warmup_epochs", "0",
        "--plain_innovation_ramp_epochs", "0",
        "--plain_innovation_aux_weight", "0.03",
        "--plain_innovation_route_weight", "0.05",
        "--plain_innovation_guard_weight", "0.10",
        "--plain_innovation_single_weight", "0.02",
    ]
    return build_model(get_args_parser().parse_args(values))


def check(backbone: str) -> None:
    model = build(backbone)
    model.set_plain_innovation_epoch(1)
    model.train()
    view_a = torch.randn(1, 3, 64, 64)
    view_b = torch.randn_like(view_a)
    target = torch.arange(15).remainder(2).float().unsqueeze(0)
    output = model(view_a, view_b)
    objective = _plain_bce_innovation_loss(output, target)
    loss = F.binary_cross_entropy_with_logits(output["logits"], target)
    loss = loss + 0.03 * objective["aux"]
    loss = loss + 0.05 * objective["route"]
    loss = loss + 0.10 * objective["guard"]
    loss = loss + 0.02 * objective["single"]
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError(f"non-finite loss for {backbone}")
    aux = output["plain_innovation_aux"]
    if aux["expert_logits"].shape != (1, 15, 4):
        raise AssertionError(
            f"unexpected expert shape for {backbone}: {aux['expert_logits'].shape}"
        )
    if aux["router_logits"].shape != (1, 15, 4):
        raise AssertionError(
            f"unexpected router shape for {backbone}: {aux['router_logits'].shape}"
        )
    if model.plain_innovation_head.base_floor != 0.0:
        raise AssertionError("locked method unexpectedly has a nonzero anchor floor")
    print(
        f"FINAL_NOANCHOR_BACKBONE_SMOKE_OK backbone={backbone} "
        f"loss={float(loss.detach()):.6f}"
    )
    del model, output, objective, loss
    gc.collect()


def main() -> None:
    torch.manual_seed(20260830)
    torch.set_num_threads(2)
    for backbone in ("resnet50", "convnextv2_tiny"):
        check(backbone)
    print("FINAL_NOANCHOR_ALL_BACKBONES_SMOKE_OK")


if __name__ == "__main__":
    main()
