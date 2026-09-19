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


def main() -> None:
    values = [
        "--model", "convnextv2_tiny",
        "--num_classes", "15",
        "--teacher_mode", "false",
        "--fuse_mode", "add",
        "--head_type", "c5",
        "--base_loss", "bce",
        "--return_intermediate", "true",
        "--use_p9_caprs", "true",
        "--plain_innovation_levels", "C4", "C5",
        "--plain_innovation_projection_dim", "32",
        "--plain_innovation_topk", "4",
        "--plain_innovation_warmup_epochs", "0",
        "--plain_innovation_ramp_epochs", "0",
        "--plain_innovation_base_floor", "0.0",
        "--plain_innovation_guard_weight", "0.10",
        "--plain_innovation_route_weight", "0.0",
        "--plain_innovation_single_weight", "0.02",
        "--plain_innovation_use_counterfactual_experts", "true",
        "--plain_innovation_use_learned_router", "false",
    ]
    torch.manual_seed(20260831)
    torch.set_num_threads(2)
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
    loss = loss + 0.10 * objective["guard"]
    loss = loss + 0.02 * objective["single"]
    loss.backward()

    if not torch.isfinite(loss):
        raise AssertionError("non-finite ConvNeXtV2 Uniform Fusion loss")
    aux = output["plain_innovation_aux"]
    if aux["expert_logits"].shape != (1, 15, 4):
        raise AssertionError(f"unexpected expert shape: {aux['expert_logits'].shape}")
    if aux["router_logits"] is not None:
        raise AssertionError("Uniform Fusion unexpectedly produced router logits")
    expected = torch.full_like(aux["router_weights"], 0.25)
    if not torch.allclose(aux["router_weights"], expected):
        raise AssertionError("Uniform Fusion weights are not fixed to 1/4")
    if float(objective["route"].detach()) != 0.0:
        raise AssertionError("Uniform Fusion route loss is not zero")
    print(f"CONVNEXTV2_UNIFORM_FUSION_SMOKE_OK loss={float(loss.detach()):.6f}")
    del model, output, objective, loss
    gc.collect()


if __name__ == "__main__":
    main()
