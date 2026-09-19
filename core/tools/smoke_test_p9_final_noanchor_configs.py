import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine_finetune import _plain_bce_innovation_loss
from main_finetune import build_model, get_args_parser


CONFIGS = {
    "final_noanchor": {},
    "final_noanchor_no_cf": {
        "counterfactual": False,
        "single_weight": 0.0,
    },
    "final_noanchor_no_router": {
        "learned_router": False,
        "route_weight": 0.0,
    },
    "final_noanchor_no_guard": {"guard_weight": 0.0},
    "final_noanchor_single_c5": {"levels": ["C5"]},
}


def build(config):
    levels = config.get("levels", ["C4", "C5"])
    values = [
        "--model", "resnet50",
        "--num_classes", "15",
        "--teacher_mode", "false",
        "--fuse_mode", "add",
        "--head_type", "c5",
        "--base_loss", "bce",
        "--return_intermediate", "true",
        "--use_p9_caprs", "true",
        "--plain_innovation_levels", *levels,
        "--plain_innovation_projection_dim", "32",
        "--plain_innovation_topk", "4",
        "--plain_innovation_warmup_epochs", "0",
        "--plain_innovation_ramp_epochs", "0",
        "--plain_innovation_base_floor", "0.0",
        "--plain_innovation_guard_weight",
        str(config.get("guard_weight", 0.1)),
        "--plain_innovation_route_weight",
        str(config.get("route_weight", 0.05)),
        "--plain_innovation_single_weight",
        str(config.get("single_weight", 0.02)),
        "--plain_innovation_use_counterfactual_experts",
        str(config.get("counterfactual", True)).lower(),
        "--plain_innovation_use_learned_router",
        str(config.get("learned_router", True)).lower(),
    ]
    return build_model(get_args_parser().parse_args(values))


def check(name, config):
    model = build(config)
    model.set_plain_innovation_epoch(1)
    model.train()
    view_a = torch.randn(2, 3, 64, 64)
    view_b = torch.randn(2, 3, 64, 64)
    target = torch.arange(15).remainder(2).float().repeat(2, 1)
    output = model(view_a, view_b)
    objective = _plain_bce_innovation_loss(output, target)
    weights = {
        "aux": 0.03,
        "route": config.get("route_weight", 0.05),
        "guard": config.get("guard_weight", 0.1),
        "single": config.get("single_weight", 0.02),
    }
    loss = F.binary_cross_entropy_with_logits(output["logits"], target)
    loss = loss + sum(weights[key] * objective[key] for key in weights)
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError(f"non-finite loss for {name}")

    aux = output["plain_innovation_aux"]
    expected_experts = 2 if not config.get("counterfactual", True) else 4
    if aux["expert_logits"].shape != (2, 15, expected_experts):
        raise AssertionError(f"unexpected expert shape for {name}")
    if name == "final_noanchor_no_router":
        if aux["router_logits"] is not None:
            raise AssertionError("NoRouter unexpectedly produced router logits")
        expected = torch.full_like(aux["router_weights"], 1.0 / expected_experts)
        if not torch.allclose(aux["router_weights"], expected):
            raise AssertionError("NoRouter weights are not uniform")
        if float(objective["route"].detach()) != 0.0:
            raise AssertionError("NoRouter route loss is not zero")
    print(f"P9_FINAL_NOANCHOR_CONFIG_OK name={name} loss={loss.item():.6f}")
    del model, output, objective, loss
    gc.collect()


def main():
    torch.manual_seed(20260829)
    torch.set_num_threads(2)
    for name, config in CONFIGS.items():
        check(name, config)
    print("P9_FINAL_NOANCHOR_CONFIGS_SMOKE_OK")


if __name__ == "__main__":
    main()
