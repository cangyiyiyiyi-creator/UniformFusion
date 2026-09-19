import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine_finetune import _plain_bce_innovation_loss
from main_finetune import build_model, get_args_parser


def build(method):
    values = [
        "--model", "resnet50", "--num_classes", "15", "--teacher_mode", "false",
        "--fuse_mode", "add", "--head_type", "c5", "--base_loss", "bce",
        "--return_intermediate", "true", "--use_p9_caprs", "true",
        "--plain_innovation_levels", "C4", "C5", "--plain_innovation_projection_dim", "32",
        "--plain_innovation_warmup_epochs", "0", "--plain_innovation_ramp_epochs", "0",
        "--plain_innovation_base_floor", "0", "--plain_innovation_use_counterfactual_experts", "true",
        "--plain_innovation_use_learned_router", "false", "--plain_innovation_route_weight", "0",
    ]
    if method == "UF_NoSingleLoss":
        values += ["--plain_innovation_topk", "8", "--plain_innovation_single_weight", "0"]
    elif method == "Plain_BCE_SingleLoss":
        values += ["--plain_innovation_topk", "8", "--plain_innovation_gamma_init", "0",
                   "--plain_innovation_gamma_trainable", "false", "--plain_innovation_aux_weight", "0",
                   "--plain_innovation_guard_weight", "0", "--plain_innovation_single_weight", "0.02"]
    elif method == "UF_DenseSelector":
        values += ["--plain_innovation_topk", "100000", "--plain_innovation_single_weight", "0.02"]
    elif method == "UF_GAP":
        values += ["--plain_innovation_topk", "8", "--plain_innovation_region_pooling", "gap",
                   "--plain_innovation_single_weight", "0.02"]
    return build_model(get_args_parser().parse_args(values))


def main():
    torch.manual_seed(20260905)
    torch.set_num_threads(2)
    target = torch.randint(0, 2, (2, 15)).float()
    for method in ("UF_NoSingleLoss", "Plain_BCE_SingleLoss", "UF_DenseSelector", "UF_GAP"):
        model = build(method).train()
        model.set_plain_innovation_epoch(1)
        output = model(torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224))
        losses = _plain_bce_innovation_loss(output, target)
        if method == "Plain_BCE_SingleLoss":
            if not torch.equal(output["logits"], output["logits_base"]):
                raise AssertionError("Plain+SingleLoss changed final logits")
            if not float(losses["single"]) > 0:
                raise AssertionError("Plain+SingleLoss did not activate single-view supervision")
        if method == "UF_DenseSelector":
            encoder = model.plain_innovation_head.encoder
            if encoder.topk < 196:
                raise AssertionError("dense selector does not retain full C4 grid")
        if method == "UF_GAP":
            encoder = model.plain_innovation_head.encoder
            if encoder.pooling != "gap":
                raise AssertionError("GAP pooling was not activated")
        print(f"REVIEWER_CONFIG_OK method={method}")
    print("REVIEWER_FIRST_BATCH_SMOKE_OK")


if __name__ == "__main__":
    main()
