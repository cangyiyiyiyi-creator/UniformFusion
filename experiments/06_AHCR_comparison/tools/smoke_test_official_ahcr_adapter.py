#!/usr/bin/env python3
from __future__ import annotations

import sys
import gc
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine_finetune import _official_ahcr_supervised_loss
from main_finetune import build_model, get_args_parser
from models.official_ahcr_adapter import OFFICIAL_COMMIT, OfficialAHCRAdapter


def main() -> None:
    torch.set_num_threads(1)
    args = get_args_parser().parse_args(
        [
            "--model",
            "official_ahcr_resnet50",
            "--input_size",
            "224",
            "--num_classes",
            "15",
            "--official_ahcr_source_dir",
            "third_party/DvXray_official",
            "--official_ahcr_source_commit",
            OFFICIAL_COMMIT,
            "--official_ahcr_pretrained_weights",
            "IMAGENET1K_V2",
        ]
    )
    model = build_model(args).cpu().eval()
    if not isinstance(model, OfficialAHCRAdapter):
        raise AssertionError(f"unexpected model type: {type(model)}")
    if model.official_source_commit != OFFICIAL_COMMIT:
        raise AssertionError("official commit was not retained by adapter")

    image_ol = torch.randn(1, 3, 224, 224)
    image_sd = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(image_ol, image_sd)
    expected_keys = {
        "logits",
        "ol_logits",
        "sd_logits",
        "official_ahcr_branch_logits",
        "official_ahcr_fusion_weights",
    }
    if set(output) != expected_keys:
        raise AssertionError(f"unexpected output keys: {sorted(output)}")
    for key in ("logits", "ol_logits", "sd_logits"):
        if tuple(output[key].shape) != (1, 15):
            raise AssertionError(f"{key} shape mismatch: {output[key].shape}")
    weights = output["official_ahcr_fusion_weights"]
    if tuple(weights.shape) != (1, 15, 2):
        raise AssertionError(f"fusion weight shape mismatch: {weights.shape}")
    if not torch.allclose(weights.sum(dim=-1), torch.ones(1, 15), atol=1e-6):
        raise AssertionError("official fusion weights do not sum to one")

    # Confirm exact equivalence with the upstream implementation for B=1.
    ol_probability = torch.sigmoid(output["ol_logits"].float())
    sd_probability = torch.sigmoid(output["sd_logits"].float())
    upstream_total = torch.cat(
        (torch.abs(ol_probability - 0.5), torch.abs(sd_probability - 0.5)), dim=0
    )
    upstream_weight = torch.softmax(upstream_total, dim=0)
    upstream_probability = (
        upstream_weight[0] * ol_probability + upstream_weight[1] * sd_probability
    )
    if not torch.allclose(torch.sigmoid(output["logits"]), upstream_probability, atol=1e-6):
        raise AssertionError("adapter fusion is not equivalent to upstream at B=1")

    model.train()
    training_output = model(image_ol, image_sd)
    training_target = torch.randint(0, 2, (1, 15), dtype=torch.float32)
    training_loss = _official_ahcr_supervised_loss(
        training_output, nn.BCEWithLogitsLoss(), training_target
    )
    if training_loss is None or not torch.isfinite(training_loss):
        raise AssertionError("official AHCR model training loss is invalid")
    training_loss.backward()
    finite_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not finite_gradients or not all(torch.isfinite(grad).all() for grad in finite_gradients):
        raise AssertionError("official AHCR model backward produced invalid gradients")
    model.zero_grad(set_to_none=True)
    model.eval()

    ol_logits = torch.randn(3, 15, requires_grad=True)
    sd_logits = torch.randn(3, 15, requires_grad=True)
    target = torch.randint(0, 2, (3, 15), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss()
    loss = _official_ahcr_supervised_loss(
        {"official_ahcr_branch_logits": (ol_logits, sd_logits)}, criterion, target
    )
    expected_loss = 0.5 * (criterion(ol_logits, target) + criterion(sd_logits, target))
    if loss is None or not torch.allclose(loss, expected_loss):
        raise AssertionError("official AHCR branch BCE mismatch")
    loss.backward()
    if ol_logits.grad is None or sd_logits.grad is None:
        raise AssertionError("official AHCR branch BCE did not backpropagate")

    # The evaluator reconstructs models from checkpoint args. Verify that the
    # exact same build path accepts a strict state-dict reload.
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    state = model.state_dict()
    del output, model
    gc.collect()
    reconstructed = build_model(args).cpu().eval()
    load_result = reconstructed.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise AssertionError(f"strict reconstruction mismatch: {load_result}")

    upstream_cache = Path("third_party/DvXray_official/__pycache__")
    if upstream_cache.exists():
        raise AssertionError("adapter wrote bytecode into the locked upstream checkout")
    print(
        "OFFICIAL_AHCR_ADAPTER_SMOKE_OK "
        f"commit={reconstructed.official_source_commit} params={parameter_count}"
    )


if __name__ == "__main__":
    main()
