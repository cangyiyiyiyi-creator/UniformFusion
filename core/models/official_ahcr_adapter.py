from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import torch
from torch import nn


OFFICIAL_COMMIT = "a6bfc1b1299d28e8226c106a94967287a8e30927"
OFFICIAL_MODEL_SHA256 = "bbc135c43b908cb33278d3f1baee510a4074e0c9b1a252d36cf23e8a973eca85"
_MODULE_CACHE: dict[Path, types.ModuleType] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source_dir(source_dir: str | Path) -> Path:
    path = Path(source_dir).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def _load_locked_official_module(
    source_dir: str | Path,
    expected_commit: str,
) -> types.ModuleType:
    source = _resolve_source_dir(source_dir)
    model_file = source / "model_ResNet.py"
    lock_file = source.parent / "DvXray_official_LOCK.json"
    if not model_file.is_file():
        raise FileNotFoundError(f"missing official AHCR model: {model_file}")
    if not lock_file.is_file():
        raise FileNotFoundError(f"missing official AHCR source lock: {lock_file}")

    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    locked_commit = str(lock.get("commit", ""))
    if locked_commit != expected_commit or expected_commit != OFFICIAL_COMMIT:
        raise RuntimeError(
            "official AHCR commit mismatch: "
            f"expected={OFFICIAL_COMMIT}, requested={expected_commit}, lock={locked_commit}"
        )
    actual_sha = _sha256(model_file)
    if actual_sha != OFFICIAL_MODEL_SHA256:
        raise RuntimeError(
            "official AHCR source hash mismatch: "
            f"expected={OFFICIAL_MODEL_SHA256}, actual={actual_sha}"
        )
    if source in _MODULE_CACHE:
        return _MODULE_CACHE[source]

    # Compile the locked source directly to avoid writing __pycache__ into the
    # pristine upstream checkout.
    module = types.ModuleType("dvxray_official_model_resnet_a6bfc1b")
    module.__file__ = str(model_file)
    source_code = model_file.read_text(encoding="utf-8")
    exec(compile(source_code, str(model_file), "exec"), module.__dict__)
    if not hasattr(module, "AHCR"):
        raise RuntimeError("locked official source does not expose AHCR")
    _MODULE_CACHE[source] = module
    return module


class OfficialAHCRAdapter(nn.Module):
    """Project adapter around the untouched official DvXray AHCR model."""

    def __init__(
        self,
        num_classes: int = 15,
        source_dir: str | Path = "third_party/DvXray_official",
        source_commit: str = OFFICIAL_COMMIT,
        pretrained_weights: str = "IMAGENET1K_V2",
    ) -> None:
        super().__init__()
        if int(num_classes) != 15:
            raise ValueError("the locked official AHCR reproduction expects 15 classes")
        module = _load_locked_official_module(source_dir, source_commit)
        weight_name = str(pretrained_weights).upper()
        weight_options = {
            "IMAGENET1K_V1": module.models.ResNet50_Weights.IMAGENET1K_V1,
            "IMAGENET1K_V2": module.models.ResNet50_Weights.IMAGENET1K_V2,
            "NONE": None,
        }
        if weight_name not in weight_options:
            raise ValueError(
                "official AHCR pretrained weights must be one of "
                f"{sorted(weight_options)}, got {pretrained_weights}"
            )

        # The upstream class hard-codes V1. The temporary builder override keeps
        # its architecture untouched while allowing the fair run to match the
        # ResNet50 V2 initialization used by this project's Plain-BCE baseline.
        original_builder = module.models.resnet50

        def locked_resnet50_builder(*_args, **_kwargs):
            return original_builder(weights=weight_options[weight_name])

        module.models.resnet50 = locked_resnet50_builder
        try:
            self.official_model = module.AHCR(num_classes=int(num_classes))
        finally:
            module.models.resnet50 = original_builder

        self.num_classes = int(num_classes)
        self.official_source_dir = str(_resolve_source_dir(source_dir))
        self.official_source_commit = source_commit
        self.official_source_sha256 = OFFICIAL_MODEL_SHA256
        self.official_pretrained_weights = weight_name
        self.official_fusion = "batch_invariant_confidence_weighted_probability"

    @staticmethod
    def _confidence_fusion(
        ol_logits: torch.Tensor,
        sd_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # This is the batch-invariant form of the official confidence-weighted
        # probability fusion. It is exactly equivalent to upstream for B=1.
        ol_probability = torch.sigmoid(ol_logits.float())
        sd_probability = torch.sigmoid(sd_logits.float())
        probabilities = torch.stack((ol_probability, sd_probability), dim=-1)
        confidence = torch.abs(probabilities - 0.5)
        weights = torch.softmax(confidence, dim=-1)
        fused_probability = (weights * probabilities).sum(dim=-1)
        fused_logits = torch.logit(fused_probability.clamp(1e-6, 1.0 - 1e-6))
        return fused_logits, weights

    def forward(self, image_ol: torch.Tensor, image_sd: torch.Tensor | None = None):
        if image_sd is None:
            raise ValueError("official AHCR requires paired OL and SD inputs")
        if image_ol.shape[-2:] != (224, 224) or image_sd.shape[-2:] != (224, 224):
            raise ValueError("official AHCR fixed pooling requires 224x224 inputs")
        ol_logits, sd_logits = self.official_model(image_ol, image_sd)
        fused_logits, fusion_weights = self._confidence_fusion(ol_logits, sd_logits)
        return {
            "logits": fused_logits,
            "ol_logits": ol_logits,
            "sd_logits": sd_logits,
            "official_ahcr_branch_logits": (ol_logits, sd_logits),
            "official_ahcr_fusion_weights": fusion_weights,
        }
