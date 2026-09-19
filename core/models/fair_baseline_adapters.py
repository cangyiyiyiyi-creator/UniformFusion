"""Project-side adapters for locked third-party fair baselines."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def _locked_source(source_dir: str, expected_commit: str) -> Path:
    root = Path(source_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing third-party source: {root}")
    head = root / ".git" / "HEAD"
    if not head.is_file():
        raise RuntimeError(f"third-party source is not a Git checkout: {root}")
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        value = (root / ".git" / value[5:]).read_text(encoding="utf-8").strip()
    if expected_commit and value != expected_commit:
        raise RuntimeError(
            f"third-party commit mismatch for {root}: {value} != {expected_commit}"
        )
    return root


class DAGNetAdapter(nn.Module):
    """Expose the locked official DAGNet model through the project contract."""

    required_input_size = 256

    def __init__(self, num_classes: int, source_dir: str, source_commit: str):
        super().__init__()
        root = _locked_source(source_dir, source_commit)
        source_text = str(root)
        sys.path.insert(0, source_text)
        try:
            module = importlib.import_module("model.model_v2")
            # expansion=2 is the configuration selected by the official train.py.
            self.model = module.MyModel(
                num_classes=num_classes,
                depth_mult=3,
                expansion=2,
                dropout=0.2,
                backbone="resnet",
            )
        finally:
            if sys.path and sys.path[0] == source_text:
                sys.path.pop(0)

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor) -> torch.Tensor:
        return self.model(view_a, view_b)


class MLDecoderDualViewAdapter(nn.Module):
    """Shared ResNet50 and additive C5 fusion with the official ML-Decoder."""

    def __init__(
        self,
        num_classes: int,
        source_dir: str,
        source_commit: str,
        decoder_embedding: int = 768,
    ):
        super().__init__()
        root = _locked_source(source_dir, source_commit)
        source_text = str(root)
        sys.path.insert(0, source_text)
        try:
            module = importlib.import_module("src_files.ml_decoder.ml_decoder")
            # PyTorch >= 2.6 inspects ``self_attn`` when cloning decoder layers;
            # the locked official implementation names the same block
            # ``multihead_attn``. A property preserves parameters and behavior.
            layer_cls = module.TransformerDecoderLayerOptimal
            if not hasattr(layer_cls, "self_attn"):
                layer_cls.self_attn = property(
                    lambda layer: layer.multihead_attn
                )
            if not getattr(layer_cls, "_project_causal_compat", False):
                official_forward = layer_cls.forward

                def compatible_forward(layer, *args, **kwargs):
                    kwargs.pop("tgt_is_causal", None)
                    kwargs.pop("memory_is_causal", None)
                    return official_forward(layer, *args, **kwargs)

                layer_cls.forward = compatible_forward
                layer_cls._project_causal_compat = True
            decoder_cls = module.MLDecoder
        finally:
            if sys.path and sys.path[0] == source_text:
                sys.path.pop(0)

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.decoder = decoder_cls(
            num_classes=num_classes,
            initial_num_features=2048,
            num_of_groups=num_classes,
            decoder_embedding=decoder_embedding,
            zsl=0,
        )

    def _features(self, image: torch.Tensor) -> torch.Tensor:
        image = self.stem(image)
        image = self.layer1(image)
        image = self.layer2(image)
        image = self.layer3(image)
        return self.layer4(image)

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor) -> torch.Tensor:
        return self.decoder(self._features(view_a) + self._features(view_b))
