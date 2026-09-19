# models/tv_backbones.py  (FIX: avoid recursion by aliasing torchvision callables)
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

# ---- use aliases to avoid name collision with our builder functions ----
from torchvision.models import (
    resnet50 as tv_resnet50, ResNet50_Weights,
    resnext50_32x4d as tv_resnext50_32x4d, ResNeXt50_32X4D_Weights,
    regnet_x_3_2gf as tv_regnet_x_3_2gf, RegNet_X_3_2GF_Weights
)

class _TVBackbone(nn.Module):
    """
    Export forward_features(x) -> [C2, C3, C4, C5]
    and compute dims=(c2,c3,c4,c5) at init time for FPN/PAN.
    """
    def __init__(self, model: nn.Module, kind: str):
        super().__init__()
        self.model = model
        self.kind = kind.lower()
        self.dims: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.num_features: int = 0

        # compute channel dims once with a dummy pass (CPU-safe)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.forward_features(dummy)
            chs = [int(f.shape[1]) for f in feats]
        self.dims = tuple(chs)
        self.num_features = int(self.dims[-1])

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.kind in ("resnet", "resnext"):
            return self._forward_resnet_family(x)
        elif self.kind in ("regnet",):
            return self._forward_regnet(x)
        else:
            raise ValueError(f"Unsupported kind: {self.kind}")

    def _forward_resnet_family(self, x: torch.Tensor) -> List[torch.Tensor]:
        m = self.model
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        c2 = m.layer1(x)   # stride 4
        c3 = m.layer2(c2)  # stride 8
        c4 = m.layer3(c3)  # stride 16
        c5 = m.layer4(c4)  # stride 32
        return [c2, c3, c4, c5]

    def _forward_regnet(self, x: torch.Tensor) -> List[torch.Tensor]:
        m = self.model
        x = m.stem(x)
        feats: List[torch.Tensor] = []
        for _, mod in m.trunk_output.named_children():
            x = mod(x)
            feats.append(x)
        if len(feats) < 4:
            raise RuntimeError(f"Unexpected RegNet stages (<4): got {len(feats)} from trunk_output")
        return feats[-4:]  # use last four as C2..C5

# ---------- public builders (names match --model) ----------
def resnet50(num_classes: int = 0, pretrained: bool = True, **kwargs) -> nn.Module:
    m = tv_resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    return _TVBackbone(m, kind="resnet")

def resnext50_32x4d(num_classes: int = 0, pretrained: bool = True, **kwargs) -> nn.Module:
    m = tv_resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None)
    return _TVBackbone(m, kind="resnext")

def regnetx_3_2gf(num_classes: int = 0, pretrained: bool = True, **kwargs) -> nn.Module:
    m = tv_regnet_x_3_2gf(weights=RegNet_X_3_2GF_Weights.IMAGENET1K_V1 if pretrained else None)
    return _TVBackbone(m, kind="regnet")
