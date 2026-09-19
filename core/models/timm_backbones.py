# models/timm_backbones.py
from __future__ import annotations
from typing import Sequence
import os
import torch
import torch.nn as nn

try:
    from timm import create_model
except Exception as e:
    raise ImportError("timm is required; pip install timm>=0.9") from e

import os
import torch

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

# read the TORCH_HOME environment variable, otherwise use the default ~/.cache/torch
TORCH_HOME = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch"))
CKPT_DIR = os.path.join(TORCH_HOME, "hub", "checkpoints")


def _load_local_pretrained(model, filename: str):
    """Load pretrained weights from a local .safetensors or .pth file; never touches Hugging Face."""
    ckpt_path = os.path.join(CKPT_DIR, filename)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"pretrained weights not found: {ckpt_path}")

    if filename.endswith(".safetensors"):
        if load_safetensors is None:
            raise ImportError("safetensors is required: pip install safetensors")
        state_dict = load_safetensors(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    # strict=False is safer and avoids classification-head mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[local pretrained] loaded {filename}, missing={len(missing)}, unexpected={len(unexpected)}")
    return model

def _build(
    timm_name: str,
    pretrained: bool = True,
    out_indices: Sequence[int] = (1, 2, 3, 4),
    checkpoint_path: str | None = None,
) -> nn.Module:
    """
    Unified wrapper around timm.create_model:
    - features_only=True by default, returning the feature-level list;
    - when checkpoint_path is given, the weights are loaded here (strict=False),
      avoiding timm's internal Hugging Face download and the FeatureListNet naming mismatch.
    """
    # build the model first (without timm's checkpoint loading)
    model = create_model(
        timm_name,
        pretrained=(pretrained if checkpoint_path is None else False),
        features_only=True,
        out_indices=out_indices,
    )

    # without a local checkpoint, return directly (timm pretrained weights or random init)
    if not checkpoint_path:
        return model

    if not os.path.isfile(checkpoint_path):
        print(f"[timm_backbones] WARNING: checkpoint not found: {checkpoint_path}")
        return model

    print(f"[timm_backbones] Manually loading checkpoint: {checkpoint_path}")

    # ---- load the checkpoint manually and remap the keys ----
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        # the official SwinV2 weights use the `layers.0.xxx` style,
        # while the modules inside FeatureListNet use `layers_0.xxx`;
        # minimal rule: prefix 'layers.' -> 'layers_'
        if k.startswith("layers."):
            new_key = "layers_" + k[len("layers."):]
        else:
            new_key = k
        new_state[new_key] = v

    msg = model.load_state_dict(new_state, strict=False)
    # msg is normally (missing_keys, unexpected_keys)
    try:
        missing, unexpected = msg
        print(
            f"[timm_backbones] load_state_dict(strict=False) "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    except Exception:
        print(f"[timm_backbones] load_state_dict(strict=False): {msg}")

    return model


class _ChannelsLastFeatureAdapter(nn.Module):
    """Convert a timm NHWC feature pyramid to the NCHW contract used here."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.feature_info = model.feature_info
        self.dims = tuple(int(channel) for channel in model.feature_info.channels())
        self.num_features = self.dims[-1]

    def forward_features(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, x: torch.Tensor):
        features = self.model(x)
        return [feature.permute(0, 3, 1, 2).contiguous() for feature in features]


def resnet50(num_classes: int = 0, pretrained: bool = True, **kwargs) -> nn.Module:
    return _build("resnet50", pretrained=pretrained)


def resnext50_32x4d(num_classes: int = 0, pretrained: bool = True, **kwargs) -> nn.Module:
    return _build("resnext50_32x4d", pretrained=pretrained)


def regnetx_3_2gf(num_classes: int = 0, pretrained: bool = True, **kwargs) -> nn.Module:
    return _build("regnetx_3_2gf", pretrained=pretrained)


def swin_tiny(num_classes: int = 0, pretrained: bool = True, **kwargs) -> nn.Module:
    model = _build(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        out_indices=(0, 1, 2, 3),
    )
    return _ChannelsLastFeatureAdapter(model)

def maxvit_tiny(pretrained: bool = True, **kwargs):
    # 1) build the model, but never let timm download weights from Hugging Face
    model = _build("maxvit_tiny_rw_224", pretrained=False, **kwargs)

    # 2) to use pretrained weights, load them from a local safetensors file
    if pretrained:
        model = _load_local_pretrained(model, "maxvit_tiny_rw_224.sw_in1k.safetensors")
    return model


def maxvit_tiny_hf(pretrained: bool = True, **kwargs):
    """MaxViT-Tiny using timm's standard pretrained-weight resolution."""
    kwargs.pop("num_classes", None)
    return _build("maxvit_tiny_rw_224", pretrained=pretrained, **kwargs)


def coatnet_0(pretrained: bool = True, **kwargs):
    model = _build("coatnet_0_rw_224", pretrained=False, **kwargs)

    if pretrained:
        model = _load_local_pretrained(model, "coatnet_0_rw_224.sw_in1k.safetensors")
    return model

def tf_efficientnet_b5_ns(pretrained: bool = True, **kwargs) -> nn.Module:
    """
    TIMM name: 'tf_efficientnet_b5_ns'
    local weight file: tf_efficientnet_b5.ns_jft_in1k.safetensors
    """
    model = _build("tf_efficientnet_b5_ns", pretrained=False, **kwargs)

    if pretrained:
        model = _load_local_pretrained(model, "tf_efficientnet_b5.ns_jft_in1k.safetensors")
    return model

def efficientnetv2_rw_s(pretrained: bool = True, **kwargs) -> nn.Module:
    """
    TIMM name: 'efficientnetv2_rw_s'
    local weight file: efficientnetv2_rw_s.ra2_in1k.safetensors
    """
    model = _build("efficientnetv2_rw_s", pretrained=False, **kwargs)

    if pretrained:
        model = _load_local_pretrained(model, "efficientnetv2_rw_s.ra2_in1k.safetensors")
    return model

def regnety_3_2gf(pretrained: bool = True, **kwargs):
    """
    TIMM name: 'regnety_032'  (RegNetY-3.2GF)
    local weight file: regnety_032.tv2_in1k.safetensors
    """
    model = _build("regnety_032", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnety_032.tv2_in1k.safetensors")
    return model


def regnetz_4_0gf(pretrained: bool = True, **kwargs):
    """
    TIMM name: 'regnetz_040'  (RegNetZ-4.0GF)
    local weight file: regnetz_040.ra3_in1k.safetensors
    """
    model = _build("regnetz_040", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetz_040.ra3_in1k.safetensors")
    return model

def seresnext26d_32x4d(pretrained: bool = True, **kwargs):
    """
    TIMM name: 'seresnext26d_32x4d'
    local weights: seresnext26d_32x4d-80fa48a3.pth
    SE-ResNeXt-D, suitable as a 20M+ backbone.
    """
    # prevent timm from downloading weights; everything comes from local files
    model = _build("seresnext26d_32x4d", pretrained=False, **kwargs)

    if pretrained:
        model = _load_local_pretrained(model, "seresnext26d_32x4d-80fa48a3.pth")
    return model

def seresnet50(pretrained: bool = True, **kwargs):
    """
    TIMM weight id: 'seresnet50.a2_in1k'
    local weight file: seresnet50.a2_in1k.safetensors
    SE-ResNet50 with 28.1M parameters, trained with the RA2 recipe.
    """
    # create the model skeleton under the timm name 'seresnet50.ra2_in1k'
    model = _build("seresnet50.a2_in1k", pretrained=False, **kwargs)

    if pretrained:
        # everything comes from a local safetensors file; timm must not download anything
        model = _load_local_pretrained(model, "seresnet50.a2_in1k.safetensors")
    return model

# SK-ResNeXt-50
def skresnext50_32x4d(pretrained: bool = True, **kwargs):
    model = _build("skresnext50_32x4d.ra_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "skresnext50_32x4d.ra_in1k.safetensors")
    return model

# SelecSLS-42b
def selecsls42b(pretrained: bool = True, **kwargs):
    model = _build("selecsls42b.in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "selecsls42b.in1k.safetensors")
    return model

# ECA-NFNet-L0
def eca_nfnet_l0(pretrained: bool = True, **kwargs):
    model = _build("eca_nfnet_l0.ra2_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "eca_nfnet_l0.ra2_in1k.safetensors")
    return model

# NF-ResNet-50
def nf_resnet50(pretrained: bool = True, **kwargs):
    model = _build("nf_resnet50.ra2_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "nf_resnet50.ra2_in1k.safetensors")
    return model

# DenseNet-201
def densenet201(pretrained: bool = True, **kwargs):
    model = _build("densenet201.tv_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "densenet201.tv_in1k.safetensors")
    return model


# =========================
# ten new backbone wrappers
# =========================

def resnest50d_1s4x24d(pretrained: bool = True, **kwargs):
    """
    ResNeSt-50d (1s4x24d), ImageNet-1k pretrained.
    timm model name: 'resnest50d_1s4x24d.in1k'
    local weights: 'resnest50d_1s4x24d.in1k.safetensors'
    """
    model = _build("resnest50d_1s4x24d.in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "resnest50d_1s4x24d.in1k.safetensors")
    return model


def resnest50d_4s2x40d(pretrained: bool = True, **kwargs):
    """
    ResNeSt-50d (4s2x40d), ImageNet-1k pretrained.
    timm model name: 'resnest50d_4s2x40d.in1k'
    local weights: 'resnest50d_4s2x40d.in1k.safetensors'
    """
    model = _build("resnest50d_4s2x40d.in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "resnest50d_4s2x40d.in1k.safetensors")
    return model


def res2net50_26w_4s(pretrained: bool = True, **kwargs):
    """
    Res2Net-50 26w4s, ImageNet-1k pretrained.
    timm model name: 'res2net50_26w_4s.in1k'
    local weights: 'res2net50_26w_4s.in1k.safetensors'
    """
    model = _build("res2net50_26w_4s.in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "res2net50_26w_4s.in1k.safetensors")
    return model


def res2net50_26w_6s(pretrained: bool = True, **kwargs):
    """
    Res2Net-50 26w6s, ImageNet-1k pretrained.
    timm model name: 'res2net50_26w_6s.in1k'
    local weights: 'res2net50_26w_6s.in1k.safetensors'
    """
    model = _build("res2net50_26w_6s.in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "res2net50_26w_6s.in1k.safetensors")
    return model


def regnetz_d8(pretrained: bool = True, **kwargs):
    """
    RegNetZ-D8, ImageNet-1k pretrained (RA3 recipe).
    timm model name: 'regnetz_d8.ra3_in1k'
    local weights: 'regnetz_d8.ra3_in1k.safetensors'
    """
    model = _build("regnetz_d8.ra3_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetz_d8.ra3_in1k.safetensors")
    return model


def regnety_080(pretrained: bool = True, **kwargs):
    """
    RegNetY-8.0GF (TV2 recipe).
    timm model name: 'regnety_080_tv.tv2_in1k'
    local weights: 'regnety_080_tv.tv2_in1k.safetensors'
    """
    model = _build("regnety_080_tv.tv2_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnety_080_tv.tv2_in1k.safetensors")
    return model


def repvgg_b1g4(pretrained: bool = True, **kwargs):
    """
    RepVGG-B1g4, ImageNet-1k pretrained.
    timm model name: 'repvgg_b1g4.rvgg_in1k'
    local weights: 'repvgg_b1g4.rvgg_in1k.safetensors'
    """
    model = _build("repvgg_b1g4.rvgg_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "repvgg_b1g4.rvgg_in1k.safetensors")
    return model


def cspresnet50(pretrained: bool = True, **kwargs):
    """
    CSP-ResNet-50, ImageNet-1k pretrained.
    timm model name: 'cspresnet50.ra_in1k'
    local weights: 'cspresnet50.ra_in1k.safetensors'
    """
    model = _build("cspresnet50.ra_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "cspresnet50.ra_in1k.safetensors")
    return model


def cspresnext50(pretrained: bool = True, **kwargs):
    """
    CSP-ResNeXt-50, ImageNet-1k pretrained.
    timm model name: 'cspresnext50.ra_in1k'
    local weights: 'cspresnext50.ra_in1k.safetensors'
    """
    model = _build("cspresnext50.ra_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "cspresnext50.ra_in1k.safetensors")
    return model


def ecaresnet50t(pretrained: bool = True, **kwargs):
    """
    ECA-ResNet-50 (T variant), ImageNet-1k pretrained.
    timm model name: 'ecaresnet50t.a1_in1k'
    local weights: 'ecaresnet50t.a1_in1k.safetensors'
    """
    model = _build("ecaresnet50t.a1_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "ecaresnet50t.a1_in1k.safetensors")
    return model

# ====================== 20-50M classic CNN backbones ======================

def resnet34(pretrained: bool = True, **kwargs):
    """
    ResNet34 (A1 recipe, 21.8M params)
    timm id: resnet34.a1_in1k
    local weight file: resnet34.a1_in1k.safetensors
    """
    model = _build("resnet34.a1_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "resnet34.a1_in1k.safetensors")
    return model


def resnet101(pretrained: bool = True, **kwargs):
    """
    ResNet101 (TV ImageNet-1k, 44.5M params)
    timm id: resnet101.tv_in1k
    local weight file: resnet101.tv_in1k.safetensors
    """
    model = _build("resnet101.tv_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "resnet101.tv_in1k.safetensors")
    return model


def resnext101_32x4d(pretrained: bool = True, **kwargs):
    """
    ResNeXt101-32x4d (Gluon ImageNet-1k, ~44M params)
    timm id: resnext101_32x4d.gluon_in1k
    local weight file: resnext101_32x4d.gluon_in1k.safetensors
    """
    model = _build("resnext101_32x4d.gluon_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "resnext101_32x4d.gluon_in1k.safetensors")
    return model


def regnetx_040(pretrained: bool = True, **kwargs):
    """
    RegNetX-4GF, ~22M params
    timm id: regnetx_040.pycls_in1k
    local weight file: regnetx_040.pycls_in1k.safetensors
    """
    model = _build("regnetx_040.pycls_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetx_040.pycls_in1k.safetensors")
    return model


def regnetx_064(pretrained: bool = True, **kwargs):
    """
    RegNetX-6.4GF, 26.2M params
    timm id: regnetx_064.pycls_in1k
    local weight file: regnetx_064.pycls_in1k.safetensors
    """
    model = _build("regnetx_064.pycls_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetx_064.pycls_in1k.safetensors")
    return model


def regnetx_080(pretrained: bool = True, **kwargs):
    """
    RegNetX-8GF, 39.6M params
    timm id: regnetx_080.tv2_in1k
    local weight file: regnetx_080.tv2_in1k.safetensors
    """
    model = _build("regnetx_080.tv2_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetx_080.tv2_in1k.safetensors")
    return model


def regnetx_120(pretrained: bool = True, **kwargs):
    """
    RegNetX-12GF, 46.1M params
    timm id: regnetx_120.pycls_in1k
    local weight file: regnetx_120.pycls_in1k.safetensors
    """
    model = _build("regnetx_120.pycls_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetx_120.pycls_in1k.safetensors")
    return model


def regnetv_040(pretrained: bool = True, **kwargs):
    """
    RegNetV-4GF, 20.6M params
    timm id: regnetv_040.ra3_in1k
    local weight file: regnetv_040.ra3_in1k.safetensors
    """
    model = _build("regnetv_040.ra3_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetv_040.ra3_in1k.safetensors")
    return model


def inception_v3(pretrained: bool = True, **kwargs):
    """
    Inception-v3 (TV, 23.8M params)
    timm id: inception_v3.tv_in1k
    local weight file: inception_v3.tv_in1k.safetensors
    """
    model = _build("inception_v3.tv_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "inception_v3.tv_in1k.safetensors")
    return model


def densenet161(pretrained: bool = True, **kwargs):
    """
    DenseNet-161 (28.7M params)
    timm id: densenet161.tv_in1k
    local weight file: densenet161.tv_in1k.safetensors
    """
    model = _build("densenet161.tv_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "densenet161.tv_in1k.safetensors")
    return model


