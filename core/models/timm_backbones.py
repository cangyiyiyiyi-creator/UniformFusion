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

# 读环境变量 TORCH_HOME，否则用默认 ~/.cache/torch
TORCH_HOME = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch"))
CKPT_DIR = os.path.join(TORCH_HOME, "hub", "checkpoints")


def _load_local_pretrained(model, filename: str):
    """从本地 .safetensors 或 .pth 加载预训练权重，完全不走 HF。"""
    ckpt_path = os.path.join(CKPT_DIR, filename)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"预训练权重不存在: {ckpt_path}")

    if filename.endswith(".safetensors"):
        if load_safetensors is None:
            raise ImportError("需要安装 safetensors：pip install safetensors")
        state_dict = load_safetensors(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    # 避免分类头不匹配之类的问题，strict=False 更稳
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
    统一封装 timm.create_model:
    - 默认 features_only=True，返回特征层列表；
    - 如果给了 checkpoint_path：我们自己加载权重（strict=False），
      避免 timm 内部的 HF 下载 / FeatureListNet 命名不匹配问题。
    """
    # 先构建模型（暂时不走 timm 的 checkpoint 加载）
    model = create_model(
        timm_name,
        pretrained=(pretrained if checkpoint_path is None else False),
        features_only=True,
        out_indices=out_indices,
    )

    # 没有本地 ckpt 就直接返回（用 timm 自己的预训练或随机初始化）
    if not checkpoint_path:
        return model

    if not os.path.isfile(checkpoint_path):
        print(f"[timm_backbones] WARNING: checkpoint not found: {checkpoint_path}")
        return model

    print(f"[timm_backbones] Manually loading checkpoint: {checkpoint_path}")

    # ---- 手动加载 checkpoint，并做 key 映射 ----
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
        # SwinV2 官方权重是 `layers.0.xxx` 风格，
        # FeatureListNet 里的模块是 `layers_0.xxx` 风格；
        # 做一个最小规则：前缀 'layers.' -> 'layers_'
        if k.startswith("layers."):
            new_key = "layers_" + k[len("layers."):]
        else:
            new_key = k
        new_state[new_key] = v

    msg = model.load_state_dict(new_state, strict=False)
    # msg 通常是 (missing_keys, unexpected_keys)
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
    # ① 先构建模型，但绝对禁止 timm 自己去 HF 下权重
    model = _build("maxvit_tiny_rw_224", pretrained=False, **kwargs)

    # ② 想用预训练就从本地 safetensors 加载
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
    TIMM 名称: 'tf_efficientnet_b5_ns'
    本地权重文件: tf_efficientnet_b5.ns_jft_in1k.safetensors
    """
    model = _build("tf_efficientnet_b5_ns", pretrained=False, **kwargs)

    if pretrained:
        model = _load_local_pretrained(model, "tf_efficientnet_b5.ns_jft_in1k.safetensors")
    return model

def efficientnetv2_rw_s(pretrained: bool = True, **kwargs) -> nn.Module:
    """
    TIMM 名称: 'efficientnetv2_rw_s'
    本地权重文件: efficientnetv2_rw_s.ra2_in1k.safetensors
    """
    model = _build("efficientnetv2_rw_s", pretrained=False, **kwargs)

    if pretrained:
        model = _load_local_pretrained(model, "efficientnetv2_rw_s.ra2_in1k.safetensors")
    return model

def regnety_3_2gf(pretrained: bool = True, **kwargs):
    """
    TIMM 名称: 'regnety_032'  （RegNetY-3.2GF）
    本地权重文件: regnety_032.tv2_in1k.safetensors
    """
    model = _build("regnety_032", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnety_032.tv2_in1k.safetensors")
    return model


def regnetz_4_0gf(pretrained: bool = True, **kwargs):
    """
    TIMM 名称: 'regnetz_040'  （RegNetZ-4.0GF）
    本地权重文件: regnetz_040.ra3_in1k.safetensors
    """
    model = _build("regnetz_040", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetz_040.ra3_in1k.safetensors")
    return model

def seresnext26d_32x4d(pretrained: bool = True, **kwargs):
    """
    TIMM 名称: 'seresnext26d_32x4d'
    本地权重: seresnext26d_32x4d-80fa48a3.pth
    SE-ResNeXt-D，适合做 20M+ 级 backbone。
    """
    # 不让 timm 自己下权重，完全走本地
    model = _build("seresnext26d_32x4d", pretrained=False, **kwargs)

    if pretrained:
        model = _load_local_pretrained(model, "seresnext26d_32x4d-80fa48a3.pth")
    return model

def seresnet50(pretrained: bool = True, **kwargs):
    """
    TIMM 权重 ID: 'seresnet50.a2_in1k'
    本地权重文件: seresnet50.a2_in1k.safetensors
    28.1M 参数的 SE-ResNet50，用 RA2 配方训练。:contentReference[oaicite:3]{index=3}
    """
    # 用 timm 里的 'seresnet50.ra2_in1k' 这个名字创建模型骨架
    model = _build("seresnet50.a2_in1k", pretrained=False, **kwargs)

    if pretrained:
        # 完全走本地 safetensors，不让 timm 自己下
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
# 新增 10 个 backbone 封装
# =========================

def resnest50d_1s4x24d(pretrained: bool = True, **kwargs):
    """
    ResNeSt-50d (1s4x24d), ImageNet-1k 预训练。
    timm 模型名: 'resnest50d_1s4x24d.in1k'
    本地权重:    'resnest50d_1s4x24d.in1k.safetensors'
    """
    model = _build("resnest50d_1s4x24d.in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "resnest50d_1s4x24d.in1k.safetensors")
    return model


def resnest50d_4s2x40d(pretrained: bool = True, **kwargs):
    """
    ResNeSt-50d (4s2x40d), ImageNet-1k 预训练。
    timm 模型名: 'resnest50d_4s2x40d.in1k'
    本地权重:    'resnest50d_4s2x40d.in1k.safetensors'
    """
    model = _build("resnest50d_4s2x40d.in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "resnest50d_4s2x40d.in1k.safetensors")
    return model


def res2net50_26w_4s(pretrained: bool = True, **kwargs):
    """
    Res2Net-50 26w4s, ImageNet-1k 预训练。
    timm 模型名: 'res2net50_26w_4s.in1k'
    本地权重:    'res2net50_26w_4s.in1k.safetensors'
    """
    model = _build("res2net50_26w_4s.in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "res2net50_26w_4s.in1k.safetensors")
    return model


def res2net50_26w_6s(pretrained: bool = True, **kwargs):
    """
    Res2Net-50 26w6s, ImageNet-1k 预训练。
    timm 模型名: 'res2net50_26w_6s.in1k'
    本地权重:    'res2net50_26w_6s.in1k.safetensors'
    """
    model = _build("res2net50_26w_6s.in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "res2net50_26w_6s.in1k.safetensors")
    return model


def regnetz_d8(pretrained: bool = True, **kwargs):
    """
    RegNetZ-D8, ImageNet-1k 预训练 (RA3 配方)。
    timm 模型名: 'regnetz_d8.ra3_in1k'
    本地权重:    'regnetz_d8.ra3_in1k.safetensors'
    """
    model = _build("regnetz_d8.ra3_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetz_d8.ra3_in1k.safetensors")
    return model


def regnety_080(pretrained: bool = True, **kwargs):
    """
    RegNetY-8.0GF (TV2 配方)。
    timm 模型名: 'regnety_080_tv.tv2_in1k'
    本地权重:    'regnety_080_tv.tv2_in1k.safetensors'
    """
    model = _build("regnety_080_tv.tv2_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnety_080_tv.tv2_in1k.safetensors")
    return model


def repvgg_b1g4(pretrained: bool = True, **kwargs):
    """
    RepVGG-B1g4, ImageNet-1k 预训练。
    timm 模型名: 'repvgg_b1g4.rvgg_in1k'
    本地权重:    'repvgg_b1g4.rvgg_in1k.safetensors'
    """
    model = _build("repvgg_b1g4.rvgg_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "repvgg_b1g4.rvgg_in1k.safetensors")
    return model


def cspresnet50(pretrained: bool = True, **kwargs):
    """
    CSP-ResNet-50, ImageNet-1k 预训练。
    timm 模型名: 'cspresnet50.ra_in1k'
    本地权重:    'cspresnet50.ra_in1k.safetensors'
    """
    model = _build("cspresnet50.ra_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "cspresnet50.ra_in1k.safetensors")
    return model


def cspresnext50(pretrained: bool = True, **kwargs):
    """
    CSP-ResNeXt-50, ImageNet-1k 预训练。
    timm 模型名: 'cspresnext50.ra_in1k'
    本地权重:    'cspresnext50.ra_in1k.safetensors'
    """
    model = _build("cspresnext50.ra_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "cspresnext50.ra_in1k.safetensors")
    return model


def ecaresnet50t(pretrained: bool = True, **kwargs):
    """
    ECA-ResNet-50 (T 版本)，ImageNet-1k 预训练。
    timm 模型名: 'ecaresnet50t.a1_in1k'
    本地权重:    'ecaresnet50t.a1_in1k.safetensors'
    """
    model = _build("ecaresnet50t.a1_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "ecaresnet50t.a1_in1k.safetensors")
    return model

# ====================== 20–50M 经典 CNN backbones ======================

def resnet34(pretrained: bool = True, **kwargs):
    """
    ResNet34 (A1 recipe, 21.8M params)
    timm id: resnet34.a1_in1k
    本地权重文件: resnet34.a1_in1k.safetensors
    """
    model = _build("resnet34.a1_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "resnet34.a1_in1k.safetensors")
    return model


def resnet101(pretrained: bool = True, **kwargs):
    """
    ResNet101 (TV ImageNet-1k, 44.5M params)
    timm id: resnet101.tv_in1k
    本地权重文件: resnet101.tv_in1k.safetensors
    """
    model = _build("resnet101.tv_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "resnet101.tv_in1k.safetensors")
    return model


def resnext101_32x4d(pretrained: bool = True, **kwargs):
    """
    ResNeXt101-32x4d (Gluon ImageNet-1k, ~44M params)
    timm id: resnext101_32x4d.gluon_in1k
    本地权重文件: resnext101_32x4d.gluon_in1k.safetensors
    """
    model = _build("resnext101_32x4d.gluon_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "resnext101_32x4d.gluon_in1k.safetensors")
    return model


def regnetx_040(pretrained: bool = True, **kwargs):
    """
    RegNetX-4GF, ~22M params
    timm id: regnetx_040.pycls_in1k
    本地权重文件: regnetx_040.pycls_in1k.safetensors
    """
    model = _build("regnetx_040.pycls_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetx_040.pycls_in1k.safetensors")
    return model


def regnetx_064(pretrained: bool = True, **kwargs):
    """
    RegNetX-6.4GF, 26.2M params
    timm id: regnetx_064.pycls_in1k
    本地权重文件: regnetx_064.pycls_in1k.safetensors
    """
    model = _build("regnetx_064.pycls_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetx_064.pycls_in1k.safetensors")
    return model


def regnetx_080(pretrained: bool = True, **kwargs):
    """
    RegNetX-8GF, 39.6M params
    timm id: regnetx_080.tv2_in1k
    本地权重文件: regnetx_080.tv2_in1k.safetensors
    """
    model = _build("regnetx_080.tv2_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetx_080.tv2_in1k.safetensors")
    return model


def regnetx_120(pretrained: bool = True, **kwargs):
    """
    RegNetX-12GF, 46.1M params
    timm id: regnetx_120.pycls_in1k
    本地权重文件: regnetx_120.pycls_in1k.safetensors
    """
    model = _build("regnetx_120.pycls_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetx_120.pycls_in1k.safetensors")
    return model


def regnetv_040(pretrained: bool = True, **kwargs):
    """
    RegNetV-4GF, 20.6M params
    timm id: regnetv_040.ra3_in1k
    本地权重文件: regnetv_040.ra3_in1k.safetensors
    """
    model = _build("regnetv_040.ra3_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "regnetv_040.ra3_in1k.safetensors")
    return model


def inception_v3(pretrained: bool = True, **kwargs):
    """
    Inception-v3 (TV, 23.8M params)
    timm id: inception_v3.tv_in1k
    本地权重文件: inception_v3.tv_in1k.safetensors
    """
    model = _build("inception_v3.tv_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "inception_v3.tv_in1k.safetensors")
    return model


def densenet161(pretrained: bool = True, **kwargs):
    """
    DenseNet-161 (28.7M params)
    timm id: densenet161.tv_in1k
    本地权重文件: densenet161.tv_in1k.safetensors
    """
    model = _build("densenet161.tv_in1k", pretrained=False, **kwargs)
    if pretrained:
        model = _load_local_pretrained(model, "densenet161.tv_in1k.safetensors")
    return model


