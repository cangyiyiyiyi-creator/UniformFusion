from typing import Dict, List, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attentions import ATTENTION_REGISTRY


# ==============================
# 基础模块
# ==============================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ParallelAttentionModule(nn.Module):
    """
    并联多个注意力模块，然后把输出求和。
    """
    def __init__(self, attn_modules: List[nn.Module]):
        super().__init__()
        self.attns = nn.ModuleList(attn_modules)

    def forward(self, x):
        outputs = [attn(x) for attn in self.attns]
        return sum(outputs)


# ==============================
# FPN
# ==============================

class FPN(nn.Module):
    """
    最小 FPN：输入 C3/C4/C5，输出 P3/P4/P5
    """
    def __init__(self, c3, c4, c5, out_channels=256):
        super().__init__()
        self.l3 = nn.Conv2d(c3, out_channels, 1)
        self.l4 = nn.Conv2d(c4, out_channels, 1)
        self.l5 = nn.Conv2d(c5, out_channels, 1)

        self.s3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.s4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.s5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, C3, C4, C5):
        P5 = self.l5(C5)
        P4 = self.l4(C4) + F.interpolate(P5, size=C4.shape[-2:], mode='nearest')
        P3 = self.l3(C3) + F.interpolate(P4, size=C3.shape[-2:], mode='nearest')
        return self.s3(P3), self.s4(P4), self.s5(P5)


# ==============================
# FPN + PAN + Attention
# ==============================

class FPN_PAN(nn.Module):
    """
    结构：
      Top-Down:  C5 -> P5, C4 + up(P5) -> P4, C3 + up(P4) -> P3
      Bottom-Up: P3 -> N3, P4 + down(N3) -> N4, P5 + down(N4) -> N5
      Attention: 分别可挂在 N3 / N4 / N5
    """
    def __init__(
        self,
        c3,
        c4,
        c5,
        out_channels=256,
        attention_config: Optional[Dict] = None,
    ):
        super().__init__()

        # ---------- Top-Down FPN ----------
        self.fpn_l3 = nn.Conv2d(c3, out_channels, 1)
        self.fpn_l4 = nn.Conv2d(c4, out_channels, 1)
        self.fpn_l5 = nn.Conv2d(c5, out_channels, 1)

        self.fpn_s3 = ConvBNAct(out_channels, out_channels, k=3, p=1, act=False)
        self.fpn_s4 = ConvBNAct(out_channels, out_channels, k=3, p=1, act=False)
        self.fpn_s5 = ConvBNAct(out_channels, out_channels, k=3, p=1, act=False)

        # ---------- Bottom-Up PAN ----------
        self.pan_d4 = ConvBNAct(out_channels, out_channels, k=3, s=2, p=1, act=False)
        self.pan_d5 = ConvBNAct(out_channels, out_channels, k=3, s=2, p=1, act=False)

        self.pan_s3 = ConvBNAct(out_channels, out_channels, k=3, p=1, act=False)
        self.pan_s4 = ConvBNAct(out_channels, out_channels, k=3, p=1, act=False)
        self.pan_s5 = ConvBNAct(out_channels, out_channels, k=3, p=1, act=False)

        # ---------- Attention ----------
        self.attn3 = None
        self.attn4 = None
        self.attn5 = None

        if attention_config is not None:
            if "N3" in attention_config:
                self.attn3 = self._build_attention_block(attention_config["N3"], out_channels)
            if "N4" in attention_config:
                self.attn4 = self._build_attention_block(attention_config["N4"], out_channels)
            if "N5" in attention_config:
                self.attn5 = self._build_attention_block(attention_config["N5"], out_channels)

    # ------------------------------
    # Attention Builder
    # ------------------------------
    def _build_attention_block(self, config: Union[str, list, dict], channels: int) -> nn.Module:
        """
        支持四种格式：

        1) 字符串
           "freq_route"

        2) 串联
           ["freq_route", "polarity"]

        3) 并联
           {"parallel": ["freq_route", "granularity"]}

        4) 参数化
           {"type": "proto_route", "num_prototypes": 8, "temperature": 1.0}
        """
        if config is None:
            return None
        if isinstance(config, str):
            key = config.lower()
            attn_class = ATTENTION_REGISTRY.get(key, None)
            if attn_class is None:
                raise ValueError(f"未知注意力类型: {config}")
            return attn_class(channels)

        elif isinstance(config, list):
            modules = [self._build_attention_block(c, channels) for c in config]
            return nn.Sequential(*modules)

        elif isinstance(config, dict):
            if "parallel" in config:
                modules = [self._build_attention_block(c, channels) for c in config["parallel"]]
                return ParallelAttentionModule(modules)

            if "type" in config:
                attn_type = config["type"].lower()
                attn_class = ATTENTION_REGISTRY.get(attn_type, None)
                if attn_class is None:
                    raise ValueError(f"未知注意力类型: {attn_type}")

                kwargs = {k: v for k, v in config.items() if k != "type"}
                return attn_class(channels, **kwargs)

            raise TypeError(f"不支持的注意力配置格式: {config}")

        else:
            raise TypeError(f"不支持的注意力配置格式: {config}")

    # ------------------------------
    # Forward
    # ------------------------------
    def forward(self, C3, C4, C5):
        # ----- Top-Down -----
        P5 = self.fpn_s5(self.fpn_l5(C5))
        P4 = self.fpn_s4(self.fpn_l4(C4) + F.interpolate(P5, size=C4.shape[-2:], mode='nearest'))
        P3 = self.fpn_s3(self.fpn_l3(C3) + F.interpolate(P4, size=C3.shape[-2:], mode='nearest'))

        # ----- Bottom-Up -----
        N3 = self.pan_s3(P3)
        N4 = self.pan_s4(P4 + self.pan_d4(N3))
        N5 = self.pan_s5(P5 + self.pan_d5(N4))

        # ----- Attention -----
        if self.attn3 is not None:
            N3 = self.attn3(N3)
        if self.attn4 is not None:
            N4 = self.attn4(N4)
        if self.attn5 is not None:
            N5 = self.attn5(N5)

        return N3, N4, N5