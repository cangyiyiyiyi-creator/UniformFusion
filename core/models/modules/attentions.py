import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# 工具模块
# ==============================

def _choose_groups(channels: int, groups: int) -> int:
    groups = max(1, int(groups))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class LayerScale(nn.Module):
    def __init__(self, channels: int, init_value: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(1, channels, 1, 1))

    def forward(self, x):
        return x * self.gamma


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ==============================
# 原有注意力（保留）
# ==============================

class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg(x))


class ECAAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = SEAttention(channels, reduction=reduction)
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.ca(x)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        s = self.sa(torch.cat([max_pool, avg_pool], dim=1))
        return x * s


# ==============================
# 1. 论文级频率路由注意力
# Complementary Frequency Routing Attention
# ==============================

class FrequencyRoutingAttention(nn.Module):
    """
    论文级升级版：
    1) same-size 低频提取，避免 shape 错位
    2) 低频 / 高频互补分解
    3) 通道组路由 + 空间路由
    4) 预算守恒竞争：w_low + w_high = 1
    5) 频率重组 + 安全残差
    """

    def __init__(
        self,
        channels,
        groups=4,
        reduction=8,
        kernel_size=3,
        init_scale=1e-3,
    ):
        super().__init__()
        self.channels = channels
        self.groups = _choose_groups(channels, groups)
        self.group_channels = channels // self.groups

        hidden = max(channels // reduction, 8)

        # same-size 可学习低通
        self.low_pass = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        with torch.no_grad():
            self.low_pass.weight.fill_(1.0 / 9.0)

        # 高频增强
        self.high_proj = ConvBNAct(
            channels,
            channels,
            k=kernel_size,
            p=kernel_size // 2,
            g=channels,
            act=False
        )

        # 通道组路由 -> [B, 2g, 1, 1]
        self.channel_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.groups * 2, 1, bias=True)
        )

        # 空间路由 -> [B, 2, H, W]
        self.spatial_router = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 2, 1, bias=True)
        )

        # 频率重组
        self.recompose = nn.Sequential(
            ConvBNAct(channels, channels, k=1, act=True),
            ConvBNAct(channels, channels, k=3, p=1, g=channels, act=False),
        )

        self.scale = LayerScale(channels, init_value=init_scale)

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.groups
        cg = self.group_channels

        # 低频 / 高频互补分解
        low = self.low_pass(x)      # [B,C,H,W]
        high = x - low              # [B,C,H,W]
        high = self.high_proj(high) # [B,C,H,W]

        # reshape 成 group 形式
        low_g = low.view(B, g, cg, H, W)    # [B,g,cg,H,W]
        high_g = high.view(B, g, cg, H, W)  # [B,g,cg,H,W]

        # 通道路由
        ch_feat = torch.cat([low, high], dim=1)          # [B,2C,H,W]
        ch_logits = self.channel_router(ch_feat)         # [B,2g,1,1]
        ch_logits = ch_logits.view(B, g, 2, 1, 1)       # [B,g,2,1,1]

        # 空间路由
        spatial_stat = torch.cat([
            torch.mean(low, dim=1, keepdim=True),        # [B,1,H,W]
            torch.mean(high.abs(), dim=1, keepdim=True), # [B,1,H,W]
        ], dim=1)                                        # [B,2,H,W]

        sp_logits = self.spatial_router(spatial_stat)    # [B,2,H,W]
        sp_logits = sp_logits.unsqueeze(1)               # [B,1,2,H,W]

        # 联合 logits -> [B,g,2,H,W]
        logits = ch_logits.expand(-1, -1, -1, H, W) + sp_logits.expand(-1, g, -1, -1, -1)
        weights = torch.softmax(logits, dim=2)

        # 关键修复：加通道维，保证和 [B,g,cg,H,W] 正确 broadcast
        w_low = weights[:, :, 0].unsqueeze(2)    # [B,g,1,H,W]
        w_high = weights[:, :, 1].unsqueeze(2)   # [B,g,1,H,W]

        fused = w_low * low_g + w_high * high_g  # [B,g,cg,H,W]
        fused = fused.reshape(B, C, H, W)

        fused = self.recompose(fused)
        return x + self.scale(fused)


# ==============================
# 2. 论文级符极关系注意力
# Support-Inhibit Polarity Attention
# ==============================

class PolarityAttention(nn.Module):
    """
    论文级升级版：
    1) 支持证据 / 抑制证据双分支
    2) 正负响应分别建模
    3) 通道门 + 空间门联合调制
    4) 最后关系重组
    """
    def __init__(self, channels, reduction=8, init_scale=1e-3):
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.pre = ConvBNAct(channels, channels, k=1, act=False)

        self.support_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.inhibit_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels * 2, 1, bias=True)
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 2, 1, bias=True)
        )

        self.recompose = ConvBNAct(channels, channels, k=1, act=False)
        self.scale = LayerScale(channels, init_value=init_scale)

    def forward(self, x):
        y = self.pre(x)

        support = self.support_proj(F.relu(y))
        inhibit = self.inhibit_proj(F.relu(-y))

        # 通道门
        ch_gate = self.channel_gate(torch.cat([support, inhibit], dim=1))
        ch_sup, ch_inh = torch.chunk(ch_gate, 2, dim=1)
        ch_sup = torch.sigmoid(ch_sup)
        ch_inh = torch.sigmoid(ch_inh)

        # 空间门
        sp_in = torch.cat([
            torch.mean(support, dim=1, keepdim=True),
            torch.max(support, dim=1, keepdim=True)[0],
            torch.mean(inhibit, dim=1, keepdim=True),
            torch.max(inhibit, dim=1, keepdim=True)[0],
        ], dim=1)

        sp_gate = self.spatial_gate(sp_in)
        sp_sup = torch.sigmoid(sp_gate[:, 0:1])
        sp_inh = torch.sigmoid(sp_gate[:, 1:2])

        out = ch_sup * sp_sup * support - ch_inh * sp_inh * inhibit
        out = self.recompose(out)

        return x + self.scale(out)


# ==============================
# 3. 论文级原型路由注意力
# Latent Prototype Routing Attention
# ==============================

class PrototypeRoutingAttention(nn.Module):
    """
    论文级升级版：
    1) 可学习原型字典
    2) 归一化相似度分配
    3) 图像级原型 gate
    4) 原型重建 + 重组
    """
    def __init__(self, channels, num_prototypes=8, temperature=1.0, reduction=8, init_scale=1e-3):
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.temperature = temperature

        self.pre = ConvBNAct(channels, channels, k=1, act=False)

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, channels))
        nn.init.normal_(self.prototypes, std=0.02)

        hidden = max(channels // reduction, 8)
        self.prototype_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, num_prototypes, 1, bias=True)
        )

        self.recompose = nn.Sequential(
            ConvBNAct(channels, channels, k=1, act=True),
            ConvBNAct(channels, channels, k=3, p=1, g=channels, act=False)
        )
        self.scale = LayerScale(channels, init_value=init_scale)

    def forward(self, x):
        B, C, H, W = x.shape

        feat = self.pre(x).flatten(2).transpose(1, 2)  # [B,N,C]
        feat_n = F.normalize(feat, dim=-1)

        proto = F.normalize(self.prototypes, dim=-1)   # [P,C]
        sim = torch.matmul(feat_n, proto.t()) / max(self.temperature, 1e-6)
        assign = F.softmax(sim, dim=-1)                # [B,N,P]

        # 图像级 prototype gate
        p_gate = self.prototype_gate(x).flatten(2).transpose(1, 2)  # [B,1,P]
        p_gate = F.softmax(p_gate, dim=-1)

        assign = assign * p_gate
        assign = assign / (assign.sum(dim=-1, keepdim=True) + 1e-6)

        recon = torch.matmul(assign, self.prototypes)  # [B,N,C]
        recon = recon.transpose(1, 2).reshape(B, C, H, W)
        recon = self.recompose(recon)

        return x + self.scale(recon)


# ==============================
# 4. 论文级自反馈注意力
# Confidence Feedback Attention
# ==============================

class SelfFeedbackAttention(nn.Module):
    """
    论文级升级版：
    1) 粗注意力 -> 置信图 -> 精注意力
    2) 边界辅助约束
    3) 二阶段反馈修正
    """
    def __init__(self, channels, reduction=8, init_scale=1e-3):
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.coarse = nn.Sequential(
            ConvBNAct(channels, channels, k=1, act=True),
            nn.Conv2d(channels, channels, 1, bias=True)
        )

        self.confidence_head = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True)
        )

        self.refine = nn.Sequential(
            ConvBNAct(channels, channels, k=3, p=1, g=channels, act=True),
            nn.Conv2d(channels, channels, 1, bias=True)
        )

        self.boundary = nn.Conv2d(channels, 1, 3, padding=1, bias=False)
        self.recompose = ConvBNAct(channels, channels, k=1, act=False)
        self.scale = LayerScale(channels, init_value=init_scale)

    def forward(self, x):
        coarse_attn = torch.sigmoid(self.coarse(x))
        coarse_feat = x * coarse_attn

        confidence = torch.sigmoid(self.confidence_head(coarse_feat))
        boundary = torch.sigmoid(self.boundary(coarse_feat.abs()))
        guide = torch.clamp(0.7 * confidence + 0.3 * boundary, 0.0, 1.0)

        refined_attn = torch.sigmoid(self.refine(coarse_feat)) * guide
        out = self.recompose(x * refined_attn)

        return x + self.scale(out)


# ==============================
# 5. 论文级粒度竞争注意力
# Granularity Competition Attention
# ==============================

class GranularityAttention(nn.Module):
    """
    论文级升级版：
    1) 多感受野专家
    2) 尺度竞争而不是简单求和
    3) 通道级 + 空间级联合尺度分配
    """
    def __init__(self, channels, kernel_sizes=(3, 5, 7), reduction=8, init_scale=1e-3):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.num_scales = len(self.kernel_sizes)

        self.experts = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(channels, channels, k=k, p=k // 2, g=channels, act=True),
                ConvBNAct(channels, channels, k=1, act=False)
            )
            for k in self.kernel_sizes
        ])

        hidden = max(channels // reduction, 8)

        self.channel_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_scales, 1, bias=True)
        )

        self.spatial_router = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_scales, 1, bias=True)
        )

        self.recompose = ConvBNAct(channels, channels, k=1, act=False)
        self.scale = LayerScale(channels, init_value=init_scale)

    def forward(self, x):
        feats = [expert(x) for expert in self.experts]    # list of [B,C,H,W]
        stack = torch.stack(feats, dim=1)                 # [B,S,C,H,W]

        ch_logits = self.channel_router(x).unsqueeze(2)   # [B,S,1,1,1]
        sp_logits = self.spatial_router(x).unsqueeze(2)   # [B,S,1,H,W]

        logits = ch_logits + sp_logits
        weights = torch.softmax(logits, dim=1)

        out = (weights * stack).sum(dim=1)
        out = self.recompose(out)

        return x + self.scale(out)

# ==============================
# GSPF 辅助缓存：用于 V3 / Full 正则项
# ==============================

_GSPF_CACHE = {
    "usage": [],
    "ortho": [],
}

def reset_gspf_cache():
    _GSPF_CACHE["usage"].clear()
    _GSPF_CACHE["ortho"].clear()

def push_gspf_stats(usage=None, prototypes=None):
    """
    usage: [B, K]，表示每个样本的 prototype 使用分布
    prototypes: [K, C] 或 [B, K, C]
    """
    if usage is not None:
        _GSPF_CACHE["usage"].append(usage.detach() if not usage.requires_grad else usage)

    if prototypes is not None:
        if prototypes.dim() == 3:
            proto = prototypes.mean(dim=0)  # [K, C]
        else:
            proto = prototypes              # [K, C]

        proto = F.normalize(proto, dim=-1)
        gram = torch.matmul(proto, proto.t())
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        ortho_loss = ((gram - eye) ** 2).mean()
        _GSPF_CACHE["ortho"].append(ortho_loss)

def pop_gspf_regularization(lambda_consistency=0.0, lambda_ortho=0.0):
    """
    返回 GSPF 正则项，并清空缓存。
    注意：这里的 consistency 是多层 prototype usage 一致性。
    若要严格双视角一致性，需要进一步改 dual 分支。
    """
    total = None

    # 多层 prototype 使用分布一致性
    if lambda_consistency > 0 and len(_GSPF_CACHE["usage"]) >= 2:
        usages = _GSPF_CACHE["usage"]
        cons_loss = 0.0
        count = 0

        for i in range(len(usages)):
            for j in range(i + 1, len(usages)):
                ui = F.normalize(usages[i], dim=-1)
                uj = F.normalize(usages[j], dim=-1)
                cons_loss = cons_loss + (1.0 - (ui * uj).sum(dim=-1)).mean()
                count += 1

        cons_loss = cons_loss / max(count, 1)
        total = lambda_consistency * cons_loss if total is None else total + lambda_consistency * cons_loss

    # prototype 正交多样性
    if lambda_ortho > 0 and len(_GSPF_CACHE["ortho"]) > 0:
        ortho_loss = sum(_GSPF_CACHE["ortho"]) / len(_GSPF_CACHE["ortho"])
        total = lambda_ortho * ortho_loss if total is None else total + lambda_ortho * ortho_loss

    reset_gspf_cache()

    if total is None:
        return 0.0

    return total


# ==============================
# V1: Dynamic Prototype Routing Attention
# ==============================

class DynamicPrototypeRoutingAttention(nn.Module):
    """
    V1:
    固定 prototype -> 动态 prototype
    P(x) = P0 + DeltaP(x)
    """
    def __init__(
        self,
        channels,
        num_prototypes=8,
        temperature=1.0,
        reduction=8,
        delta_scale=0.1,
        init_scale=1e-3,
        record_stats=False,
    ):
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.delta_scale = delta_scale
        self.record_stats = record_stats

        hidden = max(channels // reduction, 8)

        self.pre = ConvBNAct(channels, channels, k=1, act=False)

        self.base_prototypes = nn.Parameter(torch.randn(num_prototypes, channels))
        nn.init.normal_(self.base_prototypes, std=0.02)

        # 根据当前图像生成 prototype 偏移
        self.proto_delta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, num_prototypes * channels, 1, bias=True)
        )

        self.prototype_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, num_prototypes, 1, bias=True)
        )

        self.recompose = nn.Sequential(
            ConvBNAct(channels, channels, k=1, act=True),
            ConvBNAct(channels, channels, k=3, p=1, g=channels, act=False)
        )

        self.scale = LayerScale(channels, init_value=init_scale)

    def forward(self, x):
        B, C, H, W = x.shape

        feat = self.pre(x).flatten(2).transpose(1, 2)  # [B, N, C]
        feat_n = F.normalize(feat, dim=-1)

        delta = self.proto_delta(x).view(B, self.num_prototypes, C)
        dynamic_proto = self.base_prototypes.unsqueeze(0) + self.delta_scale * delta
        proto_n = F.normalize(dynamic_proto, dim=-1)  # [B, K, C]

        sim = torch.bmm(feat_n, proto_n.transpose(1, 2)) / max(self.temperature, 1e-6)
        assign = F.softmax(sim, dim=-1)  # [B, N, K]

        p_gate = self.prototype_gate(x).flatten(2).transpose(1, 2)  # [B, 1, K]
        p_gate = F.softmax(p_gate, dim=-1)

        assign = assign * p_gate
        assign = assign / (assign.sum(dim=-1, keepdim=True) + 1e-6)

        recon = torch.bmm(assign, dynamic_proto)  # [B, N, C]
        recon = recon.transpose(1, 2).reshape(B, C, H, W)
        recon = self.recompose(recon)

        if self.record_stats:
            usage = assign.mean(dim=1)  # [B, K]
            push_gspf_stats(usage=usage, prototypes=dynamic_proto)

        return x + self.scale(recon)


# ==============================
# V2: GSPF Attention
# Granularity-Scale Prototype Fusion Attention
# ==============================

class GSPFAttention(nn.Module):
    """
    V2:
    多粒度尺度分支 + 动态 prototype 分支 + 尺度-原型耦合门控
    """
    def __init__(
        self,
        channels,
        kernel_sizes=(3, 5, 7),
        num_prototypes=8,
        temperature=1.0,
        reduction=8,
        delta_scale=0.1,
        init_scale=1e-3,
        record_stats=False,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_sizes = list(kernel_sizes)
        self.num_scales = len(self.kernel_sizes)
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.delta_scale = delta_scale
        self.record_stats = record_stats

        hidden = max(channels // reduction, 8)

        # ---------- Scale branch ----------
        self.scale_experts = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(channels, channels, k=k, p=k // 2, g=channels, act=True),
                ConvBNAct(channels, channels, k=1, act=False)
            )
            for k in self.kernel_sizes
        ])

        self.scale_channel_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_scales, 1, bias=True)
        )

        self.scale_spatial_router = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_scales, 1, bias=True)
        )

        self.scale_recompose = ConvBNAct(channels, channels, k=1, act=False)

        # ---------- Dynamic prototype branch ----------
        self.pre = ConvBNAct(channels, channels, k=1, act=False)

        self.base_prototypes = nn.Parameter(torch.randn(num_prototypes, channels))
        nn.init.normal_(self.base_prototypes, std=0.02)

        self.proto_delta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, num_prototypes * channels, 1, bias=True)
        )

        self.prototype_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, num_prototypes, 1, bias=True)
        )

        self.proto_recompose = nn.Sequential(
            ConvBNAct(channels, channels, k=1, act=True),
            ConvBNAct(channels, channels, k=3, p=1, g=channels, act=False)
        )

        # ---------- Coupling gate ----------
        self.couple_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

        self.out_recompose = ConvBNAct(channels, channels, k=1, act=False)
        self.scale = LayerScale(channels, init_value=init_scale)

    def _scale_branch(self, x):
        feats = [expert(x) for expert in self.scale_experts]
        stack = torch.stack(feats, dim=1)  # [B, S, C, H, W]

        ch_logits = self.scale_channel_router(x).unsqueeze(2)  # [B,S,1,1,1]
        sp_logits = self.scale_spatial_router(x).unsqueeze(2)  # [B,S,1,H,W]

        weights = torch.softmax(ch_logits + sp_logits, dim=1)
        out = (weights * stack).sum(dim=1)
        out = self.scale_recompose(out)
        return out

    def _proto_branch(self, x):
        B, C, H, W = x.shape

        feat = self.pre(x).flatten(2).transpose(1, 2)  # [B,N,C]
        feat_n = F.normalize(feat, dim=-1)

        delta = self.proto_delta(x).view(B, self.num_prototypes, C)
        dynamic_proto = self.base_prototypes.unsqueeze(0) + self.delta_scale * delta
        proto_n = F.normalize(dynamic_proto, dim=-1)

        sim = torch.bmm(feat_n, proto_n.transpose(1, 2)) / max(self.temperature, 1e-6)
        assign = F.softmax(sim, dim=-1)

        p_gate = self.prototype_gate(x).flatten(2).transpose(1, 2)
        p_gate = F.softmax(p_gate, dim=-1)

        assign = assign * p_gate
        assign = assign / (assign.sum(dim=-1, keepdim=True) + 1e-6)

        recon = torch.bmm(assign, dynamic_proto)
        recon = recon.transpose(1, 2).reshape(B, C, H, W)
        recon = self.proto_recompose(recon)

        if self.record_stats:
            usage = assign.mean(dim=1)
            push_gspf_stats(usage=usage, prototypes=dynamic_proto)

        return recon

    def forward(self, x):
        scale_feat = self._scale_branch(x)
        proto_feat = self._proto_branch(x)

        gate = self.couple_gate(torch.cat([scale_feat, proto_feat], dim=1))
        fused = gate * scale_feat + (1.0 - gate) * proto_feat
        fused = self.out_recompose(fused)

        return x + self.scale(fused)


class DynamicPrototypeRoutingAttentionRecord(DynamicPrototypeRoutingAttention):
    def __init__(self, *args, **kwargs):
        kwargs["record_stats"] = True
        super().__init__(*args, **kwargs)





class ProtoGSPFResidualAttention(nn.Module):
    """
    Proto 主导 + GSPF 小残差增强。

    原来的 GSPF 直接替代 Proto，容易破坏 ProtoRoute 的稳定性。
    这里改为：
        Y = Proto(X) + alpha * (GSPF(X) - X)

    alpha 初始值很小，默认 1e-3。
    """
    def __init__(self, channels, alpha_init=1e-3, **kwargs):
        super().__init__()
        self.proto = PrototypeRoutingAttention(channels)
        self.gspf = GSPFAttention(channels, record_stats=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        proto_out = self.proto(x)
        gspf_out = self.gspf(x)
        return proto_out + self.alpha * (gspf_out - x)
    


# ==============================
# Liquid Adapter / LG-v2 for R2 / PG-SPR
# ==============================

class ZeroInitDWAdapter(nn.Module):
    """
    极小残差适配器：depthwise 3x3 + pointwise 1x1 + BN。
    关键：pointwise 和 BN gamma 零初始化，使初始 delta ≈ 0，整体初始几乎等价于 R2。
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

        nn.init.kaiming_normal_(self.dw.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class LiquidAdapterPGSPRAttention(nn.Module):
    """
    Liquid-Adapter PG-SPR：推荐优先测试。

    原始 R2:
        Y_R2 = A_proto(X) + alpha * (A_gspf(X) - X)

    新增极小液态适配器:
        Delta = Adapter(Y_R2)
        lambda(X) = 1 + beta * tanh(g(X) / tau(X) - c0)
        Y = Y_R2 + eps * lambda(X) * Delta

    设计目标：不改写 R2 主分支，只在 R2 后面加一个零初始化小扰动。
    """
    def __init__(
        self,
        channels,
        reduction=16,
        alpha_init=1e-3,
        beta=0.05,
        tau_min=0.5,
        tau_max=3.0,
        eps_init=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.channels = int(channels)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.beta = float(beta)

        self.proto = PrototypeRoutingAttention(channels)
        self.gspf = GSPFAttention(channels, record_stats=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.adapter = ZeroInitDWAdapter(channels)
        self.eps = nn.Parameter(torch.tensor(float(eps_init)))

        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.GELU()
        self.fc_gate = nn.Conv2d(hidden, channels, 1, bias=True)
        self.fc_tau = nn.Conv2d(hidden, channels, 1, bias=True)

        # 零初始化：gate=0.5, tau=中间值，且 centered 后初始 lambda≈1
        nn.init.zeros_(self.fc_gate.weight)
        nn.init.zeros_(self.fc_gate.bias)
        nn.init.zeros_(self.fc_tau.weight)
        nn.init.zeros_(self.fc_tau.bias)

        tau0 = self.tau_min + (self.tau_max - self.tau_min) * 0.5
        c0 = 0.5 / tau0
        self.register_buffer("liquid_base", torch.tensor(float(c0)), persistent=False)

    def _liquid_lambda(self, x):
        ctx = self.pool(x)
        h = self.act(self.fc1(ctx))
        gate = torch.sigmoid(self.fc_gate(h))
        tau_gate = torch.sigmoid(self.fc_tau(h))
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_gate
        liquid = gate / (tau + 1e-6)
        centered = liquid - self.liquid_base.to(device=x.device, dtype=x.dtype)
        return 1.0 + self.beta * torch.tanh(centered)

    def forward(self, x):
        y_r2 = self.proto(x) + self.alpha * (self.gspf(x) - x)
        delta = self.adapter(y_r2)
        lam = self._liquid_lambda(y_r2)
        return y_r2 + self.eps * lam * delta


class WeakLiquidAdapterPGSPRAttention(nn.Module):
    """
    Weak-LAdapter PG-SPR: weak-class guided Liquid Adapter on top of R2 / PG-SPR.

    Base R2:
        Y_R2 = A_proto(X) + alpha * (A_gspf(X) - X)

    Weak-LAdapter:
        Delta = Adapter(Y_R2)
        lambda_liq(X) = 1 + beta_liq * tanh(g(X)/tau(X) - c0)
        weak_ch = WeakPrior @ ClassChannelLink
        lambda_weak = 1 + beta_weak * tanh(weak_ch)
        Y = Y_R2 + eps * lambda_liq(X) * lambda_weak * Delta

    设计目标：
        1) 保留 R2 主分支；
        2) Adapter 仍然零初始化，初始几乎等价于 R2；
        3) 用 weak-class prior 生成通道级弱类调制，引导 Adapter 更关注弱类相关证据；
        4) 不改 loss，不改 dual wrapper，显存风险小。
    """
    def __init__(
        self,
        channels,
        num_classes=15,
        weak_indices=(1, 4, 5, 8),
        reduction=16,
        alpha_init=1e-3,
        beta_liq=0.05,
        beta_weak_init=0.10,
        tau_min=0.5,
        tau_max=3.0,
        eps_init=1e-4,
        weak_clip=0.25,
        **kwargs,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.beta_liq = float(beta_liq)
        self.weak_clip = float(weak_clip)

        self.register_buffer(
            "weak_prior",
            _make_weak_prior(num_classes=num_classes, weak_indices=weak_indices),
            persistent=False,
        )

        self.proto = PrototypeRoutingAttention(channels)
        self.gspf = GSPFAttention(channels, record_stats=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.adapter = ZeroInitDWAdapter(channels)
        self.eps = nn.Parameter(torch.tensor(float(eps_init)))

        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.GELU()
        self.fc_gate = nn.Conv2d(hidden, channels, 1, bias=True)
        self.fc_tau = nn.Conv2d(hidden, channels, 1, bias=True)

        # 类别-通道关联：C_cls × C_feat。
        # 由 weak_prior 汇聚成弱类通道偏置，初始化很小，避免破坏 R2。
        self.class_channel_link = nn.Parameter(torch.zeros(num_classes, channels))
        nn.init.normal_(self.class_channel_link, std=0.01)
        self.beta_weak = nn.Parameter(torch.tensor(float(beta_weak_init)))

        # 初始化液态分支为中性状态：lambda_liq ≈ 1
        nn.init.zeros_(self.fc_gate.weight)
        nn.init.zeros_(self.fc_gate.bias)
        nn.init.zeros_(self.fc_tau.weight)
        nn.init.zeros_(self.fc_tau.bias)

        tau0 = self.tau_min + (self.tau_max - self.tau_min) * 0.5
        c0 = 0.5 / tau0
        self.register_buffer("liquid_base", torch.tensor(float(c0)), persistent=False)

    def _liquid_lambda(self, x):
        ctx = self.pool(x)
        h = self.act(self.fc1(ctx))
        gate = torch.sigmoid(self.fc_gate(h))
        tau_gate = torch.sigmoid(self.fc_tau(h))
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_gate
        liquid = gate / (tau + 1e-6)
        centered = liquid - self.liquid_base.to(device=x.device, dtype=x.dtype)
        return 1.0 + self.beta_liq * torch.tanh(centered)

    def _weak_lambda(self, x):
        # weak_prior: [num_classes]
        # class_channel_link: [num_classes, channels]
        weak_prior = self.weak_prior.to(device=x.device, dtype=x.dtype)
        link = self.class_channel_link.to(device=x.device, dtype=x.dtype)

        weak_ch = torch.matmul(weak_prior, link)  # [channels]
        weak_ch = weak_ch - weak_ch.mean()
        weak_ch = torch.clamp(weak_ch, -self.weak_clip, self.weak_clip)

        weak_lambda = 1.0 + self.beta_weak.to(dtype=x.dtype) * torch.tanh(weak_ch)
        return weak_lambda.view(1, self.channels, 1, 1)

    def forward(self, x):
        # R2 主分支完全保留
        y_r2 = self.proto(x) + self.alpha * (self.gspf(x) - x)

        # 零初始化 Adapter 小残差
        delta = self.adapter(y_r2)
        lam_liq = self._liquid_lambda(y_r2)
        lam_weak = self._weak_lambda(y_r2)

        return y_r2 + self.eps * lam_liq * lam_weak * delta


class LGPGSPRv2Attention(nn.Module):
    """
    LG-PGSPR-v2：比旧 LG-PGSPR 更保守。

    与旧 LG 的区别：
    1) 不再对 residual 做额外 1x1 recompose，减少破坏 R2 残差；
    2) lambda 围绕 1 小幅波动；
    3) gate/tau 最后一层零初始化，初始近似 R2。
    """
    def __init__(
        self,
        channels,
        reduction=16,
        alpha_init=1e-3,
        beta=0.05,
        tau_min=0.5,
        tau_max=3.0,
        mod_clip=0.20,
        channel_wise=True,
        **kwargs,
    ):
        super().__init__()
        self.channels = int(channels)
        self.beta = float(beta)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.mod_clip = float(mod_clip)
        self.channel_wise = bool(channel_wise)

        self.proto = PrototypeRoutingAttention(channels)
        self.gspf = GSPFAttention(channels, record_stats=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        hidden = max(channels // reduction, 16)
        out_dim = channels if self.channel_wise else 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.GELU()
        self.fc_gate = nn.Conv2d(hidden, out_dim, 1, bias=True)
        self.fc_tau = nn.Conv2d(hidden, out_dim, 1, bias=True)

        nn.init.zeros_(self.fc_gate.weight)
        nn.init.zeros_(self.fc_gate.bias)
        nn.init.zeros_(self.fc_tau.weight)
        nn.init.zeros_(self.fc_tau.bias)

        tau0 = self.tau_min + (self.tau_max - self.tau_min) * 0.5
        c0 = 0.5 / tau0
        self.register_buffer("liquid_base", torch.tensor(float(c0)), persistent=False)

    def _liquid_lambda(self, x):
        ctx = self.pool(x)
        h = self.act(self.fc1(ctx))
        gate = torch.sigmoid(self.fc_gate(h))
        tau_gate = torch.sigmoid(self.fc_tau(h))
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_gate
        liquid = gate / (tau + 1e-6)
        centered = liquid - self.liquid_base.to(device=x.device, dtype=x.dtype)
        centered = torch.clamp(centered, -self.mod_clip, self.mod_clip)
        lam = 1.0 + self.beta * torch.tanh(centered)
        return lam

    def forward(self, x):
        proto_out = self.proto(x)
        residual = self.gspf(x) - x
        lam = self._liquid_lambda(x)
        return proto_out + self.alpha * lam * residual

# ==============================
# Weak-Class Guided Attention
# 弱类引导注意力调制
# ==============================

# ==============================
# DWR: Difficulty-aware Weak-class Expert Routing Attention
# 难度感知弱类专家路由注意力
# ==============================

def _make_weak_prior(num_classes=15, weak_indices=(1, 4, 5, 8)):
    """
    默认弱类索引：
    1: Knife
    4: Scissors
    5: Lighter
    8: Razor_blade

    如果 classes.txt 顺序不同，需要改 weak_indices。
    """
    prior = torch.zeros(num_classes, dtype=torch.float32)
    for idx in weak_indices:
        if 0 <= idx < num_classes:
            prior[idx] = 1.0

    if prior.sum() > 0:
        prior = prior / prior.sum()

    return prior


class DifficultyWeakRoutingAttention(nn.Module):
    """
    DWR: Difficulty-aware Weak-class Expert Routing Attention

    三个专家：
      E1: ProtoRoute
      E2: ProtoGSPFResidual
      E3: WeakClassPrototypeRouting

    动态路由：
      router_logits = MLP(GAP(X)) + beta * difficulty(X) * weak_class_expert_bias

    输出：
      Y = X + LayerScale( sum_e w_e * (E_e(X) - X) )
    """
    def __init__(
        self,
        channels,
        num_classes=15,
        weak_indices=(1, 4, 5, 8),
        reduction=8,
        beta_init=0.1,
        init_scale=1e-3,
    ):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.num_experts = 3

        self.register_buffer(
            "weak_prior",
            _make_weak_prior(num_classes=num_classes, weak_indices=weak_indices),
            persistent=False
        )

        # 三个专家
        self.expert_proto = PrototypeRoutingAttention(channels)
        self.expert_gspf = ProtoGSPFResidualAttention(channels)

        # 这里要求你前面已经加过 WeakClassPrototypeRoutingAttention
        self.expert_wcproto = WeakClassPrototypeRoutingAttention(
            channels,
            num_classes=num_classes,
            weak_indices=weak_indices,
        )

        hidden = max(channels // reduction, 8)

        # 图像特征驱动的专家路由
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_experts, 1, bias=True)
        )

        # 当前特征难度估计，输出 [B,1]
        self.difficulty_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # 类别-专家关联矩阵：C × E
        self.class_expert_link = nn.Parameter(torch.zeros(num_classes, self.num_experts))
        nn.init.normal_(self.class_expert_link, std=0.02)

        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
    

    def _weak_expert_bias(self, x):
        """
        weak_prior: [C]
        class_expert_link: [C, E]
        输出 expert_bias: [E]
        """
        link = F.softmax(self.class_expert_link, dim=-1)
        weak_prior = self.weak_prior.to(device=x.device, dtype=x.dtype)

        expert_bias = torch.matmul(weak_prior, link.to(dtype=x.dtype))  # [E]
        expert_bias = expert_bias - expert_bias.mean()
        return expert_bias

    def forward(self, x):
        B, C, H, W = x.shape

        # 三个专家本身已经是残差形式：Y_e = X + delta_e
        # 所以这里直接对完整专家输出做加权平均，不再额外 LayerScale
        y_proto = self.expert_proto(x)
        y_gspf = self.expert_gspf(x)
        y_wcproto = self.expert_wcproto(x)

        expert_stack = torch.stack(
            [y_proto, y_gspf, y_wcproto],
            dim=1
        )  # [B, 3, C, H, W]

        router_logits = self.router(x).flatten(1)       # [B, 3]
        difficulty = self.difficulty_head(x).flatten(1) # [B, 1]

        expert_bias = self._weak_expert_bias(x)         # [3]

        router_logits = router_logits + self.beta * difficulty * expert_bias.view(1, -1)

        weights = torch.softmax(router_logits, dim=-1)
        weights = weights.view(B, self.num_experts, 1, 1, 1)

        out = (weights * expert_stack).sum(dim=1)

        return out


class WeakClassPrototypeRoutingAttention(nn.Module):
    """
    弱类引导的原型路由注意力。

    核心思想：
    原始 ProtoRoute:
        alpha_{i,k} = softmax(sim(x_i, p_k))

    弱类调制后:
        alpha'_{i,k} = softmax(sim(x_i, p_k) + beta * b_k)

    其中 b_k 由 weak-class prior 通过 class-prototype link 映射得到。
    """
    def __init__(
        self,
        channels,
        num_classes=15,
        weak_indices=(1, 4, 5, 8),
        num_prototypes=8,
        temperature=1.0,
        reduction=8,
        beta_init=0.1,
        init_scale=1e-3,
    ):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.temperature = temperature

        self.register_buffer(
            "weak_prior",
            _make_weak_prior(num_classes=num_classes, weak_indices=weak_indices),
            persistent=False
        )

        self.pre = ConvBNAct(channels, channels, k=1, act=False)

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, channels))
        nn.init.normal_(self.prototypes, std=0.02)

        # 类别-原型关联矩阵：C × K
        self.class_proto_link = nn.Parameter(torch.zeros(num_classes, num_prototypes))
        nn.init.normal_(self.class_proto_link, std=0.02)

        self.beta_proto = nn.Parameter(torch.tensor(float(beta_init)))
        self.beta_gate = nn.Parameter(torch.tensor(float(beta_init)))

        hidden = max(channels // reduction, 8)
        self.prototype_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, num_prototypes, 1, bias=True)
        )

        self.recompose = nn.Sequential(
            ConvBNAct(channels, channels, k=1, act=True),
            ConvBNAct(channels, channels, k=3, p=1, g=channels, act=False)
        )

        self.scale = LayerScale(channels, init_value=init_scale)

    def _prototype_bias(self, x):
        # weak_prior: [C]
        # class_proto_link: [C, K]
        link = F.softmax(self.class_proto_link, dim=-1)
        weak_prior = self.weak_prior.to(device=x.device, dtype=x.dtype)

        proto_bias = torch.matmul(weak_prior, link.to(dtype=x.dtype))  # [K]
        proto_bias = proto_bias - proto_bias.mean()
        return proto_bias

    def forward(self, x):
        B, C, H, W = x.shape

        feat = self.pre(x).flatten(2).transpose(1, 2)  # [B, N, C]
        feat_n = F.normalize(feat, dim=-1)

        proto = F.normalize(self.prototypes, dim=-1)   # [K, C]

        sim = torch.matmul(feat_n, proto.t()) / max(self.temperature, 1e-6)  # [B, N, K]

        proto_bias = self._prototype_bias(x)  # [K]
        sim = sim + self.beta_proto * proto_bias.view(1, 1, -1)

        assign = F.softmax(sim, dim=-1)

        p_gate_logits = self.prototype_gate(x).flatten(2).transpose(1, 2)  # [B, 1, K]
        p_gate_logits = p_gate_logits + self.beta_gate * proto_bias.view(1, 1, -1)
        p_gate = F.softmax(p_gate_logits, dim=-1)

        assign = assign * p_gate
        assign = assign / (assign.sum(dim=-1, keepdim=True) + 1e-6)

        recon = torch.matmul(assign, self.prototypes)  # [B, N, C]
        recon = recon.transpose(1, 2).reshape(B, C, H, W)
        recon = self.recompose(recon)

        return x + self.scale(recon)


class WeakClassGranularityAttention(nn.Module):
    """
    弱类引导的粒度竞争注意力。

    核心思想：
    对 N3 的 3×3 / 5×5 / 7×7 多粒度分支加入 weak-class scale bias。
    弱类越重要，模型越倾向于细粒度和中粒度分支。
    """
    def __init__(
        self,
        channels,
        num_classes=15,
        weak_indices=(1, 4, 5, 8),
        kernel_sizes=(3, 5, 7),
        reduction=8,
        beta_init=0.1,
        init_scale=1e-3,
    ):
        super().__init__()
        self.kernel_sizes = list(kernel_sizes)
        self.num_scales = len(self.kernel_sizes)
        self.num_classes = num_classes

        self.register_buffer(
            "weak_prior",
            _make_weak_prior(num_classes=num_classes, weak_indices=weak_indices),
            persistent=False
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(channels, channels, k=k, p=k // 2, g=channels, act=True),
                ConvBNAct(channels, channels, k=1, act=False)
            )
            for k in self.kernel_sizes
        ])

        hidden = max(channels // reduction, 8)

        self.channel_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_scales, 1, bias=True)
        )

        self.spatial_router = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_scales, 1, bias=True)
        )

        # 类别-尺度关联矩阵：C × S
        self.class_scale_link = nn.Parameter(torch.zeros(num_classes, self.num_scales))
        nn.init.normal_(self.class_scale_link, std=0.02)

        # 固定弱类细粒度先验：偏向 3×3 / 5×5，抑制 7×7
        base_bias = torch.tensor([0.10, 0.05, -0.15], dtype=torch.float32)
        if self.num_scales != 3:
            base_bias = torch.zeros(self.num_scales, dtype=torch.float32)
        self.register_buffer("base_scale_bias", base_bias, persistent=False)

        self.beta_scale = nn.Parameter(torch.tensor(float(beta_init)))

        self.recompose = ConvBNAct(channels, channels, k=1, act=False)
        self.scale = LayerScale(channels, init_value=init_scale)

    def _scale_bias(self, x):
        link = F.softmax(self.class_scale_link, dim=-1)
        weak_prior = self.weak_prior.to(device=x.device, dtype=x.dtype)

        scale_bias = torch.matmul(weak_prior, link.to(dtype=x.dtype))  # [S]
        scale_bias = scale_bias - scale_bias.mean()

        base_bias = self.base_scale_bias.to(device=x.device, dtype=x.dtype)
        return scale_bias + base_bias

    def forward(self, x):
        feats = [expert(x) for expert in self.experts]
        stack = torch.stack(feats, dim=1)  # [B, S, C, H, W]

        ch_logits = self.channel_router(x).unsqueeze(2)  # [B, S, 1, 1, 1]
        sp_logits = self.spatial_router(x).unsqueeze(2)  # [B, S, 1, H, W]

        scale_bias = self._scale_bias(x).view(1, self.num_scales, 1, 1, 1)

        logits = ch_logits + sp_logits + self.beta_scale * scale_bias
        weights = torch.softmax(logits, dim=1)

        out = (weights * stack).sum(dim=1)
        out = self.recompose(out)

        return x + self.scale(out)





class DifficultyWeakRoutingV2Attention(nn.Module):
    """
    DWR-V2: R2-dominant Difficulty-aware Dual-Expert Routing Attention

    两个专家：
      E1: ProtoRoute
      E2: ProtoGSPFResidual

    设计目的：
      1) 保留 R2 的稳定优势；
      2) 去掉不稳定的 WCProto 专家；
      3) 用 difficulty gate 动态决定 Proto 与 ProtoGSPFResidual 的比例。

    输出：
      Y = w1 * E_proto(X) + w2 * E_gspfres(X)

    因为每个专家本身都是残差形式：
      E(X) = X + delta

    所以最终仍然是：
      Y = X + w1 * delta_proto + w2 * delta_gspfres
    """
    def __init__(
        self,
        channels,
        num_classes=15,
        weak_indices=(1, 4, 5, 8),
        reduction=8,
        beta_init=0.1,
        router_bias_init=1.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.num_experts = 2

        self.register_buffer(
            "weak_prior",
            _make_weak_prior(num_classes=num_classes, weak_indices=weak_indices),
            persistent=False
        )

        # 双专家：Proto + R2
        self.expert_proto = PrototypeRoutingAttention(channels)
        self.expert_gspfres = ProtoGSPFResidualAttention(channels)

        hidden = max(channels // reduction, 8)

        # 图像特征驱动的专家路由
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_experts, 1, bias=True)
        )

        # 初始化时略微偏向 R2 专家，避免一开始被 Proto 稀释
        with torch.no_grad():
            self.router[3].bias.data[0] = 0.0
            self.router[3].bias.data[1] = float(router_bias_init)

        # 当前样本难度估计
        self.difficulty_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # 类别-专家关联矩阵：C × 2
        self.class_expert_link = nn.Parameter(torch.zeros(num_classes, self.num_experts))
        nn.init.normal_(self.class_expert_link, std=0.02)

        # 弱类难度调制强度
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

        # 固定基础偏置：难样本略微偏向 GSPFRes
        base_bias = torch.tensor([-0.05, 0.05], dtype=torch.float32)
        self.register_buffer("base_expert_bias", base_bias, persistent=False)

    def _weak_expert_bias(self, x):
        """
        weak_prior: [C]
        class_expert_link: [C, 2]
        输出 expert_bias: [2]
        """
        link = F.softmax(self.class_expert_link, dim=-1)
        weak_prior = self.weak_prior.to(device=x.device, dtype=x.dtype)

        expert_bias = torch.matmul(weak_prior, link.to(dtype=x.dtype))  # [2]
        expert_bias = expert_bias - expert_bias.mean()

        base_bias = self.base_expert_bias.to(device=x.device, dtype=x.dtype)

        return expert_bias + base_bias

    def forward(self, x):
        B, C, H, W = x.shape

        y_proto = self.expert_proto(x)
        y_gspfres = self.expert_gspfres(x)

        expert_stack = torch.stack(
            [y_proto, y_gspfres],
            dim=1
        )  # [B, 2, C, H, W]

        router_logits = self.router(x).flatten(1)        # [B, 2]
        difficulty = self.difficulty_head(x).flatten(1)  # [B, 1]

        expert_bias = self._weak_expert_bias(x)          # [2]

        router_logits = router_logits + self.beta * difficulty * expert_bias.view(1, -1)

        weights = torch.softmax(router_logits, dim=-1)
        weights = weights.view(B, self.num_experts, 1, 1, 1)

        out = (weights * expert_stack).sum(dim=1)

        return out

class GSPFAttentionRecord(GSPFAttention):
    def __init__(self, *args, **kwargs):
        kwargs["record_stats"] = True
        super().__init__(*args, **kwargs)

# ==============================
# 注册表
# ==============================

ATTENTION_REGISTRY = {
    "se": SEAttention,
    "cbam": CBAM,
    "eca": ECAAttention,

    "freq_route": FrequencyRoutingAttention,
    "frequency_routing": FrequencyRoutingAttention,

    "polarity": PolarityAttention,

    "proto_route": PrototypeRoutingAttention,
    "prototype_routing": PrototypeRoutingAttention,

    "self_feedback": SelfFeedbackAttention,
    "feedback_attn": SelfFeedbackAttention,

    "granularity": GranularityAttention,
    "granularity_competition": GranularityAttention,

    "dynamic_proto": DynamicPrototypeRoutingAttention,
    "dyn_proto": DynamicPrototypeRoutingAttention,
    "gspf": GSPFAttention,
    "dynamic_scale_proto": GSPFAttention,

    "dynamic_proto_reg": DynamicPrototypeRoutingAttentionRecord,
    "gspf_reg": GSPFAttentionRecord,
    "proto_gspf_residual": ProtoGSPFResidualAttention,
    "proto_gspf_res": ProtoGSPFResidualAttention,
    "proto_gspf_aux": ProtoGSPFResidualAttention,


    "liquid_adapter_pgspr": LiquidAdapterPGSPRAttention,
    "ladapter_pgspr": LiquidAdapterPGSPRAttention,
    "liquid_adapter": LiquidAdapterPGSPRAttention,

    "weak_liquid_adapter_pgspr": WeakLiquidAdapterPGSPRAttention,
    "weak_ladapter_pgspr": WeakLiquidAdapterPGSPRAttention,
    "wladapter_pgspr": WeakLiquidAdapterPGSPRAttention,
    "weak_ladapter": WeakLiquidAdapterPGSPRAttention,

    "lg_pgspr_v2": LGPGSPRv2Attention,
    "liquid_gate_pgspr_v2": LGPGSPRv2Attention,
    "lgv2_pgspr": LGPGSPRv2Attention,

    "wc_proto_route": WeakClassPrototypeRoutingAttention,
    "weak_proto_route": WeakClassPrototypeRoutingAttention,
    "weak_class_proto": WeakClassPrototypeRoutingAttention,

    "wc_granularity": WeakClassGranularityAttention,
    "weak_granularity": WeakClassGranularityAttention,
    "weak_class_granularity": WeakClassGranularityAttention,

    "dwr_route": DifficultyWeakRoutingAttention,
    "difficulty_weak_route": DifficultyWeakRoutingAttention,
    "weak_expert_route": DifficultyWeakRoutingAttention,

    "dwr2_route": DifficultyWeakRoutingV2Attention,
    "difficulty_weak_route_v2": DifficultyWeakRoutingV2Attention,
    "weak_expert_route_v2": DifficultyWeakRoutingV2Attention,
}