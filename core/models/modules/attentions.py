import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# utility modules
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
# previously existing attentions (kept)
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
# 1. paper-level frequency-routing attention
# Complementary Frequency Routing Attention
# ==============================

class FrequencyRoutingAttention(nn.Module):
    """
    Paper-level upgraded version:
    1) same-size low-frequency extraction to avoid shape misalignment
    2) complementary low/high-frequency decomposition
    3) channel-group routing + spatial routing
    4) budget-conserving competition: w_low + w_high = 1
    5) frequency recombination + safe residual
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

        # same-size learnable low-pass filter
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

        # high-frequency enhancement
        self.high_proj = ConvBNAct(
            channels,
            channels,
            k=kernel_size,
            p=kernel_size // 2,
            g=channels,
            act=False
        )

        # channel-group routing -> [B, 2g, 1, 1]
        self.channel_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.groups * 2, 1, bias=True)
        )

        # spatial routing -> [B, 2, H, W]
        self.spatial_router = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 2, 1, bias=True)
        )

        # frequency recombination
        self.recompose = nn.Sequential(
            ConvBNAct(channels, channels, k=1, act=True),
            ConvBNAct(channels, channels, k=3, p=1, g=channels, act=False),
        )

        self.scale = LayerScale(channels, init_value=init_scale)

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.groups
        cg = self.group_channels

        # complementary low/high-frequency decomposition
        low = self.low_pass(x)      # [B,C,H,W]
        high = x - low              # [B,C,H,W]
        high = self.high_proj(high) # [B,C,H,W]

        # reshape into the group layout
        low_g = low.view(B, g, cg, H, W)    # [B,g,cg,H,W]
        high_g = high.view(B, g, cg, H, W)  # [B,g,cg,H,W]

        # channel routing
        ch_feat = torch.cat([low, high], dim=1)          # [B,2C,H,W]
        ch_logits = self.channel_router(ch_feat)         # [B,2g,1,1]
        ch_logits = ch_logits.view(B, g, 2, 1, 1)       # [B,g,2,1,1]

        # spatial routing
        spatial_stat = torch.cat([
            torch.mean(low, dim=1, keepdim=True),        # [B,1,H,W]
            torch.mean(high.abs(), dim=1, keepdim=True), # [B,1,H,W]
        ], dim=1)                                        # [B,2,H,W]

        sp_logits = self.spatial_router(spatial_stat)    # [B,2,H,W]
        sp_logits = sp_logits.unsqueeze(1)               # [B,1,2,H,W]

        # joint logits -> [B,g,2,H,W]
        logits = ch_logits.expand(-1, -1, -1, H, W) + sp_logits.expand(-1, g, -1, -1, -1)
        weights = torch.softmax(logits, dim=2)

        # key fix: add the channel dimension so it broadcasts correctly against [B,g,cg,H,W]
        w_low = weights[:, :, 0].unsqueeze(2)    # [B,g,1,H,W]
        w_high = weights[:, :, 1].unsqueeze(2)   # [B,g,1,H,W]

        fused = w_low * low_g + w_high * high_g  # [B,g,cg,H,W]
        fused = fused.reshape(B, C, H, W)

        fused = self.recompose(fused)
        return x + self.scale(fused)


# ==============================
# 2. paper-level polarity-relation attention
# Support-Inhibit Polarity Attention
# ==============================

class PolarityAttention(nn.Module):
    """
    Paper-level upgraded version:
    1) dual branch for supporting / suppressing evidence
    2) separate modelling of positive and negative responses
    3) joint modulation by a channel gate and a spatial gate
    4) final relation recombination
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

        # channel gate
        ch_gate = self.channel_gate(torch.cat([support, inhibit], dim=1))
        ch_sup, ch_inh = torch.chunk(ch_gate, 2, dim=1)
        ch_sup = torch.sigmoid(ch_sup)
        ch_inh = torch.sigmoid(ch_inh)

        # spatial gate
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
# 3. paper-level prototype-routing attention
# Latent Prototype Routing Attention
# ==============================

class PrototypeRoutingAttention(nn.Module):
    """
    Paper-level upgraded version:
    1) learnable prototype dictionary
    2) normalised similarity assignment
    3) image-level prototype gate
    4) prototype reconstruction + recombination
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

        # image-level prototype gate
        p_gate = self.prototype_gate(x).flatten(2).transpose(1, 2)  # [B,1,P]
        p_gate = F.softmax(p_gate, dim=-1)

        assign = assign * p_gate
        assign = assign / (assign.sum(dim=-1, keepdim=True) + 1e-6)

        recon = torch.matmul(assign, self.prototypes)  # [B,N,C]
        recon = recon.transpose(1, 2).reshape(B, C, H, W)
        recon = self.recompose(recon)

        return x + self.scale(recon)


# ==============================
# 4. paper-level self-feedback attention
# Confidence Feedback Attention
# ==============================

class SelfFeedbackAttention(nn.Module):
    """
    Paper-level upgraded version:
    1) coarse attention -> confidence map -> refined attention
    2) boundary auxiliary constraint
    3) two-stage feedback refinement
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
# 5. paper-level granularity-competition attention
# Granularity Competition Attention
# ==============================

class GranularityAttention(nn.Module):
    """
    Paper-level upgraded version:
    1) multi-receptive-field experts
    2) scale competition instead of plain summation
    3) joint channel-level and spatial-level scale assignment
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
# GSPF auxiliary cache: used by the V3 / Full regularisers
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
    usage: [B, K], the prototype usage distribution of every sample
    prototypes: [K, C] or [B, K, C]
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
    Return the GSPF regulariser and clear the cache.
    Note: here consistency means multi-level prototype-usage consistency.
    Strict dual-view consistency would require further changes in the dual branch.
    """
    total = None

    # multi-level prototype-usage consistency
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

    # prototype orthogonality diversity
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
    fixed prototype -> dynamic prototype
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

        # generate a prototype offset from the current image
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
    multi-granularity scale branch + dynamic prototype branch + scale-prototype coupling gate
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
    Proto dominant + small GSPF residual.

    The original GSPF replaced Proto directly, which easily destabilises ProtoRoute.
    The variant here instead:
        Y = Proto(X) + alpha * (GSPF(X) - X)

    alpha starts very small, 1e-3 by default.
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
    Tiny residual adapter: depthwise 3x3 + pointwise 1x1 + BN.
    Key detail: the pointwise and BN gamma are zero-initialised, so the initial delta is ~0 and the model starts out equivalent to R2.
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
    Liquid-Adapter PG-SPR: recommended as the first variant to test.

    Original R2:
        Y_R2 = A_proto(X) + alpha * (A_gspf(X) - X)

    Added tiny liquid adapter:
        Delta = Adapter(Y_R2)
        lambda(X) = 1 + beta * tanh(g(X) / tau(X) - c0)
        Y = Y_R2 + eps * lambda(X) * Delta

    Design goal: never rewrite the R2 main branch, only append a zero-initialised small perturbation after it.
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

        # zero initialisation: gate=0.5, tau=mid value, and after centering the initial lambda is ~1
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

    Design goal:
        1) keep the R2 main branch;
        2) the adapter stays zero-initialised so the start is nearly equivalent to R2;
        3) use the weak-class prior to build a channel-level weak-class modulation that steers the adapter towards weak-class evidence;
        4) no change to the loss or the dual wrapper, so the memory risk is small.
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

        # class-channel association: C_cls x C_feat.
        # weak_prior is aggregated into a weak-class channel bias with a tiny initial scale so R2 is not disturbed.
        self.class_channel_link = nn.Parameter(torch.zeros(num_classes, channels))
        nn.init.normal_(self.class_channel_link, std=0.01)
        self.beta_weak = nn.Parameter(torch.tensor(float(beta_weak_init)))

        # initialise the liquid branch to a neutral state: lambda_liq ~ 1
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
        # the R2 main branch is fully preserved
        y_r2 = self.proto(x) + self.alpha * (self.gspf(x) - x)

        # zero-initialised small adapter residual
        delta = self.adapter(y_r2)
        lam_liq = self._liquid_lambda(y_r2)
        lam_weak = self._weak_lambda(y_r2)

        return y_r2 + self.eps * lam_liq * lam_weak * delta


class LGPGSPRv2Attention(nn.Module):
    """
    LG-PGSPR-v2: more conservative than the old LG-PGSPR.

    Differences from the old LG:
    1) no extra 1x1 recompose on the residual, so the R2 residual is disturbed less;
    2) lambda fluctuates slightly around 1;
    3) the last layer of gate/tau is zero-initialised, so the start is close to R2.
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
# weak-class guided attention modulation
# ==============================

# ==============================
# DWR: Difficulty-aware Weak-class Expert Routing Attention
# difficulty-aware weak-class expert routing attention
# ==============================

def _make_weak_prior(num_classes=15, weak_indices=(1, 4, 5, 8)):
    """
    default weak-class indices:
    1: Knife
    4: Scissors
    5: Lighter
    8: Razor_blade

    change weak_indices if the order in classes.txt differs.
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

    three experts:
      E1: ProtoRoute
      E2: ProtoGSPFResidual
      E3: WeakClassPrototypeRouting

    dynamic routing:
      router_logits = MLP(GAP(X)) + beta * difficulty(X) * weak_class_expert_bias

    Output:
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

        # the three experts
        self.expert_proto = PrototypeRoutingAttention(channels)
        self.expert_gspf = ProtoGSPFResidualAttention(channels)

        # requires WeakClassPrototypeRoutingAttention to be defined above
        self.expert_wcproto = WeakClassPrototypeRoutingAttention(
            channels,
            num_classes=num_classes,
            weak_indices=weak_indices,
        )

        hidden = max(channels // reduction, 8)

        # image-feature driven expert routing
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_experts, 1, bias=True)
        )

        # difficulty estimate of the current features, output [B,1]
        self.difficulty_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # class-expert association matrix: C x E
        self.class_expert_link = nn.Parameter(torch.zeros(num_classes, self.num_experts))
        nn.init.normal_(self.class_expert_link, std=0.02)

        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
    

    def _weak_expert_bias(self, x):
        """
        weak_prior: [C]
        class_expert_link: [C, E]
        outputs expert_bias: [E]
        """
        link = F.softmax(self.class_expert_link, dim=-1)
        weak_prior = self.weak_prior.to(device=x.device, dtype=x.dtype)

        expert_bias = torch.matmul(weak_prior, link.to(dtype=x.dtype))  # [E]
        expert_bias = expert_bias - expert_bias.mean()
        return expert_bias

    def forward(self, x):
        B, C, H, W = x.shape

        # the three experts are already residual: Y_e = X + delta_e
        # so the full expert outputs are averaged directly, without an extra LayerScale
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
    Weak-class guided prototype-routing attention.

    Core idea:
    original ProtoRoute:
        alpha_{i,k} = softmax(sim(x_i, p_k))

    after weak-class modulation:
        alpha'_{i,k} = softmax(sim(x_i, p_k) + beta * b_k)

    where b_k comes from the weak-class prior through a class-prototype link.
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

        # class-prototype association matrix: C x K
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
    Weak-class guided granularity-competition attention.

    Core idea:
    Adds a weak-class scale bias to the 3x3 / 5x5 / 7x7 multi-granularity branches of N3.
    The more important a weak class is, the more the model favours the fine and medium granularity branches.
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

        # class-scale association matrix: C x S
        self.class_scale_link = nn.Parameter(torch.zeros(num_classes, self.num_scales))
        nn.init.normal_(self.class_scale_link, std=0.02)

        # fixed weak-class fine-granularity prior: favour 3x3 / 5x5, suppress 7x7
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

    two experts:
      E1: ProtoRoute
      E2: ProtoGSPFResidual

    Design purpose:
      1) keep the stability advantage of R2;
      2) drop the unstable WCProto expert;
      3) let a difficulty gate decide the ratio between Proto and ProtoGSPFResidual dynamically.

    Output:
      Y = w1 * E_proto(X) + w2 * E_gspfres(X)

    Since every expert is residual by construction:
      E(X) = X + delta

    the final form is still:
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

        # two experts: Proto + R2
        self.expert_proto = PrototypeRoutingAttention(channels)
        self.expert_gspfres = ProtoGSPFResidualAttention(channels)

        hidden = max(channels // reduction, 8)

        # image-feature driven expert routing
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_experts, 1, bias=True)
        )

        # bias the initialisation slightly towards the R2 expert so Proto does not dilute it early on
        with torch.no_grad():
            self.router[3].bias.data[0] = 0.0
            self.router[3].bias.data[1] = float(router_bias_init)

        # difficulty estimate of the current sample
        self.difficulty_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # class-expert association matrix: C x 2
        self.class_expert_link = nn.Parameter(torch.zeros(num_classes, self.num_experts))
        nn.init.normal_(self.class_expert_link, std=0.02)

        # weak-class difficulty modulation strength
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

        # fixed base bias: hard samples lean slightly towards GSPFRes
        base_bias = torch.tensor([-0.05, 0.05], dtype=torch.float32)
        self.register_buffer("base_expert_bias", base_bias, persistent=False)

    def _weak_expert_bias(self, x):
        """
        weak_prior: [C]
        class_expert_link: [C, 2]
        outputs expert_bias: [2]
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
# registry
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