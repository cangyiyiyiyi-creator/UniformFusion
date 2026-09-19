import math
from typing import Tuple
import torch
import torch.nn as nn
from typing import Tuple, Dict
# import Conv1x1 from the sibling common.py
from .common import Conv1x1
import torch.nn.functional as F 

# ------------------------------
# gated fusion: concat -> 1x1 projection -> SE gate
# ------------------------------
class GatedFuse(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, r: int = 8):
        super().__init__()
        hid = max(out_ch // r, 1)
        self.reduce = Conv1x1(in_ch, out_ch, act=True)   # 2C -> C
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_ch, hid, 1, bias=True)
        self.fc2 = nn.Conv2d(hid, out_ch, 1, bias=True)
        self.act = nn.GELU()
        self.sig = nn.Sigmoid()

    def forward(self, xa, xb):
        x = torch.cat([xa, xb], dim=1)      # [B,2C,H,W]
        y = self.reduce(x)                  # [B,C,H,W]
        w = self.avg(y)                     # [B,C,1,1]
        w = self.fc2(self.act(self.fc1(w)))
        w = self.sig(w)
        return y * w + y * (1 - w) * 0.0    # keeps the structure, equivalent to y * w


class ClasswiseRegionComplementFuse(nn.Module):
    """Class-specific, non-aligned region complement fusion for two views."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        projection_dim: int = 64,
        topk_ratio: float = 0.25,
        temperature: float = 0.2,
        residual_init: float = 0.05,
        residual_max: float = 0.2,
    ):
        super().__init__()
        channels = int(channels)
        num_classes = int(num_classes)
        projection_dim = max(16, int(projection_dim))
        if num_classes <= 0:
            raise ValueError("num_classes must be positive for DV-CRE")
        if not 0.0 < float(topk_ratio) <= 1.0:
            raise ValueError("DV-CRE topk_ratio must be in (0, 1]")
        if float(temperature) <= 0.0:
            raise ValueError("DV-CRE temperature must be positive")

        self.num_classes = num_classes
        self.projection_dim = projection_dim
        self.topk_ratio = float(topk_ratio)
        self.temperature = float(temperature)
        self.residual_max = max(float(residual_max), 1e-6)

        self.projection = nn.Sequential(
            nn.Conv2d(channels, projection_dim, 1, bias=False),
            nn.GroupNorm(1, projection_dim),
            nn.GELU(),
        )
        self.class_map = nn.Conv2d(projection_dim, num_classes, 1, bias=True)
        gate_in_dim = 4 * projection_dim + 2
        self.region_gate = nn.Sequential(
            nn.Linear(gate_in_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, 1),
        )
        nn.init.zeros_(self.region_gate[-1].weight)
        nn.init.zeros_(self.region_gate[-1].bias)

        self.classifier_weight = nn.Parameter(
            torch.empty(num_classes, projection_dim)
        )
        self.classifier_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_uniform_(self.classifier_weight)

        self.residual = nn.Sequential(
            nn.Conv2d(3 * projection_dim, projection_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(projection_dim, channels, 1, bias=True),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

        init_ratio = min(
            max(float(residual_init) / self.residual_max, 1e-4),
            1.0 - 1e-4,
        )
        self.residual_logit = nn.Parameter(
            torch.tensor(math.log(init_ratio / (1.0 - init_ratio)))
        )

    def _region_attention(self, scores: torch.Tensor):
        batch, classes, height, width = scores.shape
        flat = scores.flatten(2).float() / self.temperature
        token_count = flat.shape[-1]
        keep = max(1, min(token_count, int(round(token_count * self.topk_ratio))))
        if keep < token_count:
            values, indices = flat.topk(keep, dim=-1)
            selected = torch.full_like(flat, torch.finfo(flat.dtype).min)
            selected.scatter_(-1, indices, values)
            flat = selected
        attention = flat.softmax(dim=-1)
        if keep > 1:
            entropy = -(
                attention * attention.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(float(keep))
            confidence = (1.0 - entropy).clamp(0.0, 1.0)
        else:
            confidence = attention.new_ones(batch, classes)
        return attention.to(dtype=scores.dtype), confidence.to(dtype=scores.dtype)

    @staticmethod
    def _pool_regions(features: torch.Tensor, attention: torch.Tensor):
        tokens = features.flatten(2).transpose(1, 2)
        return torch.einsum("bkn,bnd->bkd", attention, tokens)

    @staticmethod
    def _support_mask(
        attention: torch.Tensor,
        weights: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        support = attention / attention.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        weighted = support * weights.unsqueeze(-1)
        mask = weighted.sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return mask.reshape(mask.shape[0], 1, height, width)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor):
        if xa.shape != xb.shape:
            raise ValueError("DV-CRE expects same-shape dual-view features")
        _, _, height, width = xa.shape
        pa = self.projection(xa)
        pb = self.projection(xb)
        score_a = self.class_map(pa)
        score_b = self.class_map(pb)
        attention_a, confidence_a = self._region_attention(score_a)
        attention_b, confidence_b = self._region_attention(score_b)
        region_a = self._pool_regions(pa, attention_a)
        region_b = self._pool_regions(pb, attention_b)

        gate_input = torch.cat(
            [
                region_a,
                region_b,
                (region_a - region_b).abs(),
                region_a * region_b,
                confidence_a.unsqueeze(-1),
                confidence_b.unsqueeze(-1),
            ],
            dim=-1,
        )
        gate_a = torch.sigmoid(self.region_gate(gate_input))
        class_features = gate_a * region_a + (1.0 - gate_a) * region_b
        aux_logits = (
            class_features * self.classifier_weight.unsqueeze(0)
        ).sum(dim=-1) / math.sqrt(float(self.projection_dim))
        aux_logits = aux_logits + self.classifier_bias.unsqueeze(0)

        weight_a = gate_a.squeeze(-1) * (0.5 + 0.5 * confidence_a)
        weight_b = (1.0 - gate_a.squeeze(-1)) * (0.5 + 0.5 * confidence_b)
        mask_a = self._support_mask(attention_a, weight_a, height, width)
        mask_b = self._support_mask(attention_b, weight_b, height, width)
        disagreement = (pa - pb).abs() * (0.5 * (mask_a + mask_b))
        correction = self.residual(
            torch.cat([pa * mask_a, pb * mask_b, disagreement], dim=1)
        )
        residual_scale = self.residual_max * torch.sigmoid(self.residual_logit)
        fused = xa + xb + residual_scale * correction
        return fused, {
            "aux_logits": aux_logits,
            "gate_a_mean": gate_a.mean(),
            "region_confidence": 0.5 * (
                confidence_a.mean() + confidence_b.mean()
            ),
            "residual_scale": residual_scale,
        }


class InterventionStableFuse(nn.Module):
    """Reliability-gated fusion initialized exactly as feature addition."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        channels = int(channels)
        hidden = max(32, min(128, channels // max(int(reduction), 1)))
        self.gate = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor):
        if xa.shape != xb.shape:
            raise ValueError("IS-CVF expects same-shape dual-view features")
        pooled_a = F.adaptive_avg_pool2d(xa, 1)
        pooled_b = F.adaptive_avg_pool2d(xb, 1)
        context = torch.cat(
            [pooled_a, pooled_b, (pooled_a - pooled_b).abs()], dim=1
        )
        gate_a = torch.sigmoid(self.gate(context))
        fused = 2.0 * (gate_a * xa + (1.0 - gate_a) * xb)
        return fused, gate_a.mean()


# ------------------------------
# XAttn fusion: bidirectional cross-view attention (downsampling reduces the token count)
# A as query and B as KV gives the A<-B message; the reverse direction is identical, then average and add a residual
# ------------------------------
class XAttnFuse(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, reduction: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.reduction = max(int(reduction), 1)

        # PyTorch MultiheadAttention expects [S,B,E]
        self.mha_ab = nn.MultiheadAttention(embed_dim=self.channels, num_heads=self.num_heads, batch_first=False)
        self.mha_ba = nn.MultiheadAttention(embed_dim=self.channels, num_heads=self.num_heads, batch_first=False)

        # normalisation and output projection
        self.norm_q = nn.LayerNorm(self.channels)
        self.norm_kv = nn.LayerNorm(self.channels)
        self.proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=self.reduction, stride=self.reduction, ceil_mode=False)

    def _upsample(self, x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        if self.reduction <= 1:
            return x
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

    def _to_seq(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [S(=H*W), B, C] with LN for Q
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # [B, HW, C]
        x = self.norm_q(x)
        x = x.transpose(0, 1)                 # [HW, B, C]
        return x, (H, W)

    def _to_seq_kv(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # [B, HW, C]
        x = self.norm_kv(x)
        x = x.transpose(0, 1)                 # [HW, B, C]
        return x

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = xa.shape

        # spatial downsampling to reduce the token count (saves GPU memory)
        xa_ds = self._downsample(xa)
        xb_ds = self._downsample(xb)
        h, w = xa_ds.shape[-2], xa_ds.shape[-1]

        # tokenisation
        qa, _ = self._to_seq(xa_ds)       # [S,B,C]
        kab = self._to_seq_kv(xb_ds)      # [S,B,C]
        qb, _ = self._to_seq(xb_ds)
        kba = self._to_seq_kv(xa_ds)

        # bidirectional cross-attention
        y_ab, _ = self.mha_ab(qa, kab, kab, need_weights=False)  # A <- B
        y_ba, _ = self.mha_ba(qb, kba, kba, need_weights=False)  # B <- A

        # back to [B,C,h,w]
        y_ab = y_ab.transpose(0, 1).transpose(1, 2).reshape(B, C, h, w)
        y_ba = y_ba.transpose(0, 1).transpose(1, 2).reshape(B, C, h, w)

        # upsample back to the original size
        y_ab = self._upsample(y_ab, (H, W))
        y_ba = self._upsample(y_ba, (H, W))

        # fusion: average, apply 1x1, add the residual (more stable)
        y = 0.5 * (y_ab + y_ba)
        y = self.proj(y)
        return xa + 0.5 * y   # residual: gently adjust the original features
    
# ==============================
# [added] AHCR core component: adaptive cross attention
# ==============================
# ==============================
# [fixed] AHCR core component: adaptive cross attention (full version)
# ==============================
class AdaptiveCrossAttention(nn.Module):
    def __init__(self, dim_query, dim_kv, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_query // num_heads # Head dimension is based on query dim
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim_query, dim_query, bias=qkv_bias)
        self.wk = nn.Linear(dim_kv, dim_query, bias=qkv_bias)
        self.wv = nn.Linear(dim_kv, dim_query, bias=qkv_bias)
        
        self.proj = nn.Linear(dim_query, dim_query)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_query, x_key_value):
        B, N_q, C_q = x_query.shape
        _, N_kv, _ = x_key_value.shape
        
        q = self.wq(x_query).reshape(B, N_q, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x_key_value).reshape(B, N_kv, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x_key_value).reshape(B, N_kv, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C_q)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x_query + self.gamma * x

# ==============================
# [fixed] AHCR fusion module (final version)
# ==============================
class AHCRFuse(nn.Module):
    def __init__(self, c3_dim, c4_dim, c5_dim, mode='intra_level'):
        super().__init__()
        self.mode = mode

        if self.mode == 'intra_level':
            # in the within-layer mode query and kv share the same dimension
            self.cross_attn_c3 = AdaptiveCrossAttention(dim_query=c3_dim, dim_kv=c3_dim)
            self.cross_attn_c3_rev = AdaptiveCrossAttention(dim_query=c3_dim, dim_kv=c3_dim)
            self.cross_attn_c4 = AdaptiveCrossAttention(dim_query=c4_dim, dim_kv=c4_dim)
            self.cross_attn_c4_rev = AdaptiveCrossAttention(dim_query=c4_dim, dim_kv=c4_dim)
            self.cross_attn_c5 = AdaptiveCrossAttention(dim_query=c5_dim, dim_kv=c5_dim)
            self.cross_attn_c5_rev = AdaptiveCrossAttention(dim_query=c5_dim, dim_kv=c5_dim)
            
            self.final_fuse_c3 = Conv1x1(in_ch=2*c3_dim, out_ch=c3_dim, act=True)
            self.final_fuse_c4 = Conv1x1(in_ch=2*c4_dim, out_ch=c4_dim, act=True)
            self.final_fuse_c5 = Conv1x1(in_ch=2*c5_dim, out_ch=c5_dim, act=True)
        
        elif self.mode == 'inter_level':
            self.cross_attn_c4_to_c3 = AdaptiveCrossAttention(dim_query=c3_dim, dim_kv=c4_dim)
            self.cross_attn_c4_to_c3_rev = AdaptiveCrossAttention(dim_query=c3_dim, dim_kv=c4_dim)
            self.cross_attn_c5_to_c4 = AdaptiveCrossAttention(dim_query=c4_dim, dim_kv=c5_dim)
            self.cross_attn_c5_to_c4_rev = AdaptiveCrossAttention(dim_query=c4_dim, dim_kv=c5_dim)
            
            self.final_fuse_c3 = Conv1x1(in_ch=2*c3_dim, out_ch=c3_dim, act=True)
            self.final_fuse_c4 = Conv1x1(in_ch=2*c4_dim, out_ch=c4_dim, act=True)
            self.final_fuse_c5 = nn.Identity()
        else:
            raise ValueError(f"unknown AHCR mode: {self.mode}")

    def _flatten(self, x):
        return x.flatten(2).transpose(1, 2)

    def _reshape(self, x, B, C, H, W):
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, feats_a: Dict[str, torch.Tensor], feats_b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c3a, c4a, c5a = feats_a['C3'], feats_a['C4'], feats_a['C5']
        c3b, c4b, c5b = feats_b['C3'], feats_b['C4'], feats_b['C5']
        
        B, C3, H3, W3 = c3a.shape
        _, C4, H4, W4 = c4a.shape
        _, C5, H5, W5 = c5a.shape
        
        if self.mode == 'intra_level':
            c3a_r = self.cross_attn_c3(self._flatten(c3a), self._flatten(c3b))
            c3b_r = self.cross_attn_c3_rev(self._flatten(c3b), self._flatten(c3a))
            fused_c3 = self.final_fuse_c3(torch.cat([
                self._reshape(c3a_r, B, C3, H3, W3),
                self._reshape(c3b_r, B, C3, H3, W3)
            ], dim=1))

            c4a_r = self.cross_attn_c4(self._flatten(c4a), self._flatten(c4b))
            c4b_r = self.cross_attn_c4_rev(self._flatten(c4b), self._flatten(c4a))
            fused_c4 = self.final_fuse_c4(torch.cat([
                self._reshape(c4a_r, B, C4, H4, W4),
                self._reshape(c4b_r, B, C4, H4, W4)
            ], dim=1))

            c5a_r = self.cross_attn_c5(self._flatten(c5a), self._flatten(c5b))
            c5b_r = self.cross_attn_c5_rev(self._flatten(c5b), self._flatten(c5a))
            fused_c5 = self.final_fuse_c5(torch.cat([
                self._reshape(c5a_r, B, C5, H5, W5),
                self._reshape(c5b_r, B, C5, H5, W5)
            ], dim=1))
            
            return {'C3': fused_c3, 'C4': fused_c4, 'C5': fused_c5}

        elif self.mode == 'inter_level':
            # --- cross-layer attention logic (example implementation) ---
            # refine C3a (fine) & C3b (fine): use c4b (coarse) and c4a (coarse) as key/value respectively
            c4b_resized = F.interpolate(c4b, size=(H3, W3), mode='bilinear', align_corners=False)
            c3a_r = self.cross_attn_c4_to_c3(self._flatten(c3a), self._flatten(c4b_resized))
            
            c4a_resized = F.interpolate(c4a, size=(H3, W3), mode='bilinear', align_corners=False)
            c3b_r = self.cross_attn_c4_to_c3_rev(self._flatten(c3b), self._flatten(c4a_resized))

            fused_c3 = self.final_fuse_c3(torch.cat([
                self._reshape(c3a_r, B, C3, H3, W3),
                self._reshape(c3b_r, B, C3, H3, W3)
            ], dim=1))

            # refine C4a (medium) & C4b (medium): use c5b (coarsest) and c5a (coarsest) as key/value respectively
            c5b_resized = F.interpolate(c5b, size=(H4, W4), mode='bilinear', align_corners=False)
            c4a_r = self.cross_attn_c5_to_c4(self._flatten(c4a), self._flatten(c5b_resized))
            
            c5a_resized = F.interpolate(c5a, size=(H4, W4), mode='bilinear', align_corners=False)
            c4b_r = self.cross_attn_c5_to_c4_rev(self._flatten(c4b), self._flatten(c5a_resized))

            fused_c4 = self.final_fuse_c4(torch.cat([
                self._reshape(c4a_r, B, C4, H4, W4),
                self._reshape(c4b_r, B, C4, H4, W4)
            ], dim=1))

            # C5 is the coarsest level, so nothing coarser can refine it; use view A directly (or a simple fusion)
            fused_c5 = self.final_fuse_c5(c5a)
            return {'C3': fused_c3, 'C4': fused_c4, 'C5': fused_c5}
