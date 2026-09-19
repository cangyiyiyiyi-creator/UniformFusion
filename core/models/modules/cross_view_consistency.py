import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_CV_CACHE = {
    "semantic": [],
    "geometry": [],
}


def reset_cv_cache():
    _CV_CACHE["semantic"].clear()
    _CV_CACHE["geometry"].clear()


def push_cv_consistency(semantic_loss=None, geometry_loss=None):
    if semantic_loss is not None:
        _CV_CACHE["semantic"].append(semantic_loss)
    if geometry_loss is not None:
        _CV_CACHE["geometry"].append(geometry_loss)


def pop_cv_consistency(lambda_sem=0.0, lambda_geo=0.0):
    total = None

    if lambda_sem > 0 and len(_CV_CACHE["semantic"]) > 0:
        sem_loss = sum(_CV_CACHE["semantic"]) / len(_CV_CACHE["semantic"])
        total = lambda_sem * sem_loss if total is None else total + lambda_sem * sem_loss

    if lambda_geo > 0 and len(_CV_CACHE["geometry"]) > 0:
        geo_loss = sum(_CV_CACHE["geometry"]) / len(_CV_CACHE["geometry"])
        total = lambda_geo * geo_loss if total is None else total + lambda_geo * geo_loss

    reset_cv_cache()

    if total is None:
        return 0.0
    return total


class CrossViewGeoSemanticAlign(nn.Module):
    """
    CV-GSC: Cross-view Geometric-Semantic Consistency Alignment.

    输入：
        f1: view-1 feature, [B, C, H, W]
        f2: view-2 feature, [B, C, H, W]

    输出：
        f1_out, f2_out

    作用：
        1) view1 查询 view2；
        2) view2 查询 view1；
        3) 训练阶段缓存 semantic consistency loss；
        4) 训练阶段缓存 geometry response consistency loss。
    """

    def __init__(
        self,
        channels,
        reduction=4,
        spatial_reduction=2,
        init_scale=1e-3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.spatial_reduction = int(spatial_reduction)

        hidden = max(self.channels // reduction, 32)

        self.coord_proj = nn.Sequential(
            nn.Conv2d(2, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.GELU(),
        )

        self.q = nn.Conv2d(self.channels, hidden, 1, bias=False)
        self.k = nn.Conv2d(self.channels, hidden, 1, bias=False)
        self.v = nn.Conv2d(self.channels, self.channels, 1, bias=False)

        self.gate1 = nn.Sequential(
            nn.Conv2d(self.channels * 4, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.gate2 = nn.Sequential(
            nn.Conv2d(self.channels * 4, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, 1, bias=True),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
        )

        geo_hidden = max(self.channels // 8, 16)
        self.geo_head = nn.Sequential(
            nn.Conv2d(self.channels, geo_hidden, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(geo_hidden, 1, 1, bias=True),
        )

        self.gamma = nn.Parameter(torch.tensor(float(init_scale)))

    def _coord(self, x):
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return grid

    def _pool(self, x):
        if self.spatial_reduction <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=self.spatial_reduction, stride=self.spatial_reduction)

    def _cross_attn(self, query_feat, key_value_feat, out_size):
        B, C, H, W = query_feat.shape

        qf = self._pool(query_feat)
        kvf = self._pool(key_value_feat)

        q = self.q(qf).flatten(2).transpose(1, 2)       # [B,N,d]
        k = self.k(kvf).flatten(2)                      # [B,d,N]
        v = self.v(kvf).flatten(2).transpose(1, 2)      # [B,N,C]

        attn = torch.matmul(q, k) / math.sqrt(max(q.size(-1), 1))
        attn = torch.softmax(attn, dim=-1)

        aligned = torch.matmul(attn, v)                 # [B,N,C]
        Hp, Wp = qf.shape[-2:]
        aligned = aligned.transpose(1, 2).reshape(B, C, Hp, Wp)

        if aligned.shape[-2:] != out_size:
            aligned = F.interpolate(aligned, size=out_size, mode="bilinear", align_corners=False)
        return aligned

    def _semantic_loss(self, f1, f2):
        z1 = F.adaptive_avg_pool2d(f1, 1).flatten(1)
        z2 = F.adaptive_avg_pool2d(f2, 1).flatten(1)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return 1.0 - (z1 * z2).sum(dim=-1).mean()

    def _geometry_loss(self, f1, f2):
        g1 = self.geo_head(f1).flatten(2)
        g2 = self.geo_head(f2).flatten(2)
        g1 = torch.softmax(g1, dim=-1)
        g2 = torch.softmax(g2, dim=-1)
        return F.mse_loss(g1, g2)

    def forward(self, f1, f2):
        B, C, H, W = f1.shape

        coord1 = self.coord_proj(self._coord(f1))
        coord2 = self.coord_proj(self._coord(f2))

        f1c = f1 + coord1
        f2c = f2 + coord2

        a12 = self._cross_attn(f1c, f2c, out_size=(H, W))
        a21 = self._cross_attn(f2c, f1c, out_size=(H, W))

        gate1 = self.gate1(torch.cat([f1, a12, torch.abs(f1 - a12), f1 * a12], dim=1))
        gate2 = self.gate2(torch.cat([f2, a21, torch.abs(f2 - a21), f2 * a21], dim=1))

        f1_out = f1 + self.gamma * gate1 * self.out_proj(a12 - f1)
        f2_out = f2 + self.gamma * gate2 * self.out_proj(a21 - f2)

        if self.training:
            push_cv_consistency(
                semantic_loss=self._semantic_loss(f1_out, f2_out),
                geometry_loss=self._geometry_loss(f1_out, f2_out),
            )

        return f1_out, f2_out
