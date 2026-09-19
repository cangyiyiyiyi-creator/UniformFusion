from __future__ import annotations

import torch

from .plain_bce_p18 import P18EWSARHead


class P19APCERHead(P18EWSARHead):
    """Anchor-preserving counterfactual expert routing."""

    mode = "p19_apcer"

    def __init__(
        self,
        *args,
        corrupt_probability: float = 0.75,
        corrupt_ratio: float = 0.25,
        full_view_drop_probability: float = 0.35,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.corrupt_probability = min(
            max(float(corrupt_probability), 0.0), 1.0
        )
        self.corrupt_ratio = min(max(float(corrupt_ratio), 0.0), 0.95)
        self.full_view_drop_probability = min(
            max(float(full_view_drop_probability), 0.0), 1.0
        )

    def prepare_training_views(self, features_a, features_b):
        if not self.training or self.corrupt_probability <= 0.0:
            return features_a, features_b

        reference = next(iter(features_a.values()))
        batch = reference.shape[0]
        corrupt = (
            torch.rand(batch, device=reference.device)
            < self.corrupt_probability
        )
        corrupt_a = corrupt & (
            torch.rand(batch, device=reference.device) < 0.5
        )
        corrupt_b = corrupt & ~corrupt_a
        full_drop = corrupt & (
            torch.rand(batch, device=reference.device)
            < self.full_view_drop_probability
        )
        output_a = {}
        output_b = {}
        for level, feature_a in features_a.items():
            feature_b = features_b[level]
            spatial_shape = (batch, 1, feature_a.shape[-2], feature_a.shape[-1])
            keep_a = (
                torch.rand(spatial_shape, device=feature_a.device)
                >= self.corrupt_ratio
            ).to(dtype=feature_a.dtype)
            keep_b = (
                torch.rand(spatial_shape, device=feature_b.device)
                >= self.corrupt_ratio
            ).to(dtype=feature_b.dtype)
            active_a = corrupt_a.reshape(batch, 1, 1, 1)
            active_b = corrupt_b.reshape(batch, 1, 1, 1)
            full_a = (corrupt_a & full_drop).reshape(batch, 1, 1, 1)
            full_b = (corrupt_b & full_drop).reshape(batch, 1, 1, 1)
            mask_a = torch.where(active_a, keep_a, torch.ones_like(keep_a))
            mask_b = torch.where(active_b, keep_b, torch.ones_like(keep_b))
            mask_a = torch.where(full_a, torch.zeros_like(mask_a), mask_a)
            mask_b = torch.where(full_b, torch.zeros_like(mask_b), mask_b)
            output_a[level] = feature_a * mask_a
            output_b[level] = feature_b * mask_b
        return output_a, output_b

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output["mode"] = self.mode
        return output
