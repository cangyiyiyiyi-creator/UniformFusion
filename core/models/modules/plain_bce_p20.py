from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .plain_bce_innovations import (
    MultiScaleClassRegionEncoder,
    ScheduledInnovationHead,
)


class P20CVCRHead(ScheduledInnovationHead):
    """Training-only cross-view complementarity ranking branch."""

    mode = "p20_cvcr"

    def __init__(
        self,
        level_dims,
        num_classes: int,
        levels=("C4", "C5"),
        projection_dim: int = 64,
        topk: int = 8,
        temperature: float = 0.2,
        dropout: float = 0.1,
        warmup_epochs: int = 5,
        ramp_epochs: int = 10,
        complement_margin: float = 0.02,
        complement_temperature: float = 0.05,
    ):
        super().__init__(warmup_epochs, ramp_epochs)
        self.num_classes = int(num_classes)
        self.complement_margin = max(float(complement_margin), 0.0)
        self.complement_temperature = max(
            float(complement_temperature), 1e-4
        )
        self.encoder = MultiScaleClassRegionEncoder(
            level_dims=level_dims,
            levels=levels,
            num_classes=num_classes,
            projection_dim=projection_dim,
            topk=topk,
            temperature=temperature,
        )
        dim = self.encoder.projection_dim
        self.view_classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.pair_expert = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.pair_expert[-1].weight)
        nn.init.zeros_(self.pair_expert[-1].bias)

    @staticmethod
    def _symmetric_pair(region_a: torch.Tensor, region_b: torch.Tensor):
        return torch.cat(
            [
                0.5 * (region_a + region_b),
                (region_a - region_b).abs(),
                region_a * region_b,
            ],
            dim=-1,
        )

    def forward(
        self,
        features_a,
        features_b,
        logits_base,
        logits_a=None,
        logits_b=None,
    ):
        if logits_a is None or logits_b is None:
            raise ValueError("P20 requires clean single-view logits")

        region_a, confidence_a = self.encoder(features_a)
        region_b, confidence_b = self.encoder(features_b)
        view_aux_a = self.view_classifier(region_a).squeeze(-1)
        view_aux_b = self.view_classifier(region_b).squeeze(-1)
        bias = self.class_bias.unsqueeze(0)
        view_aux_a = view_aux_a + bias
        view_aux_b = view_aux_b + bias

        paired = self._symmetric_pair(region_a, region_b)
        pair_delta = self.pair_expert(paired).squeeze(-1)
        pair_aux_logits = 0.5 * (view_aux_a + view_aux_b) + pair_delta

        regional_novelty = 0.5 * (
            1.0 - F.cosine_similarity(region_a, region_b, dim=-1)
        ).clamp(0.0, 1.0)
        view_disagreement = (
            torch.sigmoid(logits_a.detach().float())
            - torch.sigmoid(logits_b.detach().float())
        ).abs().to(dtype=regional_novelty.dtype)
        confidence = 0.5 * (confidence_a + confidence_b)
        complementarity = (
            0.5 * regional_novelty + 0.5 * view_disagreement
        ) * (0.5 + 0.5 * confidence)
        complementarity = complementarity.clamp(0.0, 1.0)

        zero = logits_base * 0.0
        stage = self.ramp(logits_base)
        return {
            "mode": self.mode,
            "correction": zero,
            "correction_candidate": zero,
            "aux_logits": pair_aux_logits,
            "rank_logits": logits_base,
            "single_logits_a": logits_a,
            "single_logits_b": logits_b,
            "complementarity": complementarity,
            "complement_margin": logits_base.new_tensor(
                self.complement_margin
            ),
            "complement_temperature": logits_base.new_tensor(
                self.complement_temperature
            ),
            "loss_stage": stage,
            "gate": complementarity,
            "confidence": confidence,
            "gamma": zero.new_tensor(0.0),
            "ramp": stage,
        }
