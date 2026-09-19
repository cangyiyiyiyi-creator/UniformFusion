from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .plain_bce_innovations import (
    MultiScaleClassRegionEncoder,
    ScheduledInnovationHead,
    _bounded_parameter,
    _pair_features,
)


class P16FACGRHead(ScheduledInnovationHead):
    """Frozen-anchor counterfactual gain routing for paired regions."""

    mode = "p16_facgr"

    def __init__(
        self,
        level_dims,
        num_classes: int,
        levels=("C4", "C5"),
        projection_dim: int = 64,
        topk: int = 8,
        temperature: float = 0.2,
        dropout: float = 0.1,
        base_floor: float = 0.65,
        gamma_init: float = 0.03,
        gamma_max: float = 0.20,
        warmup_epochs: int = 5,
        ramp_epochs: int = 5,
    ):
        super().__init__(warmup_epochs, ramp_epochs)
        self.num_classes = int(num_classes)
        self.base_floor = min(max(float(base_floor), 0.0), 0.99)
        self.gamma_max = max(float(gamma_max), 1e-8)
        self.encoder = MultiScaleClassRegionEncoder(
            level_dims=level_dims,
            levels=levels,
            num_classes=num_classes,
            projection_dim=projection_dim,
            topk=topk,
            temperature=temperature,
        )
        dim = self.encoder.projection_dim
        pair_dim = 4 * dim
        self.pair_residual = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 1),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(pair_dim + 6),
            nn.Linear(pair_dim + 6, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 4),
        )
        nn.init.zeros_(self.pair_residual[-1].weight)
        nn.init.zeros_(self.pair_residual[-1].bias)
        nn.init.zeros_(self.router[-1].weight)
        with torch.no_grad():
            self.router[-1].bias.copy_(
                torch.tensor([0.90, 0.025, 0.025, 0.05]).log()
            )
        gamma = _bounded_parameter(gamma_init, self.gamma_max)
        self.class_gamma_raw = nn.Parameter(
            gamma.detach().repeat(self.num_classes)
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
            raise ValueError("P16 requires counterfactual single-view logits")

        region_a, confidence_a = self.encoder(features_a)
        region_b, confidence_b = self.encoder(features_b)
        paired = _pair_features(region_a, region_b)
        single_mean = 0.5 * (logits_a + logits_b)
        pair_delta = 4.0 * torch.tanh(
            self.pair_residual(paired).squeeze(-1) / 4.0
        )
        pair_logits = single_mean + pair_delta

        base_probability = torch.sigmoid(logits_base.detach().float())
        base_uncertainty = 1.0 - 2.0 * (base_probability - 0.5).abs()
        region_confidence = 0.5 * (confidence_a + confidence_b)
        single_disagreement = (logits_a - logits_b).detach().abs().tanh()
        fusion_synergy = (logits_base.detach() - single_mean.detach()).abs().tanh()
        pair_novelty = (pair_logits.detach() - logits_base.detach()).abs().tanh()
        directional_support = (
            torch.tanh(logits_a.detach() - logits_base.detach())
            * torch.tanh(logits_b.detach() - logits_base.detach())
        )
        stats = torch.stack(
            [
                confidence_a,
                confidence_b,
                base_uncertainty.to(dtype=confidence_a.dtype),
                single_disagreement,
                fusion_synergy,
                directional_support,
            ],
            dim=-1,
        )
        router_logits = self.router(torch.cat([paired, stats], dim=-1))
        learned_weights = router_logits.softmax(dim=-1)
        alternative_weights = learned_weights[..., 1:]
        alternative_weights = alternative_weights / alternative_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        alternatives = torch.stack([logits_a, logits_b, pair_logits], dim=-1)
        alternative_logits = (alternative_weights * alternatives).sum(dim=-1)

        anchor_certainty = 1.0 - base_uncertainty
        anchor_opportunity = 0.15 + 0.85 * (
            1.0 - anchor_certainty.square()
        )
        anchor_budget = (1.0 - self.base_floor) * anchor_opportunity
        evidence_budget = 0.40 + 0.60 * region_confidence.float()
        predicted_gain = 1.0 - learned_weights[..., 0].float()
        gate = (
            anchor_budget * evidence_budget * predicted_gain
        ).clamp(0.0, 1.0)
        class_gamma = self.gamma_max * torch.sigmoid(self.class_gamma_raw)
        gamma = class_gamma.unsqueeze(0).to(dtype=alternative_logits.dtype)
        correction_candidate = gamma * gate.to(gamma.dtype) * 4.0 * torch.tanh(
            (alternative_logits - logits_base) / 4.0
        )
        ramp = self.ramp(correction_candidate)

        experts = torch.stack(
            [logits_base, logits_a, logits_b, pair_logits], dim=-1
        )
        applied_weights = torch.cat(
            [
                (1.0 - gate).unsqueeze(-1),
                gate.unsqueeze(-1) * alternative_weights.float(),
            ],
            dim=-1,
        )
        return {
            "mode": self.mode,
            "correction": ramp * correction_candidate,
            "correction_candidate": correction_candidate,
            "guard_correction": ramp * correction_candidate,
            "aux_logits": pair_logits,
            "expert_logits": experts,
            "router_logits": router_logits,
            "router_weights": applied_weights,
            "single_logits_a": logits_a,
            "single_logits_b": logits_b,
            "gate": gate,
            "confidence": region_confidence,
            "gamma": class_gamma,
            "ramp": ramp,
            "anchor_budget": anchor_budget,
            "pair_novelty": pair_novelty,
        }
