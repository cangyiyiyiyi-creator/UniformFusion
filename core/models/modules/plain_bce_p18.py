from __future__ import annotations

import torch
import torch.nn as nn

from .plain_bce_innovations import (
    MultiScaleClassRegionEncoder,
    ScheduledInnovationHead,
    _bounded_parameter,
    _pair_features,
)


class P18EWSARHead(ScheduledInnovationHead):
    """Expert-warmup soft-advantage regional routing."""

    mode = "p18_ewsar"

    def __init__(
        self,
        level_dims,
        num_classes: int,
        levels=("C4", "C5"),
        projection_dim: int = 64,
        topk: int = 8,
        temperature: float = 0.2,
        dropout: float = 0.1,
        base_floor: float = 0.85,
        gamma_init: float = 0.003,
        gamma_max: float = 0.03,
        warmup_epochs: int = 15,
        ramp_epochs: int = 10,
        router_start_epoch: int = 15,
        router_ramp_epochs: int = 5,
        advantage_temperature: float = 0.05,
    ):
        super().__init__(warmup_epochs, ramp_epochs)
        self.num_classes = int(num_classes)
        self.base_floor = min(max(float(base_floor), 0.0), 0.99)
        self.gamma_max = max(float(gamma_max), 1e-8)
        self.router_start_epoch = max(int(router_start_epoch), 0)
        self.router_ramp_epochs = max(int(router_ramp_epochs), 1)
        self.advantage_temperature = max(
            float(advantage_temperature), 1e-4
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
        pair_dim = 4 * dim
        self.region_classifier = nn.Linear(dim, 1)
        self.region_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.pair_expert = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 1),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(pair_dim + 5),
            nn.Linear(pair_dim + 5, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 4),
        )
        nn.init.zeros_(self.pair_expert[-1].weight)
        nn.init.zeros_(self.pair_expert[-1].bias)
        nn.init.zeros_(self.router[-1].weight)
        with torch.no_grad():
            self.router[-1].bias.copy_(
                torch.tensor([0.85, 0.04, 0.04, 0.07]).log()
            )
        self.gamma_raw = _bounded_parameter(gamma_init, self.gamma_max)

    def _router_stage(self, reference: torch.Tensor) -> torch.Tensor:
        epoch = int(self.innovation_epoch.item())
        if epoch < self.router_start_epoch:
            value = 0.0
        else:
            value = min(
                1.0,
                float(epoch - self.router_start_epoch + 1)
                / float(self.router_ramp_epochs),
            )
        return reference.new_tensor(value)

    def forward(
        self,
        features_a,
        features_b,
        logits_base,
        logits_a=None,
        logits_b=None,
        logits_counterfactual=None,
    ):
        if logits_a is None or logits_b is None:
            raise ValueError("P18 requires counterfactual single-view logits")

        route_base = (
            logits_base
            if logits_counterfactual is None
            else logits_counterfactual
        )
        region_a, confidence_a = self.encoder(features_a)
        region_b, confidence_b = self.encoder(features_b)
        paired = _pair_features(region_a, region_b)
        aux_a = self.region_classifier(region_a).squeeze(-1)
        aux_b = self.region_classifier(region_b).squeeze(-1)
        aux_a = aux_a + self.region_bias.unsqueeze(0)
        aux_b = aux_b + self.region_bias.unsqueeze(0)

        pair_delta = 4.0 * torch.tanh(
            self.pair_expert(paired).squeeze(-1) / 4.0
        )
        pair_logits = route_base + pair_delta
        pair_aux_logits = route_base.detach() + pair_delta

        base_probability = torch.sigmoid(route_base.detach().float())
        uncertainty = 1.0 - 2.0 * (base_probability - 0.5).abs()
        stats = torch.stack(
            [
                confidence_a,
                confidence_b,
                uncertainty.to(dtype=confidence_a.dtype),
                (logits_a - logits_b).detach().abs().tanh(),
                (aux_a - aux_b).detach().abs().tanh(),
            ],
            dim=-1,
        )
        router_logits = self.router(torch.cat([paired, stats], dim=-1))
        learned_weights = router_logits.softmax(dim=-1)
        anchor = learned_weights.new_zeros(learned_weights.shape)
        anchor[..., 0] = 1.0
        weights = self.base_floor * anchor + (
            1.0 - self.base_floor
        ) * learned_weights
        experts = torch.stack(
            [route_base, logits_a, logits_b, pair_logits], dim=-1
        )
        routed = (weights * experts).sum(dim=-1)
        gamma = self.gamma_max * torch.sigmoid(self.gamma_raw)
        correction_candidate = gamma * 4.0 * torch.tanh(
            (routed - route_base) / 4.0
        )
        ramp = self.ramp(correction_candidate)
        route_stage = self._router_stage(correction_candidate)
        correction = ramp * route_stage * correction_candidate

        return {
            "mode": self.mode,
            "correction": correction,
            "correction_candidate": correction_candidate,
            "guard_correction": correction,
            "aux_logits": 0.5 * (aux_a + aux_b),
            "pair_aux_logits": pair_aux_logits,
            "expert_logits": experts,
            "router_logits": router_logits,
            "router_weights": weights,
            "single_logits_a": logits_a,
            "single_logits_b": logits_b,
            "gate": route_stage * (1.0 - weights[..., 0]),
            "confidence": 0.5 * (confidence_a + confidence_b),
            "gamma": gamma,
            "ramp": ramp * route_stage,
            "route_stage": route_stage,
            "advantage_temperature": correction_candidate.new_tensor(
                self.advantage_temperature
            ),
        }
