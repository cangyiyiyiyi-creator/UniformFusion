from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .plain_bce_innovations import (
    MultiScaleClassRegionEncoder,
    ScheduledInnovationHead,
    _bounded_parameter,
    _pair_features,
    _probability_logit,
)


class P17DCASRHead(ScheduledInnovationHead):
    """Dual-consensus active safe rescue from a frozen visual anchor."""

    mode = "p17_dcasr"

    def __init__(
        self,
        level_dims,
        num_classes: int,
        levels=("C4", "C5"),
        projection_dim: int = 64,
        topk: int = 8,
        temperature: float = 0.2,
        dropout: float = 0.1,
        gamma_init: float = 0.03,
        gamma_max: float = 0.10,
        warmup_epochs: int = 13,
        ramp_epochs: int = 5,
        router_start_epoch: int = 8,
        router_ramp_epochs: int = 3,
        gate_init: float = 0.15,
        trust_threshold: float = 0.50,
        trust_temperature: float = 0.10,
        uncertainty_floor: float = 0.25,
        budget_target: float = 0.12,
    ):
        super().__init__(warmup_epochs, ramp_epochs)
        valid_levels = tuple(level for level in levels if level in level_dims)
        if len(valid_levels) < 2:
            raise ValueError("P17 requires two valid feature levels")
        self.levels = valid_levels[:2]
        self.num_classes = int(num_classes)
        self.gamma_max = max(float(gamma_max), 1e-8)
        self.router_start_epoch = max(int(router_start_epoch), 0)
        self.router_ramp_epochs = max(int(router_ramp_epochs), 1)
        self.trust_threshold = min(max(float(trust_threshold), 0.0), 1.0)
        self.trust_temperature = max(float(trust_temperature), 1e-3)
        self.uncertainty_floor = min(max(float(uncertainty_floor), 0.0), 1.0)
        self.budget_target = min(max(float(budget_target), 0.0), 1.0)

        self.encoder = MultiScaleClassRegionEncoder(
            level_dims=level_dims,
            levels=self.levels,
            num_classes=num_classes,
            projection_dim=projection_dim,
            topk=topk,
            temperature=temperature,
        )
        dim = self.encoder.projection_dim
        pair_dim = 4 * dim
        self.evidence_heads = nn.ModuleDict({
            level: nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(dim, 1),
            )
            for level in self.levels
        })
        router_dim = len(self.levels) * pair_dim + 7
        self.router = nn.Sequential(
            nn.LayerNorm(router_dim),
            nn.Linear(router_dim, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, _probability_logit(gate_init))
        gamma = _bounded_parameter(gamma_init, self.gamma_max)
        self.class_gamma_raw = nn.Parameter(
            gamma.detach().repeat(self.num_classes)
        )

    def _router_ramp(self, reference: torch.Tensor) -> torch.Tensor:
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

    def _level_descriptor(self, features, level):
        selected, attention, confidence = self.encoder.level_tokens(
            features, level
        )
        descriptor = (selected * attention.unsqueeze(-1)).sum(dim=2)
        return descriptor, confidence

    @staticmethod
    def _consensus(level_logits):
        directions = torch.tanh(level_logits / 2.0)
        agreement = F.relu(directions[..., 0] * directions[..., 1])
        consensus = directions.mean(dim=-1)
        return directions, agreement, consensus

    def _router_inputs(
        self,
        paired_levels,
        confidence_levels,
        level_logits,
        base_uncertainty,
    ):
        directions, agreement, consensus = self._consensus(level_logits)
        level_gap = (
            torch.sigmoid(level_logits[..., 0])
            - torch.sigmoid(level_logits[..., 1])
        ).abs()
        evidence_strength = directions.abs().mean(dim=-1)
        stats = torch.stack(
            [
                confidence_levels[..., 0],
                confidence_levels[..., 1],
                base_uncertainty.to(dtype=level_logits.dtype),
                level_gap,
                evidence_strength,
                agreement,
                consensus.abs(),
            ],
            dim=-1,
        )
        router_input = torch.cat([*paired_levels, stats], dim=-1)
        return router_input, directions, agreement, consensus

    def forward(self, features_a, features_b, logits_base, **_):
        descriptors_a = {}
        descriptors_b = {}
        confidences_a = {}
        confidences_b = {}
        paired_levels = []
        level_logits = []
        for level in self.levels:
            descriptor_a, confidence_a = self._level_descriptor(
                features_a, level
            )
            descriptor_b, confidence_b = self._level_descriptor(
                features_b, level
            )
            descriptors_a[level] = descriptor_a
            descriptors_b[level] = descriptor_b
            confidences_a[level] = confidence_a
            confidences_b[level] = confidence_b
            paired = _pair_features(descriptor_a, descriptor_b)
            paired_levels.append(paired)
            level_logits.append(
                self.evidence_heads[level](paired).squeeze(-1)
            )

        level_logits = torch.stack(level_logits, dim=-1)
        confidence_levels = torch.stack(
            [
                0.5 * (confidences_a[level] + confidences_b[level])
                for level in self.levels
            ],
            dim=-1,
        )
        base_probability = torch.sigmoid(logits_base.detach().float())
        base_uncertainty = 1.0 - 2.0 * (base_probability - 0.5).abs()
        (
            router_input,
            directions,
            agreement,
            consensus,
        ) = self._router_inputs(
            paired_levels,
            confidence_levels,
            level_logits,
            base_uncertainty,
        )
        router_logits = self.router(router_input).squeeze(-1)
        raw_trust = torch.sigmoid(router_logits)
        release = torch.sigmoid(
            (raw_trust - self.trust_threshold) / self.trust_temperature
        )
        opportunity = self.uncertainty_floor + (
            1.0 - self.uncertainty_floor
        ) * base_uncertainty
        gate = (
            release.float()
            * agreement.float()
            * opportunity.float()
        ).clamp(0.0, 1.0)

        class_gamma = self.gamma_max * torch.sigmoid(self.class_gamma_raw)
        gamma = class_gamma.unsqueeze(0).to(dtype=consensus.dtype)
        evidence_delta = torch.tanh(consensus)
        correction_candidate = gamma * gate.to(gamma.dtype) * evidence_delta
        ramp = self.ramp(correction_candidate)

        # The fixed probe decouples route supervision from a closing learned gate.
        route_probe_correction = (
            self.gamma_max
            * agreement.float()
            * opportunity.float()
            * evidence_delta.float()
        )
        route_probe_logits = (
            logits_base.detach().float() + route_probe_correction
        )
        router_ramp = self._router_ramp(correction_candidate)

        mismatch_router_logits = None
        if self.training and logits_base.shape[0] > 1:
            mismatch_pairs = []
            mismatch_logits = []
            mismatch_confidences = []
            for level in self.levels:
                descriptor_b = descriptors_b[level].roll(1, dims=0)
                paired = _pair_features(descriptors_a[level], descriptor_b)
                mismatch_pairs.append(paired)
                mismatch_logits.append(
                    self.evidence_heads[level](paired).squeeze(-1)
                )
                mismatch_confidences.append(
                    0.5 * (
                        confidences_a[level]
                        + confidences_b[level].roll(1, dims=0)
                    )
                )
            mismatch_logits = torch.stack(mismatch_logits, dim=-1)
            mismatch_confidences = torch.stack(
                mismatch_confidences, dim=-1
            )
            mismatch_input, _, _, _ = self._router_inputs(
                mismatch_pairs,
                mismatch_confidences,
                mismatch_logits,
                base_uncertainty,
            )
            mismatch_router_logits = self.router(mismatch_input).squeeze(-1)

        return {
            "mode": self.mode,
            "correction": ramp * correction_candidate,
            "correction_candidate": correction_candidate,
            "guard_correction": ramp * correction_candidate,
            "aux_logits": level_logits.mean(dim=-1),
            "level_aux_logits": level_logits,
            "rank_logits": level_logits.mean(dim=-1),
            "consistency": (
                torch.sigmoid(level_logits[..., 0])
                - torch.sigmoid(level_logits[..., 1])
            ).abs(),
            "route_probe_logits": route_probe_logits,
            "router_logits": router_logits,
            "mismatch_router_logits": mismatch_router_logits,
            "route_stage": router_ramp,
            "gate": gate,
            "raw_gate": raw_trust,
            "confidence": confidence_levels.mean(dim=-1),
            "agreement": agreement,
            "gamma": class_gamma,
            "ramp": ramp,
            "directions": directions,
            "opportunity": opportunity,
            "budget_target": correction_candidate.new_tensor(
                self.budget_target
            ),
        }
