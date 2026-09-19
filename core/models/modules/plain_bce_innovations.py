from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_evidence import CrossViewVisualEvidenceRouter


def _probability_logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


def _bounded_parameter(initial: float, maximum: float) -> nn.Parameter:
    maximum = max(float(maximum), 1e-8)
    ratio = min(max(float(initial) / maximum, 1e-4), 1.0 - 1e-4)
    return nn.Parameter(torch.tensor(_probability_logit(ratio)))


class ScheduledInnovationHead(nn.Module):
    def __init__(self, warmup_epochs: int, ramp_epochs: int):
        super().__init__()
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.ramp_epochs = max(int(ramp_epochs), 0)
        self.register_buffer("innovation_epoch", torch.zeros((), dtype=torch.long))

    def set_epoch(self, epoch: int) -> None:
        self.innovation_epoch.fill_(max(int(epoch), 0))

    def ramp(self, reference: torch.Tensor) -> torch.Tensor:
        epoch = int(self.innovation_epoch.item())
        if epoch < self.warmup_epochs:
            value = 0.0
        elif self.ramp_epochs == 0:
            value = 1.0
        else:
            value = min(
                1.0,
                float(epoch - self.warmup_epochs + 1) / self.ramp_epochs,
            )
        return reference.new_tensor(value)


class MultiScaleClassRegionEncoder(nn.Module):
    """Shared class-conditioned region pooling for paired feature pyramids."""

    def __init__(
        self,
        level_dims: Dict[str, int],
        levels: Iterable[str],
        num_classes: int,
        projection_dim: int = 64,
        topk: int = 8,
        temperature: float = 0.2,
    ):
        super().__init__()
        self.levels = tuple(level for level in levels if level in level_dims)
        if not self.levels:
            raise ValueError("region encoder requires at least one valid level")
        self.num_classes = int(num_classes)
        self.projection_dim = max(int(projection_dim), 16)
        self.topk = max(int(topk), 1)
        self.temperature = max(float(temperature), 1e-6)
        self.projections = nn.ModuleDict({
            level: nn.Sequential(
                nn.Conv2d(
                    int(level_dims[level]), self.projection_dim, 1, bias=False
                ),
                nn.GroupNorm(1, self.projection_dim),
                nn.GELU(),
            )
            for level in self.levels
        })
        self.class_queries = nn.Parameter(
            torch.empty(self.num_classes, self.projection_dim)
        )
        nn.init.trunc_normal_(self.class_queries, std=0.02)

    def _score_tokens(self, tokens: torch.Tensor):
        normalized_tokens = F.normalize(tokens, dim=-1)
        queries = F.normalize(self.class_queries, dim=-1)
        scores = torch.einsum(
            "bnd,kd->bkn", normalized_tokens, queries
        ) / self.temperature
        keep = min(self.topk, scores.shape[-1])
        values, indices = scores.topk(keep, dim=-1)
        attention = values.float().softmax(dim=-1).to(dtype=tokens.dtype)
        if keep > 1:
            entropy = -(
                attention * attention.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(float(keep))
            confidence = (1.0 - entropy).clamp(0.0, 1.0)
        else:
            confidence = attention.new_ones(attention.shape[:2])
        expanded = tokens.unsqueeze(1).expand(
            -1, self.num_classes, -1, -1
        )
        selected = expanded.gather(
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, tokens.shape[-1]),
        )
        return selected, attention, confidence

    def level_tokens(self, features: Dict[str, torch.Tensor], level: str):
        if level not in self.projections or level not in features:
            raise KeyError(f"missing region level: {level}")
        projected = self.projections[level](features[level])
        tokens = projected.flatten(2).transpose(1, 2)
        selected, attention, confidence = self._score_tokens(tokens)
        return selected, attention, confidence

    def forward(self, features: Dict[str, torch.Tensor]):
        descriptors = []
        confidences = []
        for level in self.levels:
            selected, attention, confidence = self.level_tokens(features, level)
            descriptors.append(
                (selected * attention.unsqueeze(-1)).sum(dim=2)
            )
            confidences.append(confidence)
        return (
            torch.stack(descriptors, dim=0).mean(dim=0),
            torch.stack(confidences, dim=0).mean(dim=0),
        )


def _pair_features(region_a: torch.Tensor, region_b: torch.Tensor):
    return torch.cat(
        [
            region_a,
            region_b,
            (region_a - region_b).abs(),
            region_a * region_b,
        ],
        dim=-1,
    )


class P9CAPRSHead(ScheduledInnovationHead):
    """Counterfactual anchor-preserving regional selector."""

    mode = "p9_caprs"

    def __init__(
        self,
        level_dims: Dict[str, int],
        num_classes: int,
        levels=("C4", "C5"),
        projection_dim: int = 64,
        topk: int = 8,
        temperature: float = 0.2,
        dropout: float = 0.1,
        base_floor: float = 0.8,
        use_counterfactual_experts: bool = True,
        use_learned_router: bool = True,
        gamma_init: float = 0.005,
        gamma_max: float = 0.05,
        warmup_epochs: int = 15,
        ramp_epochs: int = 10,
    ):
        super().__init__(warmup_epochs, ramp_epochs)
        self.num_classes = int(num_classes)
        self.base_floor = min(max(float(base_floor), 0.0), 0.99)
        self.use_counterfactual_experts = bool(use_counterfactual_experts)
        self.use_learned_router = bool(use_learned_router)
        self.gamma_max = max(float(gamma_max), 1e-8)
        self.encoder = MultiScaleClassRegionEncoder(
            level_dims,
            levels,
            num_classes,
            projection_dim,
            topk,
            temperature,
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
        router_stats = 5 if self.use_counterfactual_experts else 4
        router_experts = 4 if self.use_counterfactual_experts else 2
        nn.init.zeros_(self.pair_expert[-1].weight)
        nn.init.zeros_(self.pair_expert[-1].bias)
        self.router = None
        if self.use_learned_router:
            self.router = nn.Sequential(
                nn.LayerNorm(pair_dim + router_stats),
                nn.Linear(pair_dim + router_stats, dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(dim, router_experts),
            )
            nn.init.zeros_(self.router[-1].weight)
            prior = (
                torch.tensor([0.80, 0.07, 0.07, 0.06])
                if self.use_counterfactual_experts
                else torch.tensor([0.93, 0.07])
            ).log()
            with torch.no_grad():
                self.router[-1].bias.copy_(prior)
        self.gamma_raw = _bounded_parameter(gamma_init, self.gamma_max)

    def forward(
        self,
        features_a,
        features_b,
        logits_base,
        logits_a=None,
        logits_b=None,
    ):
        if self.use_counterfactual_experts and (
            logits_a is None or logits_b is None
        ):
            raise ValueError("P9 requires counterfactual single-view logits")
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
        pair_logits = logits_base + pair_delta

        base_probability = torch.sigmoid(logits_base.detach().float())
        uncertainty = 1.0 - 2.0 * (base_probability - 0.5).abs()
        stats_items = [
            confidence_a,
            confidence_b,
            uncertainty.to(dtype=confidence_a.dtype),
        ]
        if self.use_counterfactual_experts:
            stats_items.append(
                (logits_a - logits_b).detach().abs().tanh()
            )
        stats_items.append((aux_a - aux_b).detach().abs().tanh())
        stats = torch.stack(stats_items, dim=-1)
        if self.use_learned_router:
            router_logits = self.router(torch.cat([paired, stats], dim=-1))
            learned_weights = router_logits.softmax(dim=-1)
        else:
            router_logits = None
            expert_count = 4 if self.use_counterfactual_experts else 2
            learned_weights = paired.new_full(
                (*paired.shape[:-1], expert_count),
                1.0 / expert_count,
            )
        anchor = learned_weights.new_zeros(learned_weights.shape)
        anchor[..., 0] = 1.0
        weights = self.base_floor * anchor + (
            1.0 - self.base_floor
        ) * learned_weights
        expert_items = [logits_base]
        if self.use_counterfactual_experts:
            expert_items.extend([logits_a, logits_b])
        expert_items.append(pair_logits)
        experts = torch.stack(expert_items, dim=-1)
        routed = (weights * experts).sum(dim=-1)
        gamma = self.gamma_max * torch.sigmoid(self.gamma_raw)
        correction_candidate = gamma * 4.0 * torch.tanh(
            (routed - logits_base) / 4.0
        )
        ramp = self.ramp(correction_candidate)
        return {
            "mode": self.mode,
            "correction": ramp * correction_candidate,
            "correction_candidate": correction_candidate,
            "aux_logits": 0.5 * (aux_a + aux_b),
            "expert_logits": experts,
            "router_logits": router_logits,
            "router_weights": weights,
            "single_logits_a": logits_a,
            "single_logits_b": logits_b,
            "use_counterfactual_experts": self.use_counterfactual_experts,
            "use_learned_router": self.use_learned_router,
            "gate": 1.0 - weights[..., 0],
            "confidence": 0.5 * (confidence_a + confidence_b),
            "gamma": gamma,
            "ramp": ramp,
        }


class P10WGCRHead(ScheduledInnovationHead):
    """Warm-up guarded, swap-symmetric version of CA-RER."""

    mode = "p10_wgcr"

    def __init__(
        self,
        level_dims: Dict[str, int],
        num_classes: int,
        levels=("C4", "C5"),
        projection_dim: int = 64,
        topk: int = 8,
        temperature: float = 0.2,
        dropout: float = 0.1,
        gamma_init: float = 0.005,
        gamma_max: float = 0.05,
        warmup_epochs: int = 15,
        ramp_epochs: int = 10,
    ):
        super().__init__(warmup_epochs, ramp_epochs)
        self.levels = tuple(level for level in levels if level in level_dims)
        if not self.levels:
            raise ValueError("P10 requires at least one valid level")
        self.routers = nn.ModuleDict({
            level: CrossViewVisualEvidenceRouter(
                channels=level_dims[level],
                num_classes=num_classes,
                mode="ca_rer",
                projection_dim=projection_dim,
                topk_ratio=min(1.0, max(1.0 / 64.0, float(topk) / 64.0)),
                temperature=temperature,
                gate_init=0.02,
                gamma_init=gamma_init,
                gamma_max=gamma_max,
                reject_temperature=0.1,
                dropout=dropout,
            )
            for level in self.levels
        })

    @staticmethod
    def _mean(outputs, key):
        return torch.stack([item[key] for item in outputs], dim=0).mean(dim=0)

    def forward(self, features_a, features_b, logits_base, **_):
        direct = []
        swapped = []
        for level, router in self.routers.items():
            direct.append(router(features_a[level], features_b[level], logits_base))
            swapped.append(router(features_b[level], features_a[level], logits_base))
        correction_direct = self._mean(direct, "correction")
        correction_swapped = self._mean(swapped, "correction")
        correction_candidate = 0.5 * (
            correction_direct + correction_swapped
        )
        ramp = self.ramp(correction_candidate)
        return {
            "mode": self.mode,
            "correction": ramp * correction_candidate,
            "correction_candidate": correction_candidate,
            "aux_logits": 0.5 * (
                self._mean(direct, "aux_logits")
                + self._mean(swapped, "aux_logits")
            ),
            "gate": 0.5 * (
                self._mean(direct, "trust")
                + self._mean(swapped, "trust")
            ),
            "confidence": 0.5 * (
                self._mean(direct, "confidence")
                + self._mean(swapped, "confidence")
            ),
            "consistency": (correction_direct - correction_swapped).abs(),
            "gamma": self._mean(direct, "gamma"),
            "ramp": ramp,
        }


class P11OTCVRHead(ScheduledInnovationHead):
    """Optimal-transport matching of class-conditioned cross-view regions."""

    mode = "p11_otcvr"

    def __init__(
        self,
        level_dims: Dict[str, int],
        num_classes: int,
        levels=("C4",),
        projection_dim: int = 64,
        topk: int = 8,
        temperature: float = 0.2,
        dropout: float = 0.1,
        gamma_init: float = 0.005,
        gamma_max: float = 0.05,
        warmup_epochs: int = 10,
        ramp_epochs: int = 10,
        sinkhorn_iters: int = 4,
    ):
        super().__init__(warmup_epochs, ramp_epochs)
        valid_levels = tuple(level for level in levels if level in level_dims)
        if not valid_levels:
            raise ValueError("P11 requires a valid OT level")
        self.match_level = valid_levels[0]
        self.gamma_max = max(float(gamma_max), 1e-8)
        self.temperature = max(float(temperature), 1e-4)
        self.sinkhorn_iters = max(int(sinkhorn_iters), 1)
        self.encoder = MultiScaleClassRegionEncoder(
            level_dims,
            (self.match_level,),
            num_classes,
            projection_dim,
            topk,
            temperature,
        )
        dim = self.encoder.projection_dim
        pair_dim = 4 * dim
        self.aux_head = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, dim), nn.GELU(),
            nn.Dropout(float(dropout)), nn.Linear(dim, 1),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, dim), nn.GELU(),
            nn.Dropout(float(dropout)), nn.Linear(dim, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(pair_dim + 3),
            nn.Linear(pair_dim + 3, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, _probability_logit(0.02))
        self.gamma_raw = _bounded_parameter(gamma_init, self.gamma_max)

    def _sinkhorn(self, similarity: torch.Tensor):
        logits = similarity.float() / self.temperature
        logits = logits - logits.amax(dim=(-2, -1), keepdim=True)
        kernel = logits.exp().clamp_min(1e-8)
        size_a, size_b = kernel.shape[-2:]
        marginal_a = kernel.new_full(
            kernel.shape[:-2] + (size_a,), 1.0 / size_a
        )
        marginal_b = kernel.new_full(
            kernel.shape[:-2] + (size_b,), 1.0 / size_b
        )
        u = torch.ones_like(marginal_a)
        v = torch.ones_like(marginal_b)
        for _ in range(self.sinkhorn_iters):
            u = marginal_a / torch.einsum(
                "...ij,...j->...i", kernel, v
            ).clamp_min(1e-8)
            v = marginal_b / torch.einsum(
                "...ij,...i->...j", kernel, u
            ).clamp_min(1e-8)
        return u.unsqueeze(-1) * kernel * v.unsqueeze(-2)

    def forward(self, features_a, features_b, logits_base, **_):
        tokens_a, attention_a, confidence_a = self.encoder.level_tokens(
            features_a, self.match_level
        )
        tokens_b, attention_b, confidence_b = self.encoder.level_tokens(
            features_b, self.match_level
        )
        normalized_a = F.normalize(tokens_a, dim=-1)
        normalized_b = F.normalize(tokens_b, dim=-1)
        similarity = torch.einsum(
            "bkid,bkjd->bkij", normalized_a, normalized_b
        )
        transport = self._sinkhorn(similarity).to(dtype=tokens_a.dtype)
        conditional_ab = transport * float(tokens_a.shape[-2])
        conditional_ba = transport.transpose(-1, -2) * float(tokens_b.shape[-2])
        matched_b = torch.einsum("bkij,bkjd->bkid", conditional_ab, tokens_b)
        matched_a = torch.einsum("bkji,bkid->bkjd", conditional_ba, tokens_a)
        consensus = 0.25 * (
            tokens_a + matched_b + tokens_b + matched_a
        ).mean(dim=2)
        complement_a = (tokens_a - matched_b).abs().mean(dim=2)
        complement_b = (tokens_b - matched_a).abs().mean(dim=2)
        interaction = complement_a * complement_b
        paired = torch.cat(
            [consensus, complement_a, complement_b, interaction], dim=-1
        )
        aux_logits = self.aux_head(paired).squeeze(-1)
        raw_delta = self.delta_head(paired).squeeze(-1)
        match_score = (transport * similarity).sum(dim=(-2, -1))
        entropy = -(
            transport * transport.clamp_min(1e-8).log()
        ).sum(dim=(-2, -1)) / math.log(float(transport.shape[-1] ** 2))
        uncertainty = 1.0 - 2.0 * (
            torch.sigmoid(logits_base.detach().float()) - 0.5
        ).abs()
        gate_stats = torch.stack(
            [
                match_score.to(dtype=paired.dtype),
                (1.0 - entropy).to(dtype=paired.dtype),
                uncertainty.to(dtype=paired.dtype),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(
            self.gate_head(torch.cat([paired, gate_stats], dim=-1)).squeeze(-1)
        )
        gamma = self.gamma_max * torch.sigmoid(self.gamma_raw)
        correction_candidate = gamma * gate * torch.tanh(raw_delta)
        ramp = self.ramp(correction_candidate)
        return {
            "mode": self.mode,
            "correction": ramp * correction_candidate,
            "correction_candidate": correction_candidate,
            "aux_logits": aux_logits,
            "gate": gate,
            "confidence": 0.5 * (confidence_a + confidence_b),
            "match_cost": 1.0 - match_score,
            "transport_entropy": entropy,
            "gamma": gamma,
            "ramp": ramp,
        }


class P12BERFHead(ScheduledInnovationHead):
    """Class-wise Beta evidence reliability filtering."""

    mode = "p12_berf"

    def __init__(
        self,
        level_dims: Dict[str, int],
        num_classes: int,
        level: str = "C5",
        projection_dim: int = 64,
        dropout: float = 0.1,
        gamma_init: float = 0.005,
        gamma_max: float = 0.05,
        warmup_epochs: int = 10,
        ramp_epochs: int = 10,
    ):
        super().__init__(warmup_epochs, ramp_epochs)
        if level not in level_dims:
            raise ValueError(f"P12 requires feature level {level}")
        self.level = level
        self.num_classes = int(num_classes)
        self.gamma_max = max(float(gamma_max), 1e-8)
        hidden = max(int(projection_dim), 16)
        self.evidence_head = nn.Sequential(
            nn.LayerNorm(int(level_dims[level])),
            nn.Linear(int(level_dims[level]), hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 2 * self.num_classes),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, _probability_logit(0.02))
        self.gamma_raw = _bounded_parameter(gamma_init, self.gamma_max)

    def _evidence(self, features):
        vector = features[self.level].mean(dim=(-2, -1))
        raw = self.evidence_head(vector).reshape(
            vector.shape[0], self.num_classes, 2
        )
        return F.softplus(raw) + 1.0

    def forward(self, features_a, features_b, logits_base, **_):
        beta_a = self._evidence(features_a)
        beta_b = self._evidence(features_b)
        probability_a = beta_a[..., 0] / beta_a.sum(dim=-1)
        probability_b = beta_b[..., 0] / beta_b.sum(dim=-1)
        uncertainty_a = (2.0 / beta_a.sum(dim=-1)).clamp(0.0, 1.0)
        uncertainty_b = (2.0 / beta_b.sum(dim=-1)).clamp(0.0, 1.0)
        reliability_a = 1.0 - uncertainty_a
        reliability_b = 1.0 - uncertainty_b
        normalization = (reliability_a + reliability_b).clamp_min(1e-6)
        weight_a = reliability_a / normalization
        weight_b = reliability_b / normalization
        logits_a = torch.logit(probability_a.clamp(1e-5, 1.0 - 1e-5))
        logits_b = torch.logit(probability_b.clamp(1e-5, 1.0 - 1e-5))
        fused_logits = weight_a * logits_a + weight_b * logits_b
        conflict = (probability_a - probability_b).abs()
        base_uncertainty = 1.0 - 2.0 * (
            torch.sigmoid(logits_base.detach().float()) - 0.5
        ).abs()
        gate_stats = torch.stack(
            [
                reliability_a,
                reliability_b,
                conflict,
                base_uncertainty.to(dtype=conflict.dtype),
                (fused_logits - logits_base.detach()).abs().tanh(),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_head(gate_stats).squeeze(-1))
        gamma = self.gamma_max * torch.sigmoid(self.gamma_raw)
        correction_candidate = gamma * gate * 4.0 * torch.tanh(
            (fused_logits - logits_base) / 4.0
        )
        ramp = self.ramp(correction_candidate)
        return {
            "mode": self.mode,
            "correction": ramp * correction_candidate,
            "correction_candidate": correction_candidate,
            "aux_logits": fused_logits,
            "evidence_parameters": torch.stack([beta_a, beta_b], dim=2),
            "evidence_probabilities": torch.stack(
                [probability_a, probability_b], dim=-1
            ),
            "evidence_strength": torch.stack(
                [beta_a.sum(dim=-1), beta_b.sum(dim=-1)], dim=-1
            ),
            "uncertainty": 0.5 * (uncertainty_a + uncertainty_b),
            "view_weights": torch.stack([weight_a, weight_b], dim=-1),
            "gate": gate,
            "confidence": 1.0 - 0.5 * (uncertainty_a + uncertainty_b),
            "gamma": gamma,
            "ramp": ramp,
        }


class P13VDRMHead(ScheduledInnovationHead):
    """Training-only view-drop regret minimization branch."""

    mode = "p13_vdrm"

    def __init__(self):
        super().__init__(0, 0)

    def forward(
        self,
        features_a,
        features_b,
        logits_base,
        logits_a=None,
        logits_b=None,
    ):
        if logits_a is None or logits_b is None:
            raise ValueError("P13 requires counterfactual single-view logits")
        zero = logits_base * 0.0
        agreement = 1.0 - (
            torch.sigmoid(logits_a) - torch.sigmoid(logits_b)
        ).abs()
        return {
            "mode": self.mode,
            "correction": zero,
            "correction_candidate": zero,
            "aux_logits": 0.5 * (logits_a + logits_b),
            "single_logits_a": logits_a,
            "single_logits_b": logits_b,
            "consistency": 1.0 - agreement,
            "gate": zero,
            "confidence": agreement,
            "gamma": zero.new_tensor(0.0),
            "ramp": zero.new_tensor(1.0),
        }


class P14HCAERHead(ScheduledInnovationHead):
    """Data-driven hard-class auxiliary-view expert rescue."""

    mode = "p14_hcaer"

    def __init__(
        self,
        level_dims: Dict[str, int],
        num_classes: int,
        levels=("C4", "C5"),
        projection_dim: int = 64,
        topk: int = 8,
        temperature: float = 0.2,
        dropout: float = 0.1,
        gamma_init: float = 0.005,
        gamma_max: float = 0.08,
        warmup_epochs: int = 10,
        ramp_epochs: int = 10,
    ):
        super().__init__(warmup_epochs, ramp_epochs)
        self.gamma_max = max(float(gamma_max), 1e-8)
        self.encoder = MultiScaleClassRegionEncoder(
            level_dims,
            levels,
            num_classes,
            projection_dim,
            topk,
            temperature,
        )
        dim = self.encoder.projection_dim
        pair_dim = 4 * dim
        self.expert = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, dim), nn.GELU(),
            nn.Dropout(float(dropout)), nn.Linear(dim, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(pair_dim + 4),
            nn.Linear(pair_dim + 4, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, _probability_logit(0.02))
        self.gamma_raw = _bounded_parameter(gamma_init, self.gamma_max)

    def forward(self, features_a, features_b, logits_base, **_):
        region_a, confidence_a = self.encoder(features_a)
        region_b, confidence_b = self.encoder(features_b)
        paired = _pair_features(region_a, region_b)
        expert_logits = self.expert(paired).squeeze(-1)
        base_probability = torch.sigmoid(logits_base.detach().float())
        uncertainty = 1.0 - 2.0 * (base_probability - 0.5).abs()
        stats = torch.stack(
            [
                confidence_a,
                confidence_b,
                uncertainty.to(dtype=confidence_a.dtype),
                (expert_logits - logits_base.detach()).abs().tanh(),
            ],
            dim=-1,
        )
        gate_raw = self.gate_head(torch.cat([paired, stats], dim=-1)).squeeze(-1)
        gate = torch.sigmoid(gate_raw)
        gamma = self.gamma_max * torch.sigmoid(self.gamma_raw)
        correction_candidate = gamma * gate * 4.0 * torch.tanh(
            (expert_logits - logits_base) / 4.0
        )
        ramp = self.ramp(correction_candidate)
        router_logits = torch.stack(
            [torch.zeros_like(gate_raw), gate_raw], dim=-1
        )
        return {
            "mode": self.mode,
            "correction": ramp * correction_candidate,
            "correction_candidate": correction_candidate,
            "aux_logits": expert_logits,
            "expert_logits": torch.stack(
                [logits_base, expert_logits], dim=-1
            ),
            "router_logits": router_logits,
            "router_weights": router_logits.softmax(dim=-1),
            "gate": gate,
            "confidence": 0.5 * (confidence_a + confidence_b),
            "gamma": gamma,
            "ramp": ramp,
        }


class P15VTRHead(ScheduledInnovationHead):
    """Symmetric second-view visual-token reasoner."""

    mode = "p15_vtr"

    def __init__(
        self,
        level_dims: Dict[str, int],
        num_classes: int,
        level: str = "C4",
        projection_dim: int = 64,
        dropout: float = 0.1,
        gamma_init: float = 0.005,
        gamma_max: float = 0.05,
        warmup_epochs: int = 10,
        ramp_epochs: int = 10,
    ):
        super().__init__(warmup_epochs, ramp_epochs)
        if level not in level_dims:
            raise ValueError(f"P15 requires feature level {level}")
        self.level = level
        self.num_classes = int(num_classes)
        self.dim = max(int(projection_dim), 16)
        self.gamma_max = max(float(gamma_max), 1e-8)
        self.projection = nn.Sequential(
            nn.Conv2d(int(level_dims[level]), self.dim, 1, bias=False),
            nn.GroupNorm(1, self.dim),
            nn.GELU(),
        )
        heads = 4 if self.dim % 4 == 0 else 1
        self.observe = nn.MultiheadAttention(
            self.dim, heads, dropout=float(dropout), batch_first=True
        )
        self.reason = nn.MultiheadAttention(
            self.dim, heads, dropout=float(dropout), batch_first=True
        )
        self.query_norm = nn.LayerNorm(self.dim)
        self.reason_norm = nn.LayerNorm(self.dim)
        self.class_queries = nn.Parameter(
            torch.empty(self.num_classes, self.dim)
        )
        nn.init.trunc_normal_(self.class_queries, std=0.02)
        self.aux_head = nn.Linear(self.dim, 1)
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.delta_head = nn.Linear(self.dim, 1)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.dim + 2),
            nn.Linear(self.dim + 2, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, 1),
        )
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, _probability_logit(0.02))
        self.gamma_raw = _bounded_parameter(gamma_init, self.gamma_max)

    def _tokens(self, features):
        return self.projection(features[self.level]).flatten(2).transpose(1, 2)

    def _reason_direction(self, queries, source, context):
        observed, _ = self.observe(queries, source, source, need_weights=False)
        observed = self.query_norm(queries + observed)
        reasoned, _ = self.reason(observed, context, context, need_weights=False)
        return self.reason_norm(observed + reasoned)

    def forward(self, features_a, features_b, logits_base, **_):
        tokens_a = self._tokens(features_a)
        tokens_b = self._tokens(features_b)
        queries = self.class_queries.unsqueeze(0).expand(tokens_a.shape[0], -1, -1)
        reason_ab = self._reason_direction(queries, tokens_a, tokens_b)
        reason_ba = self._reason_direction(queries, tokens_b, tokens_a)
        reasoned = 0.5 * (reason_ab + reason_ba)
        aux_logits = self.aux_head(reasoned).squeeze(-1)
        aux_logits = aux_logits + self.class_bias.unsqueeze(0)
        agreement = F.cosine_similarity(reason_ab, reason_ba, dim=-1)
        uncertainty = 1.0 - 2.0 * (
            torch.sigmoid(logits_base.detach().float()) - 0.5
        ).abs()
        stats = torch.stack(
            [agreement, uncertainty.to(dtype=agreement.dtype)], dim=-1
        )
        gate = torch.sigmoid(
            self.gate_head(torch.cat([reasoned, stats], dim=-1)).squeeze(-1)
        )
        raw_delta = self.delta_head(reasoned).squeeze(-1)
        gamma = self.gamma_max * torch.sigmoid(self.gamma_raw)
        correction_candidate = gamma * gate * torch.tanh(raw_delta)
        ramp = self.ramp(correction_candidate)
        return {
            "mode": self.mode,
            "correction": ramp * correction_candidate,
            "correction_candidate": correction_candidate,
            "aux_logits": aux_logits,
            "gate": gate,
            "confidence": 0.5 * (agreement + 1.0),
            "consistency": 1.0 - agreement,
            "gamma": gamma,
            "ramp": ramp,
        }


PLAIN_BCE_INNOVATION_HEADS = {
    "p9_caprs": P9CAPRSHead,
    "p10_wgcr": P10WGCRHead,
    "p11_otcvr": P11OTCVRHead,
    "p12_berf": P12BERFHead,
    "p13_vdrm": P13VDRMHead,
    "p14_hcaer": P14HCAERHead,
    "p15_vtr": P15VTRHead,
}
