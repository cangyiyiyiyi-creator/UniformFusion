from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewVisualEvidenceRouter(nn.Module):
    """Class-conditioned visual evidence routing over paired feature maps.

    The module predicts a bounded logit correction. Its final projection is
    zero-initialized, so enabling the branch starts exactly from the R2 output.
    """

    MODES = ("pg_cver", "sa_dca", "ca_rer", "cycle_cver")

    def __init__(
        self,
        channels: int,
        num_classes: int,
        mode: str = "pg_cver",
        projection_dim: int = 64,
        topk_ratio: float = 0.25,
        temperature: float = 0.2,
        shared_axis: str = "width",
        axis_radius: int = 1,
        gate_init: float = 0.05,
        gamma_init: float = 0.05,
        gamma_max: float = 0.25,
        reject_temperature: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(f"visual route mode must be one of {self.MODES}, got {mode!r}")
        if int(num_classes) <= 0:
            raise ValueError("visual evidence routing requires num_classes > 0")
        if not 0.0 < float(topk_ratio) <= 1.0:
            raise ValueError("visual route topk_ratio must be in (0, 1]")
        if float(temperature) <= 0.0:
            raise ValueError("visual route temperature must be positive")
        if str(shared_axis).lower() not in ("width", "height"):
            raise ValueError("visual route shared_axis must be width or height")
        if int(axis_radius) < 0:
            raise ValueError("visual route axis_radius must be non-negative")

        self.mode = mode
        self.num_classes = int(num_classes)
        self.projection_dim = max(16, int(projection_dim))
        self.topk_ratio = float(topk_ratio)
        self.temperature = float(temperature)
        self.shared_axis = str(shared_axis).lower()
        self.axis_radius = int(axis_radius)
        self.gamma_max = max(float(gamma_max), 1e-6)
        self.reject_temperature = max(float(reject_temperature), 1e-4)

        self.projection = nn.Sequential(
            nn.Conv2d(int(channels), self.projection_dim, 1, bias=False),
            nn.GroupNorm(1, self.projection_dim),
            nn.GELU(),
        )
        self.class_queries = nn.Parameter(
            torch.empty(self.num_classes, self.projection_dim)
        )
        nn.init.trunc_normal_(self.class_queries, std=0.02)

        pair_dim = 4 * self.projection_dim
        self.view_selector = nn.Sequential(
            nn.LayerNorm(pair_dim + 5),
            nn.Linear(pair_dim + 5, self.projection_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.projection_dim, 1),
        )
        nn.init.zeros_(self.view_selector[-1].weight)
        nn.init.zeros_(self.view_selector[-1].bias)

        self.trust_router = nn.Sequential(
            nn.LayerNorm(pair_dim + 6),
            nn.Linear(pair_dim + 6, self.projection_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.projection_dim, 1),
        )
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.trust_router[-1].weight)
        nn.init.constant_(
            self.trust_router[-1].bias,
            math.log(gate_init / (1.0 - gate_init)),
        )

        self.correction_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, self.projection_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.projection_dim, 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

        self.aux_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.aux_bias = nn.Parameter(torch.zeros(self.num_classes))
        init_ratio = min(
            max(float(gamma_init) / self.gamma_max, 1e-4),
            1.0 - 1e-4,
        )
        self.gamma_raw = nn.Parameter(
            torch.tensor(math.log(init_ratio / (1.0 - init_ratio)))
        )
        self.reject_threshold = nn.Parameter(torch.tensor(0.0))

    def _class_attention(self, features: torch.Tensor):
        batch, _, height, width = features.shape
        tokens = F.normalize(features.flatten(2).transpose(1, 2), dim=-1)
        queries = F.normalize(self.class_queries, dim=-1)
        scores = torch.einsum("bnd,kd->bkn", tokens, queries)
        scores = scores.float() / self.temperature
        token_count = scores.shape[-1]
        keep = max(
            1,
            min(token_count, int(round(token_count * self.topk_ratio))),
        )
        if keep < token_count:
            values, indices = scores.topk(keep, dim=-1)
            selected = torch.full_like(scores, torch.finfo(scores.dtype).min)
            selected.scatter_(-1, indices, values)
            scores = selected
        attention = scores.softmax(dim=-1)
        if keep > 1:
            entropy = -(
                attention * attention.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(float(keep))
            confidence = (1.0 - entropy).clamp(0.0, 1.0)
        else:
            confidence = attention.new_ones(batch, self.num_classes)
        return (
            attention.to(dtype=features.dtype),
            confidence.to(dtype=features.dtype),
            (height, width),
        )

    @staticmethod
    def _pool_regions(features: torch.Tensor, attention: torch.Tensor):
        tokens = features.flatten(2).transpose(1, 2)
        return torch.einsum("bkn,bnd->bkd", attention, tokens)

    def _match_width(self, source: torch.Tensor, target: torch.Tensor):
        """Match target evidence while preserving source width coordinates."""
        _, _, _, width = source.shape
        source_n = F.normalize(source, dim=1)
        target_n = F.normalize(target, dim=1)
        matched_columns = []
        for column in range(width):
            left = max(0, column - self.axis_radius)
            right = min(width, column + self.axis_radius + 1)
            query = source_n[:, :, :, column].transpose(1, 2)
            keys = target_n[:, :, :, left:right].flatten(2).transpose(1, 2)
            values = target[:, :, :, left:right].flatten(2).transpose(1, 2)
            scores = torch.bmm(query, keys.transpose(1, 2)) / self.temperature
            attention = scores.float().softmax(dim=-1).to(dtype=values.dtype)
            matched = torch.bmm(attention, values).transpose(1, 2)
            matched_columns.append(matched)
        return torch.stack(matched_columns, dim=-1)

    def _shared_axis_match(self, source: torch.Tensor, target: torch.Tensor):
        if self.shared_axis == "width":
            return self._match_width(source, target)
        source_t = source.transpose(-1, -2)
        target_t = target.transpose(-1, -2)
        return self._match_width(source_t, target_t).transpose(-1, -2)

    def _axis_descriptors(self, features: torch.Tensor):
        if self.shared_axis == "width":
            return features.mean(dim=2).transpose(1, 2)
        return features.mean(dim=3).transpose(1, 2)

    def _axis_distribution(
        self,
        attention: torch.Tensor,
        spatial_shape,
    ):
        height, width = spatial_shape
        maps = attention.reshape(
            attention.shape[0], self.num_classes, height, width
        )
        distribution = maps.sum(dim=2 if self.shared_axis == "width" else 3)
        return distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _axis_transition(self, source: torch.Tensor, target: torch.Tensor):
        source_desc = F.normalize(self._axis_descriptors(source), dim=-1)
        target_desc = F.normalize(self._axis_descriptors(target), dim=-1)
        scores = torch.bmm(source_desc, target_desc.transpose(1, 2))
        scores = scores.float() / self.temperature
        positions = torch.arange(scores.shape[-1], device=scores.device)
        allowed = (
            positions[:, None] - positions[None, :]
        ).abs() <= self.axis_radius
        scores = scores.masked_fill(~allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
        return scores.softmax(dim=-1).to(dtype=source.dtype)

    def _cycle_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        attention_a: torch.Tensor,
        attention_b: torch.Tensor,
        spatial_shape,
    ):
        transition_ab = self._axis_transition(features_a, features_b)
        transition_ba = self._axis_transition(features_b, features_a)
        distribution_a = self._axis_distribution(attention_a, spatial_shape)
        distribution_b = self._axis_distribution(attention_b, spatial_shape)
        predicted_b = torch.bmm(distribution_a, transition_ab)
        predicted_a = torch.bmm(distribution_b, transition_ba)
        cycle_a = torch.bmm(predicted_b, transition_ba)
        cycle_b = torch.bmm(predicted_a, transition_ab)
        direct = 0.5 * (
            F.l1_loss(predicted_b, distribution_b)
            + F.l1_loss(predicted_a, distribution_a)
        )
        cycle = 0.5 * (
            F.l1_loss(cycle_a, distribution_a)
            + F.l1_loss(cycle_b, distribution_b)
        )
        return direct + cycle

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        logits_base: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if features_a.shape != features_b.shape:
            raise ValueError("visual evidence routing expects same-shape dual-view features")
        projected_a = self.projection(features_a)
        projected_b = self.projection(features_b)

        use_axis_match = self.mode in ("sa_dca", "cycle_cver")
        if use_axis_match:
            matched_b = self._shared_axis_match(projected_a, projected_b)
            matched_a = self._shared_axis_match(projected_b, projected_a)
            routed_a = 0.5 * (projected_a + matched_b)
            routed_b = 0.5 * (projected_b + matched_a)
        else:
            routed_a, routed_b = projected_a, projected_b

        attention_a, confidence_a, spatial_shape = self._class_attention(routed_a)
        attention_b, confidence_b, _ = self._class_attention(routed_b)
        region_a = self._pool_regions(routed_a, attention_a)
        region_b = self._pool_regions(routed_b, attention_b)
        pair_features = torch.cat(
            [region_a, region_b, (region_a - region_b).abs(), region_a * region_b],
            dim=-1,
        )

        query = F.normalize(self.class_queries, dim=-1).unsqueeze(0)
        evidence_a = (F.normalize(region_a, dim=-1) * query).sum(dim=-1)
        evidence_b = (F.normalize(region_b, dim=-1) * query).sum(dim=-1)
        agreement = F.cosine_similarity(region_a, region_b, dim=-1).clamp(-1.0, 1.0)
        base_probability = torch.sigmoid(logits_base.detach().float())
        base_uncertainty = 1.0 - 2.0 * (base_probability - 0.5).abs()
        evidence_gap = (evidence_a - evidence_b).abs()

        selector_stats = torch.stack(
            [confidence_a, confidence_b, agreement, base_uncertainty, evidence_gap],
            dim=-1,
        ).to(dtype=pair_features.dtype)
        view_a_weight = torch.sigmoid(
            self.view_selector(torch.cat([pair_features, selector_stats], dim=-1))
        ).squeeze(-1)

        selected_evidence = (
            view_a_weight * evidence_a + (1.0 - view_a_weight) * evidence_b
        )
        aux_scale = F.softplus(self.aux_scale_raw) + 1.0
        aux_logits = aux_scale * selected_evidence + self.aux_bias.unsqueeze(0)

        conflict = 0.5 * (1.0 - agreement)
        trust_stats = torch.stack(
            [
                confidence_a,
                confidence_b,
                agreement,
                base_uncertainty,
                evidence_gap,
                selected_evidence.abs(),
            ],
            dim=-1,
        ).to(dtype=pair_features.dtype)
        trust = torch.sigmoid(
            self.trust_router(torch.cat([pair_features, trust_stats], dim=-1))
        ).squeeze(-1)
        rejection = trust.new_ones(trust.shape)
        if self.mode == "ca_rer":
            quality = torch.maximum(confidence_a, confidence_b)
            rejection = torch.sigmoid(
                (quality - conflict - self.reject_threshold)
                / self.reject_temperature
            )
            trust = trust * rejection

        raw_correction = self.correction_head(pair_features).squeeze(-1)
        gamma = self.gamma_max * torch.sigmoid(self.gamma_raw)
        correction = gamma * trust * torch.tanh(raw_correction)

        cycle_loss = correction.new_tensor(0.0)
        if self.mode == "cycle_cver":
            cycle_loss = self._cycle_loss(
                projected_a,
                projected_b,
                attention_a,
                attention_b,
                spatial_shape,
            )

        return {
            "correction": correction,
            "aux_logits": aux_logits,
            "cycle_loss": cycle_loss,
            "view_a_weight": view_a_weight,
            "trust": trust,
            "rejection": rejection,
            "agreement": agreement,
            "confidence": 0.5 * (confidence_a + confidence_b),
            "gamma": gamma,
        }


class FrozenAnchorRegionEvidenceHead(nn.Module):
    """C3/C4 class-region evidence adapter for a frozen paired-view anchor."""

    def __init__(
        self,
        level_dims: Dict[str, int],
        num_classes: int,
        levels=("C3", "C4"),
        projection_dim: int = 64,
        topk_ratio: float = 0.1,
        temperature: float = 0.2,
        gamma_init: float = 0.003,
        gamma_max: float = 0.02,
        uncertainty_threshold: float = 0.35,
        gate_temperature: float = 0.1,
        trust_init: float = 0.05,
    ):
        super().__init__()
        self.levels = tuple(level for level in levels if level in level_dims)
        if not self.levels:
            raise ValueError("M7 region evidence requires at least one valid level")
        if not 0.0 < float(topk_ratio) <= 1.0:
            raise ValueError("M7 topk_ratio must be in (0, 1]")
        self.num_classes = int(num_classes)
        self.projection_dim = max(16, int(projection_dim))
        self.topk_ratio = float(topk_ratio)
        self.temperature = max(float(temperature), 1e-6)
        self.gamma_max = max(float(gamma_max), 1e-8)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.gate_temperature = max(float(gate_temperature), 1e-6)
        self.projections = nn.ModuleDict({
            level: nn.Sequential(
                nn.Conv2d(int(level_dims[level]), self.projection_dim, 1, bias=False),
                nn.GroupNorm(1, self.projection_dim),
                nn.GELU(),
            )
            for level in self.levels
        })
        self.attention_queries = nn.ParameterDict()
        self.classifier_weights = nn.ParameterDict()
        self.classifier_bias = nn.ParameterDict()
        self.logit_scale_raw = nn.ParameterDict()
        for level in self.levels:
            self.attention_queries[level] = nn.Parameter(
                torch.empty(self.num_classes, self.projection_dim)
            )
            self.classifier_weights[level] = nn.Parameter(
                torch.empty(self.num_classes, self.projection_dim)
            )
            self.classifier_bias[level] = nn.Parameter(
                torch.zeros(self.num_classes)
            )
            self.logit_scale_raw[level] = nn.Parameter(torch.tensor(4.0))
            nn.init.trunc_normal_(self.attention_queries[level], std=0.02)
            nn.init.trunc_normal_(self.classifier_weights[level], std=0.02)

        trust_hidden = max(16, self.projection_dim // 2)
        self.trust_head = nn.Sequential(
            nn.Linear(7, trust_hidden), nn.GELU(), nn.Linear(trust_hidden, 1)
        )
        trust_init = min(max(float(trust_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.trust_head[-1].weight)
        nn.init.constant_(
            self.trust_head[-1].bias,
            math.log(trust_init / (1.0 - trust_init)),
        )
        gamma_ratio = min(max(float(gamma_init) / self.gamma_max, 1e-4), 1.0 - 1e-4)
        self.gamma_raw = nn.Parameter(
            torch.full(
                (self.num_classes,),
                math.log(gamma_ratio / (1.0 - gamma_ratio)),
            )
        )

    def _decode_level(self, features: torch.Tensor, level: str):
        projected = self.projections[level](features)
        tokens = projected.flatten(2).transpose(1, 2)
        tokens_n = F.normalize(tokens, dim=-1)
        queries = F.normalize(self.attention_queries[level], dim=-1)
        scores = torch.einsum("bnd,kd->bkn", tokens_n, queries) / self.temperature
        count = scores.shape[-1]
        keep = max(1, min(count, int(round(count * self.topk_ratio))))
        if keep < count:
            values, indices = scores.topk(keep, dim=-1)
            selected = torch.full_like(scores, torch.finfo(scores.dtype).min)
            selected.scatter_(-1, indices, values)
            scores = selected
        attention = scores.float().softmax(dim=-1).to(dtype=tokens.dtype)
        region = torch.einsum("bkn,bnd->bkd", attention, tokens)
        weights = F.normalize(self.classifier_weights[level], dim=-1)
        scale = F.softplus(self.logit_scale_raw[level]) + 1.0
        logits = scale * (
            F.normalize(region, dim=-1) * weights.unsqueeze(0)
        ).sum(dim=-1) + self.classifier_bias[level].unsqueeze(0)
        if keep > 1:
            entropy = -(
                attention.float() * attention.float().clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(float(keep))
            confidence = (1.0 - entropy).clamp(0.0, 1.0)
        else:
            confidence = logits.new_ones(logits.shape)
        return logits, confidence.to(dtype=logits.dtype)

    def _decode_view(self, feats: Dict[str, torch.Tensor]):
        decoded = [self._decode_level(feats[level], level) for level in self.levels]
        logits = torch.stack([item[0] for item in decoded], dim=0).mean(dim=0)
        confidence = torch.stack([item[1] for item in decoded], dim=0).mean(dim=0)
        return logits, confidence

    def forward(self, feats_a, feats_b, logits_base):
        logits_a, confidence_a = self._decode_view(feats_a)
        logits_b, confidence_b = self._decode_view(feats_b)
        pair = torch.stack((logits_a, logits_b), dim=1)
        fusion_temperature = 0.5
        candidate_logits = fusion_temperature * torch.logsumexp(
            pair / fusion_temperature, dim=1
        ) - fusion_temperature * math.log(2.0)
        base_ref = logits_base.detach()
        base_prob = torch.sigmoid(base_ref)
        prob_a = torch.sigmoid(logits_a)
        prob_b = torch.sigmoid(logits_b)
        candidate_prob = torch.sigmoid(candidate_logits)
        positive_delta = F.relu(candidate_logits - base_ref)
        uncertainty = 4.0 * base_prob * (1.0 - base_prob)
        region_confidence = torch.maximum(confidence_a, confidence_b)
        trust_features = torch.stack((
            base_prob, prob_a, prob_b, candidate_prob,
            (prob_a - prob_b).abs(), uncertainty, region_confidence,
        ), dim=-1).detach()
        trust_logits = self.trust_head(trust_features).squeeze(-1)
        trust_gate = torch.sigmoid(trust_logits)
        uncertainty_gate = torch.sigmoid(
            (uncertainty - self.uncertainty_threshold) / self.gate_temperature
        )
        complement_gate = torch.sigmoid(
            (positive_delta - 0.25) / self.gate_temperature
        )
        quality_gate = 0.5 + 0.5 * region_confidence
        gate = uncertainty_gate * complement_gate * trust_gate * quality_gate
        gamma = self.gamma_max * torch.sigmoid(self.gamma_raw)
        correction = gamma.unsqueeze(0) * gate * torch.tanh(positive_delta)
        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "candidate_logits": candidate_logits,
            "positive_delta": positive_delta,
            "trust_logits": trust_logits,
            "trust_gate": trust_gate,
            "region_confidence": region_confidence,
            "uncertainty": uncertainty,
            "gate": gate,
            "gamma": gamma,
            "correction": correction,
        }


class FrozenAnchorCounterfactualViewRouter(nn.Module):
    """Anchor-preserving router over paired, replicated-A, and replicated-B logits."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 32,
        class_embed_dim: int = 8,
        rho_init: float = 0.05,
        rho_max: float = 0.2,
        rescue_init: float = 0.1,
        delta_clip: float = 6.0,
        region_channels: int = 0,
        region_projection_dim: int = 32,
        region_temperature: float = 0.2,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = max(8, int(hidden_dim))
        self.class_embed_dim = max(0, int(class_embed_dim))
        self.rho_max = max(float(rho_max), 1e-6)
        self.delta_clip = max(float(delta_clip), 1e-3)
        self.region_projection_dim = max(16, int(region_projection_dim))
        self.region_temperature = max(float(region_temperature), 1e-6)

        self.region_projection = None
        self.region_queries = None
        if int(region_channels) > 0:
            self.region_projection = nn.Sequential(
                nn.Conv2d(
                    int(region_channels), self.region_projection_dim, 1,
                    bias=False,
                ),
                nn.GroupNorm(1, self.region_projection_dim),
                nn.GELU(),
            )
            self.region_queries = nn.Parameter(
                torch.empty(self.num_classes, self.region_projection_dim)
            )
            nn.init.trunc_normal_(self.region_queries, std=0.02)

        if self.class_embed_dim > 0:
            self.class_embedding = nn.Parameter(
                torch.empty(self.num_classes, self.class_embed_dim)
            )
            nn.init.trunc_normal_(self.class_embedding, std=0.02)
        else:
            self.register_parameter("class_embedding", None)

        region_feature_dim = 4 if self.region_projection is not None else 0
        feature_dim = 8 + region_feature_dim + self.class_embed_dim
        self.router = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 3),
        )
        rescue_init = min(max(float(rescue_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.router[-1].weight)
        with torch.no_grad():
            self.router[-1].bias.zero_()
            self.router[-1].bias[0] = math.log(
                rescue_init / (1.0 - rescue_init)
            )

        rho_ratio = min(
            max(float(rho_init) / self.rho_max, 1e-4),
            1.0 - 1e-4,
        )
        self.rho_raw = nn.Parameter(
            torch.full(
                (self.num_classes,),
                math.log(rho_ratio / (1.0 - rho_ratio)),
            )
        )

    def _regional_counterfactual_features(self, features_a, features_b):
        if self.region_projection is None:
            return None
        if features_a is None or features_b is None:
            raise ValueError(
                "regional M8 requires both selected-view feature maps"
            )
        if features_a.shape != features_b.shape:
            raise ValueError("regional M8 expects same-shape dual-view features")

        projected_a = self.region_projection(features_a)
        projected_b = self.region_projection(features_b)
        tokens_a = projected_a.flatten(2).transpose(1, 2)
        tokens_b = projected_b.flatten(2).transpose(1, 2)
        queries = F.normalize(self.region_queries, dim=-1)
        scores_a = torch.einsum(
            "bnd,kd->bkn", F.normalize(tokens_a, dim=-1), queries
        ) / self.region_temperature
        scores_b = torch.einsum(
            "bnd,kd->bkn", F.normalize(tokens_b, dim=-1), queries
        ) / self.region_temperature
        attention_a = scores_a.float().softmax(dim=-1).to(tokens_a.dtype)
        attention_b = scores_b.float().softmax(dim=-1).to(tokens_b.dtype)
        region_a = torch.einsum("bkn,bnd->bkd", attention_a, tokens_a)
        region_b = torch.einsum("bkn,bnd->bkd", attention_b, tokens_b)

        evidence_a = (
            F.normalize(region_a, dim=-1) * queries.unsqueeze(0)
        ).sum(dim=-1)
        evidence_b = (
            F.normalize(region_b, dim=-1) * queries.unsqueeze(0)
        ).sum(dim=-1)
        agreement = F.cosine_similarity(region_a, region_b, dim=-1)
        evidence_gap = (evidence_a - evidence_b).abs()
        norm_a = region_a.float().norm(dim=-1)
        norm_b = region_b.float().norm(dim=-1)
        unique_ratio = (region_a - region_b).float().norm(dim=-1) / (
            norm_a + norm_b
        ).clamp_min(1e-6)

        token_count = max(attention_a.shape[-1], 2)
        entropy_scale = math.log(float(token_count))
        confidence_a = 1.0 - (
            -attention_a.float()
            * attention_a.float().clamp_min(1e-8).log()
        ).sum(dim=-1) / entropy_scale
        confidence_b = 1.0 - (
            -attention_b.float()
            * attention_b.float().clamp_min(1e-8).log()
        ).sum(dim=-1) / entropy_scale
        confidence = 0.5 * (confidence_a + confidence_b)
        return torch.stack((
            agreement,
            evidence_gap,
            unique_ratio,
            confidence,
        ), dim=-1).to(dtype=features_a.dtype)

    def forward(
        self,
        paired_logits,
        logits_a,
        logits_b,
        features_a=None,
        features_b=None,
    ):
        paired = paired_logits.detach()
        view_a = logits_a.detach()
        view_b = logits_b.detach()
        delta_a = (view_a - paired).clamp(-self.delta_clip, self.delta_clip)
        delta_b = (view_b - paired).clamp(-self.delta_clip, self.delta_clip)

        feature_scale = 4.0
        features = torch.stack((
            torch.tanh(paired / feature_scale),
            torch.tanh(view_a / feature_scale),
            torch.tanh(view_b / feature_scale),
            torch.tanh(delta_a / feature_scale),
            torch.tanh(delta_b / feature_scale),
            torch.tanh((view_a - view_b).abs() / feature_scale),
            (2.0 * torch.sigmoid(paired) - 1.0).abs(),
            (torch.sigmoid(view_a) - torch.sigmoid(view_b)).abs(),
        ), dim=-1)
        region_features = self._regional_counterfactual_features(
            features_a, features_b
        )
        if region_features is not None:
            features = torch.cat((features, region_features), dim=-1)
        if self.class_embedding is not None:
            embedding = self.class_embedding.unsqueeze(0).expand(
                paired.shape[0], -1, -1
            )
            features = torch.cat((features, embedding), dim=-1)

        route_output = self.router(features)
        rescue_logits = route_output[..., 0]
        view_logits = route_output[..., 1:]
        rescue_gate = torch.sigmoid(rescue_logits)
        view_weights = torch.softmax(view_logits, dim=-1)
        class_rho = self.rho_max * torch.sigmoid(self.rho_raw)
        rho = rescue_gate * class_rho.unsqueeze(0)

        counterfactual = (
            view_weights[..., 0] * view_a
            + view_weights[..., 1] * view_b
        )
        correction = rho * (counterfactual - paired).clamp(
            -self.delta_clip, self.delta_clip
        )
        routed_logits = paired + correction
        return {
            "paired_logits": paired,
            "logits_a": view_a,
            "logits_b": view_b,
            "rescue_logits": rescue_logits,
            "view_logits": view_logits,
            "rescue_gate": rescue_gate,
            "view_weights": view_weights,
            "class_rho": class_rho,
            "rho": rho,
            "counterfactual_logits": counterfactual,
            "correction": correction,
            "routed_logits": routed_logits,
        }


class FrozenAnchorRegionInteractionMoE(nn.Module):
    """Region interaction experts that preserve a frozen paired-view anchor.

    The branch creates four feature-space candidates: evidence unique to each
    view, redundant evidence shared by both views, and cross-view synergy.  The
    owning model evaluates every candidate with the original frozen neck and
    classifier before this module performs class-conditional routing.
    """

    EXPERT_NAMES = ("unique_a", "unique_b", "redundant", "synergy")

    def __init__(
        self,
        level_dims: Dict[str, int],
        num_classes: int,
        levels=("C3", "C4"),
        projection_dim: int = 64,
        temperature: float = 0.2,
        shared_axis: str = "width",
        axis_radius: int = 1,
        residual_init: float = 0.03,
        residual_max: float = 0.20,
        router_hidden_dim: int = 128,
        class_embed_dim: int = 16,
        rho_init: float = 0.08,
        rho_max: float = 0.25,
        rescue_init: float = 0.05,
        delta_clip: float = 6.0,
        include_counterfactual_candidates: bool = False,
    ):
        super().__init__()
        self.levels = tuple(level for level in levels if level in level_dims)
        if not self.levels:
            raise ValueError("M9 requires at least one valid feature level")
        if str(shared_axis).lower() not in ("width", "height"):
            raise ValueError("M9 shared_axis must be width or height")
        if int(axis_radius) < 0:
            raise ValueError("M9 axis_radius must be non-negative")

        self.num_classes = int(num_classes)
        self.projection_dim = max(16, int(projection_dim))
        self.temperature = max(float(temperature), 1e-6)
        self.shared_axis = str(shared_axis).lower()
        self.axis_radius = int(axis_radius)
        self.residual_max = max(float(residual_max), 1e-6)
        self.rho_max = max(float(rho_max), 1e-6)
        self.delta_clip = max(float(delta_clip), 1e-3)
        self.class_embed_dim = max(0, int(class_embed_dim))
        self.include_counterfactual_candidates = bool(
            include_counterfactual_candidates
        )
        self.route_candidate_names = self.EXPERT_NAMES
        if self.include_counterfactual_candidates:
            self.route_candidate_names = self.route_candidate_names + (
                "replicated_a", "replicated_b"
            )

        self.projections = nn.ModuleDict()
        self.synergy_mixers = nn.ModuleDict()
        self.region_gates = nn.ModuleDict()
        self.residual_decoders = nn.ModuleDict()
        for level in self.levels:
            channels = int(level_dims[level])
            self.projections[level] = nn.Sequential(
                nn.Conv2d(channels, self.projection_dim, 1, bias=False),
                nn.GroupNorm(1, self.projection_dim),
                nn.GELU(),
            )
            self.synergy_mixers[level] = nn.Sequential(
                nn.Conv2d(4 * self.projection_dim, self.projection_dim, 1),
                nn.GroupNorm(1, self.projection_dim),
                nn.GELU(),
                nn.Conv2d(
                    self.projection_dim,
                    self.projection_dim,
                    3,
                    padding=1,
                    groups=self.projection_dim,
                    bias=False,
                ),
                nn.GELU(),
            )
            self.region_gates[level] = nn.ModuleDict()
            self.residual_decoders[level] = nn.ModuleDict()
            for expert in self.EXPERT_NAMES:
                gate = nn.Conv2d(self.projection_dim, 1, 1)
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
                self.region_gates[level][expert] = gate
                decoder = nn.Conv2d(
                    self.projection_dim, channels, 1, bias=False
                )
                nn.init.trunc_normal_(decoder.weight, std=0.02)
                self.residual_decoders[level][expert] = decoder

        residual_ratio = min(
            max(float(residual_init) / self.residual_max, 1e-4),
            1.0 - 1e-4,
        )
        self.residual_scale_raw = nn.Parameter(
            torch.full(
                (len(self.levels), len(self.EXPERT_NAMES)),
                math.log(residual_ratio / (1.0 - residual_ratio)),
            )
        )

        if self.class_embed_dim > 0:
            self.class_embedding = nn.Parameter(
                torch.empty(self.num_classes, self.class_embed_dim)
            )
            nn.init.trunc_normal_(self.class_embedding, std=0.02)
        else:
            self.register_parameter("class_embedding", None)

        candidate_count = len(self.route_candidate_names)
        router_feature_dim = 4 + 2 * candidate_count + self.class_embed_dim
        router_hidden_dim = max(16, int(router_hidden_dim))
        self.router = nn.Sequential(
            nn.LayerNorm(router_feature_dim),
            nn.Linear(router_feature_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, 1 + candidate_count),
        )
        rescue_init = min(max(float(rescue_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.router[-1].weight)
        with torch.no_grad():
            self.router[-1].bias.zero_()
            self.router[-1].bias[0] = math.log(
                rescue_init / (1.0 - rescue_init)
            )

        rho_ratio = min(
            max(float(rho_init) / self.rho_max, 1e-4),
            1.0 - 1e-4,
        )
        self.rho_raw = nn.Parameter(
            torch.full(
                (self.num_classes,),
                math.log(rho_ratio / (1.0 - rho_ratio)),
            )
        )

    def _match_width(self, source: torch.Tensor, target: torch.Tensor):
        _, _, _, width = source.shape
        source_n = F.normalize(source, dim=1)
        target_n = F.normalize(target, dim=1)
        columns = []
        for column in range(width):
            left = max(0, column - self.axis_radius)
            right = min(width, column + self.axis_radius + 1)
            query = source_n[:, :, :, column].transpose(1, 2)
            keys = target_n[:, :, :, left:right].flatten(2).transpose(1, 2)
            values = target[:, :, :, left:right].flatten(2).transpose(1, 2)
            scores = torch.bmm(query, keys.transpose(1, 2)) / self.temperature
            attention = scores.float().softmax(dim=-1).to(values.dtype)
            columns.append(torch.bmm(attention, values).transpose(1, 2))
        return torch.stack(columns, dim=-1)

    def _shared_axis_match(self, source: torch.Tensor, target: torch.Tensor):
        if self.shared_axis == "width":
            return self._match_width(source, target)
        return self._match_width(
            source.transpose(-1, -2), target.transpose(-1, -2)
        ).transpose(-1, -2)

    def build_expert_residuals(self, feats_a, feats_b):
        residuals = {expert: {} for expert in self.EXPERT_NAMES}
        gate_values = []
        residual_values = []
        alignment_terms = []
        diversity_terms = []
        scales = self.residual_max * torch.sigmoid(self.residual_scale_raw)

        for level_index, level in enumerate(self.levels):
            projected_a = self.projections[level](feats_a[level])
            projected_b = self.projections[level](feats_b[level])
            matched_b = self._shared_axis_match(projected_a, projected_b)
            matched_a = self._shared_axis_match(projected_b, projected_a)

            common_a = 0.5 * (projected_a + matched_b)
            common_b = 0.5 * (projected_b + matched_a)
            redundant = 0.5 * (common_a + common_b)
            unique_a = 0.5 * (projected_a - matched_b)
            unique_b = 0.5 * (projected_b - matched_a)
            synergy = self.synergy_mixers[level](torch.cat((
                projected_a,
                projected_b,
                matched_a,
                matched_b,
            ), dim=1))
            latents = {
                "unique_a": unique_a,
                "unique_b": unique_b,
                "redundant": redundant,
                "synergy": synergy,
            }

            pooled_common_a = F.adaptive_avg_pool2d(common_a, 1).flatten(1)
            pooled_common_b = F.adaptive_avg_pool2d(common_b, 1).flatten(1)
            alignment_terms.append(
                1.0 - F.cosine_similarity(
                    pooled_common_a, pooled_common_b, dim=-1
                ).mean()
            )

            level_vectors = []
            for expert_index, expert in enumerate(self.EXPERT_NAMES):
                latent = latents[expert]
                gate = torch.sigmoid(self.region_gates[level][expert](latent))
                decoded = self.residual_decoders[level][expert](latent)
                residual = (
                    scales[level_index, expert_index]
                    * (2.0 * gate)
                    * decoded
                )
                residuals[expert][level] = residual
                gate_values.append(gate.mean())
                residual_values.append(residual.float().pow(2).mean().sqrt())
                level_vectors.append(
                    F.adaptive_avg_pool2d(residual, 1).flatten(1)
                )

            normalized = [F.normalize(vector, dim=-1) for vector in level_vectors]
            for first in range(len(normalized)):
                for second in range(first + 1, len(normalized)):
                    diversity_terms.append(
                        (normalized[first] * normalized[second])
                        .sum(dim=-1)
                        .pow(2)
                        .mean()
                    )

        zero = next(iter(residuals[self.EXPERT_NAMES[0]].values())).sum() * 0.0
        alignment_loss = (
            torch.stack(alignment_terms).mean() if alignment_terms else zero
        )
        diversity_loss = (
            torch.stack(diversity_terms).mean() if diversity_terms else zero
        )
        return {
            "residuals": residuals,
            "alignment_loss": alignment_loss,
            "diversity_loss": diversity_loss,
            "region_gate": torch.stack(gate_values).mean(),
            "residual_norm": torch.stack(residual_values).mean(),
            "residual_scale": scales.mean(),
        }

    def route(
        self,
        anchor_logits,
        expert_logits,
        counterfactual_logits=None,
    ):
        if expert_logits.ndim != 3:
            raise ValueError("M9 expert logits must have shape [B,E,C]")
        if expert_logits.shape[1] != len(self.EXPERT_NAMES):
            raise ValueError("M9 received an unexpected number of experts")

        route_candidates = expert_logits
        if self.include_counterfactual_candidates:
            if counterfactual_logits is None:
                raise ValueError(
                    "unified M2+M8+M9 requires A+A and B+B logits"
                )
            expected = (
                expert_logits.shape[0], 2, expert_logits.shape[2]
            )
            if tuple(counterfactual_logits.shape) != expected:
                raise ValueError(
                    "counterfactual logits must have shape [B,2,C]"
                )
            route_candidates = torch.cat((
                expert_logits, counterfactual_logits
            ), dim=1)
        elif counterfactual_logits is not None:
            raise ValueError(
                "counterfactual logits were supplied to standard M9"
            )

        anchor = anchor_logits.detach().float()
        experts = route_candidates.detach().float()
        experts_by_class = experts.permute(0, 2, 1)
        deltas = (experts_by_class - anchor.unsqueeze(-1)).clamp(
            -self.delta_clip, self.delta_clip
        )
        feature_scale = 4.0
        features = torch.cat((
            torch.tanh(anchor / feature_scale).unsqueeze(-1),
            torch.tanh(experts_by_class / feature_scale),
            torch.tanh(deltas / feature_scale),
            experts_by_class.std(dim=-1, unbiased=False).unsqueeze(-1),
            (2.0 * torch.sigmoid(anchor) - 1.0).abs().unsqueeze(-1),
            deltas.abs().amax(dim=-1, keepdim=True),
        ), dim=-1)
        if self.class_embedding is not None:
            embedding = self.class_embedding.unsqueeze(0).expand(
                anchor.shape[0], -1, -1
            )
            features = torch.cat((features, embedding), dim=-1)

        route_output = self.router(features)
        rescue_logits = route_output[..., 0]
        expert_weight_logits = route_output[..., 1:]
        rescue_gate = torch.sigmoid(rescue_logits)
        expert_weights = torch.softmax(expert_weight_logits, dim=-1)
        class_rho = self.rho_max * torch.sigmoid(self.rho_raw)
        rho = rescue_gate * class_rho.unsqueeze(0)
        candidate = (expert_weights * experts_by_class).sum(dim=-1)
        correction = rho * (candidate - anchor).clamp(
            -self.delta_clip, self.delta_clip
        )
        return {
            "anchor_logits": anchor,
            # Keep the raw tensor for Stage-A expert supervision. Routing uses
            # the detached copy above, so Stage B cannot alter frozen experts.
            "expert_logits": expert_logits,
            "route_candidate_logits": route_candidates,
            "route_candidate_names": self.route_candidate_names,
            "rescue_logits": rescue_logits,
            "expert_weight_logits": expert_weight_logits,
            "rescue_gate": rescue_gate,
            "expert_weights": expert_weights,
            "class_rho": class_rho,
            "rho": rho,
            "candidate_logits": candidate,
            "correction": correction,
            "routed_logits": anchor + correction,
        }

    def set_router_only(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.router.parameters():
            parameter.requires_grad = True
        if self.class_embedding is not None:
            self.class_embedding.requires_grad = True
        self.rho_raw.requires_grad = True
