from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# --- import all "parts" from the new module files ---
from .modules.common import Conv1x1
from .modules.fusions import (
    GatedFuse,
    XAttnFuse,
    AHCRFuse,
    ClasswiseRegionComplementFuse,
    InterventionStableFuse,
)
from .modules.necks import FPN, FPN_PAN
from .modules.cross_view_consistency import CrossViewGeoSemanticAlign
from .modules.visual_evidence import (
    CrossViewVisualEvidenceRouter,
    FrozenAnchorCounterfactualViewRouter,
    FrozenAnchorRegionInteractionMoE,
    FrozenAnchorRegionEvidenceHead,
)
from .modules.plain_bce_innovations import (
    P9CAPRSHead,
    P10WGCRHead,
    P11OTCVRHead,
    P12BERFHead,
    P13VDRMHead,
    P14HCAERHead,
    P15VTRHead,
)
from .modules.plain_bce_p16 import P16FACGRHead
from .modules.plain_bce_p17 import P17DCASRHead
from .modules.plain_bce_p18 import P18EWSARHead
from .modules.plain_bce_p19 import P19APCERHead
from .modules.plain_bce_p20 import P20CVCRHead



# ------------------------------
# Visual-Semantic Auxiliary Head
# ------------------------------
class SemanticTrustRouter(nn.Module):
    """Class-shared trust predictor built from backbone-independent evidence."""

    def __init__(
        self,
        hidden_dim: int = 16,
        init_prob: float = 0.05,
        num_classes: int = 0,
        classwise: bool = False,
    ):
        super().__init__()
        self.classwise = bool(classwise)
        hidden_dim = max(4, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        init_prob = min(max(float(init_prob), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(init_prob / (1.0 - init_prob)))
        self.class_bias = None
        if self.classwise:
            if int(num_classes) <= 0:
                raise ValueError("classwise trust routing requires num_classes > 0")
            self.class_bias = nn.Parameter(torch.zeros(int(num_classes)))

    def forward_logits(self, evidence: torch.Tensor) -> torch.Tensor:
        # evidence: [batch, classes, 5]
        logits = self.net(evidence).squeeze(-1)
        if self.class_bias is not None:
            logits = logits + self.class_bias.unsqueeze(0)
        return logits

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(evidence))


class SemanticClassEmbeddingHead(nn.Module):
    """
    Lightweight vision-semantic auxiliary branch:
    - keeps the original Linear classification head untouched;
    - maintains one learnable semantic token per class by default;
    - can initialise the semantic prompts from text embeddings of LLM class descriptions;
    - projects the global visual feature into the semantic space and takes the cosine similarity with the class tokens;
    - final logits = logits_base + semantic_correction;
    - with the optional gate, semantic_correction = gate(x) * gamma * logits_sem.
    - with the optional dual-view calibration, the A/B semantic consistency modulates semantic_correction.
    """
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        sem_dim: int = 256,
        dropout: float = 0.1,
        temperature: float = 1.0,
        gamma_init: float = 0.1,
        text_embed_path: Optional[str] = None,
        prompt_trainable: bool = True,
        prompt_residual: bool = True,
        gamma_max: float = 0.2,
        gamma_trainable: bool = True,
        classwise_gamma: bool = False,
        use_gate: bool = False,
        gate_hidden: int = 0,
        gate_init: float = 0.5,
        view_calib_min: float = 0.7,
        view_calib_max: float = 1.3,
        basc_mode: str = "none",
        basc_compat_scale: float = 2.0,
        basc_eps: float = 1e-5,
        trust_router: bool = False,
        trust_hidden: int = 16,
        trust_init: float = 0.05,
        trust_uncertainty_floor: float = 0.1,
        trust_classwise: bool = False,
        trust_candidate_mode: str = "bounded",
        text_center: bool = False,
        text_transform: str = "legacy",
        text_transform_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.sem_dim = int(sem_dim)
        self.temperature = float(max(temperature, 1e-6))
        self.prompt_trainable = bool(prompt_trainable)
        self.prompt_residual = bool(prompt_residual)
        self.gamma_max = float(gamma_max)
        self.gamma_trainable = bool(gamma_trainable)
        self.classwise_gamma = bool(classwise_gamma)
        self.use_gate = bool(use_gate)
        self.gate_init = float(gate_init)
        self.view_calib_min = float(view_calib_min)
        self.view_calib_max = float(view_calib_max)
        self.basc_mode = str(basc_mode).lower()
        self.basc_compat_scale = float(basc_compat_scale)
        self.basc_eps = float(max(basc_eps, 1e-8))
        self.use_trust_router = bool(trust_router)
        self.trust_classwise = bool(trust_classwise)
        self.trust_candidate_mode = str(trust_candidate_mode).lower()
        self.text_center = bool(text_center)
        requested_transform = str(text_transform).lower()
        if requested_transform == "legacy":
            requested_transform = "mean" if self.text_center else "none"
        if requested_transform not in ("none", "mean", "pc1", "whiten"):
            raise ValueError(
                "text_transform must be legacy, none, mean, pc1, or whiten; "
                f"got {text_transform!r}"
            )
        self.text_transform = requested_transform
        self.text_transform_eps = float(max(text_transform_eps, 1e-8))
        self.trust_uncertainty_floor = min(
            max(float(trust_uncertainty_floor), 0.0),
            1.0,
        )
        if self.trust_candidate_mode not in ("bounded", "frozen"):
            raise ValueError(
                "trust_candidate_mode must be bounded or frozen; "
                f"got {trust_candidate_mode!r}"
            )
        if self.basc_mode not in ("none", "norm", "uncertainty", "full"):
            raise ValueError(
                "basc_mode must be one of: none, norm, uncertainty, full; "
                f"got {basc_mode!r}"
            )

        hidden = max(int(in_dim), self.sem_dim)
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.sem_dim),
        )

        self.use_text_prompts = bool(text_embed_path)
        self.text_proj = nn.Identity()
        if self.use_text_prompts:
            text_obj = torch.load(text_embed_path, map_location="cpu")
            text_emb = text_obj["embeddings"] if isinstance(text_obj, dict) else text_obj
            text_emb = text_emb.float()
            if text_emb.dim() != 2:
                raise ValueError(f"embeddings in sem_text_embed_path must be [C,D], got {tuple(text_emb.shape)}")
            if text_emb.size(0) != self.num_classes:
                raise ValueError(
                    f"sem_text_embed_path class count mismatch: expected {self.num_classes}, got {text_emb.size(0)}"
                )

            text_dim = int(text_emb.size(1))
            self.register_buffer("base_text_tokens", text_emb, persistent=True)
            self.text_proj = nn.Identity() if text_dim == self.sem_dim else nn.Linear(text_dim, self.sem_dim, bias=False)

            if self.prompt_residual:
                self.prompt_delta = nn.Parameter(torch.zeros_like(text_emb), requires_grad=self.prompt_trainable)
            elif self.prompt_trainable:
                self.class_tokens = nn.Parameter(text_emb.clone())
            else:
                self.register_buffer("class_tokens", text_emb, persistent=True)
        else:
            self.class_tokens = nn.Parameter(torch.empty(self.num_classes, self.sem_dim))
            nn.init.trunc_normal_(self.class_tokens, std=0.02)

        # small-weight fusion so the R2 head is not disturbed early on; clipped to (0, gamma_max) when gamma_max>0.
        if self.gamma_max > 0:
            ratio = min(max(float(gamma_init) / self.gamma_max, 1e-4), 1.0 - 1e-4)
            gamma_raw = math.log(ratio / (1.0 - ratio))
        else:
            gamma_raw = float(gamma_init)
        if self.classwise_gamma:
            self.gamma = nn.Parameter(
                torch.full((self.num_classes,), gamma_raw, dtype=torch.float32),
                requires_grad=self.gamma_trainable,
            )
        else:
            self.gamma = nn.Parameter(
                torch.tensor(gamma_raw, dtype=torch.float32),
                requires_grad=self.gamma_trainable,
            )

        self.semantic_gate = None
        if self.use_gate:
            gate_hidden_dim = int(gate_hidden) if int(gate_hidden) > 0 else max(64, int(in_dim) // 4)
            self.semantic_gate = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(gate_hidden_dim, self.num_classes),
            )
            gate_prob = min(max(self.gate_init, 1e-4), 1.0 - 1e-4)
            gate_bias = math.log(gate_prob / (1.0 - gate_prob))
            nn.init.zeros_(self.semantic_gate[-1].weight)
            nn.init.constant_(self.semantic_gate[-1].bias, gate_bias)

        self.trust_router = None
        if self.use_trust_router:
            self.trust_router = SemanticTrustRouter(
                hidden_dim=trust_hidden,
                init_prob=trust_init,
                num_classes=self.num_classes,
                classwise=self.trust_classwise,
            )

    def _semantic_tokens(self) -> torch.Tensor:
        if self.use_text_prompts:
            if self.prompt_residual:
                tokens = self.base_text_tokens + self.prompt_delta
            else:
                tokens = self.class_tokens
            tokens = self.text_proj(tokens)
        else:
            tokens = self.class_tokens
        return self._transform_text_tokens(tokens)

    def _transform_text_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.text_transform == "none":
            return tokens

        centered = tokens - tokens.mean(dim=0, keepdim=True)
        if self.text_transform == "mean":
            return centered

        # The decomposition defines an ablation operator, not a trainable path.
        # Detaching it avoids unstable gradients through near-degenerate SVDs.
        # CUDA SVD does not support fp16/bfloat16, so decompose and project in
        # fp32 under AMP, then restore the activation dtype for the caller.
        work = (
            centered.float()
            if centered.dtype in (torch.float16, torch.bfloat16)
            else centered
        )
        with torch.no_grad():
            _, singular_values, vh = torch.linalg.svd(
                work.detach(), full_matrices=False
            )

        if self.text_transform == "pc1":
            direction = vh[:1]
            transformed = work - (work @ direction.t()) @ direction
            return transformed.to(dtype=tokens.dtype)

        variance = singular_values.square() / max(work.size(0) - 1, 1)
        inv_std = torch.rsqrt(variance + self.text_transform_eps)
        coordinates = work @ vh.t()
        transformed = (coordinates * inv_std.unsqueeze(0)) @ vh
        return transformed.to(dtype=tokens.dtype)

    def _gamma(self) -> torch.Tensor:
        if self.gamma_max > 0:
            return self.gamma_max * torch.sigmoid(self.gamma)
        return self.gamma

    def semantic_logits(self, x: torch.Tensor) -> torch.Tensor:
        v = self.visual_proj(x)
        v = F.normalize(v, dim=-1)
        t = F.normalize(self._semantic_tokens(), dim=-1)
        return torch.matmul(v, t.t()) / self.temperature

    def _normalize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        mean = logits.mean(dim=-1, keepdim=True)
        var = (logits - mean).pow(2).mean(dim=-1, keepdim=True)
        return (logits - mean) * torch.rsqrt(var + self.basc_eps)

    def _semantic_correction(
        self,
        x: torch.Tensor,
        logits_sem: torch.Tensor,
        logits_base: Optional[torch.Tensor] = None,
        view_reliability: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if self.use_trust_router:
            if logits_base is None:
                raise ValueError("Semantic trust routing requires base logits")

            sem_norm = self._normalize_logits(logits_sem)
            base_ref = logits_base.detach()
            base_norm = self._normalize_logits(base_ref)
            base_prob = torch.sigmoid(base_ref)
            uncertainty = 4.0 * base_prob * (1.0 - base_prob)
            uncertainty = self.trust_uncertainty_floor + (
                1.0 - self.trust_uncertainty_floor
            ) * uncertainty

            sem_evidence = torch.tanh(sem_norm)
            base_evidence = torch.tanh(base_norm)
            compatibility = base_evidence * sem_evidence
            if view_reliability is None:
                view_reliability = torch.ones_like(logits_sem)
            else:
                view_reliability = view_reliability.clamp(0.0, 1.0)

            router_evidence = torch.stack(
                (
                    uncertainty,
                    base_evidence.abs(),
                    sem_evidence.abs(),
                    compatibility,
                    view_reliability,
                ),
                dim=-1,
            ).detach()
            trust_logits = self.trust_router.forward_logits(router_evidence)
            trust_gate = torch.sigmoid(trust_logits)

            if self.trust_candidate_mode == "frozen":
                # Preserve the strongest observed Frozen-LLM branch and learn
                # only when each class/sample should be allowed to use it.
                trust_candidate = logits_sem * self._gamma()
            else:
                # Conservative candidate used by the original shared router.
                trust_candidate = (
                    sem_evidence
                    * self._gamma()
                    * uncertainty
                    * view_reliability
                )
            semantic_correction = trust_candidate * trust_gate
            aux = {
                "trust_gate": trust_gate,
                "trust_logits": trust_logits,
                "trust_candidate": trust_candidate,
                "router_evidence": router_evidence,
            }
        elif self.basc_mode == "none":
            semantic_correction = logits_sem * self._gamma()
            aux = None
        else:
            if logits_base is None:
                raise ValueError(f"BASC mode {self.basc_mode!r} requires base logits")

            sem_norm = self._normalize_logits(logits_sem)
            semantic_correction = torch.tanh(sem_norm) * self._gamma()

            if self.basc_mode in ("uncertainty", "full"):
                # Stop-gradient keeps the safety gate from changing the base
                # classifier merely to admit a larger semantic correction.
                base_ref = logits_base.detach()
                base_prob = torch.sigmoid(base_ref)
                uncertainty = 4.0 * base_prob * (1.0 - base_prob)
                semantic_correction = semantic_correction * uncertainty

                if self.basc_mode == "full":
                    base_norm = self._normalize_logits(base_ref)
                    compatibility = torch.sigmoid(
                        self.basc_compat_scale * base_norm * sem_norm.detach()
                    )
                    semantic_correction = semantic_correction * compatibility
            aux = None

        if self.semantic_gate is not None:
            semantic_correction = semantic_correction * torch.sigmoid(self.semantic_gate(x))
        return semantic_correction, aux

    def forward(
        self,
        x: torch.Tensor,
        logits_base: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        # x: [B, C]
        logits_sem = self.semantic_logits(x)
        correction, aux = self._semantic_correction(
            x,
            logits_sem,
            logits_base=logits_base,
        )
        if return_aux:
            aux = {} if aux is None else dict(aux)
            aux["semantic_raw_logits"] = logits_sem
            aux["semantic_correction"] = correction
        return (correction, aux) if return_aux else correction

    def forward_view_aware(
        self,
        x: torch.Tensor,
        xa: torch.Tensor,
        xb: torch.Tensor,
        logits_base: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        # x/xa/xb: [B, C]. The fused semantic correction is calibrated by
        # dual-view semantic agreement/support without changing the base head.
        logits_sem = self.semantic_logits(x)
        logits_a = self.semantic_logits(xa)
        logits_b = self.semantic_logits(xb)
        prob_a = torch.sigmoid(logits_a)
        prob_b = torch.sigmoid(logits_b)
        agreement = 1.0 - (prob_a - prob_b).abs()
        support = torch.maximum(prob_a, prob_b)
        reliability = (agreement * support).clamp(0.0, 1.0)

        semantic_correction, aux = self._semantic_correction(
            x,
            logits_sem,
            logits_base=logits_base,
            view_reliability=agreement if self.use_trust_router else None,
        )

        # Trust routing already consumes symmetric A/B agreement. Existing
        # modes retain their original view-aware calibration unchanged.
        if not self.use_trust_router:
            lo = min(self.view_calib_min, self.view_calib_max)
            hi = max(self.view_calib_min, self.view_calib_max)
            calibration = lo + (hi - lo) * reliability
            semantic_correction = semantic_correction * calibration
        if return_aux:
            aux = {} if aux is None else dict(aux)
            aux["semantic_raw_logits"] = logits_sem
            aux["semantic_correction"] = semantic_correction
        return (semantic_correction, aux) if return_aux else semantic_correction


class SpatialAttributeQueryHead(nn.Module):
    """Class-attribute queries over unpooled C3/C4 features from both views."""

    def __init__(
        self,
        c3_dim: int,
        c4_dim: int,
        num_classes: int,
        query_source: str = "random",
        attribute_embed_path: Optional[str] = None,
        query_dim: int = 256,
        text_dim: int = 384,
        num_attributes: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        c3_size: int = 14,
        c4_size: int = 7,
        gamma_init: float = 0.05,
        gamma_max: float = 0.2,
        uncertainty_floor: float = 0.1,
        view_temperature: float = 0.2,
        random_seed: int = 0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_attributes = int(num_attributes)
        self.query_dim = int(query_dim)
        self.text_dim = int(text_dim)
        self.c3_size = int(c3_size)
        self.c4_size = int(c4_size)
        self.gamma_max = float(gamma_max)
        self.uncertainty_floor = min(max(float(uncertainty_floor), 0.0), 1.0)
        self.view_temperature = max(float(view_temperature), 1e-6)
        self.query_source = str(query_source).lower()

        if self.query_source not in ("random", "llm"):
            raise ValueError(
                "spatial_query_source must be random or llm; "
                f"got {query_source!r}"
            )
        if self.query_dim % int(num_heads) != 0:
            raise ValueError(
                f"spatial_query_dim={self.query_dim} must be divisible by "
                f"spatial_query_heads={num_heads}"
            )
        if self.num_attributes <= 0:
            raise ValueError("spatial_query_attributes must be positive")

        groups = math.gcd(self.query_dim, 32)
        self.c3_proj = nn.Sequential(
            nn.Conv2d(int(c3_dim), self.query_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.query_dim),
            nn.GELU(),
        )
        self.c4_proj = nn.Sequential(
            nn.Conv2d(int(c4_dim), self.query_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.query_dim),
            nn.GELU(),
        )

        query_tokens = self._build_query_tokens(
            attribute_embed_path=attribute_embed_path,
            random_seed=int(random_seed),
        )
        self.register_buffer("base_query_tokens", query_tokens, persistent=True)
        self.query_delta = nn.Parameter(torch.zeros_like(query_tokens))
        self.query_norm = nn.LayerNorm(self.text_dim)
        self.query_proj = nn.Linear(self.text_dim, self.query_dim, bias=False)

        self.level_embed = nn.Parameter(torch.zeros(2, self.query_dim))
        self.c3_pos_embed = nn.Parameter(
            torch.zeros(1, self.c3_size * self.c3_size, self.query_dim)
        )
        self.c4_pos_embed = nn.Parameter(
            torch.zeros(1, self.c4_size * self.c4_size, self.query_dim)
        )
        nn.init.trunc_normal_(self.level_embed, std=0.02)
        nn.init.trunc_normal_(self.c3_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.c4_pos_embed, std=0.02)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(float(dropout))
        self.attn_norm = nn.LayerNorm(self.query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.query_dim, 4 * self.query_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * self.query_dim, self.query_dim),
        )
        self.ffn_dropout = nn.Dropout(float(dropout))
        self.ffn_norm = nn.LayerNorm(self.query_dim)
        self.attribute_classifier = nn.Linear(self.query_dim, 1)
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))

        if self.gamma_max > 0:
            ratio = min(max(float(gamma_init) / self.gamma_max, 1e-4), 1.0 - 1e-4)
            gamma_raw = math.log(ratio / (1.0 - ratio))
        else:
            gamma_raw = float(gamma_init)
        self.gamma = nn.Parameter(
            torch.full((self.num_classes,), gamma_raw, dtype=torch.float32)
        )

    def _build_query_tokens(
        self,
        attribute_embed_path: Optional[str],
        random_seed: int,
    ) -> torch.Tensor:
        expected = (self.num_classes, self.num_attributes, self.text_dim)
        if self.query_source == "llm":
            if not attribute_embed_path:
                raise ValueError(
                    "spatial_query_source=llm requires --spatial_attribute_embed_path"
                )
            text_obj = torch.load(attribute_embed_path, map_location="cpu")
            tokens = (
                text_obj["embeddings"]
                if isinstance(text_obj, dict)
                else text_obj
            )
            tokens = tokens.detach().float().cpu()
            if tuple(tokens.shape) != expected:
                raise ValueError(
                    "Spatial attribute embeddings must have shape "
                    f"{expected}, got {tuple(tokens.shape)}"
                )
            if not torch.isfinite(tokens).all():
                raise ValueError("Spatial attribute embeddings contain non-finite values")
            return F.normalize(tokens, dim=-1)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(random_seed)
        tokens = torch.randn(expected, generator=generator, dtype=torch.float32)
        return F.normalize(tokens, dim=-1)

    def _gamma(self) -> torch.Tensor:
        if self.gamma_max > 0:
            return self.gamma_max * torch.sigmoid(self.gamma)
        return self.gamma

    def _spatial_tokens(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "C3" not in feats or "C4" not in feats:
            raise RuntimeError("Spatial attribute queries require unpooled C3 and C4 features")
        c3 = F.adaptive_avg_pool2d(
            self.c3_proj(feats["C3"]), (self.c3_size, self.c3_size)
        ).flatten(2).transpose(1, 2)
        c4 = F.adaptive_avg_pool2d(
            self.c4_proj(feats["C4"]), (self.c4_size, self.c4_size)
        ).flatten(2).transpose(1, 2)
        c3 = c3 + self.c3_pos_embed + self.level_embed[0].view(1, 1, -1)
        c4 = c4 + self.c4_pos_embed + self.level_embed[1].view(1, 1, -1)
        return torch.cat((c3, c4), dim=1)

    def _decode_view(
        self,
        feats: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_tokens = self._spatial_tokens(feats)
        batch_size = spatial_tokens.size(0)
        query = self.query_proj(
            self.query_norm(self.base_query_tokens + self.query_delta)
        )
        query = query.reshape(-1, self.query_dim).unsqueeze(0)
        query = query.expand(batch_size, -1, -1)

        attended, weights = self.cross_attention(
            query,
            spatial_tokens,
            spatial_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        decoded = self.attn_norm(query + self.attn_dropout(attended))
        decoded = self.ffn_norm(decoded + self.ffn_dropout(self.ffn(decoded)))
        attribute_logits = self.attribute_classifier(decoded).squeeze(-1)
        attribute_logits = attribute_logits.view(
            batch_size, self.num_classes, self.num_attributes
        )
        class_logits = attribute_logits.mean(dim=-1) + self.class_bias.unsqueeze(0)

        # High confidence means the class attributes consistently attend to a
        # compact subset of spatial tokens. The resulting view weights are
        # detached later so attention cannot improve routing by collapsing.
        attention = weights.float().mean(dim=1).clamp_min(1e-8)
        entropy = -(attention * attention.log()).sum(dim=-1)
        entropy = entropy / math.log(max(int(attention.size(-1)), 2))
        confidence = 1.0 - entropy
        confidence = confidence.view(
            batch_size, self.num_classes, self.num_attributes
        ).mean(dim=-1)
        return class_logits, confidence, attribute_logits

    def forward(
        self,
        feats_a: Dict[str, torch.Tensor],
        feats_b: Dict[str, torch.Tensor],
        logits_base: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits_a, confidence_a, attribute_logits_a = self._decode_view(feats_a)
        logits_b, confidence_b, attribute_logits_b = self._decode_view(feats_b)

        confidence = torch.stack((confidence_a, confidence_b), dim=1)
        view_weights = torch.softmax(
            confidence / self.view_temperature,
            dim=1,
        )
        fusion_weights = view_weights.detach().to(dtype=logits_a.dtype)
        query_logits = (
            fusion_weights[:, 0] * logits_a
            + fusion_weights[:, 1] * logits_b
        )

        base_prob = torch.sigmoid(logits_base.detach())
        uncertainty = 4.0 * base_prob * (1.0 - base_prob)
        uncertainty = self.uncertainty_floor + (
            1.0 - self.uncertainty_floor
        ) * uncertainty
        correction = (
            torch.tanh(query_logits)
            * uncertainty
            * self._gamma().unsqueeze(0).to(dtype=query_logits.dtype)
        )
        return {
            "query_logits": query_logits,
            "query_logits_a": logits_a,
            "query_logits_b": logits_b,
            "attribute_logits_a": attribute_logits_a,
            "attribute_logits_b": attribute_logits_b,
            "correction": correction,
            "view_weights": view_weights,
            "view_confidence": confidence,
            "gamma": self._gamma(),
        }


class SharedViewEvidenceHead(nn.Module):
    """Training-only shared classifier for evidence from either X-ray view."""

    def __init__(
        self,
        level_dims: Dict[str, int],
        num_classes: int,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.levels = tuple(
            level for level in ("C3", "C4", "C5") if level in level_dims
        )
        if len(self.levels) < 2:
            raise ValueError("View evidence head requires at least two feature levels")

        projection_dim = int(projection_dim)
        hidden_dim = int(hidden_dim)
        self.level_projections = nn.ModuleDict(
            {
                level: nn.Sequential(
                    nn.LayerNorm(int(level_dims[level])),
                    nn.Linear(int(level_dims[level]), projection_dim),
                    nn.GELU(),
                )
                for level in self.levels
            }
        )
        fused_dim = projection_dim * len(self.levels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(num_classes)),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        vectors = []
        for level in self.levels:
            if level not in feats:
                raise RuntimeError(f"Missing {level} for shared view evidence head")
            pooled = feats[level].mean(dim=(-2, -1))
            vectors.append(self.level_projections[level](pooled))
        return self.classifier(torch.cat(vectors, dim=-1))


class SelectiveCrossViewRescueHead(nn.Module):
    """Bounded cross-view correction restricted to uncertain R2 decisions."""

    def __init__(self, level_dims, num_classes, projection_dim=128,
                 hidden_dim=256, dropout=0.1, gamma_max=0.05,
                 uncertainty_threshold=0.5, gate_temperature=0.1,
                 detach_features=True):
        super().__init__()
        self.evidence_head = SharedViewEvidenceHead(
            level_dims, num_classes, projection_dim, hidden_dim, dropout
        )
        self.gamma_max = max(float(gamma_max), 0.0)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.gate_temperature = max(float(gate_temperature), 1e-6)
        self.detach_features = bool(detach_features)
        self.gamma_raw = nn.Parameter(torch.zeros(int(num_classes)))

    def forward(self, feats_a, feats_b, logits_base):
        if self.detach_features:
            feats_a = {k: v.detach() for k, v in feats_a.items()}
            feats_b = {k: v.detach() for k, v in feats_b.items()}
        logits_a = self.evidence_head(feats_a)
        logits_b = self.evidence_head(feats_b)
        base_ref = logits_base.detach()
        base_prob = torch.sigmoid(base_ref)
        uncertainty = 4.0 * base_prob * (1.0 - base_prob)
        uncertainty_gate = torch.sigmoid(
            (uncertainty - self.uncertainty_threshold) / self.gate_temperature
        )
        stacked = torch.stack((logits_a, logits_b), dim=1)
        view_confidence = stacked.detach().abs()
        view_weights = torch.softmax(view_confidence / 0.2, dim=1)
        candidate = (view_weights * stacked).sum(dim=1)
        disagreement = torch.tanh((logits_a - logits_b).abs()).detach()
        complement_gate = uncertainty_gate * disagreement
        gamma = self.gamma_raw.clamp(0.0, self.gamma_max)
        delta = torch.tanh(candidate - base_ref)
        correction = gamma.unsqueeze(0) * complement_gate * delta
        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "candidate_logits": candidate,
            "correction": correction,
            "uncertainty": uncertainty,
            "gate": complement_gate,
            "gamma": gamma,
        }


class FrozenAnchorCrossViewRescueHead(nn.Module):
    """Positive-only dual-view rescue on top of an immutable R2 anchor."""

    def __init__(self, level_dims, num_classes, projection_dim=128,
                 hidden_dim=256, dropout=0.0, gamma_max=0.03,
                 uncertainty_threshold=0.35, gate_temperature=0.1,
                 trust_init=0.05):
        super().__init__()
        self.evidence_head = SharedViewEvidenceHead(
            level_dims, num_classes, projection_dim, hidden_dim, dropout
        )
        trust_hidden = max(16, hidden_dim // 8)
        self.trust_head = nn.Sequential(
            nn.Linear(6, trust_hidden), nn.GELU(), nn.Linear(trust_hidden, 1)
        )
        trust_init = min(max(float(trust_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.trust_head[-1].weight)
        nn.init.constant_(
            self.trust_head[-1].bias,
            math.log(trust_init / (1.0 - trust_init)),
        )
        self.gamma_raw = nn.Parameter(torch.zeros(int(num_classes)))
        self.gamma_max = max(float(gamma_max), 0.0)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.gate_temperature = max(float(gate_temperature), 1e-6)

    def forward(self, feats_a, feats_b, logits_base):
        logits_a = self.evidence_head(feats_a)
        logits_b = self.evidence_head(feats_b)
        base_ref = logits_base.detach()
        base_prob = torch.sigmoid(base_ref)
        prob_a = torch.sigmoid(logits_a)
        prob_b = torch.sigmoid(logits_b)
        noisy_or = 1.0 - (1.0 - prob_a) * (1.0 - prob_b)
        eps = torch.finfo(noisy_or.dtype).eps
        noisy_or = noisy_or.clamp(eps, 1.0 - eps)
        candidate_logits = torch.logit(noisy_or)
        positive_delta = F.relu(candidate_logits - base_ref)
        uncertainty = 4.0 * base_prob * (1.0 - base_prob)
        uncertainty_gate = torch.sigmoid(
            (uncertainty - self.uncertainty_threshold) / self.gate_temperature
        )
        complement_gate = torch.sigmoid(
            (positive_delta - 0.25) / self.gate_temperature
        )
        trust_features = torch.stack(
            (base_prob, prob_a, prob_b, noisy_or,
             (prob_a - prob_b).abs(), uncertainty),
            dim=-1,
        ).detach()
        trust_logits = self.trust_head(trust_features).squeeze(-1)
        trust_gate = torch.sigmoid(trust_logits)
        gate = uncertainty_gate * complement_gate * trust_gate
        gamma = self.gamma_raw.clamp(0.0, self.gamma_max)
        correction = gamma.unsqueeze(0) * gate * torch.tanh(positive_delta)
        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "candidate_logits": candidate_logits,
            "positive_delta": positive_delta,
            "trust_logits": trust_logits,
            "trust_gate": trust_gate,
            "uncertainty": uncertainty,
            "gate": gate,
            "gamma": gamma,
            "correction": correction,
        }

# ------------------------------
# Dual Wrapper
# ------------------------------
class ConvNeXtV2Dual(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        ahcr_mode: str = 'intra_level',
        return_intermediate: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3),
        fuse_mode: str = "concat",
        fuse_levels: Tuple[str, ...] = ("C3", "C4", "C5"),
        head_type: str = "c5",
        xattn_heads: int = 4,
        xattn_reduction: int = 4,
        fpn_out_channels: int = 256,
        # new parameters, passed in by main_finetune.py
        attention_config: Optional[Dict] = None,
        use_cv_gsc: bool = False,
        cv_spatial_reduction: int = 2,
        # vision-semantic auxiliary branch; identical to the original model when disabled
        use_semantic_branch: bool = False,
        sem_aux_only: bool = False,
        sem_dim: int = 256,
        sem_dropout: float = 0.1,
        sem_temperature: float = 1.0,
        sem_gamma_init: float = 0.1,
        sem_text_embed_path: Optional[str] = None,
        sem_prompt_trainable: bool = True,
        sem_prompt_residual: bool = True,
        sem_gamma_max: float = 0.2,
        sem_gamma_trainable: bool = True,
        sem_classwise_gamma: bool = False,
        sem_use_gate: bool = False,
        sem_gate_hidden: int = 0,
        sem_gate_init: float = 0.5,
        sem_view_calib: bool = False,
        sem_view_calib_min: float = 0.7,
        sem_view_calib_max: float = 1.3,
        sem_basc_mode: str = "none",
        sem_basc_compat_scale: float = 2.0,
        sem_basc_eps: float = 1e-5,
        sem_trust_router: bool = False,
        sem_trust_hidden: int = 16,
        sem_trust_init: float = 0.05,
        sem_trust_uncertainty_floor: float = 0.1,
        sem_trust_classwise: bool = False,
        sem_trust_candidate_mode: str = "bounded",
        sem_rank_calibration: bool = False,
        sem_error_calibration: bool = False,
        sem_text_center: bool = False,
        sem_text_transform: str = "legacy",
        sem_text_transform_eps: float = 1e-5,
        # Spatial attribute query branch. Disabled by default.
        use_spatial_query: bool = False,
        spatial_query_source: str = "random",
        spatial_attribute_embed_path: Optional[str] = None,
        spatial_query_dim: int = 256,
        spatial_query_text_dim: int = 384,
        spatial_query_attributes: int = 4,
        spatial_query_heads: int = 4,
        spatial_query_dropout: float = 0.1,
        spatial_query_c3_size: int = 14,
        spatial_query_c4_size: int = 7,
        spatial_query_gamma_init: float = 0.05,
        spatial_query_gamma_max: float = 0.2,
        spatial_query_uncertainty_floor: float = 0.1,
        spatial_query_view_temperature: float = 0.2,
        spatial_query_random_seed: int = 0,
        # Training-only cross-view best-evidence distillation.
        use_view_evidence_distill: bool = False,
        view_evidence_projection_dim: int = 128,
        view_evidence_hidden_dim: int = 256,
        view_evidence_dropout: float = 0.1,
        # Class-wise region complement fusion (DV-CRE).
        use_dvcre: bool = False,
        dvcre_levels: Tuple[str, ...] = ("C3", "C4"),
        dvcre_projection_dim: int = 64,
        dvcre_topk_ratio: float = 0.25,
        dvcre_temperature: float = 0.2,
        dvcre_residual_init: float = 0.05,
        dvcre_residual_max: float = 0.2,
        # Intervention-stable cross-view fusion (IS-CVF).
        use_iscvf: bool = False,
        iscvf_levels: Tuple[str, ...] = ("C3", "C4", "C5"),
        iscvf_gate_reduction: int = 16,
        iscvf_keep_prob: float = 0.75,
        # Visual-only sample/class/region evidence routing.
        use_visual_evidence_router: bool = False,
        visual_route_mode: str = "pg_cver",
        visual_route_level: str = "C4",
        visual_route_projection_dim: int = 64,
        visual_route_topk_ratio: float = 0.25,
        visual_route_temperature: float = 0.2,
        visual_route_shared_axis: str = "width",
        visual_route_axis_radius: int = 1,
        visual_route_gate_init: float = 0.05,
        visual_route_gamma_init: float = 0.05,
        visual_route_gamma_max: float = 0.25,
        visual_route_reject_temperature: float = 0.1,
        visual_route_dropout: float = 0.1,
        use_selective_view_rescue: bool = False,
        selective_rescue_projection_dim: int = 128,
        selective_rescue_hidden_dim: int = 256,
        selective_rescue_dropout: float = 0.1,
        selective_rescue_gamma_max: float = 0.05,
        selective_rescue_uncertainty_threshold: float = 0.5,
        selective_rescue_gate_temperature: float = 0.1,
        selective_rescue_detach_features: bool = True,
        use_frozen_anchor_rescue: bool = False,
        frozen_rescue_aux_only: bool = False,
        frozen_rescue_projection_dim: int = 128,
        frozen_rescue_hidden_dim: int = 256,
        frozen_rescue_dropout: float = 0.0,
        frozen_rescue_gamma_max: float = 0.03,
        frozen_rescue_uncertainty_threshold: float = 0.35,
        frozen_rescue_gate_temperature: float = 0.1,
        frozen_rescue_trust_init: float = 0.05,
        use_frozen_region_rescue: bool = False,
        frozen_region_aux_only: bool = False,
        frozen_region_levels: Tuple[str, ...] = ("C3", "C4"),
        frozen_region_projection_dim: int = 64,
        frozen_region_topk_ratio: float = 0.1,
        frozen_region_temperature: float = 0.2,
        frozen_region_gamma_init: float = 0.003,
        frozen_region_gamma_max: float = 0.02,
        frozen_region_uncertainty_threshold: float = 0.35,
        frozen_region_gate_temperature: float = 0.1,
        frozen_region_trust_init: float = 0.05,
        use_frozen_counterfactual_router: bool = False,
        frozen_counterfactual_aux_only: bool = False,
        frozen_counterfactual_hidden_dim: int = 32,
        frozen_counterfactual_class_embed_dim: int = 8,
        frozen_counterfactual_rho_init: float = 0.05,
        frozen_counterfactual_rho_max: float = 0.2,
        frozen_counterfactual_rescue_init: float = 0.1,
        frozen_counterfactual_delta_clip: float = 6.0,
        frozen_counterfactual_region_level: str = "",
        frozen_counterfactual_region_projection_dim: int = 32,
        frozen_counterfactual_region_temperature: float = 0.2,
        use_frozen_region_interaction_moe: bool = False,
        frozen_region_interaction_aux_only: bool = False,
        frozen_region_interaction_levels: Tuple[str, ...] = ("C3", "C4"),
        frozen_region_interaction_projection_dim: int = 64,
        frozen_region_interaction_temperature: float = 0.2,
        frozen_region_interaction_shared_axis: str = "width",
        frozen_region_interaction_axis_radius: int = 1,
        frozen_region_interaction_residual_init: float = 0.03,
        frozen_region_interaction_residual_max: float = 0.2,
        frozen_region_interaction_router_hidden_dim: int = 128,
        frozen_region_interaction_class_embed_dim: int = 16,
        frozen_region_interaction_rho_init: float = 0.08,
        frozen_region_interaction_rho_max: float = 0.25,
        frozen_region_interaction_rescue_init: float = 0.05,
        frozen_region_interaction_delta_clip: float = 6.0,
        frozen_region_interaction_include_counterfactual: bool = False,
        # Plain-BCE visual innovation suite. Every branch is opt-in.
        use_p9_caprs: bool = False,
        use_p10_wgcr: bool = False,
        use_p11_otcvr: bool = False,
        use_p12_berf: bool = False,
        use_p13_vdrm: bool = False,
        use_p14_hcaer: bool = False,
        use_p15_vtr: bool = False,
        use_p16_facgr: bool = False,
        use_p17_dcasr: bool = False,
        use_p18_ewsar: bool = False,
        use_p19_apcer: bool = False,
        use_p20_cvcr: bool = False,
        plain_innovation_levels: Tuple[str, ...] = ("C4", "C5"),
        plain_innovation_projection_dim: int = 64,
        plain_innovation_topk: int = 8,
        plain_innovation_temperature: float = 0.2,
        plain_innovation_dropout: float = 0.1,
        plain_innovation_gamma_init: float = 0.005,
        plain_innovation_gamma_max: float = 0.05,
        plain_innovation_base_floor: float = 0.8,
        plain_innovation_use_counterfactual_experts: bool = True,
        plain_innovation_use_learned_router: bool = True,
        plain_innovation_router_variant: str = "legacy",
        plain_innovation_gamma_trainable: bool = True,
        plain_innovation_warmup_epochs: int = 15,
        plain_innovation_ramp_epochs: int = 10,
        plain_innovation_sinkhorn_iters: int = 4,
        plain_innovation_router_start_epoch: int = 8,
        plain_innovation_router_ramp_epochs: int = 3,
        plain_innovation_gate_init: float = 0.15,
        plain_innovation_trust_threshold: float = 0.50,
        plain_innovation_trust_temperature: float = 0.10,
        plain_innovation_uncertainty_floor: float = 0.25,
        plain_innovation_budget_target: float = 0.12,
        plain_innovation_advantage_temperature: float = 0.05,
        plain_innovation_gain_margin: float = 1e-5,
        plain_innovation_score_temperature: float = 1.0,
        plain_innovation_corrupt_probability: float = 0.75,
        plain_innovation_corrupt_ratio: float = 0.25,
        plain_innovation_full_view_drop_probability: float = 0.35,
        plain_innovation_complement_margin: float = 0.02,
        plain_innovation_complement_temperature: float = 0.05,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.return_intermediate = bool(return_intermediate)
        self.out_indices = tuple(out_indices)
        self.fuse_mode = str(fuse_mode)
        self.fuse_levels = tuple(fuse_levels)
        self.head_type = str(head_type) # head_type is now a base type such as 'fpn_pan'
        self.attention_config = attention_config
        self.use_cv_gsc = bool(use_cv_gsc)
        self.cv_spatial_reduction = int(cv_spatial_reduction)
        self.use_semantic_branch = bool(use_semantic_branch)
        self.sem_aux_only = bool(sem_aux_only)
        self.sem_dim = int(sem_dim)
        self.sem_dropout = float(sem_dropout)
        self.sem_temperature = float(sem_temperature)
        self.sem_gamma_init = float(sem_gamma_init)
        self.sem_text_embed_path = sem_text_embed_path
        self.sem_prompt_trainable = bool(sem_prompt_trainable)
        self.sem_prompt_residual = bool(sem_prompt_residual)
        self.sem_gamma_max = float(sem_gamma_max)
        self.sem_gamma_trainable = bool(sem_gamma_trainable)
        self.sem_classwise_gamma = bool(sem_classwise_gamma)
        self.sem_use_gate = bool(sem_use_gate)
        self.sem_gate_hidden = int(sem_gate_hidden)
        self.sem_gate_init = float(sem_gate_init)
        self.sem_view_calib = bool(sem_view_calib)
        self.sem_view_calib_min = float(sem_view_calib_min)
        self.sem_view_calib_max = float(sem_view_calib_max)
        self.sem_basc_mode = str(sem_basc_mode).lower()
        self.sem_basc_compat_scale = float(sem_basc_compat_scale)
        self.sem_basc_eps = float(sem_basc_eps)
        self.sem_trust_router = bool(sem_trust_router)
        self.sem_trust_hidden = int(sem_trust_hidden)
        self.sem_trust_init = float(sem_trust_init)
        self.sem_trust_uncertainty_floor = float(sem_trust_uncertainty_floor)
        self.sem_trust_classwise = bool(sem_trust_classwise)
        self.sem_trust_candidate_mode = str(sem_trust_candidate_mode).lower()
        self.sem_rank_calibration = bool(sem_rank_calibration)
        self.sem_error_calibration = bool(sem_error_calibration)
        self.sem_text_center = bool(sem_text_center)
        self.sem_text_transform = str(sem_text_transform).lower()
        self.sem_text_transform_eps = float(sem_text_transform_eps)
        self.use_spatial_query = bool(use_spatial_query)
        self.use_view_evidence_distill = bool(use_view_evidence_distill)
        self.use_dvcre = bool(use_dvcre)
        self.dvcre_levels = tuple(dvcre_levels)
        self.use_iscvf = bool(use_iscvf)
        self.iscvf_levels = tuple(iscvf_levels)
        self.iscvf_keep_prob = float(iscvf_keep_prob)
        self.use_visual_evidence_router = bool(use_visual_evidence_router)
        self.visual_route_mode = str(visual_route_mode).lower()
        self.visual_route_level = str(visual_route_level)
        self.use_selective_view_rescue = bool(use_selective_view_rescue)
        self.use_frozen_anchor_rescue = bool(use_frozen_anchor_rescue)
        self.frozen_rescue_aux_only = bool(frozen_rescue_aux_only)
        self.use_frozen_region_rescue = bool(use_frozen_region_rescue)
        self.frozen_region_aux_only = bool(frozen_region_aux_only)
        self.use_frozen_counterfactual_router = bool(
            use_frozen_counterfactual_router
        )
        self.frozen_counterfactual_aux_only = bool(
            frozen_counterfactual_aux_only
        )
        self.frozen_counterfactual_region_level = str(
            frozen_counterfactual_region_level
        )
        self.use_frozen_region_interaction_moe = bool(
            use_frozen_region_interaction_moe
        )
        self.frozen_region_interaction_aux_only = bool(
            frozen_region_interaction_aux_only
        )
        innovation_flags = {
            "p9_caprs": bool(use_p9_caprs),
            "p10_wgcr": bool(use_p10_wgcr),
            "p11_otcvr": bool(use_p11_otcvr),
            "p12_berf": bool(use_p12_berf),
            "p13_vdrm": bool(use_p13_vdrm),
            "p14_hcaer": bool(use_p14_hcaer),
            "p15_vtr": bool(use_p15_vtr),
            "p16_facgr": bool(use_p16_facgr),
            "p17_dcasr": bool(use_p17_dcasr),
            "p18_ewsar": bool(use_p18_ewsar),
            "p19_apcer": bool(use_p19_apcer),
            "p20_cvcr": bool(use_p20_cvcr),
        }
        enabled_innovations = [
            name for name, enabled in innovation_flags.items() if enabled
        ]
        if len(enabled_innovations) > 1:
            raise ValueError(
                "enable only one Plain-BCE innovation at a time, got "
                + ", ".join(enabled_innovations)
            )
        self.plain_innovation_mode = (
            enabled_innovations[0] if enabled_innovations else ""
        )
        self.plain_innovation_levels = tuple(plain_innovation_levels)
        self.plain_innovation_use_counterfactual_experts = bool(
            plain_innovation_use_counterfactual_experts
        )
        self.plain_innovation_needs_single_logits = (
            (
                self.plain_innovation_mode == "p9_caprs"
                and self.plain_innovation_use_counterfactual_experts
            )
            or self.plain_innovation_mode in (
                "p13_vdrm", "p16_facgr", "p18_ewsar", "p19_apcer",
                "p20_cvcr"
            )
        )
        self.xattn_heads = int(xattn_heads)
        self.xattn_reduction = int(xattn_reduction)
        self._semantic_base_frozen = False
        self._spatial_query_base_frozen = False
        self._frozen_anchor_base_frozen = False
        self._frozen_region_base_frozen = False
        self._frozen_counterfactual_base_frozen = False
        self._frozen_region_interaction_base_frozen = False
        self._plain_innovation_base_frozen = False

        if hasattr(backbone, "dims"):
            dims = list(backbone.dims)
        elif hasattr(backbone, "feature_info"):
            chs = list(map(int, backbone.feature_info.channels()))
            if len(chs) >= 5:
                dims = [chs[1], chs[2], chs[3], chs[4]]  # aligned with C2..C5
            elif len(chs) == 4:
                dims = chs
            else:
                raise AttributeError(f"cannot infer the 4 level channel counts from feature_info, got {chs}")
        else:
            raise AttributeError("backbone has neither 'dims' nor 'feature_info'; cannot build the pyramid channel spec.")

        self.stage_to_name = {1: "C3", 2: "C4", 3: "C5"}
        self.name_to_dim = {self.stage_to_name[i]: dims[i] for i in self.out_indices}

        if (self.use_dvcre or self.use_iscvf) and self.fuse_mode != "add":
            raise ValueError("DV-CRE and IS-CVF require fuse_mode='add'")
        if self.plain_innovation_mode and self.fuse_mode != "add":
            raise ValueError("Plain-BCE innovations require fuse_mode='add'")
        if not 0.0 < self.iscvf_keep_prob <= 1.0:
            raise ValueError("iscvf_keep_prob must be in (0, 1]")

        self.dvcre_fuses = nn.ModuleDict()
        if self.use_dvcre:
            for name in self.dvcre_levels:
                if name not in self.name_to_dim:
                    raise ValueError(f"Unknown DV-CRE level: {name}")
                self.dvcre_fuses[name] = ClasswiseRegionComplementFuse(
                    channels=self.name_to_dim[name],
                    num_classes=self.num_classes,
                    projection_dim=dvcre_projection_dim,
                    topk_ratio=dvcre_topk_ratio,
                    temperature=dvcre_temperature,
                    residual_init=dvcre_residual_init,
                    residual_max=dvcre_residual_max,
                )

        self.iscvf_fuses = nn.ModuleDict()
        if self.use_iscvf:
            for name in self.iscvf_levels:
                if name not in self.name_to_dim:
                    raise ValueError(f"Unknown IS-CVF level: {name}")
                self.iscvf_fuses[name] = InterventionStableFuse(
                    channels=self.name_to_dim[name],
                    reduction=iscvf_gate_reduction,
                )

        # CV-GSC aligns same-level features before dual-view fusion, using the backbone channels of C3/C4/C5
        self.cv_gsc = nn.ModuleDict()
        if self.use_cv_gsc:
            for name in self.fuse_levels:
                if name in self.name_to_dim:
                    self.cv_gsc[name] = CrossViewGeoSemanticAlign(
                        channels=self.name_to_dim[name],
                        spatial_reduction=self.cv_spatial_reduction,
                    )

        self.xattn_fuses = nn.ModuleDict()
        self.fuse_convs = nn.ModuleDict()
        self.gated_fuses = nn.ModuleDict()

        if self.fuse_mode in ("concat", "gated"):
            for name in self.fuse_levels:
                if name not in self.name_to_dim: continue
                C = self.name_to_dim[name]
                self.fuse_convs[name] = Conv1x1(in_ch=2 * C, out_ch=C, act=True)
                if self.fuse_mode == "gated":
                    self.gated_fuses[name] = GatedFuse(in_ch=2 * C, out_ch=C)
        elif self.fuse_mode == "xattn":
            for name in self.fuse_levels:
                if name not in self.name_to_dim: continue
                C = self.name_to_dim[name]
                self.xattn_fuses[name] = XAttnFuse(
                    channels=C, num_heads=self.xattn_heads, reduction=self.xattn_reduction
                )
        elif self.fuse_mode == "ahcr":
            c3, c4, c5 = self.name_to_dim["C3"], self.name_to_dim["C4"], self.name_to_dim["C5"]
            self.ahcr_fuser = AHCRFuse(c3, c4, c5, mode=ahcr_mode)

        self.fpn = None
        self.fpn_pan = None
        ht = self.head_type.lower()
        if ht == "fpn_fuse":
            c3, c4, c5 = self.name_to_dim["C3"], self.name_to_dim["C4"], self.name_to_dim["C5"]
            self.fpn = FPN(c3, c4, c5, out_channels=fpn_out_channels)
        elif ht == "fpn_pan":
            c3, c4, c5 = self.name_to_dim["C3"], self.name_to_dim["C4"], self.name_to_dim["C5"]
            self.fpn_pan = FPN_PAN(c3, c4, c5, out_channels=fpn_out_channels, attention_config=self.attention_config)

        if ht == "c5":
            feat_ch = self.name_to_dim["C5"]
        elif ht == "fpn":
            feat_ch = sum(self.name_to_dim.get(n, 0) for n in self.fuse_levels)
        elif ht in ("fpn_fuse", "fpn_pan"):
            feat_ch = fpn_out_channels
        else:
            raise ValueError(f"head_type must be 'c5', 'fpn', 'fpn_fuse' or 'fpn_pan', got: {ht}")

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feat_ch, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.view_semantic_adapter = None
        if self.sem_view_calib:
            c5_ch = int(self.name_to_dim.get("C5", feat_ch))
            self.view_semantic_adapter = (
                nn.Identity()
                if c5_ch == feat_ch
                else nn.Sequential(nn.LayerNorm(c5_ch), nn.Linear(c5_ch, feat_ch))
            )

        self.semantic_head = None
        if self.use_semantic_branch and self.num_classes > 0:
            self.semantic_head = SemanticClassEmbeddingHead(
                in_dim=feat_ch,
                num_classes=self.num_classes,
                sem_dim=self.sem_dim,
                dropout=self.sem_dropout,
                temperature=self.sem_temperature,
                gamma_init=self.sem_gamma_init,
                text_embed_path=self.sem_text_embed_path,
                prompt_trainable=self.sem_prompt_trainable,
                prompt_residual=self.sem_prompt_residual,
                gamma_max=self.sem_gamma_max,
                gamma_trainable=self.sem_gamma_trainable,
                classwise_gamma=self.sem_classwise_gamma,
                use_gate=self.sem_use_gate,
                gate_hidden=self.sem_gate_hidden,
                gate_init=self.sem_gate_init,
                view_calib_min=self.sem_view_calib_min,
                view_calib_max=self.sem_view_calib_max,
                basc_mode=self.sem_basc_mode,
                basc_compat_scale=self.sem_basc_compat_scale,
                basc_eps=self.sem_basc_eps,
                trust_router=self.sem_trust_router,
                trust_hidden=self.sem_trust_hidden,
                trust_init=self.sem_trust_init,
                trust_uncertainty_floor=self.sem_trust_uncertainty_floor,
                trust_classwise=self.sem_trust_classwise,
                trust_candidate_mode=self.sem_trust_candidate_mode,
                text_center=self.sem_text_center,
                text_transform=self.sem_text_transform,
                text_transform_eps=self.sem_text_transform_eps,
            )

        self.spatial_query_head = None
        if self.use_spatial_query and self.num_classes > 0:
            if "C3" not in self.name_to_dim or "C4" not in self.name_to_dim:
                raise RuntimeError("Spatial attribute queries require C3 and C4 features")
            self.spatial_query_head = SpatialAttributeQueryHead(
                c3_dim=self.name_to_dim["C3"],
                c4_dim=self.name_to_dim["C4"],
                num_classes=self.num_classes,
                query_source=spatial_query_source,
                attribute_embed_path=spatial_attribute_embed_path,
                query_dim=spatial_query_dim,
                text_dim=spatial_query_text_dim,
                num_attributes=spatial_query_attributes,
                num_heads=spatial_query_heads,
                dropout=spatial_query_dropout,
                c3_size=spatial_query_c3_size,
                c4_size=spatial_query_c4_size,
                gamma_init=spatial_query_gamma_init,
                gamma_max=spatial_query_gamma_max,
                uncertainty_floor=spatial_query_uncertainty_floor,
                view_temperature=spatial_query_view_temperature,
                random_seed=spatial_query_random_seed,
            )

        self.view_evidence_head = None
        if self.use_view_evidence_distill and self.num_classes > 0:
            self.view_evidence_head = SharedViewEvidenceHead(
                level_dims=self.name_to_dim,
                num_classes=self.num_classes,
                projection_dim=view_evidence_projection_dim,
                hidden_dim=view_evidence_hidden_dim,
                dropout=view_evidence_dropout,
            )

        self.visual_evidence_router = None
        if self.use_visual_evidence_router and self.num_classes > 0:
            if self.visual_route_level not in self.name_to_dim:
                raise ValueError(
                    f"Unknown visual route level: {self.visual_route_level}"
                )
            self.visual_evidence_router = CrossViewVisualEvidenceRouter(
                channels=self.name_to_dim[self.visual_route_level],
                num_classes=self.num_classes,
                mode=self.visual_route_mode,
                projection_dim=visual_route_projection_dim,
                topk_ratio=visual_route_topk_ratio,
                temperature=visual_route_temperature,
                shared_axis=visual_route_shared_axis,
                axis_radius=visual_route_axis_radius,
                gate_init=visual_route_gate_init,
                gamma_init=visual_route_gamma_init,
                gamma_max=visual_route_gamma_max,
                reject_temperature=visual_route_reject_temperature,
                dropout=visual_route_dropout,
            )

        self.selective_view_rescue = None
        if self.use_selective_view_rescue and self.num_classes > 0:
            self.selective_view_rescue = SelectiveCrossViewRescueHead(
                self.name_to_dim, self.num_classes,
                projection_dim=selective_rescue_projection_dim,
                hidden_dim=selective_rescue_hidden_dim,
                dropout=selective_rescue_dropout,
                gamma_max=selective_rescue_gamma_max,
                uncertainty_threshold=selective_rescue_uncertainty_threshold,
                gate_temperature=selective_rescue_gate_temperature,
                detach_features=selective_rescue_detach_features,
            )

        self.frozen_anchor_rescue = None
        if self.use_frozen_anchor_rescue and self.num_classes > 0:
            self.frozen_anchor_rescue = FrozenAnchorCrossViewRescueHead(
                self.name_to_dim, self.num_classes,
                projection_dim=frozen_rescue_projection_dim,
                hidden_dim=frozen_rescue_hidden_dim,
                dropout=frozen_rescue_dropout,
                gamma_max=frozen_rescue_gamma_max,
                uncertainty_threshold=frozen_rescue_uncertainty_threshold,
                gate_temperature=frozen_rescue_gate_temperature,
                trust_init=frozen_rescue_trust_init,
            )

        self.frozen_region_rescue = None
        if self.use_frozen_region_rescue and self.num_classes > 0:
            self.frozen_region_rescue = FrozenAnchorRegionEvidenceHead(
                self.name_to_dim, self.num_classes,
                levels=frozen_region_levels,
                projection_dim=frozen_region_projection_dim,
                topk_ratio=frozen_region_topk_ratio,
                temperature=frozen_region_temperature,
                gamma_init=frozen_region_gamma_init,
                gamma_max=frozen_region_gamma_max,
                uncertainty_threshold=frozen_region_uncertainty_threshold,
                gate_temperature=frozen_region_gate_temperature,
                trust_init=frozen_region_trust_init,
            )

        self.frozen_counterfactual_router = None
        if self.use_frozen_counterfactual_router and self.num_classes > 0:
            region_channels = 0
            if self.frozen_counterfactual_region_level:
                if self.frozen_counterfactual_region_level not in self.name_to_dim:
                    raise ValueError(
                        "Unknown regional M8 level: "
                        f"{self.frozen_counterfactual_region_level}"
                    )
                region_channels = self.name_to_dim[
                    self.frozen_counterfactual_region_level
                ]
            self.frozen_counterfactual_router = (
                FrozenAnchorCounterfactualViewRouter(
                    self.num_classes,
                    hidden_dim=frozen_counterfactual_hidden_dim,
                    class_embed_dim=frozen_counterfactual_class_embed_dim,
                    rho_init=frozen_counterfactual_rho_init,
                    rho_max=frozen_counterfactual_rho_max,
                    rescue_init=frozen_counterfactual_rescue_init,
                    delta_clip=frozen_counterfactual_delta_clip,
                    region_channels=region_channels,
                    region_projection_dim=(
                        frozen_counterfactual_region_projection_dim
                    ),
                    region_temperature=frozen_counterfactual_region_temperature,
                )
            )

        self.frozen_region_interaction_moe = None
        if self.use_frozen_region_interaction_moe and self.num_classes > 0:
            if self.fuse_mode != "add":
                raise ValueError("M9 region interaction MoE requires fuse_mode='add'")
            self.frozen_region_interaction_moe = (
                FrozenAnchorRegionInteractionMoE(
                    self.name_to_dim,
                    self.num_classes,
                    levels=frozen_region_interaction_levels,
                    projection_dim=frozen_region_interaction_projection_dim,
                    temperature=frozen_region_interaction_temperature,
                    shared_axis=frozen_region_interaction_shared_axis,
                    axis_radius=frozen_region_interaction_axis_radius,
                    residual_init=frozen_region_interaction_residual_init,
                    residual_max=frozen_region_interaction_residual_max,
                    router_hidden_dim=(
                        frozen_region_interaction_router_hidden_dim
                    ),
                    class_embed_dim=(
                        frozen_region_interaction_class_embed_dim
                    ),
                    rho_init=frozen_region_interaction_rho_init,
                    rho_max=frozen_region_interaction_rho_max,
                    rescue_init=frozen_region_interaction_rescue_init,
                    delta_clip=frozen_region_interaction_delta_clip,
                    include_counterfactual_candidates=(
                        frozen_region_interaction_include_counterfactual
                    ),
                )
            )

        self.plain_innovation_head = None
        innovation_common = {
            "level_dims": self.name_to_dim,
            "num_classes": self.num_classes,
            "projection_dim": plain_innovation_projection_dim,
            "dropout": plain_innovation_dropout,
            "gamma_init": plain_innovation_gamma_init,
            "gamma_max": plain_innovation_gamma_max,
            "warmup_epochs": plain_innovation_warmup_epochs,
            "ramp_epochs": plain_innovation_ramp_epochs,
        }
        if self.plain_innovation_mode == "p9_caprs":
            self.plain_innovation_head = P9CAPRSHead(
                **innovation_common,
                levels=self.plain_innovation_levels,
                topk=plain_innovation_topk,
                temperature=plain_innovation_temperature,
                base_floor=plain_innovation_base_floor,
                use_counterfactual_experts=(
                    plain_innovation_use_counterfactual_experts
                ),
                use_learned_router=plain_innovation_use_learned_router,
                router_variant=plain_innovation_router_variant,
                gamma_trainable=plain_innovation_gamma_trainable,
                gate_init=plain_innovation_gate_init,
                advantage_temperature=(
                    plain_innovation_advantage_temperature
                ),
                gain_margin=plain_innovation_gain_margin,
                score_temperature=plain_innovation_score_temperature,
            )
        elif self.plain_innovation_mode == "p10_wgcr":
            self.plain_innovation_head = P10WGCRHead(
                **innovation_common,
                levels=self.plain_innovation_levels,
                topk=plain_innovation_topk,
                temperature=plain_innovation_temperature,
            )
        elif self.plain_innovation_mode == "p11_otcvr":
            self.plain_innovation_head = P11OTCVRHead(
                **innovation_common,
                levels=self.plain_innovation_levels,
                topk=plain_innovation_topk,
                temperature=plain_innovation_temperature,
                sinkhorn_iters=plain_innovation_sinkhorn_iters,
            )
        elif self.plain_innovation_mode == "p12_berf":
            self.plain_innovation_head = P12BERFHead(
                **innovation_common,
                level=(
                    "C5" if "C5" in self.name_to_dim
                    else next(reversed(self.name_to_dim))
                ),
            )
        elif self.plain_innovation_mode == "p13_vdrm":
            self.plain_innovation_head = P13VDRMHead()
        elif self.plain_innovation_mode == "p14_hcaer":
            self.plain_innovation_head = P14HCAERHead(
                **innovation_common,
                levels=self.plain_innovation_levels,
                topk=plain_innovation_topk,
                temperature=plain_innovation_temperature,
            )
        elif self.plain_innovation_mode == "p15_vtr":
            self.plain_innovation_head = P15VTRHead(
                **innovation_common,
                level=(
                    "C4" if "C4" in self.name_to_dim
                    else next(iter(self.name_to_dim))
                ),
            )
        elif self.plain_innovation_mode == "p16_facgr":
            self.plain_innovation_head = P16FACGRHead(
                **innovation_common,
                levels=self.plain_innovation_levels,
                topk=plain_innovation_topk,
                temperature=plain_innovation_temperature,
                base_floor=plain_innovation_base_floor,
            )
        elif self.plain_innovation_mode == "p17_dcasr":
            self.plain_innovation_head = P17DCASRHead(
                **innovation_common,
                levels=self.plain_innovation_levels,
                topk=plain_innovation_topk,
                temperature=plain_innovation_temperature,
                router_start_epoch=plain_innovation_router_start_epoch,
                router_ramp_epochs=plain_innovation_router_ramp_epochs,
                gate_init=plain_innovation_gate_init,
                trust_threshold=plain_innovation_trust_threshold,
                trust_temperature=plain_innovation_trust_temperature,
                uncertainty_floor=plain_innovation_uncertainty_floor,
                budget_target=plain_innovation_budget_target,
            )
        elif self.plain_innovation_mode == "p18_ewsar":
            self.plain_innovation_head = P18EWSARHead(
                **innovation_common,
                levels=self.plain_innovation_levels,
                topk=plain_innovation_topk,
                temperature=plain_innovation_temperature,
                base_floor=plain_innovation_base_floor,
                router_start_epoch=plain_innovation_router_start_epoch,
                router_ramp_epochs=plain_innovation_router_ramp_epochs,
                advantage_temperature=(
                    plain_innovation_advantage_temperature
                ),
            )
        elif self.plain_innovation_mode == "p19_apcer":
            self.plain_innovation_head = P19APCERHead(
                **innovation_common,
                levels=self.plain_innovation_levels,
                topk=plain_innovation_topk,
                temperature=plain_innovation_temperature,
                base_floor=plain_innovation_base_floor,
                router_start_epoch=plain_innovation_router_start_epoch,
                router_ramp_epochs=plain_innovation_router_ramp_epochs,
                advantage_temperature=(
                    plain_innovation_advantage_temperature
                ),
                corrupt_probability=(
                    plain_innovation_corrupt_probability
                ),
                corrupt_ratio=plain_innovation_corrupt_ratio,
                full_view_drop_probability=(
                    plain_innovation_full_view_drop_probability
                ),
            )
        elif self.plain_innovation_mode == "p20_cvcr":
            self.plain_innovation_head = P20CVCRHead(
                level_dims=self.name_to_dim,
                num_classes=self.num_classes,
                levels=self.plain_innovation_levels,
                projection_dim=plain_innovation_projection_dim,
                topk=plain_innovation_topk,
                temperature=plain_innovation_temperature,
                dropout=plain_innovation_dropout,
                warmup_epochs=plain_innovation_warmup_epochs,
                ramp_epochs=plain_innovation_ramp_epochs,
                complement_margin=plain_innovation_complement_margin,
                complement_temperature=(
                    plain_innovation_complement_temperature
                ),
            )

    def set_plain_innovation_epoch(self, epoch: int) -> None:
        if self.plain_innovation_head is not None:
            self.plain_innovation_head.set_epoch(epoch)

    def _apply_adapter_freeze(self) -> None:
        trainable_prefixes = []
        if self._semantic_base_frozen:
            trainable_prefixes.extend(("semantic_head.", "view_semantic_adapter."))
        if self._spatial_query_base_frozen:
            trainable_prefixes.append("spatial_query_head.")
        if self._frozen_anchor_base_frozen:
            trainable_prefixes.append("frozen_anchor_rescue.")
        if self._frozen_region_base_frozen:
            trainable_prefixes.append("frozen_region_rescue.")
        if self._frozen_counterfactual_base_frozen:
            trainable_prefixes.append("frozen_counterfactual_router.")
        if self._frozen_region_interaction_base_frozen:
            trainable_prefixes.append("frozen_region_interaction_moe.")
        if self._plain_innovation_base_frozen:
            trainable_prefixes.append("plain_innovation_head.")
        if not trainable_prefixes:
            return
        trainable_prefixes = tuple(trainable_prefixes)
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)

    def freeze_base_for_frozen_rescue(self) -> None:
        if self.frozen_anchor_rescue is None:
            raise RuntimeError(
                "frozen rescue freezing requires --use_frozen_anchor_rescue true"
            )
        self._frozen_anchor_base_frozen = True
        self._apply_adapter_freeze()
        self.train(self.training)

    def freeze_base_for_plain_innovation(self) -> None:
        if self.plain_innovation_head is None:
            raise RuntimeError(
                "plain innovation freezing requires an enabled branch"
            )
        self._plain_innovation_base_frozen = True
        self._apply_adapter_freeze()
        self.train(self.training)

    def freeze_plain_innovation_experts_for_router(self) -> None:
        if self.plain_innovation_head is None:
            raise RuntimeError(
                "plain innovation router-only mode requires an enabled branch"
            )
        if not self._plain_innovation_base_frozen:
            raise RuntimeError(
                "plain innovation router-only mode requires a frozen base"
            )
        if not hasattr(self.plain_innovation_head, "set_router_only"):
            raise RuntimeError(
                "selected innovation head does not support router-only training"
            )
        self.plain_innovation_head.set_router_only()
        self.train(self.training)

    def reset_plain_innovation_router(self) -> None:
        if self.plain_innovation_head is None or not hasattr(
            self.plain_innovation_head, "reset_router_parameters"
        ):
            raise RuntimeError("selected innovation head has no resettable router")
        self.plain_innovation_head.reset_router_parameters()

    def transfer_plain_innovation_legacy_router(self, state_dict) -> None:
        if self.plain_innovation_head is None or not hasattr(
            self.plain_innovation_head, "transfer_legacy_router_output"
        ):
            raise RuntimeError("selected innovation head cannot transfer a router")
        prefix = "plain_innovation_head.router.4."
        weight = state_dict.get(prefix + "weight")
        bias = state_dict.get(prefix + "bias")
        if weight is None or bias is None:
            raise KeyError(
                "checkpoint has no legacy P9 router output at "
                f"{prefix}{{weight,bias}}"
            )
        self.plain_innovation_head.transfer_legacy_router_output(weight, bias)

    def freeze_base_for_frozen_region_rescue(self) -> None:
        if self.frozen_region_rescue is None:
            raise RuntimeError(
                "frozen region freezing requires --use_frozen_region_rescue true"
            )
        self._frozen_region_base_frozen = True
        self._apply_adapter_freeze()
        self.train(self.training)

    def freeze_base_for_frozen_counterfactual_router(self) -> None:
        if self.frozen_counterfactual_router is None:
            raise RuntimeError(
                "counterfactual freezing requires "
                "--use_frozen_counterfactual_router true"
            )
        self._frozen_counterfactual_base_frozen = True
        self._apply_adapter_freeze()
        self.train(self.training)

    def freeze_base_for_frozen_region_interaction_moe(self) -> None:
        if self.frozen_region_interaction_moe is None:
            raise RuntimeError(
                "M9 freezing requires "
                "--use_frozen_region_interaction_moe true"
            )
        self._frozen_region_interaction_base_frozen = True
        self._apply_adapter_freeze()
        self.train(self.training)

    def freeze_region_interaction_experts_for_router(self) -> None:
        if self.frozen_region_interaction_moe is None:
            raise RuntimeError("M9 router-only mode requires the M9 branch")
        if not self._frozen_region_interaction_base_frozen:
            raise RuntimeError("M9 router-only mode requires a frozen R2 base")
        self.frozen_region_interaction_moe.set_router_only()
        self.train(self.training)

    def freeze_base_for_semantic(self) -> None:
        """Freeze the visual R2 path while leaving semantic adapters trainable."""
        if self.semantic_head is None:
            raise RuntimeError("sem_freeze_base requires --use_semantic_branch true")

        self._semantic_base_frozen = True
        self._apply_adapter_freeze()
        self.train(self.training)

    def freeze_base_for_spatial_query(self) -> None:
        """Freeze the loaded R2 path and train only the spatial query head."""
        if self.spatial_query_head is None:
            raise RuntimeError(
                "spatial_query_freeze_base requires --use_spatial_query true"
            )
        self._spatial_query_base_frozen = True
        self._apply_adapter_freeze()
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and (
            self._semantic_base_frozen
            or self._spatial_query_base_frozen
            or self._frozen_anchor_base_frozen
            or self._frozen_region_base_frozen
            or self._frozen_counterfactual_base_frozen
            or self._frozen_region_interaction_base_frozen
            or self._plain_innovation_base_frozen
        ):
            # Frozen ResNet batch-normalization statistics are part of the
            # anchored baseline and must not drift during semantic adaptation.
            trainable_children = set()
            if self._semantic_base_frozen:
                trainable_children.update(("semantic_head", "view_semantic_adapter"))
            if self._spatial_query_base_frozen:
                trainable_children.add("spatial_query_head")
            if self._frozen_anchor_base_frozen:
                trainable_children.add("frozen_anchor_rescue")
            if self._frozen_region_base_frozen:
                trainable_children.add("frozen_region_rescue")
            if self._frozen_counterfactual_base_frozen:
                trainable_children.add("frozen_counterfactual_router")
            if self._frozen_region_interaction_base_frozen:
                trainable_children.add("frozen_region_interaction_moe")
            if self._plain_innovation_base_frozen:
                trainable_children.add("plain_innovation_head")
            for name, module in self.named_children():
                if name not in trainable_children:
                    module.eval()
        return self

    # extract the multi-level features of a single view
    def _extract_single_view(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}

        # first check whether this is a native ConvNeXtV2 (explicit downsample_layers + stages)
        if hasattr(self.backbone, 'downsample_layers') and hasattr(self.backbone, 'stages'):
            x = self.backbone.downsample_layers[0](x)
            x = self.backbone.stages[0](x)
            # self.out_indices is normally (1,2,3) -> C3, C4, C5
            for i in range(1, 4):
                x = self.backbone.downsample_layers[i](x)
                x = self.backbone.stages[i](x)
                if i in self.out_indices:
                    feats[self.stage_to_name[i]] = x
        else:
            # generic FeatureList backbone (timm features_only=True)
            feature_list = None
            if hasattr(self.backbone, 'forward_features'):
                y = self.backbone.forward_features(x)
                if isinstance(y, (list, tuple)) and len(y) >= 4:
                    feature_list = y
            if feature_list is None:
                y = self.backbone(x)
                if isinstance(y, (list, tuple)) and len(y) >= 4:
                    feature_list = y
            if feature_list is None:
                raise RuntimeError("backbone did not return a feature list; expected ConvNeXtV2 or a TIMM features_only style.")

            # feature_list indices 0/1/2/3 map to C2/C3/C4/C5; only 1/2/3 -> C3/C4/C5 are used
            for i in self.out_indices:  # (1,2,3)
                feats[self.stage_to_name[i]] = feature_list[i]

        if not feats:
            raise RuntimeError("error: _extract_single_view could not extract any feature from the backbone.")
        return feats
    
    # fuse the two feature streams
    def _fuse_pair(self, xa: torch.Tensor, xb: torch.Tensor, name: str) -> torch.Tensor:
        # note: this method no longer handles the ahcr mode
        if name not in self.fuse_levels: return xa
        if self.fuse_mode in ("add", "cv_gsc_add"): return xa + xb
        elif self.fuse_mode == "mean": return (xa + xb) * 0.5
        elif self.fuse_mode == "max": return torch.maximum(xa, xb)
        elif self.fuse_mode == "concat": return self.fuse_convs[name](torch.cat([xa, xb], dim=1))
        elif self.fuse_mode == "gated": return self.gated_fuses[name](xa, xb)
        elif self.fuse_mode == "xattn":
            if name in self.xattn_fuses: return self.xattn_fuses[name](xa, xb)
            else: return xa
        # note: the ahcr elif branch has been removed
        else: raise ValueError(f"unknown fuse_mode or one that must not be handled here: {self.fuse_mode}")

    def _name_to_stage(self, name: str) -> int:
        for k, v in self.stage_to_name.items():
            if v == name:
                return k
        raise KeyError(name)

    def _view_semantic_vector(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "C5" not in feats:
            raise RuntimeError("sem_view_calib requires single-view C5 features.")
        vec = self.global_pool(feats["C5"]).flatten(1)
        if self.view_semantic_adapter is not None:
            vec = self.view_semantic_adapter(vec)
        return vec

    def _fuse_feature_dict(
        self,
        feats_a: Dict[str, torch.Tensor],
        feats_b: Dict[str, torch.Tensor],
    ):
        if self.fuse_mode == "ahcr":
            return self.ahcr_fuser(feats_a, feats_b), {}

        fused: Dict[str, torch.Tensor] = {}
        dvcre_outputs = []
        iscvf_gates = []
        for name in self.name_to_dim:
            if name not in feats_a or name not in feats_b:
                continue
            fa, fb = feats_a[name], feats_b[name]
            if self.use_cv_gsc and name in self.cv_gsc:
                fa, fb = self.cv_gsc[name](fa, fb)
            if name in self.dvcre_fuses:
                fused[name], region_aux = self.dvcre_fuses[name](fa, fb)
                dvcre_outputs.append(region_aux)
            elif name in self.iscvf_fuses:
                fused[name], gate_mean = self.iscvf_fuses[name](fa, fb)
                iscvf_gates.append(gate_mean)
            else:
                fused[name] = self._fuse_pair(fa, fb, name)

        aux = {}
        if dvcre_outputs:
            aux["dvcre_aux_logits"] = torch.stack(
                [item["aux_logits"] for item in dvcre_outputs], dim=0
            ).mean(dim=0)
            aux["dvcre_gate_a_mean"] = torch.stack(
                [item["gate_a_mean"] for item in dvcre_outputs]
            ).mean()
            aux["dvcre_region_confidence"] = torch.stack(
                [item["region_confidence"] for item in dvcre_outputs]
            ).mean()
            aux["dvcre_residual_scale"] = torch.stack(
                [item["residual_scale"] for item in dvcre_outputs]
            ).mean()
        if iscvf_gates:
            aux["iscvf_gate_a_mean"] = torch.stack(iscvf_gates).mean()
        return fused, aux

    def _aggregate_fused(self, fused: Dict[str, torch.Tensor]) -> torch.Tensor:
        ht = self.head_type.lower()
        if ht == "c5":
            return self.global_pool(fused["C5"]).flatten(1)
        if ht == "fpn":
            vecs = [
                self.global_pool(fused[name]).flatten(1)
                for name in self.fuse_levels
                if name in fused
            ]
            return torch.cat(vecs, dim=1) if len(vecs) > 1 else vecs[0]
        if self.fpn is not None:
            p3, _, _ = self.fpn(fused["C3"], fused["C4"], fused["C5"])
            return self.global_pool(p3).flatten(1)
        if self.fpn_pan is not None:
            n3, _, _ = self.fpn_pan(fused["C3"], fused["C4"], fused["C5"])
            return self.global_pool(n3).flatten(1)
        raise ValueError(f"unknown head_type: {self.head_type}")

    def _intervene_views(self, feats_a, feats_b):
        reference = next(iter(feats_a.values()))
        batch = reference.shape[0]
        device = reference.device
        intervention_type = int(torch.randint(0, 3, (1,), device=device).item())

        if intervention_type == 0:
            keep_prob = 0.5
        else:
            keep_prob = self.iscvf_keep_prob
        base_mask = (
            torch.rand(batch, 1, 8, 8, device=device) < keep_prob
        ).to(dtype=reference.dtype)
        view_selector = (
            torch.rand(batch, 1, 1, 1, device=device) < 0.5
        ).to(dtype=reference.dtype)

        intervened_a = {}
        intervened_b = {}
        for name in feats_a:
            fa, fb = feats_a[name], feats_b[name]
            if intervention_type == 2:
                selector = view_selector.to(dtype=fa.dtype)
                intervened_a[name] = 2.0 * selector * fa
                intervened_b[name] = 2.0 * (1.0 - selector) * fb
                continue

            mask = F.interpolate(
                base_mask.float(),
                size=fa.shape[-2:],
                mode="nearest",
            ).to(dtype=fa.dtype)
            if intervention_type == 0:
                intervened_a[name] = 2.0 * mask * fa
                intervened_b[name] = 2.0 * (1.0 - mask) * fb
            else:
                scale = 1.0 / max(self.iscvf_keep_prob, 1e-6)
                intervened_a[name] = scale * mask * fa
                intervened_b[name] = scale * mask * fb
        return intervened_a, intervened_b, intervention_type

    def forward(self, xa: torch.Tensor, xb: Optional[torch.Tensor] = None):
        feats_a = self._extract_single_view(xa)
        feats_b = self._extract_single_view(xb) if xb is not None else feats_a

        fused, fusion_aux = self._fuse_feature_dict(feats_a, feats_b)
        x = self._aggregate_fused(fused)

        logits_base = self.classifier(x)
        logits_sem = None
        semantic_aux = None
        if self.semantic_head is not None and (self.training or not self.sem_aux_only):
            return_semantic_aux = (
                self.sem_trust_router
                or self.sem_rank_calibration
                or self.sem_error_calibration
            )
            if self.sem_view_calib:
                x_a_sem = self._view_semantic_vector(feats_a)
                x_b_sem = self._view_semantic_vector(feats_b)
                semantic_out = self.semantic_head.forward_view_aware(
                    x,
                    x_a_sem,
                    x_b_sem,
                    logits_base=logits_base,
                    return_aux=return_semantic_aux,
                )
            else:
                semantic_out = self.semantic_head(
                    x,
                    logits_base=logits_base,
                    return_aux=return_semantic_aux,
                )
            if return_semantic_aux:
                logits_sem, semantic_aux = semantic_out
            else:
                logits_sem = semantic_out
            logits = logits_base if self.sem_aux_only else logits_base + logits_sem
        else:
            logits = logits_base

        spatial_query_aux = None
        if self.spatial_query_head is not None:
            spatial_query_aux = self.spatial_query_head(
                feats_a,
                feats_b,
                logits_base=logits_base,
            )
            logits = logits + spatial_query_aux["correction"]

        visual_route_aux = None
        if self.visual_evidence_router is not None:
            visual_route_aux = self.visual_evidence_router(
                feats_a[self.visual_route_level],
                feats_b[self.visual_route_level],
                logits_base=logits_base,
            )
            logits = logits + visual_route_aux["correction"]

        plain_innovation_aux = None
        should_run_plain_innovation = (
            self.plain_innovation_head is not None
            and (
                self.training
                or self.plain_innovation_mode not in (
                    "p13_vdrm", "p20_cvcr"
                )
            )
        )
        if should_run_plain_innovation:
            plain_features_a = feats_a
            plain_features_b = feats_b
            plain_counterfactual_logits = None
            if self.plain_innovation_mode == "p19_apcer":
                plain_features_a, plain_features_b = (
                    self.plain_innovation_head.prepare_training_views(
                        feats_a, feats_b
                    )
                )
                if self.training:
                    counterfactual_fused, _ = self._fuse_feature_dict(
                        plain_features_a, plain_features_b
                    )
                    plain_counterfactual_logits = self.classifier(
                        self._aggregate_fused(counterfactual_fused)
                    )
                else:
                    plain_counterfactual_logits = logits_base
            plain_logits_a = None
            plain_logits_b = None
            if self.plain_innovation_needs_single_logits:
                single_fused_a, _ = self._fuse_feature_dict(
                    plain_features_a, plain_features_a
                )
                single_fused_b, _ = self._fuse_feature_dict(
                    plain_features_b, plain_features_b
                )
                plain_logits_a = self.classifier(
                    self._aggregate_fused(single_fused_a)
                )
                plain_logits_b = self.classifier(
                    self._aggregate_fused(single_fused_b)
                )
            plain_innovation_kwargs = {
                "logits_a": plain_logits_a,
                "logits_b": plain_logits_b,
            }
            if self.plain_innovation_mode in ("p18_ewsar", "p19_apcer"):
                plain_innovation_kwargs["logits_counterfactual"] = (
                    plain_counterfactual_logits
                )
            plain_innovation_aux = self.plain_innovation_head(
                plain_features_a,
                plain_features_b,
                logits_base,
                **plain_innovation_kwargs,
            )
            logits = logits + plain_innovation_aux["correction"]

        selective_rescue_aux = None
        if self.selective_view_rescue is not None:
            selective_rescue_aux = self.selective_view_rescue(
                feats_a, feats_b, logits_base
            )
            logits = logits + selective_rescue_aux["correction"]

        frozen_rescue_aux = None
        if self.frozen_anchor_rescue is not None:
            frozen_rescue_aux = self.frozen_anchor_rescue(
                feats_a, feats_b, logits_base
            )
            if not self.frozen_rescue_aux_only:
                logits = logits + frozen_rescue_aux["correction"]

        frozen_region_aux = None
        if self.frozen_region_rescue is not None:
            frozen_region_aux = self.frozen_region_rescue(
                feats_a, feats_b, logits_base
            )
            if not self.frozen_region_aux_only:
                logits = logits + frozen_region_aux["correction"]

        logits_counterfactual_a = None
        logits_counterfactual_b = None
        needs_counterfactual_logits = (
            self.frozen_counterfactual_router is not None
            or (
                self.frozen_region_interaction_moe is not None
                and self.frozen_region_interaction_moe
                .include_counterfactual_candidates
            )
        )
        if needs_counterfactual_logits:
            counterfactual_a, _ = self._fuse_feature_dict(feats_a, feats_a)
            counterfactual_b, _ = self._fuse_feature_dict(feats_b, feats_b)
            logits_counterfactual_a = self.classifier(
                self._aggregate_fused(counterfactual_a)
            )
            logits_counterfactual_b = self.classifier(
                self._aggregate_fused(counterfactual_b)
            )

        frozen_counterfactual_aux = None
        if self.frozen_counterfactual_router is not None:
            region_level = self.frozen_counterfactual_region_level
            frozen_counterfactual_aux = self.frozen_counterfactual_router(
                logits_base,
                logits_counterfactual_a,
                logits_counterfactual_b,
                features_a=feats_a.get(region_level) if region_level else None,
                features_b=feats_b.get(region_level) if region_level else None,
            )
            if not self.frozen_counterfactual_aux_only:
                logits = logits + frozen_counterfactual_aux["correction"]

        frozen_region_interaction_aux = None
        if self.frozen_region_interaction_moe is not None:
            interaction_features = (
                self.frozen_region_interaction_moe.build_expert_residuals(
                    feats_a, feats_b
                )
            )
            expert_logits = []
            for expert in self.frozen_region_interaction_moe.EXPERT_NAMES:
                candidate_fused = dict(fused)
                for level, residual in interaction_features["residuals"][
                    expert
                ].items():
                    candidate_fused[level] = fused[level] + residual
                candidate_vector = self._aggregate_fused(candidate_fused)
                expert_logits.append(self.classifier(candidate_vector))
            expert_logits = torch.stack(expert_logits, dim=1)
            counterfactual_logits = None
            if (
                self.frozen_region_interaction_moe
                .include_counterfactual_candidates
            ):
                counterfactual_logits = torch.stack((
                    logits_counterfactual_a,
                    logits_counterfactual_b,
                ), dim=1)
            frozen_region_interaction_aux = (
                self.frozen_region_interaction_moe.route(
                    logits_base,
                    expert_logits,
                    counterfactual_logits=counterfactual_logits,
                )
            )
            frozen_region_interaction_aux.update({
                "expert_names": (
                    self.frozen_region_interaction_moe.EXPERT_NAMES
                ),
                "alignment_loss": interaction_features["alignment_loss"],
                "diversity_loss": interaction_features["diversity_loss"],
                "region_gate": interaction_features["region_gate"],
                "residual_norm": interaction_features["residual_norm"],
                "residual_scale": interaction_features["residual_scale"],
            })
            if not self.frozen_region_interaction_aux_only:
                logits = logits + frozen_region_interaction_aux["correction"]

        iscvf_intervention_logits = None
        iscvf_intervention_type = None
        if self.use_iscvf and self.training:
            intervened_a, intervened_b, iscvf_intervention_type = (
                self._intervene_views(feats_a, feats_b)
            )
            intervention_fused, _ = self._fuse_feature_dict(
                intervened_a, intervened_b
            )
            intervention_vector = self._aggregate_fused(intervention_fused)
            iscvf_intervention_logits = self.classifier(intervention_vector)

        # The shared single-view heads are deep supervision only. Evaluation
        # and deployment bypass them entirely, so R2 inference is unchanged.
        view_evidence_logits_a = None
        view_evidence_logits_b = None
        if self.view_evidence_head is not None and self.training:
            view_evidence_logits_a = self.view_evidence_head(feats_a)
            view_evidence_logits_b = self.view_evidence_head(feats_b)

        if not self.return_intermediate:
            return logits
        extra = {"A": feats_a, "B": feats_b, "fused": fused}
        return {
            "logits": logits,
            "logits_base": logits_base,
            "logits_sem": logits_sem,
            "semantic_aux": semantic_aux,
            "spatial_query_logits": (
                spatial_query_aux["query_logits"]
                if spatial_query_aux is not None
                else None
            ),
            "spatial_query_correction": (
                spatial_query_aux["correction"]
                if spatial_query_aux is not None
                else None
            ),
            "spatial_query_aux": spatial_query_aux,
            "view_evidence_logits_a": view_evidence_logits_a,
            "view_evidence_logits_b": view_evidence_logits_b,
            "selective_rescue_aux": selective_rescue_aux,
            "frozen_rescue_aux": frozen_rescue_aux,
            "frozen_region_aux": frozen_region_aux,
            "frozen_counterfactual_aux": frozen_counterfactual_aux,
            "frozen_region_interaction_aux": frozen_region_interaction_aux,
            "visual_route_aux_logits": (
                visual_route_aux["aux_logits"]
                if visual_route_aux is not None
                else None
            ),
            "visual_route_correction": (
                visual_route_aux["correction"]
                if visual_route_aux is not None
                else None
            ),
            "visual_route_cycle_loss": (
                visual_route_aux["cycle_loss"]
                if visual_route_aux is not None
                else None
            ),
            "visual_route_aux": visual_route_aux,
            "plain_innovation_aux": plain_innovation_aux,
            "dvcre_aux_logits": fusion_aux.get("dvcre_aux_logits"),
            "dvcre_gate_a_mean": fusion_aux.get("dvcre_gate_a_mean"),
            "dvcre_region_confidence": fusion_aux.get(
                "dvcre_region_confidence"
            ),
            "dvcre_residual_scale": fusion_aux.get("dvcre_residual_scale"),
            "iscvf_gate_a_mean": fusion_aux.get("iscvf_gate_a_mean"),
            "iscvf_intervention_logits": iscvf_intervention_logits,
            "iscvf_intervention_type": iscvf_intervention_type,
            "feats": extra,
        }
