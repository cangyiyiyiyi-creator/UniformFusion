# engine_finetune.py
# =============== 只用 EMA；兼容多种 batch 结构；评估可写 CSV；autocast 优先 BF16 ===============

import os
import csv
from typing import Tuple, Optional, List

import torch as th
import torch.nn as nn
import torch.nn.functional as F

# 从你的 utils.py 引入日志工具
from utils import MetricLogger, SmoothedValue



# === New: class-level gradient diagnostic ===
import torch as th

def _class_grad_strength(logits: th.Tensor,
                         target: th.Tensor,
                         pos_only: bool = True,
                         alpha: float = 0.75) -> th.Tensor:
    """
    计算每个类别在本 batch 内的“有效梯度强度” G_c（不参与反传，仅统计）。
    多标签 BCE 的一阶梯度对 logit 等于 |sigmoid(z) - y|：
    - 正类：|σ(z)-1| = 1-σ(z)
    - 负类：|σ(z)-0| = σ(z)
    返回: [C] 张量
    """
    with th.no_grad():
        p = th.sigmoid(logits)                         # [B, C]
        pos_mask = (target > 0.5)
        pos_cnt = pos_mask.sum(dim=0).clamp_min(1)
        g_pos = (pos_mask.float() * (1.0 - p)).sum(dim=0) / pos_cnt  # [C]

        if pos_only:
            return g_pos

        neg_mask = (~pos_mask)
        neg_cnt = neg_mask.sum(dim=0).clamp_min(1)
        g_neg = (neg_mask.float() * p).sum(dim=0) / neg_cnt          # [C]
        return alpha * g_pos + (1.0 - alpha) * g_neg


def _grad_var_from(logits: th.Tensor,
                   target: th.Tensor,
                   band: float = 0.0,
                   pos_only: bool = True,
                   alpha: float = 0.75) -> th.Tensor:
    """
    计算类级有效梯度的方差（可加容忍带 band，处于 |diff|<=band 的不计入）。
    返回标量张量（device/logits 同）。
    """
    with th.no_grad():
        g = _class_grad_strength(logits, target, pos_only=pos_only, alpha=alpha)  # [C]
        g_mean = g.mean()
        diff = g - g_mean
        if band > 0.0:
            diff = th.where(diff.abs() > band, diff, th.zeros_like(diff))
        reg = (diff ** 2).mean()
    return reg


# ----------------------------- EMA（可放 CPU） -----------------------------
class SimpleEMA:
    """
    轻量 EMA：维护 '参数名 -> EMA张量' 的字典，可放在 CPU 上节省显存。
    - update(model): 用 model 当前参数进行 EMA 更新
    - copy_to(model): 将 EMA 权重拷到传入模型（常用于验证）
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = "cpu"):
        self.decay = float(decay)
        self.device = th.device(device)
        self.ema_state = {}
        with th.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.ema_state[n] = p.detach().to(self.device, dtype=th.float32).clone()

    @th.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        one_m = 1.0 - d
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in self.ema_state:
                self.ema_state[n] = p.detach().to(self.device, dtype=th.float32).clone()
                continue
            tgt = self.ema_state[n]
            src = p.detach().to(tgt.device, dtype=tgt.dtype)
            tgt.mul_(d).add_(src, alpha=one_m)

    @th.no_grad()
    def copy_to(self, model: nn.Module):
        msd = model.state_dict()
        for n, w in self.ema_state.items():
            if n in msd:
                msd[n].copy_(w.to(msd[n].device, dtype=msd[n].dtype))


# ----------------------------- 辅助函数 -----------------------------
def _unpack_samples(samples, device: th.device) -> Tuple[th.Tensor, Optional[th.Tensor], th.Tensor]:
    """
    兼容 batch 结构：
    1) ((xa, xb), y)
    2) (xa, xb, y) / [xa, xb, y]
    3) (xa, y)（单视角）
    4) dict: {'img_a':..., 'img_b':..., 'target':...} 或 {'images':(xa,xb),'target':...}
    """
    xa = xb = target = None

    if isinstance(samples, (tuple, list)):
        if len(samples) == 2:
            x, target = samples
            if isinstance(x, (tuple, list)) and len(x) == 2:
                xa, xb = x
            else:
                xa = x
        elif len(samples) == 3:
            xa, xb, target = samples
        else:
            raise ValueError(f"Unexpected batch tuple length: {len(samples)}")
    elif isinstance(samples, dict):
        if "images" in samples:
            img = samples["images"]
            if isinstance(img, (tuple, list)) and len(img) == 2:
                xa, xb = img
            else:
                xa = img
        else:
            xa = samples.get("img_a", None)
            xb = samples.get("img_b", None)
        target = samples.get("target", None)
    else:
        raise ValueError(f"Unexpected batch structure: type={type(samples)}")

    if xa is None:
        raise ValueError("xa is None after unpacking.")
    if target is None:
        raise ValueError("target is None after unpacking.")

    xa = xa.to(device, non_blocking=True)
    if xb is not None:
        xb = xb.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    return xa, xb, target


def _extract_logits(out):
    if isinstance(out, (tuple, list)):
        logits = out[0]
    elif isinstance(out, dict) and "logits" in out:
        logits = out["logits"]
    else:
        logits = out
    return logits


def _forward_outputs(model: nn.Module, xa: th.Tensor, xb: Optional[th.Tensor], drop_feats: bool = True):
    out = model(xa, xb) if xb is not None else model(xa)
    if drop_feats and isinstance(out, dict) and "feats" in out:
        out = dict(out)
        out.pop("feats", None)
    return out, _extract_logits(out)


def _forward_logits(model: nn.Module, xa: th.Tensor, xb: Optional[th.Tensor]):
    _, logits = _forward_outputs(model, xa, xb, drop_feats=True)
    return logits


def _semantic_conflict_suppression_loss(
    model_out,
    target: th.Tensor,
    base_margin: float = 0.0,
    sem_margin: float = 0.0,
):
    """
    Target-aware semantic conflict suppression.

    Penalize semantic correction only when:
    - the visual/base branch already leans toward the label direction; and
    - the semantic correction pushes against that label direction.

    This trains semantic gate/gamma to suppress harmful semantic priors without
    changing the original supervised objective when the branch is disabled.
    """
    if not isinstance(model_out, dict):
        return None
    logits_base = model_out.get("logits_base", None)
    logits_sem = model_out.get("logits_sem", None)
    if logits_base is None or logits_sem is None:
        return None

    target_dir = target.mul(2.0).sub(1.0)
    with th.no_grad():
        base_align = target_dir * logits_base
        visual_reliable = (th.sigmoid(base_align - float(base_margin)) - 0.5).clamp_min(0.0) * 2.0

    sem_against_target = F.relu(float(sem_margin) - target_dir * logits_sem)
    return (visual_reliable * sem_against_target).mean()


def _semantic_trust_routing_loss(
    model_out,
    target: th.Tensor,
    temperature: float = 0.01,
    margin: float = 0.0,
    target_mode: str = "soft",
    harm_weight: float = 3.0,
):
    """
    Teach the inference-time router whether a bounded semantic candidate helps.

    The target is derived from the per-label BCE improvement of the candidate.
    Candidate quality is detached, so this auxiliary objective only trains the
    trust prediction rather than encouraging either branch to game its target.
    """
    if not isinstance(model_out, dict):
        return None
    logits_base = model_out.get("logits_base")
    semantic_aux = model_out.get("semantic_aux")
    if logits_base is None or not isinstance(semantic_aux, dict):
        return None

    trust_gate = semantic_aux.get("trust_gate")
    trust_logits = semantic_aux.get("trust_logits")
    trust_candidate = semantic_aux.get("trust_candidate")
    if trust_gate is None or trust_logits is None or trust_candidate is None:
        return None

    temperature = max(float(temperature), 1e-6)
    with th.no_grad():
        base_ref = logits_base.detach().float()
        candidate_ref = trust_candidate.detach().float()
        target_ref = target.detach().float()
        base_bce = F.binary_cross_entropy_with_logits(
            base_ref,
            target_ref,
            reduction="none",
        )
        candidate_bce = F.binary_cross_entropy_with_logits(
            base_ref + candidate_ref,
            target_ref,
            reduction="none",
        )
        advantage = base_bce - candidate_bce
        if str(target_mode).lower() == "asymmetric":
            active = advantage.abs() > float(margin)
            trust_target = (advantage > float(margin)).float()
            harmful = active & (trust_target < 0.5)
            target_weight = active.float()
            target_weight = target_weight * (
                1.0 + (max(float(harm_weight), 1.0) - 1.0) * harmful.float()
            )
        else:
            trust_target = th.sigmoid((advantage - float(margin)) / temperature)
            target_weight = th.tanh(advantage.abs() / temperature)
            active = target_weight > 0

    gate = trust_gate.float()
    element_loss = F.binary_cross_entropy_with_logits(
        trust_logits.float(),
        trust_target,
        reduction="none",
    )
    loss = (element_loss * target_weight).sum() / target_weight.sum().clamp_min(1.0)
    stats = {
        "gate_mean": gate.detach().mean(),
        "help_rate": (
            ((advantage > float(margin)) & active).float().sum()
            / active.float().sum().clamp_min(1.0)
        ),
        "active_rate": active.float().mean(),
    }
    return loss, stats


def _semantic_rank_calibration_loss(
    model_out,
    target: th.Tensor,
    guard_weight: float = 2.0,
    temperature: float = 0.2,
    need_temperature: float = 0.2,
):
    """
    Train raw semantic scores on positive/negative ranking pairs while keeping
    semantic residuals from reducing margins that the visual branch got right.
    """
    if not isinstance(model_out, dict):
        return None
    logits_base = model_out.get("logits_base")
    logits_sem = model_out.get("logits_sem")
    semantic_aux = model_out.get("semantic_aux")
    if (
        logits_base is None
        or logits_sem is None
        or not isinstance(semantic_aux, dict)
    ):
        return None

    semantic_raw = semantic_aux.get("semantic_raw_logits")
    if semantic_raw is None:
        return None

    temperature = max(float(temperature), 1e-6)
    need_temperature = max(float(need_temperature), 1e-6)
    guard_weight = max(float(guard_weight), 0.0)
    class_losses = []
    help_losses = []
    guard_losses = []
    need_rates = []

    base_ref = logits_base.detach().float()
    semantic_raw = semantic_raw.float()
    semantic_correction = logits_sem.float()
    positive = target > 0.5

    for class_idx in range(target.shape[1]):
        pos_mask = positive[:, class_idx]
        neg_mask = ~pos_mask
        if not pos_mask.any() or not neg_mask.any():
            continue

        base_pair = (
            base_ref[pos_mask, class_idx].unsqueeze(1)
            - base_ref[neg_mask, class_idx].unsqueeze(0)
        )
        raw_pair = (
            semantic_raw[pos_mask, class_idx].unsqueeze(1)
            - semantic_raw[neg_mask, class_idx].unsqueeze(0)
        )
        correction_pair = (
            semantic_correction[pos_mask, class_idx].unsqueeze(1)
            - semantic_correction[neg_mask, class_idx].unsqueeze(0)
        )

        # Misranked or low-margin visual pairs receive stronger semantic help.
        need = th.sigmoid(-base_pair / need_temperature)
        help_loss = (
            need
            * F.softplus(-raw_pair / temperature)
            * temperature
        ).mean()

        # For already-correct visual pairs, semantic residuals may increase the
        # margin but are penalized when they shrink it.
        visual_safe = th.sigmoid(base_pair / need_temperature)
        guard_loss = (visual_safe * F.relu(-correction_pair)).mean()

        class_losses.append(help_loss + guard_weight * guard_loss)
        help_losses.append(help_loss.detach())
        guard_losses.append(guard_loss.detach())
        need_rates.append(need.detach().mean())

    if not class_losses:
        return None

    loss = th.stack(class_losses).mean()
    stats = {
        "help_loss": th.stack(help_losses).mean(),
        "guard_loss": th.stack(guard_losses).mean(),
        "need_rate": th.stack(need_rates).mean(),
    }
    return loss, stats


def _semantic_error_correction_loss(
    model_out,
    target: th.Tensor,
    error_power: float = 2.0,
    min_weight: float = 0.05,
    guard_weight: float = 4.0,
):
    """Train semantic evidence on labels the anchored visual branch finds hard."""
    if not isinstance(model_out, dict):
        return None
    logits_base = model_out.get("logits_base")
    semantic_aux = model_out.get("semantic_aux")
    if logits_base is None or not isinstance(semantic_aux, dict):
        return None

    semantic_raw = semantic_aux.get("semantic_raw_logits")
    semantic_correction = semantic_aux.get("semantic_correction")
    if semantic_raw is None or semantic_correction is None:
        return None

    error_power = max(float(error_power), 0.0)
    min_weight = min(max(float(min_weight), 0.0), 1.0)
    guard_weight = max(float(guard_weight), 0.0)
    target_float = target.float()

    with th.no_grad():
        base_ref = logits_base.detach().float()
        base_prob = th.sigmoid(base_ref)
        base_error = (target_float - base_prob).abs()
        need_weight = min_weight + (1.0 - min_weight) * base_error.pow(error_power)
        target_dir = target_float.mul(2.0).sub(1.0)
        base_alignment = target_dir * base_ref
        base_reliability = th.sigmoid(base_alignment)

    raw_bce = F.binary_cross_entropy_with_logits(
        semantic_raw.float(),
        target_float,
        reduction="none",
    )
    help_loss = (raw_bce * need_weight).sum() / need_weight.sum().clamp_min(1.0)

    correction = semantic_correction.float()
    harmful = F.relu(-target_dir * correction)
    guard_loss = (base_reliability * harmful).mean()
    loss = help_loss + guard_weight * guard_loss
    stats = {
        "help_loss": help_loss.detach(),
        "guard_loss": guard_loss.detach(),
        "need_weight": need_weight.detach().mean(),
        "harm_rate": ((target_dir * correction) < 0).float().detach().mean(),
    }
    return loss, stats


def _class_balanced_binary_loss(
    logits: th.Tensor,
    target: th.Tensor,
    negative_weight: float = 1.0,
):
    """Average positives and negatives per class before averaging classes."""
    logits = logits.float()
    target = target.float()
    element_loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    positive = target > 0.5
    negative = ~positive

    pos_count = positive.sum(dim=0)
    neg_count = negative.sum(dim=0)
    pos_per_class = (
        element_loss * positive.float()
    ).sum(dim=0) / pos_count.clamp_min(1)
    neg_per_class = (
        element_loss * negative.float()
    ).sum(dim=0) / neg_count.clamp_min(1)

    valid_pos = pos_count > 0
    valid_neg = neg_count > 0
    pos_loss = (
        pos_per_class[valid_pos].mean()
        if valid_pos.any()
        else element_loss.new_tensor(0.0)
    )
    neg_loss = (
        neg_per_class[valid_neg].mean()
        if valid_neg.any()
        else element_loss.new_tensor(0.0)
    )
    negative_weight = max(float(negative_weight), 0.0)
    normalizer = 1.0 + negative_weight
    loss = (pos_loss + negative_weight * neg_loss) / normalizer
    return loss, pos_loss, neg_loss


def _spatial_attribute_query_loss(
    model_out,
    target: th.Tensor,
    final_weight: float = 0.25,
    guard_weight: float = 1.0,
    negative_weight: float = 1.0,
):
    """Train local attribute evidence while preserving reliable R2 decisions."""
    if not isinstance(model_out, dict):
        return None
    query_logits = model_out.get("spatial_query_logits")
    correction = model_out.get("spatial_query_correction")
    logits_base = model_out.get("logits_base")
    final_logits = model_out.get("logits")
    if (
        query_logits is None
        or correction is None
        or logits_base is None
        or final_logits is None
    ):
        return None

    query_loss, pos_loss, neg_loss = _class_balanced_binary_loss(
        query_logits,
        target,
        negative_weight=negative_weight,
    )
    final_loss, _, _ = _class_balanced_binary_loss(
        final_logits,
        target,
        negative_weight=negative_weight,
    )

    target_dir = target.float().mul(2.0).sub(1.0)
    with th.no_grad():
        base_alignment = target_dir * logits_base.detach().float()
        reliable = (
            th.sigmoid(base_alignment).sub(0.5).clamp_min(0.0) * 2.0
        )
    harmful = F.relu(-target_dir * correction.float())
    guard_loss = (reliable * harmful).sum() / reliable.sum().clamp_min(1.0)

    final_weight = max(float(final_weight), 0.0)
    guard_weight = max(float(guard_weight), 0.0)
    loss = query_loss + final_weight * final_loss + guard_weight * guard_loss

    spatial_aux = model_out.get("spatial_query_aux")
    gamma_mean = loss.detach().new_tensor(0.0)
    view_a_weight = loss.detach().new_tensor(0.5)
    if isinstance(spatial_aux, dict):
        gamma = spatial_aux.get("gamma")
        view_weights = spatial_aux.get("view_weights")
        if gamma is not None:
            gamma_mean = gamma.detach().float().mean()
        if view_weights is not None:
            view_a_weight = view_weights.detach().float()[:, 0].mean()

    stats = {
        "query_loss": query_loss.detach(),
        "positive_loss": pos_loss.detach(),
        "negative_loss": neg_loss.detach(),
        "final_loss": final_loss.detach(),
        "guard_loss": guard_loss.detach(),
        "gamma_mean": gamma_mean,
        "view_a_weight": view_a_weight,
    }
    return loss, stats


def _cross_view_best_evidence_loss(
    model_out,
    target: th.Tensor,
    epoch: int,
    warmup_epochs: int = 5,
    ramp_epochs: int = 5,
    advantage_temperature: float = 0.1,
    negative_weight: float = 1.0,
):
    """Distill only label-aligned evidence that is better than paired R2."""
    if not isinstance(model_out, dict):
        return None
    logits_a = model_out.get("view_evidence_logits_a")
    logits_b = model_out.get("view_evidence_logits_b")
    paired_logits = model_out.get("logits")
    if logits_a is None or logits_b is None or paired_logits is None:
        return None

    loss_a, _, _ = _class_balanced_binary_loss(
        logits_a,
        target,
        negative_weight=negative_weight,
    )
    loss_b, _, _ = _class_balanced_binary_loss(
        logits_b,
        target,
        negative_weight=negative_weight,
    )
    view_loss = 0.5 * (loss_a + loss_b)

    advantage_temperature = max(float(advantage_temperature), 1e-6)
    target_float = target.float()
    target_dir = target_float.mul(2.0).sub(1.0)
    with th.no_grad():
        logits_a_ref = logits_a.detach().float()
        logits_b_ref = logits_b.detach().float()
        paired_ref = paired_logits.detach().float()

        # Positive labels retain the strongest visible evidence; negatives
        # retain the least suspicious view. Labels are used only in training.
        teacher_logits = th.where(
            target_float > 0.5,
            th.maximum(logits_a_ref, logits_b_ref),
            th.minimum(logits_a_ref, logits_b_ref),
        )
        paired_label_loss = F.binary_cross_entropy_with_logits(
            paired_ref,
            target_float,
            reduction="none",
        )
        teacher_label_loss = F.binary_cross_entropy_with_logits(
            teacher_logits,
            target_float,
            reduction="none",
        )
        advantage = (paired_label_loss - teacher_label_loss).clamp_min(0.0)
        disagreement = (
            th.sigmoid(logits_a_ref) - th.sigmoid(logits_b_ref)
        ).abs()
        teacher_reliability = th.sigmoid(target_dir * teacher_logits)
        confidence = th.tanh(advantage / advantage_temperature)
        distill_weight = (
            confidence
            * (0.25 + 0.75 * disagreement)
            * teacher_reliability
        )
        teacher_prob = th.sigmoid(teacher_logits)

        teacher_better = advantage > 0
        paired_wrong = target_dir * paired_ref <= 0
        teacher_correct = target_dir * teacher_logits > 0
        rescue = paired_wrong & teacher_correct

    distill_element = F.binary_cross_entropy_with_logits(
        paired_logits.float(),
        teacher_prob,
        reduction="none",
    )
    distill_loss = (
        distill_element * distill_weight
    ).sum() / distill_weight.sum().clamp_min(1.0)

    warmup_epochs = max(int(warmup_epochs), 0)
    ramp_epochs = max(int(ramp_epochs), 1)
    if int(epoch) < warmup_epochs:
        ramp = 0.0
    else:
        ramp = min(1.0, (int(epoch) - warmup_epochs + 1) / ramp_epochs)

    stats = {
        "view_loss_a": loss_a.detach(),
        "view_loss_b": loss_b.detach(),
        "distill_ramp": paired_logits.detach().new_tensor(ramp),
        "teacher_win_rate": teacher_better.float().mean(),
        "active_weight": distill_weight.mean(),
        "advantage": advantage.mean(),
        "rescue_rate": rescue.float().mean(),
    }
    return view_loss, distill_loss, stats


def _dvcre_auxiliary_loss(model_out, target: th.Tensor):
    if not isinstance(model_out, dict):
        return None
    aux_logits = model_out.get("dvcre_aux_logits")
    if aux_logits is None:
        return None
    loss, positive, negative = _class_balanced_binary_loss(aux_logits, target)
    stats = {
        "positive_loss": positive.detach(),
        "negative_loss": negative.detach(),
        "gate_a_mean": aux_logits.detach().new_tensor(0.5),
        "region_confidence": aux_logits.detach().new_tensor(0.0),
        "residual_scale": aux_logits.detach().new_tensor(0.0),
    }
    for output_key, stat_key in (
        ("dvcre_gate_a_mean", "gate_a_mean"),
        ("dvcre_region_confidence", "region_confidence"),
        ("dvcre_residual_scale", "residual_scale"),
    ):
        value = model_out.get(output_key)
        if value is not None:
            stats[stat_key] = value.detach().float().mean()
    return loss, stats


def _selective_view_rescue_loss(model_out, target: th.Tensor):
    if not isinstance(model_out, dict):
        return None
    aux = model_out.get("selective_rescue_aux")
    base = model_out.get("logits_base")
    if not isinstance(aux, dict) or base is None:
        return None
    target = target.float()
    loss_a, _, _ = _class_balanced_binary_loss(aux["logits_a"], target)
    loss_b, _, _ = _class_balanced_binary_loss(aux["logits_b"], target)
    aux_loss = 0.5 * (loss_a + loss_b)
    direction = target.mul(2.0).sub(1.0)
    base_ref = base.detach().float()
    correction = aux["correction"].float()
    need = th.sigmoid(-direction * base_ref / 0.5) * aux["gate"].detach().float()
    element = F.binary_cross_entropy_with_logits(base_ref + correction, target, reduction="none")
    rescue_loss = (element * need).sum() / need.sum().clamp_min(1.0)
    reliable = th.sigmoid(direction * base_ref / 0.5)
    guard_loss = (reliable * F.relu(-direction * correction)).sum() / reliable.sum().clamp_min(1.0)
    return aux_loss, rescue_loss, guard_loss, {
        "gate": aux["gate"].detach().mean(),
        "gamma": aux["gamma"].detach().mean(),
    }


def _frozen_anchor_rescue_loss(
    model_out, target: th.Tensor, rank_temperature: float = 0.2
):
    if not isinstance(model_out, dict):
        return None
    aux = model_out.get("frozen_rescue_aux")
    base = model_out.get("logits_base")
    if not isinstance(aux, dict) or base is None:
        return None
    target = target.float()
    base_ref = base.detach().float()
    correction = aux["correction"].float()
    corrected = base_ref + correction
    loss_a, _, _ = _class_balanced_binary_loss(aux["logits_a"], target)
    loss_b, _, _ = _class_balanced_binary_loss(aux["logits_b"], target)
    view_loss = 0.5 * (loss_a + loss_b)

    with th.no_grad():
        base_error = F.binary_cross_entropy_with_logits(
            base_ref, target, reduction="none"
        )
        candidate_error = F.binary_cross_entropy_with_logits(
            aux["candidate_logits"].detach().float(), target, reduction="none"
        )
        trust_target = th.sigmoid((base_error - candidate_error) / 0.1)
        trust_weight = 0.25 + 0.75 * aux["uncertainty"].detach().float()
    trust_element = F.binary_cross_entropy_with_logits(
        aux["trust_logits"].float(), trust_target, reduction="none"
    )
    trust_loss = (trust_element * trust_weight).sum() / trust_weight.sum().clamp_min(1.0)

    temperature = max(float(rank_temperature), 1e-6)
    rank_terms = []
    for class_index in range(target.shape[1]):
        positive = target[:, class_index] > 0.5
        negative = ~positive
        if not positive.any() or not negative.any():
            continue
        base_margin = (
            base_ref[positive, class_index].unsqueeze(1)
            - base_ref[negative, class_index].unsqueeze(0)
        )
        corrected_margin = (
            corrected[positive, class_index].unsqueeze(1)
            - corrected[negative, class_index].unsqueeze(0)
        )
        need = th.sigmoid((0.5 - base_margin) / 0.5)
        rank_terms.append(
            (F.softplus(-corrected_margin / temperature) * need).sum()
            / need.sum().clamp_min(1.0)
        )
    rank_loss = (
        th.stack(rank_terms).mean()
        if rank_terms
        else correction.sum() * 0.0
    )

    negative = (target < 0.5).float()
    reliable_negative = negative * th.sigmoid(-base_ref)
    guard_loss = (
        reliable_negative * correction
    ).sum() / reliable_negative.sum().clamp_min(1.0)
    return view_loss, trust_loss, rank_loss, guard_loss, {
        "gate": aux["gate"].detach().mean(),
        "trust": aux["trust_gate"].detach().mean(),
        "gamma": aux["gamma"].detach().mean(),
        "correction": correction.detach().mean(),
    }


def _frozen_region_rescue_loss(
    model_out,
    target: th.Tensor,
    rank_temperature: float = 0.1,
    target_gain: float = 0.005,
):
    if not isinstance(model_out, dict):
        return None
    aux = model_out.get("frozen_region_aux")
    base = model_out.get("logits_base")
    if not isinstance(aux, dict) or base is None:
        return None
    target = target.float()
    base_ref = base.detach().float()
    correction = aux["correction"].float()
    corrected = base_ref + correction
    losses = [
        _class_balanced_binary_loss(logits, target)[0]
        for logits in (
            aux["logits_a"], aux["logits_b"], aux["candidate_logits"]
        )
    ]
    view_loss = th.stack(losses).mean()
    with th.no_grad():
        base_error = F.binary_cross_entropy_with_logits(
            base_ref, target, reduction="none"
        )
        candidate_error = F.binary_cross_entropy_with_logits(
            aux["candidate_logits"].detach().float(), target, reduction="none"
        )
        trust_target = th.sigmoid((base_error - candidate_error) / 0.1)
        trust_weight = 0.25 + 0.75 * aux["uncertainty"].detach().float()
    trust_element = F.binary_cross_entropy_with_logits(
        aux["trust_logits"].float(), trust_target, reduction="none"
    )
    trust_loss = (trust_element * trust_weight).sum() / trust_weight.sum().clamp_min(1.0)

    temperature = max(float(rank_temperature), 1e-6)
    rank_terms = []
    for class_index in range(target.shape[1]):
        positive = target[:, class_index] > 0.5
        negative = ~positive
        if not positive.any() or not negative.any():
            continue
        base_margin = (
            base_ref[positive, class_index].unsqueeze(1)
            - base_ref[negative, class_index].unsqueeze(0)
        )
        corrected_margin = (
            corrected[positive, class_index].unsqueeze(1)
            - corrected[negative, class_index].unsqueeze(0)
        )
        relative_gain = corrected_margin - base_margin
        hard_weight = th.sigmoid((1.0 - base_margin) / 1.0).clamp_min(0.05)
        pair_loss = F.softplus(
            (float(target_gain) - relative_gain) / temperature
        )
        rank_terms.append(
            (pair_loss * hard_weight).sum() / hard_weight.sum().clamp_min(1.0)
        )
    rank_loss = (
        th.stack(rank_terms).mean()
        if rank_terms
        else correction.sum() * 0.0
    )
    reliable_negative = (target < 0.5).float() * th.sigmoid(-base_ref)
    guard_loss = (
        reliable_negative * correction
    ).sum() / reliable_negative.sum().clamp_min(1.0)
    return view_loss, trust_loss, rank_loss, guard_loss, {
        "gate": aux["gate"].detach().mean(),
        "trust": aux["trust_gate"].detach().mean(),
        "gamma": aux["gamma"].detach().mean(),
        "correction": correction.detach().mean(),
        "region_confidence": aux["region_confidence"].detach().mean(),
    }


def _balanced_element_mean(element: th.Tensor, target: th.Tensor):
    positive = target > 0.5
    negative = ~positive
    positive_count = positive.sum(dim=0)
    negative_count = negative.sum(dim=0)
    positive_mean = (
        element * positive.float()
    ).sum(dim=0) / positive_count.clamp_min(1)
    negative_mean = (
        element * negative.float()
    ).sum(dim=0) / negative_count.clamp_min(1)
    parts = []
    if (positive_count > 0).any():
        parts.append(positive_mean[positive_count > 0].mean())
    if (negative_count > 0).any():
        parts.append(negative_mean[negative_count > 0].mean())
    return th.stack(parts).mean() if parts else element.mean()


def _frozen_counterfactual_router_loss(
    model_out,
    target: th.Tensor,
    queue_owner=None,
    router_temperature: float = 0.02,
    router_margin: float = 0.02,
    rank_temperature: float = 0.2,
    rank_margin: float = 0.5,
    hard_threshold: float = 2.0,
    guard_threshold: float = 4.0,
    queue_size: int = 128,
):
    if not isinstance(model_out, dict):
        return None
    aux = model_out.get("frozen_counterfactual_aux")
    if not isinstance(aux, dict):
        return None

    target = target.float()
    paired = aux["paired_logits"].float()
    logits_a = aux["logits_a"].float()
    logits_b = aux["logits_b"].float()
    routed = aux["routed_logits"].float()
    correction = aux["correction"].float()
    router_temperature = max(float(router_temperature), 1e-6)

    with th.no_grad():
        paired_error = F.binary_cross_entropy_with_logits(
            paired, target, reduction="none"
        )
        error_a = F.binary_cross_entropy_with_logits(
            logits_a, target, reduction="none"
        )
        error_b = F.binary_cross_entropy_with_logits(
            logits_b, target, reduction="none"
        )
        view_errors = th.stack((error_a, error_b), dim=-1)
        best_view_error = view_errors.min(dim=-1).values
        gain = paired_error - best_view_error
        rescue_target = th.sigmoid(
            (gain - float(router_margin)) / router_temperature
        )
        view_target = th.softmax(
            -view_errors / router_temperature, dim=-1
        )

    rescue_element = F.binary_cross_entropy_with_logits(
        aux["rescue_logits"].float(), rescue_target, reduction="none"
    )
    view_element = -(
        view_target * F.log_softmax(aux["view_logits"].float(), dim=-1)
    ).sum(dim=-1)
    route_loss = _balanced_element_mean(
        rescue_element + rescue_target * view_element,
        target,
    )

    corrected_error = F.binary_cross_entropy_with_logits(
        routed, target, reduction="none"
    )
    anchor_weight = th.sigmoid(
        (float(router_margin) - gain) / router_temperature
    )
    element_guard = _balanced_element_mean(
        anchor_weight * F.relu(corrected_error - paired_error),
        target,
    )

    rank_temperature = max(float(rank_temperature), 1e-6)
    rank_terms = []
    pair_guard_terms = []

    def add_pairs(base_positive, base_negative, routed_positive, routed_negative):
        if base_positive.numel() == 0 or base_negative.numel() == 0:
            return
        base_margin = base_positive.unsqueeze(1) - base_negative.unsqueeze(0)
        routed_margin = (
            routed_positive.unsqueeze(1) - routed_negative.unsqueeze(0)
        )
        hard_weight = th.sigmoid(
            (float(hard_threshold) - base_margin) / rank_temperature
        ).detach()
        rank_element = F.softplus(
            (float(rank_margin) - routed_margin) / rank_temperature
        ) * rank_temperature
        rank_terms.append(
            (rank_element * hard_weight).sum()
            / hard_weight.sum().clamp_min(1.0)
        )
        easy_weight = th.sigmoid(
            (base_margin - float(guard_threshold)) / rank_temperature
        ).detach()
        pair_guard_terms.append(
            (F.relu(base_margin - routed_margin) * easy_weight).sum()
            / easy_weight.sum().clamp_min(1.0)
        )

    queue_size = max(int(queue_size), 0)
    queue = None
    if queue_owner is not None and queue_size > 0:
        queue = getattr(queue_owner, "_frozen_counterfactual_rank_queue", None)
        queue_meta = (
            target.shape[1], str(target.device), queue_size
        )
        if not isinstance(queue, dict) or queue.get("meta") != queue_meta:
            empty = lambda: target.new_empty(0)
            queue = {
                "meta": queue_meta,
                "classes": [
                    {
                        "positive_base": empty(),
                        "positive_routed": empty(),
                        "negative_base": empty(),
                        "negative_routed": empty(),
                    }
                    for _ in range(target.shape[1])
                ],
            }
            setattr(queue_owner, "_frozen_counterfactual_rank_queue", queue)

    paired_ref = paired.detach()
    for class_index in range(target.shape[1]):
        positive = target[:, class_index] > 0.5
        negative = ~positive
        base_positive = paired_ref[positive, class_index]
        base_negative = paired_ref[negative, class_index]
        routed_positive = routed[positive, class_index]
        routed_negative = routed[negative, class_index]
        add_pairs(
            base_positive, base_negative, routed_positive, routed_negative
        )

        if queue is None:
            continue
        class_queue = queue["classes"][class_index]
        add_pairs(
            base_positive,
            class_queue["negative_base"],
            routed_positive,
            class_queue["negative_routed"],
        )
        add_pairs(
            class_queue["positive_base"],
            base_negative,
            class_queue["positive_routed"],
            routed_negative,
        )
        for prefix, mask in (("positive", positive), ("negative", negative)):
            new_base = paired_ref[mask, class_index]
            new_routed = routed.detach()[mask, class_index]
            class_queue[f"{prefix}_base"] = th.cat((
                class_queue[f"{prefix}_base"], new_base
            ))[-queue_size:].detach()
            class_queue[f"{prefix}_routed"] = th.cat((
                class_queue[f"{prefix}_routed"], new_routed
            ))[-queue_size:].detach()

    rank_loss = (
        th.stack(rank_terms).mean()
        if rank_terms
        else correction.sum() * 0.0
    )
    pair_guard = (
        th.stack(pair_guard_terms).mean()
        if pair_guard_terms
        else correction.sum() * 0.0
    )
    guard_loss = element_guard + pair_guard
    residual_loss = correction.abs().mean()
    return route_loss, rank_loss, guard_loss, residual_loss, {
        "rescue_gate": aux["rescue_gate"].detach().mean(),
        "rho": aux["rho"].detach().mean(),
        "view_a_weight": aux["view_weights"][..., 0].detach().mean(),
        "help_rate": (gain > float(router_margin)).float().mean(),
        "target_rate": rescue_target.mean(),
        "correction": correction.detach().abs().mean(),
        "pair_groups": correction.new_tensor(float(len(rank_terms))),
    }


def _frozen_region_interaction_moe_loss(
    model_out,
    target: th.Tensor,
    queue_owner=None,
    expert_temperature: float = 0.1,
    router_temperature: float = 0.03,
    router_margin: float = 0.01,
    rank_temperature: float = 0.2,
    rank_margin: float = 0.5,
    hard_threshold: float = 2.0,
    guard_threshold: float = 4.0,
    queue_size: int = 128,
):
    """Train M9 experts first, then learn a guarded class-wise router."""
    if not isinstance(model_out, dict):
        return None
    aux = model_out.get("frozen_region_interaction_aux")
    if not isinstance(aux, dict):
        return None

    target = target.float()
    anchor = aux["anchor_logits"].float()
    expert_logits = aux["expert_logits"].float()
    route_candidate_logits = aux.get(
        "route_candidate_logits", expert_logits
    ).float()
    routed = aux["routed_logits"].float()
    correction = aux["correction"].float()
    expanded_target = target.unsqueeze(1).expand_as(expert_logits)
    expert_errors = F.binary_cross_entropy_with_logits(
        expert_logits, expanded_target, reduction="none"
    )
    errors_by_class = expert_errors.permute(0, 2, 1)

    expert_temperature = max(float(expert_temperature), 1e-6)
    with th.no_grad():
        specialist_target = th.softmax(
            -errors_by_class.detach() / expert_temperature, dim=-1
        )
    specialist_error = (specialist_target * errors_by_class).sum(dim=-1)
    coverage_error = errors_by_class.mean(dim=-1)
    expert_loss = _balanced_element_mean(
        0.75 * specialist_error + 0.25 * coverage_error,
        target,
    )
    alignment_loss = aux.get("alignment_loss", expert_loss * 0.0)
    diversity_loss = aux.get("diversity_loss", expert_loss * 0.0)

    router_temperature = max(float(router_temperature), 1e-6)
    with th.no_grad():
        anchor_error = F.binary_cross_entropy_with_logits(
            anchor, target, reduction="none"
        )
        route_targets = target.unsqueeze(1).expand_as(route_candidate_logits)
        route_errors = F.binary_cross_entropy_with_logits(
            route_candidate_logits, route_targets, reduction="none"
        ).permute(0, 2, 1)
        detached_errors = route_errors.detach()
        best_expert_error, best_expert = detached_errors.min(dim=-1)
        gain = anchor_error - best_expert_error
        rescue_target = th.sigmoid(
            (gain - float(router_margin)) / router_temperature
        )
        expert_target = th.softmax(
            -detached_errors / router_temperature, dim=-1
        )

    rescue_element = F.binary_cross_entropy_with_logits(
        aux["rescue_logits"].float(), rescue_target, reduction="none"
    )
    selection_element = -(
        expert_target
        * F.log_softmax(aux["expert_weight_logits"].float(), dim=-1)
    ).sum(dim=-1)
    route_loss = _balanced_element_mean(
        rescue_element + rescue_target * selection_element,
        target,
    )

    routed_error = F.binary_cross_entropy_with_logits(
        routed, target, reduction="none"
    )
    anchor_weight = th.sigmoid(
        (float(router_margin) - gain) / router_temperature
    ).detach()
    element_guard = _balanced_element_mean(
        anchor_weight * F.relu(routed_error - anchor_error), target
    )

    rank_temperature = max(float(rank_temperature), 1e-6)
    rank_terms = []
    pair_guard_terms = []

    def add_pairs(base_positive, base_negative, routed_positive, routed_negative):
        if base_positive.numel() == 0 or base_negative.numel() == 0:
            return
        base_margin = base_positive.unsqueeze(1) - base_negative.unsqueeze(0)
        routed_margin = (
            routed_positive.unsqueeze(1) - routed_negative.unsqueeze(0)
        )
        hard_weight = th.sigmoid(
            (float(hard_threshold) - base_margin) / rank_temperature
        ).detach()
        rank_element = F.softplus(
            (float(rank_margin) - routed_margin) / rank_temperature
        ) * rank_temperature
        rank_terms.append(
            (rank_element * hard_weight).sum()
            / hard_weight.sum().clamp_min(1.0)
        )
        easy_weight = th.sigmoid(
            (base_margin - float(guard_threshold)) / rank_temperature
        ).detach()
        pair_guard_terms.append(
            (F.relu(base_margin - routed_margin) * easy_weight).sum()
            / easy_weight.sum().clamp_min(1.0)
        )

    queue_size = max(int(queue_size), 0)
    queue = None
    if queue_owner is not None and queue_size > 0:
        queue = getattr(queue_owner, "_m9_region_interaction_rank_queue", None)
        queue_meta = (target.shape[1], str(target.device), queue_size)
        if not isinstance(queue, dict) or queue.get("meta") != queue_meta:
            empty = lambda: target.new_empty(0)
            queue = {
                "meta": queue_meta,
                "classes": [
                    {
                        "positive_base": empty(),
                        "positive_routed": empty(),
                        "negative_base": empty(),
                        "negative_routed": empty(),
                    }
                    for _ in range(target.shape[1])
                ],
            }
            setattr(queue_owner, "_m9_region_interaction_rank_queue", queue)

    anchor_ref = anchor.detach()
    for class_index in range(target.shape[1]):
        positive = target[:, class_index] > 0.5
        negative = ~positive
        base_positive = anchor_ref[positive, class_index]
        base_negative = anchor_ref[negative, class_index]
        routed_positive = routed[positive, class_index]
        routed_negative = routed[negative, class_index]
        add_pairs(base_positive, base_negative, routed_positive, routed_negative)
        if queue is None:
            continue
        class_queue = queue["classes"][class_index]
        add_pairs(
            base_positive,
            class_queue["negative_base"],
            routed_positive,
            class_queue["negative_routed"],
        )
        add_pairs(
            class_queue["positive_base"],
            base_negative,
            class_queue["positive_routed"],
            routed_negative,
        )
        for prefix, mask in (("positive", positive), ("negative", negative)):
            class_queue[f"{prefix}_base"] = th.cat((
                class_queue[f"{prefix}_base"],
                anchor_ref[mask, class_index],
            ))[-queue_size:].detach()
            class_queue[f"{prefix}_routed"] = th.cat((
                class_queue[f"{prefix}_routed"],
                routed.detach()[mask, class_index],
            ))[-queue_size:].detach()

    zero = correction.sum() * 0.0
    rank_loss = th.stack(rank_terms).mean() if rank_terms else zero
    pair_guard = (
        th.stack(pair_guard_terms).mean() if pair_guard_terms else zero
    )
    guard_loss = element_guard + pair_guard
    residual_loss = _balanced_element_mean(correction.abs(), target)
    selected_expert = aux["expert_weights"].detach().argmax(dim=-1)
    return (
        expert_loss,
        alignment_loss,
        diversity_loss,
        route_loss,
        rank_loss,
        guard_loss,
        residual_loss,
        {
            "rescue_gate": aux["rescue_gate"].detach().mean(),
            "rho": aux["rho"].detach().mean(),
            "help_rate": (gain > float(router_margin)).float().mean(),
            "target_rate": rescue_target.mean(),
            "expert_match": (selected_expert == best_expert).float().mean(),
            "correction": correction.detach().abs().mean(),
            "pair_groups": correction.new_tensor(float(len(rank_terms))),
            "region_gate": aux["region_gate"].detach(),
            "residual_norm": aux["residual_norm"].detach(),
            "residual_scale": aux["residual_scale"].detach(),
        },
    )


def _visual_evidence_routing_loss(
    model_out,
    target: th.Tensor,
    negative_weight: float = 1.0,
):
    """Supervise local visual evidence while anchoring corrections to R2."""
    if not isinstance(model_out, dict):
        return None
    aux_logits = model_out.get("visual_route_aux_logits")
    correction = model_out.get("visual_route_correction")
    logits_base = model_out.get("logits_base")
    if aux_logits is None or correction is None or logits_base is None:
        return None

    target_float = target.float()
    aux_loss, positive_loss, negative_loss = _class_balanced_binary_loss(
        aux_logits,
        target_float,
        negative_weight=negative_weight,
    )

    target_direction = target_float.mul(2.0).sub(1.0)
    with th.no_grad():
        base_ref = logits_base.detach().float()
        base_alignment = target_direction * base_ref
        need = th.sigmoid(-base_alignment / 0.5)
        reliable = (
            th.sigmoid(base_alignment).sub(0.5).clamp_min(0.0) * 2.0
        )

    anchored_logits = base_ref + correction.float()
    rescue_element = F.binary_cross_entropy_with_logits(
        anchored_logits,
        target_float,
        reduction="none",
    )
    positive = target_float > 0.5
    negative = ~positive
    positive_weight = need * positive.float()
    negative_element_weight = need * negative.float()
    rescue_positive = (
        rescue_element * positive_weight
    ).sum(dim=0) / positive_weight.sum(dim=0).clamp_min(1.0)
    rescue_negative = (
        rescue_element * negative_element_weight
    ).sum(dim=0) / negative_element_weight.sum(dim=0).clamp_min(1.0)
    valid_positive = positive_weight.sum(dim=0) > 0
    valid_negative = negative_element_weight.sum(dim=0) > 0
    rescue_positive = (
        rescue_positive[valid_positive].mean()
        if valid_positive.any()
        else rescue_element.new_tensor(0.0)
    )
    rescue_negative = (
        rescue_negative[valid_negative].mean()
        if valid_negative.any()
        else rescue_element.new_tensor(0.0)
    )
    negative_weight = max(float(negative_weight), 0.0)
    rescue_loss = (
        rescue_positive + negative_weight * rescue_negative
    ) / (1.0 + negative_weight)

    harmful = F.relu(-target_direction * correction.float())
    guard_loss = (
        reliable * harmful
    ).sum() / reliable.sum().clamp_min(1.0)
    cycle_loss = model_out.get("visual_route_cycle_loss")
    if cycle_loss is None:
        cycle_loss = correction.new_tensor(0.0)

    route_aux = model_out.get("visual_route_aux")
    stats = {
        "positive_loss": positive_loss.detach(),
        "negative_loss": negative_loss.detach(),
        "need": need.mean(),
        "harm_rate": (
            (target_direction * correction.float()) < 0
        ).float().mean(),
        "view_a_weight": correction.detach().new_tensor(0.5),
        "trust": correction.detach().new_tensor(0.0),
        "rejection": correction.detach().new_tensor(1.0),
        "agreement": correction.detach().new_tensor(0.0),
        "confidence": correction.detach().new_tensor(0.0),
        "gamma": correction.detach().new_tensor(0.0),
    }
    if isinstance(route_aux, dict):
        for key in (
            "view_a_weight",
            "trust",
            "rejection",
            "agreement",
            "confidence",
            "gamma",
        ):
            value = route_aux.get(key)
            if value is not None:
                stats[key] = value.detach().float().mean()
    return aux_loss, rescue_loss, guard_loss, cycle_loss, stats


def _plain_bce_batch_rank_loss(scores: th.Tensor, target: th.Tensor):
    """Hard positive-negative ranking loss used by the P17 evidence heads."""
    scores = scores.float()
    target = target.float()
    losses = []
    for class_index in range(scores.shape[1]):
        positive = scores[target[:, class_index] > 0.5, class_index]
        negative = scores[target[:, class_index] <= 0.5, class_index]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        keep_positive = min(int(positive.numel()), 8)
        keep_negative = min(int(negative.numel()), 8)
        hard_positive = positive.topk(
            keep_positive, largest=False
        ).values
        hard_negative = negative.topk(
            keep_negative, largest=True
        ).values
        difference = hard_positive.unsqueeze(1) - hard_negative.unsqueeze(0)
        losses.append(F.softplus(0.5 - difference).mean())
    if not losses:
        return scores.sum() * 0.0
    return th.stack(losses).mean()


def _plain_bce_innovation_loss(model_out, target: th.Tensor):
    """Unified objectives for independently switched P9-P20 branches."""
    if not isinstance(model_out, dict):
        return None
    aux = model_out.get("plain_innovation_aux")
    logits_base = model_out.get("logits_base")
    if not isinstance(aux, dict) or logits_base is None:
        return None

    target = target.float()
    base = logits_base.float()
    zero = base.sum() * 0.0
    mode = str(aux.get("mode", ""))
    direction = target.mul(2.0).sub(1.0)

    aux_loss = zero
    aux_logits = aux.get("aux_logits")
    if aux_logits is not None:
        if mode == "p17_dcasr" and aux.get("level_aux_logits") is not None:
            level_aux_logits = aux["level_aux_logits"]
            aux_loss = th.stack([
                _class_balanced_binary_loss(
                    level_aux_logits[..., level_index], target
                )[0]
                for level_index in range(level_aux_logits.shape[-1])
            ]).mean()
        elif mode == "p14_hcaer":
            element = F.binary_cross_entropy_with_logits(
                aux_logits.float(), target, reduction="none"
            )
            need = th.sigmoid(-direction * base.detach() / 0.5)
            aux_loss = (element * need).sum() / need.sum().clamp_min(1.0)
        else:
            aux_loss = _class_balanced_binary_loss(
                aux_logits, target
            )[0]

    route_loss = zero
    route_active = zero
    expert_logits = aux.get("expert_logits")
    router_logits = aux.get("router_logits")
    if mode == "p17_dcasr" and router_logits is not None:
        route_probe_logits = aux.get("route_probe_logits")
        route_stage = aux.get("route_stage")
        if route_stage is None:
            route_stage = zero.new_tensor(1.0)
        if route_probe_logits is not None:
            with th.no_grad():
                base_element = F.binary_cross_entropy_with_logits(
                    base.detach(), target, reduction="none"
                )
                probe_element = F.binary_cross_entropy_with_logits(
                    route_probe_logits.detach().float(),
                    target,
                    reduction="none",
                )
                gain = base_element - probe_element
                gain_target = (gain > 0.0).float()
                magnitude_scale = gain.abs().mean().clamp_min(1e-7)
                magnitude_weight = (
                    gain.abs() / magnitude_scale
                ).clamp(0.25, 4.0)
                positive_fraction = gain_target.mean(dim=0).clamp(
                    0.05, 0.95
                )
                balance = th.where(
                    gain_target > 0.5,
                    0.5 / positive_fraction.unsqueeze(0),
                    0.5 / (1.0 - positive_fraction).unsqueeze(0),
                )
                route_weight = magnitude_weight * balance
            route_element = F.binary_cross_entropy_with_logits(
                router_logits.float(), gain_target, reduction="none"
            )
            route_loss = (
                route_element * route_weight
            ).sum() / route_weight.sum().clamp_min(1.0)
            mismatch_logits = aux.get("mismatch_router_logits")
            if mismatch_logits is not None:
                route_loss = route_loss + 0.10 * (
                    F.binary_cross_entropy_with_logits(
                        mismatch_logits.float(),
                        th.zeros_like(mismatch_logits, dtype=th.float32),
                    )
                )
            route_loss = route_stage.float() * route_loss
            route_active = gain_target.mean()
    elif (
        mode in ("p18_ewsar", "p19_apcer")
        and expert_logits is not None
        and router_logits is not None
    ):
        expanded_target = target.unsqueeze(-1).expand_as(expert_logits)
        expert_element = F.binary_cross_entropy_with_logits(
            expert_logits.float(), expanded_target, reduction="none"
        )
        with th.no_grad():
            base_element = expert_element[..., 0]
            positive_gain = F.relu(
                base_element.unsqueeze(-1) - expert_element[..., 1:]
            )
            temperature = aux.get("advantage_temperature")
            if temperature is None:
                temperature = zero.new_tensor(0.05)
            temperature = temperature.float().clamp_min(1e-4)
            best_gain = positive_gain.max(dim=-1).values
            non_base_mass = 1.0 - th.exp(-best_gain / temperature)
            non_base_distribution = F.softmax(
                positive_gain / temperature, dim=-1
            )
            route_target = th.cat(
                [
                    (1.0 - non_base_mass).unsqueeze(-1),
                    non_base_mass.unsqueeze(-1) * non_base_distribution,
                ],
                dim=-1,
            )
            route_weight = 1.0 + non_base_mass
        route_element = -(
            route_target * F.log_softmax(router_logits.float(), dim=-1)
        ).sum(dim=-1)
        route_loss = (
            route_element * route_weight
        ).sum() / route_weight.sum().clamp_min(1.0)
        route_stage = aux.get("route_stage")
        if route_stage is not None:
            route_loss = route_stage.float() * route_loss
        route_active = non_base_mass.mean()
    elif expert_logits is not None and router_logits is not None:
        expanded_target = target.unsqueeze(-1).expand_as(expert_logits)
        expert_element = F.binary_cross_entropy_with_logits(
            expert_logits.float(), expanded_target, reduction="none"
        )
        with th.no_grad():
            best_loss, best_index = expert_element.min(dim=-1)
            base_loss = expert_element[..., 0]
            meaningful = (base_loss - best_loss) > 0.01
            best_index = th.where(
                meaningful, best_index, th.zeros_like(best_index)
            )
        route_loss = F.cross_entropy(
            router_logits.float().reshape(-1, router_logits.shape[-1]),
            best_index.reshape(-1),
        )
        route_active = (best_index != 0).float().mean()

    correction = aux.get(
        "guard_correction", aux.get("correction_candidate")
    )
    guard_loss = zero
    correction_abs = zero
    harm_rate = zero
    if correction is not None:
        correction = correction.float()
        reliability = (
            th.sigmoid(direction * base.detach()).sub(0.5).clamp_min(0.0)
            * 2.0
        )
        harmful = F.relu(-direction * correction)
        guard_loss = (
            reliability * harmful
        ).sum() / reliability.sum().clamp_min(1.0)
        correction_abs = correction.detach().abs().mean()
        harm_rate = ((direction * correction.detach()) < 0).float().mean()

    consistency_loss = zero
    consistency = aux.get("consistency")
    if consistency is not None:
        consistency_loss = consistency.float().mean()

    rank_loss = zero
    rank_logits = aux.get("rank_logits")
    if rank_logits is not None:
        rank_loss = _plain_bce_batch_rank_loss(rank_logits, target)

    budget_loss = zero
    if mode == "p17_dcasr" and aux.get("gate") is not None:
        route_stage = aux.get("route_stage")
        if route_stage is None:
            route_stage = zero.new_tensor(1.0)
        budget_target = aux.get("budget_target")
        if budget_target is None:
            budget_target = zero.new_tensor(0.12)
        budget_loss = route_stage.float() * (
            aux["gate"].float().mean() - budget_target.float()
        ).square()

    single_loss = zero
    regret_loss = zero
    logits_a = aux.get("single_logits_a")
    logits_b = aux.get("single_logits_b")
    if logits_a is not None and logits_b is not None:
        single_loss = 0.5 * (
            _class_balanced_binary_loss(logits_a, target)[0]
            + _class_balanced_binary_loss(logits_b, target)[0]
        )
        base_element = F.binary_cross_entropy_with_logits(
            base, target, reduction="none"
        )
        best_single = th.minimum(
            F.binary_cross_entropy_with_logits(
                logits_a.float(), target, reduction="none"
            ),
            F.binary_cross_entropy_with_logits(
                logits_b.float(), target, reduction="none"
            ),
        )
        if mode == "p20_cvcr":
            complementarity = aux.get("complementarity")
            if complementarity is None:
                complementarity = th.ones_like(base_element)
            complementarity = complementarity.detach().float().clamp(0.0, 1.0)
            margin = aux.get("complement_margin")
            if margin is None:
                margin = zero.new_tensor(0.02)
            temperature = aux.get("complement_temperature")
            if temperature is None:
                temperature = zero.new_tensor(0.05)
            temperature = temperature.float().clamp_min(1e-4)
            weight = 0.25 + 0.75 * complementarity
            violation = (
                base_element
                - best_single.detach()
                + margin.float() * complementarity
            )
            regret_element = temperature * F.softplus(
                violation / temperature
            )
            regret_loss = (
                regret_element * weight
            ).sum() / weight.sum().clamp_min(1.0)
            route_active = complementarity.mean()
        else:
            regret_loss = F.relu(
                base_element - best_single.detach()
            ).mean()

    evidence_loss = zero
    pair_aux_logits = aux.get("pair_aux_logits")
    if mode in ("p18_ewsar", "p19_apcer") and pair_aux_logits is not None:
        evidence_loss = _class_balanced_binary_loss(
            pair_aux_logits.float(), target
        )[0]
    evidence_probabilities = aux.get("evidence_probabilities")
    if evidence_probabilities is not None:
        evidence_probabilities = evidence_probabilities.float().clamp(
            1e-6, 1.0 - 1e-6
        )
        evidence_target = target.unsqueeze(-1).expand_as(
            evidence_probabilities
        )
        probability_loss = F.binary_cross_entropy_with_logits(
            th.logit(evidence_probabilities),
            evidence_target,
        )
        strength = aux.get("evidence_strength")
        evidence_penalty = zero
        if strength is not None:
            error = (
                evidence_probabilities.detach() - evidence_target
            ).abs()
            evidence_penalty = (
                strength.float() * error
            ).mean() * 0.01
        evidence_loss = probability_loss + evidence_penalty

    match_loss = zero
    match_cost = aux.get("match_cost")
    if match_cost is not None:
        positive = target > 0.5
        match_loss = (
            match_cost.float() * positive.float()
        ).sum() / positive.float().sum().clamp_min(1.0)

    if mode == "p20_cvcr":
        loss_stage = aux.get("loss_stage")
        if loss_stage is None:
            loss_stage = zero.new_tensor(1.0)
        loss_stage = loss_stage.float()
        aux_loss = loss_stage * aux_loss
        consistency_loss = loss_stage * consistency_loss
        regret_loss = loss_stage * regret_loss
        single_loss = loss_stage * single_loss
        rank_loss = loss_stage * rank_loss

    gate = aux.get("gate")
    confidence = aux.get("confidence")
    gamma = aux.get("gamma")
    ramp = aux.get("ramp")
    return {
        "aux": aux_loss,
        "route": route_loss,
        "guard": guard_loss,
        "consistency": consistency_loss,
        "regret": regret_loss,
        "evidence": evidence_loss,
        "match": match_loss,
        "single": single_loss,
        "rank": rank_loss,
        "budget": budget_loss,
        "gate": gate.detach().float().mean() if gate is not None else zero,
        "confidence": (
            confidence.detach().float().mean()
            if confidence is not None else zero
        ),
        "gamma": gamma.detach().float().mean() if gamma is not None else zero,
        "ramp": ramp.detach().float().mean() if ramp is not None else zero,
        "correction": correction_abs,
        "harm": harm_rate,
        "route_active": route_active,
    }


def _iscvf_intervention_loss(
    model_out,
    target: th.Tensor,
    epoch: int,
    warmup_epochs: int = 5,
    ramp_epochs: int = 5,
):
    if not isinstance(model_out, dict):
        return None
    clean_logits = model_out.get("logits")
    intervention_logits = model_out.get("iscvf_intervention_logits")
    if clean_logits is None or intervention_logits is None:
        return None

    supervised, _, _ = _class_balanced_binary_loss(
        intervention_logits, target
    )
    with th.no_grad():
        clean_prob = th.sigmoid(clean_logits.detach().float())
        confidence = (clean_prob - 0.5).abs().mul(2.0)
    consistency_element = F.binary_cross_entropy_with_logits(
        intervention_logits.float(),
        clean_prob,
        reduction="none",
    )
    consistency = (
        consistency_element * confidence
    ).sum() / confidence.sum().clamp_min(1.0)

    warmup_epochs = max(int(warmup_epochs), 0)
    ramp_epochs = max(int(ramp_epochs), 1)
    if int(epoch) < warmup_epochs:
        ramp = 0.0
    else:
        ramp = min(1.0, (int(epoch) - warmup_epochs + 1) / ramp_epochs)

    intervention_type = model_out.get("iscvf_intervention_type")
    type_value = -1.0 if intervention_type is None else float(intervention_type)
    gate_mean = model_out.get("iscvf_gate_a_mean")
    if gate_mean is None:
        gate_mean = intervention_logits.detach().new_tensor(0.5)
    stats = {
        "consistency_ramp": intervention_logits.detach().new_tensor(ramp),
        "confidence": confidence.mean(),
        "gate_a_mean": gate_mean.detach().float().mean(),
        "intervention_type": intervention_logits.detach().new_tensor(type_value),
    }
    return supervised, consistency, stats


def _micro_accuracy(logits: th.Tensor, target: th.Tensor, thresh: float = 0.5):
    with th.no_grad():
        prob = th.sigmoid(logits)
        pred = (prob > thresh).to(target.dtype)
        acc = (pred == target).float().mean()
    return acc


def _f1_scores(logits: th.Tensor, target: th.Tensor, thresh: float = 0.5):
    with th.no_grad():
        prob = th.sigmoid(logits)
        pred = (prob > thresh).to(th.int)
        y = target.to(th.int)

        tp = (pred & y).sum(dim=0).float()
        fp = (pred & (1 - y)).sum(dim=0).float()
        fn = ((1 - pred) & y).sum(dim=0).float()

        denom = (2 * tp + fp + fn).clamp_min(1e-12)
        f1_c = (2 * tp) / denom

        TP = tp.sum(); FP = fp.sum(); FN = fn.sum()
        f1_micro = (2 * TP) / (2 * TP + FP + FN + 1e-12)
        f1_macro = f1_c.mean()
    return f1_micro.item(), f1_macro.item()


def _average_precision_score(scores: th.Tensor, targets: th.Tensor) -> float:
    s = scores.detach().cpu().float()
    y = targets.detach().cpu().float()
    if y.sum() == 0:
        return 0.0
    order = th.argsort(s, descending=True)
    y = y[order]
    tp = y; fp = 1.0 - y
    tp_cum = th.cumsum(tp, dim=0)
    fp_cum = th.cumsum(fp, dim=0)
    recalls = tp_cum / (y.sum() + 1e-12)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-12)
    ap = 0.0; prev_r = 0.0
    for r, p in zip(recalls.tolist(), precisions.tolist()):
        ap += p * max(r - prev_r, 0.0); prev_r = r
    return float(ap)


def _evaluate_multilabel_ap(all_logits: th.Tensor,
                            all_targets: th.Tensor,
                            class_names: Optional[List[str]] = None):
    prob = th.sigmoid(all_logits)
    C = prob.shape[1]
    ap_per_class = []
    for c in range(C):
        ap = _average_precision_score(prob[:, c], all_targets[:, c])
        ap_per_class.append(ap)

    print("---- Per-class AP ----")
    for idx, ap in enumerate(ap_per_class):
        name = class_names[idx] if (class_names and idx < len(class_names)) else f"C{idx}"
        print(f"{name:>18}: AP={ap:.4f}")
    mAP = float(sum(ap_per_class) / max(len(ap_per_class), 1))
    print(f"mAP={mAP:.4f}")
    return ap_per_class, mAP


# ----------------------------- 训练 / 验证 -----------------------------
def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader,
    optimizer: th.optim.Optimizer,
    device: th.device,
    epoch: int,
    drop_path_rate: float = 0.0,  # 只占位，保持签名兼容
    amp: bool = True,
    model_ema: Optional[SimpleEMA] = None,
    print_freq: int = 50,
):
    model.train()
    model_for_schedule = model.module if hasattr(model, "module") else model
    if hasattr(model_for_schedule, "set_plain_innovation_epoch"):
        model_for_schedule.set_plain_innovation_epoch(epoch)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("grad_var", SmoothedValue(window_size=20, fmt="{value:.6f}"))
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    scaler = th.amp.GradScaler('cuda', enabled=amp)

    # 优先 BF16；不可用再回落 FP16
    try:
        bf16_ok = th.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    ac_dtype = th.bfloat16 if bf16_ok else th.float16

    header = f"Epoch: [{epoch}]"
    for samples in metric_logger.log_every(data_loader, print_freq, header):
        xa, xb, target = _unpack_samples(samples, device)
        if target.dtype != th.float32:
            target = target.float()

        with th.autocast(device_type='cuda', dtype=ac_dtype, enabled=amp):
            model_out, logits = _forward_outputs(model, xa, xb, drop_feats=True)

            # --- 【CVPA/GSPF 修复版损失计算逻辑】---
            # 关键原则：
            # 1) loss_sup 只用于日志，必须绕开 GSPFRegularizedCriterion / DistillationLoss 外壳，
            #    避免提前 pop/reset CVPA/GSPF cache。
            # 2) total_loss 必须调用完整 criterion(...)，这样 CVPA/GSPF 正则才能加入，
            #    并且每个 batch 后 cache 会被清空，防止显存持续上涨。
            is_distill = False
            try:
                is_distill = 'student_inputs' in criterion.forward.__code__.co_varnames
            except Exception:
                is_distill = False

            plain_criterion = criterion
            if is_distill and hasattr(plain_criterion, 'base_criterion'):
                plain_criterion = plain_criterion.base_criterion
            while hasattr(plain_criterion, 'base_criterion'):
                plain_criterion = plain_criterion.base_criterion

            # 只计算基础监督损失，作为日志中的 loss_sup
            loss_sup = plain_criterion(logits, target)

            if is_distill:
                total_loss = criterion(
                    student_outputs=logits,
                    student_inputs=(xa, xb),
                    targets=target
                )
            else:
                # 非蒸馏模式：必须调用完整 criterion，不能直接用 criterion.base_criterion。
                # 这里会触发 GSPFRegularizedCriterion.forward()，从而加入 CVPA/GSPF 正则并清空缓存。
                total_loss = criterion(logits, target)
            sem_conflict_loss = logits.new_tensor(0.0)
            sem_conflict_lambda = float(getattr(criterion, 'sem_conflict_lambda', 0.0))
            if sem_conflict_lambda > 0:
                loss_obj = _semantic_conflict_suppression_loss(
                    model_out=model_out,
                    target=target,
                    base_margin=float(getattr(criterion, 'sem_conflict_base_margin', 0.0)),
                    sem_margin=float(getattr(criterion, 'sem_conflict_sem_margin', 0.0)),
                )
                if loss_obj is not None:
                    sem_conflict_loss = loss_obj
                    total_loss = total_loss + sem_conflict_lambda * sem_conflict_loss

            sem_trust_loss = logits.new_tensor(0.0)
            sem_trust_gate_mean = logits.new_tensor(0.0)
            sem_trust_help_rate = logits.new_tensor(0.0)
            sem_trust_active_rate = logits.new_tensor(0.0)
            sem_trust_lambda = float(getattr(criterion, 'sem_trust_lambda', 0.0))
            if sem_trust_lambda > 0:
                trust_obj = _semantic_trust_routing_loss(
                    model_out=model_out,
                    target=target,
                    temperature=float(getattr(criterion, 'sem_trust_temperature', 0.01)),
                    margin=float(getattr(criterion, 'sem_trust_margin', 0.0)),
                    target_mode=str(getattr(criterion, 'sem_trust_target_mode', 'soft')),
                    harm_weight=float(getattr(criterion, 'sem_trust_harm_weight', 3.0)),
                )
                if trust_obj is not None:
                    sem_trust_loss, trust_stats = trust_obj
                    sem_trust_gate_mean = trust_stats["gate_mean"]
                    sem_trust_help_rate = trust_stats["help_rate"]
                    sem_trust_active_rate = trust_stats["active_rate"]
                    total_loss = total_loss + sem_trust_lambda * sem_trust_loss

            sem_rank_loss = logits.new_tensor(0.0)
            sem_rank_help = logits.new_tensor(0.0)
            sem_rank_guard = logits.new_tensor(0.0)
            sem_rank_need = logits.new_tensor(0.0)
            sem_rank_lambda = float(getattr(criterion, 'sem_rank_lambda', 0.0))
            if sem_rank_lambda > 0:
                rank_obj = _semantic_rank_calibration_loss(
                    model_out=model_out,
                    target=target,
                    guard_weight=float(getattr(criterion, 'sem_rank_guard_weight', 2.0)),
                    temperature=float(getattr(criterion, 'sem_rank_temperature', 0.2)),
                    need_temperature=float(getattr(criterion, 'sem_rank_need_temperature', 0.2)),
                )
                if rank_obj is not None:
                    sem_rank_loss, rank_stats = rank_obj
                    sem_rank_help = rank_stats["help_loss"]
                    sem_rank_guard = rank_stats["guard_loss"]
                    sem_rank_need = rank_stats["need_rate"]
                    total_loss = total_loss + sem_rank_lambda * sem_rank_loss

            sem_error_loss = logits.new_tensor(0.0)
            sem_error_help = logits.new_tensor(0.0)
            sem_error_guard = logits.new_tensor(0.0)
            sem_error_need = logits.new_tensor(0.0)
            sem_error_harm = logits.new_tensor(0.0)
            sem_error_lambda = float(getattr(criterion, 'sem_error_lambda', 0.0))
            if sem_error_lambda > 0:
                error_obj = _semantic_error_correction_loss(
                    model_out=model_out,
                    target=target,
                    error_power=float(getattr(criterion, 'sem_error_power', 2.0)),
                    min_weight=float(getattr(criterion, 'sem_error_min_weight', 0.05)),
                    guard_weight=float(getattr(criterion, 'sem_error_guard_weight', 4.0)),
                )
                if error_obj is not None:
                    sem_error_loss, error_stats = error_obj
                    sem_error_help = error_stats["help_loss"]
                    sem_error_guard = error_stats["guard_loss"]
                    sem_error_need = error_stats["need_weight"]
                    sem_error_harm = error_stats["harm_rate"]
                    total_loss = total_loss + sem_error_lambda * sem_error_loss

            spatial_query_loss = logits.new_tensor(0.0)
            spatial_query_positive = logits.new_tensor(0.0)
            spatial_query_negative = logits.new_tensor(0.0)
            spatial_query_final = logits.new_tensor(0.0)
            spatial_query_guard = logits.new_tensor(0.0)
            spatial_query_gamma = logits.new_tensor(0.0)
            spatial_query_view_a = logits.new_tensor(0.5)
            spatial_query_lambda = float(
                getattr(criterion, 'spatial_query_lambda', 0.0)
            )
            if spatial_query_lambda > 0:
                spatial_obj = _spatial_attribute_query_loss(
                    model_out=model_out,
                    target=target,
                    final_weight=float(
                        getattr(criterion, 'spatial_query_final_weight', 0.25)
                    ),
                    guard_weight=float(
                        getattr(criterion, 'spatial_query_guard_weight', 1.0)
                    ),
                    negative_weight=float(
                        getattr(criterion, 'spatial_query_negative_weight', 1.0)
                    ),
                )
                if spatial_obj is not None:
                    spatial_query_loss, spatial_stats = spatial_obj
                    spatial_query_positive = spatial_stats["positive_loss"]
                    spatial_query_negative = spatial_stats["negative_loss"]
                    spatial_query_final = spatial_stats["final_loss"]
                    spatial_query_guard = spatial_stats["guard_loss"]
                    spatial_query_gamma = spatial_stats["gamma_mean"]
                    spatial_query_view_a = spatial_stats["view_a_weight"]
                    total_loss = total_loss + spatial_query_lambda * spatial_query_loss

            view_evidence_aux_loss = logits.new_tensor(0.0)
            view_evidence_distill_loss = logits.new_tensor(0.0)
            view_evidence_ramp = logits.new_tensor(0.0)
            view_evidence_teacher_win = logits.new_tensor(0.0)
            view_evidence_active = logits.new_tensor(0.0)
            view_evidence_advantage = logits.new_tensor(0.0)
            view_evidence_rescue = logits.new_tensor(0.0)
            view_evidence_aux_weight = float(
                getattr(criterion, 'view_evidence_aux_weight', 0.0)
            )
            view_evidence_distill_weight = float(
                getattr(criterion, 'view_evidence_distill_weight', 0.0)
            )
            if view_evidence_aux_weight > 0 or view_evidence_distill_weight > 0:
                evidence_obj = _cross_view_best_evidence_loss(
                    model_out=model_out,
                    target=target,
                    epoch=epoch,
                    warmup_epochs=int(
                        getattr(criterion, 'view_evidence_warmup_epochs', 5)
                    ),
                    ramp_epochs=int(
                        getattr(criterion, 'view_evidence_ramp_epochs', 5)
                    ),
                    advantage_temperature=float(
                        getattr(
                            criterion,
                            'view_evidence_advantage_temperature',
                            0.1,
                        )
                    ),
                    negative_weight=float(
                        getattr(criterion, 'view_evidence_negative_weight', 1.0)
                    ),
                )
                if evidence_obj is not None:
                    (
                        view_evidence_aux_loss,
                        view_evidence_distill_loss,
                        evidence_stats,
                    ) = evidence_obj
                    view_evidence_ramp = evidence_stats["distill_ramp"]
                    view_evidence_teacher_win = evidence_stats["teacher_win_rate"]
                    view_evidence_active = evidence_stats["active_weight"]
                    view_evidence_advantage = evidence_stats["advantage"]
                    view_evidence_rescue = evidence_stats["rescue_rate"]
                    total_loss = (
                        total_loss
                        + view_evidence_aux_weight * view_evidence_aux_loss
                        + view_evidence_distill_weight
                        * view_evidence_ramp
                        * view_evidence_distill_loss
                    )

            dvcre_aux_loss = logits.new_tensor(0.0)
            dvcre_positive = logits.new_tensor(0.0)
            dvcre_negative = logits.new_tensor(0.0)
            dvcre_gate_a = logits.new_tensor(0.5)
            dvcre_confidence = logits.new_tensor(0.0)
            dvcre_residual_scale = logits.new_tensor(0.0)
            dvcre_aux_weight = float(
                getattr(criterion, 'dvcre_aux_weight', 0.0)
            )
            if dvcre_aux_weight > 0:
                dvcre_obj = _dvcre_auxiliary_loss(model_out, target)
                if dvcre_obj is not None:
                    dvcre_aux_loss, dvcre_stats = dvcre_obj
                    dvcre_positive = dvcre_stats["positive_loss"]
                    dvcre_negative = dvcre_stats["negative_loss"]
                    dvcre_gate_a = dvcre_stats["gate_a_mean"]
                    dvcre_confidence = dvcre_stats["region_confidence"]
                    dvcre_residual_scale = dvcre_stats["residual_scale"]
                    total_loss = total_loss + dvcre_aux_weight * dvcre_aux_loss

            iscvf_intervention_loss = logits.new_tensor(0.0)
            iscvf_consistency_loss = logits.new_tensor(0.0)
            iscvf_consistency_ramp = logits.new_tensor(0.0)
            iscvf_confidence = logits.new_tensor(0.0)
            iscvf_gate_a = logits.new_tensor(0.5)
            iscvf_intervention_type = logits.new_tensor(-1.0)
            iscvf_intervention_weight = float(
                getattr(criterion, 'iscvf_intervention_weight', 0.0)
            )
            iscvf_consistency_weight = float(
                getattr(criterion, 'iscvf_consistency_weight', 0.0)
            )
            if iscvf_intervention_weight > 0 or iscvf_consistency_weight > 0:
                iscvf_obj = _iscvf_intervention_loss(
                    model_out=model_out,
                    target=target,
                    epoch=epoch,
                    warmup_epochs=int(
                        getattr(criterion, 'iscvf_warmup_epochs', 5)
                    ),
                    ramp_epochs=int(
                        getattr(criterion, 'iscvf_ramp_epochs', 5)
                    ),
                )
                if iscvf_obj is not None:
                    (
                        iscvf_intervention_loss,
                        iscvf_consistency_loss,
                        iscvf_stats,
                    ) = iscvf_obj
                    iscvf_consistency_ramp = iscvf_stats["consistency_ramp"]
                    iscvf_confidence = iscvf_stats["confidence"]
                    iscvf_gate_a = iscvf_stats["gate_a_mean"]
                    iscvf_intervention_type = iscvf_stats["intervention_type"]
                    total_loss = (
                        total_loss
                        + iscvf_intervention_weight * iscvf_intervention_loss
                        + iscvf_consistency_weight
                        * iscvf_consistency_ramp
                        * iscvf_consistency_loss
                    )

            visual_route_aux_loss = logits.new_tensor(0.0)
            visual_route_rescue_loss = logits.new_tensor(0.0)
            visual_route_guard_loss = logits.new_tensor(0.0)
            visual_route_cycle_loss = logits.new_tensor(0.0)
            visual_route_need = logits.new_tensor(0.0)
            visual_route_harm = logits.new_tensor(0.0)
            visual_route_view_a = logits.new_tensor(0.5)
            visual_route_trust = logits.new_tensor(0.0)
            visual_route_rejection = logits.new_tensor(1.0)
            visual_route_agreement = logits.new_tensor(0.0)
            visual_route_confidence = logits.new_tensor(0.0)
            visual_route_gamma = logits.new_tensor(0.0)
            visual_route_aux_weight = float(
                getattr(criterion, 'visual_route_aux_weight', 0.0)
            )
            visual_route_rescue_weight = float(
                getattr(criterion, 'visual_route_rescue_weight', 0.0)
            )
            visual_route_guard_weight = float(
                getattr(criterion, 'visual_route_guard_weight', 0.0)
            )
            visual_route_cycle_weight = float(
                getattr(criterion, 'visual_route_cycle_weight', 0.0)
            )
            if any(
                weight > 0
                for weight in (
                    visual_route_aux_weight,
                    visual_route_rescue_weight,
                    visual_route_guard_weight,
                    visual_route_cycle_weight,
                )
            ):
                route_obj = _visual_evidence_routing_loss(
                    model_out=model_out,
                    target=target,
                    negative_weight=float(
                        getattr(criterion, 'visual_route_negative_weight', 1.0)
                    ),
                )
                if route_obj is not None:
                    (
                        visual_route_aux_loss,
                        visual_route_rescue_loss,
                        visual_route_guard_loss,
                        visual_route_cycle_loss,
                        visual_route_stats,
                    ) = route_obj
                    visual_route_need = visual_route_stats["need"]
                    visual_route_harm = visual_route_stats["harm_rate"]
                    visual_route_view_a = visual_route_stats["view_a_weight"]
                    visual_route_trust = visual_route_stats["trust"]
                    visual_route_rejection = visual_route_stats["rejection"]
                    visual_route_agreement = visual_route_stats["agreement"]
                    visual_route_confidence = visual_route_stats["confidence"]
                    visual_route_gamma = visual_route_stats["gamma"]
                    total_loss = (
                        total_loss
                        + visual_route_aux_weight * visual_route_aux_loss
                        + visual_route_rescue_weight * visual_route_rescue_loss
                        + visual_route_guard_weight * visual_route_guard_loss
                        + visual_route_cycle_weight * visual_route_cycle_loss
                    )

            plain_innovation_aux_loss = logits.new_tensor(0.0)
            plain_innovation_route_loss = logits.new_tensor(0.0)
            plain_innovation_guard_loss = logits.new_tensor(0.0)
            plain_innovation_consistency_loss = logits.new_tensor(0.0)
            plain_innovation_regret_loss = logits.new_tensor(0.0)
            plain_innovation_evidence_loss = logits.new_tensor(0.0)
            plain_innovation_match_loss = logits.new_tensor(0.0)
            plain_innovation_single_loss = logits.new_tensor(0.0)
            plain_innovation_rank_loss = logits.new_tensor(0.0)
            plain_innovation_budget_loss = logits.new_tensor(0.0)
            plain_innovation_gate = logits.new_tensor(0.0)
            plain_innovation_confidence = logits.new_tensor(0.0)
            plain_innovation_gamma = logits.new_tensor(0.0)
            plain_innovation_ramp = logits.new_tensor(0.0)
            plain_innovation_correction = logits.new_tensor(0.0)
            plain_innovation_harm = logits.new_tensor(0.0)
            plain_innovation_route_active = logits.new_tensor(0.0)
            plain_innovation_weights = {
                "aux": float(getattr(
                    criterion, "plain_innovation_aux_weight", 0.0
                )),
                "route": float(getattr(
                    criterion, "plain_innovation_route_weight", 0.0
                )),
                "guard": float(getattr(
                    criterion, "plain_innovation_guard_weight", 0.0
                )),
                "consistency": float(getattr(
                    criterion, "plain_innovation_consistency_weight", 0.0
                )),
                "regret": float(getattr(
                    criterion, "plain_innovation_regret_weight", 0.0
                )),
                "evidence": float(getattr(
                    criterion, "plain_innovation_evidence_weight", 0.0
                )),
                "match": float(getattr(
                    criterion, "plain_innovation_match_weight", 0.0
                )),
                "single": float(getattr(
                    criterion, "plain_innovation_single_weight", 0.0
                )),
                "rank": float(getattr(
                    criterion, "plain_innovation_rank_weight", 0.0
                )),
                "budget": float(getattr(
                    criterion, "plain_innovation_budget_weight", 0.0
                )),
            }
            if any(weight > 0.0 for weight in plain_innovation_weights.values()):
                innovation_obj = _plain_bce_innovation_loss(model_out, target)
                if innovation_obj is not None:
                    plain_innovation_aux_loss = innovation_obj["aux"]
                    plain_innovation_route_loss = innovation_obj["route"]
                    plain_innovation_guard_loss = innovation_obj["guard"]
                    plain_innovation_consistency_loss = innovation_obj[
                        "consistency"
                    ]
                    plain_innovation_regret_loss = innovation_obj["regret"]
                    plain_innovation_evidence_loss = innovation_obj["evidence"]
                    plain_innovation_match_loss = innovation_obj["match"]
                    plain_innovation_single_loss = innovation_obj["single"]
                    plain_innovation_rank_loss = innovation_obj["rank"]
                    plain_innovation_budget_loss = innovation_obj["budget"]
                    plain_innovation_gate = innovation_obj["gate"]
                    plain_innovation_confidence = innovation_obj["confidence"]
                    plain_innovation_gamma = innovation_obj["gamma"]
                    plain_innovation_ramp = innovation_obj["ramp"]
                    plain_innovation_correction = innovation_obj["correction"]
                    plain_innovation_harm = innovation_obj["harm"]
                    plain_innovation_route_active = innovation_obj[
                        "route_active"
                    ]
                    total_loss = total_loss + sum(
                        plain_innovation_weights[name] * innovation_obj[name]
                        for name in plain_innovation_weights
                    )

            selective_rescue_aux_loss = logits.new_tensor(0.0)
            selective_rescue_loss = logits.new_tensor(0.0)
            selective_rescue_guard_loss = logits.new_tensor(0.0)
            selective_rescue_gate = logits.new_tensor(0.0)
            selective_rescue_gamma = logits.new_tensor(0.0)
            selective_rescue_ramp = logits.new_tensor(0.0)
            selective_rescue_decay = logits.new_tensor(0.0)
            sr_aux_weight = float(getattr(criterion, 'selective_rescue_aux_weight', 0.0))
            sr_loss_weight = float(getattr(criterion, 'selective_rescue_loss_weight', 0.0))
            sr_guard_weight = float(getattr(criterion, 'selective_rescue_guard_weight', 0.0))
            if sr_aux_weight > 0 or sr_loss_weight > 0 or sr_guard_weight > 0:
                sr_obj = _selective_view_rescue_loss(model_out, target)
                if sr_obj is not None:
                    selective_rescue_aux_loss, selective_rescue_loss, selective_rescue_guard_loss, sr_stats = sr_obj
                    warmup = int(getattr(criterion, 'selective_rescue_warmup_epochs', 5))
                    ramp_epochs = max(int(getattr(criterion, 'selective_rescue_ramp_epochs', 5)), 1)
                    ramp = 0.0 if epoch < warmup else min(1.0, (epoch - warmup + 1) / ramp_epochs)
                    decay_start = int(getattr(criterion, 'selective_rescue_aux_decay_start', 30))
                    decay_end = max(int(getattr(criterion, 'selective_rescue_aux_decay_end', 80)), decay_start + 1)
                    decay = 1.0 if epoch < decay_start else max(0.0, (decay_end - epoch) / (decay_end - decay_start))
                    selective_rescue_ramp = logits.new_tensor(ramp)
                    selective_rescue_decay = logits.new_tensor(decay)
                    selective_rescue_gate = sr_stats["gate"]
                    selective_rescue_gamma = sr_stats["gamma"]
                    total_loss = total_loss + sr_aux_weight * decay * selective_rescue_aux_loss
                    total_loss = total_loss + ramp * (
                        sr_loss_weight * selective_rescue_loss
                        + sr_guard_weight * selective_rescue_guard_loss
                    )

            frozen_rescue_view_loss = logits.new_tensor(0.0)
            frozen_rescue_trust_loss = logits.new_tensor(0.0)
            frozen_rescue_rank_loss = logits.new_tensor(0.0)
            frozen_rescue_guard_loss = logits.new_tensor(0.0)
            frozen_rescue_gate = logits.new_tensor(0.0)
            frozen_rescue_trust = logits.new_tensor(0.0)
            frozen_rescue_gamma = logits.new_tensor(0.0)
            frozen_rescue_correction = logits.new_tensor(0.0)
            fr_weights = (
                float(getattr(criterion, 'frozen_rescue_aux_weight', 0.0)),
                float(getattr(criterion, 'frozen_rescue_trust_weight', 0.0)),
                float(getattr(criterion, 'frozen_rescue_rank_weight', 0.0)),
                float(getattr(criterion, 'frozen_rescue_guard_weight', 0.0)),
            )
            if any(weight > 0 for weight in fr_weights):
                fr_obj = _frozen_anchor_rescue_loss(
                    model_out, target,
                    rank_temperature=float(getattr(
                        criterion, 'frozen_rescue_rank_temperature', 0.2
                    )),
                )
                if fr_obj is not None:
                    (
                        frozen_rescue_view_loss,
                        frozen_rescue_trust_loss,
                        frozen_rescue_rank_loss,
                        frozen_rescue_guard_loss,
                        fr_stats,
                    ) = fr_obj
                    total_loss = total_loss + sum(
                        weight * value for weight, value in zip(fr_weights, (
                            frozen_rescue_view_loss,
                            frozen_rescue_trust_loss,
                            frozen_rescue_rank_loss,
                            frozen_rescue_guard_loss,
                        ))
                    )
                    frozen_rescue_gate = fr_stats["gate"]
                    frozen_rescue_trust = fr_stats["trust"]
                    frozen_rescue_gamma = fr_stats["gamma"]
                    frozen_rescue_correction = fr_stats["correction"]

            frozen_region_view_loss = logits.new_tensor(0.0)
            frozen_region_trust_loss = logits.new_tensor(0.0)
            frozen_region_rank_loss = logits.new_tensor(0.0)
            frozen_region_guard_loss = logits.new_tensor(0.0)
            frozen_region_gate = logits.new_tensor(0.0)
            frozen_region_trust = logits.new_tensor(0.0)
            frozen_region_gamma = logits.new_tensor(0.0)
            frozen_region_correction = logits.new_tensor(0.0)
            frozen_region_confidence = logits.new_tensor(0.0)
            region_weights = (
                float(getattr(criterion, 'frozen_region_aux_weight', 0.0)),
                float(getattr(criterion, 'frozen_region_trust_weight', 0.0)),
                float(getattr(criterion, 'frozen_region_rank_weight', 0.0)),
                float(getattr(criterion, 'frozen_region_guard_weight', 0.0)),
            )
            if any(weight > 0 for weight in region_weights):
                region_obj = _frozen_region_rescue_loss(
                    model_out, target,
                    rank_temperature=float(getattr(
                        criterion, 'frozen_region_rank_temperature', 0.1
                    )),
                    target_gain=float(getattr(
                        criterion, 'frozen_region_target_gain', 0.005
                    )),
                )
                if region_obj is not None:
                    (
                        frozen_region_view_loss,
                        frozen_region_trust_loss,
                        frozen_region_rank_loss,
                        frozen_region_guard_loss,
                        region_stats,
                    ) = region_obj
                    total_loss = total_loss + sum(
                        weight * value for weight, value in zip(region_weights, (
                            frozen_region_view_loss,
                            frozen_region_trust_loss,
                            frozen_region_rank_loss,
                            frozen_region_guard_loss,
                        ))
                    )
                    frozen_region_gate = region_stats["gate"]
                    frozen_region_trust = region_stats["trust"]
                    frozen_region_gamma = region_stats["gamma"]
                    frozen_region_correction = region_stats["correction"]
                    frozen_region_confidence = region_stats["region_confidence"]

            frozen_counterfactual_route_loss = logits.new_tensor(0.0)
            frozen_counterfactual_rank_loss = logits.new_tensor(0.0)
            frozen_counterfactual_guard_loss = logits.new_tensor(0.0)
            frozen_counterfactual_residual_loss = logits.new_tensor(0.0)
            frozen_counterfactual_gate = logits.new_tensor(0.0)
            frozen_counterfactual_rho = logits.new_tensor(0.0)
            frozen_counterfactual_view_a = logits.new_tensor(0.5)
            frozen_counterfactual_help = logits.new_tensor(0.0)
            frozen_counterfactual_target = logits.new_tensor(0.0)
            frozen_counterfactual_correction = logits.new_tensor(0.0)
            frozen_counterfactual_pairs = logits.new_tensor(0.0)
            counterfactual_weights = (
                float(getattr(
                    criterion, 'frozen_counterfactual_route_weight', 0.0
                )),
                float(getattr(
                    criterion, 'frozen_counterfactual_rank_weight', 0.0
                )),
                float(getattr(
                    criterion, 'frozen_counterfactual_guard_weight', 0.0
                )),
                float(getattr(
                    criterion, 'frozen_counterfactual_residual_weight', 0.0
                )),
            )
            if any(weight > 0 for weight in counterfactual_weights):
                counterfactual_obj = _frozen_counterfactual_router_loss(
                    model_out,
                    target,
                    queue_owner=criterion,
                    router_temperature=float(getattr(
                        criterion,
                        'frozen_counterfactual_router_temperature',
                        0.02,
                    )),
                    router_margin=float(getattr(
                        criterion, 'frozen_counterfactual_router_margin', 0.02
                    )),
                    rank_temperature=float(getattr(
                        criterion,
                        'frozen_counterfactual_rank_temperature',
                        0.2,
                    )),
                    rank_margin=float(getattr(
                        criterion, 'frozen_counterfactual_rank_margin', 0.5
                    )),
                    hard_threshold=float(getattr(
                        criterion,
                        'frozen_counterfactual_hard_threshold',
                        2.0,
                    )),
                    guard_threshold=float(getattr(
                        criterion,
                        'frozen_counterfactual_guard_threshold',
                        4.0,
                    )),
                    queue_size=int(getattr(
                        criterion, 'frozen_counterfactual_queue_size', 128
                    )),
                )
                if counterfactual_obj is not None:
                    (
                        frozen_counterfactual_route_loss,
                        frozen_counterfactual_rank_loss,
                        frozen_counterfactual_guard_loss,
                        frozen_counterfactual_residual_loss,
                        counterfactual_stats,
                    ) = counterfactual_obj
                    total_loss = total_loss + sum(
                        weight * value
                        for weight, value in zip(
                            counterfactual_weights,
                            (
                                frozen_counterfactual_route_loss,
                                frozen_counterfactual_rank_loss,
                                frozen_counterfactual_guard_loss,
                                frozen_counterfactual_residual_loss,
                            ),
                        )
                    )
                    frozen_counterfactual_gate = counterfactual_stats[
                        "rescue_gate"
                    ]
                    frozen_counterfactual_rho = counterfactual_stats["rho"]
                    frozen_counterfactual_view_a = counterfactual_stats[
                        "view_a_weight"
                    ]
                    frozen_counterfactual_help = counterfactual_stats[
                        "help_rate"
                    ]
                    frozen_counterfactual_target = counterfactual_stats[
                        "target_rate"
                    ]
                    frozen_counterfactual_correction = counterfactual_stats[
                        "correction"
                    ]
                    frozen_counterfactual_pairs = counterfactual_stats[
                        "pair_groups"
                    ]

            m9_expert_loss = logits.new_tensor(0.0)
            m9_alignment_loss = logits.new_tensor(0.0)
            m9_diversity_loss = logits.new_tensor(0.0)
            m9_route_loss = logits.new_tensor(0.0)
            m9_rank_loss = logits.new_tensor(0.0)
            m9_guard_loss = logits.new_tensor(0.0)
            m9_residual_loss = logits.new_tensor(0.0)
            m9_gate = logits.new_tensor(0.0)
            m9_rho = logits.new_tensor(0.0)
            m9_help = logits.new_tensor(0.0)
            m9_target = logits.new_tensor(0.0)
            m9_expert_match = logits.new_tensor(0.0)
            m9_correction = logits.new_tensor(0.0)
            m9_pairs = logits.new_tensor(0.0)
            m9_region_gate = logits.new_tensor(0.0)
            m9_residual_norm = logits.new_tensor(0.0)
            m9_residual_scale = logits.new_tensor(0.0)
            m9_weights = tuple(float(getattr(criterion, name, 0.0)) for name in (
                "frozen_region_interaction_expert_weight",
                "frozen_region_interaction_alignment_weight",
                "frozen_region_interaction_diversity_weight",
                "frozen_region_interaction_route_weight",
                "frozen_region_interaction_rank_weight",
                "frozen_region_interaction_guard_weight",
                "frozen_region_interaction_residual_weight",
            ))
            if any(weight > 0 for weight in m9_weights):
                m9_obj = _frozen_region_interaction_moe_loss(
                    model_out,
                    target,
                    queue_owner=criterion,
                    expert_temperature=float(getattr(
                        criterion,
                        "frozen_region_interaction_expert_temperature",
                        0.1,
                    )),
                    router_temperature=float(getattr(
                        criterion,
                        "frozen_region_interaction_router_temperature",
                        0.03,
                    )),
                    router_margin=float(getattr(
                        criterion,
                        "frozen_region_interaction_router_margin",
                        0.01,
                    )),
                    rank_temperature=float(getattr(
                        criterion,
                        "frozen_region_interaction_rank_temperature",
                        0.2,
                    )),
                    rank_margin=float(getattr(
                        criterion,
                        "frozen_region_interaction_rank_margin",
                        0.5,
                    )),
                    hard_threshold=float(getattr(
                        criterion,
                        "frozen_region_interaction_hard_threshold",
                        2.0,
                    )),
                    guard_threshold=float(getattr(
                        criterion,
                        "frozen_region_interaction_guard_threshold",
                        4.0,
                    )),
                    queue_size=int(getattr(
                        criterion,
                        "frozen_region_interaction_queue_size",
                        128,
                    )),
                )
                if m9_obj is not None:
                    (
                        m9_expert_loss,
                        m9_alignment_loss,
                        m9_diversity_loss,
                        m9_route_loss,
                        m9_rank_loss,
                        m9_guard_loss,
                        m9_residual_loss,
                        m9_stats,
                    ) = m9_obj
                    total_loss = total_loss + sum(
                        weight * value
                        for weight, value in zip(m9_weights, (
                            m9_expert_loss,
                            m9_alignment_loss,
                            m9_diversity_loss,
                            m9_route_loss,
                            m9_rank_loss,
                            m9_guard_loss,
                            m9_residual_loss,
                        ))
                    )
                    m9_gate = m9_stats["rescue_gate"]
                    m9_rho = m9_stats["rho"]
                    m9_help = m9_stats["help_rate"]
                    m9_target = m9_stats["target_rate"]
                    m9_expert_match = m9_stats["expert_match"]
                    m9_correction = m9_stats["correction"]
                    m9_pairs = m9_stats["pair_groups"]
                    m9_region_gate = m9_stats["region_gate"]
                    m9_residual_norm = m9_stats["residual_norm"]
                    m9_residual_scale = m9_stats["residual_scale"]
            # ----------------------------------------------------

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if model_ema is not None:
            model_ema.update(model)

        acc1 = _micro_accuracy(logits, target)
        pos_only = getattr(criterion, 'pos_only', True)
        alpha    = getattr(criterion, 'alpha', 0.75)
        band     = getattr(criterion, 'band', 0.0)
        grad_var_t = _grad_var_from(logits, target, band=band, pos_only=pos_only, alpha=alpha)

        # 再更新日志
        metric_logger.update(
            loss=float(total_loss.item()),
            loss_sup=float(loss_sup.item()),
            loss_sem_conflict=float(sem_conflict_loss.item()),
            loss_sem_trust=float(sem_trust_loss.item()),
            sem_trust_gate=float(sem_trust_gate_mean.item()),
            sem_trust_help=float(sem_trust_help_rate.item()),
            sem_trust_active=float(sem_trust_active_rate.item()),
            loss_sem_rank=float(sem_rank_loss.item()),
            sem_rank_help=float(sem_rank_help.item()),
            sem_rank_guard=float(sem_rank_guard.item()),
            sem_rank_need=float(sem_rank_need.item()),
            loss_sem_error=float(sem_error_loss.item()),
            sem_error_help=float(sem_error_help.item()),
            sem_error_guard=float(sem_error_guard.item()),
            sem_error_need=float(sem_error_need.item()),
            sem_error_harm=float(sem_error_harm.item()),
            loss_spatial_query=float(spatial_query_loss.item()),
            spatial_query_pos=float(spatial_query_positive.item()),
            spatial_query_neg=float(spatial_query_negative.item()),
            spatial_query_final=float(spatial_query_final.item()),
            spatial_query_guard=float(spatial_query_guard.item()),
            spatial_query_gamma=float(spatial_query_gamma.item()),
            spatial_query_view_a=float(spatial_query_view_a.item()),
            loss_view_evidence=float(view_evidence_aux_loss.item()),
            loss_best_evidence=float(view_evidence_distill_loss.item()),
            best_evidence_ramp=float(view_evidence_ramp.item()),
            best_evidence_teacher_win=float(view_evidence_teacher_win.item()),
            best_evidence_active=float(view_evidence_active.item()),
            best_evidence_advantage=float(view_evidence_advantage.item()),
            best_evidence_rescue=float(view_evidence_rescue.item()),
            loss_selective_rescue_aux=float(selective_rescue_aux_loss.item()),
            loss_selective_rescue=float(selective_rescue_loss.item()),
            loss_selective_rescue_guard=float(selective_rescue_guard_loss.item()),
            selective_rescue_gate=float(selective_rescue_gate.item()),
            selective_rescue_gamma=float(selective_rescue_gamma.item()),
            selective_rescue_ramp=float(selective_rescue_ramp.item()),
            selective_rescue_decay=float(selective_rescue_decay.item()),
            loss_frozen_rescue_view=float(frozen_rescue_view_loss.item()),
            loss_frozen_rescue_trust=float(frozen_rescue_trust_loss.item()),
            loss_frozen_rescue_rank=float(frozen_rescue_rank_loss.item()),
            loss_frozen_rescue_guard=float(frozen_rescue_guard_loss.item()),
            frozen_rescue_gate=float(frozen_rescue_gate.item()),
            frozen_rescue_trust=float(frozen_rescue_trust.item()),
            frozen_rescue_gamma=float(frozen_rescue_gamma.item()),
            frozen_rescue_correction=float(frozen_rescue_correction.item()),
            loss_frozen_region_view=float(frozen_region_view_loss.item()),
            loss_frozen_region_trust=float(frozen_region_trust_loss.item()),
            loss_frozen_region_rank=float(frozen_region_rank_loss.item()),
            loss_frozen_region_guard=float(frozen_region_guard_loss.item()),
            frozen_region_gate=float(frozen_region_gate.item()),
            frozen_region_trust=float(frozen_region_trust.item()),
            frozen_region_gamma=float(frozen_region_gamma.item()),
            frozen_region_correction=float(frozen_region_correction.item()),
            frozen_region_confidence=float(frozen_region_confidence.item()),
            loss_frozen_counterfactual_route=float(
                frozen_counterfactual_route_loss.item()
            ),
            loss_frozen_counterfactual_rank=float(
                frozen_counterfactual_rank_loss.item()
            ),
            loss_frozen_counterfactual_guard=float(
                frozen_counterfactual_guard_loss.item()
            ),
            loss_frozen_counterfactual_residual=float(
                frozen_counterfactual_residual_loss.item()
            ),
            frozen_counterfactual_gate=float(
                frozen_counterfactual_gate.item()
            ),
            frozen_counterfactual_rho=float(
                frozen_counterfactual_rho.item()
            ),
            frozen_counterfactual_view_a=float(
                frozen_counterfactual_view_a.item()
            ),
            frozen_counterfactual_help=float(
                frozen_counterfactual_help.item()
            ),
            frozen_counterfactual_target=float(
                frozen_counterfactual_target.item()
            ),
            frozen_counterfactual_correction=float(
                frozen_counterfactual_correction.item()
            ),
            frozen_counterfactual_pairs=float(
                frozen_counterfactual_pairs.item()
            ),
            loss_m9_expert=float(m9_expert_loss.item()),
            loss_m9_alignment=float(m9_alignment_loss.item()),
            loss_m9_diversity=float(m9_diversity_loss.item()),
            loss_m9_route=float(m9_route_loss.item()),
            loss_m9_rank=float(m9_rank_loss.item()),
            loss_m9_guard=float(m9_guard_loss.item()),
            loss_m9_residual=float(m9_residual_loss.item()),
            m9_gate=float(m9_gate.item()),
            m9_rho=float(m9_rho.item()),
            m9_help=float(m9_help.item()),
            m9_target=float(m9_target.item()),
            m9_expert_match=float(m9_expert_match.item()),
            m9_correction=float(m9_correction.item()),
            m9_pairs=float(m9_pairs.item()),
            m9_region_gate=float(m9_region_gate.item()),
            m9_residual_norm=float(m9_residual_norm.item()),
            m9_residual_scale=float(m9_residual_scale.item()),
            loss_dvcre=float(dvcre_aux_loss.item()),
            dvcre_pos=float(dvcre_positive.item()),
            dvcre_neg=float(dvcre_negative.item()),
            dvcre_gate_a=float(dvcre_gate_a.item()),
            dvcre_confidence=float(dvcre_confidence.item()),
            dvcre_scale=float(dvcre_residual_scale.item()),
            loss_iscvf_intervention=float(iscvf_intervention_loss.item()),
            loss_iscvf_consistency=float(iscvf_consistency_loss.item()),
            iscvf_ramp=float(iscvf_consistency_ramp.item()),
            iscvf_confidence=float(iscvf_confidence.item()),
            iscvf_gate_a=float(iscvf_gate_a.item()),
            iscvf_intervention_type=float(iscvf_intervention_type.item()),
            loss_visual_route_aux=float(visual_route_aux_loss.item()),
            loss_visual_route_rescue=float(visual_route_rescue_loss.item()),
            loss_visual_route_guard=float(visual_route_guard_loss.item()),
            loss_visual_route_cycle=float(visual_route_cycle_loss.item()),
            visual_route_need=float(visual_route_need.item()),
            visual_route_harm=float(visual_route_harm.item()),
            visual_route_view_a=float(visual_route_view_a.item()),
            visual_route_trust=float(visual_route_trust.item()),
            visual_route_rejection=float(visual_route_rejection.item()),
            visual_route_agreement=float(visual_route_agreement.item()),
            visual_route_confidence=float(visual_route_confidence.item()),
            visual_route_gamma=float(visual_route_gamma.item()),
            loss_plain_innov_aux=float(plain_innovation_aux_loss.item()),
            loss_plain_innov_route=float(plain_innovation_route_loss.item()),
            loss_plain_innov_guard=float(plain_innovation_guard_loss.item()),
            loss_plain_innov_consistency=float(
                plain_innovation_consistency_loss.item()
            ),
            loss_plain_innov_regret=float(plain_innovation_regret_loss.item()),
            loss_plain_innov_evidence=float(
                plain_innovation_evidence_loss.item()
            ),
            loss_plain_innov_match=float(plain_innovation_match_loss.item()),
            loss_plain_innov_single=float(plain_innovation_single_loss.item()),
            loss_plain_innov_rank=float(plain_innovation_rank_loss.item()),
            loss_plain_innov_budget=float(plain_innovation_budget_loss.item()),
            plain_innov_gate=float(plain_innovation_gate.item()),
            plain_innov_confidence=float(plain_innovation_confidence.item()),
            plain_innov_gamma=float(plain_innovation_gamma.item()),
            plain_innov_ramp=float(plain_innovation_ramp.item()),
            plain_innov_correction=float(plain_innovation_correction.item()),
            plain_innov_harm=float(plain_innovation_harm.item()),
            plain_innov_route_active=float(
                plain_innovation_route_active.item()
            ),
            acc1=float(acc1.item()),
            grad_var=float(grad_var_t.item()),
        )
        for pg in optimizer.param_groups:
            if "lr" in pg:
                metric_logger.update(lr=pg["lr"])
                break

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@th.no_grad()
def evaluate(
    data_loader,
    model: nn.Module,
    device: th.device,
    criterion: Optional[nn.Module] = None,
    amp: bool = True,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
    csv_path: Optional[str] = None,
    epoch: Optional[int] = None,
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")

    try:
        bf16_ok = th.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    ac_dtype = th.bfloat16 if bf16_ok else th.float16

    all_logits = []
    all_targets = []
    all_spatial_query_logits = []
    all_m9_anchor_logits = []
    all_m9_expert_logits = []
    m9_expert_names = None

    for samples in metric_logger.log_every(data_loader, 100, header="Test:"):
        xa, xb, target = _unpack_samples(samples, device)
        if target.dtype != th.float32:
            target = target.float()

        with th.autocast(device_type='cuda', dtype=ac_dtype, enabled=amp):
            model_out, logits = _forward_outputs(model, xa, xb, drop_feats=True)
            loss = criterion(logits, target) if (criterion is not None) else th.tensor(0.0, device=device)

        all_logits.append(logits.detach().float().cpu())
        all_targets.append(target.detach().float().cpu())
        if isinstance(model_out, dict):
            query_logits = model_out.get("spatial_query_logits")
            if query_logits is not None:
                all_spatial_query_logits.append(query_logits.detach().float().cpu())
            m9_aux = model_out.get("frozen_region_interaction_aux")
            if isinstance(m9_aux, dict):
                all_m9_anchor_logits.append(
                    m9_aux["anchor_logits"].detach().float().cpu()
                )
                all_m9_expert_logits.append(
                    m9_aux["expert_logits"].detach().float().cpu()
                )
                m9_expert_names = tuple(m9_aux.get(
                    "expert_names",
                    ("unique_a", "unique_b", "redundant", "synergy"),
                ))

        metric_logger.update(loss=float(loss.item()))

    logits_cat = th.cat(all_logits, dim=0)
    targets_cat = th.cat(all_targets, dim=0)

    acc1 = _micro_accuracy(logits_cat, targets_cat, thresh=threshold)
    f1_micro, f1_macro = _f1_scores(logits_cat, targets_cat, thresh=threshold)
    per_class_ap, mAP = _evaluate_multilabel_ap(logits_cat, targets_cat, class_names=class_names)

    results = {
        "loss": metric_logger.meters["loss"].global_avg if "loss" in metric_logger.meters else 0.0,
        "acc1": float(acc1.item()),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "mAP": float(mAP),
        "per_class_ap": per_class_ap,
    }

    if len(all_spatial_query_logits) == len(all_targets):
        query_logits_cat = th.cat(all_spatial_query_logits, dim=0)
        print("---- Standalone spatial-query branch ----")
        query_per_class_ap, query_mAP = _evaluate_multilabel_ap(
            query_logits_cat,
            targets_cat,
            class_names=class_names,
        )
        results["spatial_query_mAP"] = float(query_mAP)
        results["spatial_query_per_class_ap"] = query_per_class_ap

    if (
        len(all_m9_anchor_logits) == len(all_targets)
        and len(all_m9_expert_logits) == len(all_targets)
    ):
        m9_anchor = th.cat(all_m9_anchor_logits, dim=0)
        m9_experts = th.cat(all_m9_expert_logits, dim=0)

        def quiet_ap(candidate_logits):
            probabilities = th.sigmoid(candidate_logits)
            per_class = [
                _average_precision_score(
                    probabilities[:, class_index],
                    targets_cat[:, class_index],
                )
                for class_index in range(probabilities.shape[1])
            ]
            return per_class, float(sum(per_class) / max(len(per_class), 1))

        anchor_ap, anchor_map = quiet_ap(m9_anchor)
        expert_maps = {}
        expert_per_class = {}
        candidate_per_class = [anchor_ap]
        for expert_index, expert_name in enumerate(m9_expert_names):
            expert_ap, expert_map = quiet_ap(m9_experts[:, expert_index])
            expert_maps[expert_name] = expert_map
            expert_per_class[expert_name] = expert_ap
            candidate_per_class.append(expert_ap)

        uniform_ap, uniform_map = quiet_ap(m9_experts.mean(dim=1))
        class_oracle_ap = th.tensor(candidate_per_class).amax(dim=0).tolist()
        class_oracle_map = float(sum(class_oracle_ap) / len(class_oracle_ap))
        candidates = th.cat((m9_anchor.unsqueeze(1), m9_experts), dim=1)
        candidate_targets = targets_cat.unsqueeze(1).expand_as(candidates)
        candidate_errors = F.binary_cross_entropy_with_logits(
            candidates, candidate_targets, reduction="none"
        )
        best_candidate = candidate_errors.argmin(dim=1, keepdim=True)
        element_oracle_logits = candidates.gather(
            1, best_candidate
        ).squeeze(1)
        element_oracle_ap, element_oracle_map = quiet_ap(
            element_oracle_logits
        )

        results.update({
            "m9_anchor_mAP": anchor_map,
            "m9_anchor_per_class_ap": anchor_ap,
            "m9_expert_mAPs": expert_maps,
            "m9_expert_per_class_ap": expert_per_class,
            "m9_uniform_mAP": uniform_map,
            "m9_uniform_per_class_ap": uniform_ap,
            "m9_class_oracle_mAP": class_oracle_map,
            "m9_class_oracle_per_class_ap": class_oracle_ap,
            "m9_element_oracle_mAP": element_oracle_map,
            "m9_element_oracle_per_class_ap": element_oracle_ap,
        })
        compact_experts = ", ".join(
            f"{name}={value:.4f}" for name, value in expert_maps.items()
        )
        print(
            "[M9 candidates] "
            f"anchor={anchor_map:.4f}; {compact_experts}; "
            f"uniform={uniform_map:.4f}; "
            f"class_oracle={class_oracle_map:.4f}; "
            f"element_oracle={element_oracle_map:.4f}"
        )

    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        header = ["epoch", "loss", "acc1", "f1_micro", "f1_macro", "mAP"]
        if class_names:
            header += [f"AP_{n}" for n in class_names]
        else:
            header += [f"AP_C{i}" for i in range(logits_cat.shape[1])]

        write_header = (not os.path.isfile(csv_path))
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            row = [
                -1 if epoch is None else int(epoch),
                results["loss"], results["acc1"], results["f1_micro"], results["f1_macro"], results["mAP"],
            ]
            row += [float(x) for x in per_class_ap]
            w.writerow(row)

    return results
