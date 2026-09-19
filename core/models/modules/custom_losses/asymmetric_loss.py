import torch
import torch.nn as nn


class AsymmetricLossMultiLabel(nn.Module):
    """Asymmetric Loss for multi-label classification (ICCV 2021)."""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(max(clip, 0.0))
        self.eps = float(max(eps, 1e-12))
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1.0 - pos_prob
        if self.clip > 0:
            neg_prob = (neg_prob + self.clip).clamp(max=1.0)

        loss = targets * torch.log(pos_prob.clamp_min(self.eps))
        loss = loss + (1.0 - targets) * torch.log(neg_prob.clamp_min(self.eps))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = targets * pos_prob + (1.0 - targets) * neg_prob
            gamma = targets * self.gamma_pos + (1.0 - targets) * self.gamma_neg
            loss = loss * (1.0 - pt).pow(gamma)

        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
