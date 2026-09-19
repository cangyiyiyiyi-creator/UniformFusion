# models/modules/custom_losses/dals_loss.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .signals import signal_bus

class DALSBCE(nn.Module):
    """
    DALS-BCE: a convex variant of FALS.
    y_tilde = (1 - w*eps)*y + (w*eps)*(1 - y), with w = ((1 - p_t)^gamma).detach()
    still BCEWithLogits(logits, y_tilde) -> convex, continuously differentiable and monotone in the logits.
    """
    def __init__(self, eps: float = 0.1, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.eps = float(eps)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # p_t: confidence of the prediction on the true label
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1.0 - p)              # [B,C]
        # stop-grad difficulty weight (no backprop, preserves convexity)
        w = (1.0 - pt).pow(self.gamma).detach()                  # [B,C]
        signal_bus.update_difficulty(w.mean())
        # smooth slightly towards the opposite class (1-y); hard samples have a larger w and are smoothed more
        y_tilde = (1.0 - w * self.eps) * targets + (w * self.eps) * (1.0 - targets)
        loss = self.bce(logits, y_tilde)                         # [B,C]
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
