# models/modules/custom_losses/signals.py
import torch
import torch.nn as nn

def _to_tensor(x, device=None):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach()
    return torch.as_tensor(x, dtype=torch.float32, device=device).detach()

class SignalBus(nn.Module):
    """
    A lightweight "loss side -> attention side" signal bus (EMA smoothing + detach).
    Training: updated by the loss/engine every step; attention layers read the previous EMA value (more stable).
    Inference: frozen at the last EMA value.
    """
    def __init__(self, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        # [K] class weights (MCB-Convex), [K] per-class effective gradients (GE-BCE), scalar difficulty (FALS/DALS)
        self.register_buffer("w_mcb", None)
        self.register_buffer("g_strength", None)
        self.register_buffer("difficulty", torch.tensor(0.0))
        # dual-view disagreement (scalar); extend it to a map if spatial gating is added later
        self.register_buffer("dv_disagree", torch.tensor(0.0))

    def _ema(self, old, new):
        if new is None:
            return old
        m = self.momentum
        return new.clone() if old is None else old * m + new * (1.0 - m)

    @torch.no_grad()
    def update_mcb_weights(self, w):
        w = _to_tensor(w)
        if w is not None:
            self.w_mcb = self._ema(self.w_mcb, w)

    @torch.no_grad()
    def update_ge_strength(self, g_vec):
        g_vec = _to_tensor(g_vec)
        if g_vec is not None:
            self.g_strength = self._ema(self.g_strength, g_vec)

    @torch.no_grad()
    def update_difficulty(self, d_scalar):
        d_scalar = _to_tensor(d_scalar)
        if d_scalar is not None:
            self.difficulty = self._ema(self.difficulty, d_scalar)

    @torch.no_grad()
    def update_dv_disagree_scalar(self, logits_a, logits_b):
        if logits_a is None or logits_b is None:
            return
        pa = logits_a.detach().sigmoid()
        pb = logits_b.detach().sigmoid()
        s = torch.mean(torch.abs(pa - pb))  # scalar
        self.dv_disagree = self._ema(self.dv_disagree, s)

    def get(self):
        # not cloned on read to avoid overhead (do not modify in place from outside)
        return {
            "w_mcb": self.w_mcb,                 # [K] or None
            "g_strength": self.g_strength,       # [K] or None
            "difficulty": self.difficulty,       # scalar
            "dv_disagree": self.dv_disagree,     # scalar
        }

# singleton
signal_bus = SignalBus(momentum=0.9)
