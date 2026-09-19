# models/modules/composite_loss.py
import torch
import torch.nn as nn
from .custom_losses.focal_loss import FocalLoss
from .custom_losses.asymmetric_loss import AsymmetricLossMultiLabel
from .custom_losses.fals_loss import FALSLoss
from .custom_losses.mcb_loss import MCBLoss, MCBLossConvex
from .custom_losses.gebce import GEBCELoss
from .custom_losses.dals_loss import DALSBCE as DALSLoss
from .custom_losses.consistency_losses import (
    UncertaintyConsistencyLoss, ChannelAttentionConsistencyLoss, RelationalConsistencyLoss
)


class CompositeLoss(nn.Module):
    """
    A unified, self-configuring composite loss for the "normal training" mode.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.base_loss == 'mcb_convex':
            self.base_criterion = MCBLossConvex(tau=args.mcb_tau, w_min=args.mcb_wmin, momentum=args.mcb_momentum)
        elif args.base_loss == 'gebce': 
            self.base_criterion = GEBCELoss(lambda_coef=args.ge_lambda, pos_only=args.ge_pos_only,
                                alpha=args.ge_alpha, ema=args.ge_ema,
                                momentum=args.ge_momentum, band=args.ge_band,
                                trainable=args.ge_trainable)
        elif args.base_loss == 'dals':
            self.base_criterion = DALSLoss(eps=args.dals_eps, gamma=args.dals_gamma)
        # 1. build the base supervised loss
        elif args.base_loss == 'focal':
            self.base_criterion = FocalLoss(gamma=args.focal_gamma)
        elif args.base_loss == 'asl':
            self.base_criterion = AsymmetricLossMultiLabel(
                gamma_neg=args.asl_gamma_neg,
                gamma_pos=args.asl_gamma_pos,
                clip=args.asl_clip,
            )
        elif args.base_loss == 'fals':
            self.base_criterion = FALSLoss(eps=args.fals_eps, gamma=args.fals_gamma)
        elif args.base_loss == 'mcb':
            self.base_criterion = MCBLoss(tau=args.mcb_tau, momentum=args.mcb_momentum)
        elif args.base_loss == 'mlsm':
            self.base_criterion = nn.MultiLabelSoftMarginLoss()
        else:  # 'bce'
            self.base_criterion = nn.BCEWithLogitsLoss()

        # 2. build every optional consistency loss
        self.uncert_loss = UncertaintyConsistencyLoss() if args.use_uncertainty_loss else None
        self.chan_loss = ChannelAttentionConsistencyLoss() if args.use_channel_loss else None
        self.rel_loss = RelationalConsistencyLoss() if args.use_relational_loss else None

    def forward(self, model_outputs, targets):
        # model_outputs is the raw return value of ConvNeXtV2Dual.forward
        # e.g., {"logits": ..., "feats": {"A": ..., "B": ..., "fused": ...}}

        final_logits = model_outputs['logits']
        
        # --- 1. compute the main loss ---
        total_loss = self.base_criterion(final_logits, targets)

        # --- 2. compute and accumulate every auxiliary loss as needed ---
        # auxiliary losses require various intermediate results from the model output
        if self.uncert_loss and all(k in model_outputs for k in ['A_logits','B_logits']):
            total_loss += self.args.uncertainty_lambda * self.uncert_loss(
                model_outputs['A_logits'], model_outputs['B_logits'])

        if self.uncert_loss:
            total_loss += self.args.uncertainty_lambda * self.uncert_loss(logits_a, logits_b)
        
        # ... (add channel_loss and relational_loss in the same way) ...
        
        return total_loss
