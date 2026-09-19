# models/modules/distillation/utils.py
import torch
import torch.nn.functional as F

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    """
    Core implementation of DKD (Decoupled Knowledge Distillation).
    """
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    
    # TCKD
    log_pred_student_tck = F.log_softmax(logits_student / temperature - 1000 * other_mask, dim=1)
    pred_teacher_tck = F.softmax(logits_teacher / temperature - 1000 * other_mask, dim=1)
    tckd_loss = F.kl_div(log_pred_student_tck, pred_teacher_tck, reduction='batchmean') * (temperature ** 2)

    # NCKD
    log_pred_student_nck = F.log_softmax(logits_student / temperature - 1000 * gt_mask, dim=1)
    pred_teacher_nck = F.softmax(logits_teacher / temperature - 1000 * gt_mask, dim=1)
    nckd_loss = F.kl_div(log_pred_student_nck, pred_teacher_nck, reduction='batchmean') * (temperature ** 2)
    return alpha * tckd_loss + beta * nckd_loss

def _get_gt_mask(logits, target):
    """
    Build the ground-truth mask for the multi-label setting.
    target is already a multi-hot encoding; it only has to be cast to bool.
    """
    return target.bool()

def _get_other_mask(logits, target):
    """
    Build the "other" mask for the multi-label setting.
    """
    return ~target.bool() # ~ is the boolean "NOT" operator
