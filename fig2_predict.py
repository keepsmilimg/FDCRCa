"""Open-set gate + class prediction for Fig2 evaluation (paper mode)."""
import torch

from fig2_networks import semantic_hat
from utils import combine_ZA
from clf_utils import ce_on_clfsu, clfsu_logits, w_predict_open_set


def _open_gate(args, models, f, y_w, shr):
    gate = getattr(args, "open_gate", "clfsu")
    if gate == "w" or models.get("ClfSU") is None:
        return (y_w >= shr).long()
    logits_su = clfsu_logits(models["ClfSU"], f)
    return logits_su.argmax(dim=1)


def fig2_multimodal_feat(models, z_c, clu, shr, args):
    a_hat = semantic_hat(models, z_c, clu, shr)
    if getattr(args, "combine_za", True):
        return combine_ZA(z_c, a_hat), a_hat
    return z_c, a_hat


@torch.no_grad()
def fig2_target_predictions(args, models, z_c, clu, shr):
    """
    Returns (class_pred, su_pred) for target batch.
    su: 0=shared, 1=unknown (ClfSU or W threshold per open_gate).
    """
    from clf_utils import w_logits

    f, _ = fig2_multimodal_feat(models, z_c, clu, shr, args)
    logits = w_logits(models["W"], f)
    y_w = logits.argmax(dim=1)
    su = _open_gate(args, models, f, y_w, shr)
    y_cls = w_predict_open_set(models["W"], f, shr, su)

    bank = models.get("centers")
    if bank is not None and float(bank.unk_count.sum()) > 0:
        fix = (su == 1) & (y_cls < shr)
        if fix.any():
            d = torch.cdist(z_c[fix], bank.mu_unk, p=2)
            y_cls[fix] = shr + d.argmin(dim=1)

    return y_cls, su


def clfsu_loss_on_target(models, f, clu_t, shr, su_ce):
    su_lab = (clu_t >= shr).long()
    return ce_on_clfsu(models["ClfSU"], f, su_lab, su_ce)
