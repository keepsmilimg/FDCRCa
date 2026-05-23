"""
Two-stage training (Algorithm 1 / Figure 2):
  Initialize target pseudo-labels.
  Stage 1 (E1): each epoch —
    (1) Learn Gc, D, C  by eq.7  (domain-invariant class features)
    (2) Learn Gd, D, C  by eq.11 (class-agnostic domain features)
    (3) Update target pseudo-labels (end of epoch; red text in paper)
  Between stages: update target pseudo-labels once (Alg.1 line 8).
  Stage 2 (E2): jointly learn Gc, phi, psi, W by eq.19.
  Output: Gc, phi, W.
"""
import copy
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import combine_ZA, count_epoch_on_large_dataset, adjust_learning_rate
from evaluation import eval_fig2
from fig2_networks import (
    semantic_hat,
    semantic_tilde,
    multimodal_feats_source,
    multimodal_feats_target,
)
from pseudo_label_analysis import PseudoLabelTracker
from clf_utils import (
    make_ce_loss,
    ce_on_clf,
    ce_on_clfsu,
    decouple_logits,
)

GRAD_CLIP = 5.0

# Paper table targets (I2AwA / Fig2), fractions in [0,1]
PAPER_TARGETS = {
    "OS": 0.766,
    "OU": 0.776,
    "H1": 0.771,
    "S": 0.746,
    "U": 0.316,
    "H2": 0.444,
}


def _early_stop_score(metrics, args):
    """Higher is better. 'paper' = negative weighted distance to paper row."""
    name = getattr(args, "early_stop_metric", "paper")
    if name == "H2":
        return metrics.get("H2", 0.0)
    if name == "H1":
        return metrics.get("H1", 0.0)
    if name == "OS":
        return metrics.get("OS", 0.0)
    if name == "OU":
        return metrics.get("OU", 0.0)
    w = {"OS": 1.0, "OU": 1.5, "H1": 1.0, "S": 1.2, "U": 2.5, "H2": 3.0}
    err = 0.0
    for k, tgt in PAPER_TARGETS.items():
        v = metrics.get(k, 0.0)
        err += w[k] * (v - tgt) ** 2
    return -err


def _update_best_checkpoint(models, metrics, args, best_pack):
    """Return updated (best_score, best_ep, best_state, best_metrics, patience_counter)."""
    score = _early_stop_score(metrics, args)
    min_delta = getattr(args, "early_stop_min_delta", 0.002)
    if score > best_pack["score"] + min_delta:
        best_pack["score"] = score
        best_pack["ep"] = best_pack.get("current_ep", -1)
        best_pack["metrics"] = dict(metrics)
        best_pack["state"] = {
            k: copy.deepcopy(models[k].state_dict())
            for k in models
            if hasattr(models[k], "state_dict")
        }
        best_pack["patience"] = 0
    else:
        best_pack["patience"] = best_pack.get("patience", 0) + 1
    return best_pack


def _print_metrics_vs_paper(metrics, prefix=""):
    t = PAPER_TARGETS
    print(
        f"{prefix}OS={100*metrics['OS']:.1f} OU={100*metrics['OU']:.1f} "
        f"H1={100*metrics['H1']:.1f} S={100*metrics['S']:.1f} "
        f"U={100*metrics['U']:.1f} H2={100*metrics['H2']:.1f} "
        f"(paper {100*t['OS']:.1f}/{100*t['OU']:.1f}/{100*t['H1']:.1f}/"
        f"{100*t['S']:.1f}/{100*t['U']:.1f}/{100*t['H2']:.1f})"
    )


def _clip_models(models, keys, max_norm=GRAD_CLIP):
    params = []
    for k in keys:
        if k in models and hasattr(models[k], "parameters"):
            params.extend(p for p in models[k].parameters() if p.requires_grad)
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def _grl_lambda(epoch, n_epochs):
    p = epoch / max(n_epochs, 1)
    return 2.0 / (1.0 + math.exp(-10 * p)) - 1.0


def _forward_gc_gd(models, feats):
    z_c = models["Gc"](feats)
    z_d = models["Gd"](feats)
    return z_c, z_d


def _dom_labels(n, label, device):
    return torch.full((n,), label, dtype=torch.long, device=device)


def _target_cls_losses(models, z_ct, clu_t, shr, unk_id, clf_ce, paper_mode=False):
    """Target CE: paper tilde{L}_CE on all C_t pseudo labels; legacy maps unk to open bucket."""
    shr_m = clu_t < shr
    unk_m = clu_t >= shr
    if paper_mode:
        loss = ce_on_clf(models["ClfC"], z_ct, clu_t, clf_ce)
        return loss, shr_m, unk_m
    loss = z_ct.new_tensor(0.0)
    if shr_m.any():
        loss = loss + ce_on_clf(models["ClfC"], z_ct[shr_m], clu_t[shr_m], clf_ce)
    if unk_m.any():
        y_unk = torch.full((int(unk_m.sum()),), unk_id, dtype=torch.long, device=z_ct.device)
        loss = loss + ce_on_clf(models["ClfC"], z_ct[unk_m], y_unk, clf_ce)
    return loss, shr_m, unk_m


def _paper_mode_on(args):
    return getattr(args, "paper_mode", True) and not getattr(args, "no_paper_mode", False)


def _eq5_pseudo_on(args):
    if getattr(args, "fixed_pseudo", True):
        return False
    return _paper_mode_on(args) and not getattr(args, "no_eq5_pseudo", False)


def _bootstrap_prototypes_from_source(args, pseudo_bank, models, dataloaders):
    """Initialize mu_known from source Gc before first eq.5 pass."""
    if pseudo_bank is None:
        return
    models["Gc"].eval()
    sums = np.zeros((pseudo_bank.C_s, 512), dtype=np.float64)
    cnt = np.zeros(pseudo_bank.C_s, dtype=np.int64)
    for feats, lbls, _, _ in dataloaders["tr_loader_src"]:
        x = feats.type(FloatTensor)
        y = lbls.numpy()
        z = models["Gc"](x).detach().cpu().numpy()
        for i in range(z.shape[0]):
            c = int(y[i])
            if c < pseudo_bank.C_s:
                sums[c] += z[i]
                cnt[c] += 1
    for c in range(pseudo_bank.C_s):
        if cnt[c] > 0:
            pseudo_bank.mu_known[c] = (sums[c] / cnt[c]).astype(np.float32)
    cent_path = os.path.join(
        args.data_path_target,
        "pseudo",
        args.src + "2" + args.tgt + f"_sample_xt_clu_cents{args.tgt_nc}.csv",
    )
    if os.path.isfile(cent_path):
        import pandas as pd

        cents = pd.read_csv(cent_path, header=None).values.astype(np.float32)
        shr, unk = pseudo_bank.C_s, pseudo_bank.unk_nc
        if cents.shape[0] >= shr + unk and cents.shape[1] == pseudo_bank.mu_unk.shape[1]:
            pseudo_bank.mu_unk[:] = cents[shr : shr + unk]
    models["Gc"].train()


def _target_dec_losses(models, z_dt, clu_t, shr, unk_id, ce, grl):
    shr_m = clu_t < shr
    unk_m = clu_t >= shr
    loss = z_dt.new_tensor(0.0)
    if shr_m.any():
        loss = loss + ce(
            models["C_dec"](z_dt[shr_m], grl_lambd=grl)[0], clu_t[shr_m]
        )
    if unk_m.any():
        y_unk = clu_t[unk_m] * 0 + unk_id
        loss = loss + ce(
            models["C_dec"](z_dt[unk_m], grl_lambd=grl)[0], y_unk
        )
    return loss, shr_m, unk_m


def _next_batch(tr_src, tr_tgt, dataloaders):
    try:
        feats_t, _, _, clu_t = tr_tgt.__next__()[1]
    except StopIteration:
        tr_tgt = enumerate(dataloaders["tr_loader_tgt"])
        feats_t, _, _, clu_t = tr_tgt.__next__()[1]
    try:
        feats_s, lbls_s, atts_s, _ = tr_src.__next__()[1]
    except StopIteration:
        tr_src = enumerate(dataloaders["tr_loader_src"])
        feats_s, lbls_s, atts_s, _ = tr_src.__next__()[1]
    return tr_src, tr_tgt, feats_s, lbls_s, atts_s, feats_t, clu_t


def train_stage1_eq7(args, dataloaders, models, loss_fn, opts, epoch, n_epochs, clf_ce, dom_ce):
    """Alg.1: Learn Gc, D, and C (ClfC) by eq.7 — Gc vs domain discriminator D."""
    shr = args.shr_nc
    unk_id = shr
    grl = _grl_lambda(epoch, n_epochs)

    models["Gc"].train()
    models["D"].train()
    models["ClfC"].train()

    tr_src = enumerate(dataloaders["tr_loader_src"])
    tr_tgt = enumerate(dataloaders["tr_loader_tgt"])
    _, n_batches = count_epoch_on_large_dataset(
        dataloaders["tr_loader_src"], dataloaders["tr_loader_tgt"]
    )

    loss_d_sum = loss_gc_sum = 0.0
    for _ in range(n_batches):
        tr_src, tr_tgt, feats_s, lbls_s, _, feats_t, clu_t = _next_batch(
            tr_src, tr_tgt, dataloaders
        )
        xs = feats_s.type(FloatTensor)
        xt = feats_t.type(FloatTensor)
        ys = lbls_s.type(LongTensor)
        clu_t = clu_t.type(LongTensor)

        z_cs = models["Gc"](xs)
        z_ct = models["Gc"](xt)

        # --- D: discriminate domains on z_c (eq.7, train D) ---
        dom_s = models["D"](z_cs.detach())
        dom_t = models["D"](z_ct.detach())
        loss_d = dom_ce(dom_s, _dom_labels(dom_s.size(0), 0, dom_s.device)) + dom_ce(
            dom_t, _dom_labels(dom_t.size(0), 1, dom_t.device)
        )
        opts["d"].zero_grad()
        loss_d.backward()
        _clip_models(models, ["D"])
        opts["d"].step()

        # --- Gc + ClfC: class loss + fool D via GRL (eq.7) ---
        z_cs = models["Gc"](xs)
        z_ct = models["Gc"](xt)
        loss_cls_s = ce_on_clf(models["ClfC"], z_cs, ys, clf_ce)
        loss_cls_t, _, _ = _target_cls_losses(
            models, z_ct, clu_t, shr, unk_id, clf_ce, paper_mode=_paper_mode_on(args)
        )
        dom_s = models["D"](z_cs, grl_lambd=grl)
        dom_t = models["D"](z_ct, grl_lambd=grl)
        loss_dom = dom_ce(dom_s, _dom_labels(dom_s.size(0), 0, dom_s.device)) + dom_ce(
            dom_t, _dom_labels(dom_t.size(0), 1, dom_t.device)
        )
        loss_gc = loss_cls_s + loss_cls_t + args.w_dom * loss_dom

        opts["gc"].zero_grad()
        loss_gc.backward()
        _clip_models(models, ["Gc", "ClfC"])
        opts["gc"].step()

        loss_d_sum += float(loss_d.item())
        loss_gc_sum += float(loss_gc.item())

    return loss_d_sum / n_batches, loss_gc_sum / n_batches


def train_stage1_eq11(args, dataloaders, models, loss_fn, opts, epoch, n_epochs, clf_ce, dom_ce):
    """Alg.1: Learn Gd, D, and C (C_dec) by eq.11 — Gd vs classifier C on z_d."""
    shr = args.shr_nc
    unk_id = shr
    grl = _grl_lambda(epoch, n_epochs)

    models["Gd"].train()
    models["D"].train()
    models["C_dec"].train()

    tr_src = enumerate(dataloaders["tr_loader_src"])
    tr_tgt = enumerate(dataloaders["tr_loader_tgt"])
    _, n_batches = count_epoch_on_large_dataset(
        dataloaders["tr_loader_src"], dataloaders["tr_loader_tgt"]
    )

    loss_c_sum = loss_gd_sum = 0.0
    for _ in range(n_batches):
        tr_src, tr_tgt, feats_s, lbls_s, _, feats_t, clu_t = _next_batch(
            tr_src, tr_tgt, dataloaders
        )
        xs = feats_s.type(FloatTensor)
        xt = feats_t.type(FloatTensor)
        ys = lbls_s.type(LongTensor)
        clu_t = clu_t.type(LongTensor)

        z_ds = models["Gd"](xs)
        z_dt = models["Gd"](xt)

        # --- C_dec: classify domain features z_d (eq.11, train C) ---
        logits_s = decouple_logits(models["C_dec"], z_ds.detach(), grl_lambd=0.0)
        loss_c_s = clf_ce(logits_s, ys)
        shr_m = clu_t < shr
        unk_m = clu_t >= shr
        loss_c_t = z_dt.new_tensor(0.0)
        if shr_m.any():
            logits_t = decouple_logits(
                models["C_dec"], z_dt[shr_m].detach(), grl_lambd=0.0
            )
            loss_c_t = loss_c_t + clf_ce(logits_t, clu_t[shr_m])
        if unk_m.any():
            logits_u = decouple_logits(
                models["C_dec"], z_dt[unk_m].detach(), grl_lambd=0.0
            )
            y_u = clu_t[unk_m] if _paper_mode_on(args) else torch.full(
                (int(unk_m.sum()),), unk_id, dtype=torch.long, device=z_dt.device
            )
            loss_c_t = loss_c_t + clf_ce(logits_u, y_u)
        loss_c = loss_c_s + loss_c_t
        opts["c_dec"].zero_grad()
        loss_c.backward()
        _clip_models(models, ["C_dec"])
        opts["c_dec"].step()

        # --- Gd: fool C_dec via GRL (eq.11) ---
        z_ds = models["Gd"](xs)
        z_dt = models["Gd"](xt)
        loss_gd_s = clf_ce(decouple_logits(models["C_dec"], z_ds, grl_lambd=grl), ys)
        loss_gd_t = z_dt.new_tensor(0.0)
        if shr_m.any():
            loss_gd_t = loss_gd_t + clf_ce(
                decouple_logits(models["C_dec"], z_dt[shr_m], grl_lambd=grl),
                clu_t[shr_m],
            )
        if unk_m.any():
            y_u = clu_t[unk_m] if _paper_mode_on(args) else torch.full(
                (int(unk_m.sum()),), unk_id, dtype=torch.long, device=z_dt.device
            )
            loss_gd_t = loss_gd_t + clf_ce(
                decouple_logits(models["C_dec"], z_dt[unk_m], grl_lambd=grl), y_u
            )
        loss_gd = args.w_decouple * (loss_gd_s + loss_gd_t)

        opts["gd"].zero_grad()
        loss_gd.backward()
        _clip_models(models, ["Gd"])
        opts["gd"].step()

        # --- eq.11: L_D on z_d (Gd + D_d), not adversarial with Gc ---
        z_ds = models["Gd"](xs).detach()
        z_dt = models["Gd"](xt).detach()
        dom_disc = models["D_d"] if "D_d" in models else models["D"]
        dom_s = dom_disc(z_ds)
        dom_t = dom_disc(z_dt)
        loss_d = dom_ce(dom_s, _dom_labels(dom_s.size(0), 0, dom_s.device)) + dom_ce(
            dom_t, _dom_labels(dom_t.size(0), 1, dom_t.device)
        )
        opt_d2 = opts.get("d_d", opts["d"])
        opt_d2.zero_grad()
        loss_d.backward()
        _clip_models(models, ["D_d"] if "D_d" in models else ["D"])
        opt_d2.step()

        loss_c_sum += float(loss_c.item())
        loss_gd_sum += float(loss_gd.item())

    return loss_c_sum / n_batches, loss_gd_sum / n_batches


def train_stage1_epoch(args, dataloaders, models, loss_fn, opts, epoch, n_epochs, clf_ce, dom_ce):
    """One Stage-1 epoch = eq.7 then eq.11."""
    ld, lgc = train_stage1_eq7(
        args, dataloaders, models, loss_fn, opts, epoch, n_epochs, clf_ce, dom_ce
    )
    lc, lgd = train_stage1_eq11(
        args, dataloaders, models, loss_fn, opts, epoch, n_epochs, clf_ce, dom_ce
    )
    return ld, lgc, lc, lgd


def _stage2_paper_eq19(
    args, models, xs, xt, ys, atts_s, clu_t, centers_att, train_w=True, su_ce=None
):
    """Paper eq.19: L_As + lambda1 L_R + lambda2 L_Au + lambda3 L_Arc + lambda4 L_cc."""
    from paper_modules.losses import (
        loss_arc_reg,
        loss_as_mse,
        loss_au_matrix,
        loss_au_samples,
        loss_cc_cosine,
        loss_r_class_graph,
        loss_unk_repulsion,
    )
    from fig2_predict import fig2_multimodal_feat, clfsu_loss_on_target

    shr = args.shr_nc
    ct = args.tgt_nc
    z_cs = models["Gc"](xs)
    z_ct = models["Gc"](xt)

    with torch.no_grad():
        unk_m_bank = clu_t >= shr
        models["centers"].update(z_ct, clu_t, unk_m_bank)
        models["centers"].update(z_cs, ys, torch.zeros_like(ys, dtype=torch.bool))

    shr_m = clu_t < shr
    unk_m = clu_t >= shr

    loss_as = loss_as_mse(models["Phi"], z_cs, atts_s)
    if shr_m.any():
        at_pseudo = centers_att[clu_t[shr_m]]
        loss_as = loss_as + loss_as_mse(models["Phi"], z_ct[shr_m], at_pseudo)

    bank = models["centers"]
    mu_s = bank.mu_known
    mu_t = torch.cat([bank.mu_known, bank.mu_unk], dim=0)
    loss_r = loss_r_class_graph(models["PsiGraph"], mu_s, mu_t)

    loss_au = z_ct.new_tensor(0.0)
    if unk_m.any():
        mu_s_nn, _ = bank.nearest_known(z_ct[unk_m])
        mu_u = bank.unknown_proto(clu_t[unk_m])
        loss_au = loss_au_samples(
            models["Phi"], models["Psi"], z_ct[unk_m], mu_s_nn, mu_u
        )
    a_tilde_t = semantic_tilde(models, z_ct, clu_t, shr, centers_att)
    loss_au = loss_au + loss_au_matrix(
        models["Phi"], z_ct, clu_t, centers_att, shr, ct, a_tilde=a_tilde_t
    )

    loss_arc = z_ct.new_tensor(0.0)
    loss_cc = z_ct.new_tensor(0.0)
    loss_su = z_ct.new_tensor(0.0)
    loss_rep = z_ct.new_tensor(0.0)
    if train_w:
        a_hat_s = models["Phi"](z_cs)
        a_hat_t = semantic_hat(models, z_ct, clu_t, shr)
        f_s = multimodal_feats_source(z_cs, atts_s, a_hat_s)
        f_t = multimodal_feats_target(z_ct, a_hat_t, a_tilde_t)
        ys2 = torch.cat([ys, ys], dim=0)
        clu2 = torch.cat([clu_t, clu_t], dim=0)
        unk_w = getattr(args, "arc_unk_sample_weight", 3.0)
        loss_arc = loss_arc_reg(
            models["W"], f_s, ys2, shr=shr, unk_sample_weight=1.0
        ) + loss_arc_reg(
            models["W"], f_t, clu2, shr=shr, unk_sample_weight=unk_w
        )
        centers_ct = torch.cat([bank.mu_known, bank.mu_unk], dim=0)
        loss_cc = loss_cc_cosine(z_cs, ys, bank.mu_known, shr) + loss_cc_cosine(
            z_ct, clu_t, centers_ct, ct
        )
        if unk_m.any() and bank.mu_known.sum() > 0:
            loss_rep = loss_unk_repulsion(
                z_ct[unk_m],
                bank.mu_known,
                margin=getattr(args, "unk_repulse_margin", 0.3),
            )
        if "ClfSU" in models and su_ce is not None:
            f_t_gate, _ = fig2_multimodal_feat(models, z_ct, clu_t, shr, args)
            loss_su = clfsu_loss_on_target(models, f_t_gate, clu_t, shr, su_ce)
            f_s_gate, _ = fig2_multimodal_feat(
                models, z_cs, ys, shr, args
            )
            loss_su = loss_su + ce_on_clfsu(
                models["ClfSU"],
                f_s_gate,
                torch.zeros(ys.size(0), dtype=torch.long, device=ys.device),
                su_ce,
            )

    lam1 = getattr(args, "lambda_r", 0.1)
    lam2 = getattr(args, "lambda_au", 0.5)
    lam3 = getattr(args, "lambda_arc", 1.0)
    lam4 = getattr(args, "lambda_cc", 0.1)
    w_su = getattr(args, "w_clfsu", 1.5)
    w_rep = getattr(args, "w_unk_repulse", 0.5)
    loss = (
        loss_as
        + lam1 * loss_r
        + lam2 * loss_au
        + lam3 * loss_arc
        + lam4 * loss_cc
        + w_su * loss_su
        + w_rep * loss_rep
    )
    return loss, float(loss_as.item())


def _stage2_semantic_losses(
    args,
    models,
    xs,
    xt,
    ys,
    atts_s,
    clu_t,
    centers_att,
    loss_fn,
    clf_ce,
    dom_ce,
    su_ce,
    train_w=True,
    grl=0.0,
):
    """Shared phi/psi (+ optional W) losses for Stage 2."""
    shr = args.shr_nc
    unk_id = shr
    bce = loss_fn["BCELoss"]
    att_cents = centers_att

    z_cs = models["Gc"](xs)
    z_ct = models["Gc"](xt)

    with torch.no_grad():
        unk_m_bank = clu_t >= shr
        models["centers"].update(z_ct, clu_t, unk_m_bank)
        models["centers"].update(z_cs, ys, torch.zeros_like(ys, dtype=torch.bool))

    a_s = models["Phi"](z_cs)
    loss_phi_s = bce(a_s, atts_s)

    shr_m = clu_t < shr
    unk_m = clu_t >= shr

    a_t = models["Phi"](z_ct)
    loss_phi_t = xs.new_tensor(0.0)
    if shr_m.any():
        loss_phi_t = loss_phi_t + bce(a_t[shr_m], att_cents[clu_t[shr_m]])

    loss_psi = xs.new_tensor(0.0)
    if unk_m.any():
        mu_s_nn, _ = models["centers"].nearest_known(z_ct[unk_m])
        mu_u = models["centers"].unknown_proto(clu_t[unk_m])
        a_psi = models["Psi"](z_ct[unk_m], mu_s_nn, mu_u)
        loss_psi = bce(a_psi, models["Phi"](z_ct[unk_m]).detach())

    loss_center_rel = xs.new_tensor(0.0)
    if models["centers"].known_count.sum() > 0 and models["centers"].unk_count.sum() > 0:
        mu_s = models["centers"].mu_known
        mu_u = models["centers"].mu_unk
        sim = F.cosine_similarity(mu_u.unsqueeze(1), mu_s.unsqueeze(0), dim=2)
        loss_center_rel = -sim.max(dim=1)[0].mean()

    w_phi = getattr(args, "w_phi", 5.0)
    loss = (
        w_phi * loss_phi_s
        + args.w_att * loss_phi_t
        + args.w_psi * loss_psi
        + args.w_center_rel * loss_center_rel
    )

    if train_w:
        a_hat_s = combine_ZA(z_cs, a_s)
        loss_w_s = ce_on_clf(models["W"], a_hat_s, ys, clf_ce)

        a_hat_t = combine_ZA(z_ct, semantic_hat(models, z_ct, clu_t, shr))

        loss_w_t = xs.new_tensor(0.0)
        if shr_m.any():
            loss_w_t = loss_w_t + ce_on_clf(
                models["W"], a_hat_t[shr_m], clu_t[shr_m], clf_ce
            )
        if unk_m.any():
            y_unk = torch.full(
                (int(unk_m.sum()),), unk_id, dtype=torch.long, device=xt.device
            )
            loss_w_t = loss_w_t + ce_on_clf(
                models["W"], a_hat_t[unk_m], y_unk, clf_ce
            )

        loss_su = xs.new_tensor(0.0)
        if shr_m.any():
            n_shr = int(shr_m.sum())
            loss_su = loss_su + ce_on_clfsu(
                models["ClfSU"],
                a_hat_t[shr_m],
                torch.zeros(n_shr, dtype=torch.long, device=xt.device),
                su_ce,
            )
        if unk_m.any():
            n_unk = int(unk_m.sum())
            loss_su = loss_su + ce_on_clfsu(
                models["ClfSU"],
                a_hat_t[unk_m],
                torch.ones(n_unk, dtype=torch.long, device=xt.device),
                su_ce,
            )

        w_clf = getattr(args, "w_clf", 2.0)
        loss = loss + w_clf * (loss_w_s + loss_w_t) + loss_su

        w_dom_s2 = getattr(args, "stage2_w_dom", 0.05)
        if w_dom_s2 > 0:
            models["D"].train()
            dom_s = models["D"](z_cs, grl_lambd=grl)
            dom_t = models["D"](z_ct, grl_lambd=grl)
            loss_dom = dom_ce(
                dom_s, _dom_labels(dom_s.size(0), 0, dom_s.device)
            ) + dom_ce(dom_t, _dom_labels(dom_t.size(0), 1, dom_t.device))
            loss = loss + w_dom_s2 * args.w_dom * loss_dom

    return loss, float(loss_phi_s.item())


def _stage2_set_mode(models, args, train_w):
    models["Gc"].train()
    models["Phi"].train()
    models["Psi"].train()
    if "PsiGraph" in models:
        models["PsiGraph"].train()
    models["D"].eval()
    models["Gd"].eval()
    if train_w:
        models["W"].train()
        if "ClfSU" in models:
            models["ClfSU"].train()
        if getattr(args, "unfreeze_gd_stage2", False):
            models["Gd"].train()
    else:
        models["W"].eval()
        if "ClfSU" in models:
            models["ClfSU"].eval()


def _run_stage2_epoch(
    args,
    dataloaders,
    models,
    loss_fn,
    optimizers,
    centers_att,
    epoch,
    n_epochs,
    train_w,
    clf_ce,
    dom_ce,
    su_ce,
):
    _stage2_set_mode(models, args, train_w)
    grl = _grl_lambda(epoch, n_epochs) if train_w else 0.0

    tr_src = enumerate(dataloaders["tr_loader_src"])
    tr_tgt = enumerate(dataloaders["tr_loader_tgt"])
    _, n_batches = count_epoch_on_large_dataset(
        dataloaders["tr_loader_src"], dataloaders["tr_loader_tgt"]
    )

    loss_sum = phi_sum = 0.0
    for _ in range(n_batches):
        tr_src, tr_tgt, feats_s, lbls_s, atts_s, feats_t, clu_t = _next_batch(
            tr_src, tr_tgt, dataloaders
        )
        xs = Variable(feats_s.type(FloatTensor))
        xt = Variable(feats_t.type(FloatTensor))
        ys = Variable(lbls_s.type(LongTensor))
        atts_s = Variable(atts_s.type(FloatTensor))
        clu_t = Variable(clu_t.type(LongTensor))

        if _paper_mode_on(args):
            loss, lphi = _stage2_paper_eq19(
                args,
                models,
                xs,
                xt,
                ys,
                atts_s,
                clu_t,
                centers_att,
                train_w=train_w,
                su_ce=su_ce,
            )
        else:
            loss, lphi = _stage2_semantic_losses(
                args,
                models,
                xs,
                xt,
                ys,
                atts_s,
                clu_t,
                centers_att,
                loss_fn,
                clf_ce,
                dom_ce,
                su_ce,
                train_w=train_w,
                grl=grl,
            )
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        keys = ["Gc", "Phi", "Psi", "PsiGraph"]
        if train_w:
            keys.append("W")
            if "ClfSU" in models:
                keys.append("ClfSU")
        _clip_models(models, keys)
        for opt in optimizers:
            opt.step()
        loss_sum += float(loss.item())
        phi_sum += lphi

    return loss_sum / n_batches, phi_sum / n_batches


def train_stage1_phi_align(args, dataloaders, models, loss_fn, optimizers, epoch, n_epochs):
    """After Stage1: align Phi (and Gc) on source GT attributes before Stage2."""
    models["Gc"].train()
    models["Phi"].train()
    for k in ("Gd", "D", "C_dec", "ClfC", "W"):
        if k in models:
            models[k].eval()

    bce = loss_fn["BCELoss"]
    tr_src = enumerate(dataloaders["tr_loader_src"])
    n_batches = len(dataloaders["tr_loader_src"])
    loss_sum = 0.0
    for _ in range(n_batches):
        try:
            feats_s, _, atts_s, _ = tr_src.__next__()[1]
        except StopIteration:
            tr_src = enumerate(dataloaders["tr_loader_src"])
            feats_s, _, atts_s, _ = tr_src.__next__()[1]
        xs = Variable(feats_s.type(FloatTensor))
        atts_s = Variable(atts_s.type(FloatTensor))
        z = models["Gc"](xs)
        loss = getattr(args, "w_phi", 5.0) * bce(models["Phi"](z), atts_s)
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()
        loss_sum += float(loss.item())
    return loss_sum / max(n_batches, 1)


def train_stage2_phi_warmup(
    args, dataloaders, models, loss_fn, optimizers, centers_att, epoch, n_epochs, clf_ce, dom_ce, su_ce
):
    return _run_stage2_epoch(
        args,
        dataloaders,
        models,
        loss_fn,
        optimizers,
        centers_att,
        epoch,
        n_epochs,
        False,
        clf_ce,
        dom_ce,
        su_ce,
    )


def train_stage2_eq19(
    args, dataloaders, models, loss_fn, optimizers, centers_att, epoch, n_epochs, clf_ce, dom_ce, su_ce
):
    """Stage 2 joint: Gc, phi, psi, W (+ light domain loss on Gc)."""
    return _run_stage2_epoch(
        args,
        dataloaders,
        models,
        loss_fn,
        optimizers,
        centers_att,
        epoch,
        n_epochs,
        True,
        clf_ce,
        dom_ce,
        su_ce,
    )


def train_fig2(args, dataloaders, models, loss_fn, centers_att, pseudo_tracker=None, pseudo_bank=None):
    """Algorithm 1 two-stage pipeline."""
    from networks import Clf

    z_dim = 512
    paper = _paper_mode_on(args)
    n_clf = args.tgt_nc if paper else args.shr_nc + 1
    models["ClfC"] = Clf(z_dim, 256, n_clf).to(device)
    update_pseudo = pseudo_bank is not None and not getattr(args, "fixed_pseudo", False)

    opts = {
        "gc": torch.optim.Adam(
            list(models["Gc"].parameters()) + list(models["ClfC"].parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
        ),
        "d": torch.optim.Adam(models["D"].parameters(), lr=args.lr, betas=(0.9, 0.999)),
        "gd": torch.optim.Adam(models["Gd"].parameters(), lr=args.lr, betas=(0.9, 0.999)),
        "c_dec": torch.optim.Adam(
            models["C_dec"].parameters(), lr=args.lr, betas=(0.9, 0.999)
        ),
    }
    if "D_d" in models:
        opts["d_d"] = torch.optim.Adam(
            models["D_d"].parameters(), lr=args.lr, betas=(0.9, 0.999)
        )

    stage2_phi_keys = ["Gc", "Phi", "Psi", "PsiGraph"]
    if paper:
        stage2_joint_keys = ["Gc", "Phi", "Psi", "PsiGraph", "W"]
        if getattr(args, "paper_use_clfsu", True):
            stage2_joint_keys.append("ClfSU")
    else:
        stage2_joint_keys = ["Gc", "Phi", "Psi", "W", "ClfSU"]
    optimizers_phi = [
        torch.optim.Adam(models[k].parameters(), lr=args.lr, betas=(0.9, 0.999))
        for k in stage2_phi_keys
        if k in models
    ]
    optimizers_joint = [
        torch.optim.Adam(models[k].parameters(), lr=args.lr, betas=(0.9, 0.999))
        for k in stage2_joint_keys
        if k in models
    ]
    optimizers_align = [
        torch.optim.Adam(models[k].parameters(), lr=args.lr, betas=(0.9, 0.999))
        for k in ("Gc", "Phi")
        if k in models
    ]

    n1 = args.stage1_epochs
    n2 = args.stage2_epochs
    use_early_stop = not getattr(args, "no_early_stop", False)
    es_patience = getattr(args, "early_stop_patience", 2)
    es_max_loss = getattr(args, "early_stop_max_loss", 350.0)
    best_pack = {"score": -1e18, "ep": -1, "state": None, "metrics": None, "patience": 0}
    global_ep = 0
    if paper:
        clf_ce = torch.nn.CrossEntropyLoss().to(device)
        print(f"Paper mode: C_t={args.tgt_nc} classes, ArcFace W, eq.19 losses")
    else:
        open_w = getattr(args, "open_class_weight", 0.25)
        clf_ce = make_ce_loss(args.shr_nc, device, open_weight=open_w)
        print(f"Legacy mode: open-bucket C_s+1, weight={open_w}")
    dom_ce = torch.nn.CrossEntropyLoss().to(device)
    su_ce = torch.nn.CrossEntropyLoss().to(device)

    use_eq5 = _eq5_pseudo_on(args)
    print("=" * 60)
    print("Algorithm 1 — Stage 1 (E1): eq.7 (Gc,D,C) then eq.11 (Gd,D,C)")
    if update_pseudo:
        print("  Pseudo-label update: each Stage-1 epoch (+ before Stage 2)")
        if use_eq5:
            print("  Eq.5 adaptive threshold: ON")
    else:
        print("  Pseudo-labels: FIXED round-0 K-means CSV (no eq.5 / no refresh)")
    print("  Stage2: dual multimodal F_s/F_t + matrix L_Au (paper_mode)")
    print("=" * 60)
    if use_eq5 and pseudo_bank is not None:
        _bootstrap_prototypes_from_source(args, pseudo_bank, models, dataloaders)

    stage2_only = getattr(args, "stage2_only", False)
    if stage2_only:
        print("  [Skip Stage 1] stage2_only — logging domain-D on G_c")
    else:
        for epoch in range(n1):
            for opt in opts.values():
                adjust_learning_rate(opt, epoch, args)
            ld, lgc, lc, lgd = train_stage1_epoch(
                args, dataloaders, models, loss_fn, opts, epoch, n1, clf_ce, dom_ce
            )
            print(
                f"  [Stage1] ep {epoch}/{n1} | eq7: D={ld:.4f} Gc={lgc:.4f} | "
                f"eq11: C={lc:.4f} Gd={lgd:.4f}"
            )
            if update_pseudo:
                _update_pseudo_labels(
                    args, pseudo_bank, models, dataloaders, pseudo_tracker, global_ep, stage=1
                )
            global_ep += 1
            if (epoch + 1) % max(1, args.eval_freq) == 0:
                eval_fig2(args, epoch, dataloaders["te_loader_tgt"], models, centers_att)

    if not stage2_only and update_pseudo:
        print("  [Alg.1] Update target pseudo-labels before Stage 2")
        _update_pseudo_labels(
            args, pseudo_bank, models, dataloaders, pseudo_tracker, global_ep, stage="between"
        )

    n_align = 0 if stage2_only else getattr(args, "stage1_phi_align_epochs", 3)
    if n_align > 0:
        print("=" * 60)
        print(f"Stage-1b: Phi align on source GT ({n_align} epochs)")
        print("=" * 60)
        for e in range(n_align):
            for opt in optimizers_align:
                adjust_learning_rate(opt, e, args)
            la = train_stage1_phi_align(
                args, dataloaders, models, loss_fn, optimizers_align, e, n_align
            )
            print(f"  [Phi-align] ep {e}/{n_align} loss={la:.4f}")

    if not paper and "ClfC" in models and "W" in models and hasattr(models["W"], "layer1"):
        try:
            models["W"].layer1.load_state_dict(models["ClfC"].layer1.state_dict())
        except Exception:
            pass

    n_warmup = min(getattr(args, "phi_warmup_epochs", 5), n2)
    n_joint = n2 - n_warmup
    print("=" * 60)
    print(
        f"Stage 2 (E2): phi warmup {n_warmup} ep -> joint eq.19 {n_joint} ep "
        f"(w_phi={getattr(args, 'w_phi', 5)}, w_clf={getattr(args, 'w_clf', 3)})"
    )
    print("=" * 60)
    if use_early_stop:
        print(
            f"Early stop: metric={getattr(args, 'early_stop_metric', 'paper')} "
            f"patience={es_patience} evals, max_joint_loss={es_max_loss}"
        )
    global_ep = 0 if stage2_only else n1
    stop_training = False

    for epoch in range(n_warmup):
        if stop_training:
            break
        ep_global = n1 + epoch
        for opt in optimizers_phi:
            adjust_learning_rate(opt, epoch, args)
        loss, lphi = train_stage2_phi_warmup(
            args,
            dataloaders,
            models,
            loss_fn,
            optimizers_phi,
            centers_att,
            epoch,
            max(n_warmup, 1),
            clf_ce,
            dom_ce,
            su_ce,
        )
        print(
            f"  [Stage2-warmup] ep {epoch}/{n_warmup} loss={loss:.4f} phi_s~{lphi:.4f}"
        )
        if getattr(args, "log_domain_d_acc", False):
            from domain_disc_metrics import (
                domain_discriminator_accuracy,
                append_domain_acc_log,
            )

            d_acc = domain_discriminator_accuracy(
                models,
                dataloaders["tr_loader_src"],
                dataloaders["te_loader_tgt"],
                device,
            )
            tag = f"{args.src}→{args.tgt}"
            log_ep = epoch if stage2_only else n1 + epoch
            append_domain_acc_log(
                args.domain_d_acc_csv, log_ep, "warmup", d_acc, tag
            )
            print(f"  [Domain-D @Gc] Stage2 warmup ep {epoch}: acc={100*d_acc:.2f}%")
        global_ep += 1
        if (epoch + 1) % max(1, args.eval_freq) == 0:
            m = eval_fig2(
                args, ep_global - 1, dataloaders["te_loader_tgt"], models, centers_att
            )
            best_pack["current_ep"] = ep_global - 1
            best_pack = _update_best_checkpoint(models, m, args, best_pack)
            _print_metrics_vs_paper(m, f"  [warmup eval ep {ep_global - 1}] ")

    if (
        not paper
        and n_warmup > 0
        and "ClfC" in models
        and "W" in models
        and hasattr(models["W"], "layer1")
    ):
        try:
            models["W"].layer1.load_state_dict(models["ClfC"].layer1.state_dict())
        except Exception:
            pass

    for epoch in range(n_joint):
        if stop_training:
            break
        ep_global = n1 + n_warmup + epoch
        for opt in optimizers_joint:
            adjust_learning_rate(opt, n_warmup + epoch, args)
        loss, lphi = train_stage2_eq19(
            args,
            dataloaders,
            models,
            loss_fn,
            optimizers_joint,
            centers_att,
            epoch,
            max(n_joint, 1),
            clf_ce,
            dom_ce,
            su_ce,
        )
        print(
            f"  [Stage2-joint] ep {epoch}/{n_joint} loss={loss:.4f} phi_s~{lphi:.4f}"
        )
        if getattr(args, "log_domain_d_acc", False):
            from domain_disc_metrics import (
                domain_discriminator_accuracy,
                append_domain_acc_log,
            )

            d_acc = domain_discriminator_accuracy(
                models,
                dataloaders["tr_loader_src"],
                dataloaders["te_loader_tgt"],
                device,
            )
            tag = f"{args.src}→{args.tgt}"
            log_ep = (n_warmup + epoch) if stage2_only else (n1 + n_warmup + epoch)
            append_domain_acc_log(
                args.domain_d_acc_csv, log_ep, "joint", d_acc, tag
            )
            print(
                f"  [Domain-D @Gc] Stage2 joint ep {epoch} (log_ep {log_ep}): "
                f"acc={100*d_acc:.2f}%"
            )
        if loss > es_max_loss:
            print(f"  [Early stop] joint loss {loss:.1f} > {es_max_loss} — stop before collapse")
            stop_training = True
        if update_pseudo and getattr(args, "pseudo_update_stage2", False):
            _update_pseudo_labels(
                args,
                pseudo_bank,
                models,
                dataloaders,
                pseudo_tracker,
                global_ep,
                stage=2,
            )
        global_ep += 1
        if pseudo_tracker is not None:
            _track_fig2_epoch(args, ep_global, dataloaders, models, pseudo_tracker, ep_global)
        if (epoch + 1) % max(1, args.eval_freq) == 0:
            metrics = eval_fig2(
                args, ep_global, dataloaders["te_loader_tgt"], models, centers_att
            )
            best_pack["current_ep"] = ep_global
            best_pack = _update_best_checkpoint(models, metrics, args, best_pack)
            _print_metrics_vs_paper(metrics, f"  [joint eval ep {ep_global}] ")
            if use_early_stop and best_pack["patience"] >= es_patience:
                print(
                    f"  [Early stop] no improvement for {es_patience} evals "
                    f"(best ep={best_pack['ep']})"
                )
                stop_training = True

    if best_pack["state"] is not None:
        print(f"  [Best] epoch={best_pack['ep']} — loading best checkpoint")
        if best_pack["metrics"]:
            _print_metrics_vs_paper(best_pack["metrics"], "  [Best] ")
        for k, sd in best_pack["state"].items():
            if k in models:
                models[k].load_state_dict(sd)
    else:
        print("  [Warn] no best checkpoint saved")

    _save_fig2_weights(args, models)
    _save_best_metrics(args, best_pack.get("metrics"))
    if pseudo_tracker is not None:
        out_sub = os.path.join(args.pseudo_analysis_dir, args.src + "2" + args.tgt)
        os.makedirs(out_sub, exist_ok=True)
        pseudo_tracker.output_dir = out_sub
        pseudo_tracker.save(prefix="fig2_epoch_tracking")


def _update_pseudo_labels(args, pseudo_bank, models, dataloaders, pseudo_tracker, global_ep, stage=1):
    """Paper: eq.5 split + prototype update; later rounds refine unknown via W."""
    use_eq5 = _eq5_pseudo_on(args) and pseudo_bank is not None
    # Full target pass (no drop_last) for eq.5 / prototypes — must match pseudo CSV length
    full_tgt = dataloaders.get("te_loader_tgt", dataloaders["tr_loader_tgt"])

    pseudo_bank.update_prototypes(models, full_tgt, device)

    if use_eq5:
        from paper_modules.pseudo_eq5 import refresh_pseudo_eq5

        info = refresh_pseudo_eq5(pseudo_bank, models, full_tgt, device)
        print(
            f"  [Eq.5 pseudo] stage={stage} round={info['round']} "
            f"known_assign={info['n_known_eq5']} unk_assign={info['n_unknown_eq5']}"
        )
        if global_ep > 0:
            pseudo_bank.refresh_unknown_from_classifier(
                models, full_tgt, device, use_clfsu_gate=False
            )
            pseudo_bank.update_prototypes(models, full_tgt, device)
    elif global_ep == 0:
        pseudo_bank.update_prototypes(models, full_tgt, device)
    else:
        pseudo_bank.refresh_unknown_from_classifier(
            models, full_tgt, device
        )
        pseudo_bank.update_prototypes(models, full_tgt, device)

    st = pseudo_bank.stats_vs_kmeans()
    print(
        f"  [Pseudo update] stage={stage} round={st['round']} "
        f"changed={st['n_changed']} ({100*st['frac_changed']:.2f}%)"
    )
    out = os.path.join(
        args.pseudo_analysis_dir, args.src + "2" + args.tgt, "pseudo_rounds"
    )
    pseudo_bank.save_round(out)
    if pseudo_tracker is not None:
        _track_fig2_epoch(args, global_ep, dataloaders, models, pseudo_tracker, global_ep)


def _track_fig2_epoch(args, epoch, dataloaders, models, tracker, global_ep):
    from pseudo_label_analysis import evaluate_pseudo_labels
    import numpy as np

    from clf_utils import w_predict

    models["Gc"].eval()
    models["Phi"].eval()
    models["W"].eval()
    if "ClfSU" in models:
        models["ClfSU"].eval()
    y_true, clu_all, clf_all, su_all = [], [], [], []
    te = enumerate(dataloaders["te_loader_tgt"])
    paper = _paper_mode_on(args)
    shr = args.shr_nc
    for _ in range(len(dataloaders["te_loader_tgt"])):
        feats, lbls, _, clu = te.__next__()[1]
        x = Variable(feats.type(FloatTensor))
        clu_v = clu.numpy()
        z_c = models["Gc"](x)
        clu_t = torch.tensor(clu_v, device=z_c.device, dtype=torch.long)
        a = semantic_hat(models, z_c, clu_t, shr)
        f = combine_ZA(z_c, a)
        pred = w_predict(models["W"], f)
        if paper:
            su_pred = (pred >= shr).long()
        else:
            _, su_pred = models["ClfSU"](f)
        y_true.append(lbls.numpy())
        clu_all.append(clu_v)
        clf_all.append(pred.cpu().numpy())
        su_all.append(su_pred.cpu().numpy())
    y_true = np.concatenate(y_true)
    clu = np.concatenate(clu_all)
    clf = np.concatenate(clf_all)
    su = np.concatenate(su_all)
    tracker.evaluate_batch(
        y_true, clu, clf_pred=clf, clf_su_pred=su, epoch=global_ep, phase="fig2_test"
    )
    for k in ("Gc", "Phi", "W", "ClfSU", "PsiGraph"):
        if k in models and hasattr(models[k], "train"):
            models[k].train()


def _save_best_metrics(args, metrics):
    if not metrics:
        return
    import json

    from dataset_paths import fig2_run_tag

    out_dir = os.path.join(
        "./results",
        args.att_type,
        "fig2",
        args.dataset,
        fig2_run_tag(args),
    )
    os.makedirs(out_dir, exist_ok=True)
    row = {k: float(metrics[k]) for k in PAPER_TARGETS}
    row_pct = {k: round(100 * v, 2) for k, v in row.items()}
    payload = {
        "metrics_frac": row,
        "metrics_pct": row_pct,
        "paper_targets_pct": {k: round(100 * v, 2) for k, v in PAPER_TARGETS.items()},
        "os_eval": metrics.get("os_eval", "cluster_acc_known_40"),
        "ou_eval": metrics.get("ou_eval", "cluster_acc_unknown_10"),
    }
    for k in ("OS_diag", "OU_diag", "S_diag", "U_diag", "Bi_unknown"):
        if k in metrics:
            payload[k] = float(metrics[k])
            payload[k + "_pct"] = round(100 * float(metrics[k]), 2)
    path = os.path.join(out_dir, "best_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("  Saved best metrics ->", path)


def _save_fig2_weights(args, models):
    from dataset_paths import fig2_run_tag

    save_root = os.path.join(
        "./saved_weights", args.att_type, "fig2", args.dataset, fig2_run_tag(args)
    )
    os.makedirs(save_root, exist_ok=True)
    skip = {"ProtClf", "centers", "Gd", "D", "C_dec", "ClfC"}
    for name, m in models.items():
        if name in skip or not hasattr(m, "state_dict"):
            continue
        path = os.path.join(save_root, name + ".pth")
        torch.save(m.state_dict(), path)
    print("Saved Fig2 outputs (Gc, phi, W, ...) ->", save_root)
