"""
Figure-2 style networks: feature decoupling (Gc, Gd, D, C) + semantic recovery (phi, psi, W).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import Clf, ClfSU, ProtClf
from paper_modules.arcface import ArcMarginProduct
from paper_modules.psi_graph import PsiClassGraph


class GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReverseFn.apply(x, lambd)


def _mlp(in_dim, h1, out_dim, out_act=None):
    layers = [
        nn.Linear(in_dim, h1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(h1, out_dim),
    ]
    if out_act == "sigmoid":
        layers.append(nn.Sigmoid())
    elif out_act == "softmax":
        layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)


class GenC(nn.Module):
    """G_c: domain-invariant class features."""

    def __init__(self, input_size=2048, h1=1024, h2=512):
        super().__init__()
        self.net = _mlp(input_size, h1, h2)

    def forward(self, x):
        return self.net(x)


class GenD(nn.Module):
    """G_d: class-agnostic domain features."""

    def __init__(self, input_size=2048, h1=512, h2=256):
        super().__init__()
        self.net = _mlp(input_size, h1, h2)

    def forward(self, x):
        return self.net(x)


class DomainDisc(nn.Module):
    """D: domain discriminator (on z_c in eq.7, on z_d in eq.11)."""

    def __init__(self, input_size=512, h1=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h1, 2),
        )

    def forward(self, z_c, grl_lambd=1.0):
        z = grad_reverse(z_c, grl_lambd) if self.training else z_c
        return self.net(z)


class DecoupleClf(nn.Module):
    """C: tries to predict class from z_d; G_d is trained to fool C via GRL."""

    def __init__(self, input_size=256, h1=256, n_cls=41):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h1, n_cls),
        )

    def forward(self, z_d, grl_lambd=1.0):
        z = grad_reverse(z_d, grl_lambd) if self.training else z_d
        logits = self.net(z)
        return F.softmax(logits, dim=1), torch.argmax(logits, dim=1)


class PhiSemantic(nn.Module):
    """phi: z_c -> semantic attribute vector a."""

    def __init__(self, input_size=512, h1=256, a_dim=85):
        super().__init__()
        self.net = _mlp(input_size, h1, a_dim, out_act="sigmoid")

    def forward(self, z_c):
        return self.net(z_c)


class PsiRelational(nn.Module):
    """
    psi: relational mapping using known/unknown class centers in z_c space.
    a_hat_u = phi(z) + psi([z, z-mu_s*, mu_u* - mu_s*])
    """

    def __init__(self, z_dim=512, h1=256, a_dim=85):
        super().__init__()
        self.net = _mlp(z_dim * 3, h1, a_dim, out_act="sigmoid")

    def forward(self, z_c, mu_s_nn, mu_u_proto):
        rel = torch.cat([z_c, z_c - mu_s_nn, mu_u_proto - mu_s_nn], dim=1)
        return self.net(rel)


class ClassCenterBank(nn.Module):
    """EMA class centers mu_c^s (known) and mu_c^u (unknown clusters)."""

    def __init__(self, shr_nc, unk_nc, z_dim=512, momentum=0.9):
        super().__init__()
        self.shr_nc = shr_nc
        self.unk_nc = unk_nc
        self.z_dim = z_dim
        self.momentum = momentum
        self.register_buffer("mu_known", torch.zeros(shr_nc, z_dim))
        self.register_buffer("mu_unk", torch.zeros(unk_nc, z_dim))
        self.register_buffer("known_count", torch.zeros(shr_nc))
        self.register_buffer("unk_count", torch.zeros(unk_nc))

    @torch.no_grad()
    def update(self, z_c, labels, is_unknown):
        z_c = z_c.detach()
        labels = labels.detach().long()
        is_unknown = is_unknown.detach().bool()
        for i in range(z_c.size(0)):
            zi = z_c[i]
            if is_unknown[i]:
                uidx = int(labels[i].item()) - self.shr_nc
                if uidx < 0 or uidx >= self.unk_nc:
                    continue
                if self.unk_count[uidx] == 0:
                    self.mu_unk[uidx] = zi
                else:
                    self.mu_unk[uidx] = (
                        self.momentum * self.mu_unk[uidx]
                        + (1 - self.momentum) * zi
                    )
                self.unk_count[uidx] += 1
            else:
                k = int(labels[i].item())
                if k < 0 or k >= self.shr_nc:
                    continue
                if self.known_count[k] == 0:
                    self.mu_known[k] = zi
                else:
                    self.mu_known[k] = (
                        self.momentum * self.mu_known[k]
                        + (1 - self.momentum) * zi
                    )
                self.known_count[k] += 1

    def nearest_known(self, z_c):
        dist = torch.cdist(z_c, self.mu_known, p=2)
        nn_idx = dist.argmin(dim=1)
        return self.mu_known[nn_idx], nn_idx

    def unknown_proto(self, clu_unk):
        """clu_unk: cluster id in [shr_nc, tgt_nc)."""
        uidx = (clu_unk - self.shr_nc).clamp(0, self.unk_nc - 1)
        return self.mu_unk[uidx]


def semantic_hat(models, z_c, clu, shr_nc):
    """Known: phi(z). Unknown: psi relational attributes (hat{a})."""
    a_phi = models["Phi"](z_c)
    unk_m = clu >= shr_nc
    if not unk_m.any():
        return a_phi
    a_out = a_phi.clone()
    bank = models["centers"]
    z_u = z_c[unk_m]
    clu_u = clu[unk_m]
    mu_s_nn, _ = bank.nearest_known(z_u)
    mu_u = bank.unknown_proto(clu_u)
    a_out[unk_m] = models["Psi"](z_u, mu_s_nn, mu_u)
    return a_out


def semantic_tilde(models, z_c, clu, shr_nc, centers_att):
    """
    tilde{a}: known -> class attribute prototype; unknown -> Psi relational.
    Paper F_t uses G_c(x) ⊕ tilde{a} (psi + pseudo-label semantics).
    """
    a_tilde = models["Phi"](z_c).clone()
    if centers_att is not None:
        shr_m = clu < shr_nc
        if shr_m.any():
            a_tilde[shr_m] = centers_att[clu[shr_m]]
    unk_m = clu >= shr_nc
    if unk_m.any():
        bank = models["centers"]
        z_u = z_c[unk_m]
        clu_u = clu[unk_m]
        mu_s_nn, _ = bank.nearest_known(z_u)
        mu_u = bank.unknown_proto(clu_u)
        a_tilde[unk_m] = models["Psi"](z_u, mu_s_nn, mu_u)
    return a_tilde


def multimodal_feats_source(z, atts_gt, a_hat):
    """F_s = { z⊕a , z⊕hat{a} } -> stacked batch for ArcFace."""
    from utils import combine_ZA

    f1 = combine_ZA(z, atts_gt)
    f2 = combine_ZA(z, a_hat)
    return torch.cat([f1, f2], dim=0)


def multimodal_feats_target(z, a_hat, a_tilde):
    """F_t = { z⊕hat{a} , z⊕tilde{a} }."""
    from utils import combine_ZA

    f1 = combine_ZA(z, a_hat)
    f2 = combine_ZA(z, a_tilde)
    return torch.cat([f1, f2], dim=0)


def build_fig2_models(args, x_dim=2048, z_dim=512, d_dim=256, a_dim=85):
    shr = args.shr_nc
    ct = args.tgt_nc
    f_dim = z_dim + a_dim if args.combine_za else z_dim
    paper = getattr(args, "paper_mode", True)
    n_clf = ct if paper else shr + 1
    arc_s = getattr(args, "arc_s", 30.0)
    arc_m = getattr(args, "arc_m", 0.5)
    models = {
        "Gc": GenC(x_dim, 1024, z_dim),
        "Gd": GenD(x_dim, 512, d_dim),
        "D": DomainDisc(z_dim, 256),
        "D_d": DomainDisc(d_dim, 256),
        "C_dec": DecoupleClf(d_dim, 256, n_clf),
        "Phi": PhiSemantic(z_dim, 256, a_dim),
        "Psi": PsiRelational(z_dim, 256, a_dim),
        "PsiGraph": PsiClassGraph(shr, ct, hidden=256),
        "ProtClf": ProtClf(),
        "centers": ClassCenterBank(shr, args.unk_nc, z_dim),
    }
    if paper:
        models["W"] = ArcMarginProduct(f_dim, ct, s=arc_s, m=arc_m)
        models["ClfC"] = None
        if getattr(args, "paper_use_clfsu", True):
            models["ClfSU"] = ClfSU(f_dim, 256, 2)
    else:
        models["W"] = Clf(f_dim, 256, shr + 1)
        models["ClfSU"] = ClfSU(f_dim, 256, 2)
    return models
