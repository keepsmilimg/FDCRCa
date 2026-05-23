"""
Log domain-D accuracy on G_c each Stage-2 epoch → plot Figure X.

Requires trained Fig2 weights. Runs Stage-2 only (loads checkpoint), no Stage-1.
"""
import os
import sys
import torch
from torch.autograd import Variable

from opts import opts
from prepare_data import generate_dataloader
from fig2_networks import build_fig2_models
from train_fig2 import train_fig2
from plot_stage2_domain_d import plot_domain_d_curve


def load_fig2_weights(args, models, device, required=("Gc", "D")):
    root = os.path.join(
        "./saved_weights",
        args.att_type,
        "fig2",
        args.dataset,
        args.src + "2" + args.tgt,
    )
    if not os.path.isdir(root):
        raise FileNotFoundError("Weights not found:", root)
    loaded = []
    for name, m in models.items():
        if m is None or not hasattr(m, "state_dict"):
            continue
        path = os.path.join(root, name + ".pth")
        if not os.path.isfile(path):
            continue
        try:
            sd = torch.load(path, map_location=device)
            m.load_state_dict(sd)
            loaded.append(name)
            print("  loaded", name)
        except RuntimeError as e:
            print("  skip", name, "(shape mismatch)", e)
    for r in required:
        if r not in loaded:
            raise RuntimeError(f"Required weight '{r}' not loaded from {root}")
    return root


def main():
    args = opts()
    args.log_domain_d_acc = True
    args.stage2_only = True
    args.no_early_stop = True
    args.eval_freq = 9999
    # Paper / Alg.1: Stage-2 total = stage2_epochs (default 50 = 5 warmup + 45 joint)
    if not any(a.startswith("--stage2_epochs") for a in sys.argv):
        args.stage2_epochs = 50
    if not any(a.startswith("--phi_warmup_epochs") for a in sys.argv):
        args.phi_warmup_epochs = 5

    csv_path = args.domain_d_acc_csv.replace(
        "stage2_domain_d_acc.csv", "stage2_domain_d_acc_measured.csv"
    )
    if csv_path == args.domain_d_acc_csv:
        csv_path = os.path.join(
            os.path.dirname(args.domain_d_acc_csv) or ".",
            "stage2_domain_d_acc_measured.csv",
        )
    args.domain_d_acc_csv = csv_path
    if os.path.isfile(csv_path):
        os.remove(csv_path)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    loaders = generate_dataloader(args)
    dataloaders = {
        "tr_loader_src": loaders[0],
        "tr_loader_tgt": loaders[1],
        "te_loader_tgt": loaders[2],
    }

    from i2awa_config import load_attribute_matrix

    att_cents = Variable(torch.tensor(load_attribute_matrix()).type(FloatTensor))

    models = build_fig2_models(args)
    for k, m in list(models.items()):
        if hasattr(m, "to"):
            models[k] = m.to(device)

    load_fig2_weights(args, models, device)

    loss_fn = {
        "BCELoss": torch.nn.BCELoss().to(device),
        "CELoss": torch.nn.CrossEntropyLoss().to(device),
    }

    print("=" * 60)
    print("Stage-2 only: log D(G_c) accuracy each epoch")
    print("CSV:", csv_path)
    print("=" * 60)

    train_fig2(args, dataloaders, models, loss_fn, att_cents, pseudo_bank=None)

    out_dir = os.path.dirname(csv_path)
    png = os.path.join(out_dir, "fig_stage2_domain_d_gc.png")
    plot_domain_d_curve(csv_path, png)


if __name__ == "__main__":
    main()
