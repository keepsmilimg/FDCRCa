"""
Figure: domain discriminator accuracy on G_c features across Stage-2 epochs.
Chance level = 50% (domain-invariant features).

Usage:
  python plot_stage2_domain_d.py --csv results/binary/fig2/I2AwA/3D22AwA2/stage2_domain_d_acc.csv
  python plot_stage2_domain_d.py --task_label "R→P"
"""
import argparse
import os
import sys

import numpy as np


def plot_domain_d_curve(csv_path, out_png, task_label=None, title=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return None

    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("empty CSV:", csv_path)

    df = df.sort_values("epoch").reset_index(drop=True)
    x = df["epoch"].values.astype(int)
    y = 100.0 * df["domain_d_acc"].astype(float).values

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        x,
        y,
        "o-",
        color="#1f77b4",
        linewidth=2.2,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2,
        label=r"$D(G_c(x))$ accuracy",
    )
    ax.axhline(50.0, color="#888888", linestyle="--", linewidth=1.5, label="Chance (50%)")
    ax.set_xlabel("Stage 2 epoch", fontsize=12)
    ax.set_ylabel("Domain discriminator accuracy (%)", fontsize=12)
    if title is None:
        title = r"Discriminator accuracy on $G_c$ features"
    ax.set_title(title, fontsize=13, pad=14)
    y_min, y_max = float(y.min()), float(y.max())
    pad = max(3.0, (y_max - y_min) * 0.25)
    ax.set_ylim(max(25.0, y_min - pad), min(65.0, y_max + pad))
    ax.set_xlim(-0.5, max(x) + 0.5)
    if len(x) >= 5:
        ax.axvline(4.5, color="#cccccc", linestyle=":", linewidth=1.2)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.35)
    fig.subplots_adjust(top=0.88)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("Saved:", os.path.abspath(out_png))
    print(f"  n={len(y)} points, mean={y.mean():.2f}%, std={y.std():.2f}%")
    return out_png


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--task_label", type=str, default="3D2$\\rightarrow$AwA2",
                   help="subtitle dataset tag (not in main title)")
    p.add_argument("--dataset", type=str, default="I2AwA")
    p.add_argument("--src", type=str, default="3D2")
    p.add_argument("--tgt", type=str, default="AwA2")
    p.add_argument("--att_type", type=str, default="binary")
    args = p.parse_args()

    if not args.csv:
        base = os.path.join(
            "results",
            args.att_type,
            "fig2",
            args.dataset,
            args.src + "2" + args.tgt,
        )
        measured = os.path.join(base, "stage2_domain_d_acc_measured.csv")
        args.csv = measured if os.path.isfile(measured) else os.path.join(
            base, "stage2_domain_d_acc.csv"
        )
    base = os.path.dirname(args.csv) or "."
    if not args.out:
        png = os.path.join(base, "fig_stage2_domain_d_gc.png")
    else:
        png = args.out if args.out.endswith(".png") else args.out + ".png"

    if not os.path.isfile(args.csv):
        print("CSV not found:", args.csv)
        print("Run with --log_domain_d_acc during training, or:")
        print("  python run_stage2_domain_curve.py")
        sys.exit(1)

    title = r"Discriminator accuracy on $G_c$ features"
    plot_domain_d_curve(args.csv, png, task_label=args.task_label, title=title)
    if args.out.endswith(".pdf") or "fig_stage2_domain_d_gc.pdf" in args.out:
        pdf = args.out if args.out.endswith(".pdf") else os.path.join(
            os.path.dirname(args.csv), "fig_stage2_domain_d_gc.pdf"
        )
        plot_domain_d_curve(args.csv, pdf.replace(".png", "_tmp.png"), task_label=args.task_label)
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            import pandas as pd

            df = pd.read_csv(args.csv)
            stage1_offset = int(df["epoch"].min())
            x = df["epoch"].values - stage1_offset
            y = 100.0 * df["domain_d_acc"].astype(float).values
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(x, y, "o-", color="#1f77b4", lw=2, ms=6)
            ax.axhline(50.0, color="#888", ls="--", lw=1.5)
            ax.set_xlabel("Stage 2 epoch")
            ax.set_ylabel("Domain discriminator accuracy (%)")
            ax.set_title(rf"Domain discriminator on $G_c$")
            ax.set_ylim(35, 65)
            ax.grid(True, alpha=0.35)
            fig.savefig(pdf, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("Saved:", pdf)
        except Exception as e:
            print("PDF skip:", e)


if __name__ == "__main__":
    main()
