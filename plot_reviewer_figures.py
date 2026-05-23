"""Generate all reviewer I3 figures (paper-aligned K on D_t^u)."""
import json
import os
import sys

import pandas as pd

from pseudo_label_analysis import (
    plot_init_pseudo_bars,
    plot_k_misspec,
    plot_k_sensitivity,
    evaluate_pseudo_labels,
    load_target_arrays,
    HAS_MPL,
)


def _plot_tracking(out_dir, csv_name, png_name, title_prefix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path = os.path.join(out_dir, csv_name)
    if not os.path.isfile(path):
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ep = df["epoch"]
    axes[0].plot(ep, df["clu_cluster_acc"], "o-", label="Pseudo (clu)", color="#2563eb", lw=2)
    if "clf_cluster_acc" in df.columns and df["clf_cluster_acc"].notna().any():
        axes[0].plot(ep, df["clf_cluster_acc"], "s--", label="Classifier", color="#ea580c", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Hungarian ACC")
    axes[0].set_title(f"{title_prefix}: label quality")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, df["clu_boundary_acc"], "o-", label="Pseudo boundary", color="#7c3aed", lw=2)
    if "clfsu_boundary_acc" in df.columns and df["clfsu_boundary_acc"].notna().any():
        axes[1].plot(ep, df["clfsu_boundary_acc"], "s--", label="ClfSU boundary", color="#16a34a", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Boundary ACC")
    axes[1].set_title(f"{title_prefix}: shared / unknown (Eqs. 4–6)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, png_name), dpi=200, bbox_inches="tight")
    plt.close()
    print("  saved", png_name)


def main():
    sys.argv = [sys.argv[0], "--dataset", "I2AwA"]
    from opts import opts
    args = opts()
    out = os.path.join("results", "pseudo_analysis", args.src + "2" + args.tgt)
    K_unk = args.tgt_nc - args.shr_nc

    if not HAS_MPL:
        print("matplotlib not installed; cannot plot.")
        return

    print("Output:", os.path.abspath(out))
    _, y_true, y_pseudo = load_target_arrays(args)
    init_m = evaluate_pseudo_labels(y_true, y_pseudo, shr_threshold=args.shr_nc)
    with open(os.path.join(out, "init_metrics.json"), "w") as f:
        json.dump(init_m, f, indent=2)

    plot_init_pseudo_bars(init_m, os.path.join(out, "init_pseudo_metrics.png"))

    k_sens = pd.read_csv(os.path.join(out, "k_sensitivity.csv"))
    k_miss = pd.read_csv(os.path.join(out, "k_misspecification.csv"))
    plot_k_sensitivity(k_sens, os.path.join(out, "k_sensitivity.png"), true_k_unk=K_unk)
    plot_k_misspec(k_miss, os.path.join(out, "k_misspecification.png"))

    _plot_tracking(out, "step3_epoch_tracking.csv", "step3_tracking.png", "SROSDA Step-3")
    _plot_tracking(out, "fig2_epoch_tracking.csv", "fig2_tracking.png", "Fig2 (ours)")

    # Combined panel for appendix
    try:
        import matplotlib.pyplot as plt
        from matplotlib import image as mpimg

        panels = [
            "init_pseudo_metrics.png",
            "k_sensitivity.png",
            "k_misspecification.png",
            "step3_tracking.png",
        ]
        fig = plt.figure(figsize=(14, 10))
        for i, name in enumerate(panels):
            p = os.path.join(out, name)
            if not os.path.isfile(p):
                continue
            ax = fig.add_subplot(2, 2, i + 1)
            ax.imshow(mpimg.imread(p))
            ax.axis("off")
            ax.set_title(name.replace("_", " ").replace(".png", ""), fontsize=10)
        plt.suptitle(f"I2AwA pseudo-label analysis ({args.src}→{args.tgt})", fontsize=12)
        plt.tight_layout()
        combo = os.path.join(out, "I3_analysis_panel.png")
        plt.savefig(combo, dpi=150, bbox_inches="tight")
        plt.close()
        print("  saved I3_analysis_panel.png")
    except Exception as e:
        print("  skip combined panel:", e)

    print("Done.")


if __name__ == "__main__":
    main()
