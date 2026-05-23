"""
Standalone pseudo-label analysis for SROSDA (no full Step-3 training required).

Usage (from dasa/):
  python run_pseudo_label_analysis.py --src AwA --tgt real
  python run_pseudo_label_analysis.py --src AwA --tgt real --k_min 8 --k_max 24

Requires:
  data/N2AwA/features/{tgt}_feats.csv
  data/N2AwA/features/{tgt}_labels.csv
  data/N2AwA/pseudo/{src}2{tgt}_sample_pseudo.csv
"""

import os
import argparse
import numpy as np

from pseudo_label_analysis import (
    evaluate_pseudo_labels,
    run_k_sensitivity,
    run_k_misspecification,
    load_target_arrays,
    plot_k_sensitivity,
    plot_k_misspec,
    write_summary_report,
    save_pseudo_csv,
    run_kmeans_pseudo,
    map_kmeans_to_srosda_labels,
)


def parse_args():
    p = argparse.ArgumentParser(description="SROSDA pseudo-label analysis")
    from data_paths import DEFAULT_N2AWA
    from i2awa_config import I2AWA_ROOT
    p.add_argument("--data_path", type=str, default=I2AWA_ROOT)
    p.add_argument("--dataset", type=str, default="I2AwA", choices=["N2AwA", "I2AwA"])
    p.add_argument("--src", type=str, default="3D2")
    p.add_argument("--tgt", type=str, default="AwA2")
    p.add_argument("--shr_nc", type=int, default=40, help="shared class count")
    p.add_argument("--tgt_nc", type=int, default=50, help="total target classes")
    p.add_argument("--k_min", type=int, default=None)
    p.add_argument("--k_max", type=int, default=None)
    p.add_argument("--k_step", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_init", type=int, default=10, help="K-means n_init (use 3 for faster sweeps)")
    p.add_argument("--fast", action="store_true", help="fewer seeds (0,1) and n_init=3")
    p.add_argument("--under_delta", type=int, default=3)
    p.add_argument("--over_deltas", type=int, nargs="+", default=[3, 7])
    p.add_argument("--save_alt_pseudo", action="store_true",
                   help="save K-misspec pseudo CSVs under pseudo/analysis/")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fast:
        args.seeds = [0, 1]
        args.n_init = 3
    class NS:
        pass

    ns = NS()
    ns.data_path_target = args.data_path.rstrip(os.sep) + os.sep
    ns.data_path_source = ns.data_path_target
    ns.src = args.src
    ns.tgt = args.tgt
    ns.dataset = args.dataset

    out_dir = os.path.join(
        "./results/pseudo_analysis",
        args.src + "2" + args.tgt,
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print(f"SROSDA Pseudo-Label Analysis: {args.src} -> {args.tgt}")
    print("=" * 72)

    try:
        feats, y_true, y_pseudo_init = load_target_arrays(ns)
    except FileNotFoundError as e:
        print(e)
        print("\nExtract ResNet features and labels first (see README / notebooks).")
        return

    print(f"Loaded: {feats.shape[0]} samples, dim={feats.shape[1]}")

    # (1) Initial Step-1 pseudo labels
    init_metrics = evaluate_pseudo_labels(y_true, y_pseudo_init, shr_threshold=args.shr_nc)
    print("\n[1] Step-1 initialization pseudo labels:")
    for k, v in init_metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # (2) K-sensitivity
    K_unk = args.tgt_nc - args.shr_nc  # paper: K = unknown class count, C_t = C_s + K
    k_min = args.k_min if args.k_min is not None else max(2, K_unk - 3)
    k_max = args.k_max if args.k_max is not None else K_unk + 7
    k_values = list(range(k_min, k_max + 1, args.k_step))
    print(f"\n[2] K-sensitivity (unknown-class K only, C_t=C_s+K): K in {k_values}, seeds={args.seeds}")
    k_sens_df = run_k_sensitivity(
        feats,
        y_true,
        k_values,
        shr_nc=args.shr_nc,
        tgt_nc=args.tgt_nc,
        random_seeds=args.seeds,
        n_init=args.n_init,
    )
    k_sens_path = os.path.join(out_dir, "k_sensitivity.csv")
    k_sens_df.to_csv(k_sens_path, index=False)
    plot_k_sensitivity(
        k_sens_df, os.path.join(out_dir, "k_sensitivity.png"), true_k_unk=K_unk
    )
    best = k_sens_df.loc[k_sens_df["cluster_acc"].idxmax()]
    print(f"    Best: K={int(best['K'])}, ACC={best['cluster_acc']:.4f}, "
          f"boundary={best['boundary_acc']:.4f}")

    # (3) K-misspecification
    print(f"\n[3] K-misspecification (true K_unk={K_unk}, C_t={args.tgt_nc}):")
    k_misspec_df = run_k_misspecification(
        feats,
        y_true,
        true_k=K_unk,
        shr_nc=args.shr_nc,
        tgt_nc=args.tgt_nc,
        under_delta=args.under_delta,
        over_deltas=args.over_deltas,
    )
    misspec_path = os.path.join(out_dir, "k_misspecification.csv")
    k_misspec_df.to_csv(misspec_path, index=False)
    plot_k_misspec(k_misspec_df, os.path.join(out_dir, "k_misspecification.png"))
    print(k_misspec_df[["spec", "K_used", "cluster_acc", "boundary_acc", "old_acc_v2", "new_acc_v2"]].to_string(index=False))

    if args.save_alt_pseudo:
        pseudo_dir = os.path.join(args.data_path, "pseudo", "analysis")
        os.makedirs(pseudo_dir, exist_ok=True)
        for _, row in k_misspec_df.iterrows():
            k = int(row["K_used"])
            raw, _, _ = run_kmeans_pseudo(feats, k, random_state=0)
            pseudo = map_kmeans_to_srosda_labels(raw, shr_nc=args.shr_nc, tgt_nc=args.tgt_nc)
            fname = os.path.join(
                pseudo_dir,
                f"{args.src}2{args.tgt}_{row['spec']}_K{k}_pseudo.csv",
            )
            save_pseudo_csv(pseudo, fname)
            print(f"    Saved: {fname}")

    report_path = write_summary_report(
        out_dir, init_metrics, k_sens_df, k_misspec_df
    )
    print(f"\nReport: {report_path}")
    print(f"CSV/plots: {out_dir}")


if __name__ == "__main__":
    main()
