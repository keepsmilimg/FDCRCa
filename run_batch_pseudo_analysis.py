"""
Batch pseudo-label analysis for all domain pairs under dasa/data/N2AwA/pseudo/.

Works with existing pseudo CSVs; full ACC / K-sweep when features/labels exist.

  python run_batch_pseudo_analysis.py
  python run_batch_pseudo_analysis.py --data_root D:/generalized-category-discovery/dasa/data/N2AwA
"""
import os
import argparse
import pandas as pd

from data_paths import discover_domain_pairs, paths_for_pair, DEFAULT_N2AWA
from pseudo_label_analysis import (
    evaluate_pseudo_labels,
    describe_pseudo_distribution,
    run_k_sensitivity,
    run_k_misspecification,
    load_target_arrays,
    plot_k_sensitivity,
    plot_k_misspec,
    write_summary_report,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=DEFAULT_N2AWA)
    p.add_argument("--shr_nc", type=int, default=10)
    p.add_argument("--tgt_nc", type=int, default=17)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()

    pairs = discover_domain_pairs(args.data_root)
    summary_rows = []
    out_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "pseudo_analysis",
        "batch",
    )
    os.makedirs(out_root, exist_ok=True)

    print("Found", len(pairs), "domain pairs under", args.data_root)

    for src, tgt in pairs:
        paths = paths_for_pair(src, tgt, args.data_root)
        y_pseudo = pd.read_csv(paths["pseudo"], header=None)[0].values
        dist = describe_pseudo_distribution(y_pseudo, shr_threshold=args.shr_nc)
        row = {"src": src, "tgt": tgt, **dist}
        print(f"\n{src}2{tgt}: n={dist['n_samples']} shr_ratio={dist['pseudo_shr_ratio']:.3f}")

        class NS:
            pass

        ns = NS()
        ns.data_path_target = args.data_root
        ns.src = src
        ns.tgt = tgt

        pair_dir = os.path.join(out_root, f"{src}2{tgt}")
        os.makedirs(pair_dir, exist_ok=True)

        if os.path.isfile(paths["tgt_feats"]) and os.path.isfile(paths["tgt_labels"]):
            feats, y_true, _ = load_target_arrays(ns)
            init_m = evaluate_pseudo_labels(y_true, y_pseudo, shr_threshold=args.shr_nc)
            row.update({f"init_{k}": v for k, v in init_m.items() if isinstance(v, (int, float))})
            print(
                f"  init cluster_acc={init_m['cluster_acc']:.4f} "
                f"boundary={init_m['boundary_acc']:.4f}"
            )
            k_values = list(range(args.tgt_nc - 7, args.tgt_nc + 8))
            k_sens = run_k_sensitivity(
                feats, y_true, k_values, shr_nc=args.shr_nc, tgt_nc=args.tgt_nc, random_seeds=args.seeds
            )
            k_sens.to_csv(os.path.join(pair_dir, "k_sensitivity.csv"), index=False)
            plot_k_sensitivity(k_sens, os.path.join(pair_dir, "k_sensitivity.png"))
            k_miss = run_k_misspecification(
                feats, y_true, true_k=args.tgt_nc, shr_nc=args.shr_nc, tgt_nc=args.tgt_nc
            )
            k_miss.to_csv(os.path.join(pair_dir, "k_misspecification.csv"), index=False)
            plot_k_misspec(k_miss, os.path.join(pair_dir, "k_misspecification.png"))
            write_summary_report(pair_dir, init_m, k_sens, k_miss)
        else:
            pd.DataFrame([dist]).to_csv(os.path.join(pair_dir, "pseudo_distribution.csv"), index=False)
            print("  (skip K-sweep: missing features/labels — run extract_resnet_features.ipynb)")

        summary_rows.append(row)

    summary_path = os.path.join(out_root, "all_pairs_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print("\nBatch summary:", summary_path)


if __name__ == "__main__":
    main()
