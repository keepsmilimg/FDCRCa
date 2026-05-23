"""
K-means pseudo labels for I2AwA (paper-aligned).

Notation:
  C_s = shared (known) class count
  K   = number of UNKNOWN classes only (not total clusters on all data)
  C_t = C_s + K

Round 0:
  - D_t^s (GT label < C_s): pseudo-label = semantic class id in [0, C_s-1]
  - D_t^u (GT label >= C_s): K-means with K on features of unknown samples only;
    cluster j -> pseudo-label C_s + j  (i.e. c in {C_s+1, ..., C_t})
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from i2awa_config import SHR_NC, UNK_NC, feature_paths, pseudo_path


def build_round0_pseudo(feats, gt_labels, K_unk, seed=0, n_init=10):
    """Return pseudo labels (N,), known prototypes (C_s,d), unknown centers (K,d)."""
    gt = np.asarray(gt_labels, dtype=np.int64)
    Cs = SHR_NC
    Ct = Cs + K_unk
    n, d = feats.shape

    pseudo = np.zeros(n, dtype=np.int64)
    shr_m = gt < Cs
    unk_m = gt >= Cs

    # Known target samples: use semantic class id (prototypes mu_u^c, c=1..C_s)
    pseudo[shr_m] = gt[shr_m]

    # Unknown D_t^u: K-means with K = |unknown classes|
    if unk_m.sum() == 0:
        raise ValueError("No unknown samples for K-means.")
    km = KMeans(n_clusters=K_unk, random_state=seed, n_init=n_init)
    clu_u = km.fit_predict(feats[unk_m])
    pseudo[unk_m] = Cs + clu_u  # C_s+1 .. C_t in 1-based; here 40..49

    mu_known = np.zeros((Cs, d), dtype=np.float64)
    for c in range(Cs):
        m = shr_m & (gt == c)
        if m.any():
            mu_known[c] = feats[m].mean(axis=0)

    mu_unk = km.cluster_centers_
    return pseudo, mu_known, mu_unk, shr_m, unk_m, Ct


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--k",
        type=int,
        default=UNK_NC,
        help="K = #unknown classes only (default 10); C_t = C_s + K",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_init", type=int, default=10)
    args = p.parse_args()

    fp = feature_paths("AwA2")
    feats = pd.read_csv(fp["feats"], header=None).values
    gt = pd.read_csv(fp["labels"], header=None)[0].values

    print(f"C_s={SHR_NC}, K_unk={args.k}, C_t={SHR_NC + args.k}")
    print(f"AwA2: {feats.shape[0]} samples, dim={feats.shape[1]}")

    pseudo, mu_known, mu_unk, shr_m, unk_m, ct = build_round0_pseudo(
        feats, gt, args.k, seed=args.seed, n_init=args.n_init
    )

    out = pseudo_path()
    pd.DataFrame(pseudo).to_csv(out, index=False, header=False)

    # Full C_t prototype table: rows 0..C_s-1 known, C_s..C_t-1 unknown K-means centers
    all_cents = np.vstack([mu_known, mu_unk])
    centers_path = out.replace("_pseudo.csv", f"_xt_clu_cents{ct}.csv")
    pd.DataFrame(all_cents).to_csv(centers_path, index=False, header=False)

    unk_only_path = out.replace("_pseudo.csv", f"_unk_K{args.k}_centers.csv")
    pd.DataFrame(mu_unk).to_csv(unk_only_path, index=False, header=False)

    print(f"Saved pseudo: {out}")
    print(f"  shared n={shr_m.sum()}, pseudo in [0,{SHR_NC-1}]")
    print(f"  unknown n={unk_m.sum()}, K-means K={args.k} -> pseudo in [{SHR_NC},{ct-1}]")
    print(f"  pseudo_shr_ratio (clu<C_s): {(pseudo < SHR_NC).mean():.4f}")
    dim = feats.shape[1]
    print(f"Saved all prototypes ({ct}x{dim}): {centers_path}")
    print(f"Saved unknown-only centers ({args.k}x{dim}): {unk_only_path}")


if __name__ == "__main__":
    main()
