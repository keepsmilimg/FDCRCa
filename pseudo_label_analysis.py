"""
Pseudo-label analysis for SROSDA Step 3 (ICCV 2021).

Supports:
  1. Pseudo-label accuracy tracking (vs. ground truth, Hungarian ACC, Old/New split)
  2. K-sensitivity analysis (K-means with varying K and random seeds)
  3. K-misspecification study (K too small / correct / too large)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def cluster_acc(y_true, y_pred, return_mapping=False):
    """Hungarian-matched clustering accuracy."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = np.vstack(linear_sum_assignment(w.max() - w)).T
    acc = sum(w[i, j] for i, j in ind) / y_pred.size
    if return_mapping:
        return acc, {int(j): int(i) for i, j in ind}
    return acc


def split_cluster_acc_v2(y_true, y_pred, old_mask):
    """GCD-style Old/New ACC: one global Hungarian, then per-subset recall."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    old_mask = np.asarray(old_mask, dtype=bool)

    old_classes = set(y_true[old_mask])
    new_classes = set(y_true[~old_mask])

    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = np.vstack(linear_sum_assignment(w.max() - w)).T
    ind_map = {int(j): int(i) for i, j in ind}

    all_acc = sum(w[i, j] for i, j in ind) / y_pred.size

    old_acc, old_n = 0.0, 0
    for c in old_classes:
        old_acc += w[ind_map[c], c]
        old_n += w[:, c].sum()
    old_acc = old_acc / old_n if old_n else 0.0

    new_acc, new_n = 0.0, 0
    for c in new_classes:
        new_acc += w[ind_map[c], c]
        new_n += w[:, c].sum()
    new_acc = new_acc / new_n if new_n else 0.0

    return all_acc, old_acc, new_acc


def seen_unseen_boundary_acc(y_true, y_pseudo, shr_threshold=10):
    """
  SROSDA uses clu < shr_threshold as shared (Eqs. 4-6 confident shared branch).
  Measure how well pseudo split matches GT split (GT class id < shr_threshold).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pseudo = np.asarray(y_pseudo, dtype=np.int64)
    gt_shr = y_true < shr_threshold
    pseudo_shr = y_pseudo < shr_threshold
    return float((gt_shr == pseudo_shr).mean())


def describe_pseudo_distribution(y_pseudo, shr_threshold=10):
    """Statistics without ground truth (for existing Step-1 pseudo CSV only)."""
    y_pseudo = np.asarray(y_pseudo, dtype=np.int64)
    n = y_pseudo.size
    shr = y_pseudo < shr_threshold
    counts = {int(c): int((y_pseudo == c).sum()) for c in np.unique(y_pseudo)}
    return {
        "n_samples": n,
        "n_clusters_used": len(counts),
        "pseudo_shr_ratio": float(shr.mean()),
        "pseudo_unk_ratio": float((~shr).mean()),
        "min_label": int(y_pseudo.min()),
        "max_label": int(y_pseudo.max()),
        "per_cluster_counts": counts,
    }


def evaluate_pseudo_labels(y_true, y_pseudo, shr_threshold=10, eval_funcs=("v2",)):
    """Full metric dict for one pseudo-label assignment."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pseudo = np.asarray(y_pseudo, dtype=np.int64)
    old_mask = y_true < shr_threshold

    metrics = {
        "n_samples": int(y_true.size),
        "cluster_acc": float(cluster_acc(y_true, y_pseudo)),
        "boundary_acc": float(seen_unseen_boundary_acc(y_true, y_pseudo, shr_threshold)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pseudo)),
        "ari": float(adjusted_rand_score(y_true, y_pseudo)),
        "n_unique_pseudo": int(len(np.unique(y_pseudo))),
    }
    if "v2" in eval_funcs:
        all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true, y_pseudo, old_mask)
        metrics["all_acc_v2"] = float(all_acc)
        metrics["old_acc_v2"] = float(old_acc)
        metrics["new_acc_v2"] = float(new_acc)
    return metrics


def build_paper_round0_pseudo(feats, y_true, K_unk, shr_nc, seed=0, n_init=10):
    """
    Paper: K = #unknown classes; C_t = C_s + K.
    Shared: pseudo = GT class in [0, C_s-1]; unknown: K-means on D_t^u only -> C_s + j.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    feats = np.asarray(feats, dtype=np.float64)
    Cs = shr_nc
    n = y_true.size
    pseudo = np.zeros(n, dtype=np.int64)
    shr_m = y_true < Cs
    unk_m = ~shr_m
    pseudo[shr_m] = y_true[shr_m]
    if unk_m.any():
        raw, _, _ = run_kmeans_pseudo(feats[unk_m], K_unk, random_state=seed, n_init=n_init)
        pseudo[unk_m] = Cs + raw
    return pseudo


def map_kmeans_to_srosda_labels(labels, shr_nc=10, tgt_nc=17):
    """
    Map K cluster ids to SROSDA pseudo format: [0, shr_nc) shared, [shr_nc, tgt_nc) unknown.
    First min(K, shr_nc) clusters -> 0..; remaining -> shr_nc..
    """
    labels = np.asarray(labels, dtype=np.int64)
    unique = np.sort(np.unique(labels))
    k = len(unique)
    out = np.zeros_like(labels)
    n_shr_clusters = min(k, shr_nc)
    for new_id, old_id in enumerate(unique[:n_shr_clusters]):
        out[labels == old_id] = new_id
    unk_start = shr_nc
    for j, old_id in enumerate(unique[n_shr_clusters:]):
        out[labels == old_id] = min(unk_start + j, tgt_nc - 1)
    return out


def run_kmeans_pseudo(feats, n_clusters, random_state=0, n_init=10, sil_sample=2000):
    """Fit K-means; return labels, centers, silhouette (subsampled on large n)."""
    feats = np.asarray(feats, dtype=np.float64)
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = km.fit_predict(feats)
    sil = np.nan
    n = feats.shape[0]
    if n_clusters > 1 and n > n_clusters:
        try:
            if sil_sample and n > sil_sample:
                sil = float(
                    silhouette_score(
                        feats, labels, sample_size=min(sil_sample, n), random_state=random_state
                    )
                )
            else:
                sil = float(silhouette_score(feats, labels))
        except Exception:
            pass
    return labels, km.cluster_centers_, sil


def run_k_sensitivity(
    feats,
    y_true,
    k_values,
    shr_nc=10,
    tgt_nc=17,
    random_seeds=(0, 1, 2, 3, 4),
    n_init=10,
    paper_unk_only=True,
):
    """Sweep K (unknown-class count); record pseudo-label quality vs. GT for each K and seed."""
    rows = []
    K_true_unk = tgt_nc - shr_nc
    for k in k_values:
        for seed in random_seeds:
            if paper_unk_only:
                pseudo = build_paper_round0_pseudo(
                    feats, y_true, k, shr_nc, seed=seed, n_init=n_init
                )
                sil = np.nan
            else:
                raw_labels, _, sil = run_kmeans_pseudo(
                    feats, k, random_state=seed, n_init=n_init
                )
                pseudo = map_kmeans_to_srosda_labels(raw_labels, shr_nc=shr_nc, tgt_nc=tgt_nc)
            m = evaluate_pseudo_labels(y_true, pseudo, shr_threshold=shr_nc)
            if len(rows) % 5 == 0:
                print(f"    progress: K={k} seed={seed} acc={m['cluster_acc']:.4f}", flush=True)
            m.update(
                {
                    "K": int(k),
                    "seed": int(seed),
                    "silhouette": sil,
                    "K_match_true": int(k == (tgt_nc - shr_nc)),
                }
            )
            rows.append(m)
    return pd.DataFrame(rows)


def run_k_misspecification(
    feats,
    y_true,
    true_k,
    shr_nc=10,
    tgt_nc=17,
    under_delta=3,
    over_deltas=(3, 7),
    random_state=0,
    paper_unk_only=True,
):
    """
    Compare pseudo-label quality when K (unknown-class count) is misspecified.
    - K_under: true_k - under_delta
    - K_true:  true_k
    - K_over:  true_k + delta for each over_delta
    """
    specs = [
        ("K_under", max(2, true_k - under_delta)),
        ("K_true", true_k),
    ]
    for d in over_deltas:
        specs.append((f"K_over_{d}", true_k + d))

    rows = []
    for name, k in specs:
        if paper_unk_only:
            pseudo = build_paper_round0_pseudo(
                feats, y_true, k, shr_nc, seed=random_state, n_init=10
            )
            sil = np.nan
        else:
            raw_labels, _, sil = run_kmeans_pseudo(feats, k, random_state=random_state, n_init=10)
            pseudo = map_kmeans_to_srosda_labels(raw_labels, shr_nc=shr_nc, tgt_nc=tgt_nc)
        m = evaluate_pseudo_labels(y_true, pseudo, shr_threshold=shr_nc)
        m.update({"spec": name, "K_used": int(k), "K_true": int(true_k), "silhouette": sil})
        rows.append(m)
    return pd.DataFrame(rows)


class PseudoLabelTracker:
    """Track pseudo-label and model-prediction quality across Step-3 epochs."""

    def __init__(self, shr_threshold=10, output_dir="./results/pseudo_analysis"):
        self.shr_threshold = shr_threshold
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.history = []

    def evaluate_batch(
        self,
        y_true,
        clu_pseudo,
        clf_pred=None,
        clf_su_pred=None,
        epoch=0,
        phase="train",
    ):
        """Compare GT with dataloader pseudo (clu) and optional model predictions."""
        y_true = np.asarray(y_true, dtype=np.int64)
        clu_pseudo = np.asarray(clu_pseudo, dtype=np.int64)
        record = {
            "epoch": epoch,
            "phase": phase,
            **{f"clu_{k}": v for k, v in evaluate_pseudo_labels(y_true, clu_pseudo, self.shr_threshold).items()},
        }
        if clf_pred is not None:
            clf_pred = np.asarray(clf_pred, dtype=np.int64)
            for k, v in evaluate_pseudo_labels(y_true, clf_pred, self.shr_threshold).items():
                record[f"clf_{k}"] = v
        if clf_su_pred is not None:
            clf_su_pred = np.asarray(clf_su_pred, dtype=np.int64)
            gt_shr = (y_true < self.shr_threshold).astype(np.int64)
            record["clfsu_boundary_acc"] = float((gt_shr == clf_su_pred).mean())
        self.history.append(record)
        return record

    def save(self, prefix="step3_tracking"):
        if not self.history:
            return
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.output_dir, f"{prefix}.csv")
        df.to_csv(csv_path, index=False)
        json_path = os.path.join(self.output_dir, f"{prefix}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        return csv_path

    def plot_tracking(self, prefix="step3_tracking"):
        if not HAS_MPL or not self.history:
            return
        df = pd.DataFrame(self.history)
        metrics = [
            ("clu_cluster_acc", "Pseudo (clu) Hungarian ACC"),
            ("clu_boundary_acc", "Pseudo Seen/Unseen boundary ACC"),
            ("clu_old_acc_v2", "Pseudo Old-class ACC (v2)"),
            ("clu_new_acc_v2", "Pseudo New-class ACC (v2)"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (col, title) in zip(axes.flat, metrics):
            if col in df.columns:
                for phase, g in df.groupby("phase"):
                    ax.plot(g["epoch"], g[col], marker="o", label=phase)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{prefix}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        return path


def collect_pseudo_from_loader(dataloader, max_batches=None):
    """Gather all GT labels and pseudo cluster labels from a DataLoader."""
    y_true_list, clu_list = [], []
    for bi, batch in enumerate(dataloader):
        if max_batches is not None and bi >= max_batches:
            break
        _, lbls, _, clu = batch
        y_true_list.append(lbls.numpy())
        clu_list.append(clu.numpy())
    return np.concatenate(y_true_list), np.concatenate(clu_list)


def load_target_arrays(args):
    """Load target features, GT labels, and current pseudo labels from disk."""
    try:
        from dataset_paths import feature_file, pseudo_file, dataset_name
        if dataset_name(args) == "I2AwA" or getattr(args, "dataset", "") == "I2AwA":
            feat_path = feature_file(args, "tgt", "feats")
            lbl_path = feature_file(args, "tgt", "labels")
            pseudo_path = pseudo_file(args)
        else:
            from data_paths import paths_for_pair, n2awa_root
            base = n2awa_root(getattr(args, "data_path_target", None))
            paths = paths_for_pair(args.src, args.tgt, base)
            feat_path = paths["tgt_feats"]
            lbl_path = paths["tgt_labels"]
            pseudo_path = paths["pseudo"]
    except ImportError:
        base = args.data_path_target
        feat_path = os.path.join(base, "features", f"{args.tgt}_feats.csv")
        lbl_path = os.path.join(base, "features", f"{args.tgt}_labels.csv")
        pseudo_path = os.path.join(base, "pseudo", f"{args.src}2{args.tgt}_sample_pseudo.csv")
    missing = [p for p in (feat_path, lbl_path, pseudo_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Missing data files for analysis:\n  " + "\n  ".join(missing)
        )
    feats = pd.read_csv(feat_path, header=None).values
    y_true = pd.read_csv(lbl_path, header=None)[0].values
    y_pseudo = pd.read_csv(pseudo_path, header=None)[0].values
    return feats, y_true, y_pseudo


def save_pseudo_csv(labels, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(labels).to_csv(path, index=False, header=False)


def plot_k_sensitivity(df, output_path, true_k_unk=None):
    if not HAS_MPL or df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    agg = df.groupby("K").agg(
        cluster_acc=("cluster_acc", "mean"),
        cluster_acc_std=("cluster_acc", "std"),
        boundary_acc=("boundary_acc", "mean"),
        old_acc=("old_acc_v2", "mean"),
        new_acc=("new_acc_v2", "mean"),
        silhouette=("silhouette", "mean"),
    ).reset_index()
    k_true = int(true_k_unk) if true_k_unk is not None else None

    axes[0].errorbar(
        agg["K"],
        agg["cluster_acc"],
        yerr=agg["cluster_acc_std"].fillna(0),
        marker="o",
        capsize=3,
        color="#2563eb",
    )
    if k_true is not None and k_true in agg["K"].values:
        axes[0].axvline(k_true, color="#dc2626", ls="--", lw=1.5, label=f"paper K={k_true}")
    axes[0].set_xlabel(r"$K$ (unknown classes, $C_t=C_s+K$)")
    axes[0].set_ylabel("Hungarian ACC")
    axes[0].set_title("(a) Pseudo-label accuracy vs. $K$")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(agg["K"], agg["old_acc"], "o-", label="Old (shared)", color="#16a34a")
    axes[1].plot(agg["K"], agg["new_acc"], "s-", label="New (unknown)", color="#ea580c")
    if k_true is not None:
        axes[1].axvline(k_true, color="#dc2626", ls="--", lw=1.5)
    axes[1].set_xlabel(r"$K$ (unknown classes)")
    axes[1].set_ylabel("ACC (v2)")
    axes[1].set_title("(b) Old / New class ACC")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(agg["K"], agg["boundary_acc"], marker="^", color="#7c3aed")
    if k_true is not None:
        axes[2].axvline(k_true, color="#dc2626", ls="--", lw=1.5)
    axes[2].set_xlabel(r"$K$ (unknown classes)")
    axes[2].set_ylabel("Boundary ACC (Eqs. 4–6)")
    axes[2].set_title("(c) Shared / unknown split")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_k_misspec(df, output_path):
    if not HAS_MPL or df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - 1.5 * width, df["cluster_acc"], width, label="Hungarian ACC", color="#2563eb")
    ax.bar(x - 0.5 * width, df["boundary_acc"], width, label="Boundary ACC", color="#7c3aed")
    ax.bar(x + 0.5 * width, df["old_acc_v2"], width, label="Old ACC", color="#16a34a")
    ax.bar(x + 1.5 * width, df["new_acc_v2"], width, label="New ACC", color="#ea580c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['spec']}\n$K$={int(r['K_used'])}" for _, r in df.iterrows()], rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("K-misspecification ($K$ = unknown-class count)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_init_pseudo_bars(init_metrics, output_path):
    """Bar chart for round-0 pseudo-label metrics."""
    if not HAS_MPL or not init_metrics:
        return
    keys = ["cluster_acc", "boundary_acc", "old_acc_v2", "new_acc_v2"]
    labels = ["Hungarian\nACC", "Boundary\nACC", "Old\nACC", "New\nACC"]
    vals = [init_metrics.get(k, 0) for k in keys]
    colors = ["#2563eb", "#7c3aed", "#16a34a", "#ea580c"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Round-0 pseudo labels ($K$-means on $D_t^u$, $C_t=C_s+K$)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_summary_report(
    output_dir,
    init_metrics,
    k_sens_df=None,
    k_misspec_df=None,
    tracking_df=None,
):
    """Text summary for paper / appendix."""
    lines = [
        "=" * 72,
        "SROSDA Pseudo-Label Analysis Report (Step 3)",
        "=" * 72,
        "",
        "--- Initial pseudo labels (Step 1 K-means init) ---",
        json.dumps(init_metrics, indent=2),
        "",
    ]
    if k_sens_df is not None and not k_sens_df.empty:
        best = k_sens_df.loc[k_sens_df["cluster_acc"].idxmax()]
        lines.extend(
            [
                "--- K-sensitivity (K-means on target features) ---",
                f"Best ACC: K={int(best['K'])}, seed={int(best['seed'])}, "
                f"ACC={best['cluster_acc']:.4f}, boundary={best['boundary_acc']:.4f}",
                f"Mean ACC by K:\n{k_sens_df.groupby('K')['cluster_acc'].mean().to_string()}",
                "",
            ]
        )
    if k_misspec_df is not None and not k_misspec_df.empty:
        lines.extend(
            [
                "--- K-misspecification ---",
                k_misspec_df[
                    ["spec", "K_used", "cluster_acc", "boundary_acc", "old_acc_v2", "new_acc_v2"]
                ].to_string(index=False),
                "",
            ]
        )
    if tracking_df is not None and not tracking_df.empty:
        lines.extend(
            [
                "--- Step-3 epoch tracking (final epoch) ---",
                tracking_df.iloc[-1].to_string(),
                "",
            ]
        )
    lines.append(
        "Note: Pseudo-labels in Eqs. 4-6 use clu<shr_nc as shared and clu>=shr_nc as unknown; "
        "boundary_acc measures agreement with GT shared/unknown split."
    )
    path = os.path.join(output_dir, "pseudo_label_analysis_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
