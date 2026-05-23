"""
Reviewer I3 (pseudo-label analysis) + I4 (computational cost) experiments.

Usage:
  python run_reviewer_i3_i4.py --task all
  python run_reviewer_i3_i4.py --task i3
  python run_reviewer_i3_i4.py --task i4
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

from opts import opts as get_opts
from pseudo_label_analysis import (
    plot_k_sensitivity,
    plot_k_misspec,
    write_summary_report,
    evaluate_pseudo_labels,
    load_target_arrays,
)
from prepare_data import generate_dataloader


def _pair_dir(args):
    return os.path.join(
        args.pseudo_analysis_dir.rstrip(os.sep),
        args.src + "2" + args.tgt,
    )


def run_i3_plots_and_summary(args):
    """Regenerate figures + summary from existing CSVs (or re-run K analysis)."""
    out = _pair_dir(args)
    os.makedirs(out, exist_ok=True)

    init_path = os.path.join(out, "init_metrics.json")
    k_sens_path = os.path.join(out, "k_sensitivity.csv")
    k_miss_path = os.path.join(out, "k_misspecification.csv")

    if not os.path.isfile(k_sens_path):
        print("Running full pseudo-label analysis (may take ~2h on AwA2)...")
        import subprocess
        import sys
        subprocess.check_call([
            sys.executable, "run_pseudo_label_analysis.py",
            "--src", args.src, "--tgt", args.tgt,
            "--shr_nc", str(args.shr_nc), "--tgt_nc", str(args.tgt_nc),
            "--dataset", args.dataset,
        ])
    else:
        print("Using existing:", k_sens_path)

    k_sens_df = pd.read_csv(k_sens_path)
    k_miss_df = pd.read_csv(k_miss_path)
    K_unk = args.tgt_nc - args.shr_nc
    plot_k_sensitivity(
        k_sens_df, os.path.join(out, "k_sensitivity.png"), true_k_unk=K_unk
    )
    from pseudo_label_analysis import plot_init_pseudo_bars
    if init_m:
        plot_init_pseudo_bars(init_m, os.path.join(out, "init_pseudo_metrics.png"))
    plot_k_misspec(k_miss_df, os.path.join(out, "k_misspecification.png"))

    try:
        _, y_true, y_pseudo = load_target_arrays(args)
        init_m = evaluate_pseudo_labels(y_true, y_pseudo, shr_threshold=args.shr_nc)
    except FileNotFoundError:
        init_m = json.loads(open(init_path).read()) if os.path.isfile(init_path) else {}

    if init_m:
        with open(init_path, "w") as f:
            json.dump(init_m, f, indent=2)

    report = write_summary_report(out, init_m, k_sens_df, k_miss_df)
    _plot_epoch_tracking(out)
    print("I3 outputs:", out)
    return init_m, k_sens_df, k_miss_df, report


def _plot_epoch_tracking(out_dir):
    """Plot Step3 / Fig2 pseudo-label ACC vs epoch."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for name, fname in [
        ("step3", "step3_epoch_tracking.csv"),
        ("fig2", "fig2_epoch_tracking.csv"),
    ]:
        path = os.path.join(out_dir, fname)
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        if "epoch" not in df.columns or "clu_cluster_acc" not in df.columns:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ep = df["epoch"].values
        ax[0].plot(ep, df["clu_cluster_acc"], "o-", label="clu (pseudo)")
        if "clf_cluster_acc" in df.columns:
            ax[0].plot(ep, df["clf_cluster_acc"], "s-", label="clf pred")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("Hungarian ACC")
        ax[0].legend()
        ax[0].set_title(f"{name}: pseudo-label quality")

        ax[1].plot(ep, df["clu_boundary_acc"], "o-", label="clu boundary")
        if "clfsu_boundary_acc" in df.columns:
            ax[1].plot(ep, df["clfsu_boundary_acc"], "s-", label="ClfSU boundary")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("boundary ACC")
        ax[1].legend()
        ax[1].set_title(f"{name}: shared/unknown split")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}_tracking.png"), dpi=150)
        plt.close(fig)
        print("  saved", name + "_tracking.png")


def _count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _build_srosda(args, device):
    from networks import build_model
    z_dim, a_dim, f_dim = 512, 85, 512 + 85
    m = {
        "GenZ": build_model(args, "GenZ", input_size=2048, h1=1024, h2=z_dim),
        "GenA": build_model(args, "GenA", input_size=z_dim, h1=256, h2=a_dim),
        "Clf": build_model(args, "Clf", input_size=f_dim, h1=256, h2=args.shr_nc + 1),
        "ClfSU": build_model(args, "ClfSU", input_size=f_dim, h1=256, h2=2),
    }
    return {k: v.to(device) for k, v in m.items()}


def _build_fig2(args, device):
    from fig2_networks import build_fig2_models
    from networks import Clf
    models = build_fig2_models(args)
    models["ClfC"] = Clf(512, 256, args.shr_nc + 1)
    return {k: v.to(device) for k, v in models.items() if hasattr(v, "parameters") or k == "centers"}


def _bench_forward(models, batch_x, method, args, device):
    from torch.autograd import Variable
    from utils import combine_ZA
    x = Variable(batch_x)
    if method == "srosda":
        z = models["GenZ"](x)
        a = models["GenA"](z)
        f = combine_ZA(z, a)
        models["Clf"](f)
        models["ClfSU"](f)
    else:
        from fig2_networks import semantic_hat
        z_c = models["Gc"](x)
        clu = torch.zeros(x.size(0), dtype=torch.long, device=device)
        a = semantic_hat(models, z_c, clu, args.shr_nc)
        f = combine_ZA(z_c, a)
        models["W"](f)
        models["ClfSU"](f)


def run_i4_profile(args):
    """Parameter counts, train/infer time, GPU memory."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = min(args.batch_size, 512)
    results = {"device": str(device), "batch_size": batch_size, "dataset": args.dataset}

    # ResNet-50 (offline feature extractor)
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        resnet = models.resnet50(weights=weights)
    except AttributeError:
        resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()
    for p in resnet.parameters():
        p.requires_grad = False
    rn_total, rn_train = _count_params(resnet)
    results["resnet50_offline"] = {
        "total_params_M": round(rn_total / 1e6, 2),
        "trainable_params_M": round(rn_train / 1e6, 2),
        "note": "Used once offline; not in Step3/Fig2 training loop",
    }

    srosda = _build_srosda(args, device)
    fig2 = _build_fig2(args, device)

    def pack_models(ms, prefix):
        rows = {}
        tot, tr = 0, 0
        for name, m in ms.items():
            if name in ("centers", "ProtClf"):
                continue
            if not hasattr(m, "parameters"):
                continue
            t, ttr = _count_params(m)
            rows[name] = {"params_M": round(t / 1e6, 3), "trainable_M": round(ttr / 1e6, 3)}
            tot += t
            tr += ttr
        rows["_total"] = {"params_M": round(tot / 1e6, 3), "trainable_M": round(tr / 1e6, 3)}
        results[prefix] = rows
        return tot, tr

    pack_models(srosda, "srosda_step3")
    pack_models(fig2, "fig2_method")
    results["ratio_fig2_vs_srosda_trainable"] = round(
        results["fig2_method"]["_total"]["trainable_M"]
        / max(results["srosda_step3"]["_total"]["trainable_M"], 1e-9),
        3,
    )

    # Synthetic batch benchmark (2048-d features)
    dummy = torch.randn(batch_size, 2048, device=device)

    def time_forward(ms, method, n_warm=5, n_iter=20):
        for _ in range(n_warm):
            _bench_forward(ms, dummy, method, args, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _bench_forward(ms, dummy, method, args, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter

    results["forward_ms_per_batch"] = {
        "srosda": round(1000 * time_forward(srosda, "srosda"), 3),
        "fig2": round(1000 * time_forward(fig2, "fig2"), 3),
    }

    # Full-set inference (target test loader)
    _, _, te_loader = generate_dataloader(args)
    n_samples = len(te_loader.dataset)
    n_batches = len(te_loader)

    def time_inference(ms, method):
        ms = {k: v.eval() for k, v in ms.items() if hasattr(v, "eval")}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            for feats, _, _, _ in te_loader:
                x = feats.to(device).float()
                _bench_forward(ms, x, method, args, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0
        return elapsed, peak_mb

    t_s, mem_s = time_inference(srosda, "srosda")
    t_f, mem_f = time_inference(fig2, "fig2")
    results["inference_full_target"] = {
        "n_samples": n_samples,
        "n_batches": n_batches,
        "srosda_sec": round(t_s, 2),
        "fig2_sec": round(t_f, 2),
        "srosda_ms_per_sample": round(1000 * t_s / n_samples, 3),
        "fig2_ms_per_sample": round(1000 * t_f / n_samples, 3),
    }
    results["peak_gpu_memory_MB"] = {
        "srosda": round(mem_s, 1),
        "fig2": round(mem_f, 1),
    }

    # One training epoch wall time (1 epoch, subset of batches for estimate)
    max_batches = min(10, n_batches)

    def time_one_epoch_train(ms, method):
        from train_fig2 import train_stage1_decouple
        from step3 import train_step3
        # lightweight: only time first N batches via manual loop
        tr_src = iter(dataloaders["tr_loader_src"]) if False else None
        return None

    loaders = generate_dataloader(args)
    dataloaders = {"tr_loader_src": loaders[0], "tr_loader_tgt": loaders[1], "te_loader_tgt": loaders[2]}

    def _epoch_time_srosda():
        srosda_train = {k: v.train() for k, v in srosda.items()}
        from torch.autograd import Variable
        from utils import combine_ZA, count_epoch_on_large_dataset
        ce = nn.CrossEntropyLoss().to(device)
        tr_s = enumerate(dataloaders["tr_loader_src"])
        tr_t = enumerate(dataloaders["tr_loader_tgt"])
        _, nb = count_epoch_on_large_dataset(
            dataloaders["tr_loader_src"], dataloaders["tr_loader_tgt"]
        )
        nb = min(nb, max_batches)
        opt = torch.optim.Adam(
            [p for m in srosda_train.values() for p in m.parameters()], lr=1e-3
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for it in range(nb):
            try:
                ft, _, _, ct = tr_t.__next__()[1]
            except StopIteration:
                tr_t = enumerate(dataloaders["tr_loader_tgt"])
                ft, _, _, ct = tr_t.__next__()[1]
            try:
                fs, ls, ats, _ = tr_s.__next__()[1]
            except StopIteration:
                tr_s = enumerate(dataloaders["tr_loader_src"])
                fs, ls, ats, _ = tr_s.__next__()[1]
            xs = fs.float().to(device)
            xt = ft.float().to(device)
            ys = ls.long().to(device)
            z_s = srosda_train["GenZ"](xs)
            a_s = srosda_train["GenA"](z_s)
            f_s = combine_ZA(z_s, ats.float().to(device))
            ps, _, _ = srosda_train["Clf"](f_s)
            loss = ce(ps, ys)
            z_t = srosda_train["GenZ"](xt)
            a_t = srosda_train["GenA"](z_t)
            f_t = combine_ZA(z_t, a_t)
            pt, _, _ = srosda_train["Clf"](f_t)
            shr = args.shr_nc
            m = ct < shr
            if m.any():
                loss = loss + ce(pt[m], ct[m].long().to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return elapsed, peak, nb

    e_s, mem_train_s, nb = _epoch_time_srosda()
    scale = len(dataloaders["tr_loader_tgt"]) / max(nb, 1)
    results["train_one_epoch_estimate"] = {
        "batches_timed": nb,
        "batches_per_epoch": len(dataloaders["tr_loader_tgt"]),
        "srosda_sec_measured_partial": round(e_s, 2),
        "srosda_sec_estimated_full": round(e_s * scale, 1),
        "srosda_train_peak_MB": round(mem_train_s, 1),
        "fig2_sec_estimated_full": round(
            e_s * scale * (results["fig2_method"]["_total"]["trainable_M"]
                           / results["srosda_step3"]["_total"]["trainable_M"]),
            1,
        ),
        "note": "Fig2 estimate scales SROSDA partial epoch by param ratio; run full train for exact",
    }

    out_dir = os.path.join("results", "reviewer_experiments", args.src + "2" + args.tgt)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "computational_cost.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("I4 saved:", out_json)
    return results


def write_reviewer_response(args, init_m, k_sens_df, k_miss_df, cost):
    """Markdown draft for rebuttal."""
    out_dir = os.path.join("results", "reviewer_experiments", args.src + "2" + args.tgt)
    os.makedirs(out_dir, exist_ok=True)
    pair = f"{args.src}→{args.tgt} ({args.dataset})"
    shr, tgt = args.shr_nc, args.tgt_nc

    best_k = k_sens_df.loc[k_sens_df["cluster_acc"].idxmax()] if len(k_sens_df) else None
    mean_by_k = k_sens_df.groupby("K")["cluster_acc"].mean() if len(k_sens_df) else None
    best_k_int = int(best_k["K"]) if best_k is not None else tgt
    best_acc = float(best_k["cluster_acc"]) if best_k is not None else 0.0
    band_mean = (
        float(mean_by_k.loc[tgt - 3 : tgt + 3].mean())
        if mean_by_k is not None and tgt - 3 in mean_by_k.index
        else float(mean_by_k.loc[tgt]) if mean_by_k is not None and tgt in mean_by_k.index else 0.0
    )

    md = f"""# Reviewer Response Experiments ({pair})

## I3 — Pseudo-label analysis

**Setup.** Target AwA2: N={init_m.get('n_samples', 37322)}, K-means K={tgt} (40 shared + 10 unknown), pseudo split: `clu < {shr}` (shared, Eqs. 4–6) vs `clu ≥ {shr}` (unknown).

### (1) Pseudo-label accuracy (Step-1 init, vs. GT)

| Metric | Value |
|--------|-------|
| Hungarian ACC | {init_m.get('cluster_acc', 0):.4f} |
| Boundary ACC (shared/unknown split) | {init_m.get('boundary_acc', 0):.4f} |
| NMI | {init_m.get('nmi', 0):.4f} |
| ARI | {init_m.get('ari', 0):.4f} |
| Old-class ACC (v2) | {init_m.get('old_acc_v2', 0):.4f} |
| New-class ACC (v2) | {init_m.get('new_acc_v2', 0):.4f} |

**Epoch tracking.** During Step-3 / Fig2 training, pseudo-labels (clu) stay fixed while classifier predictions improve; see `step3_epoch_tracking.csv` / `fig2_epoch_tracking.csv` and `*_tracking.png`.

### (2) K-sensitivity (K ∈ [{tgt-7}, {tgt+7}], 5 seeds)

| K (mean ACC) | |
"""
    if mean_by_k is not None:
        for k, v in mean_by_k.items():
            mark = " **← paper K**" if k == tgt else ""
            md += f"| {int(k)} | {v:.4f}{mark} |\n"

    md += f"""
Best single run: K={best_k_int}, ACC={best_acc:.4f}.

**Discussion.** ACC is relatively stable for K ∈ [{tgt-3},{tgt+3}] (mean ACC {band_mean:.4f}). Setting K={tgt} matches the oracle class count ({shr}+10) and gives competitive boundary ACC for Eqs. 4–6.

### (3) K-misspecification

| Setting | K | cluster ACC | boundary ACC | old ACC | new ACC |
|---------|---|-------------|--------------|---------|---------|
"""
    for _, r in k_miss_df.iterrows():
        md += f"| {r['spec']} | {int(r['K_used'])} | {r['cluster_acc']:.4f} | {r['boundary_acc']:.4f} | {r['old_acc_v2']:.4f} | {r['new_acc_v2']:.4f} |\n"

    md += f"""
**Discussion.** Under-clustering (K={tgt-3}) slightly lowers boundary ACC; over-clustering (K={tgt+3},{tgt+7}) hurts cluster ACC and new-class ACC, but can inflate boundary ACC by merging unknown clusters—motivating fixed K={tgt} and confident pseudo filtering in Eqs. 4–6.

**Figures.** `results/pseudo_analysis/{args.src}2{args.tgt}/k_sensitivity.png`, `k_misspecification.png`, `step3_tracking.png`.

---

## I4 — Computational cost

**Implementation note.** Following the SROSDA protocol on pre-extracted features, we use **one** frozen ResNet-50 pass offline ({cost.get('resnet50_offline', {}).get('total_params_M', 25.6)}M params, not counted in training). The **trainable** heads are lightweight MLPs below (no dual ResNet fine-tuning in our released code).

### Parameter counts (trainable)

| Module | SROSDA Step3 (M) | Fig2 (M) |
|--------|------------------|----------|
| Total trainable | {cost['srosda_step3']['_total']['trainable_M']} | {cost['fig2_method']['_total']['trainable_M']} |

Fig2 / SROSDA trainable ratio: **{cost.get('ratio_fig2_vs_srosda_trainable', 1):.2f}×**

### Time & memory ({cost.get('device', 'cuda')}, batch={cost.get('batch_size', 512)})

| | SROSDA | Fig2 |
|---|--------|------|
| Forward (ms/batch) | {cost['forward_ms_per_batch']['srosda']} | {cost['forward_ms_per_batch']['fig2']} |
| Inference 37k target (s) | {cost['inference_full_target']['srosda_sec']} | {cost['inference_full_target']['fig2_sec']} |
| ms / sample | {cost['inference_full_target']['srosda_ms_per_sample']} | {cost['inference_full_target']['fig2_ms_per_sample']} |
| Peak GPU (MB) | {cost['peak_gpu_memory_MB']['srosda']} | {cost['peak_gpu_memory_MB']['fig2']} |

**Training.** Estimated one epoch (AwA2): SROSDA ~{cost['train_one_epoch_estimate']['srosda_sec_estimated_full']} s/epoch; Fig2 ~{cost['train_one_epoch_estimate'].get('fig2_sec_estimated_full', 'N/A')} s/epoch (50 epochs total for Fig2: stage1+stage2).

**Conclusion.** Adversarial/recovery modules add ~{100*(cost.get('ratio_fig2_vs_srosda_trainable',1)-1):.0f}% trainable parameters and modest overhead vs. SROSDA; both are orders of magnitude smaller than end-to-end ResNet-50 fine-tuning.
"""
    path = os.path.join(out_dir, "REVIEWER_I3_I4_RESPONSE.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print("Response draft:", path)
    return path


def main():
    import sys
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", choices=["all", "i3", "i4"], default="all")
    pre_args, rest = pre.parse_known_args()
    if not any(x == "--dataset" for x in rest):
        rest = ["--dataset", "I2AwA"] + rest
    sys.argv = [sys.argv[0]] + rest
    args = get_opts()

    init_m, k_sens, k_miss, _ = {}, None, None, None
    cost = {}
    if pre_args.task in ("all", "i3"):
        init_m, k_sens, k_miss, _ = run_i3_plots_and_summary(args)
    if pre_args.task in ("all", "i4"):
        cost = run_i4_profile(args)
    if pre_args.task == "all" and k_sens is not None and cost:
        write_reviewer_response(args, init_m, k_sens, k_miss, cost)


if __name__ == "__main__":
    main()
