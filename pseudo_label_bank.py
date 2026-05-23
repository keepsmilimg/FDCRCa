"""
Iterative pseudo-labels and target prototypes (paper Sec. pseudo-label update).

Round 0 (first epoch): K-means pseudo-labels Y_hat_t from CSV.
Later rounds: for unknown target samples D_t^u, refresh pseudo-labels using
classifier + unknown prototypes mu_u^c, c = C_s+1 .. C_t.

Known (shared) prototypes: mu_u^c, c = 1 .. C_s  (indices 0 .. C_s-1)
Unknown prototypes: mu_u^c, c = C_s+1 .. C_t     (indices C_s .. C_t-1)
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset_paths import pseudo_file


class PseudoLabelBank:
  """Mutable target pseudo-labels aligned with feature CSV row order."""

  def __init__(self, args, init_from_csv=True):
    self.args = args
    self.C_s = args.shr_nc
    self.C_t = args.tgt_nc
    self.unk_nc = args.unk_nc
    path = getattr(args, "tgt_clu_override", "") or pseudo_file(args)
    if init_from_csv and os.path.isfile(path):
      self.labels = pd.read_csv(path, header=None)[0].values.astype(np.int64).copy()
      self.kmeans_init = self.labels.copy()
    else:
      raise FileNotFoundError("K-means pseudo CSV required for round 0:", path)
    self.round_id = 0
    self.mu_known = np.zeros((self.C_s, 512), dtype=np.float32)
    self.mu_unk = np.zeros((self.unk_nc, 512), dtype=np.float32)

  def attach_dataset(self, dataset):
    """Share label array with target FeatureDataset (in-place updates)."""
    dataset.clu_labels = self.labels
    dataset.pseudo_bank = self

  def is_unknown_label(self, y):
    return np.asarray(y, dtype=np.int64) >= self.C_s

  @torch.no_grad()
  def refresh_unknown_from_classifier(
      self,
      models,
      dataloader,
      device,
      use_clfsu_gate=True,
  ):
    """
    After an epoch: update pseudo-labels for unknown samples only.
    Unknown assignment: nearest updated mu_unk (same pseudo-class), gated by ClfSU.
    """
    from fig2_networks import semantic_hat
    from utils import combine_ZA
    from clf_utils import w_predict

    shr = self.C_s
    paper = getattr(self.args, "paper_mode", True) and not getattr(
        self.args, "no_paper_mode", False
    )
    models["Gc"].eval()
    models["Phi"].eval()
    models["W"].eval()
    if "ClfSU" in models:
        models["ClfSU"].eval()

    offset = 0
    for feats, _, _, _ in dataloader:
      n = feats.size(0)
      sl = slice(offset, offset + n)
      x = feats.float().to(device)
      z_c = models["Gc"](x)
      clu_old = torch.tensor(self.labels[sl], dtype=torch.long, device=device)
      a = semantic_hat(models, z_c, clu_old, shr)
      f = combine_ZA(z_c, a)
      pred = w_predict(models["W"], f).cpu().numpy()
      if paper:
        su = (pred >= shr).astype(np.int64)
      else:
        _, su_pred = models["ClfSU"](f)
        su = su_pred.cpu().numpy()
      z_np = z_c.cpu().numpy()

      for i in range(n):
        idx = offset + i
        y0 = self.labels[idx]
        is_unk = (y0 >= shr) or (use_clfsu_gate and su[i] == 1)
        if not is_unk:
          continue
        # Classifier marks unknown: assign to pseudo-class c in [C_s, C_t-1]
        if pred[i] < shr:
          # shared prediction on unknown gate — keep K-means cluster id
          continue
        # Nearest unknown prototype (paper: same-class aggregation)
        if self.mu_unk.sum() != 0:
          d = np.linalg.norm(self.mu_unk - z_np[i], axis=1)
          k_best = int(np.argmin(d))
          self.labels[idx] = shr + k_best
        elif y0 >= shr:
          pass  # keep kmeans cluster
        else:
          self.labels[idx] = shr + int(pred[i] - shr) if pred[i] >= shr else shr

      offset += n

    self.round_id += 1
    models["Gc"].train()
    models["Phi"].train()
    models["W"].train()
    if "ClfSU" in models:
        models["ClfSU"].train()

  @torch.no_grad()
  def update_prototypes(self, models, dataloader, device):
    """mu_u^c from samples assigned to pseudo-class c on target domain."""
    shr, unk = self.C_s, self.unk_nc
    models["Gc"].eval()
    sums_k = np.zeros((shr, 512), dtype=np.float64)
    cnt_k = np.zeros(shr, dtype=np.int64)
    sums_u = np.zeros((unk, 512), dtype=np.float64)
    cnt_u = np.zeros(unk, dtype=np.int64)
    offset = 0
    for feats, _, _, _ in dataloader:
      x = feats.float().to(device)
      z = models["Gc"](x).cpu().numpy()
      for i in range(z.shape[0]):
        y = int(self.labels[offset + i])
        if y < shr:
          sums_k[y] += z[i]
          cnt_k[y] += 1
        elif y < shr + unk:
          u = y - shr
          sums_u[u] += z[i]
          cnt_u[u] += 1
      offset += z.shape[0]
    for c in range(shr):
      if cnt_k[c] > 0:
        self.mu_known[c] = (sums_k[c] / cnt_k[c]).astype(np.float32)
    for u in range(unk):
      if cnt_u[u] > 0:
        self.mu_unk[u] = (sums_u[u] / cnt_u[u]).astype(np.float32)
    # sync to Fig2 ClassCenterBank if present
    if "centers" in models and hasattr(models["centers"], "mu_known"):
      models["centers"].mu_known.copy_(
          torch.tensor(self.mu_known, device=models["centers"].mu_known.device)
      )
      models["centers"].mu_unk.copy_(
          torch.tensor(self.mu_unk, device=models["centers"].mu_unk.device)
      )
    models["Gc"].train()

  def save_round(self, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pseudo_round{self.round_id}.csv")
    pd.DataFrame(self.labels).to_csv(path, index=False, header=False)
    return path

  def stats_vs_kmeans(self):
    changed = (self.labels != self.kmeans_init).sum()
    unk_mask = self.is_unknown_label(self.labels)
    return {
      "round": self.round_id,
      "n_changed": int(changed),
      "frac_changed": float(changed / len(self.labels)),
      "n_unknown": int(unk_mask.sum()),
    }
