# quantize_kmeans.py
# Single-purpose, reliable K-Means quantizer for feature vectors.
# No coupling to Gaussian/renderer code. Keep it small and sharp.

from __future__ import annotations
from typing import Tuple, Optional

import torch


__all__ = ["Quantize_kMeans"]


class Quantize_kMeans:
    """
    Minimal, robust K-Means for vector quantization.

    Features:
      - KMeans++ initialization on full data for optimal quality
      - Chunked distance computation to avoid VRAM spikes
      - FP32 distance math for stability (even if inputs are fp16/bf16)
      - Empty-cluster reseeding via farthest-point candidates
      - Stateful by design for training (centers/assignments kept in the instance)

    Public state (read-only in normal use):
      - self.centers      : (K, D) current codebook
      - self.nn_index     : (N,) last assignment indices
      - self.cluster_len  : (K, 1) cluster sizes
      - self.vec_dim      : int, feature dimensionality

    Notes:
      - This class is intentionally generic. No project-specific forward_* hooks.
      - For pure runtime usage (no training), prefer a stateless "apply" that only
        receives frozen centers and returns indices/quantized outputs.
    """

    def __init__(self, num_clusters: int = 100, num_iters: int = 10):
        # config
        self.num_clusters = max(0, int(num_clusters))
        self.num_kmeans_iters = int(num_iters)

        # runtime state
        self.nn_index = torch.empty(0)
        self.centers = torch.empty(0)
        self.vec_dim = 0
        self.cls_ids = torch.empty(0)
        self.cluster_len = torch.empty(0)

        # logging
        self.verbose = False  # flip to True for debug logs

    # ----------------------------
    # Utilities
    # ----------------------------
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    @staticmethod
    def _as2d(x: torch.Tensor) -> torch.Tensor:
        """Flatten to (N, D) robustly."""
        x = x.detach()
        return x.reshape(x.shape[0], -1)

    @staticmethod
    def _pairwise_sqdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Stable squared L2 distance without sqrt.
        Accepts (m,D) or (D,) vs (n,D) or (D,) and returns (m,n).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        # Distance math in fp32 for numeric stability under mixed-precision inputs.
        x32 = x.to(torch.float32)
        y32 = y.to(torch.float32)

        x2 = (x32 * x32).sum(dim=1, keepdim=True)               # (m,1)
        y2 = (y32 * y32).sum(dim=1, keepdim=True).transpose(0, 1)  # (1,n)
        d2 = x2 + y2 - 2.0 * (x32 @ y32.transpose(0, 1))
        return d2.clamp_min_(0)

    def get_dist(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        squared: bool = False,
        chunk_size: int = 32768,
    ) -> torch.Tensor:
        """
        Chunked distance: x:(m,D), y:(n,D) -> (m,n)
        """
        x = x.detach()
        y = y.detach()
        if x.ndim < 2 or y.ndim < 2:
            m = x.shape[0] if x.ndim >= 1 else 0
            n = y.shape[0] if y.ndim >= 1 else 0
            return torch.empty(m, n, device=x.device, dtype=x.dtype)

        m = x.shape[0]
        out = []
        for i in range(0, m, chunk_size):
            xi = x[i:i + chunk_size]
            d2 = self._pairwise_sqdist(xi, y)
            out.append(d2 if squared else torch.sqrt(d2))
        return torch.cat(out, dim=0)

    def _maybe_reset(self, D: int, device: torch.device, dtype: torch.dtype):
        """
        Reset internal state when dimension/device/dtype changes.
        """
        need_reset = (
            int(self.vec_dim) != int(D) or
            self.centers.numel() == 0 or
            (self.centers.is_cuda != (device.type == "cuda")) or
            (self.centers.dtype != dtype)
        )
        if need_reset:
            self.vec_dim = int(D)
            self.nn_index = torch.empty(0, device=device, dtype=torch.long)
            self.centers = torch.empty(0, device=device, dtype=dtype)
            self.cluster_len = torch.empty(0, device=device, dtype=torch.float32)
            self.cls_ids = torch.empty(0, device=device, dtype=torch.long)

    # ----------------------------
    # Initialization helpers
    # ----------------------------
    # Density sampling removed - use full data for better K-Means++ initialization

    @torch.no_grad()
    def _kmeans_plus_plus_init(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Pure K-Means++ initialization on full data. No sampling for optimal quality.
        feat is (N, D). Returns (K, D) initial centers.
        """
        x = feat.detach()
        if x.numel() == 0:
            return torch.empty(self.num_clusters, self.vec_dim, device=feat.device, dtype=feat.dtype)

        N, D = x.shape
        device, dtype = x.device, x.dtype
        K = min(self.num_clusters, N)
        if K <= 0:
            return torch.empty(0, D, device=device, dtype=dtype)

        # Use full data for best initialization quality
        n = N
        K_sub = K

        # first center
        first = torch.randint(0, n, (1,), device=device)
        centers = [x[first]]  # keep as (1, D), DO NOT squeeze to 1D
        mind2 = torch.full((n,), float('inf'), device=device, dtype=torch.float32)

        for _ in range(1, K_sub):
            d2 = self._pairwise_sqdist(x, centers[-1])  # (n,1)
            d2 = d2.squeeze(1)                              # (n,)
            d2 = torch.nan_to_num(d2, nan=0.0, posinf=0.0, neginf=0.0)
            mind2 = torch.minimum(mind2, d2)

            sum_mind2 = mind2.sum()
            if (not torch.isfinite(sum_mind2)) or (sum_mind2 <= 0):
                probs = torch.full_like(mind2, 1.0 / n)
            else:
                probs = (mind2 / (sum_mind2 + 1e-12)).clamp_min_(1e-12)

            next_idx = torch.multinomial(probs, 1)
            centers.append(x[next_idx])  # keep shape (1, D)

        C = torch.cat(centers, dim=0)  # (K_sub, D)

        # if we asked for more than sub-sample can provide, pad by repeats
        if K_sub < self.num_clusters:
            need = self.num_clusters - K_sub
            extra = C[torch.randint(0, C.shape[0], (need,), device=device)]
            C = torch.cat([C, extra], dim=0)

        return C

    # ----------------------------
    # Core loop
    # ----------------------------
    @torch.no_grad()
    def update_centers(self, feat: torch.Tensor):
        """Recompute centers from current assignments in the ORIGINAL space."""
        if self.nn_index is None or self.nn_index.numel() == 0:
            return
        feat = self._as2d(feat)
        device = feat.device
        K = self.num_clusters
        D = feat.shape[1]
        if K <= 0 or D <= 0:
            return

        counts = torch.bincount(self.nn_index, minlength=K).to(device=device)  # long
        centers = torch.zeros(K, D, device=device, dtype=feat.dtype)
        centers.index_add_(0, self.nn_index, feat)
        denom = counts.clamp_min(1).unsqueeze(1).to(centers.dtype)
        centers = centers / denom

        if not torch.isfinite(centers).all():
            self._log("Warning: NaN/Inf in centers. Keeping old centers.")
            return

        self.centers = centers
        self.cluster_len = counts.to(torch.float32, copy=False).unsqueeze(1)

        # Reseed empty clusters to avoid dead codes
        empty_clusters = (counts == 0).nonzero(as_tuple=False).flatten()
        if empty_clusters.numel() > 0:
            self._reseed_empty_clusters(feat)

    @torch.no_grad()
    def _reseed_empty_clusters(self, feat: torch.Tensor):
        """
        Reseed empties using farthest-point candidates (top-k by min-dist to centers),
        with capped candidate pool for big N.
        """
        feat = self._as2d(feat)
        K = self.num_clusters
        if K <= 0 or feat.numel() == 0 or self.centers.numel() == 0:
            return

        device = feat.device
        counts = torch.bincount(self.nn_index, minlength=K).to(device=device)
        empty = (counts == 0).nonzero(as_tuple=False).flatten()
        if empty.numel() == 0:
            return

        N = feat.shape[0]
        candidate_cap = min(N, 200_000)
        if N > candidate_cap:
            idx_cand = torch.randperm(N, device=device)[:candidate_cap]
            feat_cand = feat[idx_cand]
            d2 = self.get_dist(feat_cand, self.centers, squared=True, chunk_size=32768)
            mind2 = d2.min(dim=1).values
            k = min(max(1024, empty.numel() * 32), feat_cand.shape[0])
            _, idx_top = torch.topk(mind2, k, largest=True)
            pick_pool = idx_cand[idx_top]
        else:
            d2 = self.get_dist(feat, self.centers, squared=True, chunk_size=32768)
            mind2 = d2.min(dim=1).values
            k = min(max(1024, empty.numel() * 32), N)
            _, pick_pool = torch.topk(mind2, k, largest=True)

        perm = torch.randperm(pick_pool.shape[0], device=device)[:empty.numel()]
        pick = pick_pool[perm]
        self.centers[empty] = feat[pick]

    @torch.no_grad()
    def record_cluster_stats(self):
        """Save cluster sizes and point→cluster indices for later export (shapes consistent)."""
        if self.nn_index is None or self.nn_index.numel() == 0:
            K = self.num_clusters
            dev = self.centers.device if self.centers.numel() else 'cpu'
            self.cluster_len = torch.zeros(K, 1, dtype=torch.float32, device=dev)
            self.cls_ids = torch.empty(0, dtype=torch.long, device=dev)
            return
        K = self.num_clusters
        device = self.nn_index.device
        counts = torch.bincount(self.nn_index, minlength=K).to(device=device)
        self.cluster_len = counts.to(torch.float32).unsqueeze(1)
        self.cls_ids = self.nn_index

    @torch.no_grad()
    def cluster_assign(self, feat: torch.Tensor, chunk_size: int = 32768):
        """
        Assign points to clusters and (re)compute centers.
        feat may be (N, *) and will be flattened to (N, D).
        """
        feat = self._as2d(feat)
        device, dtype = feat.device, feat.dtype
        N, D = feat.shape

        # sync state
        self._maybe_reset(D, device, dtype)

        if self.num_clusters <= 0 or D <= 0:
            # trivial: keep consistent shapes
            self.nn_index = torch.zeros(N, dtype=torch.long, device=device)
            self.centers = torch.empty(self.num_clusters, D, device=device, dtype=dtype)
            self.cluster_len = torch.zeros(self.num_clusters, 1, device=device)
            return

        if N == 0:
            self.nn_index = torch.zeros(0, dtype=torch.long, device=device)
            self.centers = torch.empty(self.num_clusters, D, device=device, dtype=dtype)
            self.cluster_len = torch.zeros(self.num_clusters, 1, device=device)
            return

        # 1) variance guard
        gstd = feat.to(torch.float32).std()
        if (not torch.isfinite(gstd)) or gstd < 1e-6:
            # Degenerate: all to cluster 0; centers repeating first vector
            self.nn_index = torch.zeros(N, dtype=torch.long, device=device)
            base = feat[0:1]
            K = self.num_clusters
            if K > 0:
                self.centers = base.repeat(K, 1)
                self.cluster_len = torch.zeros(K, 1, device=device)
                self.cluster_len[0, 0] = float(N)
            else:
                self.centers = torch.empty(0, D, device=device, dtype=dtype)
                self.cluster_len = torch.empty(0, 1, device=device)
            self._log(f"[VQ] skip kmeans: low variance (std={float(gstd):.2e})")
            return

        # 2) standardize for assignment distance (fp32 math)
        mu = feat.mean(dim=0, keepdim=True)
        sigma = feat.std(dim=0, keepdim=True).clamp_min_(1e-6)
        feat_n = (feat - mu) / sigma

        # 3) init centers if needed
        if self.centers.numel() == 0 or self.centers.shape[1] != D:
            self.centers = self._kmeans_plus_plus_init(feat).to(device=device, dtype=dtype)
        centers_n = (self.centers - mu) / sigma

        prev_nn_index = None
        no_change_streak = 0
        max_no_change_iters = 2  # early stop if assignments repeat

        for _ in range(self.num_kmeans_iters):
            # 4) assignment (chunked, squared distance)
            nn_all = []
            for i in range(0, N, chunk_size):
                xi_n = feat_n[i:i + chunk_size]
                d2 = self.get_dist(xi_n, centers_n, squared=True, chunk_size=chunk_size)
                nn_all.append(torch.argmin(d2, dim=-1))
            self.nn_index = torch.cat(nn_all, dim=0).to(device)

            if prev_nn_index is not None and torch.equal(self.nn_index, prev_nn_index):
                no_change_streak += 1
                if no_change_streak >= max_no_change_iters:
                    self._log("  K-Means converged early")
                    break
            else:
                no_change_streak = 0
            prev_nn_index = self.nn_index.clone()

            # 5) update in original feature space
            self.update_centers(feat)
            if self.centers.numel() == 0:
                break
            centers_n = (self.centers - mu) / sigma

            # 6) stop if too many empties
            counts = torch.bincount(self.nn_index, minlength=self.num_clusters)
            empty_count = int((counts == 0).sum().item())
            if empty_count > self.num_clusters * 0.8:
                self._log(f"  too many empty clusters ({empty_count}/{self.num_clusters}), stop")
                break

        self.record_cluster_stats()
        final_counts = torch.bincount(self.nn_index, minlength=self.num_clusters)
        final_empty = int((final_counts == 0).sum().item())
        final_std = float(self.centers.std().item()) if self.centers.numel() else 0.0
        self._log(f"[VQ] done: empty={final_empty}/{self.num_clusters}, std={final_std:.4f}")

    # ----------------------------
    # Helpers for runtime use-cases
    # ----------------------------
    @torch.no_grad()
    def assign_with_frozen_centers(self, feat: torch.Tensor, chunk_size: int = 32768) -> torch.Tensor:
        """
        Return nearest-center indices for feat using current centers.
        Does NOT update internal nn_index/centers. Pure query.
        """
        assert self.centers.numel() > 0, "centers are empty; fit or load them first."
        x = self._as2d(feat)
        d2 = self.get_dist(x, self.centers, squared=True, chunk_size=chunk_size)
        return torch.argmin(d2, dim=-1)

    @torch.no_grad()
    def reconstruct(self, feat: torch.Tensor, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reconstruct quantized vectors. If indices are not provided, computes nearest centers.
        Returns (N, D).
        """
        assert self.centers.numel() > 0, "centers are empty; fit or load them first."
        x = self._as2d(feat)
        if indices is None:
            indices = self.assign_with_frozen_centers(x)
        return self.centers[indices]

    # Note: forward_dc/forward_frest methods moved to rvq_adapter.py
    # This keeps vq_module.py clean and focused on core K-Means logic