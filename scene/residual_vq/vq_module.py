from __future__ import annotations
from typing import Tuple, Optional

import torch


__all__ = ["Quantize_kMeans"]


class Quantize_kMeans:
    """
    Minimal, robust K-Means for vector quantization.

    Features:
      - KMeans++ initialization with chunked mindist update
      - Chunked distance computation to avoid VRAM spikes
      - FP32 distance math for stability (even if inputs are fp16/bf16)
      - Empty-cluster reseeding via farthest-point candidates (chunk over K)
      - Stateful by design (centers/assignments kept in the instance)

    Public state:
      - self.centers      : (K, D) current codebook
      - self.nn_index     : (N,) last assignment indices
      - self.cluster_len  : (K, 1) cluster sizes
      - self.vec_dim      : int, feature dimensionality
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

        # allow TF32 on Ampere+ (FP32 matmul path speed-up, numerically safe)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

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
        Distance math in fp32 for numeric stability.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        x32 = x.to(torch.float32)
        y32 = y.to(torch.float32)

        x2 = (x32 * x32).sum(dim=1, keepdim=True)                 # (m,1)
        y2 = (y32 * y32).sum(dim=1, keepdim=True).transpose(0, 1) # (1,n)
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
        (chunks over m; be careful with n*K product externally)
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
    @torch.no_grad()
    def _kmeans_plus_plus_init(self, feat: torch.Tensor, init_chunk: int = 262_144) -> torch.Tensor:
        """
        K-Means++ initialization with chunked mindist update (memory-safe).
        feat: (N, D) -> returns (K, D)
        """
        x = feat.detach()
        if x.numel() == 0:
            return torch.empty(self.num_clusters, self.vec_dim, device=feat.device, dtype=feat.dtype)

        N, D = x.shape
        device, dtype = x.device, x.dtype
        K = min(self.num_clusters, N)
        if K <= 0:
            return torch.empty(0, D, device=device, dtype=dtype)

        # first center
        first = torch.randint(0, N, (1,), device=device)
        centers = [x[first]]  # keep as (1, D)
        mind2 = torch.full((N,), float('inf'), device=device, dtype=torch.float32)

        def update_mind2_with_center(c_vec: torch.Tensor):
            c32 = c_vec.to(torch.float32)
            for s in range(0, N, init_chunk):
                e = min(s + init_chunk, N)
                xi = x[s:e].to(torch.float32)
                d2 = ((xi - c32) ** 2).sum(dim=1)
                mind2[s:e] = torch.minimum(mind2[s:e], torch.nan_to_num(d2))

        update_mind2_with_center(centers[-1])

        for _ in range(1, K):
            sm = mind2.sum()
            if torch.isfinite(sm) and float(sm) > 0.0:
                probs = (mind2 / (sm + 1e-12)).clamp_min_(1e-12)
            else:
                probs = torch.full_like(mind2, 1.0 / float(N))
            next_idx = torch.multinomial(probs, 1)
            c = x[next_idx]
            centers.append(c)
            update_mind2_with_center(c)

        C = torch.cat(centers, dim=0).to(device=device, dtype=dtype)
        if K < self.num_clusters:
            need = self.num_clusters - K
            extra = C[torch.randint(0, C.shape[0], (need,), device=device)]
            C = torch.cat([C, extra], dim=0)
        return C

    # ----------------------------
    # Core: single-pass Lloyd (assign + accumulate)
    # ----------------------------
    @torch.no_grad()
    def _assign_and_accumulate(
        self,
        x: torch.Tensor,                  # (N,D) original space
        mu: torch.Tensor,                 # (1,D)
        sigma: torch.Tensor,              # (1,D)
        Cn: torch.Tensor,                 # (K,D) centers in standardized space (fp32)
        c2: torch.Tensor,                 # (1,K) ||Cn||^2 (fp32)
        chunk: int,                       # chunk over N
        sums_f32: torch.Tensor,           # (K,D) fp32
        cnts_i64: torch.Tensor,           # (K,)  int64
        out_idx: torch.Tensor,            # (N,)  long
        prev_idx: Optional[torch.Tensor], # (N,) or None
    ) -> int:
        """
        One pass over data: per-chunk standardize, compute argmin to centers,
        and accumulate standardized sums for center updates. Distances in fp32.
        Returns: number of changed assignments (vs prev_idx) for early stop.
        """
        changed = 0
        N = x.shape[0]
        dev = x.device
        mu = mu.to(dev, non_blocking=True)
        sigma = sigma.to(dev, non_blocking=True)
        Cn = Cn.to(dev, non_blocking=True)            # fp32
        c2 = c2.to(dev, non_blocking=True)            # fp32

        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            Xn = ((x[s:e] - mu) / sigma).to(torch.float32)     # (c,D) fp32
            x2 = (Xn * Xn).sum(dim=1, keepdim=True)            # (c,1) fp32
            dots = Xn @ Cn.t()                                 # (c,K) fp32
            dist = x2 + c2 - 2.0 * dots                        # (c,K) fp32
            idx = dist.argmin(dim=1)                           # (c,)
            out_idx[s:e] = idx
            if prev_idx is not None:
                changed += (prev_idx[s:e] != idx).sum().item()

            # accumulate in standardized space (fp32)
            sums_f32.index_add_(0, idx, Xn)
            cnts_i64.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int64))

        return changed

    # ----------------------------
    # Centers update (kept for compatibility; no longer used in hot path)
    # ----------------------------
    @torch.no_grad()
    def update_centers(self, feat: torch.Tensor):
        """Recompute centers from current assignments in the ORIGINAL space."""
        if self.nn_index is None or self.nn_index.numel() == 0:
            return
        feat = self._as2d(feat).contiguous()
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

        # reseed empties if any
        empty_clusters = (counts == 0).nonzero(as_tuple=False).flatten()
        if empty_clusters.numel() > 0:
            self._reseed_empty_clusters(feat)

    # ----------------------------
    # Empty-cluster reseed (memory-safe)
    # ----------------------------
    @torch.no_grad()
    def _reseed_empty_clusters(self, feat: torch.Tensor):
        """
        Reseed empties using farthest-point candidates with **chunk over K**.
        Never materialize a full (cap,K) matrix.
        """
        x = self._as2d(feat).contiguous()
        K = self.num_clusters
        if K <= 0 or x.numel() == 0 or self.centers.numel() == 0:
            return

        dev = x.device
        counts = torch.bincount(self.nn_index, minlength=K).to(dev)
        empties = (counts == 0).nonzero(as_tuple=False).flatten()
        if empties.numel() == 0:
            return

        N, D = x.shape
        # candidate pool to cap memory; tune if needed
        pool_cap = 200_000
        cap = min(N, pool_cap)
        idx = torch.randperm(N, device=dev)[:cap]
        sub = x.index_select(0, idx).contiguous()             # (cap,D)

        C = self.centers.contiguous()                         # (K,D) original space; fp32 math
        # choose center-block size so that (cap × kblk × fp32) ~ 512MB
        mem_cap_mb = 512
        target_bytes = int(mem_cap_mb * 1024 * 1024)
        kblk = max(128, min(K, target_bytes // max(1, cap * 4)))

        sub32 = sub.to(torch.float32)
        x2 = (sub32 * sub32).sum(dim=1, keepdim=True)         # (cap,1)
        mind = torch.full((cap,), float('inf'), device=dev, dtype=torch.float32)

        for s in range(0, K, kblk):
            Ce = C[s:s + kblk].to(torch.float32)              # (kblk,D)
            c2 = (Ce * Ce).sum(dim=1, keepdim=True).transpose(0, 1)  # (1,kblk)
            dots = sub32 @ Ce.t()                             # (cap,kblk)
            dist_blk = x2 + c2 - 2.0 * dots                   # (cap,kblk)
            blk_min = dist_blk.min(dim=1).values              # (cap,)
            mind = torch.minimum(mind, blk_min)
            del Ce, c2, dots, dist_blk, blk_min

        pick_cnt = empties.numel()
        ksel = min(max(1024, pick_cnt * 32), cap)
        _, top = torch.topk(mind, ksel, largest=True)
        picks = idx[top[:pick_cnt]]
        self.centers[empties] = x.index_select(0, picks)

    # ----------------------------
    # Stats
    # ----------------------------
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

    # ----------------------------
    # Main loop
    # ----------------------------
    @torch.no_grad()
    def cluster_assign(self, feat: torch.Tensor, chunk_size: int = 32768):
        """
        Assign points to clusters and (re)compute centers.
        Single-pass Lloyd per iteration: assign + accumulate (standardized space).
        """
        x = self._as2d(feat).contiguous()
        device, dtype = x.device, x.dtype
        N, D = x.shape

        # sync state
        self._maybe_reset(D, device, dtype)

        if self.num_clusters <= 0 or D <= 0:
            self.nn_index = torch.zeros(N, dtype=torch.long, device=device)
            self.centers = torch.empty(self.num_clusters, D, device=device, dtype=dtype)
            self.cluster_len = torch.zeros(self.num_clusters, 1, device=device)
            return

        if N == 0:
            self.nn_index = torch.zeros(0, dtype=torch.long, device=device)
            self.centers = torch.empty(self.num_clusters, D, device=device, dtype=dtype)
            self.cluster_len = torch.zeros(self.num_clusters, 1, device=device)
            return

        # variance guard
        gstd = x.to(torch.float32).std()
        if (not torch.isfinite(gstd)) or gstd < 1e-6:
            self.nn_index = torch.zeros(N, dtype=torch.long, device=device)
            K = self.num_clusters
            if K > 0:
                base = x[0:1]
                self.centers = base.repeat(K, 1)
                self.cluster_len = torch.zeros(K, 1, device=device)
                self.cluster_len[0, 0] = float(N)
            else:
                self.centers = torch.empty(0, D, device=device, dtype=dtype)
                self.cluster_len = torch.empty(0, 1, device=device)
            self._log(f"[VQ] skip kmeans: low variance (std={float(gstd):.2e})")
            return

        # standardization params (fixed for all iters)
        mu = x.mean(dim=0, keepdims=True)
        sigma = x.std(dim=0, keepdims=True).clamp_min_(1e-6)

        # init centers if needed
        if self.centers.numel() == 0 or self.centers.shape[1] != D:
            self.centers = self._kmeans_plus_plus_init(x).to(device=device, dtype=dtype)
        self.centers = self.centers.contiguous()

        Kc = self.num_clusters
        sums_f32 = torch.empty(Kc, D, device=device, dtype=torch.float32)
        cnts_i64 = torch.empty(Kc, device=device, dtype=torch.int64)
        out_idx  = torch.empty(N,  device=device, dtype=torch.long)
        prev_idx = torch.empty_like(out_idx); prev_idx.fill_(-1)

        no_change_streak = 0
        max_no_change_iters = 2

        for it in range(self.num_kmeans_iters):
            # centers in standardized space (fp32)
            Cn = ((self.centers - mu) / sigma).to(torch.float32)   # (K,D)
            c2 = (Cn * Cn).sum(dim=1, keepdim=True).transpose(0, 1)  # (1,K)

            sums_f32.zero_(); cnts_i64.zero_()
            changed = self._assign_and_accumulate(
                x, mu, sigma, Cn, c2, chunk_size, sums_f32, cnts_i64, out_idx, prev_idx
            )

            # update centers in standardized space
            centers_std = Cn.clone()
            mask = cnts_i64 > 0
            if mask.any():
                centers_std[mask] = (sums_f32[mask] / cnts_i64[mask].clamp_min(1).unsqueeze(1))

            # optional reseed (original space)
            counts_now = torch.bincount(out_idx, minlength=Kc)
            if int((counts_now == 0).sum().item()) > 0:
                # bring centers to original space first so reseed uses consistent metric
                centers_raw_tmp = centers_std * sigma + mu
                self.centers = centers_raw_tmp.to(dtype).contiguous()
                self.nn_index = out_idx
                self._reseed_empty_clusters(x)
                # recompute standardized centers after reseed
                Cn = ((self.centers - mu) / sigma).to(torch.float32)
                centers_std = Cn

            # back to original space
            centers_raw = centers_std * sigma + mu
            self.centers = centers_raw.to(dtype).contiguous()
            self.nn_index = out_idx

            # early stop: strict repeat or tiny change ratio
            if changed == 0:
                no_change_streak += 1
                if no_change_streak >= max_no_change_iters:
                    self._log("  K-Means converged early (no changes)")
                    break
            else:
                no_change_streak = 0

            change_ratio = changed / float(N)
            if change_ratio < 0.005:
                self._log(f"  Early stop: change ratio={change_ratio:.4f}")
                break

            prev_idx.copy_(out_idx)

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
        x = self._as2d(feat).contiguous()
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