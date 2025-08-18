# rvq_apply.py
# Stateless Residual K-Means VQ: pure functions for FIT (offline) and APPLY (runtime).
# - rvq_fit  : given raw features, fit multi-layer codebooks via KMeans on residuals
# - rvq_apply: given frozen codebooks, quantize features by residual stacking
#
# Notes:
#   * No coupling to Gaussian/renderer. Input is (N, D), output indices + quantized sum.
#   * Distance math is done in fp32 for stability; outputs use original dtype/device.
#   * Use chunking to avoid VRAM spikes on large N.

from __future__ import annotations
from typing import List, Sequence, Tuple, Optional, Dict

import torch

# If you have quantize_kmeans.py in the same package, this import will work.
# It is only required for rvq_fit (to build codebooks). rvq_apply has no dependency.
try:
    from .vq_module import Quantize_kMeans  # noqa: F401
    _HAS_QKM = True
except ImportError:
    _HAS_QKM = False


# ----------------------------
# Low-level distance helpers  
# ----------------------------
# Use vq_module's distance function to avoid duplication
from .vq_module import Quantize_kMeans
_pairwise_sqdist = Quantize_kMeans._pairwise_sqdist


@torch.no_grad()
def _nearest_indices_chunked(x: torch.Tensor, C: torch.Tensor, chunk: int) -> torch.Tensor:
    """
    Return nearest-center indices for x wrt codebook C, in chunks.
    x: (N, D), C: (K, D) -> idx: (N,)
    """
    N = x.shape[0]
    idx_chunks: List[torch.Tensor] = []
    for i in range(0, N, chunk):
        xi = x[i:i + chunk].to(torch.float32)
        Ci = C.to(torch.float32)
        # d2 = ||x||^2 + ||c||^2 - 2 x·c
        x2 = (xi * xi).sum(dim=1, keepdim=True)       # (m,1)
        c2 = (Ci * Ci).sum(dim=1, keepdim=True).t()   # (1,K)
        d2 = x2 + c2 - 2.0 * (xi @ Ci.t())            # (m,K)
        idx = torch.argmin(d2, dim=1)                 # (m,)
        idx_chunks.append(idx)
    return torch.cat(idx_chunks, dim=0)


# ----------------------------
# Public stateless APIs
# ----------------------------
@torch.no_grad()
def rvq_apply(
    feat_2d: torch.Tensor,
    codebooks: Sequence[torch.Tensor],
    *,
    chunk: int = 32768,
    return_residual: bool = False,
) -> Tuple[List[torch.LongTensor], torch.Tensor] | Tuple[List[torch.LongTensor], torch.Tensor, torch.Tensor]:
    """
    Apply residual vector quantization with frozen codebooks (stateless).
    Args:
      feat_2d       : (N, D) input features.
      codebooks     : [C0 (K0,D), C1 (K1,D), ...], same D across layers.
      chunk         : per-chunk size for distance evaluation.
      return_residual: if True, also returns final residual (x - sum(q_l)).

    Returns:
      ids_list: [LongTensor(N), ...]    per-layer indices
      q_sum   : (N, D)                  quantized reconstruction
      residual: (N, D)                  optional final residual
    """
    assert feat_2d.ndim == 2, "feat_2d must be (N, D)"
    if len(codebooks) == 0:
        # Zero-layer RVQ: trivial outputs
        z = torch.zeros_like(feat_2d)
        return [], z if not return_residual else ([], z, feat_2d.clone())

    # Basic shape checks
    D = int(feat_2d.shape[1])
    for i, C in enumerate(codebooks):
        assert C.ndim == 2, f"codebook L{i} must be 2D (K,D)"
        assert int(C.shape[1]) == D, f"codebook L{i} D={C.shape[1]} != {D}"

    residual = feat_2d
    q_sum = torch.zeros_like(residual)
    ids_list: List[torch.LongTensor] = []

    for C in codebooks:
        idx = _nearest_indices_chunked(residual, C, chunk=chunk)  # (N,)
        ids_list.append(idx)
        piece = C[idx]  # (N, D) on C's device/dtype
        # Align to input dtype/device if needed (usually already aligned)
        if piece.dtype != feat_2d.dtype or piece.device != feat_2d.device:
            piece = piece.to(dtype=feat_2d.dtype, device=feat_2d.device)
        q_sum = q_sum + piece
        residual = residual - piece

    return (ids_list, q_sum, residual) if return_residual else (ids_list, q_sum)


@torch.no_grad()
def rvq_fit(
    feat_2d: torch.Tensor,
    layers: Sequence[int],
    *,
    num_iters: int = 10,
    chunk_size: int = 32768,
) -> Tuple[List[torch.Tensor], List[torch.LongTensor], torch.Tensor]:
    """
    Fit residual K-Means codebooks layer-by-layer (stateful inside, stateless API).
    Requires quantize_kmeans. Produces frozen codebooks ready for rvq_apply.

    Args:
      feat_2d    : (N, D)
      layers     : e.g., [2048, 1024] -> two layers
      num_iters  : KMeans iters per layer
      chunk_size : assignment chunk size

    Returns:
      codebooks  : [C0 (K0,D), C1 (K1,D), ...]
      ids_list   : [LongTensor(N), ...] indices from the fit pass (useful for analysis)
      q_sum      : (N, D) reconstruction from the fit pass
    """
    assert _HAS_QKM, "quantize_kmeans.Quantize_kMeans not found; ensure it is importable."
    assert feat_2d.ndim == 2, "feat_2d must be (N, D)"
    assert len(layers) >= 1, "layers must be non-empty"

    residual = feat_2d.detach()
    q_sum = torch.zeros_like(residual)

    codebooks: List[torch.Tensor] = []
    ids_list: List[torch.LongTensor] = []

    for li, K in enumerate(layers):
        q = Quantize_kMeans(num_clusters=int(K), num_iters=int(num_iters))
        q.cluster_assign(residual, chunk_size=chunk_size)

        if q.centers.numel() == 0:
            # Degenerate layer (e.g., zero variance). Stop stacking.
            break

        idx = q.nn_index.clone()
        piece = q.centers[idx]
        q_sum = q_sum + piece
        residual = residual - piece

        codebooks.append(q.centers.detach().clone())
        ids_list.append(idx)

    return codebooks, ids_list, q_sum


# ----------------------------
# Convenience: dict wrappers for per-parameter (e.g., "dc"/"sh")
# ----------------------------
@torch.no_grad()
def rvq_apply_dict(
    feats: Dict[str, torch.Tensor],
    codebooks_dict: Dict[str, Sequence[torch.Tensor]],
    *,
    chunk: int = 32768,
    return_residual: bool = False,
) -> Dict[str, tuple]:
    """
    Apply RVQ per key (e.g., {"dc": Xdc, "sh": Xsh}), returning a dict of tuples.
    Returns per-key:
      if return_residual=False: (ids_list, q_sum)
      else: (ids_list, q_sum, residual)
    """
    out: Dict[str, tuple] = {}
    for k, x in feats.items():
        cbs = codebooks_dict.get(k, [])
        out[k] = rvq_apply(x, cbs, chunk=chunk, return_residual=return_residual)
    return out


@torch.no_grad()
def rvq_fit_dict(
    feats: Dict[str, torch.Tensor],
    layers_dict: Dict[str, Sequence[int]],
    *,
    num_iters: int = 10,
    chunk_size: int = 32768,
) -> Dict[str, tuple]:
    """
    Fit RVQ per key; returns dict mapping key -> (codebooks, ids_list, q_sum)
    """
    assert _HAS_QKM, "quantize_kmeans.Quantize_kMeans not found; ensure it is importable."
    out: Dict[str, tuple] = {}
    for k, x in feats.items():
        layers = layers_dict.get(k, [])
        out[k] = rvq_fit(x, layers, num_iters=num_iters, chunk_size=chunk_size)
    return out