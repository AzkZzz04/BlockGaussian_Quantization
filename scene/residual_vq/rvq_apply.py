# rvq_apply.py
# Stateless Residual K-Means VQ: pure functions for APPLY (runtime).
# - rvq_apply: given frozen codebooks, quantize features by residual stacking

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple

import torch

# We only reuse the stable distance form and keep API parity with the project.
from .vq_module import Quantize_kMeans


# -------------------------------------
# Low-level, memory-safe NN primitives
# -------------------------------------
@torch.no_grad()
def _tile_argmin_over_columns(
    A: torch.Tensor,          # (m, D)
    B: torch.Tensor,          # (n, D)
    *,
    col_chunk: int = 65536,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each row in A, find argmin over all rows in B without materializing (m,n).
    Returns:
      best_ids  : (m,)  indices into rows of B
      best_dist2: (m,)  corresponding squared distances
    Uses stable ||a||^2 + ||b||^2 - 2 a·b in fp32.
    """
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[1]
    m = A.shape[0]

    A32 = A.to(torch.float32, copy=False).contiguous()
    B32 = B.to(torch.float32, copy=False).contiguous()

    a2 = (A32 * A32).sum(dim=1, keepdim=True)                         # (m,1)

    best_d2 = torch.full((m, 1), float("inf"), device=A.device, dtype=torch.float32)
    best_j  = torch.full((m, 1), -1,           device=A.device, dtype=torch.long)

    for s in range(0, B32.shape[0], col_chunk):
        e = min(s + col_chunk, B32.shape[0])
        Be = B32[s:e]                                                 # (b,D)
        b2 = (Be * Be).sum(dim=1, keepdim=True).transpose(0, 1)       # (1,b)
        dots = A32 @ Be.t()                                           # (m,b)
        d2   = a2 + b2 - 2.0 * dots                                   # (m,b)

        loc_best, loc_idx = d2.min(dim=1, keepdim=True)               # (m,1)
        update = loc_best < best_d2
        best_d2[update] = loc_best[update]
        best_j[update]  = (loc_idx[update] + s)

        # free temporaries early
        del Be, b2, dots, d2, loc_best, loc_idx, update

    return best_j.squeeze(1), best_d2.squeeze(1)


@torch.no_grad()
def _nearest_indices_chunked(
    x: torch.Tensor,          # (N, D)
    C: torch.Tensor,          # (K, D)
    *,
    col_chunk: int = 65536,
) -> torch.Tensor:
    """
    Return nearest-center indices for x w.r.t. codebook C, using column-chunked argmin.
    x: (N, D), C: (K, D) -> idx: (N,)
    """
    idx, _ = _tile_argmin_over_columns(x, C, col_chunk=col_chunk)
    return idx


# ----------------------------
# Public stateless APIs
# ----------------------------
@torch.no_grad()
def rvq_apply(
    feat_2d: torch.Tensor,
    codebooks: Sequence[torch.Tensor],
    *,
    chunk: int = 32768,               # kept for API parity; used as row-chunk when we extend to 2D tiling
    return_residual: bool = False,
) -> Tuple[List[torch.LongTensor], torch.Tensor] | Tuple[List[torch.LongTensor], torch.Tensor, torch.Tensor]:
    """
    Apply residual vector quantization with frozen codebooks (stateless).

    Args:
      feat_2d        : (N, D) input features.
      codebooks      : [C0 (K0,D), C1 (K1,D), ...], same D across layers.
      chunk          : kept for API compatibility; not directly used in column tiling.
      return_residual: if True, also returns final residual (x - sum(q_l)).

    Returns:
      ids_list: [LongTensor(N), ...]    per-layer indices
      q_sum   : (N, D)                  quantized reconstruction
      residual: (N, D)                  optional final residual
    """
    assert feat_2d.ndim == 2, "feat_2d must be (N, D)"
    if len(codebooks) == 0:
        z = torch.zeros_like(feat_2d)
        return ([], z, feat_2d.clone()) if return_residual else ([], z)

    # Basic checks
    D = int(feat_2d.shape[1])
    for i, C in enumerate(codebooks):
        assert C.ndim == 2, f"codebook L{i} must be 2D (K,D)"
        assert int(C.shape[1]) == D, f"codebook L{i} D={C.shape[1]} != {D}"

    # Make sure tensors are contiguous to avoid implicit copies
    x = feat_2d.contiguous()
    ids_list: List[torch.LongTensor] = []
    q_sum = torch.zeros_like(x)
    residual = x

    # Heuristic: pick column chunk to bound (N_chunk × col_chunk × 4B) within ~512MB.
    # Since we tile over columns only here, use a high default 65536; safe for large K.
    col_chunk = 65536

    for C in codebooks:
        Cc = C.contiguous()
        idx = _nearest_indices_chunked(residual, Cc, col_chunk=col_chunk)  # (N,)
        ids_list.append(idx)

        piece = Cc[idx]  # (N, D)
        if piece.dtype != x.dtype or piece.device != x.device:
            piece = piece.to(dtype=x.dtype, device=x.device)

        q_sum = q_sum + piece
        residual = residual - piece

    return (ids_list, q_sum, residual) if return_residual else (ids_list, q_sum)


# ----------------------------
# Convenience: dict wrappers for per-parameter (e.g., "dc"/"sh")
# Note: For fitting, use ResidualKMeansVQ.fit() instead of rvq_fit
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