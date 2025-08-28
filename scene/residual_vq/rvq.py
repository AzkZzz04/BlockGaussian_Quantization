# rvq.py
# Residual K-Means Vector Quantization (multi-layer, residual stacking).
# Depends on vq_module.Quantize_kMeans for single-layer clustering.
# Optional IndexPrior for candidate prediction during offline quantization.

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .vq_module import Quantize_kMeans
from .rvq_apply import rvq_apply  # pure functional apply (no prior)

try:
    from ..index_prior import IndexPrior  # optional, injected or loaded by caller
    INDEX_PRIOR_AVAILABLE = True
except Exception:
    IndexPrior = None  # type: ignore
    INDEX_PRIOR_AVAILABLE = False


__all__ = ["ResidualKMeansVQ", "rvq_apply"]


# ----------------------------
# Internal numerical utilities
# ----------------------------
@torch.no_grad()
def _tile_argmin_over_columns(
    A: torch.Tensor,    # (m, D)
    B: torch.Tensor,    # (n, D)
    col_chunk: int = 65536,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each row in A, find the argmin over all rows in B (without creating (m,n) matrix).
    Returns: (best_ids (m,), best_dist2 (m,))
    Uses stable ||a||^2 + ||b||^2 - 2 a·b form with FP32 computation.
    """
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[1]
    m, D = A.shape
    device = A.device

    A32 = A.to(torch.float32)
    B32 = B.to(torch.float32)

    a2 = (A32 * A32).sum(dim=1, keepdim=True)                 # (m,1)
    best_d2 = torch.full((m, 1), float("inf"), device=device, dtype=torch.float32)
    best_j  = torch.full((m, 1), -1, device=device, dtype=torch.long)

    for s in range(0, B32.shape[0], col_chunk):
        e = min(s + col_chunk, B32.shape[0])
        Be = B32[s:e]                                         # (b,D)
        b2 = (Be * Be).sum(dim=1, keepdim=True).transpose(0, 1)  # (1,b)
        dots = A32 @ Be.t()                                   # (m,b)
        d2 = a2 + b2 - 2.0 * dots                             # (m,b)
        # merge local best
        loc_best, loc_idx = d2.min(dim=1, keepdim=True)       # (m,1)
        update = loc_best < best_d2
        best_d2[update] = loc_best[update]
        best_j[update]  = (loc_idx[update] + s)

        del Be, b2, dots, d2, loc_best, loc_idx, update

    return best_j.squeeze(1), best_d2.squeeze(1)


@torch.no_grad()
def _tiled_knn_indices(
    X: torch.Tensor,      # (N,3)
    k: int,
    row_chunk: int = 16384,
    col_chunk: int = 16384,
) -> torch.LongTensor:
    """
    Bi-directional chunked KNN (only keep top-k per row, avoid (m,N) huge matrix).
    Returns (N,k), excludes self indices.
    """
    assert X.ndim == 2 and X.shape[1] == 3
    N = X.shape[0]
    K = min(k, max(N - 1, 1))
    device = X.device

    # maintain "current minimum K distances and indices"
    best_d = torch.full((N, K), float("inf"), device=device, dtype=torch.float32)
    best_i = torch.full((N, K), -1, device=device, dtype=torch.long)

    X32 = X.to(torch.float32)
    x2_full = (X32 * X32).sum(dim=1, keepdim=True)  # (N,1)

    for rs in range(0, N, row_chunk):
        re = min(rs + row_chunk, N)
        Ar = X32[rs:re]                              # (mr,3)
        a2 = x2_full[rs:re]                          # (mr,1)

        # current row block's best candidates
        row_best_d = torch.full((re - rs, K), float("inf"), device=device, dtype=torch.float32)
        row_best_i = torch.full((re - rs, K), -1, device=device, dtype=torch.long)

        for cs in range(0, N, col_chunk):
            ce = min(cs + col_chunk, N)
            Bc = X32[cs:ce]                          # (mc,3)
            b2 = x2_full[cs:ce].t()                  # (1,mc)
            d2 = a2 + b2 - 2.0 * (Ar @ Bc.t())      # (mr,mc)

            # exclude self
            if rs < ce and cs < re:
                ovr_start = max(rs, cs)
                ovr_end = min(re, ce)
                if ovr_start < ovr_end:
                    diag_r = ovr_start - rs
                    diag_c = ovr_start - cs
                    for i in range(ovr_end - ovr_start):
                        d2[diag_r + i, diag_c + i] = float("inf")

            # update row block's best-K
            d_cat = torch.cat([row_best_d, d2], dim=1)    # (mr, K+mc)
            i_cat = torch.cat([row_best_i, torch.arange(cs, ce, device=device).expand(re - rs, -1)], dim=1)
            topk_d, topk_idx = torch.topk(d_cat, K, dim=1, largest=False)
            row_best_d = topk_d
            row_best_i = i_cat.gather(1, topk_idx)

        # merge to global best
        d_merge = torch.cat([best_d[rs:re], row_best_d], dim=1)  # (mr, K+K)
        i_merge = torch.cat([best_i[rs:re], row_best_i], dim=1)
        global_topk_d, global_topk_idx = torch.topk(d_merge, K, dim=1, largest=False)
        best_d[rs:re] = global_topk_d
        best_i[rs:re] = i_merge.gather(1, global_topk_idx)

    return best_i


# ----------------------------
# RVQ core implementation
# ----------------------------
class ResidualKMeansVQ:
    """
    Multi-layer residual K-Means VQ.
    - fit(): learn per-layer codebooks on residuals.
    - quantize(): apply learned (or external) codebooks; no fitting.
    - get_codebooks()/set_codebooks(): access learned codebooks.

    IndexPrior (optional):
    - Uses neighbor code indices from the previous RVQ layer as context.
    - First layer runs plain K-Means (no prior).
    - Prior is expected to be pre-trained and injected via set_index_priors().
    """

    def __init__(
        self,
        num_clusters: Union[int, Sequence[int]],
        num_iters: int = 10,
        use_index_prior: bool = False,
    ):
        if isinstance(num_clusters, int):
            num_clusters = [num_clusters]
        assert len(num_clusters) >= 1, "num_clusters must be non-empty"

        self._layers_spec: List[int] = [int(k) for k in num_clusters]
        self._num_iters = int(num_iters)

        self._layers: List[Quantize_kMeans] = [
            Quantize_kMeans(num_clusters=k, num_iters=self._num_iters)
            for k in self._layers_spec
        ]
        for q in self._layers:
            if hasattr(q, "verbose"):
                q.verbose = False

        self._codebooks: List[torch.Tensor] = []

        # Prior models are optional and must be injected (pre-trained)
        self._use_index_prior = bool(use_index_prior) and INDEX_PRIOR_AVAILABLE
        self._index_priors: List[Optional[IndexPrior]] = []

    # ---------- IndexPrior injection ----------

    def set_index_priors(self, priors: Optional[List[IndexPrior]]) -> None:
        """
        Inject pre-trained IndexPrior models, one per RVQ layer.
        Pass None or [] to disable prior usage.
        """
        if (not self._use_index_prior) or (not priors):
            self._index_priors = []
            return
        assert len(priors) == self.num_layers, "priors length must match num_layers"
        self._index_priors = priors

    def _prior_ready(self, layer_idx: int) -> bool:
        return (
            self._use_index_prior
            and len(self._index_priors) == self.num_layers
            and (self._index_priors[layer_idx] is not None)
        )

    # ---------- utilities: KNN / context / candidate selection ----------

    @torch.no_grad()
    def _knn_indices(
        self,
        positions: torch.Tensor,
        k: int = 8,
        row_chunk: int = 16384,
        col_chunk: int = 16384,
    ) -> torch.LongTensor:
        """
        Bi-directional chunked KNN (avoid (m,N) huge matrix). Returns (N, k), excludes self.
        """
        return _tiled_knn_indices(positions, k=k, row_chunk=row_chunk, col_chunk=col_chunk)

    @staticmethod
    def _build_prior_context(
        prev_layer_ids: torch.LongTensor,       # (N,)
        knn_idx: torch.LongTensor,              # (N, K)
        positions: Optional[torch.Tensor] = None,  # (N, 3)
    ) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
        """
        Build prior context (neighbors only, no extra PAD token):
          tokens: (N, K) = neighbors' previous layer codes
          pos   : (N, K, 3) neighbor coordinates (optional)
        """
        N, K = knn_idx.shape
        device = knn_idx.device
        tokens = prev_layer_ids[knn_idx]                             # (N, K)
        if positions is None:
            return tokens, None
        pos = positions[knn_idx]                                     # (N, K, 3)
        return tokens, pos

    @staticmethod
    @torch.no_grad()
    def _prior_candidates_or_fallback(
        prior: "IndexPrior",
        tokens: torch.Tensor,                 # (N, K) long
        pos: Optional[torch.Tensor],          # (N, K, 3) or None
        codebook: torch.Tensor,               # (V, D)
        topk: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get candidates (row ids) and corresponding logits/probabilities, prioritizing prior's predict_relative_softmax;
        fallback to full logits_topk if method unavailable (may be slower).
        Returns: cand_ids (N, topk), cand_logits (N, topk)
        """
        # Priority: relative softmax (normalize only over candidates)
        if hasattr(prior, "predict_relative_softmax"):
            try:
                cand_ids_row, cand_probs = prior.predict_relative_softmax(
                    neighbor_indices=tokens,
                    positions=pos,
                    codebook=codebook,
                    k=topk,
                    metric="cosine",
                    temperature=1.0,
                    key_padding_mask=None,
                    query_proj=None,
                    code_id_map=None,
                )  # (N,k),(N,k)
                # use log probability as logits (monotonically equivalent)
                return cand_ids_row, cand_probs.clamp_min_(1e-9).log()
            except Exception:
                pass

        # fallback: full logits topk (take last position to be compatible with causal training)
        try:
            h = prior.encode_context(tokens, positions=pos, key_padding_mask=None)  # (N,K,D)
            logits = prior.logits_full(h)  # (N,K,V)
            logits_last = logits[:, -1, :]  # (N,V)
            v, idx = torch.topk(logits_last, k=min(topk, logits_last.shape[1]), dim=1, largest=True)
            return idx, v
        except Exception:
            # give up on prior if still failing
            N = tokens.shape[0]
            return torch.zeros(N, 0, dtype=torch.long, device=tokens.device), torch.zeros(N, 0, device=tokens.device)

    @staticmethod
    @torch.no_grad()
    def _select_from_candidates(
        residual: torch.Tensor,          # (N, D)
        codebook: torch.Tensor,          # (V, D)
        cand_ids: torch.Tensor,          # (N, K) row ids
        cand_logits: Optional[torch.Tensor] = None,  # (N, K) logits
        beta: float = 0.2,
        row_chunk: int = 32768,
    ) -> torch.LongTensor:
        """
        Argmin over candidate scores:
          score = ||r - c||^2 - beta * logits
        Returns selected row indices: (N,)
        """
        if cand_ids.numel() == 0:
            # no candidates, fallback: full NN
            best_ids, _ = _tile_argmin_over_columns(residual, codebook, col_chunk=65536)
            return best_ids

        N = residual.shape[0]
        out = torch.empty(N, dtype=torch.long, device=residual.device)
        use_prior = cand_logits is not None and cand_logits.numel() > 0 and beta > 0.0

        for s in range(0, N, row_chunk):
            e = min(s + row_chunk, N)
            r = residual[s:e]                   # (m, D)
            cids = cand_ids[s:e]                # (m, K)
            cvecs = codebook[cids]              # (m, K, D)
            dist2 = (r[:, None, :] - cvecs).pow(2).sum(-1)  # (m, K)
            if use_prior:
                score = dist2 - beta * cand_logits[s:e]     # (m, K)
            else:
                score = dist2
            loc = torch.argmin(score, dim=1)                # (m,)
            out[s:e] = cids.gather(1, loc[:, None]).squeeze(1)

        return out

    # ---------- properties ----------

    @property
    def num_layers(self) -> int:
        return len(self._layers_spec)

    @property
    def clusters_per_layer(self) -> List[int]:
        return list(self._layers_spec)

    # ---------- fitting (learn codebooks) ----------

    @torch.no_grad()
    def fit(
        self,
        feat_2d: torch.Tensor,                    # (N, D)
        chunk_size: int = 32768,
        loss_callback=None,
        positions: Optional[torch.Tensor] = None, # (N, 3)
        knn_idx: Optional[torch.LongTensor] = None,  # (N, K)
        prior_topk: int = 10,
        prior_beta: float = 0.2,
    ) -> Tuple[List[torch.LongTensor], torch.Tensor]:
        """
        Learn per-layer codebooks on residuals. If IndexPrior is ready and positions
        are provided, refine assignments for layers l>=1 using candidate-only search.
        """
        assert feat_2d.ndim == 2
        x = feat_2d.detach().contiguous()
        N, D = x.shape

        residual = x
        q_sum = torch.zeros_like(residual)
        ids_list: List[torch.LongTensor] = []
        codebooks: List[torch.Tensor] = []

        knn = None
        if positions is not None:
            pos = positions.contiguous()
            knn = knn_idx if knn_idx is not None else self._knn_indices(pos, k=8)

        for layer_idx, q in enumerate(self._layers):
            # Single-layer KMeans (already standardized and chunked within Quantize_kMeans)
            q.cluster_assign(residual, chunk_size=chunk_size)
            if q.centers.numel() == 0:
                break

            idx = q.nn_index.clone()
            piece = q.centers[idx]

            # Prior-based refinement for layers >= 1
            if knn is not None and layer_idx > 0 and self._prior_ready(layer_idx):
                prior = self._index_priors[layer_idx].to(residual.device).eval()  # type: ignore[union-attr]
                prev_ids = ids_list[layer_idx - 1]  # (N,)

                try:
                    tokens, pos_ctx = self._build_prior_context(prev_ids, knn, positions)
                    cand_ids, cand_logits = self._prior_candidates_or_fallback(
                        prior, tokens, pos_ctx, q.centers, topk=prior_topk
                    )  # (N,K),(N,K)
                    idx_refined = self._select_from_candidates(
                        residual, q.centers, cand_ids, cand_logits=cand_logits, beta=prior_beta, row_chunk=chunk_size
                    )
                    # Only adopt if strictly better
                    err_new = (residual - q.centers[idx_refined]).pow(2).sum()
                    err_old = (residual - piece).pow(2).sum()
                    if err_new < err_old:
                        idx = idx_refined
                        piece = q.centers[idx]
                except Exception:
                    pass  # Conservative fallback: keep baseline

            q_sum = q_sum + piece
            residual = residual - piece

            if loss_callback is not None:
                try:
                    loss_callback(layer_idx, q_sum.clone())
                except Exception:
                    pass

            ids_list.append(idx)
            codebooks.append(q.centers.detach().clone())

        self._codebooks = codebooks
        return ids_list, q_sum

    # ---------- runtime (apply codebooks) ----------

    @torch.no_grad()
    def quantize(
        self,
        feat_2d: torch.Tensor,                            # (N, D)
        codebooks: Optional[Sequence[torch.Tensor]] = None,
        chunk: int = 32768,
        positions: Optional[torch.Tensor] = None,         # (N, 3)
        knn_idx: Optional[torch.LongTensor] = None,       # (N, K)
        prior_topk: int = 10,
        prior_beta: float = 0.2,
    ) -> Tuple[List[torch.LongTensor], torch.Tensor]:
        """
        Apply codebooks to features. Uses self._codebooks when codebooks is None.
        If IndexPrior is ready and positions are provided, refine assignments for layers l>=1.
        """
        cbs = list(codebooks) if codebooks is not None else self._codebooks
        assert len(cbs) > 0, "No codebooks available. Fit first or pass codebooks."

        # Fast path: no prior or no positions -> pure functional apply
        if not (positions is not None and self._use_index_prior and len(self._index_priors) == self.num_layers):
            return rvq_apply(feat_2d, cbs, chunk=chunk)

        residual = feat_2d.detach().contiguous()
        q_sum = torch.zeros_like(residual)
        ids_list: List[torch.LongTensor] = []

        pos = positions.contiguous()
        knn = knn_idx if knn_idx is not None else self._knn_indices(pos, k=8)

        for layer_idx, cb in enumerate(cbs):
            if layer_idx == 0 or not self._prior_ready(layer_idx):
                # Full nearest neighbor (chunk codebook too, avoid (m,V) huge matrix)
                best_ids, _ = _tile_argmin_over_columns(residual, cb, col_chunk=65536)
                idx = best_ids
            else:
                prev_ids = ids_list[layer_idx - 1]  # (N,)
                prior = self._index_priors[layer_idx].to(residual.device).eval()  # type: ignore[union-attr]
                try:
                    tokens, pos_ctx = self._build_prior_context(prev_ids, knn, positions)
                    cand_ids, cand_logits = self._prior_candidates_or_fallback(
                        prior, tokens, pos_ctx, cb, topk=prior_topk
                    )  # (N,K),(N,K)
                    idx = self._select_from_candidates(
                        residual, cb, cand_ids, cand_logits=cand_logits, beta=prior_beta, row_chunk=chunk
                    )
                except Exception:
                    # Fallback: full NN
                    best_ids, _ = _tile_argmin_over_columns(residual, cb, col_chunk=65536)
                    idx = best_ids

            piece = cb[idx]
            q_sum = q_sum + piece
            residual = residual - piece
            ids_list.append(idx)

        return ids_list, q_sum

    # ---------- codebook access ----------

    def get_codebooks(self) -> List[torch.Tensor]:
        return list(self._codebooks)

    def set_codebooks(self, codebooks: Sequence[torch.Tensor]) -> None:
        assert len(codebooks) >= 1, "codebooks must be non-empty"
        D0 = int(codebooks[0].shape[1])
        out: List[torch.Tensor] = []
        for i, C in enumerate(codebooks):
            assert C.ndim == 2, f"codebook L{i} must be (K,D)"
            assert int(C.shape[1]) == D0, "all codebooks must share the same D"
            out.append(C.contiguous())
        self._codebooks = out

    # ---------- convenience ----------

    @torch.no_grad()
    def warm_start_from_codebooks(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Copy learned codebooks back into per-layer Quantize_kMeans instances
        to continue online refinement if needed.
        """
        if len(self._codebooks) == 0:
            return
        for q, C in zip(self._layers, self._codebooks):
            C_ = C.to(device=device or C.device, dtype=dtype or C.dtype).contiguous()
            q.centers = C_.clone()
            q.vec_dim = int(C_.shape[1])
            q.nn_index = torch.empty(0, device=C_.device, dtype=torch.long)
            q.cluster_len = torch.empty(0, device=C_.device, dtype=torch.float32)
            q.cls_ids = torch.empty(0, device=C_.device, dtype=torch.long)