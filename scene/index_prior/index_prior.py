# -*- coding: utf-8 -*-
# Minimal, stable IndexPrior (A-path: relative softmax over KNN candidates)
# Public API is preserved.

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Positional encoding (optional)
# -----------------------------
def fourier_encode_xyz(xyz: torch.Tensor, num_bands: int = 6, *, scale: float = 1.0) -> torch.Tensor:
    """
    xyz: (B, L, 3) -> (B, L, 6 * num_bands)
    6 * num_bands = 3 dims * 2 (sin, cos) * num_bands
    """
    assert xyz.ndim == 3 and xyz.shape[-1] == 3, "xyz must be (B,L,3)"
    B, L, _ = xyz.shape
    freqs = (2.0 ** torch.arange(num_bands, device=xyz.device, dtype=xyz.dtype)) * float(scale)  # match dtype
    angles = xyz.unsqueeze(-1) * freqs  # (B,L,3,F)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    # safer than view() for non-contiguous inputs
    out = torch.stack([sin, cos], dim=3).reshape(B, L, 3 * 2 * num_bands)  # (B,L,6*bands)
    return out


# -----------------------------
# Stable / chunked KNN on codebook (both query and codebook chunking)
# -----------------------------
@torch.no_grad()
def knn_on_codebook(
    query: torch.Tensor,          # (B,L,Dq)
    codebook: torch.Tensor,       # (K,Dc)
    k: int = 16,
    metric: str = "cosine",       # "cosine" or "l2"
    chunk_q: int = 65536,
    chunk_k: int = 65536,         # NEW: chunk along codebook to avoid OOM on large K
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Return top-k codebook row ids for each query: (B,L,k)
    Requires query last-dim == codebook last-dim (or project outside).
    Chunk both Q and K to avoid large temporary matrices.
    """
    assert codebook.ndim == 2, "codebook must be (K,D)"
    assert query.ndim == 3, "query must be (B,L,D)"
    B, L, Dq = query.shape
    K, Dc = codebook.shape
    if Dq != Dc:
        raise ValueError(f"[knn_on_codebook] query dim != codebook dim ({Dq}!={Dc}); "
                         f"project query to match codebook before calling.")
    if k <= 0:
        return torch.empty(B, L, 0, dtype=torch.long, device=query.device)

    Q = query.reshape(-1, Dq)  # (BL,D)

    # we always keep 'scores' as "larger is better"
    if metric == "cosine":
        Qn = F.normalize(Q, dim=-1, eps=eps)
    elif metric == "l2":
        Qn = Q.to(torch.float32)
    else:
        raise ValueError(f"[knn_on_codebook] unknown metric: {metric}")

    BL = Q.shape[0]
    topk_vals = torch.full((BL, k), -float("inf"), device=query.device)
    topk_idx  = torch.full((BL, k), -1, dtype=torch.long, device=query.device)

    for ks in range(0, K, chunk_k):
        ke = min(ks + chunk_k, K)
        C = codebook[ks:ke]
        if metric == "cosine":
            Cn = F.normalize(C, dim=-1, eps=eps)  # (kc,D)
        else:
            Cn = C.to(torch.float32)              # (kc,D)

        for qs in range(0, BL, chunk_q):
            qe = min(qs + chunk_q, BL)
            if metric == "cosine":
                scores = Qn[qs:qe] @ Cn.t()                       # (qc,kc)
            else:
                d = torch.cdist(Qn[qs:qe], Cn, p=2)               # (qc,kc)
                scores = -d                                       # larger=better

            # merge local (kc) with running top-k (k)
            idx_local = torch.arange(ks, ke, device=query.device).unsqueeze(0).expand_as(scores)
            merged_vals = torch.cat([topk_vals[qs:qe], scores], dim=1)     # (qc, k+kc)
            merged_idx  = torch.cat([topk_idx[qs:qe],  idx_local], dim=1)  # (qc, k+kc)
            mv, mi = torch.topk(merged_vals, k=min(k, K), dim=1, largest=True)
            topk_vals[qs:qe] = mv
            topk_idx[qs:qe]  = torch.gather(merged_idx, 1, mi)

    return topk_idx.reshape(B, L, -1)


# -----------------------------
# IndexPrior with relative-softmax path
# -----------------------------
class IndexPrior(nn.Module):
    """
    Context LM for RVQ index prediction.
    Training: absolute index CE over num_codes.
    Inference (A-path): KNN on codebook (row ids) -> optional row->class mapping -> local softmax over candidates.
    """
    def __init__(self,
                 num_codes: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 use_xyz: bool = True,
                 xyz_bands: int = 6,
                 causal: bool = False,
                 tie_weights: bool = False,
                 xyz_scale: float = 1.0):
        super().__init__()
        self.num_codes = int(num_codes)
        self.use_xyz = bool(use_xyz)
        self.causal = bool(causal)
        self.xyz_bands = int(xyz_bands)
        self.xyz_scale = float(xyz_scale)

        self.idx_emb = nn.Embedding(self.num_codes, d_model)

        if self.use_xyz:
            in_xyz = 3 * 2 * self.xyz_bands
            self.xyz_proj = nn.Linear(in_xyz, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out = nn.Linear(d_model, self.num_codes, bias=(not tie_weights))

        self._mask_cache: Dict[Tuple[torch.device, int], torch.Tensor] = {}
        self.reset_parameters()

        if tie_weights:
            with torch.no_grad():
                self.out.weight.copy_(self.idx_emb.weight)
            # Tie weights; bias disabled above
            self.out.weight = self.idx_emb.weight  # type: ignore[assignment]

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _build_src_mask(self, L: int, device) -> Optional[torch.Tensor]:
        if not self.causal:
            return None
        key = (device, L)
        mask = self._mask_cache.get(key, None)
        if mask is None:
            mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)  # True=masked
            self._mask_cache[key] = mask
        return mask

    def encode_context(
        self,
        neighbor_indices: torch.Tensor,               # (B,L)
        positions: Optional[torch.Tensor] = None,     # (B,L,3)
        key_padding_mask: Optional[torch.Tensor] = None  # (B,L) 1=valid,0=pad
    ) -> torch.Tensor:
        assert neighbor_indices.dtype == torch.long, "neighbor_indices must be long"
        x = self.idx_emb(neighbor_indices)  # (B,L,D)
        if self.use_xyz and positions is not None:
            xyz_feat = fourier_encode_xyz(positions, num_bands=self.xyz_bands, scale=self.xyz_scale)  # (B,L,F)
            x = x + self.xyz_proj(xyz_feat)

        src_mask = self._build_src_mask(x.size(1), x.device)
        pad_mask = (key_padding_mask == 0) if key_padding_mask is not None else None
        h = self.encoder(x, mask=src_mask, src_key_padding_mask=pad_mask)  # (B,L,D)
        return h

    # ------- absolute classification for training -------
    def logits_full(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(h)

    def loss_xent(
        self,
        neighbor_indices: torch.Tensor,
        target_indices: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert target_indices.dtype == torch.long, "target_indices must be long"
        h = self.encode_context(neighbor_indices, positions, key_padding_mask)
        logits = self.out(h)  # (B,L,V)
        B, L, V = logits.shape
        logits = logits.reshape(B * L, V)
        targets = target_indices.reshape(B * L)
        if key_padding_mask is not None:
            mask = key_padding_mask.reshape(B * L) > 0
            logits = logits[mask]
            targets = targets[mask]
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def score_candidates(self, h: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        """
        Score candidate class ids (no softmax).
        h: (B,L,D); cand_ids: (B,L,k) in class-id space aligned to out.weight rows.
        """
        W: torch.Tensor = self.out.weight   # (V,D)
        b: Optional[torch.Tensor] = self.out.bias  # (V,) or None

        # gather rows from W and b by cand_ids using index_select (safer than F.embedding for future dtype changes)
        B, L, K = cand_ids.shape
        flat_ids = cand_ids.reshape(-1)  # (B*L*K,)
        Wk = W.index_select(0, flat_ids).reshape(B, L, K, W.size(1))  # (B,L,k,D)
        scores = (h.unsqueeze(2) * Wk).sum(dim=-1)  # (B,L,k)
        if b is not None:
            bk = b.index_select(0, flat_ids).reshape(B, L, K)  # (B,L,k)
            scores = scores + bk
        return scores

    # ------- A-path: neighborhood softmax at inference -------
    @torch.no_grad()
    def predict_relative_softmax(
        self,
        neighbor_indices: torch.Tensor,            # (B,L)
        positions: Optional[torch.Tensor],         # (B,L,3) or None
        codebook: torch.Tensor,                    # (K,Dc)
        *,
        k: int = 16,
        metric: str = "cosine",                    # "cosine" or "l2"
        temperature: float = 1.0,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_proj: Optional[nn.Module] = None,    # h -> Dc
        code_id_map: Optional[torch.Tensor] = None # (K,) long; row -> class id
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (cand_ids_row, cand_probs) both (B,L,k)
        """
        assert codebook.ndim == 2, "codebook must be (K,D)"
        if code_id_map is not None:
            assert code_id_map.ndim == 1 and code_id_map.shape[0] == codebook.shape[0], \
                "[IndexPrior] code_id_map must be (K,) and match codebook rows"

        h = self.encode_context(neighbor_indices, positions, key_padding_mask)  # (B,L,D)

        # Dim align for KNN
        q_for_knn = query_proj(h) if query_proj is not None else h
        if q_for_knn.shape[-1] != codebook.shape[-1]:
            raise ValueError(
                f"[predict_relative_softmax] Dim mismatch: query={q_for_knn.shape[-1]} vs codebook={codebook.shape[-1]}. "
                "Provide query_proj to align dims."
            )

        cand_ids_row = knn_on_codebook(q_for_knn, codebook, k=k, metric=metric)  # (B,L,k)

        # Row id -> class id if needed
        K = codebook.shape[0]
        if code_id_map is None:
            if K != self.num_codes:
                raise ValueError(
                    f"[IndexPrior] codebook rows ({K}) != num_codes ({self.num_codes}); "
                    f"pass code_id_map to map row->class id."
                )
            cand_ids_class = cand_ids_row
        else:
            # FIX: simple tensor indexing (F.embedding expects float weights, not integer mapping)
            cand_ids_class = code_id_map[cand_ids_row]  # (B,L,k) long

        scores = self.score_candidates(h, cand_ids_class)  # (B,L,k)

        # Numerically stable local softmax with temperature clamp (avoid tensor alloc)
        t = float(max(1e-3, min(100.0, float(temperature))))
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = F.softmax(scores / t, dim=-1)
        return cand_ids_row, probs


# -----------------------------
# Final index selection with robust fallbacks
# -----------------------------
@torch.no_grad()
def select_indices_with_prior(
    prior: IndexPrior,
    neighbor_indices: torch.Tensor,           # (B,L)
    positions: Optional[torch.Tensor],        # (B,L,3) or None
    codebook: torch.Tensor,                   # (K,D)
    *,
    k: int = 16,
    metric: str = "cosine",
    temperature: float = 1.0,
    key_padding_mask: Optional[torch.Tensor] = None,
    query_proj: Optional[nn.Module] = None,
    refine_by_l2: bool = True,
    target_vecs: Optional[torch.Tensor] = None,   # (B,L,D)
    code_id_map: Optional[torch.Tensor] = None    # (K,)
) -> torch.Tensor:
    """
    Return final codebook row ids: (B,L)
    Pipeline: KNN candidates (row ids) -> prior local softmax on mapped class ids -> argmax -> optional L2 rerank.
    Fallbacks:
      1) If target_vecs exists: nearest neighbor over full codebook.
      2) Else: KNN with prior's hidden states (or its projection).
      3) Else: return zeros.
    """
    assert neighbor_indices.dtype == torch.long, "neighbor_indices must be long"
    assert codebook.ndim == 2, "codebook must be (K,D)"
    if target_vecs is not None:
        assert target_vecs.ndim == 3 and target_vecs.shape[-1] == codebook.shape[-1], \
            "[select_indices_with_prior] target_vecs last dim must match codebook last dim"

    B, L = neighbor_indices.shape

    try:
        cand_ids_row, cand_probs = prior.predict_relative_softmax(
            neighbor_indices, positions, codebook,
            k=k, metric=metric, temperature=temperature,
            key_padding_mask=key_padding_mask, query_proj=query_proj,
            code_id_map=code_id_map
        )  # (B,L,k),(B,L,k)

        # Initial pick by prior probability
        top1_idx = cand_probs.argmax(dim=-1)  # (B,L)
        sel_ids = torch.gather(cand_ids_row, 2, top1_idx.unsqueeze(-1)).squeeze(-1)  # (B,L)

        # Optional L2 rerank using true vectors (chunked to reduce peak memory)
        if refine_by_l2 and target_vecs is not None:
            D = codebook.shape[-1]
            BL = B * L
            flat_rows = cand_ids_row.reshape(BL, -1)   # (BL,k)
            flat_tv = target_vecs.reshape(BL, D)       # (BL,D)
            best = torch.empty(BL, dtype=torch.long, device=codebook.device)
            step = 32768
            for s in range(0, BL, step):
                e = min(s + step, BL)
                rows = flat_rows[s:e]                  # (c,k)
                vecs = codebook[rows]                  # (c,k,D)
                tv = flat_tv[s:e].unsqueeze(1)         # (c,1,D)
                d2 = ((vecs - tv) ** 2).sum(-1)        # (c,k)
                best[s:e] = d2.argmin(-1)
            sel_ids = torch.gather(cand_ids_row, 2, best.reshape(B, L, 1)).squeeze(-1)

        return sel_ids

    except RuntimeError:
        # 1) Strongest fallback: full-codebook NN with target vectors
        if target_vecs is not None:
            B, L, D = target_vecs.shape
            cb32 = codebook.to(torch.float32)
            tv_flat = target_vecs.reshape(-1, D).to(torch.float32)
            nn_ids = []
            chunk_q = 65536
            for s in range(0, tv_flat.shape[0], chunk_q):
                e = min(s + chunk_q, tv_flat.shape[0])
                d = torch.cdist(tv_flat[s:e], cb32, p=2)    # (c,K)
                nn_ids.append(d.argmin(dim=-1))
            nn = torch.cat(nn_ids, dim=0).reshape(B, L)
            return nn.to(torch.long)

        # 2) Next fallback: KNN with prior hidden (or projected) if dims match
        h = prior.encode_context(neighbor_indices, positions, key_padding_mask)  # (B,L,Dh)
        q = query_proj(h) if query_proj is not None else h
        if q.shape[-1] == codebook.shape[-1]:
            knn_ids = knn_on_codebook(q, codebook, k=1, metric=metric)
            return knn_ids.squeeze(-1)

        # 3) Conservative default
        return torch.zeros(B, L, dtype=torch.long, device=codebook.device)