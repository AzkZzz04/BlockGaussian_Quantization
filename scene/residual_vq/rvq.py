# rvq.py
# Residual K-Means Vector Quantization (multi-layer, residual stacking).
# 依赖 quantize_kmeans.Quantize_kMeans 做单层聚类；本文件管理多层残差逻辑。
# 不与 Gaussian/renderer 等上层耦合。

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union

import torch

from .vq_module import Quantize_kMeans


__all__ = ["ResidualKMeansVQ", "rvq_apply"]


# 使用 vq_module 中的距离计算，避免重复实现


# 使用 rvq_apply.py 中的实现，避免重复
from .rvq_apply import rvq_apply


class ResidualKMeansVQ:
    """
    多层残差 K-Means 量化器（训练态可 stateful，运行态可纯函数）。
    设计要点：
      - fit(): 逐层在残差空间上拟合单层 KMeans（KMeans++ + 迭代），产出每层 codebook。
      - quantize(): 给定已拟合的 codebooks（或外部注入），执行无状态残差量化。
      - get_codebooks()/set_codebooks(): 取/设当前码本（List[Tensor(K_l, D)]）。

    用法：
      # 训练阶段（拟合）
      rvq = ResidualKMeansVQ(num_clusters=[2048, 1024], num_iters=10, max_samples=10000)
      ids, q = rvq.fit(features)         # 同时返回第一轮的 ids、重建和
      codebooks = rvq.get_codebooks()    # 拿到多层码本，供导出/保存

      # 推理/导出阶段（纯应用，不再拟合）
      ids2, q2 = rvq.quantize(features)  # 内部用已有 codebooks 做查表
      # 或使用纯函数：
      ids3, q3 = rvq_apply(features, codebooks)

    注意：
      - 本类不做磁盘 IO；保存/加载放在独立的 IO 模块里（例如 rvq_io.py）。
      - 不耦合 Gaussian/renderer；上层自己 reshape/写回。
    """

    def __init__(
        self,
        num_clusters: Union[int, Sequence[int]],
        num_iters: int = 10,
    ):
        if isinstance(num_clusters, int):
            num_clusters = [num_clusters]
        assert len(num_clusters) >= 1, "num_clusters must be non-empty sequence"

        self._layers_spec: List[int] = [int(k) for k in num_clusters]
        self._num_iters = int(num_iters)

        # 每层的 kmeans 实例（仅在 fit 时使用；运行时不必依赖）
        self._layers: List[Quantize_kMeans] = [
            Quantize_kMeans(num_clusters=k, num_iters=self._num_iters)
            for k in self._layers_spec
        ]
        for q in self._layers:
            q.verbose = False

        # 拟合后的码本（运行时仅依赖这一份）
        self._codebooks: List[torch.Tensor] = []

    # ---------- properties ----------
    @property
    def num_layers(self) -> int:
        return len(self._layers_spec)

    @property
    def clusters_per_layer(self) -> List[int]:
        return list(self._layers_spec)

    # ---------- training-time ----------
    @torch.no_grad()
    def fit(
        self,
        feat_2d: torch.Tensor,
        chunk_size: int = 32768,
        loss_callback=None,
    ) -> Tuple[List[torch.LongTensor], torch.Tensor]:
        """
        Fit K-Means layers on residual space; produces codebooks and updates self._codebooks.
        Args:
            feat_2d: (N, D) input features
            chunk_size: chunk size for distance computation
            loss_callback: Optional callback(layer_idx, reconstruction) -> loss_value for layer-aware training
        Returns:
            (ids_list, q_sum): layer indices and quantized reconstruction
        """
        assert feat_2d.ndim == 2, "feat_2d must be (N, D)"

        residual = feat_2d.detach()
        q_sum = torch.zeros_like(residual)
        ids_list: List[torch.LongTensor] = []
        codebooks: List[torch.Tensor] = []

        for layer_idx, q in enumerate(self._layers):
            # 在当前残差上聚类
            q.cluster_assign(residual, chunk_size=chunk_size)
            if q.centers.numel() == 0:
                # 该层失败则提前结束（通常发生在 D=0 或方差为 0）
                break

            # 拿到该层的索引与片段
            idx = q.nn_index.clone()
            piece = q.centers[idx]
            q_sum = q_sum + piece
            residual = residual - piece

            # 如果提供了损失回调，记录当前层的重建质量
            if loss_callback is not None:
                try:
                    layer_loss = loss_callback(layer_idx, q_sum.clone())
                    # 可以在这里根据损失调整后续层的行为，但暂时只记录
                except Exception as e:
                    # 损失计算失败时静默忽略，不影响核心聚类逻辑
                    pass

            # 记录本层结果
            ids_list.append(idx)
            codebooks.append(q.centers.detach().clone())

        # 更新当前已拟合的多层码本
        self._codebooks = codebooks
        return ids_list, q_sum

    # ---------- runtime ----------
    @torch.no_grad()
    def quantize(
        self,
        feat_2d: torch.Tensor,
        codebooks: Optional[Sequence[torch.Tensor]] = None,
        chunk: int = 32768,
    ) -> Tuple[List[torch.LongTensor], torch.Tensor]:
        """
        Runtime quantization using fitted or external codebooks.
        Uses self._codebooks if codebooks parameter is None.
        """
        cbs = list(codebooks) if codebooks is not None else self._codebooks
        assert len(cbs) > 0, "No codebooks available. Fit first or pass codebooks explicitly."
        return rvq_apply(feat_2d, cbs, chunk=chunk)

    # ---------- codebook access ----------
    def get_codebooks(self) -> List[torch.Tensor]:
        """Return current fitted codebooks as List[Tensor(K_l, D)]."""
        return list(self._codebooks)

    def set_codebooks(self, codebooks: Sequence[torch.Tensor]):
        """Explicitly set multi-layer codebooks (for external codebook injection)."""
        assert len(codebooks) >= 1, "codebooks must be non-empty"
        # Basic dimension validation
        D0 = int(codebooks[0].shape[1])
        for i, C in enumerate(codebooks):
            assert C.ndim == 2, f"codebook L{i} must be (K,D)"
            assert int(C.shape[1]) == D0, "all codebooks must share the same D"
        self._codebooks = [C for C in codebooks]

    # ---------- convenience ----------
    @torch.no_grad()
    def warm_start_from_codebooks(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        （可选）把当前 codebooks 同步回每层的 Quantize_kMeans.centers，
        便于在线继续微调（例如 online RVQ 迭代）。
        """
        if len(self._codebooks) == 0:
            return
        for q, C in zip(self._layers, self._codebooks):
            C_ = C
            if device is not None or dtype is not None:
                C_ = C.to(device=device if device is not None else C.device,
                          dtype=dtype if dtype is not None else C.dtype)
            q.centers = C_.clone()
            q.vec_dim = int(C_.shape[1])
            # 清空历史 assignment；下一次 cluster_assign 会重新分配
            q.nn_index = torch.empty(0, device=C_.device, dtype=torch.long)
            q.cluster_len = torch.empty(0, device=C_.device, dtype=torch.float32)
            q.cls_ids = torch.empty(0, device=C_.device, dtype=torch.long)