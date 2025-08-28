# Adapter for legacy interface: forward_dc / forward_frest
# Internally uses ResidualKMeansVQ; when layers=1 it equals plain VQ

import math
import torch
from functools import lru_cache
from typing import List, Optional

from .rvq import ResidualKMeansVQ  # Your existing RVQ implementation


# ----------------------------
# K allocation (layer-wise decreasing + exact sum preservation)
# ----------------------------
def _k_per_layer(target_k: int, L: int) -> List[int]:
    """
    Progressive allocation (earlier layers get more weight), exactly satisfying sum(k_i)=target_k.
    Uses harmonic series weights + largest remainder method, ensuring each layer has at least 2.
    """
    if L <= 1:
        return [max(2, target_k)]

    # harmonic weights: 1, 1/2, 1/3, ...
    weights = [1.0 / (i + 1) for i in range(L)]
    total_w = sum(weights)

    raw = [target_k * w / total_w for w in weights]          # floating point quotas
    base = [max(2, int(math.floor(x))) for x in raw]         # floor and ensure minimum
    remain = target_k - sum(base)

    if remain > 0:
        # remaining quotas to allocate: supplement by "how far from expected" in descending order
        frac_gain = [(raw[i] - (base[i] - 2)) for i in range(L)]
        order = sorted(range(L), key=lambda i: frac_gain[i], reverse=True)
        for i in range(remain):
            base[order[i % L]] += 1
    elif remain < 0:
        # over-allocated: reduce by "reducible space" in descending order, keeping minimum of 2
        need = -remain
        frac_loss = [(base[i] - 2) - (raw[i] - (base[i] - 2)) for i in range(L)]
        order = sorted(range(L), key=lambda i: frac_loss[i], reverse=True)
        for i in range(L):
            can = min(base[order[i]] - 2, need)
            if can > 0:
                base[order[i]] -= can
                need -= can
            if need == 0:
                break

    # light smoothing: ensure non-increasing (front layer >= back layer), preserving total sum
    for i in range(1, L):
        if base[i] > base[i - 1]:
            delta = base[i] - base[i - 1]
            base[i] -= delta
            base[0] += delta

    return base


# ----------------------------
# SH band weights
# ----------------------------
def _compute_sh_band_weights(sh_degree: int, alpha: float = 0.15, include_dc: bool = True) -> torch.Tensor:
    """
    Compute SH band weights: w_l = 1 + α * l(l+1)
    Higher frequency components get higher weights to improve perceptual quality.
    """
    weights = []
    start_l = 0 if include_dc else 1
    for l in range(start_l, sh_degree + 1):
        for _m in range(-l, l + 1):
            weights.append(1.0 + alpha * l * (l + 1))
    return torch.tensor(weights, dtype=torch.float32)


def _compute_sh_band_weights_no_dc(sh_degree: int, alpha: float = 0.15) -> torch.Tensor:
    """SH band weights excluding DC component"""
    return _compute_sh_band_weights(sh_degree, alpha, include_dc=False)


@lru_cache(maxsize=32)
def _cached_sh_weights(C: int, alpha: float) -> torch.Tensor:
    """
    Cached SH weights (excluding DC):
    Input C is the number of SH coefficients per channel (excluding DC), verified for degree consistency.
    """
    deg = int(round(math.sqrt(C + 1) - 1))
    assert (deg + 1) * (deg + 1) - 1 == C, f"Invalid SH C={C} for degree deduction"
    return _compute_sh_band_weights_no_dc(deg, alpha)  # (C,)


# ----------------------------
# RVQ quantization adapter
# ----------------------------
class Quantize_RVQ:

    def __init__(self,
                 which: str,
                 target_k: int = 4096,
                 layers: int = 1,
                 num_iters: int = 10,
                 sh_band_weighting: bool = True,
                 band_weight_alpha: float = 0.15,
                 layer_aware_training: bool = True,
                 use_index_prior: bool = False,
                 index_prior_config: Optional[dict] = None,
                 use_multi_gpu: bool = False,
                 gpu_devices: tuple = (0, 1)):
        assert which in ("dc", "sh"), "which must be 'dc' or 'sh'"
        self.which = which
        self.layers = max(1, int(layers))
        self.target_k = int(target_k)
        self.num_iters = int(num_iters)
        self.sh_band_weighting = sh_band_weighting and (which == "sh")
        self.band_weight_alpha = float(band_weight_alpha)
        self.layer_aware_training = bool(layer_aware_training)
        self.use_index_prior = bool(use_index_prior)

        # Construct per-layer K; progressive allocation with exact sum preservation
        layer_ks = _k_per_layer(self.target_k, self.layers)

        # IndexPrior configuration (placeholder/passthrough)
        if index_prior_config is None:
            index_prior_config = {
                "d_model": 64,
                "nhead": 8,
                "num_layers": 2,
                "use_positional_encoding": True,
            }
        self.index_prior_config = index_prior_config

        # RVQ main body (should internally accept use_index_prior / positions etc.)
        self._rvq = ResidualKMeansVQ(
            num_clusters=layer_ks,
            num_iters=self.num_iters,
            use_index_prior=self.use_index_prior
        )

        # Compatibility attributes (old code might read these)
        self.num_clusters = self.target_k          # for display only
        self.num_kmeans_iters = self.num_iters     # for display only
        self.cls_ids = torch.empty(0, dtype=torch.long)  # last layer indices (shape compatible)
        self.centers = torch.empty(0)              # RVQ multi-layer codebooks, no longer single centers; placeholder to avoid breaking interface

        self._fitted = False
        self._last_ids_list = None  # multi-layer indices cache (for saving)

        # Layer-aware training state (statistics only, no backprop)
        self._layer_losses: List[List[float]] = []
        self._training_mode = False

    # ----------------------------
    # Internal flatten/restore
    # ----------------------------
    @torch.no_grad()
    def _flat_dc(self, gaussian):
        # (N,1,3) -> (N,3)
        x = gaussian.get_features_dc.detach()
        return x.reshape(-1, 3).contiguous()

    @torch.no_grad()
    def _flat_sh(self, gaussian):
        # (N,C,3) -> (N, C*3)
        fr = gaussian.get_features_rest.detach()
        N, C, _ = fr.shape
        feat = fr.reshape(N, C * 3).contiguous()

        if self.sh_band_weighting:
            sh_weights = _cached_sh_weights(C, self.band_weight_alpha).to(feat.device, non_blocking=True)  # (C,)
            weights_expanded = sh_weights.repeat_interleave(3).unsqueeze(0)  # (1, C*3)
            assert weights_expanded.shape[1] == feat.shape[1], "SH weight expansion mismatch"
            feat = feat * weights_expanded
        return feat

    @staticmethod
    def _shape_dc(x: torch.Tensor) -> torch.Tensor:
        # (N, 3) -> (N,1,3)
        return x.reshape(-1, 1, 3)

    @staticmethod
    def _shape_sh(x: torch.Tensor, C: int) -> torch.Tensor:
        # (N, C*3) -> (N,C,3)
        return x.reshape(-1, C, 3)

    def _rvq_is_fitted(self) -> bool:
        """Try not to depend on internal implementation: prefer calling is_fitted(), otherwise check if codebooks exist."""
        if hasattr(self._rvq, "is_fitted") and callable(self._rvq.is_fitted):
            try:
                return bool(self._rvq.is_fitted())
            except Exception:
                pass
        if hasattr(self._rvq, "get_codebooks"):
            try:
                cbs = self._rvq.get_codebooks()
                return (isinstance(cbs, (list, tuple)) and len(cbs) > 0)
            except Exception:
                return False
        return self._fitted

    # ----------------------------
    # Forward (DC)
    # ----------------------------
    @torch.no_grad()
    def forward_dc(self, gaussian, assign: bool = False):
        if self.which != "dc":
            return

        feat = self._flat_dc(gaussian)
        positions = gaussian.get_xyz.contiguous() if self.use_index_prior else None

        # Inter-layer loss statistics callback (no backprop)
        loss_callback = None
        if self._training_mode and assign and self.layer_aware_training:
            def dc_loss_callback(layer_idx, reconstruction):
                loss = torch.mean((reconstruction - feat) ** 2).item()
                self.record_layer_loss(layer_idx, loss)
                return loss
            loss_callback = dc_loss_callback

        if (not self._fitted) or assign:
            ids_list, q = self._rvq.fit(feat, loss_callback=loss_callback, positions=positions)
            self._fitted = True
        else:
            if not self._rvq_is_fitted():
                ids_list, q = self._rvq.fit(feat, loss_callback=loss_callback, positions=positions)
                self._fitted = True
            else:
                ids_list, q = self._rvq.quantize(feat, positions=positions)

        gaussian.set_quantized_dc(self._shape_dc(q), ids_list[-1])
        self.cls_ids = ids_list[-1]         # compatible with old save logic (only save last layer)
        self._last_ids_list = ids_list      # complete multi-layer indices cache

    # ----------------------------
    # Forward (SH / features_rest)
    # ----------------------------
    @torch.no_grad()
    def forward_frest(self, gaussian, assign: bool = False):
        if self.which != "sh":
            return

        features_rest = gaussian.get_features_rest
        C = features_rest.shape[1]
        feat = self._flat_sh(gaussian)
        positions = gaussian.get_xyz.contiguous() if self.use_index_prior else None

        # Inter-layer loss statistics callback (no backprop)
        loss_callback = None
        if self._training_mode and assign and self.layer_aware_training:
            def sh_loss_callback(layer_idx, reconstruction):
                rec_unweighted = reconstruction
                if self.sh_band_weighting:
                    sh_weights = _cached_sh_weights(C, self.band_weight_alpha).to(reconstruction.device, non_blocking=True)
                    weights_expanded = sh_weights.repeat_interleave(3).unsqueeze(0)
                    assert weights_expanded.shape[1] == reconstruction.shape[1], "SH weight expansion mismatch in callback"
                    rec_unweighted = reconstruction / weights_expanded
                loss = torch.mean((rec_unweighted - feat) ** 2).item()
                self.record_layer_loss(layer_idx, loss)
                return loss
            loss_callback = sh_loss_callback

        if (not self._fitted) or assign:
            ids_list, q = self._rvq.fit(feat, loss_callback=loss_callback, positions=positions)
            self._fitted = True
        else:
            if not self._rvq_is_fitted():
                ids_list, q = self._rvq.fit(feat, loss_callback=loss_callback, positions=positions)
                self._fitted = True
            else:
                ids_list, q = self._rvq.quantize(feat, positions=positions)

        # Restore original scale (only if reweighting was applied previously)
        if self.sh_band_weighting:
            sh_weights = _cached_sh_weights(C, self.band_weight_alpha).to(q.device, non_blocking=True)
            weights_expanded = sh_weights.repeat_interleave(3).unsqueeze(0)  # (1, C*3)
            assert weights_expanded.shape[1] == q.shape[1], "SH weight expansion mismatch in recovery"
            q = q / weights_expanded

        gaussian.set_quantized_sh(self._shape_sh(q, C), ids_list[-1])
        self.cls_ids = ids_list[-1]
        self._last_ids_list = ids_list

    # ----------------------------
    # Export/state helpers
    # ----------------------------
    def get_codebooks(self):
        return self._rvq.get_codebooks()

    def get_all_level_indices(self):
        return self._last_ids_list

    # Compatible with old calls: export pipeline will call get_all_indices()
    def get_all_indices(self):
        return self.get_all_level_indices()

    def enable_layer_aware_training(self, enable: bool = True):
        """Enable/disable layer-aware training mode (statistics only, no gradient effect)"""
        self._training_mode = bool(enable)
        if enable:
            self._layer_losses = []

    def get_layer_loss_weights(self) -> List[float]:
        """
        Calculate per-layer loss weights (for logging/visualization only): later layers focus more on perceptual quality.
        Example weight for layer t: 0.3 + 0.2 * t / (L-1) (doesn't affect training numerics)
        """
        if not self.layer_aware_training or self.layers == 1:
            return [1.0] * self.layers

        weights = []
        for t in range(self.layers):
            lpips_w = 0.3 + 0.2 * t / max(1, (self.layers - 1))
            l1_w = 1.0 - 0.3 * t / max(1, (self.layers - 1))
            ssim_w = 0.2
            weights.append(lpips_w + l1_w + ssim_w)
        return weights

    def record_layer_loss(self, layer_idx: int, loss_value: float):
        """Record loss value for specific layer (statistics only)"""
        if self._training_mode:
            while len(self._layer_losses) <= layer_idx:
                self._layer_losses.append([])
            self._layer_losses[layer_idx].append(float(loss_value))

    def get_layer_loss_stats(self) -> dict:
        """Get layer loss statistics (for logging only)"""
        if not self._layer_losses:
            return {}
        stats = {}
        for i, losses in enumerate(self._layer_losses):
            if losses:
                stats[f"layer_{i}"] = {
                    "mean": sum(losses) / len(losses),
                    "count": len(losses),
                    "recent": losses[-10:],
                }
        return stats