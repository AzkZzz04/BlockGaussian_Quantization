# 保持老接口：forward_dc / forward_frest
# 内部用 ResidualKMeansVQ 实现；当层数=1时就等价普通VQ
import math
import torch
from typing import List, Optional
from .rvq import ResidualKMeansVQ  # 你已有的 RVQ 实现

def _k_per_layer(target_k: int, L: int) -> List[int]:
    """
    计算每层的聚类数 - 渐进式策略 (第一层最大，后续层递减)
    使用简单的比例分配，更直观易懂
    """
    if L == 1:
        return [target_k]
    
    # 渐进式权重分配：第一层占主导，后续层递减
    # 基于倒数递减：[1, 1/2, 1/3, 1/4, ...]
    weights = [1.0 / (i + 1) for i in range(L)]
    total_weight = sum(weights)
    
    # 归一化权重，确保总和为1
    normalized_weights = [w / total_weight for w in weights]
    
    # 分配聚类数，确保每层至少有2个聚类
    result = []
    allocated_total = 0
    
    for i, weight in enumerate(normalized_weights):
        if i == L - 1:  # 最后一层：分配剩余的
            k = max(2, target_k - allocated_total)
        else:
            k = max(2, int(target_k * weight))
            allocated_total += k
        result.append(k)
    
    return result

def _compute_sh_band_weights(sh_degree: int, alpha: float = 0.15) -> torch.Tensor:
    """
    计算SH频带权重：w_l = 1 + α * l(l+1)
    高频分量获得更高权重，改善LPIPS
    包含DC分量 (l=0)
    """
    weights = []
    for l in range(sh_degree + 1):
        for m in range(-l, l + 1):
            weight = 1.0 + alpha * l * (l + 1)
            weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)

def _compute_sh_band_weights_no_dc(sh_degree: int, alpha: float = 0.15) -> torch.Tensor:
    """
    计算SH频带权重：w_l = 1 + α * l(l+1)  
    高频分量获得更高权重，改善LPIPS
    不包含DC分量 (跳过l=0)
    """
    weights = []
    for l in range(1, sh_degree + 1):  # 从l=1开始，跳过DC
        for m in range(-l, l + 1):
            weight = 1.0 + alpha * l * (l + 1)
            weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)

class Quantize_RVQ:
    """
    Drop-in 适配器：兼容旧的 Quantize_kMeans API。
    - forward_dc(gaussian, assign=False)
    - forward_frest(gaussian, assign=False)

    参数：
      which: 'dc' 或 'sh'，用于日志与形状处理
      target_k: 目标码本大小；当 layers=1 时就是普通 VQ 的 K
      layers: RVQ 层数（=1 等价普通 VQ）
      num_iters: 每层 KMeans 迭代数
      sh_band_weighting: 是否对SH使用频带重加权（改善高频保持）
      band_weight_alpha: 频带权重强度参数
    """
    def __init__(self,
                 which: str,
                 target_k: int = 4096,
                 layers: int = 1,
                 num_iters: int = 10,
                 sh_band_weighting: bool = True,
                 band_weight_alpha: float = 0.15,
                 layer_aware_training: bool = True):
        assert which in ("dc", "sh"), "which must be 'dc' or 'sh'"
        self.which = which
        self.layers = max(1, int(layers))
        self.target_k = int(target_k)
        self.num_iters = int(num_iters)
        self.sh_band_weighting = sh_band_weighting and (which == "sh")
        self.band_weight_alpha = band_weight_alpha
        self.layer_aware_training = layer_aware_training

        # 构造每层 K；使用渐进式分配策略
        layer_ks = _k_per_layer(self.target_k, self.layers)
        
        self._rvq = ResidualKMeansVQ(
            num_clusters=layer_ks,
            num_iters=self.num_iters
        )

        # 兼容性属性（旧代码可能会读）
        self.num_clusters = self.target_k          # 仅用于显示
        self.num_kmeans_iters = self.num_iters     # 仅用于显示
        self.cls_ids = torch.empty(0, dtype=torch.long)  # 最后一层索引（形状兼容）
        self.centers = torch.empty(0)              # RVQ 多层码本，不再单一 centers；保留占位

        self._fitted = False
        self._last_ids_list = None  # 多层索引缓存（用于落盘）
        self._sh_weights = None     # SH频带权重缓存
        
        # Layer-aware training state
        self._layer_losses = []     # 记录每层损失，用于层间对齐
        self._training_mode = False # 是否在训练模式（需要损失反馈）

    @torch.no_grad()
    def _flat_dc(self, gaussian):
        # (N,1,3) -> (N,3)
        return gaussian._features_dc.detach().reshape(-1, 3)

    @torch.no_grad()
    def _flat_sh(self, gaussian):
        # (N,C,3) -> (N, C*3)
        C = gaussian._features_rest.shape[1]
        feat = gaussian._features_rest.detach().reshape(-1, C * 3)
        
        # 应用频带重加权
        if self.sh_band_weighting:
            if self._sh_weights is None or self._sh_weights.device != feat.device:
                # 推断SH度数：处理不包含DC的SH情况
                # C=15 (degree=3, 不含DC): (3+1)^2 - 1 = 15
                # C=3 (degree=1, 不含DC): (1+1)^2 - 1 = 3  
                # 通用公式: degree = sqrt(C + 1) - 1
                sh_degree = int(math.sqrt(C + 1) - 1)
                
                # 调试信息
                expected_coeffs_with_dc = (sh_degree + 1) ** 2
                expected_coeffs_no_dc = expected_coeffs_with_dc - 1
                print(f"[SH Band Weighting] C={C}, inferred degree={sh_degree}")
                print(f"  Expected coeffs: with_DC={expected_coeffs_with_dc}, no_DC={expected_coeffs_no_dc}")
                
                # 计算权重时只对实际存在的系数分量计算
                self._sh_weights = _compute_sh_band_weights_no_dc(sh_degree, self.band_weight_alpha).to(feat.device)
                
                # 验证维度匹配
                if len(self._sh_weights) != C:
                    print(f"[WARNING] SH weight dimension mismatch: expected {C}, got {len(self._sh_weights)}")
                    print(f"  C={C}, inferred sh_degree={sh_degree}")
                    # 降级到不使用频带重加权
                    self._sh_weights = torch.ones(C, dtype=torch.float32, device=feat.device)
                else:
                    print(f"[SH Band Weighting] Successfully created weights: shape={self._sh_weights.shape}")
                    # 打印权重值分布用于调试
                    print(f"  Weight range: [{self._sh_weights.min():.3f}, {self._sh_weights.max():.3f}]")
            
            # 重加权：对每个3D点的每个SH系数应用权重
            # feat: (N, C*3), weights: (C,) -> 广播到 (N, C*3)
            if len(self._sh_weights) == C:
                weights_expanded = self._sh_weights.repeat_interleave(3).unsqueeze(0)  # (1, C*3)
                if weights_expanded.shape[1] == feat.shape[1]:  # 双重检查维度
                    feat = feat * weights_expanded
                else:
                    print(f"[WARNING] Weight expansion dimension mismatch: {weights_expanded.shape[1]} vs {feat.shape[1]}")
            else:
                print(f"[WARNING] Skipping SH band weighting due to dimension mismatch")
            
        return feat

    @staticmethod
    def _shape_dc(x):
        # (N, 3) -> (N,1,3)
        return x.reshape(-1, 1, 3)

    @staticmethod
    def _shape_sh(x, C):
        # (N, C*3) -> (N,C,3)
        return x.reshape(-1, C, 3)

    @torch.no_grad()
    def forward_dc(self, gaussian, assign: bool = False):
        if self.which != "dc":
            return
        feat = self._flat_dc(gaussian)
        
        # 准备层间感知训练的损失回调
        loss_callback = None
        if self._training_mode and assign and self.layer_aware_training:
            def dc_loss_callback(layer_idx, reconstruction):
                # 临时设置量化DC来计算损失
                original_dc = gaussian._features_dc.clone()
                try:
                    gaussian._features_dc = self._shape_dc(reconstruction)
                    # 这里需要实际的损失计算，暂时用简单的L2作为占位符
                    loss = torch.mean((reconstruction - feat) ** 2).item()
                    self.record_layer_loss(layer_idx, loss)
                    return loss
                finally:
                    gaussian._features_dc = original_dc
            loss_callback = dc_loss_callback
        
        if (not self._fitted) or assign:
            ids_list, q = self._rvq.fit(feat, loss_callback=loss_callback)
            self._fitted = True
        else:
            ids_list, q = self._rvq.quantize(feat)
        gaussian.set_quantized_dc(self._shape_dc(q), ids_list[-1])
        self.cls_ids = ids_list[-1]                    # 兼容旧保存逻辑（仅保存最后一层）
        self._last_ids_list = ids_list                 # 完整多层索引缓存

    @torch.no_grad()
    def forward_frest(self, gaussian, assign: bool = False):
        if self.which != "sh":
            return
        C = gaussian._features_rest.shape[1]
        feat = self._flat_sh(gaussian)
        
        # 准备层间感知训练的损失回调
        loss_callback = None
        if self._training_mode and assign and self.layer_aware_training:
            def sh_loss_callback(layer_idx, reconstruction):
                # 反向重加权恢复（如果启用了频带重加权）
                rec_unweighted = reconstruction
                if self.sh_band_weighting and self._sh_weights is not None:
                    if len(self._sh_weights) == C:
                        weights_expanded = self._sh_weights.repeat_interleave(3).unsqueeze(0)
                        if weights_expanded.shape[1] == reconstruction.shape[1]:
                            rec_unweighted = reconstruction / weights_expanded
                
                # 临时设置量化SH来计算损失
                original_sh = gaussian._features_rest.clone()
                try:
                    gaussian._features_rest = self._shape_sh(rec_unweighted, C)
                    # 这里需要实际的损失计算，暂时用简单的L2作为占位符
                    loss = torch.mean((rec_unweighted - feat) ** 2).item()
                    self.record_layer_loss(layer_idx, loss)
                    return loss
                finally:
                    gaussian._features_rest = original_sh
            loss_callback = sh_loss_callback
        
        if (not self._fitted) or assign:
            ids_list, q = self._rvq.fit(feat, loss_callback=loss_callback)
            self._fitted = True
        else:
            ids_list, q = self._rvq.quantize(feat)
        
        # 反向重加权恢复
        if self.sh_band_weighting and self._sh_weights is not None:
            if len(self._sh_weights) == C:
                weights_expanded = self._sh_weights.repeat_interleave(3).unsqueeze(0)  # (1, C*3)
                if weights_expanded.shape[1] == q.shape[1]:  # 维度检查
                    q = q / weights_expanded  # 恢复原始尺度
            
        gaussian.set_quantized_sh(self._shape_sh(q, C), ids_list[-1])
        self.cls_ids = ids_list[-1]
        self._last_ids_list = ids_list

    # 提供获取多层码本/索引的接口，供落盘
    def get_codebooks(self):
        return self._rvq.get_codebooks()

    def get_all_level_indices(self):
        return self._last_ids_list

    def enable_layer_aware_training(self, enable: bool = True):
        """启用/禁用层间感知训练模式"""
        self._training_mode = enable
        if enable:
            self._layer_losses = []
    
    def get_layer_loss_weights(self) -> List[float]:
        """
        计算每层的损失权重：后层更注重感知质量
        Layer t 的 LPIPS 权重: 0.3 + 0.2 * t / L
        """
        if not self.layer_aware_training or self.layers == 1:
            return [1.0] * self.layers
        
        weights = []
        for t in range(self.layers):
            # 基础权重 + 层间递增
            lpips_weight = 0.3 + 0.2 * t / (self.layers - 1)
            l1_weight = 1.0 - 0.3 * t / (self.layers - 1)  # 早层更关注L1
            ssim_weight = 0.2  # 保持固定
            
            # 组合权重 (相对于标准损失 L1 + 0.2*SSIM + 0.5*LPIPS)
            total_weight = l1_weight + ssim_weight + lpips_weight
            weights.append(total_weight)
        
        return weights
    
    def record_layer_loss(self, layer_idx: int, loss_value: float):
        """记录特定层的损失值，用于分析"""
        if self._training_mode:
            while len(self._layer_losses) <= layer_idx:
                self._layer_losses.append([])
            self._layer_losses[layer_idx].append(loss_value)
    
    def get_layer_loss_stats(self) -> dict:
        """获取层损失统计信息"""
        if not self._layer_losses:
            return {}
        
        stats = {}
        for i, losses in enumerate(self._layer_losses):
            if losses:
                stats[f"layer_{i}"] = {
                    "mean": sum(losses) / len(losses),
                    "count": len(losses),
                    "recent": losses[-10:]  # 最近10个值
                }
        return stats