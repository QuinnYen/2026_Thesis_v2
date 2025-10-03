# 正則化注意力機制模組
"""
正則化注意力機制實現

提供多種注意力正則化技術：
- 熵正則化: 控制注意力分佈的熵值，避免過度集中
- 稀疏性正則化: 促使注意力權重稀疏化，提高可解釋性
- 多樣性正則化: 增加不同注意力頭之間的多樣性
- 局部性正則化: 鼓勵注意力關注局部相關位置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class RegularizedAttention(nn.Module):
    """正則化注意力機制基礎類"""

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 regularization_config: Optional[Dict[str, Any]] = None):
        """
        初始化正則化注意力

        Args:
            hidden_dim: 隱藏層維度
            num_heads: 注意力頭數
            dropout: Dropout 比率
            regularization_config: 正則化配置
        """
        super(RegularizedAttention, self).__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim 必須能被 num_heads 整除"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # 線性變換層
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # 正則化配置
        self.regularization_config = regularization_config or {}
        self._setup_regularization()

        # 儲存注意力權重用於分析
        self.attention_weights = None
        self.regularization_losses = {}

    def _setup_regularization(self):
        """設置正則化參數"""
        config = self.regularization_config

        # 熵正則化
        self.use_entropy_reg = config.get('entropy_reg', {}).get('enabled', False)
        self.entropy_reg_weight = config.get('entropy_reg', {}).get('weight', 0.01)
        self.entropy_reg_target = config.get('entropy_reg', {}).get('target_entropy', None)

        # 稀疏性正則化
        self.use_sparsity_reg = config.get('sparsity_reg', {}).get('enabled', False)
        self.sparsity_reg_weight = config.get('sparsity_reg', {}).get('weight', 0.01)
        self.sparsity_reg_type = config.get('sparsity_reg', {}).get('type', 'l1')

        # 多樣性正則化
        self.use_diversity_reg = config.get('diversity_reg', {}).get('enabled', False)
        self.diversity_reg_weight = config.get('diversity_reg', {}).get('weight', 0.01)

        # 局部性正則化
        self.use_locality_reg = config.get('locality_reg', {}).get('enabled', False)
        self.locality_reg_weight = config.get('locality_reg', {}).get('weight', 0.01)
        self.locality_window_size = config.get('locality_reg', {}).get('window_size', 5)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向傳播

        Args:
            query: 查詢張量 [batch_size, seq_len, hidden_dim]
            key: 鍵張量 [batch_size, seq_len, hidden_dim]
            value: 值張量 [batch_size, seq_len, hidden_dim]
            mask: 注意力遮罩
            return_attention: 是否返回注意力權重

        Returns:
            output: 輸出張量
            attention_weights: 注意力權重 (如果 return_attention=True)
        """
        batch_size, seq_len, _ = query.size()

        # 線性變換並重塑
        Q = self.query_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 計算注意力分數
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 應用遮罩
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, -float('inf'))

        # 計算注意力權重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 儲存注意力權重
        self.attention_weights = attention_weights.detach()

        # 計算正則化損失
        if self.training:
            self._compute_regularization_losses(attention_weights, seq_len)

        # 應用注意力權重
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # 輸出投影
        output = self.out_proj(context)

        if return_attention:
            return output, attention_weights
        return output, None

    def _compute_regularization_losses(self, attention_weights: torch.Tensor, seq_len: int):
        """計算正則化損失"""
        self.regularization_losses.clear()

        if self.use_entropy_reg:
            self.regularization_losses['entropy'] = self._compute_entropy_regularization(attention_weights)

        if self.use_sparsity_reg:
            self.regularization_losses['sparsity'] = self._compute_sparsity_regularization(attention_weights)

        if self.use_diversity_reg:
            self.regularization_losses['diversity'] = self._compute_diversity_regularization(attention_weights)

        if self.use_locality_reg:
            self.regularization_losses['locality'] = self._compute_locality_regularization(attention_weights, seq_len)

    def _compute_entropy_regularization(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        計算熵正則化損失

        Args:
            attention_weights: 注意力權重 [batch_size, num_heads, seq_len, seq_len]

        Returns:
            entropy_loss: 熵正則化損失
        """
        # 計算注意力分佈的熵
        # 添加小常數避免 log(0)
        eps = 1e-8
        attention_weights_safe = attention_weights + eps

        # 計算每個位置的熵: H = -∑(p * log(p))
        entropy = -torch.sum(attention_weights_safe * torch.log(attention_weights_safe), dim=-1)

        # 平均熵
        mean_entropy = entropy.mean()

        if self.entropy_reg_target is not None:
            # 如果設置了目標熵，計算與目標熵的差距
            entropy_loss = self.entropy_reg_weight * torch.abs(mean_entropy - self.entropy_reg_target)
        else:
            # 否則直接使用熵作為正則化項（鼓勵高熵，即均勻分佈）
            entropy_loss = -self.entropy_reg_weight * mean_entropy

        return entropy_loss

    def _compute_sparsity_regularization(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        計算稀疏性正則化損失

        Args:
            attention_weights: 注意力權重 [batch_size, num_heads, seq_len, seq_len]

        Returns:
            sparsity_loss: 稀疏性正則化損失
        """
        if self.sparsity_reg_type == 'l1':
            # L1 正則化：促進稀疏性
            sparsity_loss = self.sparsity_reg_weight * torch.sum(attention_weights)
        elif self.sparsity_reg_type == 'l2':
            # L2 正則化：防止權重過大
            sparsity_loss = self.sparsity_reg_weight * torch.sum(attention_weights ** 2)
        elif self.sparsity_reg_type == 'gini':
            # 基尼係數：衡量分佈不均勻程度
            sorted_weights, _ = torch.sort(attention_weights.view(-1))
            n = sorted_weights.size(0)
            index = torch.arange(1, n + 1, dtype=torch.float, device=sorted_weights.device)
            gini = (2 * torch.sum(index * sorted_weights)) / (n * torch.sum(sorted_weights)) - (n + 1) / n
            sparsity_loss = self.sparsity_reg_weight * (1 - gini)  # 鼓勵高不均勻性（稀疏性）
        else:
            raise ValueError(f"不支援的稀疏性正則化類型: {self.sparsity_reg_type}")

        return sparsity_loss

    def _compute_diversity_regularization(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        計算多樣性正則化損失

        Args:
            attention_weights: 注意力權重 [batch_size, num_heads, seq_len, seq_len]

        Returns:
            diversity_loss: 多樣性正則化損失
        """
        # 計算不同注意力頭之間的相似性
        batch_size, num_heads, seq_len, _ = attention_weights.size()

        # 重塑為 [batch_size * seq_len, num_heads, seq_len]
        reshaped_weights = attention_weights.transpose(1, 2).contiguous().view(-1, num_heads, seq_len)

        # 計算注意力頭間的餘弦相似度
        similarities = []
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                head_i = reshaped_weights[:, i, :]  # [batch_size * seq_len, seq_len]
                head_j = reshaped_weights[:, j, :]  # [batch_size * seq_len, seq_len]

                # 計算餘弦相似度
                similarity = F.cosine_similarity(head_i, head_j, dim=-1).mean()
                similarities.append(similarity)

        # 平均相似度（越高表示多樣性越低）
        avg_similarity = torch.stack(similarities).mean()

        # 多樣性損失：鼓勵低相似度（高多樣性）
        diversity_loss = self.diversity_reg_weight * avg_similarity

        return diversity_loss

    def _compute_locality_regularization(self, attention_weights: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        計算局部性正則化損失

        Args:
            attention_weights: 注意力權重 [batch_size, num_heads, seq_len, seq_len]
            seq_len: 序列長度

        Returns:
            locality_loss: 局部性正則化損失
        """
        # 創建局部性遮罩：鼓勵注意力關注局部窗口內的位置
        device = attention_weights.device

        # 創建位置距離矩陣
        positions = torch.arange(seq_len, device=device).unsqueeze(1).float()
        position_diff = torch.abs(positions - positions.T)  # [seq_len, seq_len]

        # 計算基於距離的權重：距離越遠權重越小
        distance_weights = torch.exp(-position_diff / self.locality_window_size)
        distance_weights = distance_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # 計算注意力權重與理想局部性分佈的距離
        # 使用 KL 散度衡量差異
        eps = 1e-8
        attention_safe = attention_weights + eps
        distance_safe = distance_weights + eps

        # 正規化 distance_weights
        distance_weights_norm = distance_weights / distance_weights.sum(dim=-1, keepdim=True)

        # KL 散度: KL(P||Q) = ∑(P * log(P/Q))
        kl_div = torch.sum(attention_safe * torch.log(attention_safe / (distance_weights_norm + eps)), dim=-1)
        locality_loss = self.locality_reg_weight * kl_div.mean()

        return locality_loss

    def get_regularization_losses(self) -> Dict[str, torch.Tensor]:
        """獲取所有正則化損失"""
        return self.regularization_losses.copy()

    def get_total_regularization_loss(self) -> torch.Tensor:
        """獲取總正則化損失"""
        if not self.regularization_losses:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total_loss = sum(self.regularization_losses.values())
        return total_loss

    def get_attention_statistics(self) -> Dict[str, float]:
        """獲取注意力統計資訊"""
        if self.attention_weights is None:
            return {}

        with torch.no_grad():
            weights = self.attention_weights.cpu()

            stats = {
                'mean_attention': weights.mean().item(),
                'std_attention': weights.std().item(),
                'max_attention': weights.max().item(),
                'min_attention': weights.min().item(),
                'sparsity_ratio': (weights < 0.01).float().mean().item(),  # 小於 0.01 的權重比例
            }

            # 計算熵
            eps = 1e-8
            weights_safe = weights + eps
            entropy = -torch.sum(weights_safe * torch.log(weights_safe), dim=-1)
            stats['mean_entropy'] = entropy.mean().item()
            stats['std_entropy'] = entropy.std().item()

            return stats


class AttentionRegularizer:
    """注意力正則化工具類"""

    @staticmethod
    def create_regularization_config(entropy_weight: float = 0.0,
                                   sparsity_weight: float = 0.0,
                                   diversity_weight: float = 0.0,
                                   locality_weight: float = 0.0,
                                   **kwargs) -> Dict[str, Any]:
        """
        創建正則化配置

        Args:
            entropy_weight: 熵正則化權重
            sparsity_weight: 稀疏性正則化權重
            diversity_weight: 多樣性正則化權重
            locality_weight: 局部性正則化權重
            **kwargs: 其他配置參數

        Returns:
            regularization_config: 正則化配置字典
        """
        config = {
            'entropy_reg': {
                'enabled': entropy_weight > 0,
                'weight': entropy_weight,
                'target_entropy': kwargs.get('target_entropy', None)
            },
            'sparsity_reg': {
                'enabled': sparsity_weight > 0,
                'weight': sparsity_weight,
                'type': kwargs.get('sparsity_type', 'l1')
            },
            'diversity_reg': {
                'enabled': diversity_weight > 0,
                'weight': diversity_weight
            },
            'locality_reg': {
                'enabled': locality_weight > 0,
                'weight': locality_weight,
                'window_size': kwargs.get('locality_window', 5)
            }
        }

        return config

    @staticmethod
    def analyze_attention_pattern(attention_weights: torch.Tensor) -> Dict[str, Any]:
        """
        分析注意力模式

        Args:
            attention_weights: 注意力權重 [batch_size, num_heads, seq_len, seq_len]

        Returns:
            analysis: 分析結果
        """
        with torch.no_grad():
            analysis = {}

            # 基本統計
            analysis['shape'] = list(attention_weights.shape)
            analysis['mean'] = attention_weights.mean().item()
            analysis['std'] = attention_weights.std().item()
            analysis['max'] = attention_weights.max().item()
            analysis['min'] = attention_weights.min().item()

            # 熵分析
            eps = 1e-8
            weights_safe = attention_weights + eps
            entropy = -torch.sum(weights_safe * torch.log(weights_safe), dim=-1)
            analysis['entropy'] = {
                'mean': entropy.mean().item(),
                'std': entropy.std().item(),
                'min': entropy.min().item(),
                'max': entropy.max().item()
            }

            # 稀疏性分析
            sparsity_thresholds = [0.001, 0.01, 0.05, 0.1]
            analysis['sparsity'] = {}
            for threshold in sparsity_thresholds:
                ratio = (attention_weights < threshold).float().mean().item()
                analysis['sparsity'][f'below_{threshold}'] = ratio

            # 局部性分析（只對最後兩個維度分析）
            seq_len = attention_weights.size(-1)
            if seq_len > 1:
                # 計算注意力權重的「質心」
                positions = torch.arange(seq_len, dtype=torch.float, device=attention_weights.device)
                centroids = torch.sum(attention_weights * positions.view(1, 1, 1, -1), dim=-1)

                # 計算與對角線的平均距離
                diagonal_positions = torch.arange(seq_len, dtype=torch.float, device=attention_weights.device)
                diagonal_positions = diagonal_positions.view(1, 1, -1)

                locality_deviation = torch.abs(centroids - diagonal_positions).mean().item()
                analysis['locality_deviation'] = locality_deviation

            return analysis