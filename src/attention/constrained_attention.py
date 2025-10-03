# 約束注意力機制模組
"""
約束注意力機制實現

提供多種注意力約束技術：
- 語義約束: 基於語義相似性限制注意力分佈
- 位置約束: 基於位置關係約束注意力範圍
- 動態溫度調節: 根據輸入特徵動態調整注意力溫度

這些約束機制可以提高注意力的可解釋性和性能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import math
import numpy as np


class ConstrainedAttention(nn.Module):
    """約束注意力機制"""

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 constraint_config: Optional[Dict[str, Any]] = None):
        """
        初始化約束注意力

        Args:
            hidden_dim: 隱藏層維度
            num_heads: 注意力頭數
            dropout: Dropout 比率
            constraint_config: 約束配置
        """
        super(ConstrainedAttention, self).__init__()

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

        # 約束配置
        self.constraint_config = constraint_config or {}
        self._setup_constraints()

        # 儲存注意力權重和約束資訊
        self.attention_weights = None
        self.constraint_info = {}

    def _setup_constraints(self):
        """設置約束參數"""
        config = self.constraint_config

        # 語義約束
        self.use_semantic_constraint = config.get('semantic_constraint', {}).get('enabled', False)
        self.semantic_threshold = config.get('semantic_constraint', {}).get('threshold', 0.5)
        self.semantic_strength = config.get('semantic_constraint', {}).get('strength', 1.0)

        # 位置約束
        self.use_position_constraint = config.get('position_constraint', {}).get('enabled', False)
        self.position_window = config.get('position_constraint', {}).get('window_size', 10)
        self.position_decay = config.get('position_constraint', {}).get('decay_factor', 0.1)

        # 動態溫度調節
        self.use_dynamic_temperature = config.get('dynamic_temperature', {}).get('enabled', False)
        self.base_temperature = config.get('dynamic_temperature', {}).get('base_temperature', 1.0)
        self.temperature_range = config.get('dynamic_temperature', {}).get('temperature_range', [0.5, 2.0])

        # 溫度預測網路（如果使用動態溫度）
        if self.use_dynamic_temperature:
            self.temperature_predictor = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Sigmoid()
            )

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                semantic_features: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向傳播

        Args:
            query: 查詢張量 [batch_size, seq_len, hidden_dim]
            key: 鍵張量 [batch_size, seq_len, hidden_dim]
            value: 值張量 [batch_size, seq_len, hidden_dim]
            mask: 注意力遮罩
            semantic_features: 語義特徵用於語義約束
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

        # 計算基礎注意力分數
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 應用動態溫度調節
        if self.use_dynamic_temperature:
            temperature = self._compute_dynamic_temperature(query)
            attention_scores = attention_scores / temperature

        # 應用語義約束
        if self.use_semantic_constraint and semantic_features is not None:
            semantic_mask = self._compute_semantic_constraint(semantic_features)
            attention_scores = attention_scores + semantic_mask

        # 應用位置約束
        if self.use_position_constraint:
            position_mask = self._compute_position_constraint(seq_len, attention_scores.device)
            attention_scores = attention_scores + position_mask

        # 應用基本遮罩
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, -float('inf'))

        # 計算注意力權重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 儲存注意力權重和約束資訊
        self.attention_weights = attention_weights.detach()
        self._update_constraint_info(attention_weights, seq_len)

        # 應用注意力權重
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # 輸出投影
        output = self.out_proj(context)

        if return_attention:
            return output, attention_weights
        return output, None

    def _compute_dynamic_temperature(self, query: torch.Tensor) -> torch.Tensor:
        """
        計算動態溫度

        Args:
            query: 查詢張量 [batch_size, seq_len, hidden_dim]

        Returns:
            temperature: 動態溫度 [batch_size, 1, 1, 1]
        """
        # 使用查詢的平均值來預測溫度
        query_mean = query.mean(dim=1)  # [batch_size, hidden_dim]

        # 預測溫度係數 (0, 1)
        temp_coeff = self.temperature_predictor(query_mean)  # [batch_size, 1]

        # 映射到溫度範圍
        min_temp, max_temp = self.temperature_range
        temperature = min_temp + temp_coeff * (max_temp - min_temp)

        # 調整維度以便廣播
        temperature = temperature.view(-1, 1, 1, 1)

        return temperature

    def _compute_semantic_constraint(self, semantic_features: torch.Tensor) -> torch.Tensor:
        """
        計算語義約束遮罩

        Args:
            semantic_features: 語義特徵 [batch_size, seq_len, feature_dim]

        Returns:
            semantic_mask: 語義約束遮罩
        """
        batch_size, seq_len, _ = semantic_features.size()

        # 計算語義相似性矩陣
        # 正規化語義特徵
        semantic_norm = F.normalize(semantic_features, p=2, dim=-1)

        # 計算餘弦相似性
        similarity_matrix = torch.matmul(semantic_norm, semantic_norm.transpose(-2, -1))

        # 創建語義約束遮罩
        # 相似性低於閾值的位置會被懲罰
        semantic_mask = torch.where(
            similarity_matrix < self.semantic_threshold,
            -self.semantic_strength * (self.semantic_threshold - similarity_matrix),
            torch.zeros_like(similarity_matrix)
        )

        # 擴展到多頭注意力維度
        semantic_mask = semantic_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        return semantic_mask

    def _compute_position_constraint(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        計算位置約束遮罩

        Args:
            seq_len: 序列長度
            device: 設備

        Returns:
            position_mask: 位置約束遮罩
        """
        # 創建位置索引矩陣
        positions = torch.arange(seq_len, device=device).unsqueeze(1)
        position_diff = torch.abs(positions - positions.T).float()

        # 計算位置約束：距離超過窗口的位置會被懲罰
        position_mask = torch.where(
            position_diff > self.position_window,
            -self.position_decay * (position_diff - self.position_window),
            torch.zeros_like(position_diff)
        )

        # 擴展到批次和多頭維度
        position_mask = position_mask.unsqueeze(0).unsqueeze(0)

        return position_mask

    def _update_constraint_info(self, attention_weights: torch.Tensor, seq_len: int):
        """更新約束資訊統計"""
        self.constraint_info.clear()

        with torch.no_grad():
            # 計算注意力分佈的統計資訊
            weights = attention_weights.cpu()

            # 基本統計
            self.constraint_info['mean_attention'] = weights.mean().item()
            self.constraint_info['std_attention'] = weights.std().item()
            self.constraint_info['max_attention'] = weights.max().item()

            # 局部性統計
            # 計算注意力權重的「重心」位置
            positions = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
            centroids = torch.sum(weights * positions, dim=-1)
            expected_positions = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).float()

            locality_deviation = torch.abs(centroids - expected_positions).mean().item()
            self.constraint_info['locality_deviation'] = locality_deviation

            # 稀疏性統計
            sparsity_ratio = (weights < 0.01).float().mean().item()
            self.constraint_info['sparsity_ratio'] = sparsity_ratio

    def get_constraint_info(self) -> Dict[str, Any]:
        """獲取約束資訊"""
        return self.constraint_info.copy()

    def apply_semantic_guidance(self,
                              attention_weights: torch.Tensor,
                              semantic_scores: torch.Tensor,
                              guidance_strength: float = 0.5) -> torch.Tensor:
        """
        應用語義引導調整注意力權重

        Args:
            attention_weights: 原始注意力權重
            semantic_scores: 語義分數
            guidance_strength: 引導強度

        Returns:
            調整後的注意力權重
        """
        # 正規化語義分數
        semantic_scores_norm = F.softmax(semantic_scores, dim=-1)

        # 混合原始注意力和語義引導
        guided_weights = (1 - guidance_strength) * attention_weights + guidance_strength * semantic_scores_norm

        # 重新正規化
        guided_weights = F.softmax(guided_weights, dim=-1)

        return guided_weights

    def visualize_constraints(self) -> Dict[str, torch.Tensor]:
        """
        視覺化約束效果

        Returns:
            約束視覺化資料
        """
        if self.attention_weights is None:
            return {}

        visualization_data = {}

        # 注意力權重熱圖
        visualization_data['attention_heatmap'] = self.attention_weights.mean(dim=1).cpu()  # 平均所有頭

        # 如果有約束資訊，也包含進來
        if hasattr(self, 'semantic_similarity'):
            visualization_data['semantic_similarity'] = self.semantic_similarity

        return visualization_data


class AttentionConstraintManager:
    """注意力約束管理器"""

    @staticmethod
    def create_constraint_config(semantic_enabled: bool = False,
                               position_enabled: bool = False,
                               temperature_enabled: bool = False,
                               **kwargs) -> Dict[str, Any]:
        """
        創建約束配置

        Args:
            semantic_enabled: 是否啟用語義約束
            position_enabled: 是否啟用位置約束
            temperature_enabled: 是否啟用動態溫度
            **kwargs: 其他配置參數

        Returns:
            約束配置字典
        """
        config = {
            'semantic_constraint': {
                'enabled': semantic_enabled,
                'threshold': kwargs.get('semantic_threshold', 0.5),
                'strength': kwargs.get('semantic_strength', 1.0)
            },
            'position_constraint': {
                'enabled': position_enabled,
                'window_size': kwargs.get('position_window', 10),
                'decay_factor': kwargs.get('position_decay', 0.1)
            },
            'dynamic_temperature': {
                'enabled': temperature_enabled,
                'base_temperature': kwargs.get('base_temperature', 1.0),
                'temperature_range': kwargs.get('temperature_range', [0.5, 2.0])
            }
        }

        return config

    @staticmethod
    def validate_constraint_config(config: Dict[str, Any]) -> bool:
        """
        驗證約束配置

        Args:
            config: 約束配置

        Returns:
            配置是否有效
        """
        required_sections = ['semantic_constraint', 'position_constraint', 'dynamic_temperature']

        for section in required_sections:
            if section not in config:
                return False

            section_config = config[section]
            if 'enabled' not in section_config:
                return False

        return True

    @staticmethod
    def get_preset_configs() -> Dict[str, Dict[str, Any]]:
        """
        獲取預設約束配置

        Returns:
            預設配置字典
        """
        presets = {
            'no_constraint': {
                'semantic_constraint': {'enabled': False},
                'position_constraint': {'enabled': False},
                'dynamic_temperature': {'enabled': False}
            },

            'semantic_only': {
                'semantic_constraint': {
                    'enabled': True,
                    'threshold': 0.5,
                    'strength': 1.0
                },
                'position_constraint': {'enabled': False},
                'dynamic_temperature': {'enabled': False}
            },

            'position_only': {
                'semantic_constraint': {'enabled': False},
                'position_constraint': {
                    'enabled': True,
                    'window_size': 10,
                    'decay_factor': 0.1
                },
                'dynamic_temperature': {'enabled': False}
            },

            'dynamic_temperature': {
                'semantic_constraint': {'enabled': False},
                'position_constraint': {'enabled': False},
                'dynamic_temperature': {
                    'enabled': True,
                    'base_temperature': 1.0,
                    'temperature_range': [0.5, 2.0]
                }
            },

            'all_constraints': {
                'semantic_constraint': {
                    'enabled': True,
                    'threshold': 0.6,
                    'strength': 0.5
                },
                'position_constraint': {
                    'enabled': True,
                    'window_size': 8,
                    'decay_factor': 0.05
                },
                'dynamic_temperature': {
                    'enabled': True,
                    'base_temperature': 1.0,
                    'temperature_range': [0.7, 1.5]
                }
            },

            'conservative': {
                'semantic_constraint': {
                    'enabled': True,
                    'threshold': 0.7,
                    'strength': 0.3
                },
                'position_constraint': {
                    'enabled': True,
                    'window_size': 5,
                    'decay_factor': 0.2
                },
                'dynamic_temperature': {
                    'enabled': True,
                    'base_temperature': 1.2,
                    'temperature_range': [0.9, 1.3]
                }
            }
        }

        return presets

    @staticmethod
    def analyze_constraint_effectiveness(attention_weights: torch.Tensor,
                                       constraint_info: Dict[str, Any],
                                       reference_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        分析約束機制的有效性

        Args:
            attention_weights: 約束後的注意力權重
            constraint_info: 約束資訊
            reference_weights: 參考注意力權重（無約束）

        Returns:
            效果分析結果
        """
        analysis = {}

        with torch.no_grad():
            weights = attention_weights.cpu()

            # 基本統計
            analysis['mean_attention'] = weights.mean().item()
            analysis['attention_entropy'] = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean().item()
            analysis['attention_concentration'] = weights.max(dim=-1)[0].mean().item()

            # 局部性分析
            seq_len = weights.size(-1)
            positions = torch.arange(seq_len).float()
            centroids = torch.sum(weights * positions.view(1, 1, 1, -1), dim=-1)
            expected_centroids = torch.arange(seq_len).view(1, 1, -1).float()
            locality_score = 1.0 / (1.0 + torch.abs(centroids - expected_centroids).mean().item())
            analysis['locality_score'] = locality_score

            # 如果有參考權重，計算相對變化
            if reference_weights is not None:
                ref_weights = reference_weights.cpu()
                kl_divergence = F.kl_div(torch.log(weights + 1e-8), ref_weights + 1e-8, reduction='mean').item()
                analysis['kl_divergence_from_reference'] = kl_divergence

                # 計算注意力分佈的變化
                entropy_change = analysis['attention_entropy'] - (-torch.sum(ref_weights * torch.log(ref_weights + 1e-8), dim=-1).mean().item())
                analysis['entropy_change'] = entropy_change

        return analysis


class SemanticConstraintModule(nn.Module):
    """語義約束專用模組"""

    def __init__(self, feature_dim: int, constraint_strength: float = 1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.constraint_strength = constraint_strength

        # 語義特徵投影
        self.semantic_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, features: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """計算語義約束遮罩"""
        # 投影語義特徵
        projected_features = self.semantic_proj(features)
        projected_features = F.normalize(projected_features, p=2, dim=-1)

        # 計算相似性矩陣
        similarity_matrix = torch.matmul(projected_features, projected_features.transpose(-2, -1))

        # 生成約束遮罩
        constraint_mask = torch.where(
            similarity_matrix < threshold,
            -self.constraint_strength * (threshold - similarity_matrix),
            torch.zeros_like(similarity_matrix)
        )

        return constraint_mask


class PositionConstraintModule(nn.Module):
    """位置約束專用模組"""

    def __init__(self, max_seq_len: int = 512, window_size: int = 10, decay_factor: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.decay_factor = decay_factor

        # 預計算位置約束矩陣
        self.register_buffer('position_matrix', self._create_position_matrix())

    def _create_position_matrix(self) -> torch.Tensor:
        """預計算位置約束矩陣"""
        positions = torch.arange(self.max_seq_len).unsqueeze(1)
        position_diff = torch.abs(positions - positions.T).float()

        position_matrix = torch.where(
            position_diff > self.window_size,
            -self.decay_factor * (position_diff - self.window_size),
            torch.zeros_like(position_diff)
        )

        return position_matrix

    def forward(self, seq_len: int) -> torch.Tensor:
        """獲取位置約束遮罩"""
        return self.position_matrix[:seq_len, :seq_len]


class DynamicTemperatureModule(nn.Module):
    """動態溫度調節專用模組"""

    def __init__(self,
                 input_dim: int,
                 base_temperature: float = 1.0,
                 temperature_range: List[float] = None):
        super().__init__()
        self.base_temperature = base_temperature
        self.temperature_range = temperature_range or [0.5, 2.0]

        # 溫度預測網路
        self.temperature_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """預測動態溫度"""
        # 計算特徵的統計資訊
        feature_stats = features.mean(dim=1)  # [batch_size, feature_dim]

        # 預測溫度係數
        temp_coeff = self.temperature_net(feature_stats)  # [batch_size, 1]

        # 映射到溫度範圍
        min_temp, max_temp = self.temperature_range
        temperature = min_temp + temp_coeff * (max_temp - min_temp)

        return temperature.view(-1, 1, 1, 1)