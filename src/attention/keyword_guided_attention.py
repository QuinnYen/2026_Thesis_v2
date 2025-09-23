# 關鍵詞導向注意力
"""
關鍵詞導向注意力機制模組

實現基於關鍵詞和方面詞的注意力機制，包括：
- 關鍵詞權重注意力
- 方面詞感知注意力
- 位置敏感關鍵詞注意力
- 多層級關鍵詞注意力
- 動態關鍵詞發現注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np


class KeywordWeightedAttention(nn.Module):
    """
    關鍵詞權重注意力機制
    
    根據預定義的關鍵詞列表來調整注意力權重
    """
    
    def __init__(self, 
                 input_dim: int,
                 keyword_dim: int = 64,
                 dropout_rate: float = 0.1):
        """
        初始化關鍵詞權重注意力
        
        Args:
            input_dim: 輸入特徵維度
            keyword_dim: 關鍵詞嵌入維度
            dropout_rate: Dropout比率
        """
        super(KeywordWeightedAttention, self).__init__()
        
        self.input_dim = input_dim
        self.keyword_dim = keyword_dim
        self.dropout = nn.Dropout(dropout_rate)
        
        # 關鍵詞嵌入層
        self.keyword_embedding = nn.Embedding(
            num_embeddings=10000,  # 支援最多10000個關鍵詞
            embedding_dim=keyword_dim
        )
        
        # 關鍵詞權重計算網路
        self.keyword_scorer = nn.Sequential(
            nn.Linear(input_dim + keyword_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
        # 標準注意力機制
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                keyword_ids: Optional[torch.Tensor] = None,
                keyword_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len_q, input_dim]
            key: 鍵張量 [batch_size, seq_len_k, input_dim]
            value: 值張量 [batch_size, seq_len_v, input_dim]
            keyword_ids: 關鍵詞ID張量 [batch_size, seq_len_k]
            keyword_mask: 關鍵詞遮罩 [batch_size, seq_len_k]
            attention_mask: 注意力遮罩
            
        Returns:
            注意力輸出和權重
        """
        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()
        
        # 基本注意力計算
        attended_values, attention_weights = self.attention(
            query, key, value, key_padding_mask=attention_mask
        )
        
        # 如果提供關鍵詞信息，調整注意力權重
        if keyword_ids is not None:
            # 獲取關鍵詞嵌入
            keyword_embeds = self.keyword_embedding(keyword_ids)  # [batch_size, seq_len_k, keyword_dim]
            
            # 計算關鍵詞權重
            key_keyword_concat = torch.cat([key, keyword_embeds], dim=-1)
            keyword_weights = self.keyword_scorer(key_keyword_concat)  # [batch_size, seq_len_k, 1]
            
            # 應用關鍵詞遮罩
            if keyword_mask is not None:
                keyword_weights = keyword_weights * keyword_mask.unsqueeze(-1).float()
            
            # 調整注意力權重
            keyword_weights_expanded = keyword_weights.expand(-1, -1, seq_len_q).transpose(1, 2)
            adjusted_attention_weights = attention_weights * keyword_weights_expanded
            
            # 重新正規化
            adjusted_attention_weights = F.softmax(adjusted_attention_weights, dim=-1)
            adjusted_attention_weights = self.dropout(adjusted_attention_weights)
            
            # 重新計算注意力輸出
            attended_values = torch.matmul(adjusted_attention_weights, value)
            attention_weights = adjusted_attention_weights
        
        return attended_values, attention_weights


class AspectAwareAttention(nn.Module):
    """
    方面詞感知注意力機制
    
    專門針對方面詞進行注意力計算
    """
    
    def __init__(self, 
                 input_dim: int,
                 aspect_dim: int = 128,
                 max_aspect_len: int = 10,
                 dropout_rate: float = 0.1):
        """
        初始化方面詞感知注意力
        
        Args:
            input_dim: 輸入特徵維度
            aspect_dim: 方面詞嵌入維度
            max_aspect_len: 方面詞的最大長度（基於數據集統計確定）
            dropout_rate: Dropout比率
        """
        super(AspectAwareAttention, self).__init__()
        
        self.input_dim = input_dim
        self.aspect_dim = aspect_dim
        self.max_aspect_len = max_aspect_len
        self.dropout = nn.Dropout(dropout_rate)
        
        # 方面詞投影層
        self.aspect_projection = nn.Linear(input_dim, aspect_dim)
        
        # 方面詞注意力計算
        self.aspect_attention = nn.Sequential(
            nn.Linear(aspect_dim, aspect_dim // 2),
            nn.Tanh(),
            nn.Linear(aspect_dim // 2, 1)
        )
        
        # 上下文感知層
        self.context_layer = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                aspect_positions: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len_q, input_dim]
            key: 鍵張量 [batch_size, seq_len_k, input_dim]
            value: 值張量 [batch_size, seq_len_v, input_dim]
            aspect_positions: 方面詞位置 [batch_size, 2] (start, end)
            attention_mask: 注意力遮罩
            
        Returns:
            注意力輸出和權重
        """
        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()
        
        # 基本上下文注意力
        context_attended, context_weights = self.context_layer(
            query, key, value, key_padding_mask=attention_mask
        )
        
        if aspect_positions is not None:
            # 提取方面詞特徵
            aspect_features = self._extract_aspect_features(key, aspect_positions)
            
            # 計算方面詞注意力權重
            aspect_projected = self.aspect_projection(aspect_features)
            aspect_attention_scores = self.aspect_attention(aspect_projected)
            aspect_weights = F.softmax(aspect_attention_scores, dim=1)
            
            # 方面詞加權平均
            aspect_representation = torch.sum(
                aspect_features * aspect_weights, dim=1, keepdim=True
            )  # [batch_size, 1, input_dim]
            
            # 擴展方面詞表示
            aspect_expanded = aspect_representation.expand(-1, seq_len_q, -1)
            
            # 結合上下文和方面詞信息
            combined_features = context_attended + aspect_expanded
            
            return combined_features, context_weights
        else:
            return context_attended, context_weights
    
    def _extract_aspect_features(self, 
                                features: torch.Tensor, 
                                aspect_positions: torch.Tensor) -> torch.Tensor:
        """
        提取方面詞特徵
        
        Args:
            features: 特徵張量 [batch_size, seq_len, input_dim]
            aspect_positions: 方面詞位置 [batch_size, 2]
            
        Returns:
            方面詞特徵 [batch_size, max_aspect_len, input_dim]
        """
        batch_size, seq_len, input_dim = features.size()
        max_aspect_len = self.max_aspect_len  # 使用配置的方面詞最大長度
        
        aspect_features = torch.zeros(batch_size, max_aspect_len, input_dim, device=features.device)
        
        for i in range(batch_size):
            start_pos, end_pos = aspect_positions[i]
            start_pos = max(0, min(start_pos.item(), seq_len - 1))
            end_pos = max(start_pos + 1, min(end_pos.item(), seq_len))
            
            aspect_len = end_pos - start_pos
            if aspect_len > 0:
                actual_len = min(aspect_len, max_aspect_len)
                aspect_features[i, :actual_len] = features[i, start_pos:start_pos + actual_len]
        
        return aspect_features


class PositionSensitiveKeywordAttention(nn.Module):
    """
    位置敏感關鍵詞注意力機制
    
    考慮關鍵詞在序列中的位置信息
    """
    
    def __init__(self, 
                 input_dim: int,
                 position_dim: int = 64,
                 max_length: int = 512,
                 dropout_rate: float = 0.1):
        """
        初始化位置敏感關鍵詞注意力
        
        Args:
            input_dim: 輸入特徵維度
            position_dim: 位置編碼維度
            max_length: 最大序列長度
            dropout_rate: Dropout比率
        """
        super(PositionSensitiveKeywordAttention, self).__init__()
        
        self.input_dim = input_dim
        self.position_dim = position_dim
        self.max_length = max_length
        self.dropout = nn.Dropout(dropout_rate)
        
        # 位置編碼
        self.position_embedding = nn.Embedding(max_length, position_dim)
        
        # 位置敏感的注意力計算
        self.position_attention = nn.Sequential(
            nn.Linear(input_dim + position_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh()
        )
        
        # 距離權重計算
        self.distance_scorer = nn.Sequential(
            nn.Linear(position_dim, position_dim // 2),
            nn.ReLU(),
            nn.Linear(position_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 多頭注意力
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                keyword_positions: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len_q, input_dim]
            key: 鍵張量 [batch_size, seq_len_k, input_dim]
            value: 值張量 [batch_size, seq_len_v, input_dim]
            keyword_positions: 關鍵詞位置 [batch_size, seq_len_k]
            attention_mask: 注意力遮罩
            
        Returns:
            注意力輸出和權重
        """
        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()
        
        if keyword_positions is not None:
            # 獲取位置編碼
            position_embeds = self.position_embedding(keyword_positions)  # [batch_size, seq_len_k, position_dim]
            
            # 結合特徵和位置信息
            key_with_position = torch.cat([key, position_embeds], dim=-1)
            enhanced_key = self.position_attention(key_with_position)
            
            # 計算位置距離權重
            distance_weights = self._compute_distance_weights(
                keyword_positions, seq_len_q, seq_len_k
            )
            
            # 多頭注意力計算
            attended_values, attention_weights = self.multihead_attention(
                query, enhanced_key, value, key_padding_mask=attention_mask
            )
            
            # 應用距離權重
            adjusted_weights = attention_weights * distance_weights
            adjusted_weights = F.softmax(adjusted_weights, dim=-1)
            adjusted_weights = self.dropout(adjusted_weights)
            
            # 重新計算注意力輸出
            attended_values = torch.matmul(adjusted_weights, value)
            
            return attended_values, adjusted_weights
        else:
            # 如果沒有位置信息，使用標準注意力
            return self.multihead_attention(
                query, key, value, key_padding_mask=attention_mask
            )
    
    def _compute_distance_weights(self, 
                                 positions: torch.Tensor,
                                 seq_len_q: int,
                                 seq_len_k: int) -> torch.Tensor:
        """
        計算位置距離權重
        
        Args:
            positions: 位置張量 [batch_size, seq_len_k]
            seq_len_q: 查詢序列長度
            seq_len_k: 鍵序列長度
            
        Returns:
            距離權重 [batch_size, seq_len_q, seq_len_k]
        """
        batch_size = positions.size(0)
        device = positions.device
        
        # 創建查詢位置
        query_positions = torch.arange(seq_len_q, device=device).expand(batch_size, -1)
        
        # 計算距離
        query_pos_expanded = query_positions.unsqueeze(2)  # [batch_size, seq_len_q, 1]
        key_pos_expanded = positions.unsqueeze(1)          # [batch_size, 1, seq_len_k]
        
        distances = torch.abs(query_pos_expanded - key_pos_expanded)  # [batch_size, seq_len_q, seq_len_k]
        
        # 轉換為權重（距離越近權重越大）
        distance_weights = 1.0 / (1.0 + distances.float())
        
        return distance_weights


class MultiLevelKeywordAttention(nn.Module):
    """
    多層級關鍵詞注意力機制
    
    在不同層級上計算關鍵詞注意力
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_levels: int = 3,
                 level_dims: Optional[List[int]] = None,
                 dropout_rate: float = 0.1):
        """
        初始化多層級關鍵詞注意力
        
        Args:
            input_dim: 輸入特徵維度
            num_levels: 層級數量
            level_dims: 各層級維度列表
            dropout_rate: Dropout比率
        """
        super(MultiLevelKeywordAttention, self).__init__()
        
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dims = level_dims or [input_dim // (2**i) for i in range(num_levels)]
        
        # 各層級的注意力模組
        self.level_attentions = nn.ModuleList()
        for i, level_dim in enumerate(self.level_dims):
            level_attention = nn.Sequential(
                nn.Linear(input_dim, level_dim),
                nn.ReLU(),
                nn.Linear(level_dim, level_dim),
                nn.Dropout(dropout_rate)
            )
            self.level_attentions.append(level_attention)
        
        # 層級融合網路
        total_dim = sum(self.level_dims)
        self.level_fusion = nn.Sequential(
            nn.Linear(total_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 最終注意力層
        self.final_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len_q, input_dim]
            key: 鍵張量 [batch_size, seq_len_k, input_dim]
            value: 值張量 [batch_size, seq_len_v, input_dim]
            attention_mask: 注意力遮罩
            
        Returns:
            注意力輸出和權重
        """
        # 在不同層級上計算特徵
        level_features = []
        for level_attention in self.level_attentions:
            level_key = level_attention(key)
            level_features.append(level_key)
        
        # 融合多層級特徵
        fused_key = torch.cat(level_features, dim=-1)
        enhanced_key = self.level_fusion(fused_key)
        
        # 計算最終注意力
        attended_values, attention_weights = self.final_attention(
            query, enhanced_key, value, key_padding_mask=attention_mask
        )
        
        return attended_values, attention_weights


class DynamicKeywordDiscoveryAttention(nn.Module):
    """
    動態關鍵詞發現注意力機制
    
    自動發現並關注重要的關鍵詞
    """
    
    def __init__(self, 
                 input_dim: int,
                 keyword_discovery_dim: int = 128,
                 top_k: int = 10,
                 dropout_rate: float = 0.1):
        """
        初始化動態關鍵詞發現注意力
        
        Args:
            input_dim: 輸入特徵維度
            keyword_discovery_dim: 關鍵詞發現維度
            top_k: 選擇的關鍵詞數量
            dropout_rate: Dropout比率
        """
        super(DynamicKeywordDiscoveryAttention, self).__init__()
        
        self.input_dim = input_dim
        self.keyword_discovery_dim = keyword_discovery_dim
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout_rate)
        
        # 關鍵詞發現網路
        self.keyword_discovery = nn.Sequential(
            nn.Linear(input_dim, keyword_discovery_dim),
            nn.ReLU(),
            nn.Linear(keyword_discovery_dim, keyword_discovery_dim),
            nn.Tanh(),
            nn.Linear(keyword_discovery_dim, 1)
        )
        
        # 關鍵詞增強網路
        self.keyword_enhancer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 注意力計算
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len_q, input_dim]
            key: 鍵張量 [batch_size, seq_len_k, input_dim]
            value: 值張量 [batch_size, seq_len_v, input_dim]
            attention_mask: 注意力遮罩
            
        Returns:
            注意力輸出和權重
        """
        batch_size, seq_len_k, _ = key.size()
        
        # 發現關鍵詞
        keyword_scores = self.keyword_discovery(key).squeeze(-1)  # [batch_size, seq_len_k]
        
        # 選擇Top-K關鍵詞
        if self.top_k < seq_len_k:
            top_k_scores, top_k_indices = torch.topk(
                keyword_scores, k=self.top_k, dim=-1
            )
            
            # 創建關鍵詞遮罩
            keyword_mask = torch.zeros_like(keyword_scores, dtype=torch.bool)
            keyword_mask.scatter_(1, top_k_indices, True)
        else:
            keyword_mask = torch.ones_like(keyword_scores, dtype=torch.bool)
        
        # 增強關鍵詞特徵
        enhanced_key = self.keyword_enhancer(key)
        
        # 應用關鍵詞遮罩
        keyword_weights = F.sigmoid(keyword_scores).unsqueeze(-1)
        enhanced_key = enhanced_key * keyword_weights
        
        # 計算注意力
        attended_values, attention_weights = self.attention(
            query, enhanced_key, value, key_padding_mask=attention_mask
        )
        
        return attended_values, attention_weights