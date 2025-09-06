# 相似度注意力
"""
相似度注意力機制模組

實現基於語義相似度的注意力機制，包括：
- 餘弦相似度注意力
- 歐氏距離注意力  
- 點積相似度注意力
- 學習式相似度注意力
- 多層感知機相似度注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np


class CosineSimilarityAttention(nn.Module):
    """
    餘弦相似度注意力機制
    
    使用餘弦相似度計算查詢和鍵之間的注意力權重
    """
    
    def __init__(self, 
                 input_dim: int,
                 temperature: float = 1.0,
                 dropout_rate: float = 0.1):
        """
        初始化餘弦相似度注意力
        
        Args:
            input_dim: 輸入特徵維度
            temperature: 溫度參數，用於控制注意力分佈的銳利度
            dropout_rate: Dropout比率
        """
        super(CosineSimilarityAttention, self).__init__()
        
        self.input_dim = input_dim
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout_rate)
        
        # 可學習的溫度參數
        self.learnable_temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len_q, input_dim]
            key: 鍵張量 [batch_size, seq_len_k, input_dim]
            value: 值張量 [batch_size, seq_len_v, input_dim]
            mask: 注意力遮罩 [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            注意力輸出和權重
        """
        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()
        
        # 正規化查詢和鍵
        query_norm = F.normalize(query, p=2, dim=-1)  # [batch_size, seq_len_q, input_dim]
        key_norm = F.normalize(key, p=2, dim=-1)      # [batch_size, seq_len_k, input_dim]
        
        # 計算餘弦相似度
        similarity = torch.matmul(query_norm, key_norm.transpose(-2, -1))  # [batch_size, seq_len_q, seq_len_k]
        
        # 應用溫度縮放
        similarity = similarity / self.learnable_temperature
        
        # 應用遮罩
        if mask is not None:
            similarity = similarity.masked_fill(mask == 0, -1e9)
        
        # 計算注意力權重
        attention_weights = F.softmax(similarity, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算注意力輸出
        attended_values = torch.matmul(attention_weights, value)
        
        return attended_values, attention_weights


class EuclideanDistanceAttention(nn.Module):
    """
    歐氏距離注意力機制
    
    使用歐氏距離的負數作為相似度計算注意力權重
    """
    
    def __init__(self, 
                 input_dim: int,
                 temperature: float = 1.0,
                 dropout_rate: float = 0.1):
        """
        初始化歐氏距離注意力
        
        Args:
            input_dim: 輸入特徵維度
            temperature: 溫度參數
            dropout_rate: Dropout比率
        """
        super(EuclideanDistanceAttention, self).__init__()
        
        self.input_dim = input_dim
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout_rate)
        
        # 可學習的溫度參數
        self.learnable_temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len_q, input_dim]
            key: 鍵張量 [batch_size, seq_len_k, input_dim]
            value: 值張量 [batch_size, seq_len_v, input_dim]
            mask: 注意力遮罩 [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            注意力輸出和權重
        """
        batch_size, seq_len_q, input_dim = query.size()
        _, seq_len_k, _ = key.size()
        
        # 擴展維度以進行廣播計算
        query_expanded = query.unsqueeze(2)  # [batch_size, seq_len_q, 1, input_dim]
        key_expanded = key.unsqueeze(1)      # [batch_size, 1, seq_len_k, input_dim]
        
        # 計算歐氏距離的平方
        distance_squared = torch.sum((query_expanded - key_expanded) ** 2, dim=-1)
        
        # 轉換為相似度（距離越小，相似度越大）
        similarity = -distance_squared / self.learnable_temperature
        
        # 應用遮罩
        if mask is not None:
            similarity = similarity.masked_fill(mask == 0, -1e9)
        
        # 計算注意力權重
        attention_weights = F.softmax(similarity, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算注意力輸出
        attended_values = torch.matmul(attention_weights, value)
        
        return attended_values, attention_weights


class DotProductSimilarityAttention(nn.Module):
    """
    點積相似度注意力機制
    
    使用縮放點積計算相似度的標準注意力機制
    """
    
    def __init__(self, 
                 input_dim: int,
                 dropout_rate: float = 0.1,
                 scale: Optional[float] = None):
        """
        初始化點積相似度注意力
        
        Args:
            input_dim: 輸入特徵維度
            dropout_rate: Dropout比率
            scale: 縮放因子，默認為sqrt(input_dim)
        """
        super(DotProductSimilarityAttention, self).__init__()
        
        self.input_dim = input_dim
        self.scale = scale or math.sqrt(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len_q, input_dim]
            key: 鍵張量 [batch_size, seq_len_k, input_dim]
            value: 值張量 [batch_size, seq_len_v, input_dim]
            mask: 注意力遮罩 [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            注意力輸出和權重
        """
        # 計算縮放點積
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # 應用遮罩
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 計算注意力權重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算注意力輸出
        attended_values = torch.matmul(attention_weights, value)
        
        return attended_values, attention_weights


class LearnableSimilarityAttention(nn.Module):
    """
    學習式相似度注意力機制
    
    通過可學習的參數矩陣計算查詢和鍵之間的相似度
    """
    
    def __init__(self, 
                 input_dim: int,
                 similarity_dim: int = 128,
                 dropout_rate: float = 0.1):
        """
        初始化學習式相似度注意力
        
        Args:
            input_dim: 輸入特徵維度
            similarity_dim: 相似度計算的中間維度
            dropout_rate: Dropout比率
        """
        super(LearnableSimilarityAttention, self).__init__()
        
        self.input_dim = input_dim
        self.similarity_dim = similarity_dim
        self.dropout = nn.Dropout(dropout_rate)
        
        # 相似度計算的投影層
        self.query_projection = nn.Linear(input_dim, similarity_dim)
        self.key_projection = nn.Linear(input_dim, similarity_dim)
        
        # 相似度融合層
        self.similarity_fusion = nn.Sequential(
            nn.Linear(similarity_dim * 2, similarity_dim),
            nn.ReLU(),
            nn.Linear(similarity_dim, 1)
        )
        
        # 初始化權重
        self._init_weights()
        
    def _init_weights(self):
        """初始化權重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len_q, input_dim]
            key: 鍵張量 [batch_size, seq_len_k, input_dim]
            value: 值張量 [batch_size, seq_len_v, input_dim]
            mask: 注意力遮罩 [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            注意力輸出和權重
        """
        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()
        
        # 投影查詢和鍵
        projected_query = self.query_projection(query)  # [batch_size, seq_len_q, similarity_dim]
        projected_key = self.key_projection(key)        # [batch_size, seq_len_k, similarity_dim]
        
        # 擴展維度進行配對計算
        query_expanded = projected_query.unsqueeze(2).expand(-1, -1, seq_len_k, -1)
        key_expanded = projected_key.unsqueeze(1).expand(-1, seq_len_q, -1, -1)
        
        # 連接查詢和鍵
        query_key_pairs = torch.cat([query_expanded, key_expanded], dim=-1)
        
        # 計算相似度分數
        similarity_scores = self.similarity_fusion(query_key_pairs).squeeze(-1)
        
        # 應用遮罩
        if mask is not None:
            similarity_scores = similarity_scores.masked_fill(mask == 0, -1e9)
        
        # 計算注意力權重
        attention_weights = F.softmax(similarity_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算注意力輸出
        attended_values = torch.matmul(attention_weights, value)
        
        return attended_values, attention_weights


class MLPSimilarityAttention(nn.Module):
    """
    多層感知機相似度注意力機制
    
    使用MLP網路學習複雜的相似度函數
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.1,
                 activation: str = 'relu'):
        """
        初始化MLP相似度注意力
        
        Args:
            input_dim: 輸入特徵維度
            hidden_dims: 隱藏層維度列表
            dropout_rate: Dropout比率
            activation: 激活函數類型
        """
        super(MLPSimilarityAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 構建MLP相似度網路
        layers = []
        prev_dim = input_dim * 2  # 查詢和鍵連接後的維度
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 輸出層
        layers.append(nn.Linear(prev_dim, 1))
        
        self.similarity_mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout_rate)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """獲取激活函數"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len_q, input_dim]
            key: 鍵張量 [batch_size, seq_len_k, input_dim]
            value: 值張量 [batch_size, seq_len_v, input_dim]
            mask: 注意力遮罩 [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            注意力輸出和權重
        """
        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()
        
        # 擴展維度進行配對計算
        query_expanded = query.unsqueeze(2).expand(-1, -1, seq_len_k, -1)
        key_expanded = key.unsqueeze(1).expand(-1, seq_len_q, -1, -1)
        
        # 連接查詢和鍵
        query_key_pairs = torch.cat([query_expanded, key_expanded], dim=-1)
        
        # 重塑為MLP輸入格式
        mlp_input = query_key_pairs.view(-1, self.input_dim * 2)
        
        # 通過MLP計算相似度
        similarity_scores = self.similarity_mlp(mlp_input)
        
        # 重塑回注意力分數格式
        similarity_scores = similarity_scores.view(batch_size, seq_len_q, seq_len_k)
        
        # 應用遮罩
        if mask is not None:
            similarity_scores = similarity_scores.masked_fill(mask == 0, -1e9)
        
        # 計算注意力權重
        attention_weights = F.softmax(similarity_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算注意力輸出
        attended_values = torch.matmul(attention_weights, value)
        
        return attended_values, attention_weights


class AdaptiveSimilarityAttention(nn.Module):
    """
    自適應相似度注意力機制
    
    根據輸入動態選擇最適合的相似度計算方法
    """
    
    def __init__(self, 
                 input_dim: int,
                 similarity_types: List[str] = ['cosine', 'euclidean', 'dot_product'],
                 dropout_rate: float = 0.1):
        """
        初始化自適應相似度注意力
        
        Args:
            input_dim: 輸入特徵維度
            similarity_types: 支持的相似度類型列表
            dropout_rate: Dropout比率
        """
        super(AdaptiveSimilarityAttention, self).__init__()
        
        self.input_dim = input_dim
        self.similarity_types = similarity_types
        self.num_similarity_types = len(similarity_types)
        
        # 初始化各種相似度注意力機制
        self.similarity_modules = nn.ModuleDict()
        
        if 'cosine' in similarity_types:
            self.similarity_modules['cosine'] = CosineSimilarityAttention(input_dim, dropout_rate=dropout_rate)
        
        if 'euclidean' in similarity_types:
            self.similarity_modules['euclidean'] = EuclideanDistanceAttention(input_dim, dropout_rate=dropout_rate)
        
        if 'dot_product' in similarity_types:
            self.similarity_modules['dot_product'] = DotProductSimilarityAttention(input_dim, dropout_rate=dropout_rate)
        
        if 'learnable' in similarity_types:
            self.similarity_modules['learnable'] = LearnableSimilarityAttention(input_dim, dropout_rate=dropout_rate)
        
        # 選擇器網路
        self.selector_network = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, self.num_similarity_types),
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len_q, input_dim]
            key: 鍵張量 [batch_size, seq_len_k, input_dim]
            value: 值張量 [batch_size, seq_len_v, input_dim]
            mask: 注意力遮罩 [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            注意力輸出和權重
        """
        batch_size = query.size(0)
        
        # 計算全局特徵用於選擇器
        query_global = torch.mean(query, dim=1)  # [batch_size, input_dim]
        key_global = torch.mean(key, dim=1)      # [batch_size, input_dim]
        global_features = torch.cat([query_global, key_global], dim=-1)
        
        # 計算相似度類型權重
        similarity_weights = self.selector_network(global_features)  # [batch_size, num_similarity_types]
        
        # 計算各種相似度注意力的輸出
        all_attended_values = []
        all_attention_weights = []
        
        for i, similarity_type in enumerate(self.similarity_types):
            if similarity_type in self.similarity_modules:
                attended, weights = self.similarity_modules[similarity_type](query, key, value, mask)
                all_attended_values.append(attended)
                all_attention_weights.append(weights)
        
        # 加權組合所有相似度注意力的輸出
        if all_attended_values:
            stacked_attended = torch.stack(all_attended_values, dim=0)  # [num_types, batch_size, seq_len_q, input_dim]
            stacked_weights = torch.stack(all_attention_weights, dim=0)  # [num_types, batch_size, seq_len_q, seq_len_k]
            
            # 擴展相似度權重維度以進行加權
            similarity_weights_expanded = similarity_weights.view(batch_size, self.num_similarity_types, 1, 1)
            similarity_weights_attended = similarity_weights_expanded.expand(-1, -1, query.size(1), self.input_dim)
            similarity_weights_attention = similarity_weights_expanded.expand(-1, -1, query.size(1), key.size(1))
            
            # 計算加權平均
            final_attended = torch.sum(
                stacked_attended * similarity_weights_attended.permute(1, 0, 2, 3), 
                dim=0
            )
            final_attention_weights = torch.sum(
                stacked_weights * similarity_weights_attention.permute(1, 0, 2, 3), 
                dim=0
            )
        else:
            # 如果沒有可用的相似度模組，返回零張量
            final_attended = torch.zeros_like(query)
            final_attention_weights = torch.zeros(batch_size, query.size(1), key.size(1), device=query.device)
        
        return final_attended, final_attention_weights