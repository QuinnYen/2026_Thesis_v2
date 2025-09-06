# 自注意力機制模組
"""
自注意力機制實現

提供多種自注意力機制：
- 基礎自注意力: 標準 Transformer 風格自注意力
- 縮放點積自注意力: 加入縮放因子的自注意力
- 位置敏感自注意力: 結合位置編碼的自注意力
- 相對位置自注意力: 基於相對位置的自注意力
- 輕量級自注意力: 計算效率優化的自注意力
- 分組自注意力: 分組處理的自注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import math


class BasicSelfAttention(nn.Module):
    """基礎自注意力機制"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """
        初始化基礎自注意力
        
        Args:
            hidden_dim: 隱藏層維度
            dropout: Dropout 比率
        """
        super(BasicSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 線性變換層
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 輸出投影層
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [batch_size, seq_len, hidden_dim]
            mask: 注意力遮罩 [batch_size, seq_len, seq_len]
            
        Returns:
            output: 注意力輸出 [batch_size, seq_len, hidden_dim]
            attention_weights: 注意力權重 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # 線性變換
        query = self.query_projection(x)  # [batch_size, seq_len, hidden_dim]
        key = self.key_projection(x)      # [batch_size, seq_len, hidden_dim]
        value = self.value_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # 計算注意力分數
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        
        # 應用遮罩
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 正規化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算加權輸出
        context = torch.matmul(attention_weights, value)  # [batch_size, seq_len, hidden_dim]
        
        # 輸出投影
        output = self.output_projection(context)
        
        # 殘差連接和層正規化
        output = self.layer_norm(output + x)
        
        return output, attention_weights


class ScaledDotProductSelfAttention(nn.Module):
    """縮放點積自注意力機制"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """
        初始化縮放點積自注意力
        
        Args:
            hidden_dim: 隱藏層維度
            dropout: Dropout 比率
        """
        super(ScaledDotProductSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)
        
        # 線性變換層
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 輸出投影層
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [batch_size, seq_len, hidden_dim]
            mask: 注意力遮罩 [batch_size, seq_len, seq_len]
            
        Returns:
            output: 注意力輸出 [batch_size, seq_len, hidden_dim]
            attention_weights: 注意力權重 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # 線性變換
        query = self.query_projection(x)  # [batch_size, seq_len, hidden_dim]
        key = self.key_projection(x)      # [batch_size, seq_len, hidden_dim]
        value = self.value_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # 計算縮放注意力分數
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        
        # 應用遮罩
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 正規化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算加權輸出
        context = torch.matmul(attention_weights, value)  # [batch_size, seq_len, hidden_dim]
        
        # 輸出投影
        output = self.output_projection(context)
        
        # 殘差連接和層正規化
        output = self.layer_norm(output + x)
        
        return output, attention_weights


class PositionalSelfAttention(nn.Module):
    """位置敏感自注意力機制"""
    
    def __init__(self, hidden_dim: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        初始化位置敏感自注意力
        
        Args:
            hidden_dim: 隱藏層維度
            max_seq_len: 最大序列長度
            dropout: Dropout 比率
        """
        super(PositionalSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)
        
        # 位置編碼
        self.position_encoding = self._create_position_encoding(max_seq_len, hidden_dim)
        
        # 線性變換層
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 位置偏置
        self.position_bias = nn.Parameter(torch.randn(max_seq_len, max_seq_len))
        
        # 輸出投影層
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def _create_position_encoding(self, max_seq_len: int, hidden_dim: int) -> torch.Tensor:
        """創建位置編碼"""
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_seq_len, hidden_dim]
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [batch_size, seq_len, hidden_dim]
            mask: 注意力遮罩 [batch_size, seq_len, seq_len]
            
        Returns:
            output: 注意力輸出 [batch_size, seq_len, hidden_dim]
            attention_weights: 注意力權重 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # 添加位置編碼
        pos_encoding = self.position_encoding[:, :seq_len, :].to(x.device)
        x_with_pos = x + pos_encoding
        
        # 線性變換
        query = self.query_projection(x_with_pos)  # [batch_size, seq_len, hidden_dim]
        key = self.key_projection(x_with_pos)      # [batch_size, seq_len, hidden_dim]
        value = self.value_projection(x_with_pos)  # [batch_size, seq_len, hidden_dim]
        
        # 計算注意力分數
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        
        # 添加位置偏置
        position_bias = self.position_bias[:seq_len, :seq_len].unsqueeze(0)
        attention_scores = attention_scores + position_bias
        
        # 應用遮罩
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 正規化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算加權輸出
        context = torch.matmul(attention_weights, value)  # [batch_size, seq_len, hidden_dim]
        
        # 輸出投影
        output = self.output_projection(context)
        
        # 殘差連接和層正規化
        output = self.layer_norm(output + x)
        
        return output, attention_weights


class RelativePositionSelfAttention(nn.Module):
    """相對位置自注意力機制"""
    
    def __init__(self, hidden_dim: int, max_relative_position: int = 64, dropout: float = 0.1):
        """
        初始化相對位置自注意力
        
        Args:
            hidden_dim: 隱藏層維度
            max_relative_position: 最大相對位置距離
            dropout: Dropout 比率
        """
        super(RelativePositionSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_relative_position = max_relative_position
        self.scale = math.sqrt(hidden_dim)
        
        # 線性變換層
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 相對位置嵌入
        self.relative_position_k = nn.Parameter(torch.randn(2 * max_relative_position + 1, hidden_dim))
        self.relative_position_v = nn.Parameter(torch.randn(2 * max_relative_position + 1, hidden_dim))
        
        # 輸出投影層
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """獲取相對位置矩陣"""
        range_vec = torch.arange(seq_len)
        relative_matrix = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        
        # 裁剪到最大相對位置範圍
        clipped_relative_matrix = torch.clamp(
            relative_matrix, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # 轉換為正索引
        relative_matrix_clipped = clipped_relative_matrix + self.max_relative_position
        
        return relative_matrix_clipped
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [batch_size, seq_len, hidden_dim]
            mask: 注意力遮罩 [batch_size, seq_len, seq_len]
            
        Returns:
            output: 注意力輸出 [batch_size, seq_len, hidden_dim]
            attention_weights: 注意力權重 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # 線性變換
        query = self.query_projection(x)  # [batch_size, seq_len, hidden_dim]
        key = self.key_projection(x)      # [batch_size, seq_len, hidden_dim]
        value = self.value_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # 標準注意力分數
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        
        # 計算相對位置注意力分數
        relative_positions = self._get_relative_positions(seq_len).to(x.device)  # [seq_len, seq_len]
        relative_position_embeddings_k = self.relative_position_k[relative_positions]  # [seq_len, seq_len, hidden_dim]
        
        # 相對位置注意力
        relative_attention_scores = torch.einsum('bih,ijh->bij', query, relative_position_embeddings_k)  # [batch_size, seq_len, seq_len]
        
        # 合併注意力分數
        attention_scores = attention_scores + relative_attention_scores
        
        # 應用遮罩
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 正規化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 標準加權值
        context = torch.matmul(attention_weights, value)  # [batch_size, seq_len, hidden_dim]
        
        # 相對位置加權值
        relative_position_embeddings_v = self.relative_position_v[relative_positions]  # [seq_len, seq_len, hidden_dim]
        relative_context = torch.einsum('bij,ijh->bih', attention_weights, relative_position_embeddings_v)  # [batch_size, seq_len, hidden_dim]
        
        # 合併上下文
        context = context + relative_context
        
        # 輸出投影
        output = self.output_projection(context)
        
        # 殘差連接和層正規化
        output = self.layer_norm(output + x)
        
        return output, attention_weights


class LightweightSelfAttention(nn.Module):
    """輕量級自注意力機制（計算效率優化）"""
    
    def __init__(self, hidden_dim: int, reduced_dim: int = None, dropout: float = 0.1):
        """
        初始化輕量級自注意力
        
        Args:
            hidden_dim: 隱藏層維度
            reduced_dim: 降維後的維度（默認為 hidden_dim // 4）
            dropout: Dropout 比率
        """
        super(LightweightSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.reduced_dim = reduced_dim or hidden_dim // 4
        self.scale = math.sqrt(self.reduced_dim)
        
        # 降維投影層
        self.query_projection = nn.Linear(hidden_dim, self.reduced_dim)
        self.key_projection = nn.Linear(hidden_dim, self.reduced_dim)
        self.value_projection = nn.Linear(hidden_dim, self.reduced_dim)
        
        # 輸出投影層
        self.output_projection = nn.Linear(self.reduced_dim, hidden_dim)
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [batch_size, seq_len, hidden_dim]
            mask: 注意力遮罩 [batch_size, seq_len, seq_len]
            
        Returns:
            output: 注意力輸出 [batch_size, seq_len, hidden_dim]
            attention_weights: 注意力權重 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # 降維線性變換
        query = self.query_projection(x)  # [batch_size, seq_len, reduced_dim]
        key = self.key_projection(x)      # [batch_size, seq_len, reduced_dim]
        value = self.value_projection(x)  # [batch_size, seq_len, reduced_dim]
        
        # 計算注意力分數
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        
        # 應用遮罩
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 正規化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算加權輸出
        context = torch.matmul(attention_weights, value)  # [batch_size, seq_len, reduced_dim]
        
        # 輸出投影回原始維度
        output = self.output_projection(context)  # [batch_size, seq_len, hidden_dim]
        
        # 殘差連接和層正規化
        output = self.layer_norm(output + x)
        
        return output, attention_weights


class GroupedSelfAttention(nn.Module):
    """分組自注意力機制"""
    
    def __init__(self, hidden_dim: int, num_groups: int = 4, dropout: float = 0.1):
        """
        初始化分組自注意力
        
        Args:
            hidden_dim: 隱藏層維度
            num_groups: 分組數量
            dropout: Dropout 比率
        """
        super(GroupedSelfAttention, self).__init__()
        assert hidden_dim % num_groups == 0, "hidden_dim 必須能被 num_groups 整除"
        
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.group_dim = hidden_dim // num_groups
        self.scale = math.sqrt(self.group_dim)
        
        # 分組線性變換層
        self.group_projections = nn.ModuleList([
            nn.ModuleDict({
                'query': nn.Linear(self.group_dim, self.group_dim),
                'key': nn.Linear(self.group_dim, self.group_dim),
                'value': nn.Linear(self.group_dim, self.group_dim)
            }) for _ in range(num_groups)
        ])
        
        # 輸出投影層
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [batch_size, seq_len, hidden_dim]
            mask: 注意力遮罩 [batch_size, seq_len, seq_len]
            
        Returns:
            output: 注意力輸出 [batch_size, seq_len, hidden_dim]
            attention_weights: 注意力權重 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # 分組處理
        x_groups = x.view(batch_size, seq_len, self.num_groups, self.group_dim)  # [batch_size, seq_len, num_groups, group_dim]
        
        group_outputs = []
        group_attention_weights = []
        
        for i, projections in enumerate(self.group_projections):
            # 獲取當前組的輸入
            group_x = x_groups[:, :, i, :]  # [batch_size, seq_len, group_dim]
            
            # 線性變換
            query = projections['query'](group_x)  # [batch_size, seq_len, group_dim]
            key = projections['key'](group_x)      # [batch_size, seq_len, group_dim]
            value = projections['value'](group_x)  # [batch_size, seq_len, group_dim]
            
            # 計算注意力分數
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
            
            # 應用遮罩
            if mask is not None:
                attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
            # Softmax 正規化
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # 計算加權輸出
            context = torch.matmul(attention_weights, value)  # [batch_size, seq_len, group_dim]
            
            group_outputs.append(context)
            group_attention_weights.append(attention_weights)
        
        # 合併所有組的輸出
        output = torch.cat(group_outputs, dim=-1)  # [batch_size, seq_len, hidden_dim]
        
        # 平均注意力權重（用於可視化）
        avg_attention_weights = torch.stack(group_attention_weights, dim=0).mean(dim=0)  # [batch_size, seq_len, seq_len]
        
        # 輸出投影
        output = self.output_projection(output)
        
        # 殘差連接和層正規化
        output = self.layer_norm(output + x)
        
        return output, avg_attention_weights