# 多頭注意力機制模組
"""
多頭注意力機制實現

提供多種多頭注意力機制：
- 標準多頭注意力: 經典 Transformer 多頭注意力
- 可變頭數注意力: 支持動態調整頭數
- 分層多頭注意力: 不同層級的多頭處理
- 交叉多頭注意力: 跨模態多頭注意力
- 稀疏多頭注意力: 稀疏連接的多頭注意力
- 輕量級多頭注意力: 計算效率優化的多頭注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union
import math


class StandardMultiHeadAttention(nn.Module):
    """標準多頭注意力機制"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        初始化標準多頭注意力
        
        Args:
            hidden_dim: 隱藏層維度
            num_heads: 注意力頭數
            dropout: Dropout 比率
        """
        super(StandardMultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必須能被 num_heads 整除"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # 線性變換層
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 輸出投影層
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout 層
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len, hidden_dim]
            key: 鍵張量 [batch_size, seq_len, hidden_dim] (默認與 query 相同)
            value: 值張量 [batch_size, seq_len, hidden_dim] (默認與 query 相同)
            mask: 注意力遮罩 [batch_size, seq_len, seq_len]
            
        Returns:
            output: 注意力輸出 [batch_size, seq_len, hidden_dim]
            attention_weights: 注意力權重 [batch_size, num_heads, seq_len, seq_len]
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size, seq_len, _ = query.size()
        
        # 線性變換並重塑為多頭格式
        Q = self.query_projection(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_projection(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_projection(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 計算注意力分數
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        # 應用遮罩
        if mask is not None:
            # 擴展遮罩維度以匹配多頭
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 正規化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # 計算加權輸出
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重塑並合併多頭輸出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # 輸出投影
        output = self.output_projection(context)
        output = self.output_dropout(output)
        
        # 殘差連接和層正規化
        output = self.layer_norm(output + query)
        
        return output, attention_weights


class VariableHeadAttention(nn.Module):
    """可變頭數注意力機制"""
    
    def __init__(self, hidden_dim: int, min_heads: int = 1, max_heads: int = 16, dropout: float = 0.1):
        """
        初始化可變頭數注意力
        
        Args:
            hidden_dim: 隱藏層維度
            min_heads: 最小頭數
            max_heads: 最大頭數
            dropout: Dropout 比率
        """
        super(VariableHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.min_heads = min_heads
        self.max_heads = max_heads
        self.current_heads = max_heads
        
        # 為最大頭數準備投影層
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 頭數選擇器
        self.head_selector = nn.Linear(hidden_dim, max_heads)
        
        # 輸出投影層
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def set_num_heads(self, num_heads: int):
        """設置當前使用的頭數"""
        assert self.min_heads <= num_heads <= self.max_heads, f"頭數必須在 {self.min_heads} 和 {self.max_heads} 之間"
        self.current_heads = num_heads
    
    def forward(self, query: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None,
                mask: Optional[torch.Tensor] = None, adaptive_heads: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len, hidden_dim]
            key: 鍵張量 [batch_size, seq_len, hidden_dim] (默認與 query 相同)
            value: 值張量 [batch_size, seq_len, hidden_dim] (默認與 query 相同)
            mask: 注意力遮罩 [batch_size, seq_len, seq_len]
            adaptive_heads: 是否使用自適應頭數選擇
            
        Returns:
            output: 注意力輸出 [batch_size, seq_len, hidden_dim]
            attention_weights: 注意力權重 [batch_size, num_heads, seq_len, seq_len]
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size, seq_len, _ = query.size()
        
        # 自適應頭數選擇
        if adaptive_heads:
            head_scores = self.head_selector(query.mean(dim=1))  # [batch_size, max_heads]
            active_heads = torch.topk(head_scores, self.current_heads, dim=-1)[1]  # [batch_size, current_heads]
        else:
            active_heads = None
        
        # 計算每個頭的維度
        head_dim = self.hidden_dim // self.current_heads
        scale = math.sqrt(head_dim)
        
        # 線性變換
        Q = self.query_projection(query).view(batch_size, seq_len, self.max_heads, -1)
        K = self.key_projection(key).view(batch_size, -1, self.max_heads, -1)
        V = self.value_projection(value).view(batch_size, -1, self.max_heads, -1)
        
        # 選擇活躍的頭
        if active_heads is not None:
            Q = Q[:, :, active_heads[0][:self.current_heads], :head_dim]
            K = K[:, :, active_heads[0][:self.current_heads], :head_dim]
            V = V[:, :, active_heads[0][:self.current_heads], :head_dim]
        else:
            Q = Q[:, :, :self.current_heads, :head_dim]
            K = K[:, :, :self.current_heads, :head_dim]
            V = V[:, :, :self.current_heads, :head_dim]
        
        Q = Q.transpose(1, 2)  # [batch_size, current_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 計算注意力分數
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        
        # 應用遮罩
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 正規化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算加權輸出
        context = torch.matmul(attention_weights, V)  # [batch_size, current_heads, seq_len, head_dim]
        
        # 重塑輸出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.current_heads * head_dim)
        
        # 調整輸出維度
        if context.size(-1) != self.hidden_dim:
            padding = self.hidden_dim - context.size(-1)
            context = F.pad(context, (0, padding))
        
        # 輸出投影
        output = self.output_projection(context)
        
        # 殘差連接和層正規化
        output = self.layer_norm(output + query)
        
        return output, attention_weights


class HierarchicalMultiHeadAttention(nn.Module):
    """分層多頭注意力機制"""
    
    def __init__(self, hidden_dim: int, num_levels: int = 3, heads_per_level: List[int] = None, dropout: float = 0.1):
        """
        初始化分層多頭注意力
        
        Args:
            hidden_dim: 隱藏層維度
            num_levels: 層級數量
            heads_per_level: 每個層級的頭數列表
            dropout: Dropout 比率
        """
        super(HierarchicalMultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        
        if heads_per_level is None:
            heads_per_level = [8] * num_levels
        assert len(heads_per_level) == num_levels, "heads_per_level 長度必須等於 num_levels"
        
        self.heads_per_level = heads_per_level
        self.total_heads = sum(heads_per_level)
        
        # 為每個層級創建注意力模組
        self.level_attentions = nn.ModuleList([
            StandardMultiHeadAttention(hidden_dim, heads, dropout)
            for heads in heads_per_level
        ])
        
        # 層級融合模組
        self.level_fusion = nn.Linear(hidden_dim * num_levels, hidden_dim)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len, hidden_dim]
            key: 鍵張量 [batch_size, seq_len, hidden_dim] (默認與 query 相同)
            value: 值張量 [batch_size, seq_len, hidden_dim] (默認與 query 相同)
            mask: 注意力遮罩 [batch_size, seq_len, seq_len]
            
        Returns:
            output: 注意力輸出 [batch_size, seq_len, hidden_dim]
            attention_weights_list: 每個層級的注意力權重列表
        """
        if key is None:
            key = query
        if value is None:
            value = query
        
        level_outputs = []
        attention_weights_list = []
        
        current_query = query
        
        # 逐層處理
        for level_idx, level_attention in enumerate(self.level_attentions):
            level_output, level_weights = level_attention(current_query, key, value, mask)
            level_outputs.append(level_output)
            attention_weights_list.append(level_weights)
            
            # 更新查詢（殘差連接）
            current_query = level_output
        
        # 融合所有層級的輸出
        concatenated_outputs = torch.cat(level_outputs, dim=-1)  # [batch_size, seq_len, hidden_dim * num_levels]
        fused_output = self.level_fusion(concatenated_outputs)  # [batch_size, seq_len, hidden_dim]
        
        # 最終殘差連接和層正規化
        output = self.layer_norm(fused_output + query)
        
        return output, attention_weights_list


class CrossModalMultiHeadAttention(nn.Module):
    """跨模態多頭注意力機制"""
    
    def __init__(self, query_dim: int, key_dim: int, value_dim: int, hidden_dim: int, 
                 num_heads: int = 8, dropout: float = 0.1):
        """
        初始化跨模態多頭注意力
        
        Args:
            query_dim: 查詢模態維度
            key_dim: 鍵模態維度
            value_dim: 值模態維度
            hidden_dim: 隱藏層維度
            num_heads: 注意力頭數
            dropout: Dropout 比率
        """
        super(CrossModalMultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必須能被 num_heads 整除"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # 跨模態投影層
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.key_projection = nn.Linear(key_dim, hidden_dim)
        self.value_projection = nn.Linear(value_dim, hidden_dim)
        
        # 模態對齊層
        self.modal_alignment = nn.ModuleDict({
            'query': nn.Linear(hidden_dim, hidden_dim),
            'key': nn.Linear(hidden_dim, hidden_dim),
            'value': nn.Linear(hidden_dim, hidden_dim)
        })
        
        # 輸出投影層
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len_q, query_dim]
            key: 鍵張量 [batch_size, seq_len_k, key_dim]
            value: 值張量 [batch_size, seq_len_v, value_dim]
            mask: 注意力遮罩 [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            output: 注意力輸出 [batch_size, seq_len_q, hidden_dim]
            attention_weights: 注意力權重 [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)
        
        # 跨模態投影
        Q = self.query_projection(query)    # [batch_size, seq_len_q, hidden_dim]
        K = self.key_projection(key)        # [batch_size, seq_len_k, hidden_dim]
        V = self.value_projection(value)    # [batch_size, seq_len_v, hidden_dim]
        
        # 模態對齊
        Q = self.modal_alignment['query'](Q)
        K = self.modal_alignment['key'](K)
        V = self.modal_alignment['value'](V)
        
        # 重塑為多頭格式
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 計算注意力分數
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 應用遮罩
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 正規化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算加權輸出
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len_q, head_dim]
        
        # 重塑輸出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_dim)
        
        # 輸出投影
        output = self.output_projection(context)
        
        # 創建殘差連接的參考（使用投影後的查詢）
        query_projected = self.query_projection(query)
        
        # 殘差連接和層正規化
        output = self.layer_norm(output + query_projected)
        
        return output, attention_weights


class SparseMultiHeadAttention(nn.Module):
    """稀疏多頭注意力機制"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, sparsity_ratio: float = 0.1, 
                 dropout: float = 0.1):
        """
        初始化稀疏多頭注意力
        
        Args:
            hidden_dim: 隱藏層維度
            num_heads: 注意力頭數
            sparsity_ratio: 稀疏比率（保留的連接比例）
            dropout: Dropout 比率
        """
        super(SparseMultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必須能被 num_heads 整除"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.sparsity_ratio = sparsity_ratio
        
        # 線性變換層
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 稀疏化門控
        self.sparsity_gate = nn.Linear(hidden_dim * 2, 1)
        
        # 輸出投影層
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def _apply_sparsity(self, attention_scores: torch.Tensor, query: torch.Tensor, 
                       key: torch.Tensor) -> torch.Tensor:
        """應用稀疏化策略"""
        batch_size, num_heads, seq_len_q, seq_len_k = attention_scores.shape
        
        # 計算保留的連接數
        num_connections = int(seq_len_k * self.sparsity_ratio)
        
        # 基於注意力分數選擇 top-k 連接
        _, top_indices = torch.topk(attention_scores, num_connections, dim=-1)
        
        # 創建稀疏遮罩
        sparse_mask = torch.zeros_like(attention_scores)
        sparse_mask.scatter_(-1, top_indices, 1.0)
        
        # 可選：使用門控機制進一步細化
        if hasattr(self, 'use_gating') and self.use_gating:
            # 計算門控分數
            query_expanded = query.unsqueeze(-2).expand(-1, -1, -1, seq_len_k, -1)  # [B, H, seq_q, seq_k, head_dim]
            key_expanded = key.unsqueeze(-3).expand(-1, -1, seq_len_q, -1, -1)    # [B, H, seq_q, seq_k, head_dim]
            
            gate_input = torch.cat([query_expanded, key_expanded], dim=-1)  # [B, H, seq_q, seq_k, head_dim*2]
            gate_scores = torch.sigmoid(self.sparsity_gate(gate_input)).squeeze(-1)  # [B, H, seq_q, seq_k]
            
            sparse_mask = sparse_mask * gate_scores
        
        return sparse_mask
    
    def forward(self, query: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len, hidden_dim]
            key: 鍵張量 [batch_size, seq_len, hidden_dim] (默認與 query 相同)
            value: 值張量 [batch_size, seq_len, hidden_dim] (默認與 query 相同)
            mask: 注意力遮罩 [batch_size, seq_len, seq_len]
            
        Returns:
            output: 注意力輸出 [batch_size, seq_len, hidden_dim]
            attention_weights: 注意力權重 [batch_size, num_heads, seq_len, seq_len]
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size, seq_len, _ = query.size()
        
        # 線性變換並重塑為多頭格式
        Q = self.query_projection(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_projection(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_projection(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 計算注意力分數
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 應用稀疏化
        sparse_mask = self._apply_sparsity(attention_scores, Q, K)
        
        # 合併原始遮罩和稀疏遮罩
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            combined_mask = mask * sparse_mask
        else:
            combined_mask = sparse_mask
        
        # 應用遮罩
        attention_scores = attention_scores.masked_fill(combined_mask == 0, -1e9)
        
        # Softmax 正規化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算加權輸出
        context = torch.matmul(attention_weights, V)
        
        # 重塑輸出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # 輸出投影
        output = self.output_projection(context)
        
        # 殘差連接和層正規化
        output = self.layer_norm(output + query)
        
        return output, attention_weights


class LightweightMultiHeadAttention(nn.Module):
    """輕量級多頭注意力機制"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, reduction_factor: int = 4, 
                 dropout: float = 0.1):
        """
        初始化輕量級多頭注意力
        
        Args:
            hidden_dim: 隱藏層維度
            num_heads: 注意力頭數
            reduction_factor: 降維因子
            dropout: Dropout 比率
        """
        super(LightweightMultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必須能被 num_heads 整除"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.reduced_dim = hidden_dim // reduction_factor
        self.scale = math.sqrt(self.reduced_dim // num_heads)
        
        # 降維投影層
        self.input_reduction = nn.Linear(hidden_dim, self.reduced_dim)
        
        # 多頭注意力層（在降維空間）
        self.query_projection = nn.Linear(self.reduced_dim, self.reduced_dim)
        self.key_projection = nn.Linear(self.reduced_dim, self.reduced_dim)
        self.value_projection = nn.Linear(self.reduced_dim, self.reduced_dim)
        
        # 輸出投影層（恢復到原始維度）
        self.output_expansion = nn.Linear(self.reduced_dim, hidden_dim)
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len, hidden_dim]
            key: 鍵張量 [batch_size, seq_len, hidden_dim] (默認與 query 相同)
            value: 值張量 [batch_size, seq_len, hidden_dim] (默認與 query 相同)
            mask: 注意力遮罩 [batch_size, seq_len, seq_len]
            
        Returns:
            output: 注意力輸出 [batch_size, seq_len, hidden_dim]
            attention_weights: 注意力權重 [batch_size, num_heads, seq_len, seq_len]
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size, seq_len, _ = query.size()
        reduced_head_dim = self.reduced_dim // self.num_heads
        
        # 降維
        query_reduced = self.input_reduction(query)    # [batch_size, seq_len, reduced_dim]
        key_reduced = self.input_reduction(key)        # [batch_size, seq_len, reduced_dim]
        value_reduced = self.input_reduction(value)    # [batch_size, seq_len, reduced_dim]
        
        # 線性變換並重塑為多頭格式
        Q = self.query_projection(query_reduced).view(batch_size, seq_len, self.num_heads, reduced_head_dim).transpose(1, 2)
        K = self.key_projection(key_reduced).view(batch_size, -1, self.num_heads, reduced_head_dim).transpose(1, 2)
        V = self.value_projection(value_reduced).view(batch_size, -1, self.num_heads, reduced_head_dim).transpose(1, 2)
        
        # 計算注意力分數
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 應用遮罩
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 正規化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算加權輸出
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, reduced_head_dim]
        
        # 重塑輸出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.reduced_dim)
        
        # 擴展回原始維度
        output = self.output_expansion(context)  # [batch_size, seq_len, hidden_dim]
        
        # 殘差連接和層正規化
        output = self.layer_norm(output + query)
        
        return output, attention_weights