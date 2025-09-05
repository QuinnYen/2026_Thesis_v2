# 注意力層組合器
"""
注意力機制組合器模組

整合多種注意力機制，包括：
- 自注意力機制
- 多頭注意力機制
- 跨注意力機制
- 注意力融合策略
- 動態權重調整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np


class MultiAttentionCombiner(nn.Module):
    """
    多注意力機制組合器
    
    組合多種注意力機制，提供靈活的注意力計算和融合
    """
    
    def __init__(self,
                 input_dim: int,
                 attention_types: List[str] = ['self', 'cross', 'similarity'],
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 fusion_strategy: str = 'weighted_sum'):
        """
        初始化多注意力組合器
        
        Args:
            input_dim: 輸入特徵維度
            attention_types: 注意力類型列表
            num_heads: 注意力頭數
            dropout_rate: Dropout比率
            fusion_strategy: 融合策略 ('weighted_sum', 'concat', 'gated')
        """
        super(MultiAttentionCombiner, self).__init__()
        
        self.input_dim = input_dim
        self.attention_types = attention_types
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.fusion_strategy = fusion_strategy
        
        # 初始化各種注意力機制
        self.attention_modules = nn.ModuleDict()
        
        if 'self' in attention_types:
            self.attention_modules['self'] = SelfAttentionModule(
                input_dim, num_heads, dropout_rate
            )
        
        if 'cross' in attention_types:
            self.attention_modules['cross'] = CrossAttentionModule(
                input_dim, num_heads, dropout_rate
            )
        
        if 'similarity' in attention_types:
            self.attention_modules['similarity'] = SimilarityAttentionModule(
                input_dim, dropout_rate
            )
        
        if 'keyword' in attention_types:
            self.attention_modules['keyword'] = KeywordAttentionModule(
                input_dim, dropout_rate
            )
        
        # 注意力權重學習器
        self.attention_weights = nn.Parameter(torch.ones(len(attention_types)))
        
        # 融合層
        if fusion_strategy == 'weighted_sum':
            self.fusion_layer = nn.Identity()
            self.output_dim = input_dim
        elif fusion_strategy == 'concat':
            self.fusion_layer = nn.Linear(input_dim * len(attention_types), input_dim)
            self.output_dim = input_dim
        elif fusion_strategy == 'gated':
            self.fusion_layer = GatedFusion(input_dim, len(attention_types))
            self.output_dim = input_dim
        
        # 正規化和Dropout
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 位置編碼（如果需要）
        self.positional_encoding = PositionalEncoding(input_dim, dropout_rate)
    
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                keyword_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            query: 查詢張量 [batch_size, seq_len, input_dim]
            key: 鍵張量 [batch_size, key_len, input_dim]
            value: 值張量 [batch_size, value_len, input_dim]
            attention_mask: 注意力遮罩 [batch_size, seq_len, key_len]
            keyword_mask: 關鍵詞遮罩 [batch_size, seq_len]
        
        Returns:
            注意力輸出和權重
        """
        batch_size, seq_len, _ = query.size()
        
        # 如果key和value為None，則使用query
        if key is None:
            key = query
        if value is None:
            value = key
        
        # 添加位置編碼
        query_with_pos = self.positional_encoding(query)
        
        # 計算各種注意力
        attention_outputs = {}
        attention_weights = {}
        
        for i, attention_type in enumerate(self.attention_types):
            if attention_type in self.attention_modules:
                if attention_type == 'self':
                    output = self.attention_modules[attention_type](
                        query_with_pos, query_with_pos, query_with_pos, attention_mask
                    )
                elif attention_type == 'cross':
                    output = self.attention_modules[attention_type](
                        query_with_pos, key, value, attention_mask
                    )
                elif attention_type == 'similarity':
                    output = self.attention_modules[attention_type](
                        query_with_pos, key, attention_mask
                    )
                elif attention_type == 'keyword':
                    output = self.attention_modules[attention_type](
                        query_with_pos, keyword_mask
                    )
                
                attention_outputs[attention_type] = output['attended_values']
                attention_weights[attention_type] = output['attention_weights']
        
        # 融合注意力輸出
        fused_output = self._fuse_attentions(attention_outputs)
        
        # 殘差連接和層正規化
        output = self.layer_norm(query + self.dropout(fused_output))
        
        return {
            'attended_values': output,
            'attention_weights': attention_weights,
            'fusion_weights': F.softmax(self.attention_weights, dim=0)
        }
    
    def _fuse_attentions(self, attention_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        融合多個注意力輸出
        
        Args:
            attention_outputs: 注意力輸出字典
        
        Returns:
            融合後的輸出
        """
        outputs = list(attention_outputs.values())
        
        if self.fusion_strategy == 'weighted_sum':
            # 加權求和
            weights = F.softmax(self.attention_weights, dim=0)
            fused = sum(w * output for w, output in zip(weights, outputs))
        
        elif self.fusion_strategy == 'concat':
            # 連接後投影
            concatenated = torch.cat(outputs, dim=-1)
            fused = self.fusion_layer(concatenated)
        
        elif self.fusion_strategy == 'gated':
            # 門控融合
            fused = self.fusion_layer(outputs)
        
        return fused


class SelfAttentionModule(nn.Module):
    """
    自注意力機制模組
    """
    
    def __init__(self, input_dim: int, num_heads: int, dropout_rate: float):
        super(SelfAttentionModule, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, query, key, value, attention_mask=None):
        attended_values, attention_weights = self.attention(
            query, key, value, 
            key_padding_mask=attention_mask if attention_mask is not None else None
        )
        
        return {
            'attended_values': attended_values,
            'attention_weights': attention_weights
        }


class CrossAttentionModule(nn.Module):
    """
    跨注意力機制模組
    """
    
    def __init__(self, input_dim: int, num_heads: int, dropout_rate: float):
        super(CrossAttentionModule, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, query, key, value, attention_mask=None):
        attended_values, attention_weights = self.attention(
            query, key, value,
            key_padding_mask=attention_mask if attention_mask is not None else None
        )
        
        return {
            'attended_values': attended_values,
            'attention_weights': attention_weights
        }


class SimilarityAttentionModule(nn.Module):
    """
    相似度注意力機制模組
    """
    
    def __init__(self, input_dim: int, dropout_rate: float):
        super(SimilarityAttentionModule, self).__init__()
        
        self.input_dim = input_dim
        self.scale = math.sqrt(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 相似度計算投影層
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, query, key, attention_mask=None):
        batch_size, seq_len, _ = query.size()
        _, key_len, _ = key.size()
        
        # 投影
        Q = self.query_proj(query)  # [batch_size, seq_len, input_dim]
        K = self.key_proj(key)      # [batch_size, key_len, input_dim]
        V = self.value_proj(key)    # [batch_size, key_len, input_dim]
        
        # 計算相似度得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 應用遮罩
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # 計算注意力權重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算注意力輸出
        attended_values = torch.matmul(attention_weights, V)
        
        return {
            'attended_values': attended_values,
            'attention_weights': attention_weights
        }


class KeywordAttentionModule(nn.Module):
    """
    關鍵詞導向注意力機制模組
    """
    
    def __init__(self, input_dim: int, dropout_rate: float):
        super(KeywordAttentionModule, self).__init__()
        
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout_rate)
        
        # 關鍵詞權重計算層
        self.keyword_scorer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
        
        # 上下文融合層
        self.context_fusion = nn.Linear(input_dim, input_dim)
        
    def forward(self, query, keyword_mask=None):
        batch_size, seq_len, _ = query.size()
        
        # 計算關鍵詞得分
        keyword_scores = self.keyword_scorer(query).squeeze(-1)  # [batch_size, seq_len]
        
        # 應用關鍵詞遮罩
        if keyword_mask is not None:
            keyword_scores = keyword_scores * keyword_mask.float()
            keyword_scores = keyword_scores.masked_fill(keyword_mask == 0, -1e9)
        
        # 計算注意力權重
        attention_weights = F.softmax(keyword_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 計算加權表示
        attended_values = torch.sum(
            query * attention_weights.unsqueeze(-1), dim=1, keepdim=True
        ).expand(-1, seq_len, -1)
        
        # 融合原始查詢和關鍵詞注意力
        attended_values = self.context_fusion(query + attended_values)
        
        return {
            'attended_values': attended_values,
            'attention_weights': attention_weights
        }


class GatedFusion(nn.Module):
    """
    門控融合模組
    """
    
    def __init__(self, input_dim: int, num_inputs: int):
        super(GatedFusion, self).__init__()
        
        self.input_dim = input_dim
        self.num_inputs = num_inputs
        
        # 門控網路
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim * num_inputs, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_inputs),
            nn.Sigmoid()
        )
        
        # 輸出投影
        self.output_projection = nn.Linear(input_dim, input_dim)
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        門控融合前向傳播
        
        Args:
            inputs: 輸入張量列表
        
        Returns:
            融合後的張量
        """
        # 連接所有輸入
        concatenated = torch.cat(inputs, dim=-1)
        
        # 計算門控權重
        gates = self.gate_network(concatenated)  # [batch_size, seq_len, num_inputs]
        
        # 加權融合
        weighted_sum = sum(
            gate.unsqueeze(-1) * inp 
            for gate, inp in zip(gates.unbind(-1), inputs)
        )
        
        # 輸出投影
        output = self.output_projection(weighted_sum)
        
        return output


class PositionalEncoding(nn.Module):
    """
    位置編碼模組
    """
    
    def __init__(self, d_model: int, dropout_rate: float, max_length: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 建立位置編碼
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置編碼
        
        Args:
            x: 輸入張量 [batch_size, seq_len, d_model]
        
        Returns:
            添加位置編碼後的張量
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)


class AdaptiveAttentionCombiner(nn.Module):
    """
    自適應注意力組合器
    
    根據輸入動態調整注意力機制的權重
    """
    
    def __init__(self,
                 input_dim: int,
                 attention_types: List[str],
                 adaptation_strategy: str = 'learned'):
        """
        初始化自適應注意力組合器
        
        Args:
            input_dim: 輸入維度
            attention_types: 注意力類型列表
            adaptation_strategy: 適應策略 ('learned', 'entropy', 'confidence')
        """
        super(AdaptiveAttentionCombiner, self).__init__()
        
        self.input_dim = input_dim
        self.attention_types = attention_types
        self.adaptation_strategy = adaptation_strategy
        
        # 基礎注意力組合器
        self.base_combiner = MultiAttentionCombiner(
            input_dim, attention_types, fusion_strategy='weighted_sum'
        )
        
        # 適應性權重網路
        if adaptation_strategy == 'learned':
            self.adaptation_network = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, len(attention_types)),
                nn.Softmax(dim=-1)
            )
        
    def forward(self, *args, **kwargs):
        """
        自適應前向傳播
        """
        # 獲取基礎注意力輸出
        base_output = self.base_combiner(*args, **kwargs)
        
        if self.adaptation_strategy == 'learned':
            # 基於輸入學習適應權重
            query = args[0] if args else kwargs.get('query')
            pooled_input = torch.mean(query, dim=1)  # [batch_size, input_dim]
            
            # 計算適應性權重
            adaptive_weights = self.adaptation_network(pooled_input)
            
            # 更新融合權重
            base_output['fusion_weights'] = adaptive_weights.mean(0)
        
        return base_output