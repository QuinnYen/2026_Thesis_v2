# 注意力融合模組
"""
注意力融合機制實現

提供多種注意力融合策略：
- 加權融合: 基於權重的線性融合
- 門控融合: 使用門控機制的動態融合
- 階層融合: 分層次的注意力融合
- 自適應融合: 根據輸入動態調整融合策略
- 交叉融合: 跨注意力類型的融合
- 注意力蒸餾融合: 基於知識蒸餾的融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import math

from .similarity_attention import (
    CosineSimilarityAttention, EuclideanDistanceAttention, 
    DotProductSimilarityAttention, LearnableSimilarityAttention
)
from .keyword_guided_attention import (
    KeywordWeightedAttention, AspectAwareAttention,
    PositionSensitiveKeywordAttention, MultiLevelKeywordAttention
)
from .self_attention import (
    BasicSelfAttention, ScaledDotProductSelfAttention,
    PositionalSelfAttention, RelativePositionSelfAttention
)
from .multi_head_attention import (
    StandardMultiHeadAttention, VariableHeadAttention,
    HierarchicalMultiHeadAttention, CrossModalMultiHeadAttention
)


class WeightedAttentionFusion(nn.Module):
    """加權注意力融合器"""
    
    def __init__(self, attention_modules: List[nn.Module], hidden_dim: int, 
                 fusion_weights: Optional[List[float]] = None, learnable_weights: bool = True):
        """
        初始化加權注意力融合器
        
        Args:
            attention_modules: 注意力模組列表
            hidden_dim: 隱藏層維度
            fusion_weights: 預設融合權重
            learnable_weights: 是否學習權重
        """
        super(WeightedAttentionFusion, self).__init__()
        self.attention_modules = nn.ModuleList(attention_modules)
        self.num_attentions = len(attention_modules)
        self.hidden_dim = hidden_dim
        
        # 初始化融合權重
        if fusion_weights is None:
            fusion_weights = [1.0 / self.num_attentions] * self.num_attentions
        
        if learnable_weights:
            self.fusion_weights = nn.Parameter(torch.tensor(fusion_weights, dtype=torch.float32))
        else:
            self.register_buffer('fusion_weights', torch.tensor(fusion_weights, dtype=torch.float32))
        
        # 輸出投影層
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向傳播

        Args:
            x: 輸入張量 [batch_size, seq_len, hidden_dim]
            **kwargs: 其他參數，傳遞給各個注意力模組

        Returns:
            fused_output: 融合後的輸出 [batch_size, seq_len, hidden_dim]
            attention_weights_list: 各個注意力模組的權重列表
        """
        # 確保模組在正確的設備上
        device = x.device
        self.to(device)

        # 檢查並調整輸入維度
        original_shape = x.shape
        if len(x.shape) == 2:
            # 如果輸入是 [total_tokens, hidden_dim]，需要重塑為 [batch_size, seq_len, hidden_dim]
            x = x.unsqueeze(0)  # [1, total_tokens, hidden_dim]
            print(f"WeightedAttentionFusion調整輸入維度從 {original_shape} 到 {x.shape}")
        elif len(x.shape) != 3:
            raise ValueError(f"輸入維度錯誤，期望3維張量 [batch_size, seq_len, hidden_dim]，得到 {x.shape}")

        attention_outputs = []
        attention_weights_list = []
        
        # 計算各個注意力模組的輸出
        for i, attention_module in enumerate(self.attention_modules):
            try:
                # 確保注意力模組在正確的設備上
                attention_module = attention_module.to(device)

                # 對於 StandardMultiHeadAttention，只需要 query 參數
                if hasattr(attention_module, '__class__') and 'MultiHead' in attention_module.__class__.__name__:
                    result = attention_module(x)  # 只傳遞 query
                else:
                    result = attention_module(x, **kwargs)

                # 處理不同數量的返回值
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        output, weights = result[0], result[1]
                    else:
                        output = result[0]
                        weights = None
                else:
                    output = result
                    weights = None

                # 確保輸出在正確的設備上且維度正確
                output = output.to(device)
                if output.size(-1) != self.hidden_dim:
                    # 如果維度不匹配，創建投影層
                    if not hasattr(self, f'input_projection_{i}'):
                        setattr(self, f'input_projection_{i}',
                               nn.Linear(output.size(-1), self.hidden_dim).to(device))
                    projection = getattr(self, f'input_projection_{i}').to(device)
                    output = projection(output).to(device)

                attention_outputs.append(output)
                attention_weights_list.append(weights)

            except Exception as e:
                print(f"WeightedAttentionFusion注意力模組 {i} 錯誤: {e}")
                print(f"輸入形狀: {x.shape}, 模組類型: {type(attention_module)}")
                # 使用原始輸入作為備選
                attention_outputs.append(x)
                attention_weights_list.append(None)
        
        # 正規化融合權重
        normalized_weights = F.softmax(self.fusion_weights, dim=0)
        
        # 加權融合
        fused_output = torch.zeros_like(attention_outputs[0])
        for i, (output, weight) in enumerate(zip(attention_outputs, normalized_weights)):
            fused_output += weight * output

        # 輸出投影
        fused_output = self.output_projection(fused_output)

        # 殘差連接和層正規化
        fused_output = self.layer_norm(fused_output + x)

        # 如果原始輸入是2維的，恢復原始形狀
        if len(original_shape) == 2:
            fused_output = fused_output.squeeze(0)  # 移除batch維度
            print(f"WeightedAttentionFusion恢復輸出維度到 {fused_output.shape}")

        return fused_output, attention_weights_list


class GatedAttentionFusion(nn.Module):
    """門控注意力融合器"""
    
    def __init__(self, attention_modules: List[nn.Module], hidden_dim: int):
        """
        初始化門控注意力融合器
        
        Args:
            attention_modules: 注意力模組列表
            hidden_dim: 隱藏層維度
        """
        super(GatedAttentionFusion, self).__init__()
        self.attention_modules = nn.ModuleList(attention_modules)
        self.num_attentions = len(attention_modules)
        self.hidden_dim = hidden_dim
        
        # 門控網路
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_attentions),
            nn.Softmax(dim=-1)
        )
        
        # 上下文融合網路
        self.context_fusion = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(self.num_attentions)
        ])
        
        # 輸出投影層
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向傳播

        Args:
            x: 輸入張量 [batch_size, seq_len, hidden_dim]
            **kwargs: 其他參數，傳遞給各個注意力模組

        Returns:
            fused_output: 融合後的輸出 [batch_size, seq_len, hidden_dim]
            attention_weights_list: 各個注意力模組的權重列表
        """
        # 確保模組在正確的設備上
        device = x.device
        self.to(device)

        # 檢查並調整輸入維度
        original_shape = x.shape
        if len(x.shape) == 2:
            # 如果輸入是 [total_tokens, hidden_dim]，需要重塑為 [batch_size, seq_len, hidden_dim]
            x = x.unsqueeze(0)  # [1, total_tokens, hidden_dim]
            print(f"GatedAttentionFusion調整輸入維度從 {original_shape} 到 {x.shape}")
        elif len(x.shape) != 3:
            raise ValueError(f"輸入維度錯誤，期望3維張量 [batch_size, seq_len, hidden_dim]，得到 {x.shape}")

        batch_size, seq_len, _ = x.size()
        
        # 計算各個注意力模組的輸出
        attention_outputs = []
        attention_weights_list = []
        
        for i, attention_module in enumerate(self.attention_modules):
            try:
                # 確保注意力模組在正確的設備上
                attention_module = attention_module.to(device)

                # 對於 StandardMultiHeadAttention，只需要 query 參數
                if hasattr(attention_module, '__class__') and 'MultiHead' in attention_module.__class__.__name__:
                    result = attention_module(x)  # 只傳遞 query
                else:
                    result = attention_module(x, **kwargs)

                # 處理不同數量的返回值
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        output, weights = result[0], result[1]
                    else:
                        output = result[0]
                        weights = None
                else:
                    output = result
                    weights = None

                # 確保輸出在正確的設備上且維度正確
                output = output.to(device)
                if output.size(-1) != self.hidden_dim:
                    # 如果維度不匹配，創建投影層
                    if not hasattr(self, f'input_projection_{i}'):
                        setattr(self, f'input_projection_{i}',
                               nn.Linear(output.size(-1), self.hidden_dim).to(device))
                    projection = getattr(self, f'input_projection_{i}').to(device)
                    output = projection(output).to(device)

                # 上下文融合
                context_input = torch.cat([x, output], dim=-1)  # [batch_size, seq_len, hidden_dim * 2]
                # 確保context_fusion也在正確設備上
                self.context_fusion[i] = self.context_fusion[i].to(device)
                fused_context = self.context_fusion[i](context_input).to(device)  # [batch_size, seq_len, hidden_dim]

                attention_outputs.append(fused_context)
                attention_weights_list.append(weights)
            except Exception as e:
                print(f"GatedAttentionFusion注意力模組 {i} 錯誤: {e}")
                print(f"輸入形狀: {x.shape}, 模組類型: {type(attention_module)}")
                # 使用原始輸入作為備選
                attention_outputs.append(x)
                attention_weights_list.append(None)
        
        # 計算門控權重（基於原始輸入）
        gate_input = x.mean(dim=1)  # [batch_size, hidden_dim]
        gate_weights = self.gate_network(gate_input)  # [batch_size, num_attentions]
        
        # 門控融合
        fused_output = torch.zeros_like(attention_outputs[0])
        for i, output in enumerate(attention_outputs):
            weight = gate_weights[:, i].unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
            fused_output += weight * output
        
        # 輸出投影
        fused_output = self.output_projection(fused_output)

        # 殘差連接和層正規化
        fused_output = self.layer_norm(fused_output + x)

        # 如果原始輸入是2維的，恢復原始形狀
        if len(original_shape) == 2:
            fused_output = fused_output.squeeze(0)  # 移除batch維度
            print(f"GatedAttentionFusion恢復輸出維度到 {fused_output.shape}")

        return fused_output, attention_weights_list


class HierarchicalAttentionFusion(nn.Module):
    """階層注意力融合器"""
    
    def __init__(self, attention_modules: List[nn.Module], hidden_dim: int, 
                 hierarchy_levels: List[List[int]]):
        """
        初始化階層注意力融合器
        
        Args:
            attention_modules: 注意力模組列表
            hidden_dim: 隱藏層維度
            hierarchy_levels: 階層結構，每個子列表包含該層級的注意力模組索引
        """
        super(HierarchicalAttentionFusion, self).__init__()
        self.attention_modules = nn.ModuleList(attention_modules)
        self.hierarchy_levels = hierarchy_levels
        self.hidden_dim = hidden_dim
        
        # 為每個層級創建融合器
        self.level_fusers = nn.ModuleList([
            WeightedAttentionFusion(
                [attention_modules[idx] for idx in level_indices], 
                hidden_dim
            ) for level_indices in hierarchy_levels
        ])
        
        # 跨層級融合器
        self.cross_level_fusion = nn.Linear(hidden_dim * len(hierarchy_levels), hidden_dim)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [batch_size, seq_len, hidden_dim]
            **kwargs: 其他參數，傳遞給各個注意力模組
            
        Returns:
            fused_output: 融合後的輸出 [batch_size, seq_len, hidden_dim]
            hierarchical_weights: 階層注意力權重列表
        """
        level_outputs = []
        hierarchical_weights = []
        
        # 逐層級處理
        current_input = x
        for level_fuser in self.level_fusers:
            level_output, level_weights = level_fuser(current_input, **kwargs)
            level_outputs.append(level_output)
            hierarchical_weights.append(level_weights)
            
            # 更新下一層的輸入（殘差連接）
            current_input = level_output
        
        # 跨層級融合
        concatenated_levels = torch.cat(level_outputs, dim=-1)  # [batch_size, seq_len, hidden_dim * num_levels]
        fused_output = self.cross_level_fusion(concatenated_levels)  # [batch_size, seq_len, hidden_dim]
        
        # 最終殘差連接和層正規化
        fused_output = self.layer_norm(fused_output + x)
        
        return fused_output, hierarchical_weights


class AdaptiveAttentionFusion(nn.Module):
    """自適應注意力融合器"""
    
    def __init__(self, attention_modules: List[nn.Module], hidden_dim: int):
        """
        初始化自適應注意力融合器
        
        Args:
            attention_modules: 注意力模組列表
            hidden_dim: 隱藏層維度
        """
        super(AdaptiveAttentionFusion, self).__init__()
        self.attention_modules = nn.ModuleList(attention_modules)
        self.num_attentions = len(attention_modules)
        self.hidden_dim = hidden_dim
        
        # 自適應策略網路
        self.strategy_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3)  # 三種融合策略: 加權、門控、串聯
        )
        
        # 三種融合策略的實現
        self.weighted_fusion = WeightedAttentionFusion(attention_modules, hidden_dim)
        self.gated_fusion = GatedAttentionFusion(attention_modules, hidden_dim)
        
        # 串聯融合
        self.sequential_projection = nn.Linear(hidden_dim * self.num_attentions, hidden_dim)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 設備追蹤
        self.device = None
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向傳播

        Args:
            x: 輸入張量 [batch_size, seq_len, hidden_dim] 或 [seq_len, hidden_dim]
            **kwargs: 其他參數，傳遞給各個注意力模組

        Returns:
            fused_output: 融合後的輸出
            fusion_info: 融合資訊字典
        """
        try:
            # 記錄原始輸入形狀以便恢復
            original_shape = x.shape
            print(f"AdaptiveAttentionFusion 輸入形狀: {original_shape}")

            # 處理2D輸入 [seq_len, dim] -> [1, seq_len, dim]
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # 添加batch維度
                print(f"將2D輸入轉換為3D: {original_shape} -> {x.shape}")

            # 確保所有模組和輸入在同一設備上
            device = x.device
            print(f"目標設備: {device}")

            # 移動整個模組到正確的設備
            if self.device != device:
                self.device = device
                self.to(device)

            # 強制移動所有子模組和組件到正確的設備
            self.weighted_fusion = self.weighted_fusion.to(device)
            self.gated_fusion = self.gated_fusion.to(device)
            self.strategy_network = self.strategy_network.to(device)
            self.sequential_projection = self.sequential_projection.to(device)
            self.layer_norm = self.layer_norm.to(device)

            # 確保注意力模組也在正確的設備上
            for attention_module in self.attention_modules:
                attention_module.to(device)
            
            # 確保所有參數都在正確的設備上
            for param in self.parameters():
                param.data = param.data.to(device)

            # 決定融合策略
            batch_size, seq_len, feature_dim = x.size()

            # 確保輸入維度正確
            if feature_dim != self.hidden_dim:
                # 如果輸入維度不匹配，使用投影層調整
                if not hasattr(self, 'input_projection'):
                    self.input_projection = nn.Linear(feature_dim, self.hidden_dim).to(device)
                else:
                    self.input_projection = self.input_projection.to(device)
                x = self.input_projection(x).to(device)  # 確保投影後的張量在正確設備上

            # 確保所有計算都在正確的設備上
            strategy_input = x.mean(dim=1).to(device)  # [batch_size, hidden_dim]
            
            # 確保strategy_network在正確設備上
            self.strategy_network = self.strategy_network.to(device)
            strategy_logits = self.strategy_network(strategy_input)  # [batch_size, 3]
            strategy_weights = F.softmax(strategy_logits, dim=-1)  # [batch_size, 3]
            
            # 確保所有張量在正確設備上
            strategy_logits = strategy_logits.to(device)
            strategy_weights = strategy_weights.to(device)

            # 計算各種融合策略的輸出 - 穩健地處理返回值
            try:
                # 確保子融合器在正確設備上
                self.weighted_fusion = self.weighted_fusion.to(device)
                # 確保輸入張量在正確設備上再傳遞給子融合器
                x_device_checked = x.to(device)
                weighted_result = self.weighted_fusion(x_device_checked, **kwargs)
                if isinstance(weighted_result, tuple) and len(weighted_result) >= 2:
                    weighted_output, weighted_weights = weighted_result[0].to(device), weighted_result[1]
                elif isinstance(weighted_result, tuple):
                    weighted_output, weighted_weights = weighted_result[0].to(device), None
                else:
                    weighted_output, weighted_weights = weighted_result.to(device), None
            except Exception as e:
                print(f"Weighted融合錯誤: {e}")
                weighted_output, weighted_weights = x.to(device), None

            try:
                # 確保子融合器在正確設備上
                self.gated_fusion = self.gated_fusion.to(device)
                x_device_checked = x.to(device)
                gated_result = self.gated_fusion(x_device_checked, **kwargs)
                if isinstance(gated_result, tuple) and len(gated_result) >= 2:
                    gated_output, gated_weights = gated_result[0].to(device), gated_result[1]
                elif isinstance(gated_result, tuple):
                    gated_output, gated_weights = gated_result[0].to(device), None
                else:
                    gated_output, gated_weights = gated_result.to(device), None
            except Exception as e:
                print(f"Gated融合錯誤: {e}")
                gated_output, gated_weights = x.to(device), None

            # 確保輸出在正確的設備上
            weighted_output = weighted_output.to(device)
            gated_output = gated_output.to(device)

            # 串聯融合 - 穩健地處理注意力模組
            attention_outputs = []
            for i, attention_module in enumerate(self.attention_modules):
                try:
                    # 確保注意力模組在正確的設備上
                    attention_module = attention_module.to(device)

                    # 根據模組類型調用
                    if hasattr(attention_module, '__class__') and 'MultiHead' in attention_module.__class__.__name__:
                        result = attention_module(x)  # 只傳遞 query
                    else:
                        result = attention_module(x, **kwargs)

                    # 處理可變返回值
                    if isinstance(result, tuple):
                        if len(result) >= 1:
                            output = result[0]
                        else:
                            output = x  # 備用方案
                    else:
                        output = result

                    # 確保輸出維度匹配
                    output = output.to(device)
                    if output.size(-1) != self.hidden_dim:
                        if not hasattr(self, f'attention_projection_{i}'):
                            setattr(self, f'attention_projection_{i}',
                                   nn.Linear(output.size(-1), self.hidden_dim).to(device))
                        projection = getattr(self, f'attention_projection_{i}').to(device)
                        output = projection(output).to(device)

                    attention_outputs.append(output)

                except Exception as e:
                    print(f"注意力模組 {i} 錯誤: {e}")
                    # 使用原始輸入作為備用方案
                    x_fallback = x.to(device)
                    if x_fallback.size(-1) == self.hidden_dim:
                        attention_outputs.append(x_fallback)
                    else:
                        # 創建投影以匹配維度
                        if not hasattr(self, f'fallback_projection_{i}'):
                            setattr(self, f'fallback_projection_{i}',
                                   nn.Linear(x_fallback.size(-1), self.hidden_dim).to(device))
                        projection = getattr(self, f'fallback_projection_{i}').to(device)
                        attention_outputs.append(projection(x_fallback).to(device))

            # 如果沒有有效的注意力輸出，使用原始輸入
            if not attention_outputs:
                x_default = x.to(device)
                if x_default.size(-1) == self.hidden_dim:
                    attention_outputs = [x_default]
                else:
                    if not hasattr(self, 'default_projection'):
                        self.default_projection = nn.Linear(x_default.size(-1), self.hidden_dim).to(device)
                    else:
                        self.default_projection = self.default_projection.to(device)
                    attention_outputs = [self.default_projection(x_default).to(device)]

            # 確保所有attention_outputs都在正確設備上
            attention_outputs = [output.to(device) for output in attention_outputs]
            sequential_concat = torch.cat(attention_outputs, dim=-1).to(device)  # [batch_size, seq_len, hidden_dim * num_attentions]
            # 確保sequential_projection在正確設備上
            self.sequential_projection = self.sequential_projection.to(device)
            sequential_output = self.sequential_projection(sequential_concat).to(device)  # [batch_size, seq_len, hidden_dim]

            # 自適應融合
            # 正確處理 strategy_weights 的維度以匹配 outputs_stack
            strategy_weights = strategy_weights.unsqueeze(1).unsqueeze(2).to(device)  # [batch_size, 1, 1, 3]

            # 確保所有輸出都在正確設備上
            weighted_output = weighted_output.to(device)
            gated_output = gated_output.to(device)
            sequential_output = sequential_output.to(device)

            outputs_stack = torch.stack([weighted_output, gated_output, sequential_output], dim=-1).to(device)  # [batch_size, seq_len, hidden_dim, 3]
            adaptive_output = torch.sum(outputs_stack * strategy_weights, dim=-1).to(device)  # [batch_size, seq_len, hidden_dim]

            # 最終層正規化
            self.layer_norm = self.layer_norm.to(device)
            final_output = self.layer_norm(adaptive_output).to(device)

            # 如果原始輸入是2維的，恢復原始形狀
            if len(original_shape) == 2:
                final_output = final_output.squeeze(0)  # 移除batch維度
                print(f"恢復輸出維度從 {final_output.unsqueeze(0).shape} 到 {final_output.shape}")

            fusion_info = {
                'strategy_weights': strategy_weights.squeeze(),
                'weighted_weights': weighted_weights,
                'gated_weights': gated_weights
            }

            return final_output, fusion_info

        except Exception as e:
            print(f"AdaptiveAttentionFusion 前向傳播錯誤: {e}")
            # 返回原始輸入作為備用方案
            if len(original_shape) == 2:
                return x.squeeze(0), {}
            else:
                return x, {}


class CrossAttentionFusion(nn.Module):
    """跨注意力融合器"""
    
    def __init__(self, attention_modules: List[nn.Module], hidden_dim: int):
        """
        初始化跨注意力融合器
        
        Args:
            attention_modules: 注意力模組列表
            hidden_dim: 隱藏層維度
        """
        super(CrossAttentionFusion, self).__init__()
        self.attention_modules = nn.ModuleList(attention_modules)
        self.num_attentions = len(attention_modules)
        self.hidden_dim = hidden_dim
        
        # 跨注意力交互層
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            for _ in range(self.num_attentions)
        ])
        
        # 特徵融合層
        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(self.num_attentions)
        ])
        
        # 最終融合層
        self.final_fusion = nn.Linear(hidden_dim * self.num_attentions, hidden_dim)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向傳播

        Args:
            x: 輸入張量 [batch_size, seq_len, hidden_dim] 或 [seq_len, hidden_dim]
            **kwargs: 其他參數，傳遞給各個注意力模組

        Returns:
            fused_output: 融合後的輸出
            attention_weights_list: 各個注意力模組的權重列表
        """
        try:
            # 記錄原始輸入形狀以便恢復
            original_shape = x.shape
            print(f"CrossAttentionFusion 輸入形狀: {original_shape}")

            # 處理2D輸入 [seq_len, dim] -> [1, seq_len, dim]
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # 添加batch維度
                print(f"將2D輸入轉換為3D: {original_shape} -> {x.shape}")

            # 確保模組在正確的設備上
            device = x.device
            print(f"目標設備: {device}")
            self.to(device)
            
            # 強制確保所有子模組都在正確的設備上
            for name, module in self.named_modules():
                if module != self:  # 避免遞歸移動自己
                    module.to(device)
            
            # 確保所有參數都在正確的設備上
            for param in self.parameters():
                param.data = param.data.to(device)

            # 確保輸入維度正確
            batch_size, seq_len, feature_dim = x.size()
            if feature_dim != self.hidden_dim:
                # 如果輸入維度不匹配，使用投影層調整
                if not hasattr(self, 'input_projection'):
                    self.input_projection = nn.Linear(feature_dim, self.hidden_dim).to(device)
                else:
                    self.input_projection = self.input_projection.to(device)
                x = self.input_projection(x).to(device)

            # 計算各個注意力模組的輸出
            attention_outputs = []
            attention_weights_list = []

            for i, attention_module in enumerate(self.attention_modules):
                try:
                    # 確保注意力模組在正確的設備上
                    attention_module = attention_module.to(device)

                    # 根據模組類型調用
                    if hasattr(attention_module, '__class__') and 'MultiHead' in attention_module.__class__.__name__:
                        result = attention_module(x)  # 只傳遞 query
                    else:
                        result = attention_module(x, **kwargs)

                    # 穩健地處理返回值
                    if isinstance(result, tuple):
                        if len(result) >= 2:
                            output, weights = result[0], result[1]
                        elif len(result) == 1:
                            output, weights = result[0], None
                        else:
                            output, weights = x, None  # 備用方案
                    else:
                        output = result
                        weights = None

                    # 確保輸出在正確的設備上且維度匹配
                    output = output.to(device)
                    if output.size(-1) != self.hidden_dim:
                        if not hasattr(self, f'output_projection_{i}'):
                            setattr(self, f'output_projection_{i}',
                                   nn.Linear(output.size(-1), self.hidden_dim).to(device))
                        projection = getattr(self, f'output_projection_{i}').to(device)
                        output = projection(output).to(device)

                    # 確保輸出序列長度匹配
                    if output.size(1) != seq_len:
                        if output.size(1) > seq_len:
                            output = output[:, :seq_len, :]  # 截斷
                        else:
                            # 填充到匹配長度
                            padding_length = seq_len - output.size(1)
                            padding = torch.zeros(batch_size, padding_length, self.hidden_dim, device=device)
                            output = torch.cat([output, padding], dim=1)

                    attention_outputs.append(output)
                    attention_weights_list.append(weights)

                except Exception as e:
                    print(f"CrossAttentionFusion注意力模組 {i} 錯誤: {e}")
                    print(f"注意力模組類型: {type(attention_module)}, 輸入設備: {x.device}, 目標設備: {device}")
                    # 使用原始輸入作為備選，確保設備一致性
                    fallback_output = x.to(device)
                    if fallback_output.size(-1) != self.hidden_dim:
                        if not hasattr(self, f'fallback_projection_{i}'):
                            setattr(self, f'fallback_projection_{i}',
                                   nn.Linear(fallback_output.size(-1), self.hidden_dim).to(device))
                        projection = getattr(self, f'fallback_projection_{i}').to(device)
                        fallback_output = projection(fallback_output).to(device)
                    # 強制確保輸出在正確設備上
                    fallback_output = fallback_output.to(device)
                    attention_outputs.append(fallback_output)
                    attention_weights_list.append(None)

            # 如果沒有有效的注意力輸出，使用原始輸入
            if not attention_outputs:
                attention_outputs = [x.to(device)]
                attention_weights_list = [None]

            # 確保所有cross_attention_layers在正確設備上
            for cross_attn in self.cross_attention_layers:
                cross_attn.to(device)

            # 跨注意力交互 - 穩健地處理維度不匹配
            cross_enhanced_outputs = []
            for i, (output, cross_attn) in enumerate(zip(attention_outputs, self.cross_attention_layers)):
                try:
                    # 確保output在正確設備上
                    output = output.to(device)

                    # 計算與其他注意力輸出的交互
                    other_outputs = [attention_outputs[j].to(device) for j in range(len(attention_outputs)) if j != i]

                    if other_outputs and len(other_outputs) > 0:
                        # 確保所有輸出有相同的序列長度
                        target_seq_len = output.size(1)
                        aligned_other_outputs = []

                        for other_output in other_outputs:
                            if other_output.size(1) != target_seq_len:
                                if other_output.size(1) > target_seq_len:
                                    other_output = other_output[:, :target_seq_len, :]
                                else:
                                    padding_length = target_seq_len - other_output.size(1)
                                    padding = torch.zeros(batch_size, padding_length, self.hidden_dim, device=device)
                                    other_output = torch.cat([other_output, padding], dim=1)
                            aligned_other_outputs.append(other_output)

                        # 在特徵維度拼接，而不是序列維度
                        other_concat = torch.cat(aligned_other_outputs, dim=-1)  # [batch_size, seq_len, hidden_dim * (num_attentions-1)]

                        # 如果拼接後的維度不匹配hidden_dim，需要投影
                        if other_concat.size(-1) != self.hidden_dim:
                            if not hasattr(self, f'cross_projection_{i}'):
                                setattr(self, f'cross_projection_{i}',
                                       nn.Linear(other_concat.size(-1), self.hidden_dim).to(device))
                            projection = getattr(self, f'cross_projection_{i}').to(device)
                            other_concat = projection(other_concat).to(device)

                        # 跨注意力計算
                        cross_attn = cross_attn.to(device)
                        cross_output, _ = cross_attn(output, other_concat, other_concat)
                        cross_output = cross_output.to(device)

                        # 確保cross_output維度匹配
                        if cross_output.size(-1) != self.hidden_dim:
                            if not hasattr(self, f'cross_output_projection_{i}'):
                                setattr(self, f'cross_output_projection_{i}',
                                       nn.Linear(cross_output.size(-1), self.hidden_dim).to(device))
                            projection = getattr(self, f'cross_output_projection_{i}').to(device)
                            cross_output = projection(cross_output).to(device)

                        # 特徵融合
                        fusion_input = torch.cat([output, cross_output], dim=-1).to(device)  # [batch_size, seq_len, hidden_dim * 2]
                        fusion_layer = self.feature_fusion[i].to(device)
                        enhanced_output = fusion_layer(fusion_input).to(device)
                    else:
                        enhanced_output = output.to(device)

                    cross_enhanced_outputs.append(enhanced_output)

                except Exception as e:
                    print(f"CrossAttentionFusion 跨注意力交互 {i} 錯誤: {e}")
                    print(f"當前設備: {device}, 輸出設備: {output.device if hasattr(output, 'device') else 'N/A'}")
                    # 使用原始輸出作為備用方案，確保設備一致性
                    safe_output = output.to(device) if hasattr(output, 'to') else x.to(device)
                    cross_enhanced_outputs.append(safe_output)

            # 確保所有輸出在正確設備上
            cross_enhanced_outputs = [output.to(device) for output in cross_enhanced_outputs]

            # 最終融合
            final_concat = torch.cat(cross_enhanced_outputs, dim=-1).to(device)  # [batch_size, seq_len, hidden_dim * num_attentions]
            final_fusion_layer = self.final_fusion.to(device)
            fused_output = final_fusion_layer(final_concat).to(device)  # [batch_size, seq_len, hidden_dim]

            # 殘差連接和層正規化
            layer_norm = self.layer_norm.to(device)
            fused_output = layer_norm(fused_output + x).to(device)

            # 如果原始輸入是2維的，恢復原始形狀
            if len(original_shape) == 2:
                fused_output = fused_output.squeeze(0)  # 移除batch維度
                print(f"恢復輸出維度從 {fused_output.unsqueeze(0).shape} 到 {fused_output.shape}")

            return fused_output, attention_weights_list

        except Exception as e:
            print(f"CrossAttentionFusion 前向傳播錯誤: {e}")
            # 返回原始輸入作為備用方案
            if len(original_shape) == 2:
                return x.squeeze(0), []
            else:
                return x, []


class AttentionDistillationFusion(nn.Module):
    """注意力蒸餾融合器"""
    
    def __init__(self, teacher_attention: nn.Module, student_attentions: List[nn.Module], 
                 hidden_dim: int, temperature: float = 3.0, alpha: float = 0.7):
        """
        初始化注意力蒸餾融合器
        
        Args:
            teacher_attention: 教師注意力模組
            student_attentions: 學生注意力模組列表
            hidden_dim: 隱藏層維度
            temperature: 蒸餾溫度
            alpha: 蒸餾權重
        """
        super(AttentionDistillationFusion, self).__init__()
        self.teacher_attention = teacher_attention
        self.student_attentions = nn.ModuleList(student_attentions)
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.alpha = alpha
        
        # 知識融合層
        self.knowledge_fusion = nn.Sequential(
            nn.Linear(hidden_dim * (1 + len(student_attentions)), hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 注意力對齊層
        self.attention_alignment = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in student_attentions
        ])
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def compute_distillation_loss(self, teacher_weights: torch.Tensor, 
                                 student_weights_list: List[torch.Tensor]) -> torch.Tensor:
        """計算蒸餾損失"""
        distillation_loss = 0.0
        
        # 軟化教師權重
        teacher_soft = F.softmax(teacher_weights / self.temperature, dim=-1)
        
        for student_weights in student_weights_list:
            # 軟化學生權重
            student_soft = F.softmax(student_weights / self.temperature, dim=-1)
            
            # KL 散度損失
            kl_loss = F.kl_div(
                F.log_softmax(student_weights / self.temperature, dim=-1),
                teacher_soft,
                reduction='batchmean'
            )
            distillation_loss += kl_loss
        
        return distillation_loss / len(student_weights_list)
    
    def forward(self, x: torch.Tensor, compute_loss: bool = False, **kwargs) -> Union[Tuple[torch.Tensor, List[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]]:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [batch_size, seq_len, hidden_dim]
            compute_loss: 是否計算蒸餾損失
            **kwargs: 其他參數，傳遞給各個注意力模組
            
        Returns:
            fused_output: 融合後的輸出 [batch_size, seq_len, hidden_dim]
            attention_weights_list: 各個注意力模組的權重列表
            distillation_loss: 蒸餾損失（如果 compute_loss=True）
        """
        # 確保模組在正確的設備上
        device = x.device
        self.to(device)

        # 教師模型輸出
        try:
            result = self.teacher_attention(x, **kwargs)
            # 處理可變返回值
            if isinstance(result, tuple):
                if len(result) >= 2:
                    teacher_output, teacher_weights = result[0], result[1]
                else:
                    teacher_output = result[0]
                    teacher_weights = None
            else:
                teacher_output = result
                teacher_weights = None
        except:
            if hasattr(self.teacher_attention, '__class__'):
                module_name = self.teacher_attention.__class__.__name__
                if 'CrossModal' in module_name:
                    result = self.teacher_attention(x, x, x, **kwargs)
                else:
                    result = self.teacher_attention(x, **kwargs)
            else:
                result = self.teacher_attention(x, **kwargs)

            # 處理異常情況下的可變返回值
            if isinstance(result, tuple):
                if len(result) >= 2:
                    teacher_output, teacher_weights = result[0], result[1]
                else:
                    teacher_output = result[0]
                    teacher_weights = None
            else:
                teacher_output = result
                teacher_weights = None
        
        # 學生模型輸出
        student_outputs = []
        student_weights_list = []
        
        for i, student_attention in enumerate(self.student_attentions):
            try:
                result = student_attention(x, **kwargs)
                # 處理可變返回值
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        output, weights = result[0], result[1]
                    else:
                        output = result[0]
                        weights = None
                else:
                    output = result
                    weights = None
            except:
                if hasattr(student_attention, '__class__'):
                    module_name = student_attention.__class__.__name__
                    if 'CrossModal' in module_name:
                        result = student_attention(x, x, x, **kwargs)
                    else:
                        result = student_attention(x, **kwargs)
                else:
                    result = student_attention(x, **kwargs)

                # 處理異常情況下的可變返回值
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        output, weights = result[0], result[1]
                    else:
                        output = result[0]
                        weights = None
                else:
                    output = result
                    weights = None
            
            # 注意力對齊
            aligned_output = self.attention_alignment[i](output)
            student_outputs.append(aligned_output)
            student_weights_list.append(weights)
        
        # 知識融合
        all_outputs = [teacher_output] + student_outputs
        fusion_input = torch.cat(all_outputs, dim=-1)  # [batch_size, seq_len, hidden_dim * (1 + num_students)]
        
        # 融合輸出
        fused_output = self.knowledge_fusion(fusion_input)  # [batch_size, seq_len, hidden_dim]
        
        # 加權融合（教師權重更高）
        teacher_weight = self.alpha
        student_weight = (1 - self.alpha) / len(student_outputs)
        
        final_output = teacher_weight * teacher_output
        for student_output in student_outputs:
            final_output += student_weight * student_output
        
        # 結合融合輸出和加權輸出
        combined_output = 0.5 * fused_output + 0.5 * final_output
        
        # 層正規化
        combined_output = self.layer_norm(combined_output)
        
        all_weights = [teacher_weights] + student_weights_list
        
        if compute_loss:
            distillation_loss = self.compute_distillation_loss(teacher_weights, student_weights_list)
            return combined_output, all_weights, distillation_loss
        else:
            return combined_output, all_weights


class UniversalAttentionFusion(nn.Module):
    """通用注意力融合器"""
    
    def __init__(self, attention_modules: List[nn.Module], hidden_dim: int, 
                 fusion_type: str = 'adaptive'):
        """
        初始化通用注意力融合器
        
        Args:
            attention_modules: 注意力模組列表
            hidden_dim: 隱藏層維度
            fusion_type: 融合類型 ('weighted', 'gated', 'hierarchical', 'adaptive', 'cross', 'distillation')
        """
        super(UniversalAttentionFusion, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'weighted':
            self.fusion_module = WeightedAttentionFusion(attention_modules, hidden_dim)
        elif fusion_type == 'gated':
            self.fusion_module = GatedAttentionFusion(attention_modules, hidden_dim)
        elif fusion_type == 'hierarchical':
            # 預設階層結構
            hierarchy_levels = []
            for i in range(len(attention_modules)):
                hierarchy_levels.append([i])
            self.fusion_module = HierarchicalAttentionFusion(attention_modules, hidden_dim, hierarchy_levels)
        elif fusion_type == 'adaptive':
            self.fusion_module = AdaptiveAttentionFusion(attention_modules, hidden_dim)
        elif fusion_type == 'cross':
            self.fusion_module = CrossAttentionFusion(attention_modules, hidden_dim)
        elif fusion_type == 'distillation':
            # 預設第一個為教師模型，其餘為學生模型
            teacher = attention_modules[0]
            students = attention_modules[1:] if len(attention_modules) > 1 else []
            self.fusion_module = AttentionDistillationFusion(teacher, students, hidden_dim)
        else:
            raise ValueError(f"不支援的融合類型: {fusion_type}")
    
    def forward(self, x: torch.Tensor, **kwargs):
        """前向傳播"""
        return self.fusion_module(x, **kwargs)