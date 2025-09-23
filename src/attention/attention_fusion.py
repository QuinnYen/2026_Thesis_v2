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

        attention_outputs = []
        attention_weights_list = []
        
        # 計算各個注意力模組的輸出
        for attention_module in self.attention_modules:
            try:
                # 標準多頭注意力模組只需要一個輸入參數
                result = attention_module(x, **kwargs)
                if isinstance(result, tuple) and len(result) >= 2:
                    output, weights = result[0], result[1]
                else:
                    output = result
                    weights = None
                attention_outputs.append(output)
                attention_weights_list.append(weights)
            except Exception as e:
                print(f"WeightedAttentionFusion注意力模組錯誤: {e}")
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

        batch_size, seq_len, _ = x.size()
        
        # 計算各個注意力模組的輸出
        attention_outputs = []
        attention_weights_list = []
        
        for i, attention_module in enumerate(self.attention_modules):
            try:
                # 標準多頭注意力模組只需要一個輸入參數
                result = attention_module(x, **kwargs)
                if isinstance(result, tuple) and len(result) >= 2:
                    output, weights = result[0], result[1]
                else:
                    output = result
                    weights = None
                
                # 上下文融合
                context_input = torch.cat([x, output], dim=-1)  # [batch_size, seq_len, hidden_dim * 2]
                fused_context = self.context_fusion[i](context_input)  # [batch_size, seq_len, hidden_dim]
                
                attention_outputs.append(fused_context)
                attention_weights_list.append(weights)
            except Exception as e:
                print(f"GatedAttentionFusion注意力模組錯誤: {e}")
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
            x: 輸入張量 [batch_size, seq_len, hidden_dim]
            **kwargs: 其他參數，傳遞給各個注意力模組
            
        Returns:
            fused_output: 融合後的輸出 [batch_size, seq_len, hidden_dim]
            fusion_info: 融合資訊字典
        """
        # 確保所有模組和輸入在同一設備上
        device = x.device
        if self.device != device:
            self.device = device
            self.to(device)

        # 確保子模組也在正確的設備上
        self.weighted_fusion = self.weighted_fusion.to(device)
        self.gated_fusion = self.gated_fusion.to(device)
        self.strategy_network = self.strategy_network.to(device)
        self.sequential_projection = self.sequential_projection.to(device)
        self.layer_norm = self.layer_norm.to(device)
        
        # 決定融合策略
        batch_size, seq_len, feature_dim = x.size()
        
        # 確保輸入維度正確
        if feature_dim != self.hidden_dim:
            # 如果輸入維度不匹配，使用投影層調整
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(feature_dim, self.hidden_dim).to(x.device)
            x = self.input_projection(x)
        
        strategy_input = x.mean(dim=1)  # [batch_size, hidden_dim]
        strategy_logits = self.strategy_network(strategy_input)  # [batch_size, 3]
        strategy_weights = F.softmax(strategy_logits, dim=-1)  # [batch_size, 3]
        
        # 計算各種融合策略的輸出
        weighted_output, weighted_weights = self.weighted_fusion(x, **kwargs)
        gated_output, gated_weights = self.gated_fusion(x, **kwargs)
        
        # 確保輸出在正確的設備上
        weighted_output = weighted_output.to(device)
        gated_output = gated_output.to(device)
        
        # 串聯融合
        attention_outputs = []
        for attention_module in self.attention_modules:
            # 確保注意力模組在正確的設備上
            attention_module = attention_module.to(device)
            
            try:
                result = attention_module(x, **kwargs)
                # 處理可變返回值
                if isinstance(result, tuple):
                    output = result[0]
                else:
                    output = result
                attention_outputs.append(output.to(device))
            except:
                if hasattr(attention_module, '__class__'):
                    module_name = attention_module.__class__.__name__
                    if 'CrossModal' in module_name:
                        result = attention_module(x, x, x, **kwargs)
                    else:
                        result = attention_module(x, **kwargs)
                else:
                    result = attention_module(x, **kwargs)

                # 處理異常情況下的可變返回值
                if isinstance(result, tuple):
                    output = result[0]
                else:
                    output = result
                attention_outputs.append(output.to(device))
        
        sequential_concat = torch.cat(attention_outputs, dim=-1)  # [batch_size, seq_len, hidden_dim * num_attentions]
        sequential_output = self.sequential_projection(sequential_concat)  # [batch_size, seq_len, hidden_dim]
        
        # 自適應融合
        # 正確處理 strategy_weights 的維度以匹配 outputs_stack
        strategy_weights = strategy_weights.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, 3]
        
        outputs_stack = torch.stack([weighted_output, gated_output, sequential_output], dim=-1)  # [batch_size, seq_len, hidden_dim, 3]
        adaptive_output = torch.sum(outputs_stack * strategy_weights, dim=-1)  # [batch_size, seq_len, hidden_dim]
        
        # 最終層正規化
        final_output = self.layer_norm(adaptive_output)
        
        fusion_info = {
            'strategy_weights': strategy_weights.squeeze(),
            'weighted_weights': weighted_weights,
            'gated_weights': gated_weights
        }
        
        return final_output, fusion_info


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
            x: 輸入張量 [batch_size, seq_len, hidden_dim]
            **kwargs: 其他參數，傳遞給各個注意力模組

        Returns:
            fused_output: 融合後的輸出 [batch_size, seq_len, hidden_dim]
            attention_weights_list: 各個注意力模組的權重列表
        """
        # 確保模組在正確的設備上
        device = x.device
        self.to(device)

        # 計算各個注意力模組的輸出
        attention_outputs = []
        attention_weights_list = []
        
        for attention_module in self.attention_modules:
            try:
                # 標準多頭注意力模組只需要一個輸入參數
                result = attention_module(x, **kwargs)
                if isinstance(result, tuple) and len(result) >= 2:
                    output, weights = result[0], result[1]
                else:
                    output = result
                    weights = None
                attention_outputs.append(output)
                attention_weights_list.append(weights)
            except Exception as e:
                print(f"CrossAttentionFusion注意力模組錯誤: {e}")
                # 使用原始輸入作為備選
                attention_outputs.append(x)
                attention_weights_list.append(None)
        
        # 跨注意力交互
        cross_enhanced_outputs = []
        for i, (output, cross_attn) in enumerate(zip(attention_outputs, self.cross_attention_layers)):
            # 計算與其他注意力輸出的交互
            other_outputs = [attention_outputs[j] for j in range(self.num_attentions) if j != i]
            if other_outputs:
                other_concat = torch.cat(other_outputs, dim=1)  # 在序列維度拼接
                
                # 跨注意力計算
                cross_output, _ = cross_attn(output, other_concat, other_concat)
                
                # 特徵融合
                fusion_input = torch.cat([output, cross_output], dim=-1)
                enhanced_output = self.feature_fusion[i](fusion_input)
            else:
                enhanced_output = output
            
            cross_enhanced_outputs.append(enhanced_output)
        
        # 最終融合
        final_concat = torch.cat(cross_enhanced_outputs, dim=-1)  # [batch_size, seq_len, hidden_dim * num_attentions]
        fused_output = self.final_fusion(final_concat)  # [batch_size, seq_len, hidden_dim]
        
        # 殘差連接和層正規化
        fused_output = self.layer_norm(fused_output + x)
        
        return fused_output, attention_weights_list


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