# 特徵融合層
"""
特徵融合模組

整合多種特徵表示，包括：
- BERT語義特徵
- TF-IDF統計特徵  
- LDA主題特徵
- 統計語言特徵
- 跨模態特徵融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np


class MultiModalFeatureFusion(nn.Module):
    """
    多模態特徵融合器
    
    融合BERT、TF-IDF、LDA等多種特徵表示
    """
    
    def __init__(self,
                 feature_dims: Dict[str, int],
                 fusion_dim: int = 512,
                 fusion_strategy: str = 'attention',
                 dropout_rate: float = 0.1):
        """
        初始化多模態特徵融合器
        
        Args:
            feature_dims: 各特徵維度字典 {'bert': 768, 'tfidf': 1000, 'lda': 50, 'stats': 20}
            fusion_dim: 融合後維度
            fusion_strategy: 融合策略 ('concat', 'attention', 'gated', 'bilinear')
            dropout_rate: Dropout比率
        """
        super(MultiModalFeatureFusion, self).__init__()
        
        self.feature_dims = feature_dims
        self.fusion_dim = fusion_dim
        self.fusion_strategy = fusion_strategy
        self.dropout_rate = dropout_rate
        self.feature_types = list(feature_dims.keys())
        
        # 特徵投影層
        self.feature_projections = nn.ModuleDict()
        for feature_type, dim in feature_dims.items():
            self.feature_projections[feature_type] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        
        # 融合策略實現
        if fusion_strategy == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_dim * len(self.feature_types), fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        
        elif fusion_strategy == 'attention':
            self.attention_fusion = AttentionBasedFusion(
                fusion_dim, len(self.feature_types), dropout_rate
            )
            self.fusion_layer = nn.Identity()
        
        elif fusion_strategy == 'gated':
            self.gated_fusion = GatedMultiModalFusion(
                fusion_dim, len(self.feature_types), dropout_rate
            )
            self.fusion_layer = nn.Identity()
        
        elif fusion_strategy == 'bilinear':
            self.bilinear_fusion = BilinearFusion(
                fusion_dim, len(self.feature_types), dropout_rate
            )
            self.fusion_layer = nn.Identity()
        
        # 輸出層
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def calculate_actual_feature_dims(self, features: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """
        計算輸入特徵的實際維度
        """
        actual_dims = {}
        for feature_type, feature_tensor in features.items():
            if isinstance(feature_tensor, torch.Tensor):
                # 如果是多維張量，展平除了批次維度
                if feature_tensor.dim() > 2:
                    actual_dim = int(torch.prod(torch.tensor(feature_tensor.shape[1:])))
                else:
                    actual_dim = feature_tensor.shape[-1]
                actual_dims[feature_type] = actual_dim
        return actual_dims
        
    def reinitialize_for_input_dims(self, actual_feature_dims: Dict[str, int]):
        """
        根據實際特徵維度重新初始化特徵投影層
        """
        if actual_feature_dims != self.feature_dims:
            print(f"Warning: Expected feature dims {self.feature_dims}, got {actual_feature_dims}")
            print("Reinitializing MultiModalFeatureFusion with correct dimensions...")
            
            # 保存當前設備
            device = next(self.parameters()).device
            
            # 更新特徵維度
            self.feature_dims = actual_feature_dims
            self.feature_types = list(actual_feature_dims.keys())
            
            # 重建特徵投影層
            self.feature_projections = nn.ModuleDict()
            for feature_type, dim in actual_feature_dims.items():
                self.feature_projections[feature_type] = nn.Sequential(
                    nn.Linear(dim, self.fusion_dim),
                    nn.LayerNorm(self.fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate)
                ).to(device)
            
            # 重建融合層（如果需要）
            if self.fusion_strategy == 'concat':
                self.fusion_layer = nn.Sequential(
                    nn.Linear(self.fusion_dim * len(self.feature_types), self.fusion_dim),
                    nn.LayerNorm(self.fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate)
                ).to(device)
            elif self.fusion_strategy == 'attention':
                self.attention_fusion = AttentionBasedFusion(
                    self.fusion_dim, len(self.feature_types), self.dropout_rate
                ).to(device)
            elif self.fusion_strategy == 'gated':
                self.gated_fusion = GatedMultiModalFusion(
                    self.fusion_dim, len(self.feature_types), self.dropout_rate
                ).to(device)
            elif self.fusion_strategy == 'bilinear':
                self.bilinear_fusion = BilinearFusion(
                    self.fusion_dim, len(self.feature_types), self.dropout_rate
                ).to(device)
            
            return True
        return False
        
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            features: 特徵字典 {feature_type: tensor}
        
        Returns:
            融合結果字典
        """
        # 過濾掉嵌套字典（如 projected_features）
        filtered_features = {}
        for feature_type, feature_tensor in features.items():
            if isinstance(feature_tensor, torch.Tensor):
                filtered_features[feature_type] = feature_tensor
            elif isinstance(feature_tensor, dict):
                print(f"Warning: Skipping nested dict feature '{feature_type}' in forward pass")
        
        # 檢查實際特徵維度
        actual_dims = self.calculate_actual_feature_dims(filtered_features)
        if actual_dims != self.feature_dims:
            print(f"Dimension mismatch detected: expected {self.feature_dims}, got {actual_dims}")
            raise RuntimeError(
                f"Feature dimension mismatch in MultiModalFeatureFusion. "
                f"Expected {self.feature_dims}, got {actual_dims}. "
                f"Please call reinitialize_for_input_dims() before training."
            )
        
        # 投影所有特徵到相同維度
        projected_features = {}
        for feature_type, feature_tensor in filtered_features.items():
            if feature_type in self.feature_projections:
                projected_features[feature_type] = self.feature_projections[feature_type](
                    feature_tensor
                )
        
        # 根據融合策略進行特徵融合
        if self.fusion_strategy == 'concat':
            # 連接所有特徵
            concatenated = torch.cat(list(projected_features.values()), dim=-1)
            fused_features = self.fusion_layer(concatenated)
        
        elif self.fusion_strategy == 'attention':
            # 注意力融合
            feature_list = list(projected_features.values())
            fused_features, attention_weights = self.attention_fusion(feature_list)
        
        elif self.fusion_strategy == 'gated':
            # 門控融合
            feature_list = list(projected_features.values())
            fused_features = self.gated_fusion(feature_list)
        
        elif self.fusion_strategy == 'bilinear':
            # 雙線性融合
            feature_list = list(projected_features.values())
            fused_features = self.bilinear_fusion(feature_list)
        
        # 輸出投影
        output_features = self.output_projection(fused_features)
        
        return {
            'fused_features': output_features,
            'projected_features': projected_features,
            'attention_weights': attention_weights if self.fusion_strategy == 'attention' else None
        }


class AttentionBasedFusion(nn.Module):
    """
    基於注意力的特徵融合
    """
    
    def __init__(self, feature_dim: int, num_features: int, dropout_rate: float):
        super(AttentionBasedFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_features = num_features
        
        # 注意力計算層
        self.attention_weights = nn.Parameter(torch.ones(num_features))
        self.attention_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # 上下文向量
        self.context_vector = nn.Parameter(torch.randn(feature_dim))
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, feature_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        注意力融合前向傳播
        
        Args:
            feature_list: 特徵列表
        
        Returns:
            融合特徵和注意力權重
        """
        batch_size = feature_list[0].size(0)
        
        # 堆疊特徵 [batch_size, num_features, feature_dim]
        stacked_features = torch.stack(feature_list, dim=1)
        
        # 計算注意力得分
        # 方法1: 基於參數的靜態注意力
        static_weights = F.softmax(self.attention_weights, dim=0)
        
        # 方法2: 基於內容的動態注意力
        attention_scores = torch.sum(
            stacked_features * self.context_vector.unsqueeze(0).unsqueeze(0), 
            dim=-1
        )  # [batch_size, num_features]
        dynamic_weights = F.softmax(attention_scores, dim=-1)
        
        # 結合靜態和動態注意力
        combined_weights = 0.5 * static_weights.unsqueeze(0) + 0.5 * dynamic_weights
        combined_weights = self.dropout(combined_weights)
        
        # 加權融合
        fused_features = torch.sum(
            stacked_features * combined_weights.unsqueeze(-1), dim=1
        )
        
        return fused_features, combined_weights


class GatedMultiModalFusion(nn.Module):
    """
    門控多模態融合
    """
    
    def __init__(self, feature_dim: int, num_features: int, dropout_rate: float):
        super(GatedMultiModalFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_features = num_features
        
        # 門控網路
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * num_features, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid()
            ) for _ in range(num_features)
        ])
        
        # 特徵變換網路
        self.feature_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_features)
        ])
        
        # 融合投影
        self.fusion_projection = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        """
        門控融合前向傳播
        
        Args:
            feature_list: 特徵列表
        
        Returns:
            門控融合結果
        """
        # 連接所有特徵用於門控計算
        concatenated = torch.cat(feature_list, dim=-1)
        
        # 計算每個特徵的門控權重和變換
        gated_features = []
        for i, feature in enumerate(feature_list):
            # 計算門控權重
            gate = self.gate_networks[i](concatenated)
            
            # 變換特徵
            transformed = self.feature_transforms[i](feature)
            
            # 應用門控
            gated_feature = gate * transformed
            gated_features.append(gated_feature)
        
        # 求和融合
        fused = sum(gated_features)
        
        # 最終投影
        output = self.fusion_projection(fused)
        
        return output


class BilinearFusion(nn.Module):
    """
    雙線性融合模組
    """
    
    def __init__(self, feature_dim: int, num_features: int, dropout_rate: float):
        super(BilinearFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_features = num_features
        
        # 雙線性變換矩陣
        self.bilinear_layers = nn.ModuleList([
            nn.Bilinear(feature_dim, feature_dim, feature_dim)
            for _ in range(num_features - 1)
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        """
        雙線性融合前向傳播
        
        Args:
            feature_list: 特徵列表
        
        Returns:
            雙線性融合結果
        """
        if len(feature_list) < 2:
            return feature_list[0] if feature_list else torch.zeros(1)
        
        # 從第一個特徵開始
        fused = feature_list[0]
        
        # 逐步進行雙線性融合
        for i, feature in enumerate(feature_list[1:]):
            if i < len(self.bilinear_layers):
                fused = self.bilinear_layers[i](fused, feature)
                fused = self.dropout(fused)
            else:
                # 如果特徵數量超過預定義的雙線性層數，使用簡單加法
                fused = fused + feature
        
        return fused


class HierarchicalFeatureFusion(nn.Module):
    """
    階層式特徵融合
    
    按照特徵重要性進行階層式融合
    """
    
    def __init__(self,
                 feature_hierarchy: List[List[str]],
                 feature_dims: Dict[str, int],
                 fusion_dim: int = 512,
                 dropout_rate: float = 0.1):
        """
        初始化階層式特徵融合
        
        Args:
            feature_hierarchy: 特徵階層 [['bert'], ['tfidf', 'lda'], ['stats']]
            feature_dims: 特徵維度字典
            fusion_dim: 融合維度
            dropout_rate: Dropout比率
        """
        super(HierarchicalFeatureFusion, self).__init__()
        
        self.feature_hierarchy = feature_hierarchy
        self.feature_dims = feature_dims
        self.fusion_dim = fusion_dim
        
        # 為每個階層建立融合器
        self.level_fusers = nn.ModuleList()
        current_dim = 0
        
        for level, features in enumerate(feature_hierarchy):
            if level == 0:
                # 第一層直接投影
                level_dim = sum(feature_dims[f] for f in features)
                fuser = nn.Sequential(
                    nn.Linear(level_dim, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
                current_dim = fusion_dim
            else:
                # 後續層融合前一層結果和當前層特徵
                level_dim = sum(feature_dims[f] for f in features)
                fuser = nn.Sequential(
                    nn.Linear(current_dim + level_dim, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            
            self.level_fusers.append(fuser)
        
        # 特徵投影層
        self.feature_projections = nn.ModuleDict()
        for feature_type, dim in feature_dims.items():
            self.feature_projections[feature_type] = nn.Linear(dim, dim)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        階層式融合前向傳播
        
        Args:
            features: 特徵字典
        
        Returns:
            階層式融合結果
        """
        level_outputs = []
        previous_output = None
        
        for level, feature_names in enumerate(self.feature_hierarchy):
            # 收集當前層的特徵
            level_features = []
            for feature_name in feature_names:
                if feature_name in features:
                    projected = self.feature_projections[feature_name](features[feature_name])
                    level_features.append(projected)
            
            if not level_features:
                continue
            
            # 連接當前層特徵
            level_concat = torch.cat(level_features, dim=-1)
            
            # 如果不是第一層，與前一層輸出連接
            if previous_output is not None:
                level_input = torch.cat([previous_output, level_concat], dim=-1)
            else:
                level_input = level_concat
            
            # 通過對應的融合器
            level_output = self.level_fusers[level](level_input)
            level_outputs.append(level_output)
            previous_output = level_output
        
        return {
            'final_features': level_outputs[-1] if level_outputs else torch.zeros(1),
            'level_features': level_outputs
        }


class CrossModalAttentionFusion(nn.Module):
    """
    跨模態注意力融合
    
    不同模態特徵之間的交互注意力
    """
    
    def __init__(self,
                 feature_dims: Dict[str, int],
                 num_heads: int = 8,
                 dropout_rate: float = 0.1):
        """
        初始化跨模態注意力融合
        
        Args:
            feature_dims: 特徵維度字典
            num_heads: 注意力頭數
            dropout_rate: Dropout比率
        """
        super(CrossModalAttentionFusion, self).__init__()
        
        self.feature_dims = feature_dims
        self.num_heads = num_heads
        self.feature_types = list(feature_dims.keys())
        
        # 統一特徵維度
        self.unified_dim = max(feature_dims.values())
        
        # 特徵投影到統一維度
        self.feature_projections = nn.ModuleDict()
        for feature_type, dim in feature_dims.items():
            if dim != self.unified_dim:
                self.feature_projections[feature_type] = nn.Linear(dim, self.unified_dim)
            else:
                self.feature_projections[feature_type] = nn.Identity()
        
        # 跨模態注意力層
        self.cross_modal_attentions = nn.ModuleDict()
        for i, source_type in enumerate(self.feature_types):
            for j, target_type in enumerate(self.feature_types):
                if i != j:
                    attention_name = f"{source_type}_to_{target_type}"
                    self.cross_modal_attentions[attention_name] = nn.MultiheadAttention(
                        embed_dim=self.unified_dim,
                        num_heads=num_heads,
                        dropout=dropout_rate,
                        batch_first=True
                    )
        
        # 融合層
        self.fusion_projection = nn.Linear(
            self.unified_dim * len(self.feature_types), 
            self.unified_dim
        )
        
        self.layer_norm = nn.LayerNorm(self.unified_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        跨模態注意力融合前向傳播
        
        Args:
            features: 特徵字典
        
        Returns:
            跨模態融合結果
        """
        # 投影所有特徵到統一維度
        unified_features = {}
        for feature_type, feature_tensor in features.items():
            if feature_type in self.feature_projections:
                # 如果是1D特徵，擴展為2D以適配MultiheadAttention
                if feature_tensor.dim() == 2:
                    feature_tensor = feature_tensor.unsqueeze(1)
                
                projected = self.feature_projections[feature_type](feature_tensor)
                unified_features[feature_type] = projected
        
        # 跨模態注意力計算
        attended_features = {}
        for source_type in self.feature_types:
            if source_type not in unified_features:
                continue
            
            source_feature = unified_features[source_type]
            attended_list = [source_feature]  # 包含自身
            
            for target_type in self.feature_types:
                if target_type == source_type or target_type not in unified_features:
                    continue
                
                target_feature = unified_features[target_type]
                attention_name = f"{source_type}_to_{target_type}"
                
                if attention_name in self.cross_modal_attentions:
                    attended, _ = self.cross_modal_attentions[attention_name](
                        query=source_feature,
                        key=target_feature,
                        value=target_feature
                    )
                    attended_list.append(attended)
            
            # 融合所有注意力結果
            if len(attended_list) > 1:
                attended_features[source_type] = torch.cat(attended_list, dim=-1)
            else:
                attended_features[source_type] = attended_list[0]
        
        # 最終融合
        if attended_features:
            all_attended = torch.cat(list(attended_features.values()), dim=-1)
            fused = self.fusion_projection(all_attended)
            fused = self.layer_norm(fused)
            fused = self.dropout(fused)
        else:
            fused = torch.zeros(1, 1, self.unified_dim)
        
        return {
            'fused_features': fused.squeeze(1) if fused.dim() == 3 else fused,
            'attended_features': attended_features
        }