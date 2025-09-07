# 情感分類器
"""
情感分類器模組

提供多種情感分類架構，包括：
- 基礎分類器
- 多層感知機分類器
- 注意力增強分類器
- 階層式分類器
- 跨領域分類器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from abc import ABC, abstractmethod


class BaseClassifier(nn.Module, ABC):
    """
    分類器基礎類
    
    提供所有分類器的通用接口和功能
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 dropout_rate: float = 0.1):
        """
        初始化基礎分類器
        
        Args:
            input_dim: 輸入特徵維度
            num_classes: 分類數量
            dropout_rate: Dropout比率
        """
        super(BaseClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 通用Dropout層
        self.dropout = nn.Dropout(dropout_rate)
        
    def _process_input_features(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        處理輸入特徵，將字典格式轉換為張量
        
        Args:
            x: 輸入特徵或特徵字典
            
        Returns:
            處理後的張量
        """
        if isinstance(x, dict):
            # 將所有特徵拼接成單一張量
            feature_tensors = []
            for key in sorted(x.keys()):  # 保持順序一致性
                feature = x[key]
                # 確保特徵是張量
                if isinstance(feature, torch.Tensor):
                    # 如果是多維張量，展平除了批次維度
                    if feature.dim() > 2:
                        feature = feature.view(feature.size(0), -1)
                    feature_tensors.append(feature)
                elif isinstance(feature, np.ndarray):
                    feature_tensor = torch.from_numpy(feature).float()
                    if feature_tensor.dim() > 2:
                        feature_tensor = feature_tensor.view(feature_tensor.size(0), -1)
                    feature_tensors.append(feature_tensor)
                elif isinstance(feature, dict):
                    # 如果特徵本身是字典，遞歸展平處理
                    try:
                        flattened_features = self._flatten_dict_features(feature)
                        if flattened_features is not None and len(flattened_features) > 0:
                            feature_tensors.extend(flattened_features)
                        else:
                            print(f"Warning: 巢狀字典特徵 '{key}' 無有效張量，跳過")
                    except Exception as e:
                        print(f"Warning: 處理巢狀字典特徵 '{key}' 時發生錯誤: {e}，跳過")
                    continue
                else:
                    print(f"Warning: Skipping unsupported feature type '{key}': {type(feature)}")
                    continue
            
            if not feature_tensors:
                # 如果沒有有效的特徵張量，創建一個零張量
                batch_size = 1  # 默認批次大小
                return torch.zeros(batch_size, 1)
            else:
                return torch.cat(feature_tensors, dim=-1)
        else:
            return x
    
    def _flatten_dict_features(self, feature_dict: Dict) -> List[torch.Tensor]:
        """
        遞歸展平字典特徵
        
        Args:
            feature_dict: 包含特徵的字典
            
        Returns:
            展平後的張量列表
        """
        feature_tensors = []
        
        for key, value in feature_dict.items():
            try:
                if isinstance(value, torch.Tensor):
                    # 如果是多維張量，展平除了批次維度
                    if value.dim() > 2:
                        value = value.view(value.size(0), -1)
                    feature_tensors.append(value)
                elif isinstance(value, np.ndarray):
                    tensor_value = torch.from_numpy(value).float()
                    if tensor_value.dim() > 2:
                        tensor_value = tensor_value.view(tensor_value.size(0), -1)
                    feature_tensors.append(tensor_value)
                elif isinstance(value, dict):
                    # 遞歸處理巢狀字典
                    nested_tensors = self._flatten_dict_features(value)
                    feature_tensors.extend(nested_tensors)
                else:
                    # 跳過不支援的資料類型（但不輸出警告，避免過多輸出）
                    continue
            except Exception as e:
                print(f"Warning: 處理字典特徵 '{key}' 時發生錯誤: {e}")
                continue
        
        return feature_tensors
        
    @abstractmethod
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        前向傳播抽象方法
        
        Args:
            x: 輸入特徵或特徵字典
            
        Returns:
            分類結果字典
        """
        pass
        
    def predict(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        預測方法
        
        Args:
            x: 輸入特徵或特徵字典
            
        Returns:
            預測標籤
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = torch.argmax(outputs['logits'], dim=-1)
        return predictions
    
    def predict_proba(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        預測機率方法
        
        Args:
            x: 輸入特徵或特徵字典
            
        Returns:
            預測機率
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs['logits'], dim=-1)
        return probabilities


class MLPClassifier(BaseClassifier):
    """
    多層感知機分類器
    
    使用多層全連接網路進行情感分類
    """
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: List[int] = [512, 256],
                 dropout_rate: float = 0.1,
                 activation: str = 'relu'):
        """
        初始化MLP分類器
        
        Args:
            input_dim: 輸入特徵維度
            num_classes: 分類數量
            hidden_dims: 隱藏層維度列表
            dropout_rate: Dropout比率
            activation: 激活函數類型
        """
        super(MLPClassifier, self).__init__(input_dim, num_classes, dropout_rate)
        
        self.hidden_dims = hidden_dims
        self.activation = activation
        self._expected_input_dim = input_dim
        
        # 直接建立網路層
        self._build_network(input_dim)
        
    def _build_network(self, input_dim: int):
        """建立網路層"""
        # 建立MLP層
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(self.activation),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 輸出層
        layers.append(nn.Linear(prev_dim, self.num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # 特徵提取層（輸出層之前的部分）
        self.feature_extractor = nn.Sequential(*layers[:-1])
        self.output_layer = layers[-1]
        
    def reinitialize_for_input_dim(self, actual_input_dim: int):
        """
        根據實際輸入維度重新初始化網路（當維度不匹配時使用）
        
        Args:
            actual_input_dim: 實際輸入特徵維度
        """
        if self._expected_input_dim != actual_input_dim:
            print(f"⚠️  維度不匹配：預期輸入維度 {self._expected_input_dim}，實際得到 {actual_input_dim}")
            print("🔄 正在使用正確維度重新初始化分類器...")
            
            # 保存當前設備
            device = next(self.parameters()).device
            
            # 重建網路
            self._expected_input_dim = actual_input_dim
            self._build_network(actual_input_dim)
            
            # 移動到正確設備
            self.to(device)
            
            return True
        return False
    
    def _get_activation(self, activation: str) -> nn.Module:
        """
        獲取激活函數
        
        Args:
            activation: 激活函數名稱
            
        Returns:
            激活函數模組
        """
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入特徵 [batch_size, input_dim] 或特徵字典
            
        Returns:
            分類結果字典
        """
        # 處理輸入特徵
        x = self._process_input_features(x)
        
        # 檢查輸入維度是否與期望的匹配
        actual_input_dim = x.size(-1)
        if actual_input_dim != self._expected_input_dim:
            print(f"⚠️ 前向傳播中檢測到維度不匹配：預期 {self._expected_input_dim}，實際得到 {actual_input_dim}")
            print("🔄 自動重新初始化模型維度...")
            # 自動重新初始化
            if hasattr(self, 'reinitialize_for_input_dim'):
                self.reinitialize_for_input_dim(actual_input_dim)
                print("✅ 模型維度重新初始化完成")
            else:
                raise RuntimeError(
                    f"輸入維度不匹配：預期 {self._expected_input_dim}，實際得到 {actual_input_dim}。"
                    f"請在訓練前呼叫 reinitialize_for_input_dim() 方法。"
                    f"這個錯誤表示模型期望的輸入特徵數量與實際提供的不同。"
                )
        
        # 提取特徵
        features = self.feature_extractor(x)
        
        # 輸出logits
        logits = self.output_layer(features)
        
        return {
            'logits': logits,
            'features': features,
            'probabilities': F.softmax(logits, dim=-1)
        }


class AttentionEnhancedClassifier(BaseClassifier):
    """
    注意力增強分類器
    
    使用注意力機制增強特徵表示的分類器
    """
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 num_heads: int = 8,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.1):
        """
        初始化注意力增強分類器
        
        Args:
            input_dim: 輸入特徵維度
            num_classes: 分類數量
            num_heads: 注意力頭數
            hidden_dim: 隱藏層維度
            dropout_rate: Dropout比率
        """
        super(AttentionEnhancedClassifier, self).__init__(input_dim, num_classes, dropout_rate)
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # 輸入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 自注意力層
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 前饋網路
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 層正規化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 分類頭
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 位置編碼（可選）
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 保存初始化參數
        self._expected_input_dim = input_dim
        
    def reinitialize_for_input_dim(self, actual_input_dim: int):
        """
        根據實際輸入維度重新初始化網路（當維度不匹配時使用）
        
        Args:
            actual_input_dim: 實際輸入特徵維度
        """
        if self._expected_input_dim != actual_input_dim:
            print(f"⚠️  注意力增強分類器維度不匹配：預期 {self._expected_input_dim}，實際得到 {actual_input_dim}")
            print("🔄 正在使用正確維度重新初始化注意力增強分類器...")
            
            # 保存當前設備
            device = next(self.parameters()).device
            
            # 重建輸入投影層
            self._expected_input_dim = actual_input_dim
            self.input_projection = nn.Linear(actual_input_dim, self.hidden_dim).to(device)
            
            # 重建其他層（保持原有架構）
            self.self_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
                batch_first=True
            ).to(device)
            
            self.feed_forward = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            ).to(device)
            
            self.layer_norm1 = nn.LayerNorm(self.hidden_dim).to(device)
            self.layer_norm2 = nn.LayerNorm(self.hidden_dim).to(device)
            
            self.classifier_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.num_classes)
            ).to(device)
            
            # 重新初始化位置編碼
            self.positional_encoding = nn.Parameter(torch.randn(1, 1, self.hidden_dim).to(device))
            
            return True
        return False
    
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入特徵 [batch_size, input_dim] 或 [batch_size, seq_len, input_dim] 或特徵字典
            
        Returns:
            分類結果字典
        """
        # 處理輸入特徵
        x = self._process_input_features(x)
        
        # 如果是2D輸入，擴展為3D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        batch_size, seq_len, _ = x.size()
        
        # 輸入投影
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # 添加位置編碼
        x = x + self.positional_encoding.expand(batch_size, seq_len, -1)
        
        # 自注意力
        attended, attention_weights = self.self_attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attended))
        
        # 前饋網路
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        # 池化（如果序列長度 > 1）
        if seq_len > 1:
            # 平均池化
            pooled_features = torch.mean(x, dim=1)
        else:
            pooled_features = x.squeeze(1)
        
        # 分類
        logits = self.classifier_head(pooled_features)
        
        return {
            'logits': logits,
            'features': pooled_features,
            'attention_weights': attention_weights,
            'probabilities': F.softmax(logits, dim=-1)
        }


class HierarchicalClassifier(BaseClassifier):
    """
    階層式分類器
    
    進行粗粒度到細粒度的階層式分類
    """
    
    def __init__(self,
                 input_dim: int,
                 coarse_classes: int,
                 fine_classes: int,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.1):
        """
        初始化階層式分類器
        
        Args:
            input_dim: 輸入特徵維度
            coarse_classes: 粗粒度分類數
            fine_classes: 細粒度分類數
            hidden_dim: 隱藏層維度
            dropout_rate: Dropout比率
        """
        # 使用fine_classes作為num_classes
        super(HierarchicalClassifier, self).__init__(input_dim, fine_classes, dropout_rate)
        
        self.coarse_classes = coarse_classes
        self.fine_classes = fine_classes
        self.hidden_dim = hidden_dim
        
        # 共享特徵提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 粗粒度分類器
        self.coarse_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, coarse_classes)
        )
        
        # 細粒度分類器
        self.fine_classifier = nn.Sequential(
            nn.Linear(hidden_dim + coarse_classes, hidden_dim),  # 融合粗分類結果
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, fine_classes)
        )
        
        # 保存初始化參數
        self._expected_input_dim = input_dim
        
    def reinitialize_for_input_dim(self, actual_input_dim: int):
        """
        根據實際輸入維度重新初始化網路（當維度不匹配時使用）
        """
        if self._expected_input_dim != actual_input_dim:
            print(f"⚠️  階層式分類器維度不匹配：預期 {self._expected_input_dim}，實際得到 {actual_input_dim}")
            print("🔄 正在使用正確維度重新初始化階層式分類器...")
            
            # 保存當前設備
            device = next(self.parameters()).device
            
            # 重建特徵提取器
            self._expected_input_dim = actual_input_dim
            self.feature_extractor = nn.Sequential(
                nn.Linear(actual_input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ).to(device)
            
            # 重建分類器（保持原有結構）
            self.coarse_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.coarse_classes)
            ).to(device)
            
            self.fine_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim + self.coarse_classes, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.fine_classes)
            ).to(device)
            
            return True
        return False
    
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入特徵 [batch_size, input_dim] 或特徵字典
            
        Returns:
            分類結果字典
        """
        # 處理輸入特徵
        x = self._process_input_features(x)
        
        # 特徵提取
        features = self.feature_extractor(x)
        
        # 粗粒度分類
        coarse_logits = self.coarse_classifier(features)
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        
        # 融合粗分類結果進行細分類
        fine_input = torch.cat([features, coarse_probs], dim=-1)
        fine_logits = self.fine_classifier(fine_input)
        
        return {
            'logits': fine_logits,  # 主要輸出是細分類結果
            'coarse_logits': coarse_logits,
            'fine_logits': fine_logits,
            'features': features,
            'coarse_probabilities': coarse_probs,
            'fine_probabilities': F.softmax(fine_logits, dim=-1),
            'probabilities': F.softmax(fine_logits, dim=-1)  # 保持與基類一致
        }


class CrossDomainClassifier(BaseClassifier):
    """
    跨領域分類器
    
    針對跨領域情感分析設計的分類器
    """
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 num_domains: int,
                 domain_adaptation: bool = True,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.1):
        """
        初始化跨領域分類器
        
        Args:
            input_dim: 輸入特徵維度
            num_classes: 分類數量
            num_domains: 領域數量
            domain_adaptation: 是否使用領域適應
            hidden_dim: 隱藏層維度
            dropout_rate: Dropout比率
        """
        super(CrossDomainClassifier, self).__init__(input_dim, num_classes, dropout_rate)
        
        self.num_domains = num_domains
        self.domain_adaptation = domain_adaptation
        self.hidden_dim = hidden_dim
        
        # 共享特徵提取器
        self.shared_feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 領域特定特徵提取器
        if domain_adaptation:
            self.domain_feature_extractors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ) for _ in range(num_domains)
            ])
            
            # 領域判別器（用於對抗訓練）
            self.domain_discriminator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, num_domains)
            )
            
            feature_dim = hidden_dim
        else:
            feature_dim = hidden_dim
        
        # 情感分類器
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 梯度反轉層權重（用於對抗訓練）
        self.lambda_grl = nn.Parameter(torch.tensor(1.0))
        
        # 保存初始化參數
        self._expected_input_dim = input_dim
        
    def reinitialize_for_input_dim(self, actual_input_dim: int):
        """
        根據實際輸入維度重新初始化網路（當維度不匹配時使用）
        """
        if self._expected_input_dim != actual_input_dim:
            print(f"⚠️  跨領域分類器維度不匹配：預期 {self._expected_input_dim}，實際得到 {actual_input_dim}")
            print("🔄 正在使用正確維度重新初始化跨領域分類器...")
            
            # 保存當前設備
            device = next(self.parameters()).device
            
            # 重建共享特徵提取器
            self._expected_input_dim = actual_input_dim
            self.shared_feature_extractor = nn.Sequential(
                nn.Linear(actual_input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ).to(device)
            
            # 重建其他層（保持原有結構）
            if self.domain_adaptation:
                self.domain_feature_extractors = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.LayerNorm(self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(self.dropout_rate)
                    ).to(device) for _ in range(self.num_domains)
                ])
                
                self.domain_discriminator = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.hidden_dim // 2, self.num_domains)
                ).to(device)
            
            self.sentiment_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.num_classes)
            ).to(device)
            
            return True
        return False
    
    def forward(self, 
                x: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                domain_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入特徵 [batch_size, input_dim] 或特徵字典
            domain_ids: 領域ID [batch_size] （可選）
            
        Returns:
            分類結果字典
        """
        # 處理輸入特徵
        x = self._process_input_features(x)
        
        batch_size = x.size(0)
        
        # 共享特徵提取
        shared_features = self.shared_feature_extractor(x)
        
        if self.domain_adaptation and domain_ids is not None:
            # 領域特定特徵提取
            domain_features = []
            for i in range(batch_size):
                domain_id = domain_ids[i].item()
                if domain_id < len(self.domain_feature_extractors):
                    domain_feature = self.domain_feature_extractors[domain_id](
                        shared_features[i:i+1]
                    )
                else:
                    domain_feature = shared_features[i:i+1]
                domain_features.append(domain_feature)
            
            domain_features = torch.cat(domain_features, dim=0)
            final_features = domain_features
            
            # 領域判別
            domain_logits = self.domain_discriminator(
                self._gradient_reversal(shared_features)
            )
        else:
            final_features = shared_features
            domain_logits = None
        
        # 情感分類
        sentiment_logits = self.sentiment_classifier(final_features)
        
        result = {
            'logits': sentiment_logits,
            'features': final_features,
            'shared_features': shared_features,
            'probabilities': F.softmax(sentiment_logits, dim=-1)
        }
        
        if domain_logits is not None:
            result.update({
                'domain_logits': domain_logits,
                'domain_probabilities': F.softmax(domain_logits, dim=-1)
            })
        
        return result
    
    def _gradient_reversal(self, x: torch.Tensor) -> torch.Tensor:
        """
        梯度反轉層
        
        Args:
            x: 輸入張量
            
        Returns:
            梯度反轉後的張量
        """
        return GradientReversalFunction.apply(x, self.lambda_grl)


class GradientReversalFunction(torch.autograd.Function):
    """
    梯度反轉函數
    
    用於對抗訓練的梯度反轉層
    """
    
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class EnsembleClassifier(nn.Module):
    """
    集成分類器
    
    組合多個分類器進行預測
    """
    
    def __init__(self,
                 classifiers: List[BaseClassifier],
                 ensemble_method: str = 'average',
                 weights: Optional[List[float]] = None):
        """
        初始化集成分類器
        
        Args:
            classifiers: 分類器列表
            ensemble_method: 集成方法 ('average', 'voting', 'learned')
            weights: 分類器權重（可選）
        """
        super(EnsembleClassifier, self).__init__()
        
        self.classifiers = nn.ModuleList(classifiers)
        self.ensemble_method = ensemble_method
        self.num_classifiers = len(classifiers)
        
        # 設定權重
        if weights is None:
            self.weights = torch.ones(self.num_classifiers) / self.num_classifiers
        else:
            self.weights = torch.tensor(weights)
            self.weights = self.weights / self.weights.sum()  # 正規化
        
        # 如果是學習式集成，添加元學習器
        if ensemble_method == 'learned':
            # 假設所有分類器有相同的類別數
            num_classes = classifiers[0].num_classes
            self.meta_learner = nn.Sequential(
                nn.Linear(num_classes * self.num_classifiers, num_classes * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(num_classes * 2, num_classes)
            )
    
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入特徵或特徵字典
            **kwargs: 額外參數
            
        Returns:
            集成分類結果
        """
        # 獲取所有分類器的預測
        all_outputs = []
        all_probabilities = []
        
        for classifier in self.classifiers:
            output = classifier(x, **kwargs)
            all_outputs.append(output)
            all_probabilities.append(output['probabilities'])
        
        # 根據集成方法組合結果
        if self.ensemble_method == 'average':
            # 加權平均
            weights = self.weights.to(x.device)
            ensemble_probs = sum(
                w * prob for w, prob in zip(weights, all_probabilities)
            )
            ensemble_logits = torch.log(ensemble_probs + 1e-8)
            
        elif self.ensemble_method == 'voting':
            # 多數投票
            predictions = [torch.argmax(prob, dim=-1) for prob in all_probabilities]
            stacked_preds = torch.stack(predictions, dim=1)  # [batch_size, num_classifiers]
            
            # 計算每個類別的投票數
            batch_size, num_classes = all_probabilities[0].shape
            vote_counts = torch.zeros(batch_size, num_classes, device=x.device)
            
            for i in range(batch_size):
                votes = stacked_preds[i]
                for vote in votes:
                    vote_counts[i, vote] += 1
            
            ensemble_probs = vote_counts / self.num_classifiers
            ensemble_logits = torch.log(ensemble_probs + 1e-8)
            
        elif self.ensemble_method == 'learned':
            # 使用元學習器
            concatenated_probs = torch.cat(all_probabilities, dim=-1)
            ensemble_logits = self.meta_learner(concatenated_probs)
            ensemble_probs = F.softmax(ensemble_logits, dim=-1)
        
        return {
            'logits': ensemble_logits,
            'probabilities': ensemble_probs,
            'individual_outputs': all_outputs
        }
    
    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        預測方法
        
        Args:
            x: 輸入特徵
            **kwargs: 額外參數
            
        Returns:
            預測標籤
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, **kwargs)
            predictions = torch.argmax(outputs['logits'], dim=-1)
        return predictions
    
    def predict_proba(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        預測機率方法
        
        Args:
            x: 輸入特徵
            **kwargs: 額外參數
            
        Returns:
            預測機率
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, **kwargs)
            probabilities = outputs['probabilities']
        return probabilities