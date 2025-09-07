# 訓練管理器
"""
訓練管理器模組

提供完整的模型訓練功能，包括：
- 損失函數定義
- 優化器配置
- 學習率調度器
- 早停機制
- 訓練過程管理
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, List, Optional, Any, Callable
import numpy as np
from pathlib import Path
import json


class TrainingManager:
    """訓練管理器"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 config: Dict[str, Any]):
        """
        初始化訓練管理器
        
        Args:
            model: 要訓練的模型
            train_loader: 訓練數據載入器
            val_loader: 驗證數據載入器
            config: 訓練配置
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 設備配置
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        # 優化器設置
        self.optimizer = self._setup_optimizer()
        
        # 學習率調度器
        self.scheduler = self._setup_scheduler()
        
        # 損失函數
        self.criterion = self._setup_criterion()
        
        # 早停設置
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 1e-4),
            restore_best_weights=config.get('restore_best_weights', True)
        )
        
        # 訓練歷史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """設置優化器"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        learning_rate = self.config.get('learning_rate', 2e-5)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def _setup_scheduler(self):
        """設置學習率調度器"""
        scheduler_name = self.config.get('scheduler', 'reduce_on_plateau').lower()
        
        if scheduler_name == 'step':
            step_size = self.config.get('step_size', 10)
            gamma = self.config.get('gamma', 0.1)
            return StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        elif scheduler_name == 'cosine':
            T_max = self.config.get('T_max', 50)
            return CosineAnnealingLR(self.optimizer, T_max=T_max)
        
        elif scheduler_name == 'reduce_on_plateau':
            factor = self.config.get('factor', 0.1)
            patience = self.config.get('scheduler_patience', 5)
            return ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience)
        
        else:
            return ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
    
    def _setup_criterion(self):
        """設置損失函數"""
        criterion_name = self.config.get('criterion', 'cross_entropy').lower()
        
        if criterion_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif criterion_name == 'focal':
            return FocalLoss(alpha=1, gamma=2)
        elif criterion_name == 'label_smoothing':
            smoothing = self.config.get('label_smoothing', 0.1)
            return LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            return nn.CrossEntropyLoss()
    
    def _check_and_fix_model_dimensions(self, inputs):
        """檢查並修復模型維度不匹配問題"""
        try:
            print(f"Checking model dimensions with input type: {type(inputs)}")
            
            # 完全複製 _process_input_features 的邏輯來獲取實際維度
            if isinstance(inputs, dict):
                print(f"Processing dictionary input with keys: {sorted(inputs.keys())}")
                feature_tensors = []
                for key in sorted(inputs.keys()):
                    feature = inputs[key]
                    print(f"Processing feature '{key}' with shape: {feature.shape if hasattr(feature, 'shape') else 'N/A'}")
                    
                    # 確保特徵是張量
                    if isinstance(feature, torch.Tensor):
                        # 如果是多維張量，展平除了批次維度
                        if feature.dim() > 2:
                            feature = feature.view(feature.size(0), -1)
                            print(f"  Flattened '{key}' to shape: {feature.shape}")
                        feature_tensors.append(feature)
                    else:
                        print(f"  Skipping non-tensor feature '{key}': {type(feature)}")
                        continue
                
                if not feature_tensors:
                    print("No valid feature tensors found")
                    return
                    
                # 計算拼接後的總維度
                individual_dims = [t.size(-1) for t in feature_tensors]
                actual_dim = sum(individual_dims)
                print(f"Individual feature dims: {individual_dims}, total: {actual_dim}")
            else:
                actual_dim = inputs.size(-1)
                print(f"Single tensor input dimension: {actual_dim}")
            
            # 檢查模型類型和重新初始化方法
            model_reinitialized = False
            
            # 處理MLP分類器
            if hasattr(self.model, 'reinitialize_for_input_dim'):
                expected_dim = getattr(self.model, '_expected_input_dim', None)
                print(f"MLP Classifier expects dimension: {expected_dim}, actual: {actual_dim}")
                
                model_reinitialized = self.model.reinitialize_for_input_dim(actual_dim)
                
            # 處理MultiModalFeatureFusion (nn.Sequential)
            elif hasattr(self.model, '__len__') and len(self.model) >= 1:
                # 檢查是否是Sequential模型，第一層是MultiModalFeatureFusion
                first_layer = self.model[0]
                if hasattr(first_layer, 'reinitialize_for_input_dims'):
                    actual_feature_dims = first_layer.calculate_actual_feature_dims(inputs)
                    print(f"MultiModalFeatureFusion expects dims: {first_layer.feature_dims}")
                    print(f"Actual feature dims: {actual_feature_dims}")
                    
                    fusion_reinitialized = first_layer.reinitialize_for_input_dims(actual_feature_dims)
                    
                    # 如果融合層重新初始化，可能也需要重新初始化分類器
                    if fusion_reinitialized and len(self.model) >= 2:
                        classifier = self.model[1]
                        if hasattr(classifier, 'reinitialize_for_input_dim'):
                            # 重新初始化分類器以匹配融合層輸出
                            fusion_output_dim = first_layer.fusion_dim
                            classifier.reinitialize_for_input_dim(fusion_output_dim)
                            print("Also reinitialized classifier for fusion output dimension")
                    
                    model_reinitialized = fusion_reinitialized
                else:
                    print(f"First layer type: {type(first_layer)}, does not support reinitialization")
            else:
                print(f"Model type: {type(self.model)}, does not support dimension reinitialization")
            
            if model_reinitialized:
                # 如果模型重新初始化，需要重新創建優化器
                print("Model reinitialized, recreating optimizer...")
                self.optimizer = self._setup_optimizer()
                self.scheduler = self._setup_scheduler()
                print("Optimizer and scheduler recreated successfully")
            else:
                print("Model dimensions already match, no reinitialization needed")
                    
        except Exception as e:
            print(f"Error in dimension checking: {e}")
            import traceback
            traceback.print_exc()

    def train_epoch(self) -> Dict[str, float]:
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        first_batch = True
        
        for batch in self.train_loader:
            # 移動數據到設備
            if isinstance(batch['features'], dict):
                # 處理特徵字典：將每個特徵張量移動到設備
                inputs = {key: tensor.to(self.device) for key, tensor in batch['features'].items()}
            else:
                # 處理單一張量
                inputs = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 在第一個批次檢查並修復維度不匹配
            if first_batch:
                self._check_and_fix_model_dimensions(inputs)
                first_batch = False
            
            # 前向傳播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 計算損失
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            loss = self.criterion(logits, labels)
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('gradient_clipping'):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
            
            self.optimizer.step()
            
            # 統計
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate_epoch(self) -> Dict[str, float]:
        """驗證一個epoch"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 移動數據到設備
                if isinstance(batch['features'], dict):
                    # 處理特徵字典：將每個特徵張量移動到設備
                    inputs = {key: tensor.to(self.device) for key, tensor in batch['features'].items()}
                else:
                    # 處理單一張量
                    inputs = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(inputs)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, epochs: int) -> Dict[str, List]:
        """執行完整訓練"""
        print(f"開始訓練，共 {epochs} 個 epoch")
        
        for epoch in range(epochs):
            print(f"\\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # 訓練階段
            train_metrics = self.train_epoch()
            print(f"訓練 - 損失: {train_metrics['loss']:.4f}, 準確率: {train_metrics['accuracy']:.4f}")
            
            # 驗證階段
            val_metrics = self.validate_epoch()
            print(f"驗證 - 損失: {val_metrics['loss']:.4f}, 準確率: {val_metrics['accuracy']:.4f}")
            
            # 更新歷史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # 學習率調度
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # 早停檢查
            if self.early_stopping(val_metrics['loss'], self.model):
                print(f"\\n早停觸發，在第 {epoch + 1} epoch 停止訓練")
                break
        
        return self.history
    
    def save_checkpoint(self, filepath: str):
        """保存檢查點"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"檢查點已保存到: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """載入檢查點"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        print(f"檢查點已從 {filepath} 載入")


class EarlyStopping:
    """早停類"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """檢查是否應該早停"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print("已恢復最佳權重")
            return True
        
        return False


class FocalLoss(nn.Module):
    """Focal Loss 實現"""
    
    def __init__(self, alpha: float = 1, gamma: float = 2, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """標籤平滑交叉熵損失"""
    
    def __init__(self, smoothing: float = 0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        log_prob = nn.functional.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size()) * self.smoothing / (inputs.size(-1) - 1.)
        weight.scatter_(-1, targets.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss