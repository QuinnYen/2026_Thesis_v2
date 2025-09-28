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
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation.standard_evaluator import StandardEvaluator


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

        # 評估器設置 (用於計算F1分數等詳細指標)
        # 動態確定類別標籤
        class_labels = config.get('class_labels', ['negative', 'neutral', 'positive'])
        self.evaluator = StandardEvaluator(class_labels=class_labels)
        
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
            # 計算實際輸入維度（移除詳細除錯輸出）
            if isinstance(inputs, dict):
                feature_tensors = []
                for key in sorted(inputs.keys()):
                    feature = inputs[key]
                    
                    # 確保特徵是張量
                    if isinstance(feature, torch.Tensor):
                        # 如果是多維張量，展平除了批次維度
                        if feature.dim() > 2:
                            feature = feature.view(feature.size(0), -1)
                        feature_tensors.append(feature)
                    else:
                        continue
                
                if not feature_tensors:
                    return
                    
                # 計算拼接後的總維度
                individual_dims = [t.size(-1) for t in feature_tensors]
                actual_dim = sum(individual_dims)
            else:
                actual_dim = inputs.size(-1)
            
            # 檢查模型類型和重新初始化方法
            model_reinitialized = False
            
            # 檢查並重新初始化模型（僅在需要時顯示訊息）
            if hasattr(self.model, 'reinitialize_for_input_dim'):
                model_reinitialized = self.model.reinitialize_for_input_dim(actual_dim)
                # 強制更新預期維度
                if hasattr(self.model, '_expected_input_dim'):
                    self.model._expected_input_dim = actual_dim
                
            elif hasattr(self.model, '__len__') and len(self.model) >= 1:
                # 處理Sequential模型，第一層是MultiModalFeatureFusion
                first_layer = self.model[0]
                if hasattr(first_layer, 'reinitialize_for_input_dims'):
                    actual_feature_dims = first_layer.calculate_actual_feature_dims(inputs)
                    fusion_reinitialized = first_layer.reinitialize_for_input_dims(actual_feature_dims)
                    
                    # 如果融合層重新初始化，同時重新初始化分類器
                    if fusion_reinitialized and len(self.model) >= 2:
                        classifier = self.model[1]
                        if hasattr(classifier, 'reinitialize_for_input_dim'):
                            fusion_output_dim = first_layer.fusion_dim
                            classifier.reinitialize_for_input_dim(fusion_output_dim)
                    
                    model_reinitialized = fusion_reinitialized
                else:
                    model_reinitialized = False
            else:
                model_reinitialized = False
            
            if model_reinitialized:
                # 重新初始化優化器和排程器
                self.optimizer = self._setup_optimizer()
                self.scheduler = self._setup_scheduler()
            else:
                # 除錯：顯示為什麼沒有重新初始化
                expected_dim = getattr(self.model, '_expected_input_dim', '未知')
                    
        except Exception as e:
            pass  # 靜默處理維度檢查錯誤

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
            try:
                outputs = self.model(inputs)
            except RuntimeError as e:
                if "Input dimension mismatch" in str(e) or "輸入維度不匹配" in str(e):
                    # 如果維度檢查失敗，再次嘗試修復
                    self._check_and_fix_model_dimensions(inputs)
                    # 確保模型在正確設備上
                    self.model = self.model.to(self.device)
                    try:
                        outputs = self.model(inputs)
                    except RuntimeError as retry_e:
                        # 靜默處理重試失敗
                        raise retry_e
                else:
                    raise e
            
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

        # 收集所有預測和真實標籤用於計算詳細指標
        all_predictions = []
        all_labels = []

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

                # 收集預測和標籤
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples

        # 使用評估器計算詳細指標
        try:
            detailed_metrics = self.evaluator.evaluate_predictions(
                y_true=np.array(all_labels),
                y_pred=np.array(all_predictions)
            )

            # 合併基本指標和詳細指標
            result = {
                'loss': avg_loss,
                'accuracy': accuracy,
                **detailed_metrics  # 包含 f1, f1_macro, f1_micro, f1_weighted 等
            }

        except Exception as e:
            # 如果評估器出錯，回退到基本指標
            print(f"警告：評估器計算失敗: {e}")
            result = {'loss': avg_loss, 'accuracy': accuracy}

        return result
    
    def train(self, epochs: int) -> Dict[str, List]:
        """執行完整訓練"""

        # 使用進度條顯示訓練進度
        pbar = tqdm(range(epochs), desc="訓練進度", unit="epoch")

        for epoch in pbar:
            # 訓練階段
            train_metrics = self.train_epoch()

            # 驗證階段
            val_metrics = self.validate_epoch()

            # 更新進度條顯示
            pbar.set_postfix({
                'Train Loss': f"{train_metrics['loss']:.4f}",
                'Train Acc': f"{train_metrics['accuracy']:.4f}",
                'Val Loss': f"{val_metrics['loss']:.4f}",
                'Val Acc': f"{val_metrics['accuracy']:.4f}"
            })
            
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