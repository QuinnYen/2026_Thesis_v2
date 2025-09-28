# 標準評估器模組
"""
標準評估器實現

提供以下評估功能：
- 準確率計算: 整體和類別準確率
- F1 分數計算: 宏平均、微平均、加權平均
- 精確率與召回率: 多類別精確率和召回率
- 混淆矩陣生成: 詳細的混淆矩陣分析
- ROC 和 AUC 計算: 接收者操作特性曲線
- 分類報告: 完整的分類性能報告
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


class StandardEvaluator:
    """標準評估器"""
    
    def __init__(self, class_labels: List[str] = None, save_plots: bool = True, 
                 plot_dir: str = "evaluation_plots"):
        """
        初始化標準評估器
        
        Args:
            class_labels: 類別標籤列表
            save_plots: 是否保存圖表
            plot_dir: 圖表保存目錄
        """
        self.class_labels = class_labels or ['負面', '中性', '正面']
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        
        if self.save_plots:
            self.plot_dir.mkdir(exist_ok=True)
        
        # 評估結果存儲
        self.evaluation_history = []
    
    def evaluate_predictions(self, y_true: Union[torch.Tensor, np.ndarray], 
                           y_pred: Union[torch.Tensor, np.ndarray],
                           y_prob: Union[torch.Tensor, np.ndarray] = None,
                           domain_name: str = "unknown") -> Dict[str, float]:
        """
        評估預測結果
        
        Args:
            y_true: 真實標籤 [batch_size]
            y_pred: 預測標籤 [batch_size]  
            y_prob: 預測機率 [batch_size, num_classes] (可選)
            domain_name: 領域名稱
            
        Returns:
            evaluation_metrics: 評估指標字典
        """
        # 轉換為 numpy 陣列
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
        
        # 確保標籤為整數類型
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        
        # 計算基本指標
        metrics = {}
        
        # 準確率
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 精確率、召回率、F1 分數（不同平均方式）
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        # 添加簡單的 precision 和 recall 鍵，使用宏平均作為默認值
        metrics['precision'] = metrics['precision_macro']
        metrics['recall'] = metrics['recall_macro']
        
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # 添加簡單的 f1 鍵，使用宏平均作為默認值
        metrics['f1'] = metrics['f1_macro']
        
        # 各類別的詳細指標
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, label in enumerate(self.class_labels):
            if i < len(precision_per_class):
                metrics[f'precision_{label}'] = precision_per_class[i]
                metrics[f'recall_{label}'] = recall_per_class[i]
                metrics[f'f1_{label}'] = f1_per_class[i]
        
        # 混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 如果有機率預測，計算 AUC 相關指標
        if y_prob is not None:
            try:
                # 多類別 AUC（一對其餘）
                if y_prob.shape[1] > 2:  # 多類別
                    metrics['auc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                    metrics['auc_weighted'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                else:  # 二分類
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                
                # 平均精確率分數
                metrics['average_precision_macro'] = average_precision_score(
                    np.eye(len(self.class_labels))[y_true], y_prob, average='macro'
                )
                metrics['average_precision_weighted'] = average_precision_score(
                    np.eye(len(self.class_labels))[y_true], y_prob, average='weighted'
                )
            except Exception as e:
                print(f"警告：無法計算 AUC 指標: {e}")
        
        # 分類報告
        try:
            class_report = classification_report(
                y_true, y_pred, 
                target_names=self.class_labels,
                output_dict=True,
                zero_division=0
            )
            metrics['classification_report'] = class_report
        except Exception as e:
            print(f"警告：無法生成分類報告: {e}")
        
        # 添加領域信息
        metrics['domain'] = domain_name
        metrics['total_samples'] = len(y_true)
        
        # 保存評估結果
        evaluation_result = {
            'domain': domain_name,
            'metrics': metrics,
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist() if y_prob is not None else None
        }
        self.evaluation_history.append(evaluation_result)
        
        # 生成圖表
        if self.save_plots:
            self._generate_plots(y_true, y_pred, y_prob, domain_name, metrics)
        
        return metrics
    
    def _generate_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_prob: np.ndarray, domain_name: str, metrics: Dict):
        """生成評估圖表"""
        try:
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 混淆矩陣熱力圖
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_labels, 
                       yticklabels=self.class_labels)
            plt.title(f'混淆矩陣 - {domain_name}')
            plt.xlabel('預測標籤')
            plt.ylabel('真實標籤')
            plt.tight_layout()
            plt.savefig(self.plot_dir / f'confusion_matrix_{domain_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 各類別性能柱狀圖
            if 'classification_report' in metrics:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                class_names = self.class_labels
                precision_values = [metrics.get(f'precision_{name}', 0) for name in class_names]
                recall_values = [metrics.get(f'recall_{name}', 0) for name in class_names]
                f1_values = [metrics.get(f'f1_{name}', 0) for name in class_names]
                
                # 精確率
                axes[0].bar(class_names, precision_values, color='skyblue', alpha=0.7)
                axes[0].set_title('各類別精確率')
                axes[0].set_ylabel('精確率')
                axes[0].set_ylim(0, 1)
                
                # 召回率
                axes[1].bar(class_names, recall_values, color='lightgreen', alpha=0.7)
                axes[1].set_title('各類別召回率')
                axes[1].set_ylabel('召回率')
                axes[1].set_ylim(0, 1)
                
                # F1 分數
                axes[2].bar(class_names, f1_values, color='salmon', alpha=0.7)
                axes[2].set_title('各類別 F1 分數')
                axes[2].set_ylabel('F1 分數')
                axes[2].set_ylim(0, 1)
                
                plt.suptitle(f'各類別性能指標 - {domain_name}')
                plt.tight_layout()
                plt.savefig(self.plot_dir / f'class_performance_{domain_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. ROC 曲線（如果有機率預測）
            if y_prob is not None:
                plt.figure(figsize=(10, 8))
                
                if y_prob.shape[1] == 2:  # 二分類
                    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                    auc_score = roc_auc_score(y_true, y_prob[:, 1])
                    plt.plot(fpr, tpr, label=f'ROC 曲線 (AUC = {auc_score:.3f})')
                else:  # 多類別
                    for i, class_name in enumerate(self.class_labels):
                        if i < y_prob.shape[1]:
                            # 一對其餘 ROC
                            y_true_binary = (y_true == i).astype(int)
                            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
                            auc_score = roc_auc_score(y_true_binary, y_prob[:, i])
                            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.3f})')
                
                plt.plot([0, 1], [0, 1], 'k--', label='隨機分類器')
                plt.xlabel('偽陽性率')
                plt.ylabel('真陽性率')
                plt.title(f'ROC 曲線 - {domain_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.plot_dir / f'roc_curve_{domain_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 4. 精確率-召回率曲線
                plt.figure(figsize=(10, 8))
                
                if y_prob.shape[1] == 2:  # 二分類
                    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                    ap_score = average_precision_score(y_true, y_prob[:, 1])
                    plt.plot(recall, precision, label=f'PR 曲線 (AP = {ap_score:.3f})')
                else:  # 多類別
                    for i, class_name in enumerate(self.class_labels):
                        if i < y_prob.shape[1]:
                            y_true_binary = (y_true == i).astype(int)
                            precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, i])
                            ap_score = average_precision_score(y_true_binary, y_prob[:, i])
                            plt.plot(recall, precision, label=f'{class_name} (AP = {ap_score:.3f})')
                
                plt.xlabel('召回率')
                plt.ylabel('精確率')
                plt.title(f'精確率-召回率曲線 - {domain_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.plot_dir / f'precision_recall_curve_{domain_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            print(f"警告：生成圖表時發生錯誤: {e}")
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        比較多個模型的性能
        
        Args:
            model_results: 模型結果字典 {model_name: metrics_dict}
            
        Returns:
            comparison_df: 比較結果 DataFrame
        """
        comparison_data = []
        
        for model_name, metrics in model_results.items():
            row = {'模型': model_name}
            
            # 主要指標
            row['準確率'] = metrics.get('accuracy', 0)
            row['F1 宏平均'] = metrics.get('f1_macro', 0)
            row['F1 微平均'] = metrics.get('f1_micro', 0)
            row['F1 加權平均'] = metrics.get('f1_weighted', 0)
            row['精確率 宏平均'] = metrics.get('precision_macro', 0)
            row['召回率 宏平均'] = metrics.get('recall_macro', 0)
            
            # AUC 指標（如果有）
            if 'auc_macro' in metrics:
                row['AUC 宏平均'] = metrics['auc_macro']
            if 'auc_weighted' in metrics:
                row['AUC 加權平均'] = metrics['auc_weighted']
            
            # 各類別 F1 分數
            for class_label in self.class_labels:
                f1_key = f'f1_{class_label}'
                if f1_key in metrics:
                    row[f'F1_{class_label}'] = metrics[f1_key]
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存比較結果
        if self.save_plots:
            comparison_df.to_csv(self.plot_dir / 'model_comparison.csv', index=False, encoding='utf-8-sig')
            
            # 生成比較圖表
            self._generate_comparison_plots(comparison_df)
        
        return comparison_df
    
    def _generate_comparison_plots(self, comparison_df: pd.DataFrame):
        """生成模型比較圖表"""
        try:
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 主要指標比較
            main_metrics = ['準確率', 'F1 宏平均', 'F1 微平均', 'F1 加權平均']
            available_metrics = [m for m in main_metrics if m in comparison_df.columns]
            
            if available_metrics:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for i, metric in enumerate(available_metrics[:4]):
                    if i < len(axes):
                        ax = axes[i]
                        ax.bar(comparison_df['模型'], comparison_df[metric], alpha=0.7)
                        ax.set_title(f'{metric} 比較')
                        ax.set_ylabel(metric)
                        ax.tick_params(axis='x', rotation=45)
                        
                        # 添加數值標籤
                        for j, v in enumerate(comparison_df[metric]):
                            ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(self.plot_dir / 'model_comparison_main_metrics.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 各類別 F1 分數比較
            f1_columns = [col for col in comparison_df.columns if col.startswith('F1_')]
            if f1_columns:
                plt.figure(figsize=(12, 8))
                
                x = range(len(comparison_df))
                width = 0.8 / len(f1_columns)
                
                for i, col in enumerate(f1_columns):
                    class_name = col.replace('F1_', '')
                    plt.bar([pos + i * width for pos in x], comparison_df[col], 
                           width, label=class_name, alpha=0.7)
                
                plt.xlabel('模型')
                plt.ylabel('F1 分數')
                plt.title('各模型各類別 F1 分數比較')
                plt.xticks([pos + width * (len(f1_columns) - 1) / 2 for pos in x], 
                          comparison_df['模型'], rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.plot_dir / 'model_comparison_class_f1.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"警告：生成比較圖表時發生錯誤: {e}")
    
    def generate_summary_report(self) -> Dict[str, any]:
        """生成評估總結報告"""
        if not self.evaluation_history:
            return {"error": "沒有評估歷史記錄"}
        
        summary = {
            "評估總數": len(self.evaluation_history),
            "領域列表": list(set([result['domain'] for result in self.evaluation_history])),
            "平均指標": {},
            "最佳性能": {},
            "詳細結果": []
        }
        
        # 計算平均指標
        all_metrics = {}
        for result in self.evaluation_history:
            metrics = result['metrics']
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        # 計算平均值和標準差
        for key, values in all_metrics.items():
            if values:
                summary["平均指標"][key] = {
                    "平均值": np.mean(values),
                    "標準差": np.std(values),
                    "最小值": np.min(values),
                    "最大值": np.max(values)
                }
        
        # 找出最佳性能
        best_accuracy = max(self.evaluation_history, 
                          key=lambda x: x['metrics'].get('accuracy', 0))
        best_f1_macro = max(self.evaluation_history, 
                          key=lambda x: x['metrics'].get('f1_macro', 0))
        
        summary["最佳性能"] = {
            "最佳準確率": {
                "領域": best_accuracy['domain'],
                "準確率": best_accuracy['metrics'].get('accuracy', 0),
                "F1_宏平均": best_accuracy['metrics'].get('f1_macro', 0)
            },
            "最佳F1宏平均": {
                "領域": best_f1_macro['domain'],
                "準確率": best_f1_macro['metrics'].get('accuracy', 0),
                "F1_宏平均": best_f1_macro['metrics'].get('f1_macro', 0)
            }
        }
        
        # 詳細結果
        for result in self.evaluation_history:
            domain_summary = {
                "領域": result['domain'],
                "樣本數": result['metrics'].get('total_samples', 0),
                "準確率": result['metrics'].get('accuracy', 0),
                "F1_宏平均": result['metrics'].get('f1_macro', 0),
                "F1_微平均": result['metrics'].get('f1_micro', 0),
                "F1_加權平均": result['metrics'].get('f1_weighted', 0)
            }
            summary["詳細結果"].append(domain_summary)
        
        # 保存報告
        if self.save_plots:
            with open(self.plot_dir / 'evaluation_summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        return summary
    
    def save_evaluation_history(self, filepath: str):
        """保存評估歷史"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_history, f, ensure_ascii=False, indent=2, default=str)
    
    def load_evaluation_history(self, filepath: str):
        """載入評估歷史"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.evaluation_history = json.load(f)
    
    def reset_history(self):
        """重置評估歷史"""
        self.evaluation_history = []


class MetricCalculator:
    """指標計算器"""
    
    @staticmethod
    def calculate_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                          num_classes: int = 3) -> float:
        """計算宏平均 F1 分數"""
        f1_scores = []
        for class_idx in range(num_classes):
            # 將多類別問題轉換為二分類
            true_binary = (y_true == class_idx).astype(int)
            pred_binary = (y_pred == class_idx).astype(int)
            
            # 計算 F1 分數
            tp = np.sum((true_binary == 1) & (pred_binary == 1))
            fp = np.sum((true_binary == 0) & (pred_binary == 1))
            fn = np.sum((true_binary == 1) & (pred_binary == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    @staticmethod
    def calculate_weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 num_classes: int = 3) -> Dict[str, float]:
        """計算加權指標"""
        metrics = {'precision': 0, 'recall': 0, 'f1': 0}
        class_weights = []
        class_metrics = {'precision': [], 'recall': [], 'f1': []}
        
        for class_idx in range(num_classes):
            # 計算類別權重（基於支援度）
            class_support = np.sum(y_true == class_idx)
            class_weights.append(class_support)
            
            # 二分類指標
            true_binary = (y_true == class_idx).astype(int)
            pred_binary = (y_pred == class_idx).astype(int)
            
            tp = np.sum((true_binary == 1) & (pred_binary == 1))
            fp = np.sum((true_binary == 0) & (pred_binary == 1))
            fn = np.sum((true_binary == 1) & (pred_binary == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics['precision'].append(precision)
            class_metrics['recall'].append(recall)
            class_metrics['f1'].append(f1)
        
        # 計算加權平均
        total_samples = len(y_true)
        for metric_name in metrics.keys():
            weighted_sum = sum(w * m for w, m in zip(class_weights, class_metrics[metric_name]))
            metrics[metric_name] = weighted_sum / total_samples if total_samples > 0 else 0
        
        return metrics
    
    @staticmethod
    def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """計算信賴區間"""
        if len(scores) < 2:
            return (0, 0)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)
        n = len(scores)
        
        # t 分佈臨界值（近似）
        if confidence == 0.95:
            t_critical = 1.96  # 近似值，適用於大樣本
        elif confidence == 0.99:
            t_critical = 2.576
        else:
            t_critical = 1.645  # 90% 信賴區間
        
        margin_error = t_critical * (std_score / np.sqrt(n))
        
        return (mean_score - margin_error, mean_score + margin_error)


class PerformanceMonitor:
    """性能監控器"""
    
    def __init__(self, window_size: int = 100):
        """
        初始化性能監控器
        
        Args:
            window_size: 滑動視窗大小
        """
        self.window_size = window_size
        self.accuracy_history = []
        self.f1_history = []
        self.loss_history = []
    
    def update(self, accuracy: float, f1_score: float, loss: float = None):
        """更新性能指標"""
        self.accuracy_history.append(accuracy)
        self.f1_history.append(f1_score)
        
        if loss is not None:
            self.loss_history.append(loss)
        
        # 保持視窗大小
        if len(self.accuracy_history) > self.window_size:
            self.accuracy_history.pop(0)
            self.f1_history.pop(0)
            if self.loss_history:
                self.loss_history.pop(0)
    
    def get_recent_performance(self) -> Dict[str, float]:
        """獲取近期性能"""
        if not self.accuracy_history:
            return {}
        
        recent_window = min(10, len(self.accuracy_history))
        
        performance = {
            'recent_accuracy_mean': np.mean(self.accuracy_history[-recent_window:]),
            'recent_f1_mean': np.mean(self.f1_history[-recent_window:]),
            'recent_accuracy_std': np.std(self.accuracy_history[-recent_window:]),
            'recent_f1_std': np.std(self.f1_history[-recent_window:]),
        }
        
        if self.loss_history:
            performance['recent_loss_mean'] = np.mean(self.loss_history[-recent_window:])
            performance['recent_loss_std'] = np.std(self.loss_history[-recent_window:])
        
        return performance
    
    def detect_performance_degradation(self, threshold: float = 0.05) -> bool:
        """檢測性能退化"""
        if len(self.accuracy_history) < 20:
            return False
        
        # 比較前半和後半的平均性能
        mid_point = len(self.accuracy_history) // 2
        first_half_acc = np.mean(self.accuracy_history[:mid_point])
        second_half_acc = np.mean(self.accuracy_history[mid_point:])
        
        first_half_f1 = np.mean(self.f1_history[:mid_point])
        second_half_f1 = np.mean(self.f1_history[mid_point:])
        
        # 如果後半段性能明顯下降，則判定為性能退化
        acc_degradation = (first_half_acc - second_half_acc) > threshold
        f1_degradation = (first_half_f1 - second_half_f1) > threshold
        
        return acc_degradation or f1_degradation