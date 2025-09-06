# 錯誤分析器模組
"""
錯誤分析器實現

提供以下錯誤分析功能：
- 錯誤樣本分析: 識別模型預測錯誤的樣本並分析錯誤模式
- 混淆分析: 深度分析類別混淆情況
- 困難樣本識別: 找出模型難以正確分類的樣本
- 錯誤歸因: 分析錯誤的可能原因
- 改進建議: 基於錯誤分析提供模型改進建議
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


class ErrorAnalyzer:
    """錯誤分析器"""
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "error_analysis_plots"):
        """
        初始化錯誤分析器
        
        Args:
            save_plots: 是否保存圖表
            plot_dir: 圖表保存目錄
        """
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        
        if self.save_plots:
            self.plot_dir.mkdir(exist_ok=True)
        
        # 分析結果存儲
        self.analysis_results = []
        
        # 情感極性標籤映射
        self.sentiment_labels = {
            0: "負面",
            1: "中性", 
            2: "正面",
            -1: "負面",  # 某些數據集使用 -1, 0, 1
            "negative": "負面",
            "neutral": "中性",
            "positive": "正面"
        }
    
    def analyze_prediction_errors(self, y_true: Union[torch.Tensor, np.ndarray, List],
                                y_pred: Union[torch.Tensor, np.ndarray, List],
                                y_prob: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
                                texts: Optional[List[str]] = None,
                                domain: str = "unknown") -> Dict[str, Any]:
        """
        分析預測錯誤
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤  
            y_prob: 預測機率（可選）
            texts: 文本內容（可選）
            domain: 領域名稱
            
        Returns:
            error_analysis: 錯誤分析結果
        """
        print(f"進行錯誤分析 - {domain}")
        
        # 轉換為numpy數組
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 識別錯誤樣本
        error_mask = y_true != y_pred
        error_indices = np.where(error_mask)[0]
        correct_indices = np.where(~error_mask)[0]
        
        results = {
            "domain": domain,
            "total_samples": len(y_true),
            "error_samples": len(error_indices),
            "correct_samples": len(correct_indices),
            "error_rate": len(error_indices) / len(y_true),
            "accuracy": len(correct_indices) / len(y_true),
            "error_analysis": {},
            "confusion_analysis": {},
            "difficult_samples": {},
            "confidence_analysis": {}
        }
        
        # 1. 錯誤類型分析
        error_types = self._analyze_error_types(y_true, y_pred, error_indices)
        results["error_analysis"] = error_types
        
        # 2. 混淆矩陣分析
        confusion_analysis = self._analyze_confusion_matrix(y_true, y_pred)
        results["confusion_analysis"] = confusion_analysis
        
        # 3. 信心度分析（如果提供機率）
        if y_prob is not None:
            confidence_analysis = self._analyze_prediction_confidence(
                y_true, y_pred, y_prob, error_indices
            )
            results["confidence_analysis"] = confidence_analysis
        
        # 4. 困難樣本識別
        difficult_samples = self._identify_difficult_samples(
            y_true, y_pred, y_prob, error_indices, texts
        )
        results["difficult_samples"] = difficult_samples
        
        # 5. 錯誤分佈分析
        error_distribution = self._analyze_error_distribution(y_true, y_pred, error_indices)
        results["error_distribution"] = error_distribution
        
        # 6. 改進建議
        improvement_suggestions = self._generate_improvement_suggestions(results)
        results["improvement_suggestions"] = improvement_suggestions
        
        # 保存結果
        self.analysis_results.append({
            "analysis_type": "prediction_errors",
            "domain": domain,
            "results": results,
            "timestamp": pd.Timestamp.now()
        })
        
        # 生成視覺化
        if self.save_plots:
            self._plot_error_analysis(results, domain)
        
        return results
    
    def compare_model_errors(self, models_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        比較多個模型的錯誤模式
        
        Args:
            models_results: 模型結果 {model_name: {y_true, y_pred, y_prob, texts}}
            
        Returns:
            comparison_results: 比較分析結果
        """
        print("進行多模型錯誤比較分析")
        
        results = {
            "models": list(models_results.keys()),
            "error_rates": {},
            "common_errors": {},
            "unique_errors": {},
            "error_overlap": {},
            "complementary_analysis": {}
        }
        
        # 計算各模型錯誤率
        model_errors = {}
        for model_name, model_data in models_results.items():
            y_true = np.array(model_data["y_true"])
            y_pred = np.array(model_data["y_pred"])
            error_mask = y_true != y_pred
            error_indices = set(np.where(error_mask)[0])
            
            model_errors[model_name] = error_indices
            results["error_rates"][model_name] = len(error_indices) / len(y_true)
        
        # 分析錯誤重疊
        model_names = list(models_results.keys())
        
        # 兩兩比較
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                errors1 = model_errors[model1]
                errors2 = model_errors[model2]
                
                # 共同錯誤
                common = errors1.intersection(errors2)
                # 獨特錯誤
                unique1 = errors1 - errors2
                unique2 = errors2 - errors1
                
                pair_key = f"{model1}_vs_{model2}"
                results["error_overlap"][pair_key] = {
                    "common_errors": len(common),
                    "unique_to_model1": len(unique1),
                    "unique_to_model2": len(unique2),
                    "overlap_ratio": len(common) / len(errors1.union(errors2)) if errors1.union(errors2) else 0
                }
        
        # 全體共同錯誤和獨特錯誤
        if len(model_names) >= 2:
            all_error_sets = list(model_errors.values())
            common_to_all = set.intersection(*all_error_sets) if all_error_sets else set()
            
            results["common_errors"] = {
                "indices": list(common_to_all),
                "count": len(common_to_all),
                "examples": self._get_error_examples(common_to_all, models_results)
            }
            
            # 每個模型的獨特錯誤
            for model_name in model_names:
                other_models_errors = set()
                for other_model in model_names:
                    if other_model != model_name:
                        other_models_errors.update(model_errors[other_model])
                
                unique_to_model = model_errors[model_name] - other_models_errors
                results["unique_errors"][model_name] = {
                    "indices": list(unique_to_model),
                    "count": len(unique_to_model),
                    "examples": self._get_error_examples(unique_to_model, {model_name: models_results[model_name]})
                }
        
        # 模型互補性分析
        complementary_analysis = self._analyze_model_complementarity(model_errors, models_results)
        results["complementary_analysis"] = complementary_analysis
        
        # 保存結果
        self.analysis_results.append({
            "analysis_type": "model_error_comparison",
            "results": results,
            "timestamp": pd.Timestamp.now()
        })
        
        # 生成視覺化
        if self.save_plots:
            self._plot_model_error_comparison(results)
        
        return results
    
    def analyze_cross_domain_errors(self, source_results: Dict[str, Any], 
                                  target_results: Dict[str, Any],
                                  domain_pair: Tuple[str, str]) -> Dict[str, Any]:
        """
        分析跨領域錯誤模式
        
        Args:
            source_results: 源領域結果 {y_true, y_pred, y_prob, texts}
            target_results: 目標領域結果 {y_true, y_pred, y_prob, texts}
            domain_pair: 領域對 (source, target)
            
        Returns:
            cross_domain_analysis: 跨領域錯誤分析結果
        """
        print(f"進行跨領域錯誤分析: {domain_pair[0]} -> {domain_pair[1]}")
        
        # 分析各領域錯誤
        source_analysis = self.analyze_prediction_errors(
            source_results["y_true"], source_results["y_pred"], 
            source_results.get("y_prob"), source_results.get("texts"),
            domain=domain_pair[0]
        )
        
        target_analysis = self.analyze_prediction_errors(
            target_results["y_true"], target_results["y_pred"],
            target_results.get("y_prob"), target_results.get("texts"), 
            domain=domain_pair[1]
        )
        
        results = {
            "domain_pair": domain_pair,
            "source_analysis": source_analysis,
            "target_analysis": target_analysis,
            "transfer_effects": {},
            "domain_specific_errors": {},
            "adaptation_insights": {}
        }
        
        # 1. 遷移效果分析
        transfer_effects = {
            "accuracy_drop": target_analysis["accuracy"] - source_analysis["accuracy"],
            "error_rate_increase": target_analysis["error_rate"] - source_analysis["error_rate"],
            "confusion_pattern_change": self._compare_confusion_patterns(
                source_analysis["confusion_analysis"], 
                target_analysis["confusion_analysis"]
            )
        }
        results["transfer_effects"] = transfer_effects
        
        # 2. 領域特異性錯誤
        domain_specific = self._identify_domain_specific_errors(
            source_analysis, target_analysis
        )
        results["domain_specific_errors"] = domain_specific
        
        # 3. 適應性洞察
        adaptation_insights = self._generate_adaptation_insights(
            source_analysis, target_analysis, transfer_effects
        )
        results["adaptation_insights"] = adaptation_insights
        
        # 保存結果
        self.analysis_results.append({
            "analysis_type": "cross_domain_errors",
            "domain_pair": domain_pair,
            "results": results,
            "timestamp": pd.Timestamp.now()
        })
        
        # 生成視覺化
        if self.save_plots:
            self._plot_cross_domain_error_analysis(results)
        
        return results
    
    def cluster_error_samples(self, features: Union[torch.Tensor, np.ndarray],
                            y_true: Union[torch.Tensor, np.ndarray],
                            y_pred: Union[torch.Tensor, np.ndarray],
                            n_clusters: int = 5) -> Dict[str, Any]:
        """
        對錯誤樣本進行聚類分析
        
        Args:
            features: 特徵向量
            y_true: 真實標籤
            y_pred: 預測標籤
            n_clusters: 聚類數量
            
        Returns:
            clustering_results: 聚類分析結果
        """
        print("對錯誤樣本進行聚類分析")
        
        # 轉換為numpy數組
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        # 識別錯誤樣本
        error_mask = y_true != y_pred
        error_indices = np.where(error_mask)[0]
        
        if len(error_indices) < n_clusters:
            return {"error": f"錯誤樣本數量 ({len(error_indices)}) 少於聚類數量 ({n_clusters})"}
        
        error_features = features[error_indices]
        error_true_labels = y_true[error_indices]
        error_pred_labels = y_pred[error_indices]
        
        results = {
            "n_error_samples": len(error_indices),
            "n_clusters": n_clusters,
            "cluster_analysis": {},
            "dimensional_reduction": {},
            "cluster_characteristics": {}
        }
        
        # 1. K-means聚類
        try:
            # 降維以提高聚類效果
            if error_features.shape[1] > 50:
                pca = PCA(n_components=50)
                reduced_features = pca.fit_transform(error_features)
                results["dimensional_reduction"]["pca_variance_ratio"] = pca.explained_variance_ratio_[:10].tolist()
            else:
                reduced_features = error_features
            
            # 執行聚類
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(reduced_features)
            
            results["cluster_analysis"] = {
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "inertia": kmeans.inertia_,
                "cluster_labels": cluster_labels.tolist()
            }
            
            # 2. 分析每個聚類的特徵
            cluster_characteristics = {}
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = error_indices[cluster_mask]
                
                # 聚類內樣本的真實標籤和預測標籤分佈
                cluster_true = error_true_labels[cluster_mask]
                cluster_pred = error_pred_labels[cluster_mask]
                
                cluster_characteristics[f"cluster_{cluster_id}"] = {
                    "size": np.sum(cluster_mask),
                    "proportion": np.sum(cluster_mask) / len(error_indices),
                    "true_label_distribution": dict(Counter(cluster_true)),
                    "pred_label_distribution": dict(Counter(cluster_pred)),
                    "common_error_types": self._analyze_cluster_error_types(
                        cluster_true, cluster_pred
                    ),
                    "sample_indices": cluster_indices.tolist()[:10]  # 前10個樣本索引
                }
            
            results["cluster_characteristics"] = cluster_characteristics
            
            # 3. t-SNE降維視覺化
            try:
                if len(error_features) > 5:  # 至少需要5個樣本才能進行t-SNE
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(error_features)-1))
                    tsne_coords = tsne.fit_transform(reduced_features)
                    results["dimensional_reduction"]["tsne_coords"] = tsne_coords.tolist()
            except:
                pass
                
        except Exception as e:
            results["clustering_error"] = str(e)
        
        # 保存結果
        self.analysis_results.append({
            "analysis_type": "error_clustering",
            "results": results,
            "timestamp": pd.Timestamp.now()
        })
        
        # 生成視覺化
        if self.save_plots:
            self._plot_error_clustering(results)
        
        return results
    
    # ===== 輔助方法 =====
    
    def _analyze_error_types(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           error_indices: np.ndarray) -> Dict[str, Any]:
        """分析錯誤類型"""
        error_types = {}
        
        # 計算各類錯誤的數量
        for true_label in np.unique(y_true):
            for pred_label in np.unique(y_pred):
                if true_label != pred_label:
                    error_mask = (y_true == true_label) & (y_pred == pred_label)
                    error_count = np.sum(error_mask)
                    
                    if error_count > 0:
                        error_key = f"{self._get_label_name(true_label)}_to_{self._get_label_name(pred_label)}"
                        error_types[error_key] = {
                            "count": int(error_count),
                            "true_label": int(true_label),
                            "pred_label": int(pred_label),
                            "proportion": error_count / len(error_indices) if len(error_indices) > 0 else 0
                        }
        
        return error_types
    
    def _analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """分析混淆矩陣"""
        cm = confusion_matrix(y_true, y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        
        # 計算各類別的精確率、召回率
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        
        return {
            "confusion_matrix": cm.tolist(),
            "labels": labels.tolist(),
            "classification_report": report,
            "most_confused_pair": self._find_most_confused_pair(cm, labels)
        }
    
    def _find_most_confused_pair(self, cm: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """找出最容易混淆的類別對"""
        max_confusion = 0
        confused_pair = None
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > max_confusion:
                    max_confusion = cm[i, j]
                    confused_pair = (labels[i], labels[j])
        
        if confused_pair:
            return {
                "true_label": int(confused_pair[0]),
                "pred_label": int(confused_pair[1]),
                "confusion_count": int(max_confusion),
                "description": f"{self._get_label_name(confused_pair[0])} 最常被誤判為 {self._get_label_name(confused_pair[1])}"
            }
        return {}
    
    def _analyze_prediction_confidence(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_prob: np.ndarray, error_indices: np.ndarray) -> Dict[str, Any]:
        """分析預測信心度"""
        # 獲取預測機率的最大值作為信心度
        if len(y_prob.shape) > 1:
            confidence_scores = np.max(y_prob, axis=1)
        else:
            confidence_scores = y_prob
        
        error_confidence = confidence_scores[error_indices]
        correct_indices = np.setdiff1d(np.arange(len(y_true)), error_indices)
        correct_confidence = confidence_scores[correct_indices]
        
        return {
            "error_confidence": {
                "mean": float(np.mean(error_confidence)),
                "std": float(np.std(error_confidence)),
                "median": float(np.median(error_confidence)),
                "min": float(np.min(error_confidence)),
                "max": float(np.max(error_confidence))
            },
            "correct_confidence": {
                "mean": float(np.mean(correct_confidence)),
                "std": float(np.std(correct_confidence)),
                "median": float(np.median(correct_confidence)),
                "min": float(np.min(correct_confidence)),
                "max": float(np.max(correct_confidence))
            },
            "confidence_difference": float(np.mean(correct_confidence) - np.mean(error_confidence)),
            "low_confidence_errors": int(np.sum(error_confidence < 0.6)),
            "high_confidence_errors": int(np.sum(error_confidence > 0.8))
        }
    
    def _identify_difficult_samples(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_prob: Optional[np.ndarray], error_indices: np.ndarray,
                                  texts: Optional[List[str]]) -> Dict[str, Any]:
        """識別困難樣本"""
        difficult_samples = {}
        
        if y_prob is not None and len(y_prob.shape) > 1:
            # 基於預測機率識別困難樣本
            confidence_scores = np.max(y_prob, axis=1)
            
            # 低信心度但正確的樣本（邊界樣本）
            correct_mask = y_true == y_pred
            low_confidence_correct = np.where(correct_mask & (confidence_scores < 0.6))[0]
            
            # 高信心度但錯誤的樣本（誤導性樣本）  
            high_confidence_errors = np.where(~correct_mask & (confidence_scores > 0.8))[0]
            
            difficult_samples = {
                "low_confidence_correct": {
                    "count": len(low_confidence_correct),
                    "indices": low_confidence_correct.tolist()[:20],  # 前20個
                    "description": "模型對這些樣本信心度低但預測正確，可能是邊界樣本"
                },
                "high_confidence_errors": {
                    "count": len(high_confidence_errors),
                    "indices": high_confidence_errors.tolist()[:20],  # 前20個
                    "description": "模型對這些樣本信心度高但預測錯誤，可能存在標籤問題或特殊情況"
                }
            }
            
            # 如果有文本，添加樣本內容
            if texts:
                for category in ["low_confidence_correct", "high_confidence_errors"]:
                    indices = difficult_samples[category]["indices"][:5]  # 只取前5個
                    samples = []
                    for idx in indices:
                        if idx < len(texts):
                            samples.append({
                                "index": int(idx),
                                "text": texts[idx][:200] + "..." if len(texts[idx]) > 200 else texts[idx],
                                "true_label": self._get_label_name(y_true[idx]),
                                "pred_label": self._get_label_name(y_pred[idx]),
                                "confidence": float(confidence_scores[idx])
                            })
                    difficult_samples[category]["examples"] = samples
        
        return difficult_samples
    
    def _analyze_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  error_indices: np.ndarray) -> Dict[str, Any]:
        """分析錯誤分佈"""
        labels = np.unique(y_true)
        distribution = {}
        
        for label in labels:
            label_mask = y_true == label
            label_errors = np.sum(error_indices[np.isin(error_indices, np.where(label_mask)[0])] >= 0)
            label_total = np.sum(label_mask)
            
            distribution[self._get_label_name(label)] = {
                "total_samples": int(label_total),
                "error_samples": int(label_errors),
                "error_rate": label_errors / label_total if label_total > 0 else 0,
                "proportion_of_all_errors": label_errors / len(error_indices) if len(error_indices) > 0 else 0
            }
        
        return distribution
    
    def _generate_improvement_suggestions(self, results: Dict[str, Any]) -> List[str]:
        """生成改進建議"""
        suggestions = []
        
        # 基於錯誤率
        if results["error_rate"] > 0.3:
            suggestions.append("錯誤率較高(>30%)，建議考慮增加訓練數據或調整模型架構")
        elif results["error_rate"] > 0.2:
            suggestions.append("錯誤率偏高(>20%)，建議優化超參數或使用更好的特徵工程")
        
        # 基於混淆矩陣
        confusion_analysis = results.get("confusion_analysis", {})
        most_confused = confusion_analysis.get("most_confused_pair", {})
        if most_confused:
            suggestions.append(f"最常見錯誤: {most_confused.get('description', '')}，建議增加這類樣本的訓練數據")
        
        # 基於信心度分析
        confidence_analysis = results.get("confidence_analysis", {})
        if confidence_analysis:
            high_conf_errors = confidence_analysis.get("high_confidence_errors", 0)
            if high_conf_errors > results["error_samples"] * 0.2:
                suggestions.append("存在較多高信心度錯誤，建議檢查數據質量或模型過度自信問題")
        
        # 基於困難樣本
        difficult_samples = results.get("difficult_samples", {})
        if difficult_samples.get("low_confidence_correct", {}).get("count", 0) > 10:
            suggestions.append("存在許多低信心度但正確的預測，建議使用集成方法或增強模型信心度")
        
        if not suggestions:
            suggestions.append("模型表現良好，可考慮在更多數據上進一步驗證")
        
        return suggestions
    
    def _get_error_examples(self, error_indices: set, models_results: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """獲取錯誤樣本例子"""
        examples = []
        
        for model_name, model_data in models_results.items():
            texts = model_data.get("texts", [])
            y_true = np.array(model_data["y_true"])
            y_pred = np.array(model_data["y_pred"])
            
            for idx in list(error_indices)[:5]:  # 只取前5個
                if idx < len(texts) and texts:
                    examples.append({
                        "model": model_name,
                        "index": int(idx),
                        "text": texts[idx][:150] + "..." if len(texts[idx]) > 150 else texts[idx],
                        "true_label": self._get_label_name(y_true[idx]),
                        "pred_label": self._get_label_name(y_pred[idx])
                    })
                    break  # 每個錯誤索引只取一個例子
                    
        return examples
    
    def _analyze_model_complementarity(self, model_errors: Dict[str, set], 
                                     models_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """分析模型互補性"""
        model_names = list(model_errors.keys())
        
        if len(model_names) < 2:
            return {"error": "需要至少兩個模型才能分析互補性"}
        
        # 計算模型組合的潛在改進
        total_samples = len(list(models_results.values())[0]["y_true"])
        union_errors = set.union(*model_errors.values())
        
        complementarity = {
            "individual_error_rates": {
                name: len(errors) / total_samples 
                for name, errors in model_errors.items()
            },
            "union_error_rate": len(union_errors) / total_samples,
            "potential_ensemble_improvement": {},
            "best_combination": None
        }
        
        # 分析最佳二元組合
        best_combo_error_rate = float('inf')
        best_combo = None
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                # 簡單多數投票的錯誤（兩個模型都錯才算錯）
                intersection_errors = model_errors[model1].intersection(model_errors[model2])
                combo_error_rate = len(intersection_errors) / total_samples
                
                complementarity["potential_ensemble_improvement"][f"{model1}+{model2}"] = {
                    "individual_avg_error": (len(model_errors[model1]) + len(model_errors[model2])) / (2 * total_samples),
                    "ensemble_error": combo_error_rate,
                    "improvement": (len(model_errors[model1]) + len(model_errors[model2])) / (2 * total_samples) - combo_error_rate
                }
                
                if combo_error_rate < best_combo_error_rate:
                    best_combo_error_rate = combo_error_rate
                    best_combo = f"{model1}+{model2}"
        
        complementarity["best_combination"] = {
            "models": best_combo,
            "error_rate": best_combo_error_rate,
            "improvement_over_best_individual": min(complementarity["individual_error_rates"].values()) - best_combo_error_rate
        }
        
        return complementarity
    
    def _compare_confusion_patterns(self, source_confusion: Dict[str, Any], 
                                  target_confusion: Dict[str, Any]) -> Dict[str, Any]:
        """比較混淆模式"""
        source_cm = np.array(source_confusion["confusion_matrix"])
        target_cm = np.array(target_confusion["confusion_matrix"])
        
        # 確保矩陣大小相同
        if source_cm.shape != target_cm.shape:
            return {"error": "混淆矩陣維度不同"}
        
        # 計算混淆模式變化
        cm_difference = target_cm - source_cm
        
        return {
            "confusion_change": cm_difference.tolist(),
            "major_changes": self._identify_major_confusion_changes(cm_difference, source_confusion["labels"])
        }
    
    def _identify_major_confusion_changes(self, cm_diff: np.ndarray, labels: List[int]) -> List[Dict[str, Any]]:
        """識別主要混淆變化"""
        changes = []
        threshold = 5  # 變化超過5個樣本才算主要變化
        
        for i in range(cm_diff.shape[0]):
            for j in range(cm_diff.shape[1]):
                if i != j and abs(cm_diff[i, j]) >= threshold:
                    changes.append({
                        "true_label": self._get_label_name(labels[i]),
                        "pred_label": self._get_label_name(labels[j]),
                        "change": int(cm_diff[i, j]),
                        "interpretation": "增加" if cm_diff[i, j] > 0 else "減少"
                    })
        
        return sorted(changes, key=lambda x: abs(x["change"]), reverse=True)
    
    def _identify_domain_specific_errors(self, source_analysis: Dict[str, Any], 
                                       target_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """識別領域特異性錯誤"""
        source_errors = source_analysis.get("error_analysis", {})
        target_errors = target_analysis.get("error_analysis", {})
        
        # 找出目標領域特有的錯誤類型
        source_error_types = set(source_errors.keys())
        target_error_types = set(target_errors.keys())
        
        return {
            "target_specific_errors": list(target_error_types - source_error_types),
            "source_specific_errors": list(source_error_types - target_error_types),
            "common_errors": list(source_error_types.intersection(target_error_types)),
            "error_severity_changes": self._analyze_error_severity_changes(source_errors, target_errors)
        }
    
    def _analyze_error_severity_changes(self, source_errors: Dict[str, Any], 
                                      target_errors: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析錯誤嚴重程度變化"""
        changes = []
        
        for error_type in set(source_errors.keys()).intersection(set(target_errors.keys())):
            source_prop = source_errors[error_type]["proportion"]
            target_prop = target_errors[error_type]["proportion"]
            change = target_prop - source_prop
            
            if abs(change) > 0.05:  # 比例變化超過5%
                changes.append({
                    "error_type": error_type,
                    "source_proportion": source_prop,
                    "target_proportion": target_prop,
                    "change": change,
                    "severity": "惡化" if change > 0 else "改善"
                })
        
        return sorted(changes, key=lambda x: abs(x["change"]), reverse=True)
    
    def _generate_adaptation_insights(self, source_analysis: Dict[str, Any],
                                    target_analysis: Dict[str, Any],
                                    transfer_effects: Dict[str, Any]) -> List[str]:
        """生成適應性洞察"""
        insights = []
        
        # 基於準確率下降
        accuracy_drop = transfer_effects["accuracy_drop"]
        if accuracy_drop < -0.1:
            insights.append("準確率顯著下降(>10%)，建議進行領域適應訓練")
        elif accuracy_drop < -0.05:
            insights.append("準確率有所下降，可考慮少量目標領域數據進行微調")
        
        # 基於錯誤模式變化
        confusion_changes = transfer_effects.get("confusion_pattern_change", {}).get("major_changes", [])
        if confusion_changes:
            insights.append("混淆模式發生顯著變化，建議重點關注變化最大的類別對")
        
        # 基於錯誤分佈
        source_dist = source_analysis.get("error_distribution", {})
        target_dist = target_analysis.get("error_distribution", {})
        
        for label in source_dist.keys():
            if label in target_dist:
                source_rate = source_dist[label]["error_rate"]
                target_rate = target_dist[label]["error_rate"]
                if target_rate - source_rate > 0.2:
                    insights.append(f"{label}類別錯誤率大幅增加，建議增加該類別的訓練樣本")
        
        if not insights:
            insights.append("跨領域遷移效果良好，模型具有較強的泛化能力")
        
        return insights
    
    def _analyze_cluster_error_types(self, cluster_true: np.ndarray, 
                                   cluster_pred: np.ndarray) -> Dict[str, int]:
        """分析聚類內的錯誤類型"""
        error_types = {}
        
        for true_label, pred_label in zip(cluster_true, cluster_pred):
            error_key = f"{self._get_label_name(true_label)}_to_{self._get_label_name(pred_label)}"
            error_types[error_key] = error_types.get(error_key, 0) + 1
        
        return dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True))
    
    def _get_label_name(self, label: Union[int, str]) -> str:
        """獲取標籤名稱"""
        return self.sentiment_labels.get(label, str(label))
    
    # ===== 視覺化方法 =====
    
    def _plot_error_analysis(self, results: Dict[str, Any], domain: str):
        """繪製錯誤分析圖表"""
        try:
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 錯誤類型分佈
            error_analysis = results.get("error_analysis", {})
            if error_analysis:
                error_types = list(error_analysis.keys())
                error_counts = [error_analysis[et]["count"] for et in error_types]
                
                bars = axes[0, 0].bar(range(len(error_types)), error_counts)
                axes[0, 0].set_xticks(range(len(error_types)))
                axes[0, 0].set_xticklabels(error_types, rotation=45)
                axes[0, 0].set_title(f'{domain} - 錯誤類型分佈')
                axes[0, 0].set_ylabel('錯誤數量')
                
                # 添加數值標籤
                for bar, count in zip(bars, error_counts):
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                   str(count), ha='center', va='bottom')
            
            # 2. 混淆矩陣
            confusion_analysis = results.get("confusion_analysis", {})
            if confusion_analysis:
                cm = np.array(confusion_analysis["confusion_matrix"])
                labels = confusion_analysis["labels"]
                
                sns.heatmap(cm, annot=True, fmt='d', 
                           xticklabels=[self._get_label_name(l) for l in labels],
                           yticklabels=[self._get_label_name(l) for l in labels],
                           ax=axes[0, 1])
                axes[0, 1].set_title(f'{domain} - 混淆矩陣')
                axes[0, 1].set_ylabel('真實標籤')
                axes[0, 1].set_xlabel('預測標籤')
            
            # 3. 信心度分佈比較
            confidence_analysis = results.get("confidence_analysis", {})
            if confidence_analysis:
                error_conf = confidence_analysis.get("error_confidence", {})
                correct_conf = confidence_analysis.get("correct_confidence", {})
                
                if error_conf and correct_conf:
                    axes[1, 0].hist([error_conf["mean"]], alpha=0.7, label='錯誤預測', bins=20)
                    axes[1, 0].hist([correct_conf["mean"]], alpha=0.7, label='正確預測', bins=20)
                    axes[1, 0].set_title(f'{domain} - 預測信心度比較')
                    axes[1, 0].set_xlabel('平均信心度')
                    axes[1, 0].set_ylabel('頻率')
                    axes[1, 0].legend()
            
            # 4. 各類別錯誤率
            error_distribution = results.get("error_distribution", {})
            if error_distribution:
                categories = list(error_distribution.keys())
                error_rates = [error_distribution[cat]["error_rate"] for cat in categories]
                
                bars = axes[1, 1].bar(categories, error_rates, alpha=0.7)
                axes[1, 1].set_title(f'{domain} - 各類別錯誤率')
                axes[1, 1].set_ylabel('錯誤率')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                # 添加數值標籤
                for bar, rate in zip(bars, error_rates):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{rate:.2%}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.plot_dir / f'error_analysis_{domain}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"警告：錯誤分析圖表生成失敗: {e}")
    
    def _plot_model_error_comparison(self, results: Dict[str, Any]):
        """繪製模型錯誤比較圖表"""
        try:
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 模型錯誤率比較
            models = results["models"]
            error_rates = [results["error_rates"][model] for model in models]
            
            bars = axes[0, 0].bar(models, error_rates, alpha=0.7)
            axes[0, 0].set_title('模型錯誤率比較')
            axes[0, 0].set_ylabel('錯誤率')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 添加數值標籤
            for bar, rate in zip(bars, error_rates):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{rate:.2%}', ha='center', va='bottom')
            
            # 2. 錯誤重疊分析
            overlap_data = results.get("error_overlap", {})
            if overlap_data:
                pairs = list(overlap_data.keys())
                overlap_ratios = [overlap_data[pair]["overlap_ratio"] for pair in pairs]
                
                bars = axes[0, 1].bar(range(len(pairs)), overlap_ratios, alpha=0.7)
                axes[0, 1].set_xticks(range(len(pairs)))
                axes[0, 1].set_xticklabels([pair.replace('_vs_', ' vs ') for pair in pairs], rotation=45)
                axes[0, 1].set_title('模型間錯誤重疊比例')
                axes[0, 1].set_ylabel('重疊比例')
                
                for bar, ratio in zip(bars, overlap_ratios):
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{ratio:.2%}', ha='center', va='bottom')
            
            # 3. 共同錯誤 vs 獨特錯誤
            common_errors = results.get("common_errors", {}).get("count", 0)
            unique_errors = results.get("unique_errors", {})
            
            if unique_errors:
                unique_counts = [unique_errors[model]["count"] for model in models]
                
                x_pos = np.arange(len(models))
                axes[1, 0].bar(x_pos, unique_counts, alpha=0.7, label='獨特錯誤')
                axes[1, 0].axhline(y=common_errors, color='red', linestyle='--', label=f'共同錯誤: {common_errors}')
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels(models, rotation=45)
                axes[1, 0].set_title('共同錯誤 vs 獨特錯誤')
                axes[1, 0].set_ylabel('錯誤數量')
                axes[1, 0].legend()
            
            # 4. 模型互補性潛力
            complementarity = results.get("complementary_analysis", {})
            if complementarity and "potential_ensemble_improvement" in complementarity:
                ensemble_data = complementarity["potential_ensemble_improvement"]
                combos = list(ensemble_data.keys())
                improvements = [ensemble_data[combo]["improvement"] for combo in combos]
                
                bars = axes[1, 1].bar(range(len(combos)), improvements, alpha=0.7)
                axes[1, 1].set_xticks(range(len(combos)))
                axes[1, 1].set_xticklabels([combo.replace('+', '\n+') for combo in combos], rotation=0)
                axes[1, 1].set_title('集成方法潛在改進')
                axes[1, 1].set_ylabel('錯誤率改進')
                
                for bar, imp in zip(bars, improvements):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                                   f'{imp:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.plot_dir / 'model_error_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"警告：模型錯誤比較圖表生成失敗: {e}")
    
    def _plot_cross_domain_error_analysis(self, results: Dict[str, Any]):
        """繪製跨領域錯誤分析圖表"""
        try:
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            domain_pair = results["domain_pair"]
            source_analysis = results["source_analysis"]
            target_analysis = results["target_analysis"]
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 錯誤率比較
            domains = [domain_pair[0], domain_pair[1]]
            error_rates = [source_analysis["error_rate"], target_analysis["error_rate"]]
            
            bars = axes[0, 0].bar(domains, error_rates, alpha=0.7, color=['blue', 'orange'])
            axes[0, 0].set_title('跨領域錯誤率比較')
            axes[0, 0].set_ylabel('錯誤率')
            
            for bar, rate in zip(bars, error_rates):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{rate:.2%}', ha='center', va='bottom')
            
            # 2. 混淆矩陣比較
            source_cm = np.array(source_analysis["confusion_analysis"]["confusion_matrix"])
            target_cm = np.array(target_analysis["confusion_analysis"]["confusion_matrix"])
            
            # 計算差異
            cm_diff = target_cm - source_cm
            
            sns.heatmap(cm_diff, annot=True, fmt='d', center=0, cmap='RdBu_r', ax=axes[0, 1])
            axes[0, 1].set_title('混淆矩陣變化 (目標 - 源)')
            
            # 3. 各類別錯誤率變化
            source_dist = source_analysis.get("error_distribution", {})
            target_dist = target_analysis.get("error_distribution", {})
            
            if source_dist and target_dist:
                categories = list(set(source_dist.keys()).intersection(set(target_dist.keys())))
                source_rates = [source_dist[cat]["error_rate"] for cat in categories]
                target_rates = [target_dist[cat]["error_rate"] for cat in categories]
                
                x_pos = np.arange(len(categories))
                width = 0.35
                
                axes[1, 0].bar(x_pos - width/2, source_rates, width, label=domain_pair[0], alpha=0.7)
                axes[1, 0].bar(x_pos + width/2, target_rates, width, label=domain_pair[1], alpha=0.7)
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels(categories, rotation=45)
                axes[1, 0].set_title('各類別錯誤率對比')
                axes[1, 0].set_ylabel('錯誤率')
                axes[1, 0].legend()
            
            # 4. 遷移效果總結
            transfer_effects = results.get("transfer_effects", {})
            accuracy_drop = transfer_effects.get("accuracy_drop", 0)
            error_rate_increase = transfer_effects.get("error_rate_increase", 0)
            
            metrics = ['準確率變化', '錯誤率變化']
            values = [accuracy_drop, error_rate_increase]
            colors = ['red' if v < 0 else 'green' for v in values]
            
            bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
            axes[1, 1].set_title('遷移效果總結')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            for bar, value in zip(bars, values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + (0.01 if value >= 0 else -0.02),
                               f'{value:.3f}', ha='center', 
                               va='bottom' if value >= 0 else 'top')
            
            plt.tight_layout()
            plt.savefig(self.plot_dir / f'cross_domain_error_analysis_{domain_pair[0]}_to_{domain_pair[1]}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"警告：跨領域錯誤分析圖表生成失敗: {e}")
    
    def _plot_error_clustering(self, results: Dict[str, Any]):
        """繪製錯誤聚類圖表"""
        try:
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 聚類大小分佈
            cluster_chars = results.get("cluster_characteristics", {})
            if cluster_chars:
                cluster_ids = list(cluster_chars.keys())
                cluster_sizes = [cluster_chars[cid]["size"] for cid in cluster_ids]
                
                bars = axes[0, 0].bar(range(len(cluster_ids)), cluster_sizes, alpha=0.7)
                axes[0, 0].set_xticks(range(len(cluster_ids)))
                axes[0, 0].set_xticklabels(cluster_ids, rotation=45)
                axes[0, 0].set_title('錯誤樣本聚類大小分佈')
                axes[0, 0].set_ylabel('樣本數量')
                
                for bar, size in zip(bars, cluster_sizes):
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                   str(size), ha='center', va='bottom')
            
            # 2. t-SNE 視覺化
            tsne_coords = results.get("dimensional_reduction", {}).get("tsne_coords")
            cluster_labels = results.get("cluster_analysis", {}).get("cluster_labels")
            
            if tsne_coords and cluster_labels:
                tsne_coords = np.array(tsne_coords)
                cluster_labels = np.array(cluster_labels)
                
                scatter = axes[0, 1].scatter(tsne_coords[:, 0], tsne_coords[:, 1], 
                                           c=cluster_labels, cmap='Set1', alpha=0.7)
                axes[0, 1].set_title('錯誤樣本 t-SNE 聚類視覺化')
                axes[0, 1].set_xlabel('t-SNE 維度 1')
                axes[0, 1].set_ylabel('t-SNE 維度 2')
                plt.colorbar(scatter, ax=axes[0, 1])
            
            # 3. 各聚類錯誤類型分佈
            if cluster_chars:
                # 選擇前3個最大的聚類進行詳細分析
                sorted_clusters = sorted(cluster_chars.items(), 
                                       key=lambda x: x[1]["size"], reverse=True)[:3]
                
                all_error_types = set()
                for _, char in sorted_clusters:
                    all_error_types.update(char["common_error_types"].keys())
                
                error_types = list(all_error_types)[:5]  # 最多顯示5種錯誤類型
                
                if error_types:
                    x_pos = np.arange(len(error_types))
                    width = 0.25
                    
                    for i, (cluster_id, char) in enumerate(sorted_clusters):
                        error_counts = [char["common_error_types"].get(et, 0) for et in error_types]
                        axes[1, 0].bar(x_pos + i * width, error_counts, width, 
                                     label=cluster_id, alpha=0.7)
                    
                    axes[1, 0].set_xticks(x_pos + width)
                    axes[1, 0].set_xticklabels(error_types, rotation=45)
                    axes[1, 0].set_title('各聚類主要錯誤類型')
                    axes[1, 0].set_ylabel('錯誤數量')
                    axes[1, 0].legend()
            
            # 4. 聚類質量指標
            cluster_analysis = results.get("cluster_analysis", {})
            if cluster_analysis:
                inertia = cluster_analysis.get("inertia", 0)
                n_clusters = results.get("n_clusters", 0)
                n_samples = results.get("n_error_samples", 0)
                
                metrics = ['聚類內平方和', '平均聚類大小']
                values = [inertia, n_samples / n_clusters if n_clusters > 0 else 0]
                
                bars = axes[1, 1].bar(metrics, values, alpha=0.7)
                axes[1, 1].set_title('聚類質量指標')
                
                for bar, value in zip(bars, values):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.01,
                                   f'{value:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.plot_dir / 'error_clustering_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"警告：錯誤聚類圖表生成失敗: {e}")
    
    def generate_error_analysis_report(self) -> Dict[str, Any]:
        """生成錯誤分析總結報告"""
        if not self.analysis_results:
            return {"error": "沒有錯誤分析歷史記錄"}
        
        report = {
            "分析總覽": {
                "總分析次數": len(self.analysis_results),
                "分析類型": list(set([result["analysis_type"] for result in self.analysis_results])),
                "分析時間範圍": {
                    "開始": str(min([result["timestamp"] for result in self.analysis_results])),
                    "結束": str(max([result["timestamp"] for result in self.analysis_results]))
                }
            },
            "關鍵發現": [],
            "改進建議匯總": [],
            "最常見錯誤模式": [],
            "跨領域差異": []
        }
        
        # 收集關鍵發現
        for result in self.analysis_results:
            if result["analysis_type"] == "prediction_errors":
                domain = result.get("domain", "unknown")
                error_rate = result["results"]["error_rate"]
                
                report["關鍵發現"].append({
                    "領域": domain,
                    "錯誤率": f"{error_rate:.2%}",
                    "分析時間": str(result["timestamp"])
                })
                
                # 收集改進建議
                suggestions = result["results"].get("improvement_suggestions", [])
                report["改進建議匯總"].extend(suggestions)
        
        # 去重改進建議
        report["改進建議匯總"] = list(set(report["改進建議匯總"]))
        
        # 保存報告
        if self.save_plots:
            with open(self.plot_dir / 'error_analysis_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        return report