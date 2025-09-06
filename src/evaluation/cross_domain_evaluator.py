# 跨領域評估器模組 【論文主要創新點】
"""
跨領域對齊評估器實現

這是本論文的主要創新點，提供以下評估功能：

【核心創新指標】：
- 內聚性指標 (Cohesion Metrics): 測量同一抽象方面內的語義一致性
- 區分性指標 (Discrimination Metrics): 測量不同抽象方面間的差異性
- 穩定性指標 (Stability Metrics): 測量跨領域預測一致性
- 對齊品質指標 (Alignment Quality): 評估抽象方面對齊的有效性

【評估維度】：
1. 語義空間對齊品質
2. 方面間語義距離分析
3. 跨領域遷移效果評估
4. 抽象表示學習效果
5. 領域不變特徵品質
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import wasserstein_distance, ks_2samp
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict


class CrossDomainEvaluator:
    """跨領域評估器 - 論文主要創新點"""
    
    def __init__(self, abstract_aspects: List[str] = None, domains: List[str] = None, 
                 save_plots: bool = True, plot_dir: str = "cross_domain_plots"):
        """
        初始化跨領域評估器
        
        Args:
            abstract_aspects: 抽象方面列表 ['品質', '價格', '服務', '環境', '便利性']
            domains: 領域列表 ['餐廳', '筆記型電腦', '手機']  
            save_plots: 是否保存圖表
            plot_dir: 圖表保存目錄
        """
        self.abstract_aspects = abstract_aspects or ['品質', '價格', '服務', '環境', '便利性']
        self.domains = domains or ['餐廳', '筆記型電腦', '手機']
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        
        if self.save_plots:
            self.plot_dir.mkdir(exist_ok=True)
        
        # 評估結果存儲
        self.evaluation_results = []
        self.alignment_history = []
    
    def evaluate_cross_domain_alignment(self, 
                                      source_features: Dict[str, torch.Tensor],
                                      target_features: Dict[str, torch.Tensor],
                                      source_labels: Dict[str, torch.Tensor],
                                      target_labels: Dict[str, torch.Tensor],
                                      aspect_predictions: Dict[str, Dict[str, torch.Tensor]],
                                      domain_pair: Tuple[str, str]) -> Dict[str, float]:
        """
        評估跨領域對齊品質 - 核心創新方法
        
        Args:
            source_features: 源領域特徵 {aspect: features [batch_size, feature_dim]}
            target_features: 目標領域特徵 {aspect: features [batch_size, feature_dim]}
            source_labels: 源領域標籤 {aspect: labels [batch_size]}
            target_labels: 目標領域標籤 {aspect: labels [batch_size]}
            aspect_predictions: 方面預測結果 {domain: {aspect: predictions}}
            domain_pair: 領域對 (source_domain, target_domain)
            
        Returns:
            alignment_metrics: 跨領域對齊評估指標
        """
        source_domain, target_domain = domain_pair
        print(f"評估跨領域對齊: {source_domain} -> {target_domain}")
        
        alignment_metrics = {}
        
        # 1. 【創新】內聚性指標計算
        cohesion_metrics = self._calculate_cohesion_metrics(
            source_features, target_features, source_labels, target_labels
        )
        alignment_metrics.update(cohesion_metrics)
        
        # 2. 【創新】區分性指標計算  
        discrimination_metrics = self._calculate_discrimination_metrics(
            source_features, target_features
        )
        alignment_metrics.update(discrimination_metrics)
        
        # 3. 【創新】穩定性指標計算
        stability_metrics = self._calculate_stability_metrics(
            aspect_predictions, source_labels, target_labels, domain_pair
        )
        alignment_metrics.update(stability_metrics)
        
        # 4. 【創新】對齊品質綜合評估
        alignment_quality = self._calculate_alignment_quality(
            source_features, target_features, source_labels, target_labels
        )
        alignment_metrics.update(alignment_quality)
        
        # 5. 語義空間分析
        semantic_analysis = self._analyze_semantic_space(
            source_features, target_features, domain_pair
        )
        alignment_metrics.update(semantic_analysis)
        
        # 6. 計算綜合對齊分數
        composite_score = self._calculate_composite_alignment_score(alignment_metrics)
        alignment_metrics['composite_alignment_score'] = composite_score
        
        # 添加元資訊
        alignment_metrics['source_domain'] = source_domain
        alignment_metrics['target_domain'] = target_domain
        alignment_metrics['domain_pair'] = f"{source_domain}->{target_domain}"
        
        # 保存結果
        self.alignment_history.append({
            'domain_pair': domain_pair,
            'metrics': alignment_metrics,
            'timestamp': pd.Timestamp.now()
        })
        
        # 生成視覺化圖表
        if self.save_plots:
            self._generate_alignment_visualizations(
                source_features, target_features, source_labels, target_labels, 
                alignment_metrics, domain_pair
            )
        
        return alignment_metrics
    
    def _calculate_cohesion_metrics(self, source_features: Dict[str, torch.Tensor],
                                   target_features: Dict[str, torch.Tensor],
                                   source_labels: Dict[str, torch.Tensor],
                                   target_labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        【創新】計算內聚性指標 - 測量同一抽象方面內的語義一致性
        
        內聚性指標衡量：
        1. 同一抽象方面在不同領域中的特徵相似性
        2. 同一情感極性在不同領域中的表示一致性
        3. 方面內部的緊湊性和連貫性
        """
        cohesion_metrics = {}
        
        for aspect in self.abstract_aspects:
            if aspect in source_features and aspect in target_features:
                # 轉換為 numpy
                src_feat = source_features[aspect].cpu().numpy()
                tgt_feat = target_features[aspect].cpu().numpy()
                
                if aspect in source_labels and aspect in target_labels:
                    src_labels = source_labels[aspect].cpu().numpy()
                    tgt_labels = target_labels[aspect].cpu().numpy()
                    
                    # 1. 同極性樣本的跨領域相似性
                    sentiment_cohesion = self._calculate_sentiment_cohesion(
                        src_feat, tgt_feat, src_labels, tgt_labels
                    )
                    cohesion_metrics[f'sentiment_cohesion_{aspect}'] = sentiment_cohesion
                    
                    # 2. 方面內部緊湊性
                    internal_compactness = self._calculate_internal_compactness(
                        src_feat, tgt_feat, src_labels, tgt_labels
                    )
                    cohesion_metrics[f'internal_compactness_{aspect}'] = internal_compactness
                
                # 3. 跨領域特徵分佈一致性
                distribution_consistency = self._calculate_distribution_consistency(
                    src_feat, tgt_feat
                )
                cohesion_metrics[f'distribution_consistency_{aspect}'] = distribution_consistency
        
        # 計算平均內聚性
        aspect_cohesions = [v for k, v in cohesion_metrics.items() if 'sentiment_cohesion_' in k]
        cohesion_metrics['average_cohesion'] = np.mean(aspect_cohesions) if aspect_cohesions else 0.0
        
        return cohesion_metrics
    
    def _calculate_discrimination_metrics(self, source_features: Dict[str, torch.Tensor],
                                        target_features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        【創新】計算區分性指標 - 測量不同抽象方面間的差異性
        
        區分性指標衡量：
        1. 不同抽象方面間的特徵可區分性
        2. 跨領域方面邊界的清晰度
        3. 方面間語義距離的合理性
        """
        discrimination_metrics = {}
        
        # 1. 方面間平均距離
        aspect_distances = []
        for i, aspect1 in enumerate(self.abstract_aspects):
            for j, aspect2 in enumerate(self.abstract_aspects[i+1:], i+1):
                if aspect1 in source_features and aspect2 in source_features:
                    if aspect1 in target_features and aspect2 in target_features:
                        # 源領域方面距離
                        src_dist = self._calculate_aspect_distance(
                            source_features[aspect1].cpu().numpy(),
                            source_features[aspect2].cpu().numpy()
                        )
                        
                        # 目標領域方面距離
                        tgt_dist = self._calculate_aspect_distance(
                            target_features[aspect1].cpu().numpy(),
                            target_features[aspect2].cpu().numpy()
                        )
                        
                        # 距離一致性
                        distance_consistency = 1 - abs(src_dist - tgt_dist) / max(src_dist, tgt_dist, 1e-8)
                        discrimination_metrics[f'distance_consistency_{aspect1}_{aspect2}'] = distance_consistency
                        
                        aspect_distances.append((src_dist + tgt_dist) / 2)
        
        # 2. 平均方面間區分度
        discrimination_metrics['average_aspect_discrimination'] = np.mean(aspect_distances) if aspect_distances else 0.0
        
        # 3. 方面可分離性 (Silhouette Score)
        separability_scores = []
        all_features = []
        all_labels = []
        
        for domain_features in [source_features, target_features]:
            for aspect_idx, aspect in enumerate(self.abstract_aspects):
                if aspect in domain_features:
                    features = domain_features[aspect].cpu().numpy()
                    labels = [aspect_idx] * len(features)
                    all_features.extend(features)
                    all_labels.extend(labels)
        
        if len(all_features) > 0 and len(set(all_labels)) > 1:
            try:
                all_features_array = np.array(all_features)
                silhouette = silhouette_score(all_features_array, all_labels)
                discrimination_metrics['aspect_separability'] = max(0, silhouette)  # 確保非負
            except:
                discrimination_metrics['aspect_separability'] = 0.0
        else:
            discrimination_metrics['aspect_separability'] = 0.0
        
        return discrimination_metrics
    
    def _calculate_stability_metrics(self, aspect_predictions: Dict[str, Dict[str, torch.Tensor]],
                                   source_labels: Dict[str, torch.Tensor],
                                   target_labels: Dict[str, torch.Tensor],
                                   domain_pair: Tuple[str, str]) -> Dict[str, float]:
        """
        【創新】計算穩定性指標 - 測量跨領域預測一致性
        
        穩定性指標衡量：
        1. 同一樣本在不同領域中的預測一致性
        2. 相似樣本的跨領域預測穩定性
        3. 決策邊界的跨領域穩定性
        """
        stability_metrics = {}
        source_domain, target_domain = domain_pair
        
        # 1. 預測一致性
        prediction_consistencies = []
        
        for aspect in self.abstract_aspects:
            if (source_domain in aspect_predictions and 
                aspect in aspect_predictions[source_domain] and
                target_domain in aspect_predictions and 
                aspect in aspect_predictions[target_domain]):
                
                src_pred = aspect_predictions[source_domain][aspect].cpu().numpy()
                tgt_pred = aspect_predictions[target_domain][aspect].cpu().numpy()
                
                # 計算預測分佈的相似性
                consistency = self._calculate_prediction_consistency(src_pred, tgt_pred)
                stability_metrics[f'prediction_consistency_{aspect}'] = consistency
                prediction_consistencies.append(consistency)
        
        # 2. 平均預測穩定性
        stability_metrics['average_prediction_stability'] = (
            np.mean(prediction_consistencies) if prediction_consistencies else 0.0
        )
        
        # 3. 標籤分佈穩定性
        label_stability_scores = []
        for aspect in self.abstract_aspects:
            if aspect in source_labels and aspect in target_labels:
                src_labels = source_labels[aspect].cpu().numpy()
                tgt_labels = target_labels[aspect].cpu().numpy()
                
                # 計算標籤分佈的KL散度
                label_stability = self._calculate_label_distribution_stability(src_labels, tgt_labels)
                stability_metrics[f'label_stability_{aspect}'] = label_stability
                label_stability_scores.append(label_stability)
        
        stability_metrics['average_label_stability'] = (
            np.mean(label_stability_scores) if label_stability_scores else 0.0
        )
        
        return stability_metrics
    
    def _calculate_alignment_quality(self, source_features: Dict[str, torch.Tensor],
                                   target_features: Dict[str, torch.Tensor],
                                   source_labels: Dict[str, torch.Tensor],
                                   target_labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        【創新】計算對齊品質指標 - 評估抽象方面對齊的有效性
        
        對齊品質指標衡量：
        1. 跨領域特徵空間的對齊程度
        2. 語義保持的完整性
        3. 領域不變表示的品質
        """
        alignment_quality = {}
        
        # 1. 特徵空間對齊品質
        space_alignment_scores = []
        
        for aspect in self.abstract_aspects:
            if aspect in source_features and aspect in target_features:
                src_feat = source_features[aspect].cpu().numpy()
                tgt_feat = target_features[aspect].cpu().numpy()
                
                # 計算特徵空間對齊分數
                alignment_score = self._calculate_feature_space_alignment(src_feat, tgt_feat)
                alignment_quality[f'space_alignment_{aspect}'] = alignment_score
                space_alignment_scores.append(alignment_score)
        
        alignment_quality['average_space_alignment'] = (
            np.mean(space_alignment_scores) if space_alignment_scores else 0.0
        )
        
        # 2. 語義保持品質
        semantic_preservation_scores = []
        
        for aspect in self.abstract_aspects:
            if (aspect in source_features and aspect in target_features and
                aspect in source_labels and aspect in target_labels):
                
                src_feat = source_features[aspect].cpu().numpy()
                tgt_feat = target_features[aspect].cpu().numpy()
                src_labels = source_labels[aspect].cpu().numpy()
                tgt_labels = target_labels[aspect].cpu().numpy()
                
                # 計算語義保持分數
                semantic_score = self._calculate_semantic_preservation(
                    src_feat, tgt_feat, src_labels, tgt_labels
                )
                alignment_quality[f'semantic_preservation_{aspect}'] = semantic_score
                semantic_preservation_scores.append(semantic_score)
        
        alignment_quality['average_semantic_preservation'] = (
            np.mean(semantic_preservation_scores) if semantic_preservation_scores else 0.0
        )
        
        # 3. 領域不變性品質
        domain_invariance = self._calculate_domain_invariance_quality(
            source_features, target_features
        )
        alignment_quality['domain_invariance_quality'] = domain_invariance
        
        return alignment_quality
    
    def _analyze_semantic_space(self, source_features: Dict[str, torch.Tensor],
                              target_features: Dict[str, torch.Tensor],
                              domain_pair: Tuple[str, str]) -> Dict[str, float]:
        """分析語義空間的品質和特性"""
        semantic_analysis = {}
        
        # 1. 語義空間維度分析
        dimensionality_scores = []
        
        for aspect in self.abstract_aspects:
            if aspect in source_features and aspect in target_features:
                src_feat = source_features[aspect].cpu().numpy()
                tgt_feat = target_features[aspect].cpu().numpy()
                
                # 計算有效維度
                effective_dim = self._calculate_effective_dimensionality(src_feat, tgt_feat)
                semantic_analysis[f'effective_dimensionality_{aspect}'] = effective_dim
                dimensionality_scores.append(effective_dim)
        
        semantic_analysis['average_effective_dimensionality'] = (
            np.mean(dimensionality_scores) if dimensionality_scores else 0.0
        )
        
        # 2. 語義密度分析
        density_scores = []
        
        for aspect in self.abstract_aspects:
            if aspect in source_features and aspect in target_features:
                src_feat = source_features[aspect].cpu().numpy()
                tgt_feat = target_features[aspect].cpu().numpy()
                
                # 計算語義密度
                semantic_density = self._calculate_semantic_density(src_feat, tgt_feat)
                semantic_analysis[f'semantic_density_{aspect}'] = semantic_density
                density_scores.append(semantic_density)
        
        semantic_analysis['average_semantic_density'] = (
            np.mean(density_scores) if density_scores else 0.0
        )
        
        return semantic_analysis
    
    def _calculate_composite_alignment_score(self, alignment_metrics: Dict[str, float]) -> float:
        """
        【創新】計算綜合對齊分數 - 加權組合各項評估指標
        
        綜合分數考慮：
        - 內聚性指標 (權重: 0.3)
        - 區分性指標 (權重: 0.25) 
        - 穩定性指標 (權重: 0.25)
        - 對齊品質指標 (權重: 0.2)
        """
        weights = {
            'cohesion': 0.3,
            'discrimination': 0.25,
            'stability': 0.25,
            'alignment_quality': 0.2
        }
        
        # 提取各類指標
        cohesion_score = alignment_metrics.get('average_cohesion', 0.0)
        discrimination_score = alignment_metrics.get('average_aspect_discrimination', 0.0)
        stability_score = alignment_metrics.get('average_prediction_stability', 0.0)
        alignment_quality_score = alignment_metrics.get('average_space_alignment', 0.0)
        
        # 正規化到 [0, 1] 範圍
        cohesion_score = max(0, min(1, cohesion_score))
        discrimination_score = max(0, min(1, discrimination_score))
        stability_score = max(0, min(1, stability_score))
        alignment_quality_score = max(0, min(1, alignment_quality_score))
        
        # 計算加權綜合分數
        composite_score = (
            weights['cohesion'] * cohesion_score +
            weights['discrimination'] * discrimination_score +
            weights['stability'] * stability_score +
            weights['alignment_quality'] * alignment_quality_score
        )
        
        return composite_score
    
    # ===== 輔助計算方法 =====
    
    def _calculate_sentiment_cohesion(self, src_feat: np.ndarray, tgt_feat: np.ndarray,
                                    src_labels: np.ndarray, tgt_labels: np.ndarray) -> float:
        """計算同極性樣本的跨領域相似性"""
        cohesion_scores = []
        
        for sentiment in [0, 1, 2]:  # 負面、中性、正面
            src_sentiment_mask = (src_labels == sentiment)
            tgt_sentiment_mask = (tgt_labels == sentiment)
            
            if np.sum(src_sentiment_mask) > 0 and np.sum(tgt_sentiment_mask) > 0:
                src_sentiment_feat = src_feat[src_sentiment_mask]
                tgt_sentiment_feat = tgt_feat[tgt_sentiment_mask]
                
                # 計算同極性特徵的平均相似性
                src_centroid = np.mean(src_sentiment_feat, axis=0)
                tgt_centroid = np.mean(tgt_sentiment_feat, axis=0)
                
                similarity = 1 - cosine(src_centroid, tgt_centroid)
                cohesion_scores.append(max(0, similarity))
        
        return np.mean(cohesion_scores) if cohesion_scores else 0.0
    
    def _calculate_internal_compactness(self, src_feat: np.ndarray, tgt_feat: np.ndarray,
                                      src_labels: np.ndarray, tgt_labels: np.ndarray) -> float:
        """計算方面內部緊湊性"""
        # 合併兩個領域的特徵
        combined_feat = np.vstack([src_feat, tgt_feat])
        combined_labels = np.hstack([src_labels, tgt_labels])
        
        # 計算每個類別內部的緊湊性
        compactness_scores = []
        
        for sentiment in [0, 1, 2]:
            sentiment_mask = (combined_labels == sentiment)
            if np.sum(sentiment_mask) > 1:
                sentiment_feat = combined_feat[sentiment_mask]
                
                # 計算內部方差
                centroid = np.mean(sentiment_feat, axis=0)
                distances = [euclidean(feat, centroid) for feat in sentiment_feat]
                compactness = 1 / (1 + np.mean(distances))  # 距離越小，緊湊性越高
                
                compactness_scores.append(compactness)
        
        return np.mean(compactness_scores) if compactness_scores else 0.0
    
    def _calculate_distribution_consistency(self, src_feat: np.ndarray, tgt_feat: np.ndarray) -> float:
        """計算跨領域特徵分佈一致性"""
        try:
            # 使用 Wasserstein 距離測量分佈差異
            src_mean = np.mean(src_feat, axis=0)
            tgt_mean = np.mean(tgt_feat, axis=0)
            
            # 計算各維度的分佈一致性
            consistency_scores = []
            for dim in range(src_feat.shape[1]):
                ks_stat, _ = ks_2samp(src_feat[:, dim], tgt_feat[:, dim])
                consistency = 1 - ks_stat  # KS統計量越小，一致性越高
                consistency_scores.append(max(0, consistency))
            
            return np.mean(consistency_scores)
        except:
            return 0.0
    
    def _calculate_aspect_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """計算兩個方面間的特徵距離"""
        centroid1 = np.mean(feat1, axis=0)
        centroid2 = np.mean(feat2, axis=0)
        return euclidean(centroid1, centroid2)
    
    def _calculate_prediction_consistency(self, src_pred: np.ndarray, tgt_pred: np.ndarray) -> float:
        """計算預測一致性"""
        # 計算預測分佈的相似性
        if len(src_pred.shape) > 1:  # 機率預測
            src_dist = np.mean(src_pred, axis=0)
            tgt_dist = np.mean(tgt_pred, axis=0)
        else:  # 類別預測
            src_dist = np.bincount(src_pred, minlength=3) / len(src_pred)
            tgt_dist = np.bincount(tgt_pred, minlength=3) / len(tgt_pred)
        
        # 計算分佈間的相似性
        similarity = 1 - np.sum(np.abs(src_dist - tgt_dist)) / 2
        return max(0, similarity)
    
    def _calculate_label_distribution_stability(self, src_labels: np.ndarray, tgt_labels: np.ndarray) -> float:
        """計算標籤分佈穩定性"""
        src_dist = np.bincount(src_labels, minlength=3) / len(src_labels)
        tgt_dist = np.bincount(tgt_labels, minlength=3) / len(tgt_labels)
        
        # 使用 Jensen-Shannon 散度
        m = (src_dist + tgt_dist) / 2
        js_div = 0.5 * np.sum(src_dist * np.log(src_dist / (m + 1e-10) + 1e-10)) + \
                 0.5 * np.sum(tgt_dist * np.log(tgt_dist / (m + 1e-10) + 1e-10))
        
        return max(0, 1 - js_div)  # 散度越小，穩定性越高
    
    def _calculate_feature_space_alignment(self, src_feat: np.ndarray, tgt_feat: np.ndarray) -> float:
        """計算特徵空間對齊分數"""
        try:
            # 使用 PCA 分析主成分相似性
            from sklearn.preprocessing import StandardScaler
            
            # 標準化特徵
            scaler = StandardScaler()
            src_feat_scaled = scaler.fit_transform(src_feat)
            tgt_feat_scaled = scaler.fit_transform(tgt_feat)
            
            # PCA 分析
            pca = PCA(n_components=min(10, src_feat.shape[1]))
            src_pca = pca.fit_transform(src_feat_scaled)
            tgt_pca = pca.fit_transform(tgt_feat_scaled)
            
            # 計算主成分空間中的相似性
            src_centroid = np.mean(src_pca, axis=0)
            tgt_centroid = np.mean(tgt_pca, axis=0)
            
            alignment = 1 - cosine(src_centroid, tgt_centroid)
            return max(0, alignment)
        except:
            return 0.0
    
    def _calculate_semantic_preservation(self, src_feat: np.ndarray, tgt_feat: np.ndarray,
                                       src_labels: np.ndarray, tgt_labels: np.ndarray) -> float:
        """計算語義保持品質"""
        # 計算同類樣本的跨領域相似性
        preservation_scores = []
        
        for sentiment in [0, 1, 2]:
            src_mask = (src_labels == sentiment)
            tgt_mask = (tgt_labels == sentiment)
            
            if np.sum(src_mask) > 0 and np.sum(tgt_mask) > 0:
                src_sentiment = src_feat[src_mask]
                tgt_sentiment = tgt_feat[tgt_mask]
                
                # 計算類內相似性
                intra_class_sim = self._calculate_intra_class_similarity(src_sentiment, tgt_sentiment)
                preservation_scores.append(intra_class_sim)
        
        return np.mean(preservation_scores) if preservation_scores else 0.0
    
    def _calculate_domain_invariance_quality(self, source_features: Dict[str, torch.Tensor],
                                           target_features: Dict[str, torch.Tensor]) -> float:
        """計算領域不變性品質"""
        invariance_scores = []
        
        for aspect in self.abstract_aspects:
            if aspect in source_features and aspect in target_features:
                src_feat = source_features[aspect].cpu().numpy()
                tgt_feat = target_features[aspect].cpu().numpy()
                
                # 計算特徵統計量的相似性
                src_mean = np.mean(src_feat, axis=0)
                tgt_mean = np.mean(tgt_feat, axis=0)
                src_std = np.std(src_feat, axis=0)
                tgt_std = np.std(tgt_feat, axis=0)
                
                # 均值相似性
                mean_sim = 1 - cosine(src_mean, tgt_mean)
                
                # 標準差相似性  
                std_sim = 1 - np.mean(np.abs(src_std - tgt_std) / (src_std + tgt_std + 1e-8))
                
                invariance_score = (mean_sim + std_sim) / 2
                invariance_scores.append(max(0, invariance_score))
        
        return np.mean(invariance_scores) if invariance_scores else 0.0
    
    def _calculate_effective_dimensionality(self, src_feat: np.ndarray, tgt_feat: np.ndarray) -> float:
        """計算有效維度"""
        try:
            # 合併特徵
            combined_feat = np.vstack([src_feat, tgt_feat])
            
            # PCA 分析
            pca = PCA()
            pca.fit(combined_feat)
            
            # 計算累積方差解釋比例
            cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
            effective_dim = np.argmax(cumsum_ratio >= 0.95) + 1  # 解釋95%方差的維度數
            
            return effective_dim / combined_feat.shape[1]  # 正規化
        except:
            return 0.0
    
    def _calculate_semantic_density(self, src_feat: np.ndarray, tgt_feat: np.ndarray) -> float:
        """計算語義密度"""
        try:
            # 計算特徵空間中的樣本密度
            combined_feat = np.vstack([src_feat, tgt_feat])
            
            # 使用 KMeans 估計密度
            n_clusters = min(10, len(combined_feat) // 5)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(combined_feat)
                
                # 計算平均類內距離
                intra_cluster_distances = []
                for cluster in range(n_clusters):
                    cluster_points = combined_feat[labels == cluster]
                    if len(cluster_points) > 1:
                        cluster_center = kmeans.cluster_centers_[cluster]
                        distances = [euclidean(point, cluster_center) for point in cluster_points]
                        intra_cluster_distances.extend(distances)
                
                # 密度與距離成反比
                avg_distance = np.mean(intra_cluster_distances) if intra_cluster_distances else 1.0
                density = 1 / (1 + avg_distance)
                return density
            else:
                return 0.5
        except:
            return 0.0
    
    def _calculate_intra_class_similarity(self, src_feat: np.ndarray, tgt_feat: np.ndarray) -> float:
        """計算類內相似性"""
        try:
            # 計算類內樣本的平均相似性
            src_centroid = np.mean(src_feat, axis=0)
            tgt_centroid = np.mean(tgt_feat, axis=0)
            
            # 跨領域類內相似性
            cross_domain_sim = 1 - cosine(src_centroid, tgt_centroid)
            
            return max(0, cross_domain_sim)
        except:
            return 0.0
    
    def _generate_alignment_visualizations(self, source_features: Dict[str, torch.Tensor],
                                         target_features: Dict[str, torch.Tensor],
                                         source_labels: Dict[str, torch.Tensor],
                                         target_labels: Dict[str, torch.Tensor],
                                         metrics: Dict[str, float],
                                         domain_pair: Tuple[str, str]):
        """生成跨領域對齊視覺化圖表"""
        try:
            source_domain, target_domain = domain_pair
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 對齊品質綜合評估雷達圖
            self._plot_alignment_radar_chart(metrics, domain_pair)
            
            # 2. t-SNE 降維視覺化
            self._plot_tsne_visualization(source_features, target_features, 
                                        source_labels, target_labels, domain_pair)
            
            # 3. 方面間距離矩陣熱力圖
            self._plot_aspect_distance_heatmap(source_features, target_features, domain_pair)
            
            # 4. 穩定性分析圖
            self._plot_stability_analysis(metrics, domain_pair)
            
        except Exception as e:
            print(f"警告：生成對齊視覺化時發生錯誤: {e}")
    
    def _plot_alignment_radar_chart(self, metrics: Dict[str, float], domain_pair: Tuple[str, str]):
        """繪製對齊品質雷達圖"""
        # 提取主要指標
        categories = ['內聚性', '區分性', '穩定性', '對齊品質', '語義密度']
        values = [
            metrics.get('average_cohesion', 0),
            metrics.get('average_aspect_discrimination', 0),
            metrics.get('average_prediction_stability', 0),
            metrics.get('average_space_alignment', 0),
            metrics.get('average_semantic_density', 0)
        ]
        
        # 正規化到 [0, 1]
        values = [max(0, min(1, v)) for v in values]
        
        # 繪製雷達圖
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 閉合圖形
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label='對齊品質')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        
        plt.title(f'跨領域對齊品質評估\n{domain_pair[0]} → {domain_pair[1]}', size=16, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        domain_str = f"{domain_pair[0]}_{domain_pair[1]}"
        plt.savefig(self.plot_dir / f'alignment_radar_{domain_str}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tsne_visualization(self, source_features: Dict[str, torch.Tensor],
                               target_features: Dict[str, torch.Tensor],
                               source_labels: Dict[str, torch.Tensor],
                               target_labels: Dict[str, torch.Tensor],
                               domain_pair: Tuple[str, str]):
        """繪製 t-SNE 降維視覺化"""
        try:
            # 選擇一個代表性方面進行視覺化
            aspect = self.abstract_aspects[0]  # 使用第一個方面作為示例
            
            if (aspect in source_features and aspect in target_features and
                aspect in source_labels and aspect in target_labels):
                
                # 合併特徵和標籤
                src_feat = source_features[aspect].cpu().numpy()
                tgt_feat = target_features[aspect].cpu().numpy()
                src_labels = source_labels[aspect].cpu().numpy()
                tgt_labels = target_labels[aspect].cpu().numpy()
                
                all_features = np.vstack([src_feat, tgt_feat])
                all_labels = np.hstack([src_labels, tgt_labels])
                domain_labels = ['源領域'] * len(src_feat) + ['目標領域'] * len(tgt_feat)
                
                # t-SNE 降維
                if len(all_features) > 50:  # 只有足夠樣本時才進行 t-SNE
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
                    features_2d = tsne.fit_transform(all_features)
                    
                    # 繪製
                    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                    
                    # 按情感標籤著色
                    sentiment_colors = ['red', 'gray', 'blue']  # 負面、中性、正面
                    sentiment_names = ['負面', '中性', '正面']
                    
                    for sentiment, color, name in zip([0, 1, 2], sentiment_colors, sentiment_names):
                        mask = (all_labels == sentiment)
                        axes[0].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                      c=color, alpha=0.6, s=50, label=name)
                    
                    axes[0].set_title(f'{aspect} - 按情感標籤')
                    axes[0].legend()
                    
                    # 按領域著色
                    domain_colors = ['orange', 'purple']
                    for domain, color in zip(['源領域', '目標領域'], domain_colors):
                        mask = [d == domain for d in domain_labels]
                        axes[1].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                      c=color, alpha=0.6, s=50, label=domain)
                    
                    axes[1].set_title(f'{aspect} - 按領域')
                    axes[1].legend()
                    
                    plt.suptitle(f't-SNE 特徵空間視覺化: {domain_pair[0]} → {domain_pair[1]}')
                    
                    domain_str = f"{domain_pair[0]}_{domain_pair[1]}"
                    plt.savefig(self.plot_dir / f'tsne_visualization_{domain_str}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                
        except Exception as e:
            print(f"警告：t-SNE 視覺化失敗: {e}")
    
    def _plot_aspect_distance_heatmap(self, source_features: Dict[str, torch.Tensor],
                                    target_features: Dict[str, torch.Tensor],
                                    domain_pair: Tuple[str, str]):
        """繪製方面間距離矩陣熱力圖"""
        try:
            # 計算方面間距離矩陣
            n_aspects = len(self.abstract_aspects)
            distance_matrix = np.zeros((n_aspects, n_aspects))
            
            for i, aspect1 in enumerate(self.abstract_aspects):
                for j, aspect2 in enumerate(self.abstract_aspects):
                    if aspect1 in source_features and aspect2 in source_features:
                        if aspect1 in target_features and aspect2 in target_features:
                            # 計算平均距離
                            src_dist = self._calculate_aspect_distance(
                                source_features[aspect1].cpu().numpy(),
                                source_features[aspect2].cpu().numpy()
                            )
                            tgt_dist = self._calculate_aspect_distance(
                                target_features[aspect1].cpu().numpy(),
                                target_features[aspect2].cpu().numpy()
                            )
                            distance_matrix[i, j] = (src_dist + tgt_dist) / 2
            
            # 繪製熱力圖
            plt.figure(figsize=(10, 8))
            sns.heatmap(distance_matrix, 
                       xticklabels=self.abstract_aspects,
                       yticklabels=self.abstract_aspects,
                       annot=True, fmt='.3f', cmap='YlOrRd')
            
            plt.title(f'方面間語義距離矩陣\n{domain_pair[0]} → {domain_pair[1]}')
            
            domain_str = f"{domain_pair[0]}_{domain_pair[1]}"
            plt.savefig(self.plot_dir / f'aspect_distance_heatmap_{domain_str}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"警告：方面距離熱力圖生成失敗: {e}")
    
    def _plot_stability_analysis(self, metrics: Dict[str, float], domain_pair: Tuple[str, str]):
        """繪製穩定性分析圖"""
        try:
            # 提取各方面的穩定性指標
            stability_metrics = []
            aspect_names = []
            
            for aspect in self.abstract_aspects:
                stability_key = f'prediction_consistency_{aspect}'
                if stability_key in metrics:
                    stability_metrics.append(metrics[stability_key])
                    aspect_names.append(aspect)
            
            if stability_metrics:
                plt.figure(figsize=(12, 6))
                bars = plt.bar(aspect_names, stability_metrics, alpha=0.7, color='skyblue')
                
                # 添加數值標籤
                for bar, value in zip(bars, stability_metrics):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                plt.title(f'各方面預測穩定性分析\n{domain_pair[0]} → {domain_pair[1]}')
                plt.ylabel('穩定性分數')
                plt.xlabel('抽象方面')
                plt.xticks(rotation=45)
                plt.ylim(0, 1.1)
                plt.grid(True, alpha=0.3)
                
                domain_str = f"{domain_pair[0]}_{domain_pair[1]}"
                plt.tight_layout()
                plt.savefig(self.plot_dir / f'stability_analysis_{domain_str}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"警告：穩定性分析圖生成失敗: {e}")
    
    def generate_cross_domain_report(self) -> Dict[str, any]:
        """生成跨領域評估總結報告"""
        if not self.alignment_history:
            return {"error": "沒有跨領域評估歷史記錄"}
        
        report = {
            "評估總覽": {
                "評估次數": len(self.alignment_history),
                "領域對數量": len(set([result['domain_pair'] for result in self.alignment_history])),
                "評估時間範圍": {
                    "開始": str(min([result['timestamp'] for result in self.alignment_history])),
                    "結束": str(max([result['timestamp'] for result in self.alignment_history]))
                }
            },
            "性能分析": {},
            "最佳對齊": {},
            "創新指標統計": {},
            "詳細結果": []
        }
        
        # 性能分析
        all_composite_scores = [result['metrics']['composite_alignment_score'] 
                              for result in self.alignment_history]
        
        report["性能分析"] = {
            "平均綜合對齊分數": np.mean(all_composite_scores),
            "最佳綜合對齊分數": np.max(all_composite_scores),
            "對齊分數標準差": np.std(all_composite_scores),
            "對齊品質分佈": {
                "優秀 (>0.8)": sum(1 for s in all_composite_scores if s > 0.8),
                "良好 (0.6-0.8)": sum(1 for s in all_composite_scores if 0.6 <= s <= 0.8),
                "一般 (0.4-0.6)": sum(1 for s in all_composite_scores if 0.4 <= s <= 0.6),
                "待改善 (<0.4)": sum(1 for s in all_composite_scores if s < 0.4)
            }
        }
        
        # 找出最佳對齊
        best_alignment = max(self.alignment_history, 
                           key=lambda x: x['metrics']['composite_alignment_score'])
        
        report["最佳對齊"] = {
            "領域對": best_alignment['domain_pair'],
            "綜合分數": best_alignment['metrics']['composite_alignment_score'],
            "內聚性": best_alignment['metrics'].get('average_cohesion', 0),
            "區分性": best_alignment['metrics'].get('average_aspect_discrimination', 0),
            "穩定性": best_alignment['metrics'].get('average_prediction_stability', 0),
            "對齊品質": best_alignment['metrics'].get('average_space_alignment', 0)
        }
        
        # 創新指標統計
        cohesion_scores = [result['metrics'].get('average_cohesion', 0) 
                          for result in self.alignment_history]
        discrimination_scores = [result['metrics'].get('average_aspect_discrimination', 0) 
                               for result in self.alignment_history]
        stability_scores = [result['metrics'].get('average_prediction_stability', 0) 
                          for result in self.alignment_history]
        
        report["創新指標統計"] = {
            "內聚性指標": {
                "平均值": np.mean(cohesion_scores),
                "標準差": np.std(cohesion_scores),
                "最大值": np.max(cohesion_scores),
                "最小值": np.min(cohesion_scores)
            },
            "區分性指標": {
                "平均值": np.mean(discrimination_scores),
                "標準差": np.std(discrimination_scores),
                "最大值": np.max(discrimination_scores),
                "最小值": np.min(discrimination_scores)
            },
            "穩定性指標": {
                "平均值": np.mean(stability_scores),
                "標準差": np.std(stability_scores),
                "最大值": np.max(stability_scores),
                "最小值": np.min(stability_scores)
            }
        }
        
        # 詳細結果
        for result in self.alignment_history:
            detail = {
                "領域對": result['domain_pair'],
                "時間": str(result['timestamp']),
                "綜合分數": result['metrics']['composite_alignment_score'],
                "主要指標": {
                    "內聚性": result['metrics'].get('average_cohesion', 0),
                    "區分性": result['metrics'].get('average_aspect_discrimination', 0),
                    "穩定性": result['metrics'].get('average_prediction_stability', 0),
                    "對齊品質": result['metrics'].get('average_space_alignment', 0)
                }
            }
            report["詳細結果"].append(detail)
        
        # 保存報告
        if self.save_plots:
            with open(self.plot_dir / 'cross_domain_evaluation_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        return report