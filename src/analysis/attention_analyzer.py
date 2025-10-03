# 注意力權重分析器
"""
注意力權重分析器模組

提供深入的注意力機制分析功能：
- 注意力聚焦度分析：量化注意力的集中程度
- 注意力分佈熵計算：衡量注意力分佈的均勻性
- 異常模式檢測：識別異常的注意力模式
- 有意義詞彙識別：找出獲得高注意力權重的重要詞彙
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import Counter, defaultdict
import scipy.stats as stats
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class AttentionAnalyzer:
    """注意力權重分析器"""

    def __init__(self,
                 language: str = 'en',
                 stop_words: Optional[List[str]] = None,
                 min_attention_threshold: float = 0.01):
        """
        初始化注意力分析器

        Args:
            language: 語言類型
            stop_words: 停用詞列表
            min_attention_threshold: 最小注意力閾值
        """
        self.language = language
        self.min_attention_threshold = min_attention_threshold

        # 預設停用詞
        default_stop_words = {
            'en': ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'],
            'zh': ['的', '了', '和', '是', '在', '有', '也', '都', '不', '要', '可以', '就', '這', '那']
        }
        self.stop_words = set(stop_words or default_stop_words.get(language, []))

        # 分析結果快取
        self.analysis_cache = {}

    def analyze_attention_weights(self,
                                attention_weights: torch.Tensor,
                                tokens: Optional[List[str]] = None,
                                layer_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        綜合分析注意力權重

        Args:
            attention_weights: 注意力權重張量 [batch_size, num_heads, seq_len, seq_len]
            tokens: 對應的詞彙列表
            layer_names: 層名稱列表

        Returns:
            完整的注意力分析結果
        """
        if attention_weights.dim() != 4:
            raise ValueError("注意力權重應該是4維張量 [batch_size, num_heads, seq_len, seq_len]")

        batch_size, num_heads, seq_len, _ = attention_weights.shape

        analysis_results = {
            'basic_stats': self._compute_basic_statistics(attention_weights),
            'focus_analysis': self._analyze_attention_focus(attention_weights),
            'entropy_analysis': self._compute_entropy_metrics(attention_weights),
            'anomaly_detection': self._detect_attention_anomalies(attention_weights),
            'head_comparison': self._compare_attention_heads(attention_weights),
            'pattern_analysis': self._analyze_attention_patterns(attention_weights)
        }

        # 如果提供了詞彙，進行詞彙相關分析
        if tokens is not None:
            analysis_results['token_analysis'] = self._analyze_token_attention(
                attention_weights, tokens
            )

        # 添加元數據
        analysis_results['metadata'] = {
            'batch_size': batch_size,
            'num_heads': num_heads,
            'seq_len': seq_len,
            'total_weights': attention_weights.numel(),
            'layer_names': layer_names
        }

        return analysis_results

    def _compute_basic_statistics(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """計算基本統計指標"""
        with torch.no_grad():
            weights = attention_weights.detach().cpu()

            stats = {
                'mean': weights.mean().item(),
                'std': weights.std().item(),
                'min': weights.min().item(),
                'max': weights.max().item(),
                'median': weights.median().item(),
                'q25': weights.quantile(0.25).item(),
                'q75': weights.quantile(0.75).item(),
                'skewness': float(stats.skew(weights.flatten().numpy())),
                'kurtosis': float(stats.kurtosis(weights.flatten().numpy())),
                'sparsity_ratio': (weights < self.min_attention_threshold).float().mean().item()
            }

            return stats

    def _analyze_attention_focus(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """分析注意力聚焦度"""
        with torch.no_grad():
            weights = attention_weights.detach().cpu()
            batch_size, num_heads, seq_len, _ = weights.shape

            focus_metrics = {}

            # 計算每個頭的聚焦度指標
            head_focus_scores = []
            head_concentration_ratios = []
            head_effective_ranges = []

            for head_idx in range(num_heads):
                head_weights = weights[:, head_idx, :, :]  # [batch_size, seq_len, seq_len]

                # 1. 聚焦度分數：最大權重的平均值
                max_weights = head_weights.max(dim=-1)[0]  # [batch_size, seq_len]
                focus_score = max_weights.mean().item()
                head_focus_scores.append(focus_score)

                # 2. 集中度比率：前10%權重占總權重的比例
                flattened = head_weights.flatten()
                sorted_weights, _ = torch.sort(flattened, descending=True)
                top_10_percent = int(len(sorted_weights) * 0.1)
                concentration_ratio = sorted_weights[:top_10_percent].sum() / sorted_weights.sum()
                head_concentration_ratios.append(concentration_ratio.item())

                # 3. 有效範圍：超過閾值的權重比例
                effective_ratio = (head_weights > self.min_attention_threshold).float().mean()
                head_effective_ranges.append(effective_ratio.item())

            focus_metrics['head_focus_scores'] = head_focus_scores
            focus_metrics['head_concentration_ratios'] = head_concentration_ratios
            focus_metrics['head_effective_ranges'] = head_effective_ranges

            # 整體聚焦度指標
            focus_metrics['overall_focus_score'] = float(np.mean(head_focus_scores))
            focus_metrics['overall_concentration_ratio'] = float(np.mean(head_concentration_ratios))
            focus_metrics['overall_effective_range'] = float(np.mean(head_effective_ranges))

            # 聚焦度變異性
            focus_metrics['focus_score_variance'] = float(np.var(head_focus_scores))
            focus_metrics['concentration_variance'] = float(np.var(head_concentration_ratios))

            # 位置聚焦分析
            position_focus = self._analyze_positional_focus(weights)
            focus_metrics['position_focus'] = position_focus

            return focus_metrics

    def _analyze_positional_focus(self, weights: torch.Tensor) -> Dict[str, Any]:
        """分析位置聚焦模式"""
        batch_size, num_heads, seq_len, _ = weights.shape

        position_metrics = {}

        # 計算每個位置的平均注意力權重
        position_attention = weights.mean(dim=(0, 1))  # [seq_len, seq_len]

        # 對角線注意力（自我注意）
        diagonal_attention = torch.diag(position_attention)
        position_metrics['self_attention_ratio'] = diagonal_attention.mean().item()

        # 局部性分析：計算注意力的「重心」偏移
        positions = torch.arange(seq_len, dtype=torch.float)
        centroids = []

        for i in range(seq_len):
            weights_i = position_attention[i, :]
            if weights_i.sum() > 0:
                centroid = (weights_i * positions).sum() / weights_i.sum()
                centroids.append(abs(centroid - i).item())

        position_metrics['average_centroid_deviation'] = float(np.mean(centroids))
        position_metrics['centroid_deviation_std'] = float(np.std(centroids))

        # 局部窗口分析
        window_sizes = [3, 5, 7, 10]
        local_attention_ratios = {}

        for window_size in window_sizes:
            local_ratios = []
            half_window = window_size // 2

            for i in range(seq_len):
                start = max(0, i - half_window)
                end = min(seq_len, i + half_window + 1)
                local_attention = position_attention[i, start:end].sum()
                local_ratios.append(local_attention.item())

            local_attention_ratios[f'window_{window_size}'] = float(np.mean(local_ratios))

        position_metrics['local_attention_ratios'] = local_attention_ratios

        return position_metrics

    def _compute_entropy_metrics(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """計算注意力分佈熵相關指標"""
        with torch.no_grad():
            weights = attention_weights.detach().cpu()
            batch_size, num_heads, seq_len, _ = weights.shape

            entropy_metrics = {}

            # 為每個頭計算熵指標
            head_entropies = []
            head_normalized_entropies = []
            head_perplexities = []

            eps = 1e-8  # 防止 log(0)

            for head_idx in range(num_heads):
                head_weights = weights[:, head_idx, :, :]  # [batch_size, seq_len, seq_len]

                # 計算每行的熵（每個查詢位置的注意力分佈）
                row_entropies = []
                row_perplexities = []

                for batch_idx in range(batch_size):
                    for pos_idx in range(seq_len):
                        attention_dist = head_weights[batch_idx, pos_idx, :]
                        attention_dist = attention_dist + eps  # 數值穩定性

                        # Shannon 熵
                        entropy = -torch.sum(attention_dist * torch.log(attention_dist))
                        row_entropies.append(entropy.item())

                        # 困惑度 (Perplexity)
                        perplexity = torch.exp(entropy)
                        row_perplexities.append(perplexity.item())

                # 頭級別的平均指標
                head_entropy = float(np.mean(row_entropies))
                head_entropies.append(head_entropy)

                # 標準化熵（相對於最大可能熵）
                max_entropy = np.log(seq_len)
                normalized_entropy = head_entropy / max_entropy if max_entropy > 0 else 0
                head_normalized_entropies.append(normalized_entropy)

                head_perplexity = float(np.mean(row_perplexities))
                head_perplexities.append(head_perplexity)

            entropy_metrics['head_entropies'] = head_entropies
            entropy_metrics['head_normalized_entropies'] = head_normalized_entropies
            entropy_metrics['head_perplexities'] = head_perplexities

            # 整體熵指標
            entropy_metrics['overall_entropy'] = float(np.mean(head_entropies))
            entropy_metrics['overall_normalized_entropy'] = float(np.mean(head_normalized_entropies))
            entropy_metrics['overall_perplexity'] = float(np.mean(head_perplexities))

            # 熵的變異性
            entropy_metrics['entropy_variance'] = float(np.var(head_entropies))
            entropy_metrics['entropy_range'] = float(np.max(head_entropies) - np.min(head_entropies))

            # 熵分類
            entropy_metrics['entropy_classification'] = self._classify_entropy_levels(
                head_normalized_entropies, seq_len
            )

            return entropy_metrics

    def _classify_entropy_levels(self, normalized_entropies: List[float], seq_len: int) -> Dict[str, Any]:
        """對熵水平進行分類"""
        classification = {
            'very_low_entropy': 0,    # < 0.3
            'low_entropy': 0,         # 0.3 - 0.5
            'medium_entropy': 0,      # 0.5 - 0.7
            'high_entropy': 0,        # 0.7 - 0.9
            'very_high_entropy': 0    # > 0.9
        }

        for entropy in normalized_entropies:
            if entropy < 0.3:
                classification['very_low_entropy'] += 1
            elif entropy < 0.5:
                classification['low_entropy'] += 1
            elif entropy < 0.7:
                classification['medium_entropy'] += 1
            elif entropy < 0.9:
                classification['high_entropy'] += 1
            else:
                classification['very_high_entropy'] += 1

        # 轉換為比例
        total_heads = len(normalized_entropies)
        for key in classification:
            classification[key] = classification[key] / total_heads if total_heads > 0 else 0

        return classification

    def _detect_attention_anomalies(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """檢測異常的注意力模式"""
        with torch.no_grad():
            weights = attention_weights.detach().cpu()
            batch_size, num_heads, seq_len, _ = weights.shape

            anomaly_results = {}

            # 1. 檢測極端權重值
            flattened_weights = weights.flatten()
            q99 = flattened_weights.quantile(0.99)
            q01 = flattened_weights.quantile(0.01)

            extreme_high_count = (flattened_weights > q99).sum().item()
            extreme_low_count = (flattened_weights < q01).sum().item()

            anomaly_results['extreme_weights'] = {
                'high_threshold': q99.item(),
                'low_threshold': q01.item(),
                'extreme_high_count': extreme_high_count,
                'extreme_low_count': extreme_low_count,
                'extreme_high_ratio': extreme_high_count / flattened_weights.numel(),
                'extreme_low_ratio': extreme_low_count / flattened_weights.numel()
            }

            # 2. 檢測不均勻分佈
            uniformity_scores = []
            for head_idx in range(num_heads):
                head_weights = weights[:, head_idx, :, :]
                # 計算與均勻分佈的 KL 散度
                uniform_dist = torch.ones_like(head_weights) / seq_len
                kl_div = F.kl_div(torch.log(head_weights + 1e-8), uniform_dist, reduction='batchmean')
                uniformity_scores.append(kl_div.item())

            anomaly_results['uniformity_analysis'] = {
                'head_uniformity_scores': uniformity_scores,
                'average_kl_divergence': float(np.mean(uniformity_scores)),
                'uniformity_variance': float(np.var(uniformity_scores))
            }

            # 3. 檢測頭間相似性異常
            head_similarity_matrix = self._compute_head_similarity_matrix(weights)

            # 找出異常相似的頭對
            similarity_threshold = 0.9
            high_similarity_pairs = []

            for i in range(num_heads):
                for j in range(i + 1, num_heads):
                    similarity = head_similarity_matrix[i, j]
                    if similarity > similarity_threshold:
                        high_similarity_pairs.append((i, j, similarity))

            anomaly_results['head_similarity_anomalies'] = {
                'similarity_matrix': head_similarity_matrix.tolist(),
                'high_similarity_pairs': high_similarity_pairs,
                'average_head_similarity': float(head_similarity_matrix.mean()),
                'max_head_similarity': float(head_similarity_matrix.max())
            }

            # 4. 檢測位置偏差
            position_bias_scores = []
            for head_idx in range(num_heads):
                head_weights = weights[:, head_idx, :, :]
                # 計算每個位置作為查詢時的總權重分佈
                query_totals = head_weights.sum(dim=-1).mean(dim=0)  # [seq_len]

                # 檢測是否有顯著的位置偏差
                position_variance = query_totals.var().item()
                position_bias_scores.append(position_variance)

            anomaly_results['position_bias'] = {
                'head_position_bias_scores': position_bias_scores,
                'average_position_bias': float(np.mean(position_bias_scores)),
                'max_position_bias': float(np.max(position_bias_scores))
            }

            return anomaly_results

    def _compute_head_similarity_matrix(self, weights: torch.Tensor) -> torch.Tensor:
        """計算注意力頭之間的相似性矩陣"""
        batch_size, num_heads, seq_len, _ = weights.shape

        similarity_matrix = torch.zeros(num_heads, num_heads)

        for i in range(num_heads):
            for j in range(num_heads):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    head_i = weights[:, i, :, :].flatten()
                    head_j = weights[:, j, :, :].flatten()

                    # 計算餘弦相似度
                    cosine_sim = F.cosine_similarity(head_i, head_j, dim=0)
                    similarity_matrix[i, j] = cosine_sim.item()

        return similarity_matrix

    def _compare_attention_heads(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """比較不同注意力頭的特性"""
        with torch.no_grad():
            weights = attention_weights.detach().cpu()
            batch_size, num_heads, seq_len, _ = weights.shape

            comparison_results = {}

            # 為每個頭計算特徵向量
            head_features = []

            for head_idx in range(num_heads):
                head_weights = weights[:, head_idx, :, :]

                features = {
                    'mean_weight': head_weights.mean().item(),
                    'std_weight': head_weights.std().item(),
                    'max_weight': head_weights.max().item(),
                    'sparsity': (head_weights < self.min_attention_threshold).float().mean().item(),
                    'self_attention_ratio': torch.diag(head_weights.mean(dim=0)).mean().item()
                }

                head_features.append(features)

            comparison_results['head_features'] = head_features

            # 頭聚類分析
            feature_matrix = np.array([
                [f['mean_weight'], f['std_weight'], f['max_weight'], f['sparsity'], f['self_attention_ratio']]
                for f in head_features
            ])

            if num_heads > 2:
                # 標準化特徵
                scaler = StandardScaler()
                normalized_features = scaler.fit_transform(feature_matrix)

                # K-means 聚類
                n_clusters = min(3, num_heads)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(normalized_features)

                comparison_results['head_clustering'] = {
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'n_clusters': n_clusters
                }

            # 頭多樣性分析
            diversity_metrics = self._compute_head_diversity(weights)
            comparison_results['diversity_metrics'] = diversity_metrics

            return comparison_results

    def _compute_head_diversity(self, weights: torch.Tensor) -> Dict[str, float]:
        """計算注意力頭的多樣性指標"""
        batch_size, num_heads, seq_len, _ = weights.shape

        # 計算頭間的平均相似度
        similarity_matrix = self._compute_head_similarity_matrix(weights)

        # 移除對角線元素（自相似性）
        mask = ~torch.eye(num_heads, dtype=torch.bool)
        off_diagonal_similarities = similarity_matrix[mask]

        diversity_metrics = {
            'average_head_similarity': off_diagonal_similarities.mean().item(),
            'min_head_similarity': off_diagonal_similarities.min().item(),
            'max_head_similarity': off_diagonal_similarities.max().item(),
            'similarity_std': off_diagonal_similarities.std().item(),
            'diversity_score': 1.0 - off_diagonal_similarities.mean().item()  # 多樣性 = 1 - 相似性
        }

        return diversity_metrics

    def _analyze_attention_patterns(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """分析注意力模式"""
        with torch.no_grad():
            weights = attention_weights.detach().cpu()
            batch_size, num_heads, seq_len, _ = weights.shape

            pattern_results = {}

            # 1. 注意力模式分類
            pattern_types = {
                'local_pattern': 0,      # 局部模式
                'global_pattern': 0,     # 全局模式
                'diagonal_pattern': 0,   # 對角模式
                'random_pattern': 0      # 隨機模式
            }

            for head_idx in range(num_heads):
                head_weights = weights[:, head_idx, :, :].mean(dim=0)  # 平均所有批次

                pattern_type = self._classify_attention_pattern(head_weights)
                pattern_types[pattern_type] += 1

            # 轉換為比例
            total_heads = num_heads
            for pattern_type in pattern_types:
                pattern_types[pattern_type] /= total_heads

            pattern_results['pattern_distribution'] = pattern_types

            # 2. 注意力流分析
            attention_flow = self._analyze_attention_flow(weights)
            pattern_results['attention_flow'] = attention_flow

            # 3. 週期性模式檢測
            periodicity_analysis = self._detect_periodic_patterns(weights)
            pattern_results['periodicity'] = periodicity_analysis

            return pattern_results

    def _classify_attention_pattern(self, attention_matrix: torch.Tensor) -> str:
        """分類單個注意力頭的模式"""
        seq_len = attention_matrix.shape[0]

        # 計算對角線權重比例
        diagonal_ratio = torch.diag(attention_matrix).sum() / attention_matrix.sum()

        # 計算局部性指標
        local_weight = 0
        window_size = 3
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            local_weight += attention_matrix[i, start:end].sum()

        local_ratio = local_weight / attention_matrix.sum()

        # 分類邏輯
        if diagonal_ratio > 0.5:
            return 'diagonal_pattern'
        elif local_ratio > 0.7:
            return 'local_pattern'
        elif attention_matrix.std() < 0.1:  # 權重分佈很均勻
            return 'global_pattern'
        else:
            return 'random_pattern'

    def _analyze_attention_flow(self, weights: torch.Tensor) -> Dict[str, Any]:
        """分析注意力流動模式"""
        batch_size, num_heads, seq_len, _ = weights.shape

        # 計算整體注意力流
        avg_weights = weights.mean(dim=(0, 1))  # [seq_len, seq_len]

        flow_metrics = {}

        # 前向流動 vs 後向流動
        forward_flow = 0
        backward_flow = 0

        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:  # 前向
                    forward_flow += avg_weights[i, j].item()
                elif j < i:  # 後向
                    backward_flow += avg_weights[i, j].item()

        total_flow = forward_flow + backward_flow
        if total_flow > 0:
            flow_metrics['forward_flow_ratio'] = forward_flow / total_flow
            flow_metrics['backward_flow_ratio'] = backward_flow / total_flow
        else:
            flow_metrics['forward_flow_ratio'] = 0.0
            flow_metrics['backward_flow_ratio'] = 0.0

        # 計算注意力的"中心"
        positions = torch.arange(seq_len, dtype=torch.float)
        attention_centers = []

        for i in range(seq_len):
            weights_i = avg_weights[i, :]
            if weights_i.sum() > 0:
                center = (weights_i * positions).sum() / weights_i.sum()
                attention_centers.append(center.item())

        flow_metrics['attention_centers'] = attention_centers
        flow_metrics['center_variance'] = float(np.var(attention_centers))

        return flow_metrics

    def _detect_periodic_patterns(self, weights: torch.Tensor) -> Dict[str, Any]:
        """檢測週期性注意力模式"""
        batch_size, num_heads, seq_len, _ = weights.shape

        periodicity_results = {}

        # 對每個頭分析週期性
        head_periodicities = []

        for head_idx in range(num_heads):
            head_weights = weights[:, head_idx, :, :].mean(dim=0)  # [seq_len, seq_len]

            # 分析對角線附近的週期性
            diag_weights = []
            for offset in range(-(seq_len//4), seq_len//4 + 1):
                if offset == 0:
                    continue
                diag_sum = torch.diagonal(head_weights, offset=offset).sum().item()
                diag_weights.append(diag_sum)

            if len(diag_weights) > 0:
                # 使用FFT檢測週期性
                fft_result = np.fft.fft(diag_weights)
                power_spectrum = np.abs(fft_result) ** 2

                # 找出主要頻率
                dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                dominant_period = len(diag_weights) / dominant_freq_idx if dominant_freq_idx > 0 else 0

                head_periodicities.append({
                    'dominant_period': dominant_period,
                    'power_spectrum_max': float(np.max(power_spectrum[1:]))
                })

        periodicity_results['head_periodicities'] = head_periodicities

        if head_periodicities:
            periods = [h['dominant_period'] for h in head_periodicities]
            periodicity_results['average_period'] = float(np.mean(periods))
            periodicity_results['period_variance'] = float(np.var(periods))

        return periodicity_results

    def _analyze_token_attention(self,
                               attention_weights: torch.Tensor,
                               tokens: List[str]) -> Dict[str, Any]:
        """分析詞彙級別的注意力模式"""
        with torch.no_grad():
            weights = attention_weights.detach().cpu()
            batch_size, num_heads, seq_len, _ = weights.shape

            if len(tokens) != seq_len:
                raise ValueError(f"詞彙數量 ({len(tokens)}) 與序列長度 ({seq_len}) 不匹配")

            token_analysis = {}

            # 1. 詞彙注意力分數
            token_scores = self._compute_token_attention_scores(weights, tokens)
            token_analysis['token_scores'] = token_scores

            # 2. 重要詞彙識別
            important_tokens = self._identify_important_tokens(weights, tokens)
            token_analysis['important_tokens'] = important_tokens

            # 3. 詞性注意力模式
            pos_patterns = self._analyze_pos_attention_patterns(weights, tokens)
            token_analysis['pos_patterns'] = pos_patterns

            # 4. 詞彙關聯分析
            token_associations = self._analyze_token_associations(weights, tokens)
            token_analysis['token_associations'] = token_associations

            return token_analysis

    def _compute_token_attention_scores(self,
                                      weights: torch.Tensor,
                                      tokens: List[str]) -> Dict[str, Any]:
        """計算每個詞彙的注意力分數"""
        batch_size, num_heads, seq_len, _ = weights.shape

        # 計算每個詞彙接收到的總注意力
        incoming_attention = weights.sum(dim=(0, 1, 2))  # [seq_len]

        # 計算每個詞彙發出的總注意力
        outgoing_attention = weights.sum(dim=(0, 1, 3))  # [seq_len]

        token_scores = {}
        for i, token in enumerate(tokens):
            token_scores[token] = {
                'incoming_attention': incoming_attention[i].item(),
                'outgoing_attention': outgoing_attention[i].item(),
                'total_attention': (incoming_attention[i] + outgoing_attention[i]).item(),
                'position': i
            }

        # 排序
        sorted_by_total = sorted(token_scores.items(),
                               key=lambda x: x[1]['total_attention'], reverse=True)

        return {
            'token_scores': token_scores,
            'top_tokens_by_attention': sorted_by_total[:10],
            'attention_statistics': {
                'mean_incoming': incoming_attention.mean().item(),
                'std_incoming': incoming_attention.std().item(),
                'mean_outgoing': outgoing_attention.mean().item(),
                'std_outgoing': outgoing_attention.std().item()
            }
        }

    def _identify_important_tokens(self,
                                 weights: torch.Tensor,
                                 tokens: List[str]) -> Dict[str, Any]:
        """識別重要詞彙"""
        batch_size, num_heads, seq_len, _ = weights.shape

        # 過濾停用詞
        content_tokens = [(i, token) for i, token in enumerate(tokens)
                         if token.lower() not in self.stop_words]

        if not content_tokens:
            return {'important_tokens': [], 'filtered_tokens': []}

        # 計算內容詞的注意力分數
        content_indices = [idx for idx, _ in content_tokens]
        content_weights = weights[:, :, content_indices, :]
        content_attention_scores = content_weights.sum(dim=(0, 1, 3))  # [num_content_tokens]

        # 識別高注意力詞彙
        threshold = content_attention_scores.quantile(0.8)  # 前20%
        high_attention_indices = (content_attention_scores > threshold).nonzero().flatten()

        important_tokens = []
        for idx in high_attention_indices:
            original_idx = content_indices[idx.item()]
            token = tokens[original_idx]
            score = content_attention_scores[idx].item()

            important_tokens.append({
                'token': token,
                'position': original_idx,
                'attention_score': score,
                'normalized_score': score / content_attention_scores.sum().item()
            })

        # 按分數排序
        important_tokens.sort(key=lambda x: x['attention_score'], reverse=True)

        return {
            'important_tokens': important_tokens,
            'threshold': threshold.item(),
            'num_important_tokens': len(important_tokens),
            'importance_ratio': len(important_tokens) / len(content_tokens)
        }

    def _analyze_pos_attention_patterns(self,
                                      weights: torch.Tensor,
                                      tokens: List[str]) -> Dict[str, Any]:
        """分析詞性注意力模式（簡化版本）"""
        # 簡單的詞性識別（基於規則）
        pos_categories = {
            'noun': [],
            'verb': [],
            'adjective': [],
            'adverb': [],
            'function_word': [],
            'other': []
        }

        # 基本詞性分類規則
        for i, token in enumerate(tokens):
            token_lower = token.lower()

            if token_lower in self.stop_words:
                pos_categories['function_word'].append(i)
            elif token.endswith(('ing', 'ed', 'es', 's')) and len(token) > 3:
                pos_categories['verb'].append(i)
            elif token.endswith(('ly',)) and len(token) > 3:
                pos_categories['adverb'].append(i)
            elif token.endswith(('ive', 'ous', 'ful', 'less')) and len(token) > 4:
                pos_categories['adjective'].append(i)
            else:
                pos_categories['noun'].append(i)

        # 計算每個詞性類別的平均注意力
        pos_attention = {}
        total_attention = weights.sum()

        for pos, indices in pos_categories.items():
            if indices:
                pos_weights = weights[:, :, indices, :].sum()
                pos_attention[pos] = {
                    'total_attention': pos_weights.item(),
                    'average_attention': pos_weights.item() / len(indices),
                    'relative_attention': pos_weights.item() / total_attention.item(),
                    'token_count': len(indices)
                }

        return pos_attention

    def _analyze_token_associations(self,
                                  weights: torch.Tensor,
                                  tokens: List[str]) -> Dict[str, Any]:
        """分析詞彙間的關聯模式"""
        batch_size, num_heads, seq_len, _ = weights.shape

        # 計算平均注意力矩陣
        avg_attention = weights.mean(dim=(0, 1))  # [seq_len, seq_len]

        # 找出強關聯的詞彙對
        association_threshold = avg_attention.quantile(0.9)
        strong_associations = []

        for i in range(seq_len):
            for j in range(seq_len):
                if i != j and avg_attention[i, j] > association_threshold:
                    strong_associations.append({
                        'from_token': tokens[i],
                        'to_token': tokens[j],
                        'from_position': i,
                        'to_position': j,
                        'attention_weight': avg_attention[i, j].item(),
                        'bidirectional_weight': (avg_attention[i, j] + avg_attention[j, i]).item()
                    })

        # 按權重排序
        strong_associations.sort(key=lambda x: x['attention_weight'], reverse=True)

        # 構建關聯網絡統計
        association_stats = {
            'total_associations': len(strong_associations),
            'threshold': association_threshold.item(),
            'top_associations': strong_associations[:20],
            'average_association_weight': float(np.mean([a['attention_weight']
                                                       for a in strong_associations])) if strong_associations else 0
        }

        return association_stats

    def generate_attention_report(self, analysis_results: Dict[str, Any]) -> str:
        """生成注意力分析報告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("注意力權重分析報告")
        report_lines.append("=" * 60)
        report_lines.append("")

        # 基本資訊
        metadata = analysis_results.get('metadata', {})
        report_lines.append(f"數據概覽:")
        report_lines.append(f"  批次大小: {metadata.get('batch_size', 'N/A')}")
        report_lines.append(f"  注意力頭數: {metadata.get('num_heads', 'N/A')}")
        report_lines.append(f"  序列長度: {metadata.get('seq_len', 'N/A')}")
        report_lines.append("")

        # 基本統計
        basic_stats = analysis_results.get('basic_stats', {})
        report_lines.append("基本統計:")
        report_lines.append(f"  平均權重: {basic_stats.get('mean', 0):.4f}")
        report_lines.append(f"  標準差: {basic_stats.get('std', 0):.4f}")
        report_lines.append(f"  稀疏度: {basic_stats.get('sparsity_ratio', 0):.4f}")
        report_lines.append("")

        # 聚焦度分析
        focus_analysis = analysis_results.get('focus_analysis', {})
        report_lines.append("聚焦度分析:")
        report_lines.append(f"  整體聚焦分數: {focus_analysis.get('overall_focus_score', 0):.4f}")
        report_lines.append(f"  集中度比率: {focus_analysis.get('overall_concentration_ratio', 0):.4f}")
        report_lines.append("")

        # 熵分析
        entropy_analysis = analysis_results.get('entropy_analysis', {})
        report_lines.append("熵分析:")
        report_lines.append(f"  整體熵: {entropy_analysis.get('overall_entropy', 0):.4f}")
        report_lines.append(f"  標準化熵: {entropy_analysis.get('overall_normalized_entropy', 0):.4f}")

        entropy_classification = entropy_analysis.get('entropy_classification', {})
        report_lines.append(f"  熵分類:")
        for level, ratio in entropy_classification.items():
            report_lines.append(f"    {level}: {ratio:.2%}")
        report_lines.append("")

        # 異常檢測
        anomaly_detection = analysis_results.get('anomaly_detection', {})
        if anomaly_detection:
            report_lines.append("異常檢測:")
            extreme_weights = anomaly_detection.get('extreme_weights', {})
            report_lines.append(f"  極端高權重比例: {extreme_weights.get('extreme_high_ratio', 0):.4f}")

            head_similarity = anomaly_detection.get('head_similarity_anomalies', {})
            report_lines.append(f"  頭間平均相似度: {head_similarity.get('average_head_similarity', 0):.4f}")
            report_lines.append("")

        # 詞彙分析（如果有）
        token_analysis = analysis_results.get('token_analysis', {})
        if token_analysis:
            important_tokens = token_analysis.get('important_tokens', {})
            top_tokens = important_tokens.get('important_tokens', [])[:5]

            report_lines.append("重要詞彙:")
            for token_info in top_tokens:
                report_lines.append(f"  {token_info['token']}: {token_info['attention_score']:.4f}")
            report_lines.append("")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)