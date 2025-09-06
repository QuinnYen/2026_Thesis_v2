# 統計分析器模組
"""
統計分析器實現

提供以下統計分析功能：
- 顯著性檢驗: t檢定、卡方檢定、Mann-Whitney U 檢定
- 效果量計算: Cohen's d、Cramer's V、效果大小評估
- 置信區間估計: 均值、比例、差異的置信區間
- 多重比較校正: Bonferroni、FDR 校正
- 分佈分析: 正態性檢驗、分佈參數估計
- 相關性分析: Pearson、Spearman 相關係數
"""

import torch
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, chi2_contingency, mannwhitneyu,
    shapiro, normaltest, ks_2samp, wilcoxon,
    pearsonr, spearmanr, kendalltau,
    bootstrap, sem
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """統計分析器"""
    
    def __init__(self, alpha: float = 0.05, save_plots: bool = True, 
                 plot_dir: str = "statistical_plots"):
        """
        初始化統計分析器
        
        Args:
            alpha: 顯著性水準
            save_plots: 是否保存圖表
            plot_dir: 圖表保存目錄
        """
        self.alpha = alpha
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        
        if self.save_plots:
            self.plot_dir.mkdir(exist_ok=True)
        
        # 分析結果存儲
        self.analysis_results = []
    
    def compare_model_performance(self, model_results: Dict[str, List[float]], 
                                metric_name: str = "F1分數") -> Dict[str, Any]:
        """
        比較多個模型的性能差異
        
        Args:
            model_results: 模型結果 {model_name: [score1, score2, ...]}
            metric_name: 指標名稱
            
        Returns:
            comparison_results: 比較分析結果
        """
        print(f"進行模型性能比較分析 - {metric_name}")
        
        results = {
            "metric_name": metric_name,
            "models": list(model_results.keys()),
            "descriptive_stats": {},
            "significance_tests": {},
            "effect_sizes": {},
            "confidence_intervals": {},
            "post_hoc_tests": {}
        }
        
        # 1. 描述性統計
        for model_name, scores in model_results.items():
            scores_array = np.array(scores)
            results["descriptive_stats"][model_name] = {
                "mean": np.mean(scores_array),
                "std": np.std(scores_array, ddof=1),
                "median": np.median(scores_array),
                "min": np.min(scores_array),
                "max": np.max(scores_array),
                "n": len(scores_array),
                "sem": sem(scores_array)
            }
        
        # 2. 正態性檢驗
        normality_results = {}
        for model_name, scores in model_results.items():
            if len(scores) >= 3:  # 至少需要3個樣本
                shapiro_stat, shapiro_p = shapiro(scores)
                normality_results[model_name] = {
                    "shapiro_stat": shapiro_stat,
                    "shapiro_p": shapiro_p,
                    "is_normal": shapiro_p > self.alpha
                }
        
        results["normality_tests"] = normality_results
        
        # 3. 兩兩比較
        model_names = list(model_results.keys())
        pairwise_comparisons = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                pair_key = f"{model1}_vs_{model2}"
                
                scores1 = np.array(model_results[model1])
                scores2 = np.array(model_results[model2])
                
                # 選擇合適的檢驗方法
                is_normal1 = normality_results.get(model1, {}).get("is_normal", False)
                is_normal2 = normality_results.get(model2, {}).get("is_normal", False)
                
                if is_normal1 and is_normal2 and len(scores1) >= 3 and len(scores2) >= 3:
                    # 使用 t 檢定
                    test_stat, p_value = ttest_ind(scores1, scores2)
                    test_type = "independent_t_test"
                else:
                    # 使用 Mann-Whitney U 檢定
                    test_stat, p_value = mannwhitneyu(scores1, scores2, alternative='two-sided')
                    test_type = "mann_whitney_u"
                
                # 計算效果量 (Cohen's d)
                pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                                    (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                                   (len(scores1) + len(scores2) - 2))
                cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
                
                pairwise_comparisons[pair_key] = {
                    "test_type": test_type,
                    "test_statistic": test_stat,
                    "p_value": p_value,
                    "cohens_d": cohens_d,
                    "effect_size_interpretation": self._interpret_cohens_d(cohens_d),
                    "significant": p_value < self.alpha
                }
        
        results["pairwise_comparisons"] = pairwise_comparisons
        
        # 4. 多重比較校正
        p_values = [comp["p_value"] for comp in pairwise_comparisons.values()]
        if p_values:
            # Bonferroni 校正
            bonferroni_corrected = multipletests(p_values, method='bonferroni')[1]
            
            # FDR 校正
            fdr_corrected = multipletests(p_values, method='fdr_bh')[1]
            
            for i, pair_key in enumerate(pairwise_comparisons.keys()):
                pairwise_comparisons[pair_key]["bonferroni_p"] = bonferroni_corrected[i]
                pairwise_comparisons[pair_key]["fdr_p"] = fdr_corrected[i]
                pairwise_comparisons[pair_key]["bonferroni_significant"] = bonferroni_corrected[i] < self.alpha
                pairwise_comparisons[pair_key]["fdr_significant"] = fdr_corrected[i] < self.alpha
        
        # 5. 置信區間計算
        confidence_intervals = {}
        for model_name, scores in model_results.items():
            scores_array = np.array(scores)
            if len(scores_array) >= 2:
                ci = stats.t.interval(
                    1 - self.alpha,
                    len(scores_array) - 1,
                    loc=np.mean(scores_array),
                    scale=sem(scores_array)
                )
                confidence_intervals[model_name] = {
                    "lower": ci[0],
                    "upper": ci[1],
                    "width": ci[1] - ci[0]
                }
        
        results["confidence_intervals"] = confidence_intervals
        
        # 6. 最佳模型識別
        best_model = max(model_results.keys(), 
                        key=lambda k: results["descriptive_stats"][k]["mean"])
        
        results["best_model"] = {
            "name": best_model,
            "mean_score": results["descriptive_stats"][best_model]["mean"],
            "confidence_interval": confidence_intervals.get(best_model, {})
        }
        
        # 保存結果
        self.analysis_results.append({
            "analysis_type": "model_comparison",
            "metric": metric_name,
            "results": results,
            "timestamp": pd.Timestamp.now()
        })
        
        # 生成視覺化圖表
        if self.save_plots:
            self._plot_model_comparison(model_results, results, metric_name)
        
        return results
    
    def analyze_cross_domain_transfer(self, transfer_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        分析跨領域遷移效果
        
        Args:
            transfer_results: 遷移結果 {source_domain: {target_domain: performance}}
            
        Returns:
            transfer_analysis: 遷移分析結果
        """
        print("進行跨領域遷移效果分析")
        
        # 準備數據
        transfer_data = []
        for source, targets in transfer_results.items():
            for target, performance in targets.items():
                if source != target:  # 排除同域
                    transfer_data.append({
                        "source": source,
                        "target": target,
                        "performance": performance,
                        "transfer_pair": f"{source}->{target}"
                    })
        
        transfer_df = pd.DataFrame(transfer_data)
        
        results = {
            "transfer_matrix": transfer_results,
            "descriptive_stats": {},
            "domain_analysis": {},
            "transfer_difficulty": {},
            "symmetric_analysis": {}
        }
        
        # 1. 描述性統計
        performances = transfer_df["performance"].values
        results["descriptive_stats"] = {
            "mean": np.mean(performances),
            "std": np.std(performances, ddof=1),
            "median": np.median(performances),
            "min": np.min(performances),
            "max": np.max(performances),
            "range": np.max(performances) - np.min(performances)
        }
        
        # 2. 各領域作為源領域的效果
        source_analysis = {}
        for source in transfer_df["source"].unique():
            source_data = transfer_df[transfer_df["source"] == source]["performance"]
            source_analysis[source] = {
                "mean_performance": np.mean(source_data),
                "std_performance": np.std(source_data, ddof=1),
                "transferability_rank": None  # 稍後計算
            }
        
        # 排名源領域的可遷移性
        source_means = {k: v["mean_performance"] for k, v in source_analysis.items()}
        sorted_sources = sorted(source_means.items(), key=lambda x: x[1], reverse=True)
        for rank, (source, _) in enumerate(sorted_sources, 1):
            source_analysis[source]["transferability_rank"] = rank
        
        results["domain_analysis"]["as_source"] = source_analysis
        
        # 3. 各領域作為目標領域的效果
        target_analysis = {}
        for target in transfer_df["target"].unique():
            target_data = transfer_df[transfer_df["target"] == target]["performance"]
            target_analysis[target] = {
                "mean_performance": np.mean(target_data),
                "std_performance": np.std(target_data, ddof=1),
                "difficulty_rank": None  # 稍後計算
            }
        
        # 排名目標領域的難度（性能越低，難度越高）
        target_means = {k: v["mean_performance"] for k, v in target_analysis.items()}
        sorted_targets = sorted(target_means.items(), key=lambda x: x[1])
        for rank, (target, _) in enumerate(sorted_targets, 1):
            target_analysis[target]["difficulty_rank"] = rank
        
        results["domain_analysis"]["as_target"] = target_analysis
        
        # 4. 遷移對稱性分析
        symmetric_pairs = []
        asymmetric_pairs = []
        
        domains = list(set(transfer_df["source"].tolist() + transfer_df["target"].tolist()))
        for i, domain1 in enumerate(domains):
            for j, domain2 in enumerate(domains[i+1:], i+1):
                perf_1_to_2 = transfer_results.get(domain1, {}).get(domain2)
                perf_2_to_1 = transfer_results.get(domain2, {}).get(domain1)
                
                if perf_1_to_2 is not None and perf_2_to_1 is not None:
                    diff = abs(perf_1_to_2 - perf_2_to_1)
                    asymmetry_ratio = diff / max(perf_1_to_2, perf_2_to_1, 1e-8)
                    
                    pair_info = {
                        "domain1": domain1,
                        "domain2": domain2,
                        "perf_1_to_2": perf_1_to_2,
                        "perf_2_to_1": perf_2_to_1,
                        "difference": diff,
                        "asymmetry_ratio": asymmetry_ratio
                    }
                    
                    if asymmetry_ratio < 0.1:  # 相對差異小於 10%
                        symmetric_pairs.append(pair_info)
                    else:
                        asymmetric_pairs.append(pair_info)
        
        results["symmetric_analysis"] = {
            "symmetric_pairs": symmetric_pairs,
            "asymmetric_pairs": asymmetric_pairs,
            "symmetry_ratio": len(symmetric_pairs) / (len(symmetric_pairs) + len(asymmetric_pairs))
                              if (len(symmetric_pairs) + len(asymmetric_pairs)) > 0 else 0
        }
        
        # 5. 統計檢驗
        # 檢驗不同源領域的遷移效果是否有顯著差異
        source_groups = [transfer_df[transfer_df["source"] == source]["performance"].values 
                        for source in transfer_df["source"].unique()]
        
        if len(source_groups) > 2 and all(len(group) > 0 for group in source_groups):
            try:
                # Kruskal-Wallis 檢驗（非參數）
                kw_stat, kw_p = stats.kruskal(*source_groups)
                results["statistical_tests"] = {
                    "kruskal_wallis": {
                        "statistic": kw_stat,
                        "p_value": kw_p,
                        "significant": kw_p < self.alpha,
                        "interpretation": "源領域間遷移效果有顯著差異" if kw_p < self.alpha else "源領域間遷移效果無顯著差異"
                    }
                }
            except:
                results["statistical_tests"] = {"error": "無法執行統計檢驗"}
        
        # 保存結果
        self.analysis_results.append({
            "analysis_type": "cross_domain_transfer",
            "results": results,
            "timestamp": pd.Timestamp.now()
        })
        
        # 生成視覺化
        if self.save_plots:
            self._plot_transfer_analysis(transfer_df, results)
        
        return results
    
    def calculate_confidence_intervals(self, data: Dict[str, List[float]], 
                                     confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
        """
        計算置信區間
        
        Args:
            data: 數據 {group_name: [values]}
            confidence_level: 信賴水準
            
        Returns:
            confidence_intervals: 置信區間結果
        """
        alpha = 1 - confidence_level
        results = {}
        
        for group_name, values in data.items():
            values_array = np.array(values)
            n = len(values_array)
            
            if n >= 2:
                # 均值的置信區間
                mean_val = np.mean(values_array)
                std_err = sem(values_array)
                
                # t 分佈置信區間
                t_ci = stats.t.interval(confidence_level, n-1, loc=mean_val, scale=std_err)
                
                # Bootstrap 置信區間
                try:
                    bootstrap_result = bootstrap(
                        (values_array,), 
                        np.mean, 
                        n_resamples=1000, 
                        confidence_level=confidence_level,
                        random_state=42
                    )
                    bootstrap_ci = (bootstrap_result.confidence_interval.low, 
                                  bootstrap_result.confidence_interval.high)
                except:
                    bootstrap_ci = t_ci
                
                results[group_name] = {
                    "mean": mean_val,
                    "std": np.std(values_array, ddof=1),
                    "n": n,
                    "confidence_level": confidence_level,
                    "t_interval": {
                        "lower": t_ci[0],
                        "upper": t_ci[1],
                        "width": t_ci[1] - t_ci[0]
                    },
                    "bootstrap_interval": {
                        "lower": bootstrap_ci[0],
                        "upper": bootstrap_ci[1],
                        "width": bootstrap_ci[1] - bootstrap_ci[0]
                    }
                }
        
        return results
    
    def perform_correlation_analysis(self, variables: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        執行相關性分析
        
        Args:
            variables: 變數數據 {variable_name: [values]}
            
        Returns:
            correlation_results: 相關性分析結果
        """
        print("進行相關性分析")
        
        # 轉換為 DataFrame
        min_length = min(len(values) for values in variables.values())
        correlation_data = {}
        for var_name, values in variables.items():
            correlation_data[var_name] = values[:min_length]
        
        df = pd.DataFrame(correlation_data)
        
        results = {
            "variables": list(variables.keys()),
            "sample_size": min_length,
            "pearson_correlations": {},
            "spearman_correlations": {},
            "kendall_correlations": {},
            "correlation_matrix": {},
            "significant_correlations": []
        }
        
        var_names = list(variables.keys())
        
        # 計算各種相關係數
        for i, var1 in enumerate(var_names):
            for j, var2 in enumerate(var_names[i+1:], i+1):
                data1 = df[var1]
                data2 = df[var2]
                
                # Pearson 相關係數
                pearson_r, pearson_p = pearsonr(data1, data2)
                
                # Spearman 相關係數  
                spearman_r, spearman_p = spearmanr(data1, data2)
                
                # Kendall tau 相關係數
                kendall_tau, kendall_p = kendalltau(data1, data2)
                
                pair_key = f"{var1}_vs_{var2}"
                
                results["pearson_correlations"][pair_key] = {
                    "correlation": pearson_r,
                    "p_value": pearson_p,
                    "significant": pearson_p < self.alpha
                }
                
                results["spearman_correlations"][pair_key] = {
                    "correlation": spearman_r,
                    "p_value": spearman_p,
                    "significant": spearman_p < self.alpha
                }
                
                results["kendall_correlations"][pair_key] = {
                    "correlation": kendall_tau,
                    "p_value": kendall_p,
                    "significant": kendall_p < self.alpha
                }
                
                # 記錄顯著相關
                if pearson_p < self.alpha:
                    results["significant_correlations"].append({
                        "variables": [var1, var2],
                        "pearson_r": pearson_r,
                        "p_value": pearson_p,
                        "strength": self._interpret_correlation_strength(abs(pearson_r))
                    })
        
        # 相關矩陣
        pearson_matrix = df.corr(method='pearson')
        spearman_matrix = df.corr(method='spearman')
        
        results["correlation_matrix"] = {
            "pearson": pearson_matrix.to_dict(),
            "spearman": spearman_matrix.to_dict()
        }
        
        # 保存結果
        self.analysis_results.append({
            "analysis_type": "correlation_analysis",
            "results": results,
            "timestamp": pd.Timestamp.now()
        })
        
        # 生成視覺化
        if self.save_plots:
            self._plot_correlation_analysis(df, results)
        
        return results
    
    def test_distribution_differences(self, groups: Dict[str, List[float]], 
                                    test_type: str = "auto") -> Dict[str, Any]:
        """
        檢驗分佈差異
        
        Args:
            groups: 組別數據 {group_name: [values]}
            test_type: 檢驗類型 ("auto", "parametric", "nonparametric")
            
        Returns:
            test_results: 檢驗結果
        """
        print(f"進行分佈差異檢驗 - {test_type}")
        
        group_names = list(groups.keys())
        group_values = [np.array(values) for values in groups.values()]
        
        results = {
            "groups": group_names,
            "sample_sizes": [len(values) for values in group_values],
            "normality_tests": {},
            "homogeneity_tests": {},
            "main_tests": {},
            "effect_sizes": {},
            "post_hoc_tests": {}
        }
        
        # 1. 正態性檢驗
        all_normal = True
        for i, (group_name, values) in enumerate(groups.items()):
            if len(values) >= 3:
                shapiro_stat, shapiro_p = shapiro(values)
                is_normal = shapiro_p > self.alpha
                
                results["normality_tests"][group_name] = {
                    "shapiro_stat": shapiro_stat,
                    "shapiro_p": shapiro_p,
                    "is_normal": is_normal
                }
                
                if not is_normal:
                    all_normal = False
        
        # 2. 方差齊性檢驗 (Levene's test)
        if len(group_values) >= 2:
            try:
                levene_stat, levene_p = stats.levene(*group_values)
                equal_variances = levene_p > self.alpha
                
                results["homogeneity_tests"]["levene"] = {
                    "statistic": levene_stat,
                    "p_value": levene_p,
                    "equal_variances": equal_variances
                }
            except:
                equal_variances = True
        else:
            equal_variances = True
        
        # 3. 選擇主要檢驗方法
        if test_type == "auto":
            use_parametric = all_normal and equal_variances and all(len(values) >= 3 for values in group_values)
        elif test_type == "parametric":
            use_parametric = True
        else:  # nonparametric
            use_parametric = False
        
        # 4. 執行主要檢驗
        if len(group_values) == 2:
            # 兩組比較
            if use_parametric:
                # 獨立樣本 t 檢定
                test_stat, p_value = ttest_ind(group_values[0], group_values[1], 
                                             equal_var=equal_variances)
                test_name = "independent_t_test"
            else:
                # Mann-Whitney U 檢定
                test_stat, p_value = mannwhitneyu(group_values[0], group_values[1], 
                                                alternative='two-sided')
                test_name = "mann_whitney_u"
            
            # 計算效果量
            pooled_std = np.sqrt((np.var(group_values[0], ddof=1) + np.var(group_values[1], ddof=1)) / 2)
            cohens_d = (np.mean(group_values[0]) - np.mean(group_values[1])) / pooled_std if pooled_std > 0 else 0
            
            results["main_tests"] = {
                "test_name": test_name,
                "test_statistic": test_stat,
                "p_value": p_value,
                "significant": p_value < self.alpha
            }
            
            results["effect_sizes"] = {
                "cohens_d": cohens_d,
                "interpretation": self._interpret_cohens_d(cohens_d)
            }
            
        elif len(group_values) > 2:
            # 多組比較
            if use_parametric:
                # 單因素 ANOVA
                test_stat, p_value = stats.f_oneway(*group_values)
                test_name = "one_way_anova"
            else:
                # Kruskal-Wallis 檢定
                test_stat, p_value = stats.kruskal(*group_values)
                test_name = "kruskal_wallis"
            
            results["main_tests"] = {
                "test_name": test_name,
                "test_statistic": test_stat,
                "p_value": p_value,
                "significant": p_value < self.alpha
            }
            
            # 如果主檢驗顯著，進行事後檢驗
            if p_value < self.alpha:
                post_hoc_results = {}
                for i, group1_name in enumerate(group_names):
                    for j, group2_name in enumerate(group_names[i+1:], i+1):
                        if use_parametric:
                            # Tukey HSD 的簡化版本（配對 t 檢定）
                            test_stat, p_val = ttest_ind(group_values[i], group_values[j])
                            post_hoc_test = "pairwise_t_test"
                        else:
                            # 配對 Mann-Whitney U 檢定
                            test_stat, p_val = mannwhitneyu(group_values[i], group_values[j], 
                                                          alternative='two-sided')
                            post_hoc_test = "pairwise_mann_whitney"
                        
                        pair_key = f"{group1_name}_vs_{group2_name}"
                        post_hoc_results[pair_key] = {
                            "test": post_hoc_test,
                            "test_statistic": test_stat,
                            "p_value": p_val,
                            "significant": p_val < self.alpha
                        }
                
                # 多重比較校正
                p_values = [result["p_value"] for result in post_hoc_results.values()]
                bonferroni_corrected = multipletests(p_values, method='bonferroni')[1]
                
                for i, pair_key in enumerate(post_hoc_results.keys()):
                    post_hoc_results[pair_key]["bonferroni_p"] = bonferroni_corrected[i]
                    post_hoc_results[pair_key]["bonferroni_significant"] = bonferroni_corrected[i] < self.alpha
                
                results["post_hoc_tests"] = post_hoc_results
        
        # 保存結果
        self.analysis_results.append({
            "analysis_type": "distribution_test",
            "results": results,
            "timestamp": pd.Timestamp.now()
        })
        
        return results
    
    # ===== 輔助方法 =====
    
    def _interpret_cohens_d(self, d: float) -> str:
        """解釋 Cohen's d 效果量"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "極小效果量"
        elif abs_d < 0.5:
            return "小效果量"
        elif abs_d < 0.8:
            return "中等效果量"
        else:
            return "大效果量"
    
    def _interpret_correlation_strength(self, r: float) -> str:
        """解釋相關係數強度"""
        if r < 0.3:
            return "弱相關"
        elif r < 0.7:
            return "中等相關"
        else:
            return "強相關"
    
    def _plot_model_comparison(self, model_results: Dict[str, List[float]], 
                              results: Dict[str, Any], metric_name: str):
        """繪製模型比較圖表"""
        try:
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 箱線圖
            data_for_boxplot = []
            labels_for_boxplot = []
            for model_name, scores in model_results.items():
                data_for_boxplot.extend(scores)
                labels_for_boxplot.extend([model_name] * len(scores))
            
            df_boxplot = pd.DataFrame({
                'Model': labels_for_boxplot,
                'Score': data_for_boxplot
            })
            
            sns.boxplot(data=df_boxplot, x='Model', y='Score', ax=axes[0, 0])
            axes[0, 0].set_title(f'{metric_name} 分佈比較')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. 均值比較條形圖
            models = list(results["descriptive_stats"].keys())
            means = [results["descriptive_stats"][model]["mean"] for model in models]
            stds = [results["descriptive_stats"][model]["std"] for model in models]
            
            bars = axes[0, 1].bar(models, means, yerr=stds, alpha=0.7, capsize=5)
            axes[0, 1].set_title(f'{metric_name} 均值比較')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 添加數值標籤
            for bar, mean in zip(bars, means):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{mean:.3f}', ha='center', va='bottom')
            
            # 3. 置信區間圖
            ci_data = results.get("confidence_intervals", {})
            if ci_data:
                models_ci = list(ci_data.keys())
                means_ci = [results["descriptive_stats"][model]["mean"] for model in models_ci]
                ci_lower = [ci_data[model]["lower"] for model in models_ci]
                ci_upper = [ci_data[model]["upper"] for model in models_ci]
                
                x_pos = range(len(models_ci))
                axes[1, 0].errorbar(x_pos, means_ci, 
                                   yerr=[np.array(means_ci) - np.array(ci_lower),
                                         np.array(ci_upper) - np.array(means_ci)],
                                   fmt='o', capsize=5, capthick=2)
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels(models_ci, rotation=45)
                axes[1, 0].set_title(f'{metric_name} 95% 置信區間')
            
            # 4. 顯著性檢驗結果熱力圖
            pairwise_results = results.get("pairwise_comparisons", {})
            if pairwise_results:
                models_for_heatmap = list(model_results.keys())
                n_models = len(models_for_heatmap)
                p_matrix = np.ones((n_models, n_models))
                
                for pair_key, pair_result in pairwise_results.items():
                    model1, model2 = pair_key.split("_vs_")
                    i = models_for_heatmap.index(model1)
                    j = models_for_heatmap.index(model2)
                    p_val = pair_result["p_value"]
                    p_matrix[i, j] = p_val
                    p_matrix[j, i] = p_val
                
                sns.heatmap(p_matrix, annot=True, fmt='.3f', 
                           xticklabels=models_for_heatmap,
                           yticklabels=models_for_heatmap,
                           cmap='RdYlBu_r', ax=axes[1, 1])
                axes[1, 1].set_title('配對檢驗 p 值')
            
            plt.tight_layout()
            plt.savefig(self.plot_dir / f'model_comparison_{metric_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"警告：模型比較圖表生成失敗: {e}")
    
    def _plot_transfer_analysis(self, transfer_df: pd.DataFrame, results: Dict[str, Any]):
        """繪製遷移分析圖表"""
        try:
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 遷移矩陣熱力圖
            transfer_matrix = results["transfer_matrix"]
            domains = list(transfer_matrix.keys())
            matrix_data = np.zeros((len(domains), len(domains)))
            
            for i, source in enumerate(domains):
                for j, target in enumerate(domains):
                    if target in transfer_matrix[source]:
                        matrix_data[i, j] = transfer_matrix[source][target]
                    else:
                        matrix_data[i, j] = np.nan
            
            sns.heatmap(matrix_data, annot=True, fmt='.3f',
                       xticklabels=domains, yticklabels=domains,
                       cmap='YlOrRd', ax=axes[0, 0])
            axes[0, 0].set_title('跨領域遷移性能矩陣')
            axes[0, 0].set_xlabel('目標領域')
            axes[0, 0].set_ylabel('源領域')
            
            # 2. 源領域可遷移性排名
            source_analysis = results["domain_analysis"]["as_source"]
            sources = list(source_analysis.keys())
            source_performances = [source_analysis[s]["mean_performance"] for s in sources]
            
            bars = axes[0, 1].bar(sources, source_performances, alpha=0.7)
            axes[0, 1].set_title('各領域作為源領域的平均遷移性能')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 添加排名標籤
            for bar, source in zip(bars, sources):
                rank = source_analysis[source]["transferability_rank"]
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'#{rank}', ha='center', va='bottom', weight='bold')
            
            # 3. 目標領域難度排名
            target_analysis = results["domain_analysis"]["as_target"]
            targets = list(target_analysis.keys())
            target_performances = [target_analysis[t]["mean_performance"] for t in targets]
            
            bars = axes[1, 0].bar(targets, target_performances, alpha=0.7, color='orange')
            axes[1, 0].set_title('各領域作為目標領域的平均接收性能')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 添加難度排名標籤
            for bar, target in zip(bars, targets):
                rank = target_analysis[target]["difficulty_rank"]
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'#{rank}', ha='center', va='bottom', weight='bold')
            
            # 4. 遷移對稱性分析
            symmetric_analysis = results["symmetric_analysis"]
            symmetric_count = len(symmetric_analysis["symmetric_pairs"])
            asymmetric_count = len(symmetric_analysis["asymmetric_pairs"])
            
            axes[1, 1].pie([symmetric_count, asymmetric_count], 
                          labels=['對稱遷移', '非對稱遷移'],
                          autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('遷移對稱性分佈')
            
            plt.tight_layout()
            plt.savefig(self.plot_dir / 'cross_domain_transfer_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"警告：遷移分析圖表生成失敗: {e}")
    
    def _plot_correlation_analysis(self, df: pd.DataFrame, results: Dict[str, Any]):
        """繪製相關性分析圖表"""
        try:
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Pearson 相關矩陣熱力圖
            pearson_corr = df.corr(method='pearson')
            sns.heatmap(pearson_corr, annot=True, fmt='.3f', center=0,
                       cmap='RdBu_r', ax=axes[0, 0])
            axes[0, 0].set_title('Pearson 相關係數矩陣')
            
            # 2. Spearman 相關矩陣熱力圖
            spearman_corr = df.corr(method='spearman')
            sns.heatmap(spearman_corr, annot=True, fmt='.3f', center=0,
                       cmap='RdBu_r', ax=axes[0, 1])
            axes[0, 1].set_title('Spearman 相關係數矩陣')
            
            # 3. 散佈圖矩陣（選擇前幾個變數）
            if len(df.columns) <= 4:
                pd.plotting.scatter_matrix(df, ax=axes[1, 0], alpha=0.6, diagonal='hist')
                axes[1, 0].set_title('變數間散佈圖')
            else:
                # 如果變數太多，只顯示前4個
                subset_df = df.iloc[:, :4]
                pd.plotting.scatter_matrix(subset_df, ax=axes[1, 0], alpha=0.6, diagonal='hist')
                axes[1, 0].set_title('前4個變數散佈圖')
            
            # 4. 顯著相關性總結
            significant_corrs = results["significant_correlations"]
            if significant_corrs:
                var_pairs = [f"{corr['variables'][0]} vs {corr['variables'][1]}" 
                           for corr in significant_corrs]
                corr_values = [corr['pearson_r'] for corr in significant_corrs]
                
                bars = axes[1, 1].barh(var_pairs, corr_values)
                axes[1, 1].set_title('顯著相關性 (p < 0.05)')
                axes[1, 1].set_xlabel('Pearson 相關係數')
                
                # 添加顏色編碼
                for bar, corr_val in zip(bars, corr_values):
                    if corr_val > 0:
                        bar.set_color('red' if corr_val > 0.5 else 'pink')
                    else:
                        bar.set_color('blue' if corr_val < -0.5 else 'lightblue')
            else:
                axes[1, 1].text(0.5, 0.5, '沒有發現顯著相關性', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('顯著相關性總結')
            
            plt.tight_layout()
            plt.savefig(self.plot_dir / 'correlation_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"警告：相關性分析圖表生成失敗: {e}")
    
    def generate_statistical_report(self) -> Dict[str, Any]:
        """生成統計分析總結報告"""
        if not self.analysis_results:
            return {"error": "沒有統計分析歷史記錄"}
        
        report = {
            "分析總覽": {
                "總分析次數": len(self.analysis_results),
                "分析類型": list(set([result["analysis_type"] for result in self.analysis_results])),
                "分析時間範圍": {
                    "開始": str(min([result["timestamp"] for result in self.analysis_results])),
                    "結束": str(max([result["timestamp"] for result in self.analysis_results]))
                }
            },
            "分析摘要": {},
            "關鍵發現": [],
            "統計檢驗結果": {},
            "建議": []
        }
        
        # 按分析類型整理結果
        analysis_by_type = {}
        for result in self.analysis_results:
            analysis_type = result["analysis_type"]
            if analysis_type not in analysis_by_type:
                analysis_by_type[analysis_type] = []
            analysis_by_type[analysis_type].append(result)
        
        report["分析摘要"] = {
            "模型比較分析": len(analysis_by_type.get("model_comparison", [])),
            "跨領域遷移分析": len(analysis_by_type.get("cross_domain_transfer", [])),
            "相關性分析": len(analysis_by_type.get("correlation_analysis", [])),
            "分佈檢驗": len(analysis_by_type.get("distribution_test", []))
        }
        
        # 收集關鍵發現
        for result in self.analysis_results:
            if result["analysis_type"] == "model_comparison":
                best_model = result["results"].get("best_model", {})
                if best_model:
                    report["關鍵發現"].append({
                        "類型": "模型比較",
                        "發現": f"最佳模型: {best_model['name']}，平均性能: {best_model['mean_score']:.3f}"
                    })
            
            elif result["analysis_type"] == "cross_domain_transfer":
                symmetry_ratio = result["results"]["symmetric_analysis"]["symmetry_ratio"]
                report["關鍵發現"].append({
                    "類型": "跨領域遷移",
                    "發現": f"遷移對稱性比例: {symmetry_ratio:.1%}"
                })
            
            elif result["analysis_type"] == "correlation_analysis":
                significant_corrs = result["results"]["significant_correlations"]
                if significant_corrs:
                    strongest_corr = max(significant_corrs, key=lambda x: abs(x['pearson_r']))
                    report["關鍵發現"].append({
                        "類型": "相關性分析",
                        "發現": f"最強相關: {strongest_corr['variables'][0]} vs {strongest_corr['variables'][1]} (r={strongest_corr['pearson_r']:.3f})"
                    })
        
        # 保存報告
        if self.save_plots:
            with open(self.plot_dir / 'statistical_analysis_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        return report