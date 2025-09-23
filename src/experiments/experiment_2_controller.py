# 實驗2控制器：注意力機制比較實驗
"""
實驗2：注意力機制比較測試

實驗設計：
- 固定條件：Attention Fusion 融合策略 (基於實驗1最佳結果)
- 變數：7種注意力機制
- 評估指標：各機制的優劣勢分析

注意力機制：
1. 自注意力 (Self-Attention)
2. 多頭注意力 (Multi-Head Attention) 
3. 跨模態注意力 (Cross-Modal Attention)
4. 相似度注意力 (Similarity Attention)
5. 關鍵詞導向注意力 (Keyword-Guided Attention)
6. 位置感知注意力 (Position-Aware Attention)
7. 階層注意力 (Hierarchical Attention)

本控制器整合到主系統中，提供完整的注意力機制比較流程。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
import os
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

import sys
from pathlib import Path

# 添加上級目錄到路徑
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from attention_mechanism_experiment import (
    AttentionMechanismComparator, AttentionMechanismProfile, AttentionAnalyzer
)
from experiment_1_controller import Experiment1Controller
from data import DatasetManager, SemEval2014Loader, SemEval2016Loader
from data.data_converter import create_experiment_data_converter


class Experiment2Controller:
    """實驗2控制器：注意力機制比較實驗"""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "experiment2_results"):
        """
        初始化實驗2控制器
        
        Args:
            config: 實驗配置
            output_dir: 輸出目錄
        """
        self.config = config
        self.output_dir = output_dir
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 設置日誌
        self._setup_logging()
        
        # 初始化注意力機制比較器
        self.attention_comparator = AttentionMechanismComparator(
            hidden_dim=config.get('hidden_dim', 768),
            device=self.device
        )
        
        # 資料集管理器
        self.dataset_manager = None
        self.test_datasets = {}
        
        # 數據轉換器
        self.data_converter = create_experiment_data_converter(config)
        
        # 實驗結果
        self.experiment_results = {}
        self.attention_profiles = {}
        
        # 設置中文字體支援
        self._setup_chinese_fonts()
        
        self.logger.info("實驗2控制器初始化完成")
    
    def _setup_logging(self):
        """設置日誌系統"""
        log_file = os.path.join(self.output_dir, "experiment2.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('Experiment2')
    
    def _setup_chinese_fonts(self):
        """設置中文字體支援"""
        try:
            # 嘗試設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            self.logger.warning(f"中文字體設置失敗: {str(e)}")
    
    def setup_datasets(self):
        """設置實驗資料集 (復用實驗1的設置)"""
        self.logger.info("設置實驗資料集...")
        
        data_config = self.config.get('data', {})
        
        # 初始化資料集載入器
        loaders = {}
        
        # SemEval 2014 資料集
        if data_config.get('use_semeval2014', True):
            loader_2014 = SemEval2014Loader(
                data_dir=data_config.get('semeval2014_path', 'data/SemEval2014')
            )
            loaders['semeval2014'] = loader_2014
        
        # SemEval 2016 資料集  
        if data_config.get('use_semeval2016', True):
            loader_2016 = SemEval2016Loader(
                data_dir=data_config.get('semeval2016_path', 'data/SemEval2016')
            )
            loaders['semeval2016'] = loader_2016
        
        # 創建資料集管理器
        self.dataset_manager = DatasetManager(data_config.get('base_data_dir', 'data'))
        
        # 載入測試資料
        try:
            all_data = self.dataset_manager.load_all_datasets()
            
            domains = set()
            for dataset_name, domain_data in all_data.items():
                domains.update(domain_data.keys())
            
            for domain in domains:
                domain_test_data = []
                
                for dataset_name, domain_data in all_data.items():
                    if domain in domain_data:
                        # 使用實際的測試分割
                        domain_splits = domain_data[domain]
                        if 'test' in domain_splits:
                            domain_test_data.extend(domain_splits['test'])
                
                # 轉換AspectSentiment數據為features/labels格式
                if domain_test_data:
                    self.test_datasets[domain] = self.data_converter.convert_to_features_labels(
                        domain_test_data, use_bert=True
                    )
            
            self.logger.info(f"成功載入 {len(domains)} 個領域的測試資料")
            
        except Exception as e:
            self.logger.error(f"載入真實資料集失敗: {str(e)}")
            raise ValueError(
                f"實驗2無法運行，因為數據載入失敗：{str(e)}\n"
                f"請確保以下數據集可用：\n"
                f"1. SemEval-2014數據集 (restaurants, laptops)\n"
                f"2. SemEval-2016數據集 (restaurants, laptops)\n"
                f"3. 數據路徑配置正確\n"
                f"4. XML文件格式完整"
            )
    
    def _combine_domain_test_data(self, test_datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """合併多個領域的測試數據"""
        if not test_datasets:
            return {'features': np.array([]), 'labels': np.array([])}
        
        all_features = []
        all_labels = []
        all_texts = []
        all_aspect_terms = []
        all_domains = []
        
        for domain, data in test_datasets.items():
            if 'features' in data and 'labels' in data and len(data['features']) > 0:
                all_features.append(data['features'])
                all_labels.append(data['labels'])
                
                # 可選的附加信息
                if 'texts' in data:
                    all_texts.extend(data['texts'])
                if 'aspect_terms' in data:
                    all_aspect_terms.extend(data['aspect_terms'])
                if 'domains' in data:
                    all_domains.extend(data['domains'])
                else:
                    # 如果沒有域信息，使用當前域名
                    all_domains.extend([domain] * len(data['labels']))
        
        if not all_features:
            return {'features': np.array([]), 'labels': np.array([])}
        
        # 合併特徵和標籤
        import numpy as np
        combined_features = np.vstack(all_features)
        combined_labels = np.hstack(all_labels)
        
        result = {
            'features': combined_features,
            'labels': combined_labels
        }
        
        # 添加可選信息
        if all_texts:
            result['texts'] = all_texts
        if all_aspect_terms:
            result['aspect_terms'] = all_aspect_terms
        if all_domains:
            result['domains'] = all_domains
            
        return result
    
    def run_experiment(self, best_fusion_strategy: str = None) -> Dict[str, Any]:
        """運行實驗2：注意力機制比較"""
        self.logger.info("開始運行實驗2：注意力機制比較")
        
        # 設置資料集
        self.setup_datasets()
        
        # 如果沒有指定最佳融合策略，使用預設值
        if best_fusion_strategy is None:
            best_fusion_strategy = self.config.get('best_fusion_strategy', 'weighted')
            self.logger.info(f"使用預設融合策略: {best_fusion_strategy}")
        else:
            self.logger.info(f"使用指定的最佳融合策略: {best_fusion_strategy}")
        
        # 獲取實驗參數
        experiment_config = self.config.get('experiment', {})
        num_epochs = experiment_config.get('num_epochs', 5)
        
        # 運行注意力機制比較實驗
        self.logger.info("開始注意力機制比較測試...")
        
        try:
            # 合併所有領域的測試數據
            combined_test_data = self._combine_domain_test_data(self.test_datasets)
            
            # 執行注意力機制比較
            attention_profiles = self.attention_comparator.compare_all_mechanisms(
                combined_test_data, num_epochs
            )
            
            # 生成比較報告
            comparison_report = self.attention_comparator.generate_comparison_report(attention_profiles)
            
            # 存儲結果
            self.attention_profiles = attention_profiles
            self.experiment_results = {
                'attention_profiles': attention_profiles,
                'comparison_report': comparison_report,
                'best_fusion_strategy': best_fusion_strategy,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存結果
            self._save_results()
            
            # 生成可視化
            self._generate_visualizations()
            
            self.logger.info("實驗2完成")
            return self.experiment_results
            
        except Exception as e:
            self.logger.error(f"實驗2執行失敗: {str(e)}")
            raise
    
    def _save_results(self):
        """保存實驗結果"""
        # 保存注意力機制檔案
        profiles_data = {}
        for mechanism, profile in self.attention_profiles.items():
            profiles_data[mechanism] = {
                'mechanism_name': profile.mechanism_name,
                'accuracy': float(profile.accuracy),
                'precision': float(profile.precision),
                'recall': float(profile.recall),
                'f1_score': float(profile.f1_score),
                'computational_complexity': {k: float(v) for k, v in profile.computational_complexity.items()},
                'memory_usage': float(profile.memory_usage),
                'inference_time': float(profile.inference_time),
                'training_time': float(profile.training_time),
                'attention_coverage': float(profile.attention_coverage),
                'attention_focus': float(profile.attention_focus),
                'attention_stability': float(profile.attention_stability),
                'strengths': profile.strengths,
                'weaknesses': profile.weaknesses,
                'use_cases': profile.use_cases,
                'domain_performance': {
                    domain: {k: float(v) for k, v in metrics.items()}
                    for domain, metrics in profile.domain_performance.items()
                }
            }
        
        profiles_path = os.path.join(self.output_dir, "attention_mechanism_profiles.json")
        with open(profiles_path, 'w', encoding='utf-8') as f:
            json.dump(profiles_data, f, indent=2, ensure_ascii=False)
        
        # 保存比較報告
        report_path = os.path.join(self.output_dir, "attention_comparison_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_results['comparison_report'], f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"實驗結果已保存到 {self.output_dir}")
    
    def _generate_visualizations(self):
        """生成可視化圖表"""
        self.logger.info("生成可視化圖表...")
        
        try:
            # 1. 性能比較圖
            self._plot_performance_comparison()
            
            # 2. 計算複雜度比較圖
            self._plot_complexity_comparison()
            
            # 3. 注意力品質比較圖
            self._plot_attention_quality_comparison()
            
            # 4. 領域適應性比較圖
            self._plot_domain_adaptation_comparison()
            
            # 5. 綜合雷達圖
            self._plot_comprehensive_radar_chart()
            
            self.logger.info("可視化圖表生成完成")
            
        except Exception as e:
            self.logger.error(f"可視化生成失敗: {str(e)}")
    
    def _plot_performance_comparison(self):
        """繪製性能比較圖"""
        mechanisms = list(self.attention_profiles.keys())
        accuracies = [profile.accuracy for profile in self.attention_profiles.values()]
        f1_scores = [profile.f1_score for profile in self.attention_profiles.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 準確率比較
        ax1.bar(mechanisms, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('注意力機制準確率比較')
        ax1.set_ylabel('準確率')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # F1分數比較
        ax2.bar(mechanisms, f1_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('注意力機制F1分數比較')
        ax2.set_ylabel('F1分數')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_complexity_comparison(self):
        """繪製計算複雜度比較圖"""
        mechanisms = list(self.attention_profiles.keys())
        parameters = [profile.computational_complexity['total_parameters'] for profile in self.attention_profiles.values()]
        inference_times = [profile.inference_time for profile in self.attention_profiles.values()]
        memory_usage = [profile.memory_usage for profile in self.attention_profiles.values()]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 參數量比較
        ax1.bar(mechanisms, parameters, color='lightgreen', alpha=0.7)
        ax1.set_title('模型參數量比較')
        ax1.set_ylabel('參數量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 推理時間比較
        ax2.bar(mechanisms, inference_times, color='orange', alpha=0.7)
        ax2.set_title('推理時間比較')
        ax2.set_ylabel('推理時間 (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 記憶體使用比較
        ax3.bar(mechanisms, memory_usage, color='purple', alpha=0.7)
        ax3.set_title('記憶體使用比較')
        ax3.set_ylabel('記憶體 (MB)')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'complexity_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_quality_comparison(self):
        """繪製注意力品質比較圖"""
        mechanisms = list(self.attention_profiles.keys())
        coverage = [profile.attention_coverage for profile in self.attention_profiles.values()]
        focus = [profile.attention_focus for profile in self.attention_profiles.values()]
        stability = [profile.attention_stability for profile in self.attention_profiles.values()]
        
        x = np.arange(len(mechanisms))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, coverage, width, label='覆蓋度', alpha=0.7)
        ax.bar(x, focus, width, label='集中度', alpha=0.7)
        ax.bar(x + width, stability, width, label='穩定性', alpha=0.7)
        
        ax.set_title('注意力品質比較')
        ax.set_ylabel('品質分數')
        ax.set_xlabel('注意力機制')
        ax.set_xticks(x)
        ax.set_xticklabels(mechanisms, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'attention_quality_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_domain_adaptation_comparison(self):
        """繪製領域適應性比較圖"""
        mechanisms = list(self.attention_profiles.keys())
        # 使用實際載入的領域
        domains = list(self.attention_profiles.keys()) if hasattr(self, 'attention_profiles') and self.attention_profiles else ['restaurant', 'laptop']
        
        # 計算每個機制在各領域的平均F1分數
        domain_scores = defaultdict(list)
        
        for mechanism in mechanisms:
            profile = self.attention_profiles[mechanism]
            for domain in domains:
                if domain in profile.domain_performance:
                    domain_scores[domain].append(profile.domain_performance[domain]['f1_score'])
                else:
                    domain_scores[domain].append(0.0)
        
        x = np.arange(len(mechanisms))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, domain in enumerate(domains):
            ax.bar(x + i * width, domain_scores[domain], width, 
                  label=f'{domain}領域', alpha=0.7)
        
        ax.set_title('各領域適應性比較')
        ax.set_ylabel('F1分數')
        ax.set_xlabel('注意力機制')
        ax.set_xticks(x + width)
        ax.set_xticklabels(mechanisms, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'domain_adaptation_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_radar_chart(self):
        """繪製綜合雷達圖"""
        mechanisms = list(self.attention_profiles.keys())
        
        # 定義評估維度
        dimensions = ['準確率', '推理速度', '記憶體效率', '注意力品質', '領域適應性']
        
        # 計算各維度的正規化分數
        scores = {}
        for mechanism in mechanisms:
            profile = self.attention_profiles[mechanism]
            
            # 正規化各項指標 (0-1範圍)
            accuracy_score = profile.accuracy
            speed_score = 1 - min(profile.inference_time / 100, 1)  # 推理時間越短越好
            memory_score = 1 - min(profile.memory_usage / 1000, 1)  # 記憶體越少越好
            attention_score = (profile.attention_coverage + profile.attention_focus + profile.attention_stability) / 3
            
            # 計算領域適應性分數 (各領域F1分數的標準差，越小越好)
            domain_f1s = [metrics['f1_score'] for metrics in profile.domain_performance.values()]
            domain_score = 1 - min(np.std(domain_f1s), 0.5) / 0.5
            
            scores[mechanism] = [accuracy_score, speed_score, memory_score, attention_score, domain_score]
        
        # 繪製雷達圖
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]  # 閉合圖形
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(mechanisms)))
        
        for i, (mechanism, score_list) in enumerate(scores.items()):
            score_list += score_list[:1]  # 閉合數據
            ax.plot(angles, score_list, 'o-', linewidth=2, label=mechanism, color=colors[i])
            ax.fill(angles, score_list, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions)
        ax.set_ylim(0, 1)
        ax.set_title('注意力機制綜合評估雷達圖', size=16, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_radar_chart.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_results(self) -> Dict[str, Any]:
        """深度分析實驗結果"""
        if not self.experiment_results:
            raise ValueError("尚未運行實驗，無法分析結果")
        
        analysis = {
            'mechanism_ranking': self._rank_mechanisms(),
            'trade_off_analysis': self._analyze_trade_offs(),
            'scenario_recommendations': self._generate_scenario_recommendations(),
            'future_directions': self._suggest_future_directions()
        }
        
        # 保存分析結果
        analysis_path = os.path.join(self.output_dir, "experiment2_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"深度分析完成，結果保存到: {analysis_path}")
        return analysis
    
    def _rank_mechanisms(self) -> Dict[str, List[str]]:
        """對注意力機制進行排名"""
        mechanisms = list(self.attention_profiles.keys())
        
        rankings = {
            'overall_performance': sorted(mechanisms, 
                                        key=lambda x: self.attention_profiles[x].f1_score, reverse=True),
            'computational_efficiency': sorted(mechanisms,
                                             key=lambda x: self.attention_profiles[x].computational_complexity['total_parameters']),
            'inference_speed': sorted(mechanisms,
                                    key=lambda x: self.attention_profiles[x].inference_time),
            'attention_quality': sorted(mechanisms,
                                      key=lambda x: (self.attention_profiles[x].attention_coverage +
                                                   self.attention_profiles[x].attention_focus +
                                                   self.attention_profiles[x].attention_stability) / 3, reverse=True)
        }
        
        return rankings
    
    def _analyze_trade_offs(self) -> Dict[str, Any]:
        """分析性能權衡"""
        trade_offs = {
            'performance_vs_efficiency': [],
            'quality_vs_speed': [],
            'complexity_vs_effectiveness': []
        }
        
        for mechanism, profile in self.attention_profiles.items():
            # 性能 vs 效率
            performance_score = profile.f1_score
            efficiency_score = 1 / (profile.computational_complexity['total_parameters'] / 1000000)  # 參數效率
            trade_offs['performance_vs_efficiency'].append({
                'mechanism': mechanism,
                'performance': performance_score,
                'efficiency': efficiency_score,
                'balance_score': (performance_score + efficiency_score) / 2
            })
            
            # 品質 vs 速度
            quality_score = (profile.attention_coverage + profile.attention_focus + profile.attention_stability) / 3
            speed_score = 1 / max(profile.inference_time, 1)
            trade_offs['quality_vs_speed'].append({
                'mechanism': mechanism,
                'quality': quality_score,
                'speed': speed_score,
                'balance_score': (quality_score + speed_score) / 2
            })
        
        return trade_offs
    
    def _generate_scenario_recommendations(self) -> Dict[str, Dict[str, str]]:
        """生成場景化推薦"""
        recommendations = {
            'high_performance_required': {
                'recommended_mechanism': '',
                'reason': '',
                'scenario': '需要最高準確率的關鍵應用'
            },
            'resource_constrained': {
                'recommended_mechanism': '',
                'reason': '',
                'scenario': '計算資源受限的環境'
            },
            'real_time_processing': {
                'recommended_mechanism': '',
                'reason': '',
                'scenario': '需要實時處理的應用'
            },
            'multi_domain_deployment': {
                'recommended_mechanism': '',
                'reason': '',
                'scenario': '跨多個領域部署'
            },
            'research_exploration': {
                'recommended_mechanism': '',
                'reason': '',
                'scenario': '研究和探索新方法'
            }
        }
        
        # 基於分析結果填充推薦
        rankings = self._rank_mechanisms()
        
        recommendations['high_performance_required']['recommended_mechanism'] = rankings['overall_performance'][0]
        recommendations['resource_constrained']['recommended_mechanism'] = rankings['computational_efficiency'][0]
        recommendations['real_time_processing']['recommended_mechanism'] = rankings['inference_speed'][0]
        
        return recommendations
    
    def _suggest_future_directions(self) -> List[str]:
        """建議未來研究方向"""
        directions = [
            "探索注意力機制的混合策略，結合不同機制的優勢",
            "研究自適應注意力機制，根據輸入動態選擇最佳策略",
            "開發輕量化注意力機制，降低計算複雜度",
            "研究領域特定的注意力機制優化方法",
            "探索注意力機制的可解釋性分析",
            "開發多尺度注意力機制處理不同長度的序列",
            "研究注意力機制在零樣本學習中的應用"
        ]
        
        return directions
    
    def generate_summary_report(self) -> str:
        """生成實驗摘要報告"""
        if not self.experiment_results:
            return "尚未運行實驗"
        
        report = self.experiment_results['comparison_report']
        
        summary = f"""
實驗2：注意力機制比較實驗 - 摘要報告
=====================================

實驗時間：{self.experiment_results['timestamp']}
測試機制數量：{report['total_mechanisms']}
使用融合策略：{self.experiment_results.get('best_fusion_strategy', '未指定')}

實驗狀態：{report['recommendations'].get('status', '正常完成') if 'status' in report.get('recommendations', {}) else '正常完成'}
{report['recommendations'].get('message', '') if 'message' in report.get('recommendations', {}) else ''}

性能摘要：
- 平均準確率：{report.get('summary_statistics', {}).get('average_accuracy', '無法計算（缺少真實數據）')}
- 平均F1分數：{report.get('summary_statistics', {}).get('average_f1_score', '無法計算（缺少真實數據）')}

最佳機制推薦：{'' if 'status' not in report.get('recommendations', {}) else '（由於缺少真實數據，無法提供具體推薦）'}
{f"- 最佳整體性能：{report['recommendations']['best_overall_performance']['mechanism']} ({report['recommendations']['best_overall_performance']['reason']})" if 'best_overall_performance' in report.get('recommendations', {}) else ''}
{f"- 最快推理速度：{report['recommendations']['fastest_inference']['mechanism']} ({report['recommendations']['fastest_inference']['reason']})" if 'fastest_inference' in report.get('recommendations', {}) else ''}
{f"- 最高參數效率：{report['recommendations']['most_parameter_efficient']['mechanism']} ({report['recommendations']['most_parameter_efficient']['reason']})" if 'most_parameter_efficient' in report.get('recommendations', {}) else ''}
{f"- 最佳注意力品質：{report['recommendations']['best_attention_quality']['mechanism']}" if 'best_attention_quality' in report.get('recommendations', {}) else ''}

性能排名（按F1分數）：{'（由於缺少真實數據，無法提供排名）' if 'f1_score' not in report.get('rankings', {}) else ''}
"""
        if 'rankings' in report and 'f1_score' in report['rankings'] and report['rankings']['f1_score']:
            for i, mechanism in enumerate(report['rankings']['f1_score'], 1):
                f1_score = self.attention_profiles[mechanism].f1_score
                summary += f"{i}. {mechanism}: {f1_score:.4f}\n"
        else:
            summary += "（由於缺少真實數據，無法提供排名）\n"
        
        summary += f"""
主要發現：{'' if 'rankings' in report and 'f1_score' in report['rankings'] and report['rankings']['f1_score'] else '由於缺少真實數據，無法進行注意力機制的性能比較和分析。'}
{f"1. {report['rankings']['f1_score'][0]} 在整體性能上表現最佳" if 'rankings' in report and 'f1_score' in report['rankings'] and report['rankings']['f1_score'] else ''}
{f"2. {report['rankings']['inference_speed'][0]} 在推理速度上領先" if 'rankings' in report and 'inference_speed' in report['rankings'] and report['rankings']['inference_speed'] else ''}
{f"3. {report['rankings']['parameter_efficiency'][0]} 在參數效率上最優" if 'rankings' in report and 'parameter_efficiency' in report['rankings'] and report['rankings']['parameter_efficiency'] else ''}
{f"4. 不同機制在不同應用場景下各有優勢" if 'rankings' in report and 'f1_score' in report['rankings'] and report['rankings']['f1_score'] else ''}

建議：
{f"- 對於性能優先的應用，推薦使用 {report['recommendations']['best_overall_performance']['mechanism']}" if 'best_overall_performance' in report.get('recommendations', {}) else '請提供真實的多領域情感分析數據集以進行嚴謹的注意力機制比較研究。'}
{f"- 對於實時應用，推薦使用 {report['recommendations']['fastest_inference']['mechanism']}" if 'fastest_inference' in report.get('recommendations', {}) else '本系統現在要求使用真實數據以確保學術研究的可靠性。'}
{f"- 對於資源受限環境，推薦使用 {report['recommendations']['most_parameter_efficient']['mechanism']}" if 'most_parameter_efficient' in report.get('recommendations', {}) else ''}
"""
        
        # 保存摘要報告
        summary_path = os.path.join(self.output_dir, "experiment2_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return summary


def create_experiment2_config() -> Dict[str, Any]:
    """創建實驗2的預設配置"""
    return {
        'experiment': {
            'name': 'attention_mechanism_comparison',
            'num_epochs': 5,
            'batch_size': 16
        },
        'model': {
            'hidden_dim': 768,
            'num_attention_heads': 8,
            'dropout': 0.1
        },
        'data': {
            'use_semeval2014': True,
            'use_semeval2016': True,
            'semeval2014_path': 'data/SemEval2014',
            'semeval2016_path': 'data/SemEval2016',
            'max_length': 512
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'best_fusion_strategy': 'weighted'  # 來自實驗1的結果
    }


def integrate_experiments(experiment1_result: Dict[str, Any] = None) -> Dict[str, Any]:
    """整合實驗1和實驗2的結果"""
    # 實驗1配置
    config1 = create_experiment1_config() if 'create_experiment1_config' in globals() else {}
    
    # 實驗2配置
    config2 = create_experiment2_config()
    
    # 如果有實驗1結果，使用其最佳融合策略
    if experiment1_result:
        best_fusion = experiment1_result.get('report', {}).get('analysis', {}).get('best_accuracy', {}).get('strategy')
        if best_fusion:
            config2['best_fusion_strategy'] = best_fusion
    
    return {
        'experiment1_config': config1,
        'experiment2_config': config2,
        'integration_complete': True
    }


if __name__ == "__main__":
    # 示範整合使用
    config = create_experiment2_config()
    controller = Experiment2Controller(config)
    
    # 運行實驗
    results = controller.run_experiment()
    
    # 分析結果
    analysis = controller.analyze_results()
    
    # 生成摘要
    summary = controller.generate_summary_report()
    print(summary)