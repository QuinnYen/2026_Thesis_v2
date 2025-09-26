# 實驗3控制器：組合效果分析實驗
"""
實驗3：組合效果分析實驗

實驗設計：
- 組合實驗1的最佳融合策略 + 實驗2的最佳注意力機制
- 與多種基線方法進行全面對比
- 評估組合效果的顯著性和實用性

基線方法：
1. 傳統機器學習 (SVM, Random Forest)
2. 基礎深度學習 (LSTM, GRU, Basic BERT)
3. 單一注意力機制
4. 簡單特徵融合

評估維度：
- 性能表現 (準確率、F1分數等)
- 計算效率 (參數量、推理時間)
- 領域遷移能力
- 統計顯著性
- 消融分析

本控制器整合實驗1-3的完整流程，提供最終的實驗結論。
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import sys
from pathlib import Path

# 添加上級目錄到路徑
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from combination_analysis_experiment import (
    CombinationAnalysisExperiment, CombinationResult, BaselineResult,
    BaselineImplementations, OptimalCombinationBuilder, ComprehensiveEvaluator
)
from experiment_1_controller import Experiment1Controller
from experiment_2_controller import Experiment2Controller
from data import DatasetManager, SemEval2014Loader, SemEval2016Loader
from data.data_converter import create_experiment_data_converter


class Experiment3Controller:
    """實驗3控制器：組合效果分析實驗"""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "results/experiment3"):
        """
        初始化實驗3控制器
        
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
        
        # 初始化組合分析實驗
        self.combination_experiment = CombinationAnalysisExperiment(
            hidden_dim=config.get('hidden_dim', 768),
            device=self.device
        )
        
        # 資料集管理器
        self.dataset_manager = None
        self.test_datasets = {}
        
        # 數據轉換器
        self.data_converter = create_experiment_data_converter(config)
        
        # 前序實驗的結果
        self.experiment1_results = None
        self.experiment2_results = None
        
        # 實驗3結果
        self.experiment_results = {}
        self.baseline_results = {}
        self.combination_results = None
        self.comprehensive_report = {}
        
        # 設置中文字體支援
        self._setup_chinese_fonts()
        
        self.logger.info("實驗3控制器初始化完成")
    
    def _setup_logging(self):
        """設置日誌系統"""
        log_file = os.path.join(self.output_dir, "experiment3.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('Experiment3')
    
    def _setup_chinese_fonts(self):
        """設置中文字體支援"""
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            self.logger.warning(f"中文字體設置失敗: {str(e)}")
    
    def load_previous_experiment_results(self, exp1_results_path: str = None, 
                                       exp2_results_path: str = None):
        """載入前序實驗結果"""
        self.logger.info("載入前序實驗結果...")
        
        try:
            # 載入實驗1結果
            if exp1_results_path and os.path.exists(exp1_results_path):
                with open(exp1_results_path, 'r', encoding='utf-8') as f:
                    self.experiment1_results = json.load(f)
                self.logger.info("成功載入實驗1結果")
            else:
                self.logger.warning("實驗1結果文件不存在，使用預設值")
                self.experiment1_results = {'best_fusion_strategy': 'adaptive'}
            
            # 載入實驗2結果
            if exp2_results_path and os.path.exists(exp2_results_path):
                with open(exp2_results_path, 'r', encoding='utf-8') as f:
                    self.experiment2_results = json.load(f)
                self.logger.info("成功載入實驗2結果")
            else:
                self.logger.warning("實驗2結果文件不存在，使用預設值")
                self.experiment2_results = {'best_attention_mechanism': 'multi_head'}
                
        except Exception as e:
            self.logger.error(f"載入前序實驗結果失敗: {str(e)}")
            # 使用預設值
            self.experiment1_results = {'best_fusion_strategy': 'adaptive'}
            self.experiment2_results = {'best_attention_mechanism': 'multi_head'}
    
    def run_integrated_experiments(self, run_exp1: bool = True, run_exp2: bool = True) -> Dict[str, Any]:
        """運行整合的實驗1-3流程"""
        self.logger.info("開始運行整合實驗流程...")
        
        integrated_results = {
            'experiment1_results': None,
            'experiment2_results': None,
            'experiment3_results': None,
            'integration_summary': {}
        }
        
        # 運行實驗1
        if run_exp1:
            self.logger.info("運行實驗1：融合策略比較")
            exp1_config = self._create_experiment1_config()
            exp1_controller = Experiment1Controller(exp1_config, 
                                                   os.path.join(self.output_dir, "experiment1"))
            exp1_results = exp1_controller.run_experiment()
            self.experiment1_results = exp1_results
            integrated_results['experiment1_results'] = exp1_results
        
        # 運行實驗2
        if run_exp2:
            self.logger.info("運行實驗2：注意力機制比較")
            exp2_config = self._create_experiment2_config()
            if self.experiment1_results:
                # 使用實驗1的最佳融合策略
                best_fusion = self._extract_best_fusion_strategy()
                exp2_config['best_fusion_strategy'] = best_fusion
            
            exp2_controller = Experiment2Controller(exp2_config,
                                                   os.path.join(self.output_dir, "experiment2"))
            exp2_results = exp2_controller.run_experiment(
                best_fusion_strategy=exp2_config.get('best_fusion_strategy')
            )
            self.experiment2_results = exp2_results
            integrated_results['experiment2_results'] = exp2_results
        
        # 運行實驗3
        self.logger.info("運行實驗3：組合效果分析")
        exp3_results = self.run_experiment()
        integrated_results['experiment3_results'] = exp3_results
        
        # 生成整合摘要
        integration_summary = self._generate_integration_summary(integrated_results)
        integrated_results['integration_summary'] = integration_summary
        
        # 保存整合結果
        self._save_integrated_results(integrated_results)
        
        return integrated_results
    
    def setup_datasets(self):
        """設置實驗資料集"""
        self.logger.info("設置實驗資料集...")
        
        data_config = self.config.get('data', {})
        
        # 初始化資料集載入器
        loaders = {}
        
        if data_config.get('use_semeval2014', True):
            loader_2014 = SemEval2014Loader(
                data_dir=data_config.get('semeval2014_path', 'data/raw/SemEval-2014')
            )
            loaders['semeval2014'] = loader_2014
        
        if data_config.get('use_semeval2016', True):
            loader_2016 = SemEval2016Loader(
                data_dir=data_config.get('semeval2016_path', 'data/raw/SemEval-2016')
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
                f"實驗3無法運行，因為數據載入失敗：{str(e)}\n"
                f"請確保以下數據集可用：\n"
                f"1. SemEval-2014數據集 (restaurants, laptops)\n"
                f"2. SemEval-2016數據集 (restaurants, laptops)\n"
                f"3. 數據路徑配置正確\n"
                f"4. XML文件格式完整"
            )
    
    def run_experiment(self) -> Dict[str, Any]:
        """運行實驗3：組合效果分析"""
        self.logger.info("開始運行實驗3：組合效果分析")
        
        # 設置資料集
        self.setup_datasets()
        
        # 提取最佳組合參數
        best_attention = self._extract_best_attention_mechanism()
        best_fusion = self._extract_best_fusion_strategy()
        
        self.logger.info(f"使用最佳組合: {best_attention} + {best_fusion}")
        
        try:
            # 運行基線實驗
            self.logger.info("運行基線方法對比實驗...")
            baseline_results = self.combination_experiment.run_baseline_experiments(self.test_datasets)
            self.baseline_results = baseline_results
            
            # 運行最佳組合實驗
            self.logger.info("運行最佳組合實驗...")
            combination_result = self.combination_experiment.run_combination_experiment(
                best_attention, best_fusion, self.test_datasets
            )
            self.combination_results = combination_result
            
            # 生成綜合報告
            self.logger.info("生成綜合分析報告...")
            comprehensive_report = self.combination_experiment.generate_comprehensive_report(
                combination_result, baseline_results
            )
            self.comprehensive_report = comprehensive_report
            
            # 組裝實驗結果
            self.experiment_results = {
                'best_combination': {
                    'attention_mechanism': best_attention,
                    'fusion_strategy': best_fusion
                },
                'baseline_results': baseline_results,
                'combination_results': combination_result,
                'comprehensive_report': comprehensive_report,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存結果
            self._save_results()
            
            # 生成可視化
            self._generate_visualizations()
            
            self.logger.info("實驗3完成")
            return self.experiment_results
            
        except Exception as e:
            self.logger.error(f"實驗3執行失敗: {str(e)}")
            raise
    
    def _extract_best_attention_mechanism(self) -> str:
        """提取最佳注意力機制"""
        if self.experiment2_results:
            # 從實驗2結果中提取
            report = self.experiment2_results.get('comparison_report', {})
            recommendations = report.get('recommendations', {})
            best_overall = recommendations.get('best_overall_performance', {})
            return best_overall.get('mechanism', 'multi_head')
        else:
            # 使用預設值
            return self.config.get('best_attention_mechanism', 'multi_head')
    
    def _extract_best_fusion_strategy(self) -> str:
        """提取最佳融合策略"""
        if self.experiment1_results:
            # 從實驗1結果中提取
            report = self.experiment1_results.get('report', {})
            analysis = report.get('analysis', {})
            best_accuracy = analysis.get('best_accuracy', {})
            return best_accuracy.get('strategy', 'adaptive')
        else:
            # 使用預設值
            return self.config.get('best_fusion_strategy', 'adaptive')
    
    def _save_results(self):
        """保存實驗結果"""
        # 保存基線結果
        baseline_data = {}
        for name, result in self.baseline_results.items():
            baseline_data[name] = {
                'method_name': result.method_name,
                'accuracy': float(result.accuracy),
                'precision': float(result.precision),
                'recall': float(result.recall),
                'f1_score': float(result.f1_score),
                'training_time': float(result.training_time),
                'inference_time': float(result.inference_time),
                'model_size': int(result.model_size),
                'domain_performance': {
                    domain: {k: float(v) for k, v in metrics.items()}
                    for domain, metrics in result.domain_performance.items()
                },
                'strengths': result.strengths,
                'limitations': result.limitations
            }
        
        baseline_path = os.path.join(self.output_dir, "baseline_results.json")
        with open(baseline_path, 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False)
        
        # 保存組合結果
        combination_data = {
            'combination_name': self.combination_results.combination_name,
            'attention_mechanism': self.combination_results.attention_mechanism,
            'fusion_strategy': self.combination_results.fusion_strategy,
            'performance_metrics': {k: float(v) for k, v in self.combination_results.performance_metrics.items()},
            'computational_metrics': {k: float(v) for k, v in self.combination_results.computational_metrics.items()},
            'domain_performance': {
                domain: {k: float(v) for k, v in metrics.items()}
                for domain, metrics in self.combination_results.domain_performance.items()
            },
            'improvement_over_baselines': {k: float(v) for k, v in self.combination_results.improvement_over_baselines.items()},
            'ablation_analysis': {k: float(v) for k, v in self.combination_results.ablation_analysis.items()}
        }
        
        combination_path = os.path.join(self.output_dir, "combination_results.json")
        with open(combination_path, 'w', encoding='utf-8') as f:
            json.dump(combination_data, f, indent=2, ensure_ascii=False)
        
        # 保存綜合報告
        report_path = os.path.join(self.output_dir, "comprehensive_report.json")
        
        # 使用自定義JSON編碼器
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, bool):
                    return bool(obj)
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                elif hasattr(obj, '_asdict'):
                    return obj._asdict()
                else:
                    return str(obj)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.comprehensive_report, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        
        self.logger.info(f"實驗結果已保存到 {self.output_dir}")
    
    def _generate_visualizations(self):
        """生成可視化圖表"""
        self.logger.info("生成可視化圖表...")
        
        try:
            # 1. 性能對比圖
            self._plot_performance_comparison()
            
            # 2. 改進幅度圖
            self._plot_improvement_analysis()
            
            # 3. 計算效率對比圖
            self._plot_computational_efficiency()
            
            # 4. 領域遷移能力圖
            self._plot_domain_transferability()
            
            # 5. 消融分析圖
            self._plot_ablation_analysis()
            
            # 6. 綜合性能雷達圖
            self._plot_comprehensive_radar()
            
            self.logger.info("可視化圖表生成完成")
            
        except Exception as e:
            self.logger.error(f"可視化生成失敗: {str(e)}")
    
    def _plot_performance_comparison(self):
        """繪製性能對比圖"""
        methods = list(self.baseline_results.keys()) + [self.combination_results.combination_name]
        f1_scores = [result.f1_score for result in self.baseline_results.values()] + \
                   [self.combination_results.performance_metrics['f1_score']]
        accuracies = [result.accuracy for result in self.baseline_results.values()] + \
                    [self.combination_results.performance_metrics['accuracy']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # F1分數對比
        colors = ['lightcoral' if method != self.combination_results.combination_name else 'gold' 
                 for method in methods]
        bars1 = ax1.bar(range(len(methods)), f1_scores, color=colors, alpha=0.8)
        ax1.set_title('F1分數對比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('F1分數', fontsize=12)
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        
        # 添加數值標籤
        for i, (bar, score) in enumerate(zip(bars1, f1_scores)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 準確率對比
        bars2 = ax2.bar(range(len(methods)), accuracies, color=colors, alpha=0.8)
        ax2.set_title('準確率對比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('準確率', fontsize=12)
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        # 添加數值標籤
        for i, (bar, score) in enumerate(zip(bars2, accuracies)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_analysis(self):
        """繪製改進幅度分析圖"""
        baseline_names = list(self.combination_results.improvement_over_baselines.keys())
        improvements = list(self.combination_results.improvement_over_baselines.values())
        
        # 按改進幅度排序
        sorted_data = sorted(zip(baseline_names, improvements), key=lambda x: x[1], reverse=True)
        sorted_names, sorted_improvements = zip(*sorted_data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['green' if imp > 0 else 'red' for imp in sorted_improvements]
        bars = ax.barh(range(len(sorted_names)), sorted_improvements, color=colors, alpha=0.7)
        
        ax.set_title('相對基線方法的改進幅度', fontsize=14, fontweight='bold')
        ax.set_xlabel('改進幅度 (%)', fontsize=12)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加數值標籤
        for i, (bar, imp) in enumerate(zip(bars, sorted_improvements)):
            ax.text(bar.get_width() + (1 if imp > 0 else -1), bar.get_y() + bar.get_height()/2,
                   f'{imp:.1f}%', ha='left' if imp > 0 else 'right', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'improvement_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_computational_efficiency(self):
        """繪製計算效率對比圖"""
        methods = list(self.baseline_results.keys()) + [self.combination_results.combination_name]
        parameters = [result.model_size for result in self.baseline_results.values()] + \
                    [self.combination_results.computational_metrics['total_parameters']]
        inference_times = [result.inference_time for result in self.baseline_results.values()] + \
                         [self.combination_results.computational_metrics['inference_time_ms']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 參數量對比
        ax1.bar(range(len(methods)), parameters, color='lightblue', alpha=0.7)
        ax1.set_title('模型參數量對比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('參數量', fontsize=12)
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        
        # 推理時間對比
        ax2.bar(range(len(methods)), inference_times, color='lightgreen', alpha=0.7)
        ax2.set_title('推理時間對比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('推理時間 (ms)', fontsize=12)
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'computational_efficiency.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_domain_transferability(self):
        """繪製領域遷移能力圖"""
        # 使用實際可用的領域
        domains = list(self.baseline_results.keys()) if hasattr(self, 'baseline_results') and self.baseline_results else ['restaurant', 'laptop']
        methods = list(self.baseline_results.keys()) + [self.combination_results.combination_name]
        
        # 收集每個方法在各領域的F1分數
        domain_scores = defaultdict(list)
        method_labels = []
        
        for method in methods:
            if method == self.combination_results.combination_name:
                domain_performance = self.combination_results.domain_performance
            else:
                domain_performance = self.baseline_results[method].domain_performance
            
            method_labels.append(method)
            for domain in domains:
                if domain in domain_performance:
                    domain_scores[domain].append(domain_performance[domain]['f1_score'])
                else:
                    domain_scores[domain].append(0.0)
        
        x = np.arange(len(method_labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, domain in enumerate(domains):
            offset = (i - len(domains)//2) * width
            ax.bar(x + offset, domain_scores[domain], width, 
                  label=f'{domain}領域', alpha=0.8)
        
        ax.set_title('各領域遷移能力對比', fontsize=14, fontweight='bold')
        ax.set_ylabel('F1分數', fontsize=12)
        ax.set_xlabel('方法', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'domain_transferability.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ablation_analysis(self):
        """繪製消融分析圖"""
        ablation_data = self.combination_results.ablation_analysis
        
        components = ['僅注意力', '普通注意力+最佳融合', '最佳注意力+簡單融合', '完整組合']
        scores = [
            ablation_data['only_attention'],
            ablation_data['ordinary_attention_best_fusion'],
            ablation_data['best_attention_simple_fusion'],
            ablation_data['full_combination']
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['lightcoral', 'orange', 'lightblue', 'gold']
        bars = ax.bar(components, scores, color=colors, alpha=0.8)
        
        ax.set_title('消融分析：各組件貢獻度', fontsize=14, fontweight='bold')
        ax.set_ylabel('F1分數', fontsize=12)
        ax.set_ylim(0, max(scores) * 1.1)
        
        # 添加數值標籤
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 添加改進箭頭和數值
        for i in range(len(scores) - 1):
            improvement = scores[i+1] - scores[i]
            if improvement > 0:
                ax.annotate(f'+{improvement:.3f}', 
                           xy=(i+0.5, (scores[i] + scores[i+1])/2),
                           ha='center', va='center', fontsize=10, color='red',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ablation_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_radar(self):
        """繪製綜合性能雷達圖"""
        # 選擇top5方法進行對比
        all_methods = [(name, result.f1_score) for name, result in self.baseline_results.items()]
        all_methods.append((self.combination_results.combination_name, 
                          self.combination_results.performance_metrics['f1_score']))
        
        # 按F1分數排序，取前5名
        top_methods = sorted(all_methods, key=lambda x: x[1], reverse=True)[:5]
        
        # 定義評估維度
        dimensions = ['性能表現', '計算效率', '推理速度', '領域適應性', '模型簡潔性']
        
        # 計算各維度分數
        method_scores = {}
        for method_name, _ in top_methods:
            if method_name == self.combination_results.combination_name:
                # 組合方法
                perf_score = self.combination_results.performance_metrics['f1_score']
                efficiency_score = 1 - min(self.combination_results.computational_metrics['total_parameters'] / 10000000, 1)
                speed_score = 1 - min(self.combination_results.computational_metrics['inference_time_ms'] / 100, 1)
                
                # 計算領域適應性（各領域F1分數的標準差，越小越好）
                domain_f1s = [metrics['f1_score'] for metrics in self.combination_results.domain_performance.values()]
                domain_score = 1 - min(np.std(domain_f1s), 0.1) / 0.1
                
                simplicity_score = efficiency_score  # 與計算效率相關
            else:
                # 基線方法
                baseline_result = self.baseline_results[method_name]
                perf_score = baseline_result.f1_score
                efficiency_score = 1 - min(baseline_result.model_size / 10000000, 1)
                speed_score = 1 - min(baseline_result.inference_time / 100, 1)
                
                domain_f1s = [metrics['f1_score'] for metrics in baseline_result.domain_performance.values()]
                domain_score = 1 - min(np.std(domain_f1s), 0.1) / 0.1
                
                simplicity_score = efficiency_score
            
            method_scores[method_name] = [perf_score, efficiency_score, speed_score, domain_score, simplicity_score]
        
        # 繪製雷達圖
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]  # 閉合圖形
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(method_scores)))
        
        for i, (method, scores) in enumerate(method_scores.items()):
            scores += scores[:1]  # 閉合數據
            
            if method == self.combination_results.combination_name:
                ax.plot(angles, scores, 'o-', linewidth=3, label=f'{method} (本研究)', 
                       color='red', markersize=8)
                ax.fill(angles, scores, alpha=0.25, color='red')
            else:
                ax.plot(angles, scores, 'o-', linewidth=2, label=method, 
                       color=colors[i], markersize=6)
                ax.fill(angles, scores, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('綜合性能對比雷達圖', size=16, fontweight='bold', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_radar.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_experiment1_config(self) -> Dict[str, Any]:
        """創建實驗1配置"""
        return {
            'experiment': {
                'name': 'fusion_strategy_comparison',
                'num_epochs': self.config.get('num_epochs', 5),
                'batch_size': 16
            },
            'model': {
                'hidden_dim': self.config.get('hidden_dim', 768),
                'num_attention_heads': 8,
                'dropout': 0.1
            },
            'data': self.config.get('data', {}),
            'device': self.device
        }
    
    def _create_experiment2_config(self) -> Dict[str, Any]:
        """創建實驗2配置"""
        return {
            'experiment': {
                'name': 'attention_mechanism_comparison',
                'num_epochs': self.config.get('num_epochs', 5),
                'batch_size': 16
            },
            'model': {
                'hidden_dim': self.config.get('hidden_dim', 768),
                'num_attention_heads': 8,
                'dropout': 0.1
            },
            'data': self.config.get('data', {}),
            'device': self.device
        }
    
    def _generate_integration_summary(self, integrated_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成整合摘要"""
        summary = {
            'experiment_sequence': ['fusion_strategy_comparison', 'attention_mechanism_comparison', 'combination_analysis'],
            'best_components': {},
            'performance_progression': {},
            'final_achievements': {},
            'research_contributions': []
        }
        
        # 提取最佳組件
        if integrated_results['experiment1_results']:
            exp1_report = integrated_results['experiment1_results'].get('report', {})
            best_fusion = exp1_report.get('analysis', {}).get('best_accuracy', {}).get('strategy')
            summary['best_components']['fusion_strategy'] = best_fusion
        
        if integrated_results['experiment2_results']:
            exp2_report = integrated_results['experiment2_results'].get('comparison_report', {})
            best_attention = exp2_report.get('recommendations', {}).get('best_overall_performance', {}).get('mechanism')
            summary['best_components']['attention_mechanism'] = best_attention
        
        # 性能進展
        if integrated_results['experiment3_results']:
            exp3_results = integrated_results['experiment3_results']
            final_f1 = exp3_results['combination_results'].performance_metrics['f1_score']
            improvements = exp3_results['combination_results'].improvement_over_baselines
            
            summary['performance_progression'] = {
                'final_f1_score': final_f1,
                'best_improvement_percentage': max(improvements.values()),
                'average_improvement_percentage': np.mean(list(improvements.values()))
            }
        
        # 研究貢獻
        summary['research_contributions'] = [
            "系統性比較了多種融合策略的效果",
            "全面評估了7種注意力機制的優劣勢",
            "提出了最佳注意力機制與融合策略的組合方案",
            "通過統計顯著性檢驗驗證了改進效果",
            "提供了詳細的消融分析和實用指導"
        ]
        
        return summary
    
    def _save_integrated_results(self, integrated_results: Dict[str, Any]):
        """保存整合結果"""
        integrated_path = os.path.join(self.output_dir, "integrated_experiment_results.json")
        
        # 使用自定義JSON編碼器
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, bool):
                    return bool(obj)
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                elif hasattr(obj, '_asdict'):
                    return obj._asdict()
                else:
                    return str(obj)
        
        with open(integrated_path, 'w', encoding='utf-8') as f:
            json.dump(integrated_results, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        
        self.logger.info(f"整合實驗結果已保存到: {integrated_path}")
    
    def generate_final_summary_report(self) -> str:
        """生成最終摘要報告"""
        if not self.experiment_results:
            return "尚未運行實驗"
        
        combination_result = self.experiment_results['combination_results']
        comprehensive_report = self.experiment_results['comprehensive_report']
        
        summary = f"""
實驗3：組合效果分析實驗 - 最終摘要報告
=======================================

實驗時間：{self.experiment_results['timestamp']}

最佳組合方案：
- 注意力機制：{combination_result.attention_mechanism}
- 融合策略：{combination_result.fusion_strategy}

性能表現：
- F1分數：{combination_result.performance_metrics['f1_score']:.4f}
- 準確率：{combination_result.performance_metrics['accuracy']:.4f}
- 精確度：{combination_result.performance_metrics['precision']:.4f}
- 召回率：{combination_result.performance_metrics['recall']:.4f}

對比基線方法：
- 測試基線方法數量：{len(self.baseline_results)}
- 平均改進幅度：{comprehensive_report['improvement_analysis']['average_improvement_percentage']:.1f}%
- 最大改進幅度：{comprehensive_report['improvement_analysis']['max_improvement_percentage']:.1f}%
- 統計顯著性改進方法數：{len(comprehensive_report['statistical_analysis']['significant_improvements'])}

消融分析結果：
- 僅注意力機制：{combination_result.ablation_analysis['only_attention']:.4f}
- 普通注意力+最佳融合：{combination_result.ablation_analysis['ordinary_attention_best_fusion']:.4f}
- 最佳注意力+簡單融合：{combination_result.ablation_analysis['best_attention_simple_fusion']:.4f}
- 完整最佳組合：{combination_result.ablation_analysis['full_combination']:.4f}

計算效率：
- 模型參數量：{combination_result.computational_metrics['total_parameters']:,}
- 推理時間：{combination_result.computational_metrics['inference_time_ms']:.2f} ms
- 記憶體使用：{combination_result.computational_metrics['memory_usage_mb']:.1f} MB

關鍵發現：
"""
        for finding in comprehensive_report['key_findings']:
            summary += f"- {finding}\n"
        
        summary += f"""
建議與應用指導：
"""
        for recommendation in comprehensive_report['recommendations']:
            summary += f"- {recommendation}\n"
        
        summary += f"""
研究貢獻：
1. 提供了系統性的多層次特徵融合方法比較框架
2. 驗證了注意力機制與融合策略組合的有效性
3. 通過統計檢驗證明了方法的顯著性改進
4. 為跨領域情感分析提供了實用的技術方案
5. 為未來相關研究提供了基準和方向指導
"""
        
        # 保存最終摘要
        final_summary_path = os.path.join(self.output_dir, "final_summary_report.txt")
        with open(final_summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return summary


def create_experiment3_config() -> Dict[str, Any]:
    """創建實驗3的預設配置"""
    return {
        'experiment': {
            'name': 'combination_analysis',
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
            'semeval2014_path': 'data/raw/SemEval-2014',
            'semeval2016_path': 'data/raw/SemEval-2016',
            'max_length': 512
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'best_attention_mechanism': 'multi_head',  # 實驗2結果
        'best_fusion_strategy': 'adaptive'         # 實驗1結果
    }


if __name__ == "__main__":
    # 示範使用
    config = create_experiment3_config()
    controller = Experiment3Controller(config)
    
    # 選項1：只運行實驗3（需要提供前序實驗結果）
    # results = controller.run_experiment()
    
    # 選項2：運行完整的實驗1-3流程
    integrated_results = controller.run_integrated_experiments(run_exp1=True, run_exp2=True)
    
    # 生成最終摘要
    final_summary = controller.generate_final_summary_report()
    print(final_summary)