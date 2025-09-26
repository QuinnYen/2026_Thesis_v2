# 實驗1控制器：融合策略比較實驗
"""
實驗1：融合策略比較測試

實驗設計：
- 固定條件：Cross Attention 注意力機制
- 變數：融合策略 (Concatenation, Weighted, Gated, Adaptive, Cross-Attention)
- 評估指標：準確率、計算複雜度、跨領域穩定性

本控制器整合到主系統中，提供完整的實驗流程管理。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path
import logging
from datetime import datetime

import sys
from pathlib import Path

# 添加上級目錄到路徑
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fusion_strategy_experiment import (
    FusionStrategyExperiment, ExperimentResult, 
    ComputationalComplexityAnalyzer, CrossDomainStabilityEvaluator
)
from data import DatasetManager, SemEval2014Loader, SemEval2016Loader
from data.data_converter import create_experiment_data_converter
from models import TrainingManager


class Experiment1Controller:
    """實驗1控制器：融合策略比較實驗"""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "results/experiment1"):
        """
        初始化實驗1控制器
        
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
        
        # 初始化實驗組件
        self.fusion_experiment = FusionStrategyExperiment(
            hidden_dim=config.get('hidden_dim', 768),
            device=self.device
        )
        
        # 資料集管理器
        self.dataset_manager = None
        self.train_datasets = {}
        self.test_datasets = {}
        
        # 數據轉換器
        self.data_converter = create_experiment_data_converter(config)
        
        # 實驗結果
        self.experiment_results = {}
        
        self.logger.info("實驗1控制器初始化完成")
    
    def _setup_logging(self):
        """設置日誌系統"""
        log_file = os.path.join(self.output_dir, "experiment1.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('Experiment1')
    
    def setup_datasets(self):
        """設置實驗資料集"""
        self.logger.info("設置實驗資料集...")
        
        data_config = self.config.get('data', {})
        
        # 初始化資料集載入器
        loaders = {}
        
        # SemEval 2014 資料集
        if data_config.get('use_semeval2014', True):
            loader_2014 = SemEval2014Loader(
                data_dir=data_config.get('semeval2014_path', 'data/raw/SemEval-2014')
            )
            loaders['semeval2014'] = loader_2014
        
        # SemEval 2016 資料集  
        if data_config.get('use_semeval2016', True):
            loader_2016 = SemEval2016Loader(
                data_dir=data_config.get('semeval2016_path', 'data/raw/SemEval-2016')
            )
            loaders['semeval2016'] = loader_2016
        
        # 創建資料集管理器
        self.dataset_manager = DatasetManager(data_config.get('base_data_dir', 'data'))
        
        # 載入和分割資料
        try:
            # 載入所有資料
            all_data = self.dataset_manager.load_all_datasets()
            
            # 按領域分割訓練和測試資料
            domains = set()
            for dataset_name, domain_data in all_data.items():
                domains.update(domain_data.keys())
            
            for domain in domains:
                domain_train_data = []
                domain_test_data = []
                
                for dataset_name, domain_data in all_data.items():
                    if domain in domain_data:
                        # 使用實際的訓練/測試分割
                        domain_splits = domain_data[domain]
                        if 'train' in domain_splits:
                            domain_train_data.extend(domain_splits['train'])
                        if 'test' in domain_splits:
                            domain_test_data.extend(domain_splits['test'])
                
                # 轉換AspectSentiment數據為features/labels格式
                if domain_train_data:
                    self.train_datasets[domain] = self.data_converter.convert_to_features_labels(
                        domain_train_data, use_bert=True
                    )
                
                if domain_test_data:
                    self.test_datasets[domain] = self.data_converter.convert_to_features_labels(
                        domain_test_data, use_bert=True
                    )
            
            self.logger.info(f"成功載入 {len(domains)} 個領域的資料")
            
        except Exception as e:
            self.logger.error(f"載入真實資料集失敗: {str(e)}")
            raise ValueError(
                f"實驗1無法運行，因為數據載入失敗：{str(e)}\n"
                f"請確保以下數據集可用：\n"
                f"1. SemEval-2014數據集 (restaurants, laptops)\n"
                f"2. SemEval-2016數據集 (restaurants, laptops)\n"
                f"3. 數據路徑配置正確\n"
                f"4. XML文件格式完整"
            )
    
    def run_experiment(self) -> Dict[str, Any]:
        """運行實驗1：融合策略比較"""
        self.logger.info("開始運行實驗1：融合策略比較")
        
        # 設置資料集
        self.setup_datasets()
        
        # 獲取實驗參數
        experiment_config = self.config.get('experiment', {})
        num_epochs = experiment_config.get('num_epochs', 10)
        
        # 運行融合策略比較實驗
        self.logger.info("開始融合策略比較測試...")
        
        try:
            results = self.fusion_experiment.run_all_experiments(
                self.train_datasets,
                self.test_datasets,
                num_epochs
            )
            
            # 生成實驗報告
            report = self.fusion_experiment.generate_experiment_report(results)
            
            # 保存結果
            self.fusion_experiment.save_results(results, report, self.output_dir)
            
            # 存儲結果
            self.experiment_results = {
                'results': results,
                'report': report,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("實驗1完成")
            return self.experiment_results
            
        except Exception as e:
            self.logger.error(f"實驗1執行失敗: {str(e)}")
            raise
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析實驗結果"""
        if not self.experiment_results:
            raise ValueError("尚未運行實驗，無法分析結果")
        
        results = self.experiment_results['results']
        report = self.experiment_results['report']
        
        analysis = {
            'performance_analysis': self._analyze_performance(results),
            'complexity_analysis': self._analyze_complexity(results),
            'stability_analysis': self._analyze_stability(results),
            'efficiency_analysis': self._analyze_efficiency(results),
            'recommendations': self._generate_recommendations(results, report)
        }
        
        # 保存分析結果
        analysis_path = os.path.join(self.output_dir, "experiment1_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"實驗分析完成，結果保存到: {analysis_path}")
        return analysis
    
    def _analyze_performance(self, results: Dict[str, ExperimentResult]) -> Dict[str, Any]:
        """分析性能表現"""
        performance_metrics = {}
        
        for strategy, result in results.items():
            performance_metrics[strategy] = {
                'accuracy': result.accuracy,
                'f1_score': result.f1_score,
                'precision': result.precision,
                'recall': result.recall
            }
        
        # 找出最佳性能策略
        best_accuracy = max(results.items(), key=lambda x: x[1].accuracy)
        best_f1 = max(results.items(), key=lambda x: x[1].f1_score)
        
        return {
            'metrics': performance_metrics,
            'best_accuracy': {'strategy': best_accuracy[0], 'value': best_accuracy[1].accuracy},
            'best_f1': {'strategy': best_f1[0], 'value': best_f1[1].f1_score},
            'performance_ranking': sorted(
                results.keys(), 
                key=lambda x: results[x].f1_score, 
                reverse=True
            )
        }
    
    def _analyze_complexity(self, results: Dict[str, ExperimentResult]) -> Dict[str, Any]:
        """分析計算複雜度"""
        complexity_metrics = {}
        
        for strategy, result in results.items():
            complexity_metrics[strategy] = {
                'parameters': result.model_parameters,
                'memory_mb': result.memory_usage,
                'inference_time_ms': result.inference_time,
                'training_time_s': result.training_time
            }
        
        # 找出最高效策略
        most_efficient = min(results.items(), key=lambda x: x[1].model_parameters)
        fastest_inference = min(results.items(), key=lambda x: x[1].inference_time)
        
        return {
            'metrics': complexity_metrics,
            'most_efficient': {'strategy': most_efficient[0], 'parameters': most_efficient[1].model_parameters},
            'fastest_inference': {'strategy': fastest_inference[0], 'time_ms': fastest_inference[1].inference_time},
            'efficiency_ranking': sorted(
                results.keys(),
                key=lambda x: results[x].model_parameters
            )
        }
    
    def _analyze_stability(self, results: Dict[str, ExperimentResult]) -> Dict[str, Any]:
        """分析跨領域穩定性"""
        stability_metrics = {}
        
        for strategy, result in results.items():
            stability_metrics[strategy] = {
                'cross_domain_stability': result.cross_domain_stability,
                'domain_performance': result.domain_specific_metrics
            }
        
        # 找出最穩定策略
        most_stable = max(results.items(), key=lambda x: x[1].cross_domain_stability)
        
        return {
            'metrics': stability_metrics,
            'most_stable': {'strategy': most_stable[0], 'stability': most_stable[1].cross_domain_stability},
            'stability_ranking': sorted(
                results.keys(),
                key=lambda x: results[x].cross_domain_stability,
                reverse=True
            )
        }
    
    def _analyze_efficiency(self, results: Dict[str, ExperimentResult]) -> Dict[str, Any]:
        """分析效率平衡"""
        efficiency_scores = {}
        
        # 計算綜合效率分數 (性能 vs 複雜度)
        for strategy, result in results.items():
            # 正規化指標
            performance_score = result.f1_score  # 已經是 0-1 範圍
            
            # 計算效率分數 (參數越少越好)
            max_params = max(r.model_parameters for r in results.values())
            efficiency_score = 1 - (result.model_parameters / max_params)
            
            # 計算速度分數 (推理時間越短越好)
            max_time = max(r.inference_time for r in results.values())
            speed_score = 1 - (result.inference_time / max_time)
            
            # 綜合分數 (可調整權重)
            composite_score = (
                0.5 * performance_score +
                0.3 * efficiency_score + 
                0.2 * speed_score
            )
            
            efficiency_scores[strategy] = {
                'performance_score': performance_score,
                'efficiency_score': efficiency_score,
                'speed_score': speed_score,
                'composite_score': composite_score
            }
        
        # 找出最佳平衡策略
        best_balance = max(efficiency_scores.items(), key=lambda x: x[1]['composite_score'])
        
        return {
            'scores': efficiency_scores,
            'best_balance': {'strategy': best_balance[0], 'score': best_balance[1]['composite_score']},
            'balance_ranking': sorted(
                efficiency_scores.keys(),
                key=lambda x: efficiency_scores[x]['composite_score'],
                reverse=True
            )
        }
    
    def _generate_recommendations(self, results: Dict[str, ExperimentResult], 
                                report: Dict[str, Any]) -> Dict[str, Any]:
        """生成實驗建議"""
        recommendations = {
            'best_overall': None,
            'best_for_accuracy': None,
            'best_for_efficiency': None,
            'best_for_stability': None,
            'trade_offs': [],
            'next_steps': []
        }
        
        # 基於分析結果生成建議
        best_accuracy = report['analysis']['best_accuracy']['strategy']
        best_stability = report['analysis']['best_stability']['strategy']
        most_efficient = report['analysis']['most_efficient']['strategy']
        fastest = report['analysis']['fastest_inference']['strategy']
        
        recommendations['best_for_accuracy'] = best_accuracy
        recommendations['best_for_stability'] = best_stability
        recommendations['best_for_efficiency'] = most_efficient
        
        # 綜合建議
        strategy_scores = {}
        for strategy in results.keys():
            score = 0
            if strategy == best_accuracy:
                score += 3
            if strategy == best_stability:
                score += 2
            if strategy == most_efficient:
                score += 1
            strategy_scores[strategy] = score
        
        recommendations['best_overall'] = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        # 權衡分析
        recommendations['trade_offs'] = [
            f"{best_accuracy} 策略準確率最高，但可能計算複雜度較高",
            f"{most_efficient} 策略最高效，但可能性能有所犧牲",
            f"{best_stability} 策略跨領域穩定性最佳，適合多領域應用"
        ]
        
        # 後續步驟建議
        recommendations['next_steps'] = [
            "基於最佳融合策略進行實驗2：注意力機制比較",
            "進一步優化最佳策略的超參數設置",
            "在更大規模資料集上驗證結果",
            "考慮策略組合或混合方法"
        ]
        
        return recommendations
    
    def generate_summary_report(self) -> str:
        """生成實驗摘要報告"""
        if not self.experiment_results:
            return "尚未運行實驗"
        
        report = self.experiment_results['report']
        results = self.experiment_results['results']
        
        summary = f"""
實驗1：融合策略比較實驗 - 摘要報告
=====================================

實驗時間：{self.experiment_results['timestamp']}
測試策略數量：{len(results)}

性能摘要：
- 平均準確率：{report['summary'].get('average_accuracy', '無法計算（缺少真實數據）')}
- 平均跨領域穩定性：{report['summary'].get('average_stability', '無法計算（缺少真實數據）')}

實驗狀態：{report['analysis'].get('status', '正常完成') if 'status' in report['analysis'] else '正常完成'}
{report['analysis'].get('message', '') if 'message' in report['analysis'] else ''}

最佳策略：{'' if 'status' not in report['analysis'] else '（由於缺少真實數據，無法提供具體分析）'}
{f"- 準確率最高：{report['analysis']['best_accuracy']['strategy']} ({report['analysis']['best_accuracy']['value']:.4f})" if 'best_accuracy' in report['analysis'] else ''}
{f"- 穩定性最佳：{report['analysis']['best_stability']['strategy']} ({report['analysis']['best_stability']['value']:.4f})" if 'best_stability' in report['analysis'] else ''}
{f"- 效率最高：{report['analysis']['most_efficient']['strategy']} ({report['analysis']['most_efficient']['parameters']} 參數)" if 'most_efficient' in report['analysis'] else ''}
{f"- 推理最快：{report['analysis']['fastest_inference']['strategy']} ({report['analysis']['fastest_inference']['time_ms']:.2f} ms)" if 'fastest_inference' in report['analysis'] else ''}

準確率排名：
"""
        if 'rankings' in report and 'accuracy' in report['rankings']:
            for i, strategy in enumerate(report['rankings']['accuracy'], 1):
                accuracy = results[strategy].accuracy
                summary += f"{i}. {strategy}: {accuracy:.4f}\n"
        else:
            summary += "（由於缺少真實數據，無法提供排名）\n"
        
        if 'best_accuracy' in report.get('analysis', {}):
            summary += f"""
建議：
基於綜合考慮準確率、穩定性和效率，建議在後續實驗中優先考慮 {report['analysis']['best_accuracy']['strategy']} 策略。
"""
        else:
            summary += """
建議：
請提供真實的多領域情感分析數據集以進行嚴謹的融合策略比較研究。
本系統現在要求使用真實數據以確保學術研究的可靠性。
"""
        
        # 保存摘要報告
        summary_path = os.path.join(self.output_dir, "experiment1_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return summary


def create_experiment1_config() -> Dict[str, Any]:
    """創建實驗1的預設配置"""
    return {
        'experiment': {
            'name': 'fusion_strategy_comparison',
            'num_epochs': 10,
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
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }


if __name__ == "__main__":
    # 示範使用
    config = create_experiment1_config()
    controller = Experiment1Controller(config)
    
    # 運行實驗
    results = controller.run_experiment()
    
    # 分析結果
    analysis = controller.analyze_results()
    
    # 生成摘要
    summary = controller.generate_summary_report()
    print(summary)