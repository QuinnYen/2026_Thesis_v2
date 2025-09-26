"""
實驗管理器
負責批量實驗執行、結果對比分析、超參數搜索、實驗重現驗證
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from dataclasses import dataclass

from .config_manager import ConfigManager
from .logging_system import logging_manager, ExperimentLogger
from .utility_functions import set_random_seed, save_json, load_json


@dataclass
class ExperimentResult:
    """實驗結果數據結構"""
    experiment_name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    start_time: str
    end_time: str
    duration: float
    status: str  # 'completed', 'failed', 'running'
    error_message: Optional[str] = None
    

class HyperparameterSearcher:
    """超參數搜索器"""
    
    def __init__(self):
        self.search_methods = {
            'grid_search': self._grid_search,
            'random_search': self._random_search,
            'bayesian_search': self._bayesian_search
        }
    
    def _grid_search(self, param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """網格搜索"""
        keys = list(param_space.keys())
        values = list(param_space.values())
        
        configurations = []
        for combination in itertools.product(*values):
            config = dict(zip(keys, combination))
            configurations.append(config)
        
        return configurations
    
    def _random_search(self, param_space: Dict[str, List[Any]], 
                      n_samples: int = 10) -> List[Dict[str, Any]]:
        """隨機搜索"""
        import random
        
        configurations = []
        for _ in range(n_samples):
            config = {}
            for param, values in param_space.items():
                config[param] = random.choice(values)
            configurations.append(config)
        
        return configurations
    
    def _bayesian_search(self, param_space: Dict[str, List[Any]], 
                        n_samples: int = 10) -> List[Dict[str, Any]]:
        """貝氏最佳化搜索（簡化版本）"""
        # 這裡實現簡化版本，實際使用時可以整合 scikit-optimize 等函式庫
        return self._random_search(param_space, n_samples)
    
    def generate_configurations(self, param_space: Dict[str, List[Any]],
                              method: str = 'grid_search',
                              n_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        生成超參數配置
        
        Args:
            param_space: 參數空間
            method: 搜索方法
            n_samples: 樣本數量（隨機搜索和貝氏搜索使用）
            
        Returns:
            配置列表
        """
        if method not in self.search_methods:
            raise ValueError(f"不支援的搜索方法: {method}")
        
        if method in ['random_search', 'bayesian_search'] and n_samples is None:
            n_samples = 10
        
        if method == 'grid_search':
            return self.search_methods[method](param_space)
        else:
            return self.search_methods[method](param_space, n_samples)


class ExperimentManager:
    """實驗管理器"""
    
    def __init__(self, base_config_dir: str = "configs",
                 results_dir: str = "results"):
        """
        初始化實驗管理器
        
        Args:
            base_config_dir: 配置文件目錄
            results_dir: 結果存放目錄
        """
        self.config_manager = ConfigManager(base_config_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.hyperparameter_searcher = HyperparameterSearcher()
        self.experiment_results: List[ExperimentResult] = []
        
        # 載入已有的實驗結果
        self.load_experiment_history()
    
    def create_experiment_series(self, base_experiment_name: str,
                               param_space: Dict[str, List[Any]],
                               search_method: str = 'grid_search',
                               n_samples: Optional[int] = None) -> List[str]:
        """
        建立實驗系列
        
        Args:
            base_experiment_name: 基礎實驗名稱
            param_space: 參數空間
            search_method: 搜索方法
            n_samples: 樣本數量
            
        Returns:
            實驗名稱列表
        """
        # 生成超參數配置
        configurations = self.hyperparameter_searcher.generate_configurations(
            param_space, search_method, n_samples
        )
        
        # 獲取基礎配置
        base_config = self.config_manager.get_default_config()
        
        experiment_names = []
        for i, param_config in enumerate(configurations):
            # 建立實驗名稱
            experiment_name = f"{base_experiment_name}_exp_{i+1:03d}"
            
            # 合併配置
            experiment_config = self.config_manager.merge_configs(
                base_config, {'hyperparameters': param_config}
            )
            
            # 添加實驗元資訊
            experiment_config['experiment_info'] = {
                'series_name': base_experiment_name,
                'experiment_id': i + 1,
                'total_experiments': len(configurations),
                'search_method': search_method,
                'parameter_config': param_config
            }
            
            # 保存配置
            self.config_manager.save_config(experiment_config, experiment_name)
            experiment_names.append(experiment_name)
        
        return experiment_names
    
    def run_single_experiment(self, experiment_name: str,
                            experiment_function: Callable[[Dict[str, Any], ExperimentLogger], Dict[str, float]],
                            force_rerun: bool = False) -> ExperimentResult:
        """
        執行單個實驗
        
        Args:
            experiment_name: 實驗名稱
            experiment_function: 實驗執行函數
            force_rerun: 是否強制重新執行
            
        Returns:
            實驗結果
        """
        # 檢查是否已經執行過
        if not force_rerun:
            existing_result = self._get_existing_result(experiment_name)
            if existing_result and existing_result.status == 'completed':
                print(f"實驗 {experiment_name} 已完成，跳過執行")
                return existing_result
        
        # 載入實驗配置
        config = self.config_manager.get_experiment_config(experiment_name)
        
        # 設定隨機種子（如果有指定）
        if 'random_seed' in config:
            set_random_seed(config['random_seed'])
        
        # 建立日誌記錄器
        logger = logging_manager.get_logger(experiment_name)
        
        # 記錄實驗開始
        start_time = datetime.now()
        logger.log_experiment_start(config)
        logger.start_performance_monitoring()
        
        try:
            # 執行實驗
            metrics = experiment_function(config, logger)
            
            # 記錄實驗完成
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.log_experiment_end(metrics)
            
            # 建立結果對象
            result = ExperimentResult(
                experiment_name=experiment_name,
                config=config,
                metrics=metrics,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                status='completed'
            )
            
        except Exception as e:
            # 處理實驗失敗
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"實驗執行失敗: {str(e)}", e)
            
            result = ExperimentResult(
                experiment_name=experiment_name,
                config=config,
                metrics={},
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                status='failed',
                error_message=str(e)
            )
        
        # 保存結果
        self._save_experiment_result(result)
        self.experiment_results.append(result)
        
        return result
    
    def run_experiment_batch(self, experiment_names: List[str],
                           experiment_function: Callable[[Dict[str, Any], ExperimentLogger], Dict[str, float]],
                           parallel: bool = False,
                           max_workers: int = 4,
                           force_rerun: bool = False) -> List[ExperimentResult]:
        """
        批量執行實驗
        
        Args:
            experiment_names: 實驗名稱列表
            experiment_function: 實驗執行函數
            parallel: 是否並行執行
            max_workers: 最大工作執行緒數
            force_rerun: 是否強制重新執行
            
        Returns:
            實驗結果列表
        """
        results = []
        
        if parallel:
            # 並行執行
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_name = {
                    executor.submit(
                        self.run_single_experiment, 
                        name, 
                        experiment_function, 
                        force_rerun
                    ): name
                    for name in experiment_names
                }
                
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        result = future.result()
                        results.append(result)
                        print(f"實驗 {name} 完成")
                    except Exception as e:
                        print(f"實驗 {name} 執行出錯: {e}")
        else:
            # 串行執行
            for name in experiment_names:
                try:
                    result = self.run_single_experiment(
                        name, experiment_function, force_rerun
                    )
                    results.append(result)
                    print(f"實驗 {name} 完成 ({len(results)}/{len(experiment_names)})")
                except Exception as e:
                    print(f"實驗 {name} 執行出錯: {e}")
        
        return results
    
    def compare_experiments(self, experiment_names: List[str],
                          metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        比較實驗結果
        
        Args:
            experiment_names: 要比較的實驗名稱列表
            metrics: 要比較的指標列表
            
        Returns:
            比較結果的DataFrame
        """
        comparison_data = []
        
        for name in experiment_names:
            result = self._get_existing_result(name)
            if result is None:
                print(f"找不到實驗結果: {name}")
                continue
            
            row_data = {
                'experiment_name': name,
                'status': result.status,
                'duration': result.duration
            }
            
            # 添加指標數據
            if metrics is None:
                row_data.update(result.metrics)
            else:
                for metric in metrics:
                    row_data[metric] = result.metrics.get(metric, None)
            
            # 添加超參數資訊
            if 'hyperparameters' in result.config:
                for param, value in result.config['hyperparameters'].items():
                    row_data[f"param_{param}"] = value
            
            comparison_data.append(row_data)
        
        return pd.DataFrame(comparison_data)
    
    def find_best_experiment(self, metric_name: str,
                           higher_is_better: bool = True) -> Optional[ExperimentResult]:
        """
        找出最佳實驗結果
        
        Args:
            metric_name: 評估指標名稱
            higher_is_better: 是否數值越高越好
            
        Returns:
            最佳實驗結果
        """
        valid_results = [
            result for result in self.experiment_results
            if result.status == 'completed' and metric_name in result.metrics
        ]
        
        if not valid_results:
            return None
        
        if higher_is_better:
            best_result = max(valid_results, key=lambda r: r.metrics[metric_name])
        else:
            best_result = min(valid_results, key=lambda r: r.metrics[metric_name])
        
        return best_result
    
    def reproduce_experiment(self, experiment_name: str,
                           experiment_function: Callable[[Dict[str, Any], ExperimentLogger], Dict[str, float]],
                           n_runs: int = 5) -> Dict[str, Any]:
        """
        重現實驗驗證
        
        Args:
            experiment_name: 原始實驗名稱
            experiment_function: 實驗執行函數
            n_runs: 重複執行次數
            
        Returns:
            重現性分析結果
        """
        original_result = self._get_existing_result(experiment_name)
        if original_result is None:
            raise ValueError(f"找不到原始實驗結果: {experiment_name}")
        
        # 執行多次重現實驗
        reproduction_results = []
        for i in range(n_runs):
            reproduction_name = f"{experiment_name}_reproduce_{i+1}"
            
            try:
                result = self.run_single_experiment(
                    reproduction_name, experiment_function, force_rerun=True
                )
                reproduction_results.append(result)
            except Exception as e:
                print(f"重現實驗 {reproduction_name} 失敗: {e}")
        
        # 分析重現性
        analysis = self._analyze_reproducibility(original_result, reproduction_results)
        
        # 保存重現性分析結果
        analysis_file = self.results_dir / f"{experiment_name}_reproducibility.json"
        save_json(analysis, analysis_file)
        
        return analysis
    
    def _analyze_reproducibility(self, original_result: ExperimentResult,
                               reproduction_results: List[ExperimentResult]) -> Dict[str, Any]:
        """分析重現性"""
        from .utility_functions import calculate_mean_std, calculate_confidence_interval
        
        analysis = {
            'original_experiment': original_result.experiment_name,
            'reproduction_count': len(reproduction_results),
            'successful_reproductions': sum(1 for r in reproduction_results if r.status == 'completed'),
            'metrics_analysis': {}
        }
        
        # 對每個指標進行分析
        for metric_name in original_result.metrics:
            reproduction_values = [
                r.metrics.get(metric_name, 0) for r in reproduction_results
                if r.status == 'completed' and metric_name in r.metrics
            ]
            
            if reproduction_values:
                original_value = original_result.metrics[metric_name]
                mean_value, std_value = calculate_mean_std(reproduction_values)
                ci_lower, ci_upper = calculate_confidence_interval(reproduction_values)
                
                analysis['metrics_analysis'][metric_name] = {
                    'original_value': original_value,
                    'reproduction_mean': mean_value,
                    'reproduction_std': std_value,
                    'confidence_interval': [ci_lower, ci_upper],
                    'original_in_ci': ci_lower <= original_value <= ci_upper,
                    'relative_error': abs(original_value - mean_value) / original_value if original_value != 0 else 0
                }
        
        return analysis
    
    def _get_existing_result(self, experiment_name: str) -> Optional[ExperimentResult]:
        """獲取已存在的實驗結果"""
        # 先從記憶體中查找
        for result in self.experiment_results:
            if result.experiment_name == experiment_name:
                return result
        
        # 從文件中載入
        result_file = self.results_dir / f"{experiment_name}.json"
        if result_file.exists():
            result_data = load_json(result_file)
            return ExperimentResult(**result_data)
        
        return None
    
    def _save_experiment_result(self, result: ExperimentResult):
        """保存實驗結果"""
        result_file = self.results_dir / f"{result.experiment_name}.json"
        result_data = {
            'experiment_name': result.experiment_name,
            'config': result.config,
            'metrics': result.metrics,
            'start_time': result.start_time,
            'end_time': result.end_time,
            'duration': result.duration,
            'status': result.status,
            'error_message': result.error_message
        }
        save_json(result_data, result_file)
    
    def load_experiment_history(self):
        """載入實驗歷史"""
        if not self.results_dir.exists():
            return
        
        for result_file in self.results_dir.glob("*.json"):
            # 跳過特殊檔案
            if result_file.stem.endswith('_reproducibility'):
                continue
                
            try:
                result_data = load_json(result_file)
                result = ExperimentResult(**result_data)
                self.experiment_results.append(result)
            except Exception as e:
                print(f"載入實驗結果失敗 {result_file}: {e}")
    
    def export_results_summary(self, output_file: str = "experiments_summary.csv"):
        """匯出實驗結果摘要"""
        if not self.experiment_results:
            print("沒有實驗結果可匯出")
            return
        
        # 準備數據
        summary_data = []
        for result in self.experiment_results:
            row = {
                'experiment_name': result.experiment_name,
                'status': result.status,
                'start_time': result.start_time,
                'duration': result.duration
            }
            row.update(result.metrics)
            summary_data.append(row)
        
        # 建立DataFrame並保存
        df = pd.DataFrame(summary_data)
        output_path = self.results_dir / output_file
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"實驗結果摘要已匯出至: {output_path}")
    
    def cleanup_failed_experiments(self):
        """清理失敗的實驗文件"""
        removed_count = 0
        
        for result in self.experiment_results[:]:  # 使用副本進行迭代
            if result.status == 'failed':
                # 移除結果文件
                result_file = self.results_dir / f"{result.experiment_name}.json"
                if result_file.exists():
                    result_file.unlink()
                
                # 移除配置文件
                config_file = Path(self.config_manager.config_dir) / f"{result.experiment_name}.yaml"
                if config_file.exists():
                    config_file.unlink()
                
                # 從記憶體中移除
                self.experiment_results.remove(result)
                removed_count += 1
        
        print(f"已清理 {removed_count} 個失敗的實驗")


# 建立全域實驗管理器實例
experiment_manager = ExperimentManager()