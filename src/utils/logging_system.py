"""
日誌系統
負責實驗進度追蹤、錯誤日誌記錄、性能監控、結果版本控制
"""

import os
import logging
import time
import json
import psutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
from contextlib import contextmanager


class ExperimentLogger:
    """實驗日誌記錄器"""
    
    def __init__(self, experiment_name: str, log_dir: str = "results/logs"):
        """
        初始化實驗日誌記錄器
        
        Args:
            experiment_name: 實驗名稱
            log_dir: 日誌存放目錄
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 建立實驗專用目錄
        self.experiment_log_dir = self.log_dir / experiment_name
        self.experiment_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定日誌格式
        self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S'
        
        # 初始化各種日誌記錄器
        self._setup_loggers()
        
        # 實驗開始時間
        self.start_time = time.time()
        
        # 性能監控數據
        self.performance_data = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 實驗結果版本控制
        self.result_versions = []
        
    def _setup_loggers(self):
        """設定各種日誌記錄器"""
        # 主實驗日誌
        self.main_logger = self._create_logger(
            'main', 
            self.experiment_log_dir / 'experiment.log',
            level=logging.INFO
        )
        
        # 錯誤日誌
        self.error_logger = self._create_logger(
            'error',
            self.experiment_log_dir / 'error.log',
            level=logging.ERROR
        )
        
        # 性能日誌
        self.performance_logger = self._create_logger(
            'performance',
            self.experiment_log_dir / 'performance.log',
            level=logging.INFO
        )
        
        # 結果日誌
        self.result_logger = self._create_logger(
            'result',
            self.experiment_log_dir / 'results.log',
            level=logging.INFO
        )
        
    def _create_logger(self, name: str, log_file: Path, 
                      level: int = logging.INFO) -> logging.Logger:
        """創建日誌記錄器"""
        logger = logging.getLogger(f"{self.experiment_name}_{name}")
        logger.setLevel(level)
        
        # 清除既有的處理器
        logger.handlers = []
        
        # 文件處理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(self.log_format, self.date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 控制台處理器（僅主日誌）
        if name == 'main':
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message: str):
        """記錄資訊"""
        self.main_logger.info(message)
    
    def warning(self, message: str):
        """記錄警告"""
        self.main_logger.warning(message)
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """記錄錯誤"""
        self.main_logger.error(message)
        self.error_logger.error(message)
        
        if exception:
            self.error_logger.exception(exception)
    
    def debug(self, message: str):
        """記錄除錯資訊"""
        self.main_logger.debug(message)
    
    def log_experiment_start(self, config: Dict[str, Any]):
        """記錄實驗開始"""
        self.info("=" * 60)
        self.info(f"實驗開始: {self.experiment_name}")
        self.info(f"開始時間: {datetime.now().strftime(self.date_format)}")
        self.info("實驗配置:")
        
        # 記錄配置資訊
        for key, value in config.items():
            if isinstance(value, dict):
                self.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.info(f"    {sub_key}: {sub_value}")
            else:
                self.info(f"  {key}: {value}")
        
        self.info("=" * 60)
    
    def log_experiment_end(self, final_results: Optional[Dict[str, Any]] = None):
        """記錄實驗結束"""
        duration = time.time() - self.start_time
        
        self.info("=" * 60)
        self.info(f"實驗結束: {self.experiment_name}")
        self.info(f"結束時間: {datetime.now().strftime(self.date_format)}")
        self.info(f"總執行時間: {self._format_duration(duration)}")
        
        if final_results:
            self.info("最終結果:")
            for key, value in final_results.items():
                self.info(f"  {key}: {value}")
        
        self.info("=" * 60)
        
        # 停止性能監控
        self.stop_performance_monitoring()
    
    def log_epoch_progress(self, epoch: int, total_epochs: int, 
                          metrics: Dict[str, float]):
        """記錄訓練進度"""
        progress = (epoch + 1) / total_epochs * 100
        
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Epoch {epoch+1}/{total_epochs} ({progress:.1f}%) - {metrics_str}")
    
    def log_validation_results(self, epoch: int, metrics: Dict[str, float]):
        """記錄驗證結果"""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"驗證結果 (Epoch {epoch+1}): {metrics_str}")
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """記錄模型性能"""
        self.info(f"模型 {model_name} 性能:")
        for metric, value in metrics.items():
            self.info(f"  {metric}: {value:.4f}")
    
    def start_performance_monitoring(self, interval: float = 10.0):
        """開始系統性能監控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_performance,
            args=(interval,)
        )
        self.monitoring_thread.start()
        self.info(f"開始系統性能監控 (間隔: {interval}秒)")
    
    def stop_performance_monitoring(self):
        """停止系統性能監控"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        self.info("停止系統性能監控")
        
        # 保存性能數據摘要
        self._save_performance_summary()
    
    def _monitor_performance(self, interval: float):
        """監控系統性能的背景執行緒"""
        while self.monitoring_active:
            try:
                # 獲取系統資源使用情況
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                performance_info = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / (1024**3)
                }
                
                self.performance_data.append(performance_info)
                
                # 記錄到性能日誌
                self.performance_logger.info(
                    f"CPU: {cpu_percent:.1f}% | "
                    f"Memory: {memory.percent:.1f}% | "
                    f"Disk: {disk.percent:.1f}%"
                )
                
                time.sleep(interval)
                
            except Exception as e:
                self.error(f"性能監控錯誤: {str(e)}", e)
                time.sleep(interval)
    
    def _save_performance_summary(self):
        """保存性能監控摘要"""
        if not self.performance_data:
            return
        
        # 計算統計資訊
        cpu_values = [d['cpu_percent'] for d in self.performance_data]
        memory_values = [d['memory_percent'] for d in self.performance_data]
        
        summary = {
            'monitoring_duration': len(self.performance_data),
            'cpu_stats': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory_stats': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            }
        }
        
        # 保存詳細數據
        performance_file = self.experiment_log_dir / 'performance_data.json'
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'detailed_data': self.performance_data
            }, f, indent=2, ensure_ascii=False)
        
        self.info(f"性能監控摘要已保存: {performance_file}")
    
    def save_result_version(self, results: Dict[str, Any], 
                          version_note: str = ""):
        """保存結果版本"""
        version = {
            'timestamp': datetime.now().isoformat(),
            'version': len(self.result_versions) + 1,
            'note': version_note,
            'results': results
        }
        
        self.result_versions.append(version)
        
        # 保存到文件
        results_file = self.experiment_log_dir / 'result_versions.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.result_versions, f, indent=2, ensure_ascii=False)
        
        self.result_logger.info(f"結果版本 {version['version']} 已保存: {version_note}")
    
    @contextmanager
    def time_block(self, block_name: str):
        """時間監控上下文管理器"""
        start_time = time.time()
        self.info(f"開始執行: {block_name}")
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.info(f"完成執行: {block_name} (耗時: {self._format_duration(duration)})")
    
    def _format_duration(self, seconds: float) -> str:
        """格式化時間長度"""
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes}分{seconds:.2f}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = seconds % 60
            return f"{hours}小時{minutes}分{seconds:.2f}秒"


class LoggingManager:
    """日誌管理器"""
    
    def __init__(self):
        self.loggers: Dict[str, ExperimentLogger] = {}
    
    def get_logger(self, experiment_name: str) -> ExperimentLogger:
        """獲取或創建實驗日誌記錄器"""
        if experiment_name not in self.loggers:
            self.loggers[experiment_name] = ExperimentLogger(experiment_name)
        return self.loggers[experiment_name]
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """清理舊的日誌文件"""
        log_dir = Path("results/logs")
        if not log_dir.exists():
            return
        
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        for log_file in log_dir.rglob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    print(f"已刪除舊日誌文件: {log_file}")
                except Exception as e:
                    print(f"刪除日誌文件失敗 {log_file}: {e}")


# 全域日誌管理器實例
logging_manager = LoggingManager()