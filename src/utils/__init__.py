"""
配置與工具模組
提供實驗配置管理、日誌系統、工具函數、實驗管理等功能
"""

from .config_manager import ConfigManager
from .logging_system import ExperimentLogger, LoggingManager, logging_manager
from .utility_functions import (
    # 隨機種子管理
    set_random_seed, get_random_state, restore_random_state,
    
    # 文件輸入輸出工具
    ensure_dir, save_json, load_json, save_pickle, load_pickle, get_file_hash,
    
    # 文本處理工具
    clean_text, extract_keywords, calculate_text_similarity,
    
    # 數學計算工具
    calculate_mean_std, calculate_confidence_interval, normalize_scores,
    calculate_weighted_average,
    
    # 數據結構工具
    flatten_dict, unflatten_dict, merge_dicts_deep,
    
    # 性能工具
    time_function, memory_usage
)
from .experiment_manager import (
    ExperimentResult, HyperparameterSearcher, ExperimentManager, experiment_manager
)

__all__ = [
    # 配置管理
    'ConfigManager',
    
    # 日誌系統
    'ExperimentLogger', 'LoggingManager', 'logging_manager',
    
    # 隨機種子管理
    'set_random_seed', 'get_random_state', 'restore_random_state',
    
    # 文件工具
    'ensure_dir', 'save_json', 'load_json', 'save_pickle', 'load_pickle', 'get_file_hash',
    
    # 文本處理
    'clean_text', 'extract_keywords', 'calculate_text_similarity',
    
    # 數學計算
    'calculate_mean_std', 'calculate_confidence_interval', 'normalize_scores',
    'calculate_weighted_average',
    
    # 數據結構
    'flatten_dict', 'unflatten_dict', 'merge_dicts_deep',
    
    # 性能工具
    'time_function', 'memory_usage',
    
    # 實驗管理
    'ExperimentResult', 'HyperparameterSearcher', 'ExperimentManager', 'experiment_manager'
]