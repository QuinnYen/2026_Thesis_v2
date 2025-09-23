# 實驗框架模組
"""
跨領域情感分析系統的實驗框架

提供三個主要實驗：
1. 實驗1：融合策略比較實驗
2. 實驗2：注意力機制比較實驗  
3. 實驗3：組合效果分析實驗

每個實驗都有獨立的控制器和完整的評估框架。
"""

try:
    import sys
    import os
    from pathlib import Path
    
    # 確保當前目錄在路徑中
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    from experiment_1_controller import Experiment1Controller, create_experiment1_config
    from experiment_2_controller import Experiment2Controller, create_experiment2_config
    from experiment_3_controller import Experiment3Controller, create_experiment3_config
    
    from fusion_strategy_experiment import FusionStrategyExperiment
    from attention_mechanism_experiment import AttentionMechanismComparator
    from combination_analysis_experiment import CombinationAnalysisExperiment
    
    EXPERIMENTS_AVAILABLE = True
    
except ImportError as e:
    print(f"警告：部分實驗模組導入失敗 - {str(e)}")
    EXPERIMENTS_AVAILABLE = False
    
    # 創建空的佔位符類
    class Experiment1Controller:
        def __init__(self, *args, **kwargs):
            raise ImportError("實驗1控制器不可用")
    
    class Experiment2Controller:
        def __init__(self, *args, **kwargs):
            raise ImportError("實驗2控制器不可用")
    
    class Experiment3Controller:
        def __init__(self, *args, **kwargs):
            raise ImportError("實驗3控制器不可用")
    
    def create_experiment1_config():
        return {}
    
    def create_experiment2_config():
        return {}
    
    def create_experiment3_config():
        return {}

__all__ = [
    'Experiment1Controller',
    'Experiment2Controller', 
    'Experiment3Controller',
    'create_experiment1_config',
    'create_experiment2_config',
    'create_experiment3_config',
    'FusionStrategyExperiment',
    'AttentionMechanismComparator',
    'CombinationAnalysisExperiment',
    'EXPERIMENTS_AVAILABLE'
]