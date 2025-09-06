# 評估分析模組
"""
跨領域情感分析系統的評估分析模組

提供以下功能：
- 標準評估: 常規分類指標評估
- 跨領域評估: 創新的跨領域對齊評估指標
- 統計分析: 顯著性檢驗和效果量計算
- 錯誤分析: 深度錯誤模式分析和改進建議
"""

# 標準評估器
from .standard_evaluator import StandardEvaluator

# 跨領域評估器  
from .cross_domain_evaluator import CrossDomainEvaluator

# 統計分析器
from .statistical_analyzer import StatisticalAnalyzer

# 錯誤分析器
from .error_analyzer import ErrorAnalyzer

__all__ = [
    # 標準評估
    'StandardEvaluator',
    
    # 跨領域評估（論文主要創新點）
    'CrossDomainEvaluator',
    
    # 統計分析
    'StatisticalAnalyzer',
    
    # 錯誤分析
    'ErrorAnalyzer'
]