# 可視化分析模組
"""
跨領域情感分析系統的可視化分析模組

提供以下功能：
- 結果可視化: 跨領域準確度比較、注意力機制效果對比等
- 注意力可視化: 注意力熱點圖、多頭注意力分析、流程圖等  
- 語義空間可視化: t-SNE降維、3D語義分布、跨領域對齊等
- 圖表生成: PNG格式的學術質量圖表輸出
"""

# 結果可視化器
from .result_visualizer import ResultVisualizer

# 注意力可視化器  
from .attention_visualizer import AttentionVisualizer

# 語義空間可視化器
from .semantic_space_visualizer import SemanticSpaceVisualizer

# 報告生成器已移除，改用直接圖表輸出

__all__ = [
    # 結果可視化
    'ResultVisualizer',
    
    # 注意力機制可視化
    'AttentionVisualizer',
    
    # 語義空間可視化
    'SemanticSpaceVisualizer'
]