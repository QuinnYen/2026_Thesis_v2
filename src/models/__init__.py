# 模型訓練模組
"""
跨領域情感分析系統的模型訓練模組

提供以下功能：
- BERT編碼器: 方面感知和階層式編碼
- 注意力組合器: 多種注意力機制融合
- 特徵融合: 多模態特徵整合
- 分類器: 多種分類架構支援
- 訓練管理: 完整的訓練流程控制
- 模型快取: 模型保存和載入
"""

from .bert_encoder import (
    AspectAwareBERTEncoder,
    HierarchicalBERTEncoder
)

from .attention_combiner import (
    MultiAttentionCombiner,
    SelfAttentionModule,
    CrossAttentionModule,
    SimilarityAttentionModule,
    KeywordAttentionModule,
    GatedFusion,
    PositionalEncoding,
    AdaptiveAttentionCombiner
)

from .feature_fusion import (
    MultiModalFeatureFusion,
    AttentionBasedFusion,
    GatedMultiModalFusion,
    BilinearFusion,
    HierarchicalFeatureFusion,
    CrossModalAttentionFusion
)

from .classifier import (
    BaseClassifier,
    MLPClassifier,
    AttentionEnhancedClassifier,
    HierarchicalClassifier,
    CrossDomainClassifier,
    GradientReversalFunction,
    EnsembleClassifier
)

from .training_manager import (
    TrainingManager,
    EarlyStopping,
    FocalLoss,
    LabelSmoothingCrossEntropy
)

from .model_cache import ModelCache

__all__ = [
    # BERT編碼器相關
    'AspectAwareBERTEncoder',
    'HierarchicalBERTEncoder',
    
    # 注意力機制相關
    'MultiAttentionCombiner',
    'SelfAttentionModule',
    'CrossAttentionModule', 
    'SimilarityAttentionModule',
    'KeywordAttentionModule',
    'GatedFusion',
    'PositionalEncoding',
    'AdaptiveAttentionCombiner',
    
    # 特徵融合相關
    'MultiModalFeatureFusion',
    'AttentionBasedFusion',
    'GatedMultiModalFusion',
    'BilinearFusion',
    'HierarchicalFeatureFusion',
    'CrossModalAttentionFusion',
    
    # 分類器相關
    'BaseClassifier',
    'MLPClassifier',
    'AttentionEnhancedClassifier',
    'HierarchicalClassifier',
    'CrossDomainClassifier',
    'GradientReversalFunction',
    'EnsembleClassifier',
    
    # 訓練管理相關
    'TrainingManager',
    'EarlyStopping',
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    
    # 模型快取相關
    'ModelCache'
]