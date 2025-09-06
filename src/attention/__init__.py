# 注意力機制模組
"""
跨領域情感分析系統的注意力機制模組

提供以下功能：
- 相似度注意力: 基於相似度的注意力機制
- 關鍵詞導向注意力: 基於關鍵詞和方面的注意力機制
- 自注意力: 多種自注意力實現
- 多頭注意力: 不同類型的多頭注意力
- 注意力融合: 多種注意力融合策略
"""

# 相似度注意力機制
from .similarity_attention import (
    CosineSimilarityAttention,
    EuclideanDistanceAttention,
    DotProductSimilarityAttention,
    LearnableSimilarityAttention,
    MLPSimilarityAttention,
    AdaptiveSimilarityAttention
)

# 關鍵詞導向注意力機制
from .keyword_guided_attention import (
    KeywordWeightedAttention,
    AspectAwareAttention,
    PositionSensitiveKeywordAttention,
    MultiLevelKeywordAttention,
    DynamicKeywordDiscoveryAttention
)

# 自注意力機制
from .self_attention import (
    BasicSelfAttention,
    ScaledDotProductSelfAttention,
    PositionalSelfAttention,
    RelativePositionSelfAttention,
    LightweightSelfAttention,
    GroupedSelfAttention
)

# 多頭注意力機制
from .multi_head_attention import (
    StandardMultiHeadAttention,
    VariableHeadAttention,
    HierarchicalMultiHeadAttention,
    CrossModalMultiHeadAttention,
    SparseMultiHeadAttention,
    LightweightMultiHeadAttention
)

# 注意力融合機制
from .attention_fusion import (
    WeightedAttentionFusion,
    GatedAttentionFusion,
    HierarchicalAttentionFusion,
    AdaptiveAttentionFusion,
    CrossAttentionFusion,
    AttentionDistillationFusion,
    UniversalAttentionFusion
)

__all__ = [
    # 相似度注意力相關
    'CosineSimilarityAttention',
    'EuclideanDistanceAttention',
    'DotProductSimilarityAttention',
    'LearnableSimilarityAttention',
    'MLPSimilarityAttention',
    'AdaptiveSimilarityAttention',
    
    # 關鍵詞導向注意力相關
    'KeywordWeightedAttention',
    'AspectAwareAttention',
    'PositionSensitiveKeywordAttention',
    'MultiLevelKeywordAttention',
    'DynamicKeywordDiscoveryAttention',
    
    # 自注意力相關
    'BasicSelfAttention',
    'ScaledDotProductSelfAttention',
    'PositionalSelfAttention',
    'RelativePositionSelfAttention',
    'LightweightSelfAttention',
    'GroupedSelfAttention',
    
    # 多頭注意力相關
    'StandardMultiHeadAttention',
    'VariableHeadAttention',
    'HierarchicalMultiHeadAttention',
    'CrossModalMultiHeadAttention',
    'SparseMultiHeadAttention',
    'LightweightMultiHeadAttention',
    
    # 注意力融合相關
    'WeightedAttentionFusion',
    'GatedAttentionFusion',
    'HierarchicalAttentionFusion',
    'AdaptiveAttentionFusion',
    'CrossAttentionFusion',
    'AttentionDistillationFusion',
    'UniversalAttentionFusion'
]