# 數據處理模組
"""
跨領域情感分析系統的數據處理模組

提供以下功能：
- 數據載入: SemEval 2014/2016 數據集載入器
- 預處理: 文本清理、分詞、BIO標記、詞彙表構建
- 特徵提取: BERT、TF-IDF、LDA、統計特徵提取
- 跨領域對齊: 抽象方面對齊、語義相似度計算
"""

from .data_loader import (
    AspectSentiment,
    SemEval2014Loader,
    SemEval2016Loader,
    DataSplitter
)

from .preprocessor import (
    AspectDataPreprocessor,
    TextCleaner,
    BIOTagger,
    VocabularyBuilder
)

from .feature_extractor import (
    BERTFeatureExtractor,
    TFIDFFeatureExtractor,
    LDAFeatureExtractor,
    StatisticalFeatureExtractor,
    MultiModalFeatureExtractor
)

from .cross_domain_aligner import (
    AbstractAspectDefinition,
    CrossDomainAligner,
    AlignmentQualityEvaluator
)

__all__ = [
    # 數據載入相關
    'AspectSentiment',
    'SemEval2014Loader', 
    'SemEval2016Loader',
    'DataSplitter',
    
    # 預處理相關
    'AspectDataPreprocessor',
    'TextCleaner',
    'BIOTagger', 
    'VocabularyBuilder',
    
    # 特徵提取相關
    'BERTFeatureExtractor',
    'TFIDFFeatureExtractor',
    'LDAFeatureExtractor',
    'StatisticalFeatureExtractor',
    'MultiModalFeatureExtractor',
    
    # 跨領域對齊相關
    'AbstractAspectDefinition',
    'CrossDomainAligner',
    'AlignmentQualityEvaluator'
]