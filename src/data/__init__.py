# 數據處理模組
"""
跨領域情感分析系統的數據處理模組

提供以下功能：
- 數據載入: SemEval 2014/2016 數據集載入器
- 預處理: 文本清理、分詞、BIO標記、詞彙表構建
- 特徵提取: BERT、TF-IDF、LDA、統計特徵提取、多層次特徵融合
- 跨領域對齊: 抽象方面對齊、語義相似度計算
"""

from .data_loader import (
    AspectSentiment,
    SemEval2014Loader,
    SemEval2016Loader,
    CustomDataLoader,
    DatasetManager
)

from .preprocessor import (
    AspectDataPreprocessor,
    TextCleaner,
    DataSplitter,
    ProcessedText,
    AspectTermExtractor,
    TokenizerWithAlignment
)

from .feature_extractor import (
    BERTFeatureExtractor,
    TFIDFFeatureExtractor,
    LDAFeatureExtractor,
    StatisticalFeatureExtractor,
    FeatureExtractor as MultiLevelFeatureExtractor,
    FeatureVector
)

from .cross_domain_aligner import (
    AbstractAspectDefinition,
    CrossDomainAligner
)

from .data_converter import (
    AspectSentimentConverter,
    create_experiment_data_converter
)

__all__ = [
    # 數據載入相關
    'AspectSentiment',
    'SemEval2014Loader', 
    'SemEval2016Loader',
    'CustomDataLoader',
    'DatasetManager',
    'DataSplitter',
    
    # 預處理相關
    'AspectDataPreprocessor',
    'TextCleaner',
    'ProcessedText',
    'AspectTermExtractor',
    'TokenizerWithAlignment',
    
    # 特徵提取相關
    'BERTFeatureExtractor',
    'TFIDFFeatureExtractor',
    'LDAFeatureExtractor',
    'StatisticalFeatureExtractor',
    'MultiLevelFeatureExtractor',
    'FeatureVector',
    
    # 跨領域對齊相關
    'AbstractAspectDefinition',
    'CrossDomainAligner',
    
    # 數據轉換相關
    'AspectSentimentConverter',
    'create_experiment_data_converter'
]