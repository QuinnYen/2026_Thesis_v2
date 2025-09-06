# 主控制器 (Main Controller)
"""
跨領域情感分析系統主控制器

串聯所有模組，提供完整的實驗流程：
- 實驗配置管理
- 數據處理流程協調
- 模型訓練和評估
- 結果彙整和保存
- 跨領域對齊評估
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from datetime import datetime
import json
import yaml
from tqdm import tqdm

# 導入所有模組
from .utils import (
    ConfigManager, ExperimentLogger, 
    ExperimentManager, set_random_seed
)
from .data import (
    SemEval2014Loader, SemEval2016Loader, DataSplitter,
    AspectDataPreprocessor, BERTFeatureExtractor, 
    TFIDFFeatureExtractor, LDAFeatureExtractor,
    StatisticalFeatureExtractor, MultiModalFeatureExtractor,
    CrossDomainAligner, AbstractAspectDefinition
)
from .models import (
    AspectAwareBERTEncoder, MultiModalFeatureFusion,
    MultiAttentionCombiner, MLPClassifier, 
    AttentionEnhancedClassifier, CrossDomainClassifier,
    TrainingManager, ModelCache
)


class CrossDomainSentimentAnalysisController:
    """
    跨領域情感分析系統主控制器
    
    協調整個實驗流程，從數據載入到結果輸出
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化主控制器
        
        Args:
            config_path: 配置文件路徑（可選）
        """
        # 初始化配置管理器
        self.config_manager = ConfigManager()
        
        # 載入配置
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self.config_manager.get_default_config()
        
        # 設定隨機種子
        set_random_seed(self.config.get('random_seed', 42))
        
        # 初始化實驗管理器
        self.experiment_manager = ExperimentManager()
        
        # 初始化日誌器
        self.logger = ExperimentLogger(self.config.get('experiment_name', 'cross_domain_sentiment'))
        
        # 初始化模型快取
        self.model_cache = ModelCache(self.config.get('model_cache_dir', 'outputs/models'))
        
        # 組件初始化標記
        self._components_initialized = False
        self._data_loaded = False
        self._features_extracted = False
        
        # 數據相關屬性
        self.datasets = {}
        self.data_loaders = {}
        self.features = {}
        self.models = {}
        
        # 結果存儲
        self.experiment_results = {}
    
    def initialize_components(self):
        """初始化所有系統組件"""
        if self._components_initialized:
            return
        
        self.logger.info("初始化系統組件...")
        
        # 初始化數據預處理器
        self.preprocessor = AspectDataPreprocessor(
            max_length=self.config['data'].get('max_length', 512),
            language=self.config['data'].get('language', 'en')
        )
        
        # 初始化特徵提取器
        self._initialize_feature_extractors()
        
        # 初始化跨領域對齊器
        self.cross_domain_aligner = CrossDomainAligner()
        
        self._components_initialized = True
        self.logger.info("系統組件初始化完成")
    
    def _initialize_feature_extractors(self):
        """初始化特徵提取器"""
        feature_config = self.config['features']
        
        # BERT特徵提取器
        if feature_config.get('use_bert', True):
            self.bert_extractor = BERTFeatureExtractor(
                model_name=feature_config.get('bert_model', 'bert-base-uncased'),
                max_length=self.config['data'].get('max_length', 512)
            )
        
        # TF-IDF特徵提取器
        if feature_config.get('use_tfidf', True):
            self.tfidf_extractor = TFIDFFeatureExtractor(
                max_features=feature_config.get('tfidf_max_features', 1000),
                ngram_range=tuple(feature_config.get('tfidf_ngram_range', [1, 2]))
            )
        
        # LDA特徵提取器
        if feature_config.get('use_lda', True):
            self.lda_extractor = LDAFeatureExtractor(
                n_topics=feature_config.get('lda_topics', 50),
                random_state=self.config.get('random_seed', 42)
            )
        
        # 統計特徵提取器
        if feature_config.get('use_statistical', True):
            self.stats_extractor = StatisticalFeatureExtractor()
        
        # 多模態特徵提取器
        self.multi_modal_extractor = MultiModalFeatureExtractor(
            feature_extractors={
                'bert': getattr(self, 'bert_extractor', None),
                'tfidf': getattr(self, 'tfidf_extractor', None),
                'lda': getattr(self, 'lda_extractor', None),
                'statistical': getattr(self, 'stats_extractor', None)
            }
        )
    
    def load_datasets(self) -> Dict[str, Any]:
        """
        載入所有數據集
        
        Returns:
            數據集信息字典
        """
        if self._data_loaded:
            return self.datasets
        
        self.logger.info("開始載入數據集...")
        
        data_config = self.config['data']
        datasets_to_load = data_config.get('datasets', ['SemEval-2014', 'SemEval-2016'])
        
        # 載入SemEval數據集
        for dataset_name in datasets_to_load:
            if dataset_name == 'SemEval-2014':
                loader = SemEval2014Loader(data_config.get('semeval_2014_path', 'data/raw/SemEval-2014'))
                self.datasets['semeval_2014'] = {
                    'train': loader.load_data('train'),
                    'test': loader.load_data('test')
                }
                
            elif dataset_name == 'SemEval-2016':
                loader = SemEval2016Loader(data_config.get('semeval_2016_path', 'data/raw/SemEval-2016'))
                self.datasets['semeval_2016'] = {
                    'train': loader.load_data('train'),
                    'test': loader.load_data('test')
                }
        
        # 數據分割
        self._split_datasets()
        
        self._data_loaded = True
        self.logger.info(f"數據集載入完成，共載入 {len(self.datasets)} 個數據集")
        
        return self.datasets
    
    def _split_datasets(self):
        """分割數據集為訓練、驗證、測試集"""
        splitter = DataSplitter(
            train_ratio=self.config['data'].get('train_ratio', 0.7),
            val_ratio=self.config['data'].get('val_ratio', 0.15),
            test_ratio=self.config['data'].get('test_ratio', 0.15),
            random_state=self.config.get('random_seed', 42)
        )
        
        for dataset_name, dataset_splits in self.datasets.items():
            if 'train' in dataset_splits and 'test' in dataset_splits:
                # 合併訓練和測試數據後重新分割
                all_data = dataset_splits['train'] + dataset_splits['test']
                train_data, val_data, test_data = splitter.split_data(all_data)
                
                self.datasets[dataset_name] = {
                    'train': train_data,
                    'val': val_data, 
                    'test': test_data
                }
    
    def preprocess_data(self) -> Dict[str, Any]:
        """
        預處理所有數據
        
        Returns:
            預處理後的數據
        """
        self.logger.info("開始數據預處理...")
        
        if not self._data_loaded:
            self.load_datasets()
        
        if not self._components_initialized:
            self.initialize_components()
        
        preprocessed_data = {}
        
        for dataset_name, dataset_splits in self.datasets.items():
            preprocessed_data[dataset_name] = {}
            
            for split_name, split_data in dataset_splits.items():
                self.logger.info(f"預處理 {dataset_name} - {split_name} ({len(split_data)} 條數據)")
                
                # 文本預處理
                processed_data = []
                for item in tqdm(split_data, desc=f"處理 {dataset_name} {split_name}"):
                    processed_item = self.preprocessor.preprocess_aspect_data(
                        text=item.text,
                        aspect_term=item.aspect_term,
                        aspect_category=item.aspect_category
                    )
                    processed_item.update({
                        'sentiment': item.sentiment,
                        'original_item': item
                    })
                    processed_data.append(processed_item)
                
                preprocessed_data[dataset_name][split_name] = processed_data
        
        self.preprocessed_data = preprocessed_data
        self.logger.info("數據預處理完成")
        
        return preprocessed_data
    
    def extract_features(self) -> Dict[str, Any]:
        """
        提取所有特徵
        
        Returns:
            提取的特徵字典
        """
        if self._features_extracted:
            return self.features
        
        self.logger.info("開始特徵提取...")
        
        if not hasattr(self, 'preprocessed_data'):
            self.preprocess_data()
        
        self.features = {}
        
        for dataset_name, dataset_splits in self.preprocessed_data.items():
            self.features[dataset_name] = {}
            
            for split_name, split_data in dataset_splits.items():
                self.logger.info(f"提取特徵 {dataset_name} - {split_name}")
                
                # 準備文本數據
                texts = [item['processed_text'] for item in split_data]
                aspect_terms = [item['aspect_term'] for item in split_data]
                aspect_positions = [item['aspect_positions'] for item in split_data if item['aspect_positions']]
                
                # 使用多模態特徵提取器
                extracted_features = self.multi_modal_extractor.extract_features(
                    texts=texts,
                    aspect_terms=aspect_terms,
                    aspect_positions=aspect_positions if aspect_positions else None
                )
                
                self.features[dataset_name][split_name] = {
                    'features': extracted_features,
                    'labels': [self._sentiment_to_label(item['sentiment']) for item in split_data],
                    'metadata': [{
                        'text': item['processed_text'],
                        'aspect_term': item['aspect_term'], 
                        'aspect_category': item['aspect_category'],
                        'sentiment': item['sentiment']
                    } for item in split_data]
                }
        
        self._features_extracted = True
        self.logger.info("特徵提取完成")
        
        return self.features
    
    def _sentiment_to_label(self, sentiment: str) -> int:
        """將情感標籤轉換為數字"""
        sentiment_map = {
            'positive': 2,
            'neutral': 1, 
            'negative': 0,
            'conflict': 3  # 如果有衝突標籤
        }
        return sentiment_map.get(sentiment.lower(), 1)
    
    def build_models(self) -> Dict[str, nn.Module]:
        """
        構建所有模型
        
        Returns:
            構建的模型字典
        """
        self.logger.info("開始構建模型...")
        
        model_config = self.config['model']
        feature_dims = self._get_feature_dimensions()
        num_classes = self.config['model'].get('num_classes', 3)
        
        models = {}
        
        # 基礎MLP分類器
        if model_config.get('use_mlp', True):
            models['mlp'] = MLPClassifier(
                input_dim=sum(feature_dims.values()),
                num_classes=num_classes,
                hidden_dims=model_config.get('mlp_hidden_dims', [512, 256]),
                dropout_rate=model_config.get('dropout_rate', 0.1)
            )
        
        # 注意力增強分類器
        if model_config.get('use_attention', True):
            models['attention'] = AttentionEnhancedClassifier(
                input_dim=sum(feature_dims.values()),
                num_classes=num_classes,
                num_heads=model_config.get('attention_heads', 8),
                hidden_dim=model_config.get('attention_hidden_dim', 512),
                dropout_rate=model_config.get('dropout_rate', 0.1)
            )
        
        # 跨領域分類器
        if model_config.get('use_cross_domain', True):
            num_domains = len(self.datasets)  # 根據數據集數量確定領域數
            models['cross_domain'] = CrossDomainClassifier(
                input_dim=sum(feature_dims.values()),
                num_classes=num_classes,
                num_domains=num_domains,
                domain_adaptation=model_config.get('domain_adaptation', True),
                hidden_dim=model_config.get('cross_domain_hidden_dim', 512),
                dropout_rate=model_config.get('dropout_rate', 0.1)
            )
        
        # 多模態特徵融合 + 分類器
        if model_config.get('use_multimodal_fusion', True):
            fusion_model = MultiModalFeatureFusion(
                feature_dims=feature_dims,
                fusion_dim=model_config.get('fusion_dim', 512),
                fusion_strategy=model_config.get('fusion_strategy', 'attention'),
                dropout_rate=model_config.get('dropout_rate', 0.1)
            )
            
            classifier = MLPClassifier(
                input_dim=model_config.get('fusion_dim', 512),
                num_classes=num_classes,
                hidden_dims=model_config.get('fusion_classifier_dims', [256]),
                dropout_rate=model_config.get('dropout_rate', 0.1)
            )
            
            models['multimodal'] = nn.Sequential(fusion_model, classifier)
        
        self.models = models
        self.logger.info(f"模型構建完成，共構建 {len(models)} 個模型")
        
        return models
    
    def _get_feature_dimensions(self) -> Dict[str, int]:
        """獲取特徵維度"""
        if not self.features:
            self.extract_features()
        
        # 從第一個數據集的第一個分割獲取特徵維度信息
        sample_features = None
        for dataset_name, dataset_splits in self.features.items():
            for split_name, split_data in dataset_splits.items():
                if split_data['features']:
                    sample_features = split_data['features'][0]
                    break
            if sample_features:
                break
        
        if sample_features is None:
            # 默認維度
            return {
                'bert': 768,
                'tfidf': 1000, 
                'lda': 50,
                'statistical': 20
            }
        
        # 計算實際維度
        feature_dims = {}
        for feature_type, feature_tensor in sample_features.items():
            if isinstance(feature_tensor, torch.Tensor):
                feature_dims[feature_type] = feature_tensor.shape[-1]
            elif isinstance(feature_tensor, np.ndarray):
                feature_dims[feature_type] = feature_tensor.shape[-1]
            else:
                feature_dims[feature_type] = len(feature_tensor) if hasattr(feature_tensor, '__len__') else 1
        
        return feature_dims
    
    def create_data_loaders(self) -> Dict[str, Dict[str, DataLoader]]:
        """
        創建PyTorch數據載入器
        
        Returns:
            數據載入器字典
        """
        self.logger.info("創建數據載入器...")
        
        if not self._features_extracted:
            self.extract_features()
        
        batch_size = self.config['training'].get('batch_size', 16)
        self.data_loaders = {}
        
        for dataset_name, dataset_splits in self.features.items():
            self.data_loaders[dataset_name] = {}
            
            for split_name, split_data in dataset_splits.items():
                dataset = SentimentDataset(
                    features=split_data['features'],
                    labels=split_data['labels'],
                    metadata=split_data['metadata']
                )
                
                shuffle = (split_name == 'train')
                self.data_loaders[dataset_name][split_name] = DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=0  # Windows環境設為0避免多進程問題
                )
        
        self.logger.info("數據載入器創建完成")
        return self.data_loaders
    
    def train_models(self) -> Dict[str, Dict[str, Any]]:
        """
        訓練所有模型
        
        Returns:
            訓練結果字典
        """
        self.logger.info("開始模型訓練...")
        
        if not self.models:
            self.build_models()
        
        if not self.data_loaders:
            self.create_data_loaders()
        
        training_config = self.config['training']
        results = {}
        
        # 對每個模型進行訓練
        for model_name, model in self.models.items():
            self.logger.info(f"訓練模型: {model_name}")
            model_results = {}
            
            # 在每個數據集上訓練
            for dataset_name, dataset_loaders in self.data_loaders.items():
                self.logger.info(f"在數據集 {dataset_name} 上訓練 {model_name}")
                
                # 設置訓練管理器
                trainer = TrainingManager(
                    model=model,
                    train_loader=dataset_loaders['train'],
                    val_loader=dataset_loaders['val'],
                    config=training_config
                )
                
                # 開始訓練
                history = trainer.train(epochs=training_config.get('epochs', 10))
                
                # 在測試集上評估
                test_metrics = trainer.validate_epoch()
                
                model_results[dataset_name] = {
                    'history': history,
                    'test_metrics': test_metrics
                }
                
                # 保存模型
                model_save_name = f"{model_name}_{dataset_name}"
                self.model_cache.save_model(
                    model=model,
                    model_name=model_save_name,
                    config={
                        'model_type': model_name,
                        'dataset': dataset_name,
                        'training_config': training_config
                    }
                )
            
            results[model_name] = model_results
        
        self.experiment_results['training'] = results
        self.logger.info("模型訓練完成")
        
        return results
    
    def evaluate_cross_domain_alignment(self) -> Dict[str, Any]:
        """
        評估跨領域對齊效果
        
        Returns:
            跨領域對齊評估結果
        """
        self.logger.info("開始跨領域對齊評估...")
        
        if not self._features_extracted:
            self.extract_features()
        
        alignment_results = {}
        
        # 獲取所有數據集的特徵
        all_features = {}
        all_metadata = {}
        
        for dataset_name, dataset_splits in self.features.items():
            # 合併所有分割的數據
            combined_features = []
            combined_metadata = []
            
            for split_name, split_data in dataset_splits.items():
                combined_features.extend(split_data['features'])
                combined_metadata.extend(split_data['metadata'])
            
            all_features[dataset_name] = combined_features
            all_metadata[dataset_name] = combined_metadata
        
        # 進行跨領域對齊評估
        for source_domain in all_features.keys():
            for target_domain in all_features.keys():
                if source_domain != target_domain:
                    alignment_key = f"{source_domain}_to_{target_domain}"
                    
                    # 計算對齊分數
                    alignment_score = self.cross_domain_aligner.evaluate_alignment_quality(
                        source_features=all_features[source_domain],
                        target_features=all_features[target_domain],
                        source_metadata=all_metadata[source_domain],
                        target_metadata=all_metadata[target_domain]
                    )
                    
                    alignment_results[alignment_key] = alignment_score
        
        self.experiment_results['cross_domain_alignment'] = alignment_results
        self.logger.info("跨領域對齊評估完成")
        
        return alignment_results
    
    def generate_experiment_report(self) -> Dict[str, Any]:
        """
        生成實驗報告
        
        Returns:
            完整的實驗報告
        """
        self.logger.info("生成實驗報告...")
        
        report = {
            'experiment_info': {
                'name': self.config.get('experiment_name', 'cross_domain_sentiment'),
                'timestamp': datetime.now().isoformat(),
                'config': self.config
            },
            'data_statistics': self._generate_data_statistics(),
            'model_performance': self.experiment_results.get('training', {}),
            'cross_domain_alignment': self.experiment_results.get('cross_domain_alignment', {}),
            'feature_analysis': self._analyze_features()
        }
        
        # 保存報告
        self._save_experiment_report(report)
        
        self.logger.info("實驗報告生成完成")
        return report
    
    def _generate_data_statistics(self) -> Dict[str, Any]:
        """生成數據統計信息"""
        if not hasattr(self, 'datasets') or not self.datasets:
            return {}
        
        stats = {}
        for dataset_name, dataset_splits in self.datasets.items():
            dataset_stats = {}
            for split_name, split_data in dataset_splits.items():
                sentiment_counts = {}
                for item in split_data:
                    sentiment = item.sentiment
                    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                
                dataset_stats[split_name] = {
                    'total_samples': len(split_data),
                    'sentiment_distribution': sentiment_counts
                }
            
            stats[dataset_name] = dataset_stats
        
        return stats
    
    def _analyze_features(self) -> Dict[str, Any]:
        """分析特徵統計信息"""
        if not self._features_extracted:
            return {}
        
        feature_analysis = {}
        
        for dataset_name, dataset_splits in self.features.items():
            dataset_analysis = {}
            
            for split_name, split_data in dataset_splits.items():
                if split_data['features']:
                    sample_features = split_data['features'][0]
                    split_analysis = {}
                    
                    for feature_type, feature_tensor in sample_features.items():
                        if isinstance(feature_tensor, (torch.Tensor, np.ndarray)):
                            split_analysis[feature_type] = {
                                'dimension': feature_tensor.shape[-1] if hasattr(feature_tensor, 'shape') else len(feature_tensor),
                                'type': str(type(feature_tensor).__name__)
                            }
                    
                    dataset_analysis[split_name] = split_analysis
            
            feature_analysis[dataset_name] = dataset_analysis
        
        return feature_analysis
    
    def _save_experiment_report(self, report: Dict[str, Any]):
        """保存實驗報告"""
        # 創建輸出目錄
        output_dir = Path(self.config.get('output_dir', 'outputs/reports'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成報告文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.config.get('experiment_name', 'cross_domain_sentiment')
        report_filename = f"{experiment_name}_{timestamp}_report.json"
        
        # 保存JSON報告
        report_path = output_dir / report_filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"實驗報告已保存到: {report_path}")
    
    def run_complete_experiment(self) -> Dict[str, Any]:
        """
        運行完整的實驗流程
        
        Returns:
            完整的實驗結果
        """
        self.logger.info("開始運行完整實驗流程...")
        self.logger.start_performance_monitoring()
        
        try:
            # 1. 初始化組件
            self.initialize_components()
            
            # 2. 載入和預處理數據
            self.load_datasets()
            self.preprocess_data()
            
            # 3. 特徵提取
            self.extract_features()
            
            # 4. 構建模型
            self.build_models()
            
            # 5. 創建數據載入器
            self.create_data_loaders()
            
            # 6. 訓練模型
            self.train_models()
            
            # 7. 跨領域對齊評估
            self.evaluate_cross_domain_alignment()
            
            # 8. 生成實驗報告
            final_report = self.generate_experiment_report()
            
            self.logger.info("完整實驗流程運行成功")
            self.logger.log_experiment_end(final_report)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"實驗運行失敗: {str(e)}")
            raise e
        
        finally:
            self.logger.stop_performance_monitoring()


class SentimentDataset(Dataset):
    """情感分析數據集類"""
    
    def __init__(self, features: List[Dict[str, torch.Tensor]], 
                 labels: List[int], 
                 metadata: List[Dict[str, Any]]):
        """
        初始化數據集
        
        Args:
            features: 特徵列表
            labels: 標籤列表
            metadata: 元數據列表
        """
        self.features = features
        self.labels = labels
        self.metadata = metadata
        
        assert len(features) == len(labels) == len(metadata), "特徵、標籤、元數據長度必須相同"
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 合併所有特徵為一個向量
        feature_tensors = []
        for feature_type, feature_tensor in self.features[idx].items():
            if isinstance(feature_tensor, torch.Tensor):
                feature_tensors.append(feature_tensor.flatten())
            elif isinstance(feature_tensor, np.ndarray):
                feature_tensors.append(torch.from_numpy(feature_tensor).float().flatten())
            else:
                # 處理其他類型的特徵
                feature_tensors.append(torch.tensor([feature_tensor], dtype=torch.float32))
        
        combined_features = torch.cat(feature_tensors, dim=0)
        
        return {
            'features': combined_features,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'metadata': self.metadata[idx]
        }


def main(config_path: Optional[str] = None):
    """
    主函數 - 運行完整的跨領域情感分析實驗
    
    Args:
        config_path: 配置文件路徑（可選）
    """
    try:
        # 創建主控制器
        controller = CrossDomainSentimentAnalysisController(config_path)
        
        # 運行完整實驗
        results = controller.run_complete_experiment()
        
        print("\\n" + "="*50)
        print("實驗完成！")
        print("="*50)
        print(f"實驗名稱: {results['experiment_info']['name']}")
        print(f"完成時間: {results['experiment_info']['timestamp']}")
        
        # 打印關鍵結果
        if 'model_performance' in results:
            print("\\n模型性能總結:")
            for model_name, model_results in results['model_performance'].items():
                print(f"  {model_name}:")
                for dataset_name, dataset_results in model_results.items():
                    test_acc = dataset_results['test_metrics'].get('accuracy', 0)
                    print(f"    {dataset_name}: 測試準確率 = {test_acc:.4f}")
        
        if 'cross_domain_alignment' in results:
            print("\\n跨領域對齊評估:")
            for alignment_key, alignment_score in results['cross_domain_alignment'].items():
                print(f"  {alignment_key}: {alignment_score:.4f}")
        
        return results
        
    except Exception as e:
        print(f"實驗運行失敗: {str(e)}")
        raise e


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="跨領域情感分析實驗")
    parser.add_argument('--config', type=str, default=None, help='配置文件路徑')
    
    args = parser.parse_args()
    
    main(args.config)