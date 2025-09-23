"""
實驗配置管理器
負責管理實驗的各項配置，包括超參數、數據集路徑、模型參數、評估指標等
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """實驗配置管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件存放目錄
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        載入配置文件
        
        Args:
            config_name: 配置文件名稱 (不含副檔名)
            
        Returns:
            配置字典
        """
        # 支援多種配置文件格式
        for ext in ['.yaml', '.yml', '.json']:
            config_path = self.config_dir / f"{config_name}{ext}"
            if config_path.exists():
                return self._load_file(config_path)
        
        raise FileNotFoundError(f"找不到配置文件: {config_name}")
    
    def save_config(self, config: Dict[str, Any], config_name: str, format: str = "yaml"):
        """
        保存配置文件
        
        Args:
            config: 配置字典
            config_name: 配置文件名稱
            format: 文件格式 ('yaml' 或 'json')
        """
        if format.lower() == "yaml":
            config_path = self.config_dir / f"{config_name}.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif format.lower() == "json":
            config_path = self.config_dir / f"{config_name}.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支援的格式: {format}")
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """載入配置文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                return json.load(f)
        return {}
    
    def get_default_config(self) -> Dict[str, Any]:
        """獲取預設配置"""
        return {
            # 數據集配置
            'data': {
                'datasets': ['SemEval-2014', 'SemEval-2016'],
                'domains': ['restaurant', 'laptop', 'hotel'],
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'batch_size': 32,
                'max_seq_length': 128
            },
            
            # 模型配置
            'model': {
                'bert_model': 'bert-base-uncased',
                'hidden_size': 768,
                'num_attention_heads': 12,
                'dropout': 0.1,
                'num_classes': 3  # positive, negative, neutral
            },
            
            # 訓練配置
            'training': {
                'learning_rate': 2e-5,
                'num_epochs': 10,
                'warmup_steps': 500,
                'weight_decay': 0.01,
                'early_stopping_patience': 3,
                'gradient_clip_norm': 1.0
            },
            
            # 注意力機制配置
            'attention': {
                'similarity_attention': True,
                'keyword_guided_attention': True,
                'self_attention': True,
                'multi_head_attention': True,
                'fusion_method': 'weighted_sum'
            },
            
            # 跨領域對齊配置
            'cross_domain': {
                'abstract_aspects': [
                    'quality', 'price', 'service', 'ambiance', 'convenience'
                ],
                'alignment_weight': 0.5,
                'similarity_threshold': 0.7
            },
            
            # 評估配置
            'evaluation': {
                'metrics': ['accuracy', 'f1', 'precision', 'recall'],
                'cross_domain_metrics': ['cohesion', 'distinction', 'stability']
            },
            
            # 路徑配置
            'paths': {
                'data_dir': 'data',
                'raw_data_dir': 'data/raw',
                'processed_data_dir': 'data/processed',
                'cache_dir': 'data/cache',
                'output_dir': 'outputs',
                'model_dir': 'outputs/models',
                'figure_dir': 'outputs/figures',
                'report_dir': 'outputs/reports'
            },
            
            # 日誌配置
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'log_dir': 'experiments/logs'
            }
        }
    
    def merge_configs(self, base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合併配置，override_config 覆蓋 base_config
        
        Args:
            base_config: 基礎配置
            override_config: 覆蓋配置
            
        Returns:
            合併後的配置
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        驗證配置的有效性
        
        Args:
            config: 待驗證的配置
            
        Returns:
            配置是否有效
        """
        required_sections = ['data', 'model', 'training', 'paths']
        
        for section in required_sections:
            if section not in config:
                print(f"缺少必要配置段落: {section}")
                return False
        
        # 驗證數據比例總和
        data_config = config.get('data', {})
        ratios = [
            data_config.get('train_ratio', 0),
            data_config.get('val_ratio', 0),
            data_config.get('test_ratio', 0)
        ]
        
        if abs(sum(ratios) - 1.0) > 1e-6:
            print(f"數據集分割比例總和應為1.0，當前為: {sum(ratios)}")
            return False
        
        return True
    
    def get_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """
        獲取特定實驗的配置
        
        Args:
            experiment_name: 實驗名稱
            
        Returns:
            實驗配置
        """
        try:
            config = self.load_config(experiment_name)
        except FileNotFoundError:
            # 使用預設配置
            config = self.get_default_config()
            self.save_config(config, experiment_name)
            
        return config
    
    def create_experiment_config(self, experiment_name: str, 
                               base_config: Optional[Dict[str, Any]] = None,
                               **kwargs) -> Dict[str, Any]:
        """
        建立新實驗配置
        
        Args:
            experiment_name: 實驗名稱
            base_config: 基礎配置，若為None則使用預設配置
            **kwargs: 額外的配置覆蓋項目
            
        Returns:
            新建立的實驗配置
        """
        if base_config is None:
            base_config = self.get_default_config()
            
        # 覆蓋指定參數
        config = self.merge_configs(base_config, kwargs)
        
        # 驗證配置
        if not self.validate_config(config):
            raise ValueError("配置驗證失敗")
        
        # 保存配置
        self.save_config(config, experiment_name)
        
        return config