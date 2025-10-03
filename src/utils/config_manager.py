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
                'dropout': 0.3,
                'num_classes': 3  # positive, negative, neutral
            },
            
            # 訓練配置
            'training': {
                'learning_rate': 2e-5,
                'num_epochs': 10,
                'warmup_steps': 500,
                'weight_decay': 0.05,
                'early_stopping_patience': 2,
                'gradient_clip_norm': 1.0,

                # K折交叉驗證配置
                'use_k_fold_cv': True,  # 啟用K折交叉驗證
                'k_folds': 5,  # 5折交叉驗證
                
                # 注意力監控配置
                'enable_attention_monitoring': False,  # 默認關閉，可通過配置文件啟用
                'monitoring_output_dir': 'monitoring_output',
                'tensorboard_log_dir': 'logs/attention_monitoring',
                'monitoring_save_frequency': 100,
                'monitoring_history_window': 1000,

                'attention_monitoring': {
                    'monitor_enabled': True,
                    'save_attention_weights': True,
                    'compute_statistics': True,
                    'detect_degradation': True,
                    'generate_plots': True,
                    'alert_on_issues': True,
                    'degradation_thresholds': {
                        'entropy_drop': 0.3,
                        'sparsity_increase': 0.5,
                        'variance_drop': 0.4,
                        'uniformity_increase': 0.3
                    }
                }
            },
            
            # 注意力機制配置
            'attention': {
                'similarity_attention': True,
                'keyword_guided_attention': True,
                'self_attention': True,
                'multi_head_attention': True,
                'fusion_method': 'weighted_sum',

                # 注意力正則化配置
                'regularization': {
                    # 熵正則化：控制注意力分佈的熵值
                    'entropy_reg': {
                        'enabled': True,  # 啟用熵正則化
                        'weight': 0.01,
                        'target_entropy': None  # None 表示鼓勵高熵（均勻分佈）
                    },

                    # 稀疏性正則化：促使注意力權重稀疏化
                    'sparsity_reg': {
                        'enabled': True,  # 啟用稀疏性正則化
                        'weight': 0.005,  # 降低權重避免過度約束
                        'type': 'l1'  # 'l1', 'l2', 'gini'
                    },

                    # 多樣性正則化：增加不同注意力頭之間的多樣性
                    'diversity_reg': {
                        'enabled': True,  # 啟用多樣性正則化
                        'weight': 0.01
                    },

                    # 局部性正則化：鼓勵注意力關注局部相關位置
                    'locality_reg': {
                        'enabled': False,  # 暫時保持關閉
                        'weight': 0.01,
                        'window_size': 5  # 局部窗口大小
                    }
                },

                # 預設正則化配置組合
                'preset_configs': {
                    # 基礎配置：輕微的熵正則化
                    'basic': {
                        'entropy_reg': {'enabled': True, 'weight': 0.005},
                        'sparsity_reg': {'enabled': False},
                        'diversity_reg': {'enabled': False},
                        'locality_reg': {'enabled': False}
                    },

                    # 稀疏配置：促進稀疏注意力
                    'sparse': {
                        'entropy_reg': {'enabled': False},
                        'sparsity_reg': {'enabled': True, 'weight': 0.01, 'type': 'l1'},
                        'diversity_reg': {'enabled': False},
                        'locality_reg': {'enabled': False}
                    },

                    # 局部配置：鼓勵局部注意力
                    'local': {
                        'entropy_reg': {'enabled': False},
                        'sparsity_reg': {'enabled': False},
                        'diversity_reg': {'enabled': False},
                        'locality_reg': {'enabled': True, 'weight': 0.02, 'window_size': 3}
                    },

                    # 多樣性配置：增加注意力頭多樣性
                    'diverse': {
                        'entropy_reg': {'enabled': False},
                        'sparsity_reg': {'enabled': False},
                        'diversity_reg': {'enabled': True, 'weight': 0.01},
                        'locality_reg': {'enabled': False}
                    },

                    # 組合配置：多種正則化結合
                    'combined': {
                        'entropy_reg': {'enabled': True, 'weight': 0.005},
                        'sparsity_reg': {'enabled': True, 'weight': 0.005, 'type': 'l1'},
                        'diversity_reg': {'enabled': True, 'weight': 0.005},
                        'locality_reg': {'enabled': True, 'weight': 0.01, 'window_size': 5}
                    }
                },

                # 注意力約束配置
                'constraints': {
                    # 語義約束：基於語義相似性限制注意力分佈
                    'semantic_constraint': {
                        'enabled': False,
                        'threshold': 0.5,  # 語義相似性閾值
                        'strength': 1.0    # 約束強度
                    },

                    # 位置約束：基於位置關係約束注意力範圍
                    'position_constraint': {
                        'enabled': False,
                        'window_size': 10,    # 位置窗口大小
                        'decay_factor': 0.1   # 距離衰減因子
                    },

                    # 動態溫度調節：根據輸入特徵動態調整注意力溫度
                    'dynamic_temperature': {
                        'enabled': False,
                        'base_temperature': 1.0,           # 基礎溫度
                        'temperature_range': [0.5, 2.0]   # 溫度範圍
                    }
                },

                # 預設約束配置組合
                'constraint_presets': {
                    # 無約束
                    'no_constraint': {
                        'semantic_constraint': {'enabled': False},
                        'position_constraint': {'enabled': False},
                        'dynamic_temperature': {'enabled': False}
                    },

                    # 語義導向配置
                    'semantic_guided': {
                        'semantic_constraint': {
                            'enabled': True,
                            'threshold': 0.6,
                            'strength': 0.8
                        },
                        'position_constraint': {'enabled': False},
                        'dynamic_temperature': {'enabled': False}
                    },

                    # 位置約束配置
                    'position_constrained': {
                        'semantic_constraint': {'enabled': False},
                        'position_constraint': {
                            'enabled': True,
                            'window_size': 8,
                            'decay_factor': 0.15
                        },
                        'dynamic_temperature': {'enabled': False}
                    },

                    # 適應性溫度配置
                    'adaptive_temperature': {
                        'semantic_constraint': {'enabled': False},
                        'position_constraint': {'enabled': False},
                        'dynamic_temperature': {
                            'enabled': True,
                            'base_temperature': 1.0,
                            'temperature_range': [0.7, 1.5]
                        }
                    },

                    # 組合約束配置
                    'combined_constraints': {
                        'semantic_constraint': {
                            'enabled': True,
                            'threshold': 0.5,
                            'strength': 0.5
                        },
                        'position_constraint': {
                            'enabled': True,
                            'window_size': 12,
                            'decay_factor': 0.08
                        },
                        'dynamic_temperature': {
                            'enabled': True,
                            'base_temperature': 1.0,
                            'temperature_range': [0.8, 1.3]
                        }
                    },

                    # 保守約束配置
                    'conservative_constraints': {
                        'semantic_constraint': {
                            'enabled': True,
                            'threshold': 0.7,
                            'strength': 0.3
                        },
                        'position_constraint': {
                            'enabled': True,
                            'window_size': 6,
                            'decay_factor': 0.2
                        },
                        'dynamic_temperature': {
                            'enabled': True,
                            'base_temperature': 1.1,
                            'temperature_range': [0.9, 1.2]
                        }
                    }
                }
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
                'output_dir': 'results',
                'model_dir': 'results/models',
                'figure_dir': 'results/visualizations',
                'report_dir': 'results'
            },
            
            # 日誌配置
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'log_dir': 'results/logs'
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

    def apply_attention_regularization_preset(self, config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
        """
        應用預設的注意力正則化配置

        Args:
            config: 基礎配置
            preset_name: 預設配置名稱 ('basic', 'sparse', 'local', 'diverse', 'combined')

        Returns:
            更新後的配置
        """
        preset_configs = self.get_default_config()['attention']['preset_configs']

        if preset_name not in preset_configs:
            raise ValueError(f"未知的預設配置: {preset_name}. 可用選項: {list(preset_configs.keys())}")

        preset = preset_configs[preset_name]

        # 更新注意力正則化配置
        if 'attention' not in config:
            config['attention'] = {}
        if 'regularization' not in config['attention']:
            config['attention']['regularization'] = {}

        # 應用預設配置
        for reg_type, reg_config in preset.items():
            if reg_type in config['attention']['regularization']:
                config['attention']['regularization'][reg_type].update(reg_config)
            else:
                config['attention']['regularization'][reg_type] = reg_config.copy()

        return config

    def get_attention_regularization_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        從完整配置中提取注意力正則化配置

        Args:
            config: 完整配置

        Returns:
            注意力正則化配置
        """
        return config.get('attention', {}).get('regularization', {})

    def validate_attention_regularization_config(self, reg_config: Dict[str, Any]) -> bool:
        """
        驗證注意力正則化配置的有效性

        Args:
            reg_config: 正則化配置

        Returns:
            配置是否有效
        """
        required_keys = ['entropy_reg', 'sparsity_reg', 'diversity_reg', 'locality_reg']

        for key in required_keys:
            if key not in reg_config:
                return False

            reg_item = reg_config[key]
            if not isinstance(reg_item, dict):
                return False

            # 檢查必要的子鍵
            if 'enabled' not in reg_item or 'weight' not in reg_item:
                return False

            # 檢查權重是否為非負數
            if not isinstance(reg_item['weight'], (int, float)) or reg_item['weight'] < 0:
                return False

        # 檢查稀疏性正則化的類型
        if (reg_config['sparsity_reg'].get('enabled', False) and
            reg_config['sparsity_reg'].get('type', 'l1') not in ['l1', 'l2', 'gini']):
            return False

        # 檢查局部性正則化的窗口大小
        if (reg_config['locality_reg'].get('enabled', False) and
            not isinstance(reg_config['locality_reg'].get('window_size', 5), int)):
            return False

        return True

    def create_regularization_experiment_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        創建一系列正則化實驗配置

        Returns:
            實驗配置字典
        """
        base_config = self.get_default_config()
        experiment_configs = {}

        # 無正則化基準實驗
        experiment_configs['no_regularization'] = base_config.copy()

        # 各種預設配置實驗
        preset_names = ['basic', 'sparse', 'local', 'diverse', 'combined']
        for preset_name in preset_names:
            config = base_config.copy()
            config = self.apply_attention_regularization_preset(config, preset_name)
            experiment_configs[f'regularization_{preset_name}'] = config

        # 權重掃描實驗
        for weight in [0.001, 0.005, 0.01, 0.05]:
            config = base_config.copy()
            config = self.apply_attention_regularization_preset(config, 'basic')
            config['attention']['regularization']['entropy_reg']['weight'] = weight
            experiment_configs[f'entropy_weight_{weight}'] = config

        return experiment_configs

    def apply_attention_constraint_preset(self, config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
        """
        應用預設的注意力約束配置

        Args:
            config: 基礎配置
            preset_name: 預設配置名稱

        Returns:
            更新後的配置
        """
        preset_configs = self.get_default_config()['attention']['constraint_presets']

        if preset_name not in preset_configs:
            raise ValueError(f"未知的約束預設配置: {preset_name}. 可用選項: {list(preset_configs.keys())}")

        preset = preset_configs[preset_name]

        # 更新注意力約束配置
        if 'attention' not in config:
            config['attention'] = {}
        if 'constraints' not in config['attention']:
            config['attention']['constraints'] = {}

        # 應用預設配置
        for constraint_type, constraint_config in preset.items():
            if constraint_type in config['attention']['constraints']:
                config['attention']['constraints'][constraint_type].update(constraint_config)
            else:
                config['attention']['constraints'][constraint_type] = constraint_config.copy()

        return config

    def get_attention_constraint_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        從完整配置中提取注意力約束配置

        Args:
            config: 完整配置

        Returns:
            注意力約束配置
        """
        return config.get('attention', {}).get('constraints', {})

    def validate_attention_constraint_config(self, constraint_config: Dict[str, Any]) -> bool:
        """
        驗證注意力約束配置的有效性

        Args:
            constraint_config: 約束配置

        Returns:
            配置是否有效
        """
        required_keys = ['semantic_constraint', 'position_constraint', 'dynamic_temperature']

        for key in required_keys:
            if key not in constraint_config:
                return False

            constraint_item = constraint_config[key]
            if not isinstance(constraint_item, dict):
                return False

            # 檢查必要的子鍵
            if 'enabled' not in constraint_item:
                return False

        # 檢查語義約束的特定參數
        semantic_config = constraint_config['semantic_constraint']
        if semantic_config.get('enabled', False):
            if 'threshold' not in semantic_config or 'strength' not in semantic_config:
                return False
            if not (0 <= semantic_config['threshold'] <= 1):
                return False
            if semantic_config['strength'] < 0:
                return False

        # 檢查位置約束的特定參數
        position_config = constraint_config['position_constraint']
        if position_config.get('enabled', False):
            if 'window_size' not in position_config or 'decay_factor' not in position_config:
                return False
            if position_config['window_size'] <= 0:
                return False
            if position_config['decay_factor'] < 0:
                return False

        # 檢查動態溫度的特定參數
        temp_config = constraint_config['dynamic_temperature']
        if temp_config.get('enabled', False):
            if 'base_temperature' not in temp_config or 'temperature_range' not in temp_config:
                return False
            temp_range = temp_config['temperature_range']
            if not isinstance(temp_range, list) or len(temp_range) != 2:
                return False
            if temp_range[0] >= temp_range[1] or temp_range[0] <= 0:
                return False

        return True

    def create_constraint_experiment_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        創建一系列約束實驗配置

        Returns:
            實驗配置字典
        """
        base_config = self.get_default_config()
        experiment_configs = {}

        # 無約束基準實驗
        experiment_configs['no_constraints'] = base_config.copy()

        # 各種預設配置實驗
        preset_names = ['semantic_guided', 'position_constrained', 'adaptive_temperature',
                       'combined_constraints', 'conservative_constraints']
        for preset_name in preset_names:
            config = base_config.copy()
            config = self.apply_attention_constraint_preset(config, preset_name)
            experiment_configs[f'constraint_{preset_name}'] = config

        # 語義閾值掃描實驗
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            config = base_config.copy()
            config = self.apply_attention_constraint_preset(config, 'semantic_guided')
            config['attention']['constraints']['semantic_constraint']['threshold'] = threshold
            experiment_configs[f'semantic_threshold_{threshold}'] = config

        # 位置窗口大小掃描實驗
        for window_size in [5, 10, 15, 20]:
            config = base_config.copy()
            config = self.apply_attention_constraint_preset(config, 'position_constrained')
            config['attention']['constraints']['position_constraint']['window_size'] = window_size
            experiment_configs[f'position_window_{window_size}'] = config

        return experiment_configs

    def create_comprehensive_attention_experiment_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        創建包含正則化和約束的綜合注意力實驗配置

        Returns:
            綜合實驗配置字典
        """
        base_config = self.get_default_config()
        experiment_configs = {}

        # 基準配置（無正則化和約束）
        experiment_configs['baseline'] = base_config.copy()

        # 僅正則化配置
        reg_presets = ['basic', 'sparse', 'local', 'diverse']
        for preset in reg_presets:
            config = base_config.copy()
            config = self.apply_attention_regularization_preset(config, preset)
            experiment_configs[f'regularization_only_{preset}'] = config

        # 僅約束配置
        constraint_presets = ['semantic_guided', 'position_constrained', 'adaptive_temperature']
        for preset in constraint_presets:
            config = base_config.copy()
            config = self.apply_attention_constraint_preset(config, preset)
            experiment_configs[f'constraint_only_{preset}'] = config

        # 正則化 + 約束組合配置
        for reg_preset in ['basic', 'sparse']:
            for constraint_preset in ['semantic_guided', 'position_constrained']:
                config = base_config.copy()
                config = self.apply_attention_regularization_preset(config, reg_preset)
                config = self.apply_attention_constraint_preset(config, constraint_preset)
                experiment_configs[f'combined_{reg_preset}_{constraint_preset}'] = config

        return experiment_configs