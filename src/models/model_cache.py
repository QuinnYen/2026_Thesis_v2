# 模型快取系統
"""
模型快取系統

提供預訓練模型管理和檢查點恢復機制
"""

import torch
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
import json


class ModelCache:
    """模型快取管理器"""
    
    def __init__(self, cache_dir: str = 'outputs/models'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model, model_name: str, config: Dict[str, Any]):
        """保存模型和配置"""
        model_path = self.cache_dir / f"{model_name}.pth"
        config_path = self.cache_dir / f"{model_name}_config.json"
        
        torch.save(model.state_dict(), model_path)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_model(self, model_class, model_name: str):
        """載入模型"""
        model_path = self.cache_dir / f"{model_name}.pth"
        config_path = self.cache_dir / f"{model_name}_config.json"
        
        if not model_path.exists() or not config_path.exists():
            return None
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = model_class(**config)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        return model
    
    def cache_features(self, features: Dict[str, torch.Tensor], cache_key: str):
        """快取特徵"""
        cache_path = self.cache_dir / f"features_{cache_key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
    
    def load_features(self, cache_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """載入特徵"""
        cache_path = self.cache_dir / f"features_{cache_key}.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None