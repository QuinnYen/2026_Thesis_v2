"""
工具函數庫
提供文件輸入輸出工具、數學計算工具、文本處理工具、隨機種子管理等通用功能
"""

import os
import json
import pickle
import random
import numpy as np
import torch
import re
import string
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import hashlib
from collections import defaultdict, Counter
import unicodedata


# ==================== 隨機種子管理 ====================

def set_random_seed(seed: int = 42):
    """
    設定所有隨機數生成器的種子，確保實驗可重現
    
    Args:
        seed: 隨機種子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_random_state() -> Dict[str, Any]:
    """
    獲取當前隨機數生成器狀態
    
    Returns:
        包含各個隨機數生成器狀態的字典
    """
    return {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }


def restore_random_state(state: Dict[str, Any]):
    """
    恢復隨機數生成器狀態
    
    Args:
        state: 隨機數生成器狀態字典
    """
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['torch_cuda'])


# ==================== 文件輸入輸出工具 ====================

def ensure_dir(path: Union[str, Path]):
    """
    確保目錄存在，如果不存在則創建
    
    Args:
        path: 目錄路徑
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, file_path: Union[str, Path], **kwargs):
    """
    保存數據為JSON格式
    
    Args:
        data: 要保存的數據
        file_path: 保存路徑
        **kwargs: json.dump的額外參數
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, **kwargs)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    從JSON文件載入數據
    
    Args:
        file_path: JSON文件路徑
        
    Returns:
        載入的數據
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, file_path: Union[str, Path]):
    """
    保存數據為pickle格式
    
    Args:
        data: 要保存的數據
        file_path: 保存路徑
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    從pickle文件載入數據
    
    Args:
        file_path: pickle文件路徑
        
    Returns:
        載入的數據
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    計算文件的哈希值
    
    Args:
        file_path: 文件路徑
        algorithm: 哈希算法 ('md5', 'sha1', 'sha256')
        
    Returns:
        文件的哈希值
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


# ==================== 文本處理工具 ====================

def clean_text(text: str, 
               remove_punctuation: bool = False,
               remove_numbers: bool = False,
               normalize_unicode: bool = True,
               lowercase: bool = True) -> str:
    """
    清理文本
    
    Args:
        text: 待清理的文本
        remove_punctuation: 是否移除標點符號
        remove_numbers: 是否移除數字
        normalize_unicode: 是否標準化Unicode
        lowercase: 是否轉為小寫
        
    Returns:
        清理後的文本
    """
    if normalize_unicode:
        text = unicodedata.normalize('NFKD', text)
    
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # 移除多餘的空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_keywords(text: str, 
                    stop_words: Optional[List[str]] = None,
                    min_length: int = 2,
                    max_keywords: Optional[int] = None) -> List[str]:
    """
    從文本中提取關鍵詞
    
    Args:
        text: 輸入文本
        stop_words: 停用詞列表
        min_length: 關鍵詞最小長度
        max_keywords: 返回的最大關鍵詞數量
        
    Returns:
        關鍵詞列表
    """
    if stop_words is None:
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    
    # 清理文本並分詞
    words = clean_text(text, remove_punctuation=True).split()
    
    # 過濾停用詞和短詞
    keywords = [word for word in words 
                if word not in stop_words and len(word) >= min_length]
    
    # 計算詞頻
    word_freq = Counter(keywords)
    
    # 按頻率排序
    sorted_keywords = [word for word, freq in word_freq.most_common()]
    
    if max_keywords is not None:
        sorted_keywords = sorted_keywords[:max_keywords]
    
    return sorted_keywords


def calculate_text_similarity(text1: str, text2: str, method: str = 'jaccard') -> float:
    """
    計算兩個文本的相似度
    
    Args:
        text1: 文本1
        text2: 文本2
        method: 相似度計算方法 ('jaccard', 'cosine', 'overlap')
        
    Returns:
        相似度分數 (0-1)
    """
    # 分詞
    words1 = set(clean_text(text1, remove_punctuation=True).split())
    words2 = set(clean_text(text2, remove_punctuation=True).split())
    
    if method == 'jaccard':
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    elif method == 'overlap':
        intersection = len(words1 & words2)
        min_len = min(len(words1), len(words2))
        return intersection / min_len if min_len > 0 else 0.0
    
    elif method == 'cosine':
        # 簡單的詞袋模型餘弦相似度
        all_words = words1 | words2
        vec1 = [1 if word in words1 else 0 for word in all_words]
        vec2 = [1 if word in words2 else 0 for word in all_words]
        
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        norm1 = sum(v ** 2 for v in vec1) ** 0.5
        norm2 = sum(v ** 2 for v in vec2) ** 0.5
        
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
    
    else:
        raise ValueError(f"不支援的相似度計算方法: {method}")


# ==================== 數學計算工具 ====================

def calculate_mean_std(values: List[float]) -> Tuple[float, float]:
    """
    計算均值和標準差
    
    Args:
        values: 數值列表
        
    Returns:
        (均值, 標準差)
    """
    if not values:
        return 0.0, 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = variance ** 0.5
    
    return mean, std


def calculate_confidence_interval(values: List[float], 
                                confidence: float = 0.95) -> Tuple[float, float]:
    """
    計算信賴區間
    
    Args:
        values: 數值列表
        confidence: 信心水準 (0-1)
        
    Returns:
        (下界, 上界)
    """
    import scipy.stats as stats
    
    mean, std = calculate_mean_std(values)
    n = len(values)
    
    # t分布的臨界值
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, n - 1)
    
    margin_of_error = t_critical * (std / (n ** 0.5))
    
    return (mean - margin_of_error, mean + margin_of_error)


def normalize_scores(scores: List[float], 
                    method: str = 'minmax') -> List[float]:
    """
    標準化分數
    
    Args:
        scores: 分數列表
        method: 標準化方法 ('minmax', 'zscore')
        
    Returns:
        標準化後的分數
    """
    if not scores:
        return []
    
    if method == 'minmax':
        min_score = min(scores)
        max_score = max(scores)
        range_score = max_score - min_score
        
        if range_score == 0:
            return [0.5] * len(scores)  # 所有分數相同時返回中間值
        
        return [(score - min_score) / range_score for score in scores]
    
    elif method == 'zscore':
        mean, std = calculate_mean_std(scores)
        
        if std == 0:
            return [0.0] * len(scores)  # 所有分數相同時返回0
        
        return [(score - mean) / std for score in scores]
    
    else:
        raise ValueError(f"不支援的標準化方法: {method}")


def calculate_weighted_average(values: List[float], 
                              weights: List[float]) -> float:
    """
    計算加權平均
    
    Args:
        values: 數值列表
        weights: 權重列表
        
    Returns:
        加權平均值
    """
    if len(values) != len(weights):
        raise ValueError("數值和權重的長度必須相同")
    
    if not values:
        return 0.0
    
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    total_weight = sum(weights)
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


# ==================== 數據結構工具 ====================

def flatten_dict(d: Dict[str, Any], 
                separator: str = '.', 
                prefix: str = '') -> Dict[str, Any]:
    """
    展平嵌套字典
    
    Args:
        d: 嵌套字典
        separator: 鍵值分隔符
        prefix: 前綴
        
    Returns:
        展平後的字典
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{prefix}{separator}{k}" if prefix else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, separator, new_key).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def unflatten_dict(d: Dict[str, Any], 
                  separator: str = '.') -> Dict[str, Any]:
    """
    將展平的字典恢復為嵌套結構
    
    Args:
        d: 展平的字典
        separator: 鍵值分隔符
        
    Returns:
        嵌套字典
    """
    result = {}
    
    for key, value in d.items():
        keys = key.split(separator)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


def merge_dicts_deep(dict1: Dict[str, Any], 
                    dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合併兩個字典
    
    Args:
        dict1: 字典1
        dict2: 字典2
        
    Returns:
        合併後的字典
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_deep(result[key], value)
        else:
            result[key] = value
    
    return result


# ==================== 性能工具 ====================

def time_function(func):
    """
    函數執行時間裝飾器
    
    Args:
        func: 要計時的函數
        
    Returns:
        裝飾後的函數
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"函數 {func.__name__} 執行時間: {end_time - start_time:.4f} 秒")
        return result
    
    return wrapper


def memory_usage():
    """
    獲取當前記憶體使用情況
    
    Returns:
        記憶體使用資訊字典
    """
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }