"""
數據轉換器
將 AspectSentiment 對象轉換為實驗所需的 features/labels 格式
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from .data_loader import AspectSentiment


class AspectSentimentConverter:
    """將 AspectSentiment 對象轉換為實驗所需格式的轉換器"""
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 max_length: int = 128,
                 device: str = "cuda"):
        """
        初始化數據轉換器
        
        Args:
            model_name: 預訓練模型名稱
            max_length: 最大序列長度
            device: 計算設備
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # 初始化tokenizer和模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.to(device)
            self.bert_model.eval()
        except Exception as e:
            print(f"警告：無法載入BERT模型 {model_name}，將使用TF-IDF特徵: {e}")
            self.tokenizer = None
            self.bert_model = None
        
        # 情感標籤編碼器
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['positive', 'negative', 'neutral', 'conflict'])
        
        # TF-IDF向量化器（作為備選）
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.is_fitted = False
    
    def _preprocess_text(self, text: str, aspect_term: str) -> str:
        """
        預處理文本，突出aspect term
        
        Args:
            text: 原始文本
            aspect_term: aspect術語
            
        Returns:
            預處理後的文本
        """
        # 清理文本
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = ' '.join(text.split())
        
        # 突出aspect term（用特殊標記包圍）
        if aspect_term and aspect_term.lower() in text:
            text = text.replace(aspect_term.lower(), f"[ASPECT] {aspect_term.lower()} [/ASPECT]")
        
        return text
    
    def _extract_bert_features(self, texts: List[str]) -> torch.Tensor:
        """
        使用BERT提取特徵
        
        Args:
            texts: 文本列表
            
        Returns:
            特徵張量
        """
        if self.bert_model is None:
            raise ValueError("BERT模型未正確初始化")
        
        all_features = []
        batch_size = 16  # 避免記憶體不足
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 編碼文本
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # 移動到設備
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 獲取BERT特徵
                outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 使用[CLS]標記的特徵作為句子表示
                batch_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
                all_features.append(batch_features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def _extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """
        使用TF-IDF提取特徵
        
        Args:
            texts: 文本列表
            
        Returns:
            特徵矩陣
        """
        if not self.is_fitted:
            # 首次使用時fit vectorizer
            self.tfidf_vectorizer.fit(texts)
            self.is_fitted = True
        
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def convert_to_features_labels(self, 
                                 sentiment_data: List[AspectSentiment],
                                 use_bert: bool = True) -> Dict[str, Any]:
        """
        將 AspectSentiment 列表轉換為 features/labels 格式
        
        Args:
            sentiment_data: AspectSentiment對象列表
            use_bert: 是否使用BERT特徵（否則使用TF-IDF）
            
        Returns:
            包含features和labels的字典
        """
        if not sentiment_data:
            return {'features': np.array([]), 'labels': np.array([])}
        
        # 提取文本和標籤
        texts = []
        labels = []
        
        for item in sentiment_data:
            # 預處理文本
            processed_text = self._preprocess_text(item.text, item.aspect_term)
            texts.append(processed_text)
            
            # 編碼標籤
            try:
                label = self.label_encoder.transform([item.sentiment])[0]
                labels.append(label)
            except ValueError:
                # 處理未知標籤，映射到最接近的類別
                sentiment_lower = item.sentiment.lower()
                if sentiment_lower in ['conflict', 'mixed']:
                    # conflict標籤映射到neutral
                    label = self.label_encoder.transform(['neutral'])[0]
                elif sentiment_lower in ['pos', 'positive']:
                    label = self.label_encoder.transform(['positive'])[0]
                elif sentiment_lower in ['neg', 'negative']:
                    label = self.label_encoder.transform(['negative'])[0]
                else:
                    # 其他未知標籤默認為neutral
                    label = self.label_encoder.transform(['neutral'])[0]
                labels.append(label)
        
        # 提取特徵
        try:
            if use_bert and self.bert_model is not None:
                features = self._extract_bert_features(texts)
                # 轉換為numpy
                if isinstance(features, torch.Tensor):
                    features = features.detach().cpu().numpy()
            else:
                features = self._extract_tfidf_features(texts)
        except Exception as e:
            print(f"特徵提取失敗，使用TF-IDF作為備選: {e}")
            features = self._extract_tfidf_features(texts)
        
        return {
            'features': features,
            'labels': np.array(labels),
            'texts': texts,  # 保留原始文本用於分析
            'aspect_terms': [item.aspect_term for item in sentiment_data],
            'domains': [item.domain for item in sentiment_data]
        }
    
    def convert_domain_data(self, 
                           domain_data: Dict[str, List[AspectSentiment]],
                           use_bert: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        轉換整個領域數據
        
        Args:
            domain_data: 領域到AspectSentiment列表的映射
            use_bert: 是否使用BERT特徵
            
        Returns:
            轉換後的領域數據
        """
        converted_data = {}
        
        for domain, sentiment_list in domain_data.items():
            print(f"轉換領域 {domain} 的數據 ({len(sentiment_list)} 個樣本)...")
            converted_data[domain] = self.convert_to_features_labels(
                sentiment_list, use_bert=use_bert
            )
        
        return converted_data
    
    def get_feature_dim(self) -> int:
        """獲取特徵維度"""
        if self.bert_model is not None:
            return self.bert_model.config.hidden_size
        else:
            return self.tfidf_vectorizer.max_features if hasattr(self.tfidf_vectorizer, 'max_features') else 1000
    
    def get_num_classes(self) -> int:
        """獲取類別數量"""
        return len(self.label_encoder.classes_)
    
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """解碼標籤"""
        return self.label_encoder.inverse_transform(encoded_labels).tolist()


def create_experiment_data_converter(config: Dict[str, Any]) -> AspectSentimentConverter:
    """
    根據配置創建數據轉換器
    
    Args:
        config: 實驗配置
        
    Returns:
        數據轉換器實例
    """
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    return AspectSentimentConverter(
        model_name=model_config.get('pretrained_model', 'bert-base-uncased'),
        max_length=data_config.get('max_length', 128),
        device=config.get('device', 'cuda')
    )