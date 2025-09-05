"""
特徵提取器
負責 BERT 嵌入向量生成、LDA 主題模型建構、領域特定詞彙表建構、統計特徵計算等
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter, defaultdict
import pickle
from pathlib import Path
from dataclasses import dataclass

from .data_loader import AspectSentiment
from .preprocessor import ProcessedText


@dataclass
class FeatureVector:
    """特徵向量數據結構"""
    bert_features: torch.Tensor           # BERT 嵌入特徵
    tfidf_features: Optional[np.ndarray]  # TF-IDF 特徵
    lda_features: Optional[np.ndarray]    # LDA 主題特徵
    statistical_features: Optional[np.ndarray]  # 統計特徵
    domain_features: Optional[np.ndarray] # 領域特定特徵
    text_length: int                      # 文本長度
    aspect_positions: List[Tuple[int, int]]  # 面向位置
    
    def to_dict(self) -> Dict:
        """轉換為字典格式（用於保存）"""
        return {
            'bert_features': self.bert_features.cpu().numpy() if isinstance(self.bert_features, torch.Tensor) else self.bert_features,
            'tfidf_features': self.tfidf_features,
            'lda_features': self.lda_features,
            'statistical_features': self.statistical_features,
            'domain_features': self.domain_features,
            'text_length': self.text_length,
            'aspect_positions': self.aspect_positions
        }


class BERTFeatureExtractor:
    """BERT 嵌入向量生成器"""
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 max_length: int = 128,
                 device: str = "cpu"):
        """
        初始化 BERT 特徵提取器
        
        Args:
            model_name: BERT 模型名稱
            max_length: 最大序列長度
            device: 運算裝置
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device)
        
        # 載入 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"已載入 BERT 模型: {model_name}")
    
    def encode_texts(self, texts: List[str], 
                    batch_size: int = 16) -> torch.Tensor:
        """
        批量編碼文本
        
        Args:
            texts: 文本列表
            batch_size: 批量大小
            
        Returns:
            編碼後的特徵張量 [batch_size, hidden_size]
        """
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 分詞和編碼
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
                
                # 前向傳播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 使用 [CLS] token 的表示作為句子表示
                cls_features = outputs.last_hidden_state[:, 0, :]
                all_features.append(cls_features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def encode_with_aspect_attention(self, texts: List[str], 
                                   aspect_positions: List[List[Tuple[int, int]]],
                                   batch_size: int = 16) -> torch.Tensor:
        """
        帶面向注意力的編碼
        
        Args:
            texts: 文本列表
            aspect_positions: 面向位置列表
            batch_size: 批量大小
            
        Returns:
            編碼後的特徵張量
        """
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_positions = aspect_positions[i:i + batch_size]
                
                # 分詞和編碼
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 前向傳播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 獲取序列表示
                sequence_output = outputs.last_hidden_state
                
                # 計算面向權重的平均表示
                batch_features = []
                for j, positions in enumerate(batch_positions):
                    if positions:
                        # 有面向術語，使用面向位置的平均表示
                        aspect_features = []
                        for start, end in positions:
                            # 確保位置在有效範圍內
                            start = max(0, min(start, sequence_output.size(1) - 1))
                            end = max(start + 1, min(end, sequence_output.size(1)))
                            aspect_repr = sequence_output[j, start:end, :].mean(dim=0)
                            aspect_features.append(aspect_repr)
                        
                        if aspect_features:
                            combined_features = torch.stack(aspect_features).mean(dim=0)
                        else:
                            combined_features = sequence_output[j, 0, :]  # 使用 [CLS]
                    else:
                        # 沒有面向術語，使用 [CLS] 表示
                        combined_features = sequence_output[j, 0, :]
                    
                    batch_features.append(combined_features)
                
                batch_tensor = torch.stack(batch_features)
                all_features.append(batch_tensor.cpu())
        
        return torch.cat(all_features, dim=0)


class TFIDFFeatureExtractor:
    """TF-IDF 特徵提取器"""
    
    def __init__(self, 
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.95):
        """
        初始化 TF-IDF 特徵提取器
        
        Args:
            max_features: 最大特徵數量
            ngram_range: N-gram 範圍
            min_df: 最小文檔頻率
            max_df: 最大文檔頻率
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        self.fitted = False
    
    def fit(self, texts: List[str]):
        """
        訓練 TF-IDF 向量化器
        
        Args:
            texts: 訓練文本列表
        """
        self.vectorizer.fit(texts)
        self.fitted = True
        print(f"TF-IDF 向量化器已訓練，特徵維度: {len(self.vectorizer.vocabulary_)}")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        轉換文本為 TF-IDF 特徵
        
        Args:
            texts: 輸入文本列表
            
        Returns:
            TF-IDF 特徵矩陣
        """
        if not self.fitted:
            raise ValueError("TF-IDF 向量化器尚未訓練")
        
        return self.vectorizer.transform(texts).toarray()
    
    def get_feature_names(self) -> List[str]:
        """獲取特徵名稱"""
        if not self.fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()


class LDAFeatureExtractor:
    """LDA 主題模型建構器"""
    
    def __init__(self, 
                 n_topics: int = 10,
                 max_features: int = 1000,
                 random_state: int = 42):
        """
        初始化 LDA 特徵提取器
        
        Args:
            n_topics: 主題數量
            max_features: 最大特徵數量
            random_state: 隨機種子
        """
        self.n_topics = n_topics
        self.random_state = random_state
        
        # TF-IDF 向量化器（LDA 的輸入）
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        # LDA 模型
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=10,
            learning_method='online'
        )
        
        self.fitted = False
    
    def fit(self, texts: List[str]):
        """
        訓練 LDA 模型
        
        Args:
            texts: 訓練文本列表
        """
        # 先進行 TF-IDF 轉換
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # 訓練 LDA 模型
        self.lda_model.fit(tfidf_matrix)
        self.fitted = True
        
        print(f"LDA 模型已訓練，主題數: {self.n_topics}")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        轉換文本為主題分布特徵
        
        Args:
            texts: 輸入文本列表
            
        Returns:
            主題分布特徵矩陣 [n_samples, n_topics]
        """
        if not self.fitted:
            raise ValueError("LDA 模型尚未訓練")
        
        # TF-IDF 轉換
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        # LDA 轉換
        return self.lda_model.transform(tfidf_matrix)
    
    def get_top_words_per_topic(self, n_words: int = 10) -> Dict[int, List[str]]:
        """
        獲取每個主題的頂部詞彙
        
        Args:
            n_words: 每個主題返回的詞彙數量
            
        Returns:
            主題詞彙字典 {主題索引: [詞彙列表]}
        """
        if not self.fitted:
            return {}
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        topic_words = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_words[topic_idx] = top_words
        
        return topic_words


class DomainSpecificVocabulary:
    """領域特定詞彙表建構器"""
    
    def __init__(self):
        self.domain_vocabularies: Dict[str, Dict[str, float]] = {}
        self.global_vocabulary: Dict[str, int] = {}
        self.domain_tfidf_scores: Dict[str, Dict[str, float]] = {}
    
    def build_vocabularies(self, data: List[AspectSentiment]):
        """
        建構領域特定詞彙表
        
        Args:
            data: AspectSentiment 對象列表
        """
        # 按領域分組
        domain_texts = defaultdict(list)
        all_texts = []
        
        for sample in data:
            domain_texts[sample.domain].append(sample.text)
            all_texts.append(sample.text)
        
        # 建立全局詞彙表
        all_words = []
        for text in all_texts:
            words = text.lower().split()
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        self.global_vocabulary = {word: count for word, count in word_counts.items() if count >= 2}
        
        # 為每個領域建立 TF-IDF 詞彙表
        for domain, texts in domain_texts.items():
            if len(texts) < 10:  # 跳過文本太少的領域
                continue
            
            # 使用 TF-IDF 找出領域特定詞彙
            tfidf = TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
            
            tfidf_matrix = tfidf.fit_transform(texts)
            feature_names = tfidf.get_feature_names_out()
            
            # 計算平均 TF-IDF 分數
            mean_scores = tfidf_matrix.mean(axis=0).A1
            domain_vocab = {}
            
            for i, word in enumerate(feature_names):
                domain_vocab[word] = mean_scores[i]
            
            self.domain_vocabularies[domain] = domain_vocab
            
            # 儲存 TF-IDF 分數用於特徵提取
            self.domain_tfidf_scores[domain] = dict(zip(feature_names, mean_scores))
        
        print(f"已建立 {len(self.domain_vocabularies)} 個領域的詞彙表")
    
    def get_domain_specific_words(self, domain: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        獲取領域特定詞彙
        
        Args:
            domain: 領域名稱
            top_k: 返回的詞彙數量
            
        Returns:
            領域特定詞彙列表 [(詞彙, 分數)]
        """
        if domain not in self.domain_vocabularies:
            return []
        
        vocab = self.domain_vocabularies[domain]
        sorted_words = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_k]
    
    def extract_domain_features(self, text: str, domain: str) -> np.ndarray:
        """
        提取領域特定特徵
        
        Args:
            text: 輸入文本
            domain: 領域名稱
            
        Returns:
            領域特定特徵向量
        """
        if domain not in self.domain_tfidf_scores:
            return np.zeros(100)  # 預設特徵維度
        
        words = text.lower().split()
        domain_scores = self.domain_tfidf_scores[domain]
        
        # 計算文本中詞彙的領域特定分數
        word_scores = [domain_scores.get(word, 0.0) for word in words]
        
        if not word_scores:
            return np.zeros(100)
        
        # 統計特徵
        features = [
            np.mean(word_scores),          # 平均分數
            np.max(word_scores),           # 最大分數
            np.std(word_scores),           # 標準差
            len([s for s in word_scores if s > 0]),  # 領域詞彙數量
            len([s for s in word_scores if s > np.mean(word_scores)])  # 高分詞彙數量
        ]
        
        # 填充到固定維度
        features.extend([0.0] * (100 - len(features)))
        return np.array(features[:100])


class StatisticalFeatureExtractor:
    """統計特徵計算器"""
    
    @staticmethod
    def extract_statistical_features(text: str, processed_text: ProcessedText) -> np.ndarray:
        """
        提取統計特徵
        
        Args:
            text: 原始文本
            processed_text: 預處理後的文本
            
        Returns:
            統計特徵向量
        """
        features = []
        
        # 基本長度特徵
        features.append(len(text))                    # 字符長度
        features.append(len(processed_text.tokens))   # 詞彙數量
        features.append(len([t for t in processed_text.tokens if len(t) > 5]))  # 長詞數量
        
        # 標點符號特徵
        punctuation_count = sum(1 for c in text if c in '.,!?;:')
        features.append(punctuation_count)
        features.append(punctuation_count / len(text) if len(text) > 0 else 0)
        
        # 大寫字母特徵
        uppercase_count = sum(1 for c in text if c.isupper())
        features.append(uppercase_count)
        features.append(uppercase_count / len(text) if len(text) > 0 else 0)
        
        # 詞性特徵
        pos_counts = Counter(processed_text.pos_tags)
        features.extend([
            pos_counts.get('NOUN', 0),     # 名詞數量
            pos_counts.get('VERB', 0),     # 動詞數量
            pos_counts.get('ADJ', 0),      # 形容詞數量
            pos_counts.get('ADV', 0),      # 副詞數量
        ])
        
        # 面向術語特徵
        aspect_count = len(processed_text.aspect_positions)
        features.append(aspect_count)
        
        # 句子複雜度特徵
        avg_word_length = np.mean([len(token) for token in processed_text.tokens]) if processed_text.tokens else 0
        features.append(avg_word_length)
        
        # 詞彙多樣性
        unique_words = len(set(processed_text.tokens))
        vocab_diversity = unique_words / len(processed_text.tokens) if processed_text.tokens else 0
        features.append(vocab_diversity)
        
        return np.array(features, dtype=np.float32)


class FeatureExtractor:
    """綜合特徵提取器"""
    
    def __init__(self, 
                 bert_model: str = "bert-base-uncased",
                 max_length: int = 128,
                 device: str = "cpu",
                 use_tfidf: bool = True,
                 use_lda: bool = True,
                 use_domain_vocab: bool = True):
        """
        初始化特徵提取器
        
        Args:
            bert_model: BERT 模型名稱
            max_length: 最大序列長度
            device: 運算裝置
            use_tfidf: 是否使用 TF-IDF 特徵
            use_lda: 是否使用 LDA 特徵
            use_domain_vocab: 是否使用領域詞彙特徵
        """
        self.bert_extractor = BERTFeatureExtractor(bert_model, max_length, device)
        self.use_tfidf = use_tfidf
        self.use_lda = use_lda
        self.use_domain_vocab = use_domain_vocab
        
        if use_tfidf:
            self.tfidf_extractor = TFIDFFeatureExtractor()
        
        if use_lda:
            self.lda_extractor = LDAFeatureExtractor()
        
        if use_domain_vocab:
            self.domain_vocab = DomainSpecificVocabulary()
        
        self.statistical_extractor = StatisticalFeatureExtractor()
        self.fitted = False
    
    def fit(self, data: List[AspectSentiment], processed_data: List[ProcessedText]):
        """
        訓練特徵提取器
        
        Args:
            data: AspectSentiment 對象列表
            processed_data: 預處理後的數據列表
        """
        texts = [sample.text for sample in data]
        
        if self.use_tfidf:
            self.tfidf_extractor.fit(texts)
        
        if self.use_lda:
            self.lda_extractor.fit(texts)
        
        if self.use_domain_vocab:
            self.domain_vocab.build_vocabularies(data)
        
        self.fitted = True
        print("特徵提取器訓練完成")
    
    def extract_features(self, 
                        data: List[AspectSentiment], 
                        processed_data: List[ProcessedText],
                        batch_size: int = 16) -> List[FeatureVector]:
        """
        提取特徵
        
        Args:
            data: AspectSentiment 對象列表
            processed_data: 預處理後的數據列表
            batch_size: 批量大小
            
        Returns:
            特徵向量列表
        """
        if not self.fitted:
            raise ValueError("特徵提取器尚未訓練")
        
        texts = [sample.text for sample in data]
        aspect_positions = [processed.aspect_positions for processed in processed_data]
        
        # BERT 特徵
        bert_features = self.bert_extractor.encode_with_aspect_attention(
            texts, aspect_positions, batch_size
        )
        
        # TF-IDF 特徵
        tfidf_features = None
        if self.use_tfidf:
            tfidf_features = self.tfidf_extractor.transform(texts)
        
        # LDA 特徵
        lda_features = None
        if self.use_lda:
            lda_features = self.lda_extractor.transform(texts)
        
        # 建立特徵向量
        feature_vectors = []
        for i, (sample, processed) in enumerate(zip(data, processed_data)):
            # 領域特定特徵
            domain_features = None
            if self.use_domain_vocab:
                domain_features = self.domain_vocab.extract_domain_features(
                    sample.text, sample.domain
                )
            
            # 統計特徵
            statistical_features = self.statistical_extractor.extract_statistical_features(
                sample.text, processed
            )
            
            feature_vector = FeatureVector(
                bert_features=bert_features[i],
                tfidf_features=tfidf_features[i] if tfidf_features is not None else None,
                lda_features=lda_features[i] if lda_features is not None else None,
                statistical_features=statistical_features,
                domain_features=domain_features,
                text_length=len(processed.tokens),
                aspect_positions=processed.aspect_positions
            )
            
            feature_vectors.append(feature_vector)
        
        return feature_vectors
    
    def save(self, save_path: str):
        """保存特徵提取器"""
        save_data = {
            'tfidf_extractor': self.tfidf_extractor if self.use_tfidf else None,
            'lda_extractor': self.lda_extractor if self.use_lda else None,
            'domain_vocab': self.domain_vocab if self.use_domain_vocab else None,
            'fitted': self.fitted
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"特徵提取器已保存至: {save_path}")
    
    def load(self, load_path: str):
        """載入特徵提取器"""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        if self.use_tfidf and save_data['tfidf_extractor']:
            self.tfidf_extractor = save_data['tfidf_extractor']
        
        if self.use_lda and save_data['lda_extractor']:
            self.lda_extractor = save_data['lda_extractor']
        
        if self.use_domain_vocab and save_data['domain_vocab']:
            self.domain_vocab = save_data['domain_vocab']
        
        self.fitted = save_data['fitted']
        
        print(f"特徵提取器已載入: {load_path}")