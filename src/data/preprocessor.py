"""
預處理器
負責文本清理、分詞與詞性標註、面向術語提取與標註、數據集分割等預處理工作
"""

import re
import string
import unicodedata
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from dataclasses import dataclass
import spacy
from spacy.lang.en import English

from .data_loader import AspectSentiment


@dataclass
class ProcessedText:
    """預處理後的文本數據結構"""
    original_text: str           # 原始文本
    cleaned_text: str            # 清理後的文本
    tokens: List[str]            # 分詞結果
    pos_tags: List[str]          # 詞性標籤
    lemmas: List[str]            # 詞元化結果
    aspect_positions: List[Tuple[int, int]]  # 面向術語在token中的位置
    aspect_labels: List[str]     # 面向標籤 (BIO格式)


class TextCleaner:
    """文本清理器"""
    
    def __init__(self, 
                 remove_html: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 normalize_whitespace: bool = True,
                 lowercase: bool = False,
                 remove_punctuation: bool = False,
                 remove_numbers: bool = False):
        """
        初始化文本清理器
        
        Args:
            remove_html: 是否移除HTML標籤
            remove_urls: 是否移除URL
            remove_emails: 是否移除電子郵件
            normalize_whitespace: 是否標準化空白字符
            lowercase: 是否轉為小寫
            remove_punctuation: 是否移除標點符號
            remove_numbers: 是否移除數字
        """
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        
        # 預編譯正規表達式
        self.html_pattern = re.compile(r'<[^<>]*>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def clean(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理後的文本
        """
        if not text:
            return ""
        
        # Unicode標準化
        text = unicodedata.normalize('NFKD', text)
        
        # 移除HTML標籤
        if self.remove_html:
            text = self.html_pattern.sub(' ', text)
        
        # 移除URL
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        # 移除電子郵件
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        # 移除數字
        if self.remove_numbers:
            text = self.number_pattern.sub(' ', text)
        
        # 移除標點符號
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 轉為小寫
        if self.lowercase:
            text = text.lower()
        
        # 標準化空白字符
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text


class AspectTermExtractor:
    """面向術語提取器"""
    
    def __init__(self, nlp_model: str = "en_core_web_sm"):
        """
        初始化面向術語提取器
        
        Args:
            nlp_model: spaCy模型名稱
        """
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            # 如果模型未安裝，使用基礎英文模型
            print(f"警告: 未找到 {nlp_model}，使用基礎英文分詞器")
            self.nlp = English()
            self.nlp.add_pipe('sentencizer')
        
        # 面向術語的詞性標籤
        self.aspect_pos_tags = {'NOUN', 'PROPN', 'ADJ'}
        
        # 停用詞集合
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 
                              'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def extract_candidate_terms(self, text: str) -> List[Tuple[str, int, int]]:
        """
        提取候選面向術語
        
        Args:
            text: 輸入文本
            
        Returns:
            候選術語列表 [(術語, 開始位置, 結束位置)]
        """
        doc = self.nlp(text)
        candidates = []
        
        # 單詞候選
        for token in doc:
            if (token.pos_ in self.aspect_pos_tags and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2 and
                token.text.lower() not in self.stop_words):
                
                candidates.append((
                    token.text,
                    token.idx,
                    token.idx + len(token.text)
                ))
        
        # 名詞短語候選
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 2 and chunk.text.lower() not in self.stop_words:
                candidates.append((
                    chunk.text,
                    chunk.start_char,
                    chunk.end_char
                ))
        
        # 去重並排序
        candidates = list(set(candidates))
        candidates.sort(key=lambda x: x[1])  # 按開始位置排序
        
        return candidates
    
    def create_bio_labels(self, text: str, tokens: List[str], 
                         aspect_terms: List[Tuple[str, int, int]]) -> List[str]:
        """
        創建BIO格式的面向標籤
        
        Args:
            text: 原始文本
            tokens: 分詞結果
            aspect_terms: 面向術語列表 [(術語, 開始位置, 結束位置)]
            
        Returns:
            BIO標籤列表
        """
        doc = self.nlp(text)
        bio_labels = ['O'] * len(tokens)
        
        # 建立token位置映射
        token_positions = []
        for token in doc:
            token_positions.append((token.idx, token.idx + len(token.text)))
        
        # 為每個面向術語標記BIO標籤
        for aspect_term, start_pos, end_pos in aspect_terms:
            # 找到與面向術語重疊的token
            overlapping_tokens = []
            for i, (token_start, token_end) in enumerate(token_positions):
                if token_start < end_pos and token_end > start_pos:
                    overlapping_tokens.append(i)
            
            # 標記BIO標籤
            if overlapping_tokens:
                bio_labels[overlapping_tokens[0]] = 'B-ASPECT'
                for i in overlapping_tokens[1:]:
                    bio_labels[i] = 'I-ASPECT'
        
        return bio_labels


class TokenizerWithAlignment:
    """帶對齊功能的分詞器"""
    
    def __init__(self, nlp_model: str = "en_core_web_sm"):
        """
        初始化分詞器
        
        Args:
            nlp_model: spaCy模型名稱
        """
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            print(f"警告: 未找到 {nlp_model}，使用基礎英文分詞器")
            self.nlp = English()
            self.nlp.add_pipe('sentencizer')
    
    def tokenize_with_positions(self, text: str) -> Tuple[List[str], List[str], List[str], List[Tuple[int, int]]]:
        """
        分詞並保持位置對齊
        
        Args:
            text: 輸入文本
            
        Returns:
            (tokens, pos_tags, lemmas, positions)
        """
        doc = self.nlp(text)
        
        tokens = []
        pos_tags = []
        lemmas = []
        positions = []
        
        for token in doc:
            tokens.append(token.text)
            pos_tags.append(token.pos_)
            lemmas.append(token.lemma_)
            positions.append((token.idx, token.idx + len(token.text)))
        
        return tokens, pos_tags, lemmas, positions
    
    def align_aspect_terms(self, text: str, tokens: List[str], 
                          aspect_terms: List[Tuple[str, int, int]]) -> List[Tuple[int, int]]:
        """
        將面向術語對齊到token位置
        
        Args:
            text: 原始文本
            tokens: token列表
            aspect_terms: 面向術語列表
            
        Returns:
            對齊後的token位置列表
        """
        doc = self.nlp(text)
        token_positions = [(token.idx, token.idx + len(token.text)) for token in doc]
        
        aligned_positions = []
        
        for aspect_term, char_start, char_end in aspect_terms:
            # 找到與面向術語重疊的token範圍
            start_token = None
            end_token = None
            
            for i, (token_start, token_end) in enumerate(token_positions):
                if start_token is None and token_start >= char_start:
                    start_token = i
                if token_end <= char_end:
                    end_token = i + 1
            
            if start_token is not None and end_token is not None:
                aligned_positions.append((start_token, end_token))
            else:
                # 如果無法對齊，使用模糊匹配
                best_match = self._fuzzy_align(aspect_term, tokens)
                if best_match:
                    aligned_positions.append(best_match)
                else:
                    aligned_positions.append((0, 0))  # 無法對齊
        
        return aligned_positions
    
    def _fuzzy_align(self, aspect_term: str, tokens: List[str]) -> Optional[Tuple[int, int]]:
        """模糊對齊面向術語到token"""
        aspect_words = aspect_term.lower().split()
        
        for i in range(len(tokens)):
            # 檢查從位置i開始是否匹配
            match_length = 0
            for j, word in enumerate(aspect_words):
                if i + j < len(tokens) and tokens[i + j].lower() == word:
                    match_length += 1
                else:
                    break
            
            # 如果匹配超過一半的詞，認為是匹配的
            if match_length >= len(aspect_words) // 2 + 1:
                return (i, i + match_length)
        
        return None


class DataSplitter:
    """數據集分割器"""
    
    @staticmethod
    def split_data(data: List[AspectSentiment], 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_state: int = 42,
                  stratify_by: str = 'sentiment') -> Tuple[List[AspectSentiment], List[AspectSentiment], List[AspectSentiment]]:
        """
        分割數據集
        
        Args:
            data: AspectSentiment對象列表
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例
            test_ratio: 測試集比例
            random_state: 隨機種子
            stratify_by: 分層依據 ('sentiment', 'domain', 'category', None)
            
        Returns:
            (train_data, val_data, test_data)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("比例總和必須等於1.0")
        
        if not data:
            return [], [], []
        
        # 準備分層標籤
        stratify_labels = None
        if stratify_by == 'sentiment':
            stratify_labels = [sample.sentiment for sample in data]
        elif stratify_by == 'domain':
            stratify_labels = [sample.domain for sample in data]
        elif stratify_by == 'category':
            stratify_labels = [sample.aspect_category for sample in data]
        
        # 第一次分割: 分出訓練集
        if val_ratio + test_ratio > 0:
            train_data, temp_data, train_labels, temp_labels = train_test_split(
                data, stratify_labels,
                test_size=(val_ratio + test_ratio),
                random_state=random_state,
                stratify=stratify_labels
            )
        else:
            train_data, temp_data = data, []
            train_labels, temp_labels = stratify_labels, []
        
        # 第二次分割: 從剩餘數據中分出驗證集和測試集
        if len(temp_data) > 0 and val_ratio > 0 and test_ratio > 0:
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_ratio_adjusted),
                random_state=random_state,
                stratify=temp_labels
            )
        elif len(temp_data) > 0:
            if val_ratio > 0:
                val_data, test_data = temp_data, []
            else:
                val_data, test_data = [], temp_data
        else:
            val_data, test_data = [], []
        
        return train_data, val_data, test_data
    
    @staticmethod
    def split_by_domain(data: List[AspectSentiment], 
                       source_domains: List[str],
                       target_domain: str) -> Tuple[List[AspectSentiment], List[AspectSentiment]]:
        """
        按領域分割數據（用於跨領域實驗）
        
        Args:
            data: AspectSentiment對象列表
            source_domains: 源領域列表
            target_domain: 目標領域
            
        Returns:
            (source_data, target_data)
        """
        source_data = [sample for sample in data if sample.domain in source_domains]
        target_data = [sample for sample in data if sample.domain == target_domain]
        
        return source_data, target_data

    @staticmethod
    def stratified_k_fold_split(data: List[AspectSentiment],
                               k_folds: int = 5,
                               random_state: int = 42,
                               stratify_by: str = 'sentiment',
                               shuffle: bool = True) -> List[Tuple[List[AspectSentiment], List[AspectSentiment]]]:
        """
        分層 K-fold 交叉驗證分割

        確保每個折都包含足夠的各類樣本，特別是 negative 樣本

        Args:
            data: AspectSentiment對象列表
            k_folds: 折數，預設為 5
            random_state: 隨機種子
            stratify_by: 分層依據 ('sentiment', 'domain', 'category')
            shuffle: 是否打亂數據

        Returns:
            List of (train_data, val_data) tuples for each fold
        """
        if not data:
            return []

        # 準備分層標籤
        stratify_labels = []
        if stratify_by == 'sentiment':
            stratify_labels = [sample.sentiment for sample in data]
        elif stratify_by == 'domain':
            stratify_labels = [sample.domain for sample in data]
        elif stratify_by == 'category':
            stratify_labels = [sample.aspect_category for sample in data]
        else:
            raise ValueError(f"不支援的分層依據: {stratify_by}")

        # 計算各類別的樣本數
        from collections import Counter
        label_counts = Counter(stratify_labels)
        print(f"各類別樣本數: {dict(label_counts)}")

        # 檢查是否有類別樣本數少於折數
        min_samples = min(label_counts.values())
        if min_samples < k_folds:
            print(f"警告: 最小類別樣本數 ({min_samples}) 小於折數 ({k_folds})")
            print("建議減少折數或增加樣本數")
            k_folds = min_samples
            print(f"自動調整折數為: {k_folds}")

        # 建立分層 K-fold
        skf = StratifiedKFold(
            n_splits=k_folds,
            shuffle=shuffle,
            random_state=random_state
        )

        fold_splits = []
        data_array = np.array(data)
        labels_array = np.array(stratify_labels)

        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(data_array, labels_array)):
            train_data = data_array[train_indices].tolist()
            val_data = data_array[val_indices].tolist()

            # 統計每個折的類別分佈
            train_labels = labels_array[train_indices]
            val_labels = labels_array[val_indices]

            train_counts = Counter(train_labels)
            val_counts = Counter(val_labels)

            print(f"第 {fold_idx + 1} 折:")
            print(f"  訓練集: {dict(train_counts)} (總計: {len(train_data)})")
            print(f"  驗證集: {dict(val_counts)} (總計: {len(val_data)})")

            # 檢查 negative 樣本分佈
            if 'negative' in train_counts and 'negative' in val_counts:
                neg_train_ratio = train_counts['negative'] / len(train_data)
                neg_val_ratio = val_counts['negative'] / len(val_data)
                print(f"  Negative 比例 - 訓練集: {neg_train_ratio:.3f}, 驗證集: {neg_val_ratio:.3f}")

            fold_splits.append((train_data, val_data))

        return fold_splits

    @staticmethod
    def analyze_class_distribution(data: List[AspectSentiment],
                                 stratify_by: str = 'sentiment') -> Dict:
        """
        分析類別分佈情況

        Args:
            data: AspectSentiment對象列表
            stratify_by: 分析依據

        Returns:
            類別分佈統計資訊
        """
        if not data:
            return {}

        # 提取標籤
        if stratify_by == 'sentiment':
            labels = [sample.sentiment for sample in data]
        elif stratify_by == 'domain':
            labels = [sample.domain for sample in data]
        elif stratify_by == 'category':
            labels = [sample.aspect_category for sample in data]
        else:
            raise ValueError(f"不支援的分析依據: {stratify_by}")

        from collections import Counter
        label_counts = Counter(labels)
        total_samples = len(data)

        distribution_info = {
            'total_samples': total_samples,
            'class_counts': dict(label_counts),
            'class_ratios': {k: v/total_samples for k, v in label_counts.items()},
            'min_class_size': min(label_counts.values()),
            'max_class_size': max(label_counts.values()),
            'imbalance_ratio': max(label_counts.values()) / min(label_counts.values())
        }

        return distribution_info


class AspectDataPreprocessor:
    """面向情感分析數據預處理器"""
    
    def __init__(self, 
                 clean_config: Optional[Dict] = None,
                 nlp_model: str = "en_core_web_sm"):
        """
        初始化預處理器
        
        Args:
            clean_config: 文本清理配置
            nlp_model: spaCy模型名稱
        """
        # 預設清理配置
        default_clean_config = {
            'remove_html': True,
            'remove_urls': True,
            'remove_emails': True,
            'normalize_whitespace': True,
            'lowercase': False,  # 保持原始大小寫
            'remove_punctuation': False,  # 保留標點符號
            'remove_numbers': False  # 保留數字
        }
        
        if clean_config:
            default_clean_config.update(clean_config)
        
        self.text_cleaner = TextCleaner(**default_clean_config)
        self.tokenizer = TokenizerWithAlignment(nlp_model)
        self.aspect_extractor = AspectTermExtractor(nlp_model)
        self.data_splitter = DataSplitter()
    
    def preprocess_sample(self, sample: AspectSentiment) -> ProcessedText:
        """
        預處理單個樣本
        
        Args:
            sample: AspectSentiment對象
            
        Returns:
            ProcessedText對象
        """
        # 清理文本
        cleaned_text = self.text_cleaner.clean(sample.text)
        
        # 分詞和詞性標註
        tokens, pos_tags, lemmas, positions = self.tokenizer.tokenize_with_positions(cleaned_text)
        
        # 處理面向術語
        aspect_terms = []
        if sample.aspect_term and sample.start_position >= 0 and sample.end_position > sample.start_position:
            aspect_terms.append((sample.aspect_term, sample.start_position, sample.end_position))
        
        # 對齊面向術語到token位置
        aspect_positions = self.tokenizer.align_aspect_terms(cleaned_text, tokens, aspect_terms)
        
        # 創建BIO標籤
        aspect_labels = self.aspect_extractor.create_bio_labels(cleaned_text, tokens, aspect_terms)
        
        return ProcessedText(
            original_text=sample.text,
            cleaned_text=cleaned_text,
            tokens=tokens,
            pos_tags=pos_tags,
            lemmas=lemmas,
            aspect_positions=aspect_positions,
            aspect_labels=aspect_labels
        )
    
    def preprocess_dataset(self, data: List[AspectSentiment]) -> List[ProcessedText]:
        """
        預處理數據集
        
        Args:
            data: AspectSentiment對象列表
            
        Returns:
            ProcessedText對象列表
        """
        processed_data = []
        
        for sample in data:
            try:
                processed_sample = self.preprocess_sample(sample)
                processed_data.append(processed_sample)
            except Exception as e:
                print(f"預處理樣本失敗: {sample.sentence_id}, 錯誤: {e}")
                continue
        
        return processed_data
    
    def get_vocabulary(self, processed_data: List[ProcessedText], 
                      min_freq: int = 2, 
                      max_vocab_size: Optional[int] = None) -> Dict[str, int]:
        """
        建立詞彙表
        
        Args:
            processed_data: 預處理後的數據
            min_freq: 最小詞頻
            max_vocab_size: 最大詞彙表大小
            
        Returns:
            詞彙表 {詞: 索引}
        """
        # 統計詞頻
        word_freq = {}
        for sample in processed_data:
            for token in sample.tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # 過濾低頻詞
        filtered_words = [word for word, freq in word_freq.items() if freq >= min_freq]
        
        # 按頻率排序
        filtered_words.sort(key=lambda w: word_freq[w], reverse=True)
        
        # 限制詞彙表大小
        if max_vocab_size:
            filtered_words = filtered_words[:max_vocab_size]
        
        # 建立詞彙表，保留特殊token
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<CLS>': 2,
            '<SEP>': 3
        }
        
        for i, word in enumerate(filtered_words):
            vocab[word] = i + 4
        
        return vocab