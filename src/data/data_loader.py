"""
數據載入器
負責載入 SemEval-2014、SemEval-2016 數據集，以及自定義數據格式轉換
"""

import os
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import re

@dataclass
class AspectSentiment:
    """面向情感分析數據結構"""
    text: str                    # 原始文本
    aspect_term: str             # 面向術語
    aspect_category: str         # 面向類別
    sentiment: str               # 情感極性 (positive, negative, neutral)
    start_position: int          # 面向術語開始位置
    end_position: int            # 面向術語結束位置
    domain: str                  # 領域 (restaurant, laptop, hotel 等)
    sentence_id: str             # 句子ID
    
    def to_dict(self) -> Dict:
        """轉換為字典格式"""
        return {
            'text': self.text,
            'aspect_term': self.aspect_term,
            'aspect_category': self.aspect_category,
            'sentiment': self.sentiment,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'domain': self.domain,
            'sentence_id': self.sentence_id
        }


class SemEval2014Loader:
    """SemEval-2014 數據載入器"""
    
    def __init__(self, data_dir: str):
        """
        初始化 SemEval-2014 載入器
        
        Args:
            data_dir: SemEval-2014 數據目錄路徑
        """
        self.data_dir = Path(data_dir)
        self.domain_files = {
            'restaurant': {
                'train': 'Restaurants_Train_v2.xml',
                'test': 'Restaurants_Test_Data_phaseB.xml'
            },
            'laptop': {
                'train': 'Laptop_Train_v2.xml',
                'test': 'Laptops_Test_Data_phaseB.xml'
            }
        }
    
    def load_domain_data(self, domain: str, split: str = 'train') -> List[AspectSentiment]:
        """
        載入特定領域的數據
        
        Args:
            domain: 領域名稱 (restaurant, laptop)
            split: 數據分割 (train, test)
            
        Returns:
            AspectSentiment 對象列表
        """
        if domain not in self.domain_files:
            raise ValueError(f"不支援的領域: {domain}")
        
        if split not in self.domain_files[domain]:
            raise ValueError(f"不支援的數據分割: {split}")
        
        file_path = self.data_dir / self.domain_files[domain][split]
        if not file_path.exists():
            raise FileNotFoundError(f"找不到數據文件: {file_path}")
        
        return self._parse_xml_file(file_path, domain)
    
    def _parse_xml_file(self, file_path: Path, domain: str) -> List[AspectSentiment]:
        """解析 XML 文件"""
        data_samples = []
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            for sentence in root.findall('.//sentence'):
                sentence_id = sentence.get('id', '')
                text = sentence.find('text').text if sentence.find('text') is not None else ""
                
                # 處理面向術語 (aspectTerms)
                aspect_terms = sentence.find('aspectTerms')
                if aspect_terms is not None:
                    for aspect_term in aspect_terms.findall('aspectTerm'):
                        term = aspect_term.get('term', '')
                        polarity = aspect_term.get('polarity', 'neutral')
                        start_pos = int(aspect_term.get('from', 0))
                        end_pos = int(aspect_term.get('to', 0))
                        
                        # 根據面向術語推斷類別
                        category = self._infer_category(term, domain)
                        
                        sample = AspectSentiment(
                            text=text,
                            aspect_term=term,
                            aspect_category=category,
                            sentiment=polarity,
                            start_position=start_pos,
                            end_position=end_pos,
                            domain=domain,
                            sentence_id=sentence_id
                        )
                        data_samples.append(sample)
                
                # 處理面向類別 (aspectCategories)
                aspect_categories = sentence.find('aspectCategories')
                if aspect_categories is not None:
                    for aspect_category in aspect_categories.findall('aspectCategory'):
                        category = aspect_category.get('category', '')
                        polarity = aspect_category.get('polarity', 'neutral')
                        
                        sample = AspectSentiment(
                            text=text,
                            aspect_term='',  # 類別級別沒有具體術語
                            aspect_category=category,
                            sentiment=polarity,
                            start_position=0,
                            end_position=0,
                            domain=domain,
                            sentence_id=sentence_id
                        )
                        data_samples.append(sample)
                        
        except ET.ParseError as e:
            raise ValueError(f"XML 解析錯誤: {e}")
        
        return data_samples
    
    def _infer_category(self, aspect_term: str, domain: str) -> str:
        """根據面向術語推斷類別"""
        term_lower = aspect_term.lower()
        
        if domain == 'restaurant':
            if any(word in term_lower for word in ['food', 'dish', 'meal', 'cuisine']):
                return 'FOOD'
            elif any(word in term_lower for word in ['service', 'staff', 'waiter', 'waitress']):
                return 'SERVICE'
            elif any(word in term_lower for word in ['price', 'cost', 'expensive', 'cheap']):
                return 'PRICE'
            elif any(word in term_lower for word in ['atmosphere', 'ambiance', 'decor']):
                return 'AMBIANCE'
            else:
                return 'GENERAL'
        
        elif domain == 'laptop':
            if any(word in term_lower for word in ['screen', 'display', 'monitor']):
                return 'DISPLAY'
            elif any(word in term_lower for word in ['battery', 'power']):
                return 'BATTERY'
            elif any(word in term_lower for word in ['keyboard', 'key']):
                return 'KEYBOARD'
            elif any(word in term_lower for word in ['performance', 'speed', 'fast', 'slow']):
                return 'PERFORMANCE'
            else:
                return 'GENERAL'
        
        return 'GENERAL'
    
    def load_all_data(self) -> Dict[str, Dict[str, List[AspectSentiment]]]:
        """載入所有領域和分割的數據"""
        all_data = {}
        
        for domain in self.domain_files.keys():
            all_data[domain] = {}
            for split in self.domain_files[domain].keys():
                try:
                    all_data[domain][split] = self.load_domain_data(domain, split)
                except (FileNotFoundError, ValueError) as e:
                    print(f"載入 {domain} {split} 數據失敗: {e}")
                    all_data[domain][split] = []
        
        return all_data


class SemEval2016Loader:
    """SemEval-2016 數據載入器"""
    
    def __init__(self, data_dir: str):
        """
        初始化 SemEval-2016 載入器
        
        Args:
            data_dir: SemEval-2016 數據目錄路徑
        """
        self.data_dir = Path(data_dir)
        self.domain_files = {
            'restaurant': {
                'train': 'restaurants_train_sb1.xml',
                'test': 'restaurants_test_sb1.xml'
            },
            'laptop': {
                'train': 'Laptops_Train_sb1.xml',
                'test': 'laptops_test_sb1.xml'
            }
        }
    
    def load_domain_data(self, domain: str, split: str = 'train') -> List[AspectSentiment]:
        """載入特定領域的數據"""
        if domain not in self.domain_files:
            raise ValueError(f"不支援的領域: {domain}")
        
        if split not in self.domain_files[domain]:
            raise ValueError(f"不支援的數據分割: {split}")
        
        file_path = self.data_dir / self.domain_files[domain][split]
        if not file_path.exists():
            raise FileNotFoundError(f"找不到數據文件: {file_path}")
        
        return self._parse_xml_file(file_path, domain)
    
    def _parse_xml_file(self, file_path: Path, domain: str) -> List[AspectSentiment]:
        """解析 SemEval-2016 XML 文件"""
        data_samples = []
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            for review in root.findall('.//Review'):
                for sentences in review.findall('sentences'):
                    for sentence in sentences.findall('sentence'):
                        sentence_id = sentence.get('id', '')
                        text = sentence.find('text').text if sentence.find('text') is not None else ""
                        
                        # 處理 Opinions
                        opinions = sentence.find('Opinions')
                        if opinions is not None:
                            for opinion in opinions.findall('Opinion'):
                                category = opinion.get('category', '')
                                polarity = opinion.get('polarity', 'neutral')
                                target = opinion.get('target', '')
                                
                                # 查找目標在文本中的位置
                                start_pos, end_pos = self._find_target_position(text, target)
                                
                                sample = AspectSentiment(
                                    text=text,
                                    aspect_term=target,
                                    aspect_category=category,
                                    sentiment=polarity,
                                    start_position=start_pos,
                                    end_position=end_pos,
                                    domain=domain,
                                    sentence_id=sentence_id
                                )
                                data_samples.append(sample)
                                
        except ET.ParseError as e:
            raise ValueError(f"XML 解析錯誤: {e}")
        
        return data_samples
    
    def _find_target_position(self, text: str, target: str) -> Tuple[int, int]:
        """查找目標詞在文本中的位置"""
        if not target or target == "NULL":
            return 0, 0
        
        # 嘗試精確匹配
        start_pos = text.lower().find(target.lower())
        if start_pos != -1:
            return start_pos, start_pos + len(target)
        
        # 如果精確匹配失敗，嘗試部分匹配
        words = target.lower().split()
        for word in words:
            start_pos = text.lower().find(word)
            if start_pos != -1:
                return start_pos, start_pos + len(word)
        
        return 0, 0
    
    def load_all_data(self) -> Dict[str, Dict[str, List[AspectSentiment]]]:
        """載入所有領域和分割的數據"""
        all_data = {}
        
        for domain in self.domain_files.keys():
            all_data[domain] = {}
            for split in self.domain_files[domain].keys():
                try:
                    all_data[domain][split] = self.load_domain_data(domain, split)
                except (FileNotFoundError, ValueError) as e:
                    print(f"載入 {domain} {split} 數據失敗: {e}")
                    all_data[domain][split] = []
        
        return all_data


class CustomDataLoader:
    """自定義數據格式載入器"""
    
    @staticmethod
    def load_json_data(file_path: str) -> List[AspectSentiment]:
        """
        載入 JSON 格式的數據
        
        Args:
            file_path: JSON 文件路徑
            
        Returns:
            AspectSentiment 對象列表
        """
        data_samples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            sample = AspectSentiment(
                text=item.get('text', ''),
                aspect_term=item.get('aspect_term', ''),
                aspect_category=item.get('aspect_category', ''),
                sentiment=item.get('sentiment', 'neutral'),
                start_position=item.get('start_position', 0),
                end_position=item.get('end_position', 0),
                domain=item.get('domain', ''),
                sentence_id=item.get('sentence_id', '')
            )
            data_samples.append(sample)
        
        return data_samples
    
    @staticmethod
    def load_csv_data(file_path: str) -> List[AspectSentiment]:
        """
        載入 CSV 格式的數據
        
        Args:
            file_path: CSV 文件路徑
            
        Returns:
            AspectSentiment 對象列表
        """
        data_samples = []
        
        df = pd.read_csv(file_path)
        
        for _, row in df.iterrows():
            sample = AspectSentiment(
                text=str(row.get('text', '')),
                aspect_term=str(row.get('aspect_term', '')),
                aspect_category=str(row.get('aspect_category', '')),
                sentiment=str(row.get('sentiment', 'neutral')),
                start_position=int(row.get('start_position', 0)),
                end_position=int(row.get('end_position', 0)),
                domain=str(row.get('domain', '')),
                sentence_id=str(row.get('sentence_id', ''))
            )
            data_samples.append(sample)
        
        return data_samples
    
    @staticmethod
    def save_to_json(data: List[AspectSentiment], file_path: str):
        """
        將數據保存為 JSON 格式
        
        Args:
            data: AspectSentiment 對象列表
            file_path: 保存路徑
        """
        json_data = [sample.to_dict() for sample in data]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def save_to_csv(data: List[AspectSentiment], file_path: str):
        """
        將數據保存為 CSV 格式
        
        Args:
            data: AspectSentiment 對象列表
            file_path: 保存路徑
        """
        df_data = [sample.to_dict() for sample in data]
        df = pd.DataFrame(df_data)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')


class DatasetManager:
    """數據集管理器"""
    
    def __init__(self, base_data_dir: str):
        """
        初始化數據集管理器
        
        Args:
            base_data_dir: 基礎數據目錄
        """
        self.base_data_dir = Path(base_data_dir)
        self.semeval2014_loader = SemEval2014Loader(self.base_data_dir / 'raw' / 'SemEval-2014')
        self.semeval2016_loader = SemEval2016Loader(self.base_data_dir / 'raw' / 'SemEval-2016')
        self.custom_loader = CustomDataLoader()
    
    def load_dataset(self, dataset_name: str, domain: str = None, 
                    split: str = 'train') -> List[AspectSentiment]:
        """
        載入指定數據集
        
        Args:
            dataset_name: 數據集名稱 ('semeval2014', 'semeval2016', 或文件路徑)
            domain: 領域名稱 (僅用於 SemEval 數據集)
            split: 數據分割 (僅用於 SemEval 數據集)
            
        Returns:
            AspectSentiment 對象列表
        """
        if dataset_name == 'semeval2014':
            if domain is None:
                raise ValueError("SemEval-2014 數據集需要指定領域")
            return self.semeval2014_loader.load_domain_data(domain, split)
        
        elif dataset_name == 'semeval2016':
            if domain is None:
                raise ValueError("SemEval-2016 數據集需要指定領域")
            return self.semeval2016_loader.load_domain_data(domain, split)
        
        else:
            # 嘗試作為文件路徑載入
            file_path = Path(dataset_name)
            if file_path.suffix.lower() == '.json':
                return self.custom_loader.load_json_data(str(file_path))
            elif file_path.suffix.lower() == '.csv':
                return self.custom_loader.load_csv_data(str(file_path))
            else:
                raise ValueError(f"不支援的數據集或文件格式: {dataset_name}")
    
    def get_dataset_statistics(self, data: List[AspectSentiment]) -> Dict:
        """
        獲取數據集統計資訊
        
        Args:
            data: AspectSentiment 對象列表
            
        Returns:
            統計資訊字典
        """
        if not data:
            return {}
        
        # 基本統計
        total_samples = len(data)
        domains = list(set(sample.domain for sample in data))
        sentiments = list(set(sample.sentiment for sample in data))
        categories = list(set(sample.aspect_category for sample in data if sample.aspect_category))
        
        # 情感分布
        sentiment_dist = {}
        for sentiment in sentiments:
            sentiment_dist[sentiment] = sum(1 for sample in data if sample.sentiment == sentiment)
        
        # 領域分布
        domain_dist = {}
        for domain in domains:
            domain_dist[domain] = sum(1 for sample in data if sample.domain == domain)
        
        # 類別分布
        category_dist = {}
        for category in categories:
            category_dist[category] = sum(1 for sample in data if sample.aspect_category == category)
        
        # 文本長度統計
        text_lengths = [len(sample.text.split()) for sample in data]
        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        
        return {
            'total_samples': total_samples,
            'domains': domains,
            'sentiments': sentiments,
            'categories': categories,
            'sentiment_distribution': sentiment_dist,
            'domain_distribution': domain_dist,
            'category_distribution': category_dist,
            'average_text_length': avg_text_length,
            'text_length_range': (min(text_lengths), max(text_lengths)) if text_lengths else (0, 0)
        }
    
    def load_all_datasets(self) -> Dict[str, Dict[str, List[AspectSentiment]]]:
        """
        載入所有可用的數據集
        
        Returns:
            包含所有數據集的字典，格式為：
            {
                'semeval2014': {
                    'restaurants': {'train': [...], 'test': [...]},
                    'laptops': {'train': [...], 'test': [...]}
                },
                'semeval2016': {
                    'restaurants': {'train': [...], 'test': [...]},
                    'laptops': {'train': [...], 'test': [...]}
                }
            }
        """
        all_datasets = {}
        
        try:
            # 載入 SemEval-2014 數據
            semeval2014_data = self.semeval2014_loader.load_all_data()
            if semeval2014_data:
                all_datasets['semeval2014'] = semeval2014_data
                
        except Exception as e:
            print(f"載入 SemEval-2014 數據失敗: {str(e)}")
        
        try:
            # 載入 SemEval-2016 數據
            semeval2016_data = self.semeval2016_loader.load_all_data()
            if semeval2016_data:
                all_datasets['semeval2016'] = semeval2016_data
                
        except Exception as e:
            print(f"載入 SemEval-2016 數據失敗: {str(e)}")
        
        return all_datasets