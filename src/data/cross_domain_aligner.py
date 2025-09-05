"""
跨領域對齊器
負責五個抽象面向定義、面向術語對應表建構、語義相似度計算等跨領域對齊工作
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional, Union
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dataclasses import dataclass
import pickle
from pathlib import Path

from .data_loader import AspectSentiment
from .preprocessor import ProcessedText
from .feature_extractor import FeatureVector


@dataclass
class AbstractAspect:
    """抽象面向數據結構"""
    name: str                           # 抽象面向名稱
    description: str                    # 描述
    domain_mappings: Dict[str, List[str]]  # 領域映射 {領域: [具體面向列表]}
    keywords: List[str]                 # 關鍵詞
    embedding: Optional[np.ndarray]     # 抽象面向的向量表示
    
    def to_dict(self) -> Dict:
        """轉換為字典格式"""
        return {
            'name': self.name,
            'description': self.description,
            'domain_mappings': self.domain_mappings,
            'keywords': self.keywords,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }


class AbstractAspectDefinition:
    """五個抽象面向定義器"""
    
    def __init__(self):
        """初始化抽象面向定義器"""
        self.abstract_aspects = self._define_abstract_aspects()
        self.domain_aspect_mapping = {}
        
    def _define_abstract_aspects(self) -> Dict[str, AbstractAspect]:
        """定義五個抽象面向"""
        abstract_aspects = {}
        
        # 1. 品質面向 (Quality)
        quality_aspect = AbstractAspect(
            name="quality",
            description="產品或服務的整體品質，包括功能性、可靠性、耐用性等",
            domain_mappings={
                "restaurant": ["FOOD", "food quality", "taste", "freshness"],
                "laptop": ["GENERAL", "build quality", "performance", "reliability"], 
                "hotel": ["ROOM", "service quality", "cleanliness", "comfort"]
            },
            keywords=[
                "quality", "good", "bad", "excellent", "poor", "great", "terrible",
                "high", "low", "best", "worst", "perfect", "awful", "amazing",
                "outstanding", "horrible", "superb", "inferior", "superior"
            ],
            embedding=None
        )
        abstract_aspects["quality"] = quality_aspect
        
        # 2. 價格面向 (Price)
        price_aspect = AbstractAspect(
            name="price",
            description="產品或服務的價格合理性、性價比等",
            domain_mappings={
                "restaurant": ["PRICE", "cost", "value", "expensive"],
                "laptop": ["PRICE", "cost", "value", "budget", "worth"],
                "hotel": ["PRICE", "rate", "cost", "value", "expensive"]
            },
            keywords=[
                "price", "cost", "expensive", "cheap", "affordable", "reasonable",
                "value", "money", "budget", "worth", "overpriced", "underpriced",
                "deal", "bargain", "pricey", "costly", "inexpensive", "economical"
            ],
            embedding=None
        )
        abstract_aspects["price"] = price_aspect
        
        # 3. 服務面向 (Service)
        service_aspect = AbstractAspect(
            name="service", 
            description="服務態度、響應速度、專業性等人員服務相關",
            domain_mappings={
                "restaurant": ["SERVICE", "staff", "waiter", "waitress", "server"],
                "laptop": ["SUPPORT", "customer service", "technical support"],
                "hotel": ["SERVICE", "staff", "reception", "housekeeping"]
            },
            keywords=[
                "service", "staff", "waiter", "waitress", "server", "employee",
                "friendly", "rude", "helpful", "professional", "attentive",
                "support", "assistance", "courteous", "responsive", "prompt"
            ],
            embedding=None
        )
        abstract_aspects["service"] = service_aspect
        
        # 4. 氛圍面向 (Ambiance)
        ambiance_aspect = AbstractAspect(
            name="ambiance",
            description="環境氛圍、外觀設計、舒適度等環境相關特徵",
            domain_mappings={
                "restaurant": ["AMBIANCE", "atmosphere", "decor", "ambience"],
                "laptop": ["DESIGN", "appearance", "look", "aesthetics"],
                "hotel": ["AMBIANCE", "atmosphere", "decor", "environment"]
            },
            keywords=[
                "ambiance", "atmosphere", "environment", "decor", "design",
                "comfortable", "cozy", "elegant", "modern", "stylish",
                "beautiful", "ugly", "pleasant", "nice", "lovely", "gorgeous"
            ],
            embedding=None
        )
        abstract_aspects["ambiance"] = ambiance_aspect
        
        # 5. 便利性面向 (Convenience)
        convenience_aspect = AbstractAspect(
            name="convenience",
            description="使用便利性、可及性、實用功能等便利性相關特徵",
            domain_mappings={
                "restaurant": ["LOCATION", "parking", "reservation", "accessibility"],
                "laptop": ["PORTABILITY", "battery", "connectivity", "usability"],
                "hotel": ["LOCATION", "transportation", "amenities", "accessibility"]
            },
            keywords=[
                "convenient", "easy", "accessible", "location", "portable",
                "battery", "connectivity", "wifi", "parking", "transportation",
                "nearby", "close", "far", "difficult", "inconvenient", "handy"
            ],
            embedding=None
        )
        abstract_aspects["convenience"] = convenience_aspect
        
        return abstract_aspects
    
    def get_abstract_aspects(self) -> Dict[str, AbstractAspect]:
        """獲取所有抽象面向"""
        return self.abstract_aspects
    
    def get_domain_mappings(self, domain: str) -> Dict[str, List[str]]:
        """
        獲取特定領域的面向映射
        
        Args:
            domain: 領域名稱
            
        Returns:
            抽象面向到具體面向的映射
        """
        mappings = {}
        for abstract_name, abstract_aspect in self.abstract_aspects.items():
            if domain in abstract_aspect.domain_mappings:
                mappings[abstract_name] = abstract_aspect.domain_mappings[domain]
        return mappings
    
    def map_aspect_to_abstract(self, aspect: str, domain: str) -> Optional[str]:
        """
        將具體面向映射到抽象面向
        
        Args:
            aspect: 具體面向名稱
            domain: 領域名稱
            
        Returns:
            抽象面向名稱，如果無法映射則返回None
        """
        aspect_lower = aspect.lower()
        
        for abstract_name, abstract_aspect in self.abstract_aspects.items():
            if domain in abstract_aspect.domain_mappings:
                domain_aspects = [a.lower() for a in abstract_aspect.domain_mappings[domain]]
                if aspect_lower in domain_aspects:
                    return abstract_name
                    
                # 檢查是否包含關鍵詞
                for keyword in abstract_aspect.keywords:
                    if keyword in aspect_lower:
                        return abstract_name
        
        # 如果無法直接映射，返回最相似的
        return self._find_most_similar_abstract_aspect(aspect, domain)
    
    def _find_most_similar_abstract_aspect(self, aspect: str, domain: str) -> Optional[str]:
        """找到最相似的抽象面向"""
        aspect_words = set(aspect.lower().split())
        max_similarity = 0
        best_match = None
        
        for abstract_name, abstract_aspect in self.abstract_aspects.items():
            # 計算與關鍵詞的重疊度
            keyword_words = set([kw.lower() for kw in abstract_aspect.keywords])
            similarity = len(aspect_words.intersection(keyword_words)) / len(aspect_words.union(keyword_words))
            
            if similarity > max_similarity and similarity > 0.1:  # 閾值
                max_similarity = similarity
                best_match = abstract_name
        
        return best_match


class AspectTermMappingBuilder:
    """面向術語對應表建構器"""
    
    def __init__(self, abstract_definition: AbstractAspectDefinition):
        """
        初始化對應表建構器
        
        Args:
            abstract_definition: 抽象面向定義器
        """
        self.abstract_definition = abstract_definition
        self.term_mappings: Dict[str, Dict[str, List[str]]] = {}  # {領域: {抽象面向: [術語列表]}}
        self.term_embeddings: Dict[str, np.ndarray] = {}  # 術語向量表示
        self.similarity_matrix: Optional[np.ndarray] = None
        
    def build_mappings(self, data: List[AspectSentiment], 
                      feature_vectors: Optional[List[FeatureVector]] = None):
        """
        建構面向術語對應表
        
        Args:
            data: AspectSentiment 數據列表
            feature_vectors: 特徵向量列表（用於計算語義相似度）
        """
        # 收集所有面向術語
        domain_terms = defaultdict(lambda: defaultdict(list))
        
        for i, sample in enumerate(data):
            if not sample.aspect_term:
                continue
                
            # 映射到抽象面向
            abstract_aspect = self.abstract_definition.map_aspect_to_abstract(
                sample.aspect_category or sample.aspect_term, 
                sample.domain
            )
            
            if abstract_aspect:
                domain_terms[sample.domain][abstract_aspect].append(sample.aspect_term)
                
                # 如果有特徵向量，保存術語的向量表示
                if feature_vectors and i < len(feature_vectors):
                    term_key = f"{sample.domain}_{sample.aspect_term}"
                    self.term_embeddings[term_key] = feature_vectors[i].bert_features.numpy()
        
        # 去重並統計頻率
        for domain in domain_terms:
            self.term_mappings[domain] = {}
            for abstract_aspect in domain_terms[domain]:
                term_counts = Counter(domain_terms[domain][abstract_aspect])
                # 保留頻率大於1的術語
                frequent_terms = [term for term, count in term_counts.items() if count > 1]
                self.term_mappings[domain][abstract_aspect] = frequent_terms
        
        print(f"已建構 {len(self.term_mappings)} 個領域的術語對應表")
    
    def get_cross_domain_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """
        獲取跨領域術語映射
        
        Returns:
            {抽象面向: {領域: [術語列表]}}
        """
        cross_domain_mappings = defaultdict(dict)
        
        for domain, domain_mappings in self.term_mappings.items():
            for abstract_aspect, terms in domain_mappings.items():
                cross_domain_mappings[abstract_aspect][domain] = terms
        
        return dict(cross_domain_mappings)
    
    def find_similar_terms_across_domains(self, 
                                        term: str, 
                                        source_domain: str,
                                        target_domains: List[str],
                                        top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        查找跨領域相似術語
        
        Args:
            term: 源術語
            source_domain: 源領域
            target_domains: 目標領域列表
            top_k: 返回的相似術語數量
            
        Returns:
            {目標領域: [(相似術語, 相似度分數)]}
        """
        if not self.term_embeddings:
            return {}
        
        source_key = f"{source_domain}_{term}"
        if source_key not in self.term_embeddings:
            return {}
        
        source_embedding = self.term_embeddings[source_key].reshape(1, -1)
        similar_terms = {}
        
        for target_domain in target_domains:
            domain_similarities = []
            
            for key, embedding in self.term_embeddings.items():
                if key.startswith(f"{target_domain}_"):
                    target_term = key.split(f"{target_domain}_", 1)[1]
                    target_embedding = embedding.reshape(1, -1)
                    
                    # 計算餘弦相似度
                    similarity = cosine_similarity(source_embedding, target_embedding)[0][0]
                    domain_similarities.append((target_term, similarity))
            
            # 按相似度排序並取前k個
            domain_similarities.sort(key=lambda x: x[1], reverse=True)
            similar_terms[target_domain] = domain_similarities[:top_k]
        
        return similar_terms
    
    def cluster_terms_by_similarity(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """
        基於語義相似度聚類術語
        
        Args:
            n_clusters: 聚類數量
            
        Returns:
            {聚類ID: [術語列表]}
        """
        if not self.term_embeddings:
            return {}
        
        # 準備數據
        terms = list(self.term_embeddings.keys())
        embeddings = np.array(list(self.term_embeddings.values()))
        
        # K-means 聚類
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 整理聚類結果
        clusters = defaultdict(list)
        for term, label in zip(terms, cluster_labels):
            # 移除領域前綴
            clean_term = term.split('_', 1)[1] if '_' in term else term
            clusters[label].append(clean_term)
        
        return dict(clusters)


class SemanticSimilarityCalculator:
    """語義相似度計算器"""
    
    def __init__(self, embedding_dim: int = 768):
        """
        初始化語義相似度計算器
        
        Args:
            embedding_dim: 嵌入向量維度
        """
        self.embedding_dim = embedding_dim
        
    def calculate_cosine_similarity(self, 
                                   embedding1: np.ndarray, 
                                   embedding2: np.ndarray) -> float:
        """
        計算兩個向量的餘弦相似度
        
        Args:
            embedding1: 向量1
            embedding2: 向量2
            
        Returns:
            餘弦相似度分數
        """
        # 確保向量是2D
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
            
        return cosine_similarity(embedding1, embedding2)[0][0]
    
    def calculate_batch_similarity(self, 
                                 embeddings1: np.ndarray, 
                                 embeddings2: np.ndarray) -> np.ndarray:
        """
        批量計算相似度矩陣
        
        Args:
            embeddings1: 嵌入矩陣1 [n1, dim]
            embeddings2: 嵌入矩陣2 [n2, dim]
            
        Returns:
            相似度矩陣 [n1, n2]
        """
        return cosine_similarity(embeddings1, embeddings2)
    
    def find_most_similar(self, 
                         query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        找到最相似的候選項
        
        Args:
            query_embedding: 查詢向量
            candidate_embeddings: 候選向量矩陣
            top_k: 返回的相似項數量
            
        Returns:
            [(索引, 相似度分數)] 列表
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # 獲取最相似的top_k個
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def calculate_aspect_coherence(self, 
                                 aspect_embeddings: List[np.ndarray]) -> float:
        """
        計算面向內聚性（同一面向術語的相似度）
        
        Args:
            aspect_embeddings: 同一面向的術語嵌入列表
            
        Returns:
            內聚性分數
        """
        if len(aspect_embeddings) < 2:
            return 1.0
        
        embeddings_matrix = np.array(aspect_embeddings)
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # 計算平均相似度（排除對角線）
        mask = np.ones_like(similarity_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        
        return similarity_matrix[mask].mean()
    
    def calculate_aspect_separation(self, 
                                  aspect1_embeddings: List[np.ndarray],
                                  aspect2_embeddings: List[np.ndarray]) -> float:
        """
        計算面向分離度（不同面向術語的差異性）
        
        Args:
            aspect1_embeddings: 面向1的術語嵌入列表
            aspect2_embeddings: 面向2的術語嵌入列表
            
        Returns:
            分離度分數（值越小表示分離度越高）
        """
        if not aspect1_embeddings or not aspect2_embeddings:
            return 0.0
        
        embeddings1 = np.array(aspect1_embeddings)
        embeddings2 = np.array(aspect2_embeddings)
        
        cross_similarity = cosine_similarity(embeddings1, embeddings2)
        
        return cross_similarity.mean()


class CrossDomainAligner:
    """跨領域對齊器主類"""
    
    def __init__(self):
        """初始化跨領域對齊器"""
        self.abstract_definition = AbstractAspectDefinition()
        self.mapping_builder = AspectTermMappingBuilder(self.abstract_definition)
        self.similarity_calculator = SemanticSimilarityCalculator()
        
        self.aligned_data: Dict[str, List[AspectSentiment]] = {}
        self.alignment_statistics: Dict = {}
        
    def align_domains(self, 
                     data: List[AspectSentiment],
                     feature_vectors: Optional[List[FeatureVector]] = None):
        """
        執行跨領域對齊
        
        Args:
            data: AspectSentiment 數據列表
            feature_vectors: 特徵向量列表
        """
        # 建構術語對應表
        self.mapping_builder.build_mappings(data, feature_vectors)
        
        # 按抽象面向重新組織數據
        self._reorganize_by_abstract_aspects(data)
        
        # 計算對齊統計資訊
        self._calculate_alignment_statistics()
        
        print("跨領域對齊完成")
    
    def _reorganize_by_abstract_aspects(self, data: List[AspectSentiment]):
        """按抽象面向重新組織數據"""
        for abstract_aspect in self.abstract_definition.get_abstract_aspects():
            self.aligned_data[abstract_aspect] = []
        
        for sample in data:
            abstract_aspect = self.abstract_definition.map_aspect_to_abstract(
                sample.aspect_category or sample.aspect_term,
                sample.domain
            )
            
            if abstract_aspect and abstract_aspect in self.aligned_data:
                # 創建對齊後的樣本
                aligned_sample = AspectSentiment(
                    text=sample.text,
                    aspect_term=sample.aspect_term,
                    aspect_category=abstract_aspect,  # 使用抽象面向
                    sentiment=sample.sentiment,
                    start_position=sample.start_position,
                    end_position=sample.end_position,
                    domain=sample.domain,
                    sentence_id=sample.sentence_id
                )
                
                self.aligned_data[abstract_aspect].append(aligned_sample)
    
    def _calculate_alignment_statistics(self):
        """計算對齊統計資訊"""
        self.alignment_statistics = {
            'total_samples': sum(len(samples) for samples in self.aligned_data.values()),
            'abstract_aspects': list(self.aligned_data.keys()),
            'samples_per_aspect': {
                aspect: len(samples) for aspect, samples in self.aligned_data.items()
            },
            'domains_per_aspect': {},
            'cross_domain_coverage': {}
        }
        
        # 計算每個抽象面向涵蓋的領域
        for aspect, samples in self.aligned_data.items():
            domains = set(sample.domain for sample in samples)
            self.alignment_statistics['domains_per_aspect'][aspect] = list(domains)
            self.alignment_statistics['cross_domain_coverage'][aspect] = len(domains)
    
    def get_aligned_data(self, abstract_aspect: Optional[str] = None) -> Union[Dict[str, List[AspectSentiment]], List[AspectSentiment]]:
        """
        獲取對齊後的數據
        
        Args:
            abstract_aspect: 指定的抽象面向，如果為None則返回所有
            
        Returns:
            對齊後的數據
        """
        if abstract_aspect:
            return self.aligned_data.get(abstract_aspect, [])
        return self.aligned_data
    
    def get_cross_domain_term_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """獲取跨領域術語映射"""
        return self.mapping_builder.get_cross_domain_mappings()
    
    def evaluate_alignment_quality(self, feature_vectors: List[FeatureVector]) -> Dict[str, float]:
        """
        評估對齊品質
        
        Args:
            feature_vectors: 特徵向量列表
            
        Returns:
            評估指標字典
        """
        if not feature_vectors:
            return {}
        
        evaluation_metrics = {}
        
        # 為每個抽象面向計算內聚性和分離度
        aspect_embeddings = defaultdict(list)
        
        # 收集每個抽象面向的嵌入向量
        for aspect, samples in self.aligned_data.items():
            for sample in samples:
                # 找到對應的特徵向量（這裡需要根據實際情況調整索引邏輯）
                for fv in feature_vectors:
                    if fv.text_length > 0:  # 簡化的匹配邏輯
                        aspect_embeddings[aspect].append(fv.bert_features.numpy())
                        break
        
        # 計算內聚性
        cohesion_scores = {}
        for aspect, embeddings in aspect_embeddings.items():
            if len(embeddings) > 1:
                cohesion_scores[aspect] = self.similarity_calculator.calculate_aspect_coherence(embeddings)
        
        evaluation_metrics['average_cohesion'] = np.mean(list(cohesion_scores.values())) if cohesion_scores else 0.0
        evaluation_metrics['cohesion_per_aspect'] = cohesion_scores
        
        # 計算分離度
        aspect_pairs = []
        aspects = list(aspect_embeddings.keys())
        
        for i in range(len(aspects)):
            for j in range(i + 1, len(aspects)):
                if len(aspect_embeddings[aspects[i]]) > 0 and len(aspect_embeddings[aspects[j]]) > 0:
                    separation = self.similarity_calculator.calculate_aspect_separation(
                        aspect_embeddings[aspects[i]], 
                        aspect_embeddings[aspects[j]]
                    )
                    aspect_pairs.append(separation)
        
        evaluation_metrics['average_separation'] = 1.0 - np.mean(aspect_pairs) if aspect_pairs else 1.0
        evaluation_metrics['alignment_statistics'] = self.alignment_statistics
        
        return evaluation_metrics
    
    def save_alignment(self, save_path: str):
        """保存對齊結果"""
        alignment_data = {
            'abstract_definition': self.abstract_definition,
            'mapping_builder': self.mapping_builder,
            'aligned_data': self.aligned_data,
            'alignment_statistics': self.alignment_statistics
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(alignment_data, f)
        
        print(f"對齊結果已保存至: {save_path}")
    
    def load_alignment(self, load_path: str):
        """載入對齊結果"""
        with open(load_path, 'rb') as f:
            alignment_data = pickle.load(f)
        
        self.abstract_definition = alignment_data['abstract_definition']
        self.mapping_builder = alignment_data['mapping_builder']
        self.aligned_data = alignment_data['aligned_data']
        self.alignment_statistics = alignment_data['alignment_statistics']
        
        print(f"對齊結果已載入: {load_path}")