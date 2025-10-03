# 語義空間可視化器模組
"""
語義空間可視化器實現

提供以下語義空間可視化功能：
- t-SNE 降維圖生成: 高維語義空間的2D/3D可視化
- 方面聚類可視化: 展示不同方面的語義聚類效果
- 跨領域語義分布圖: 可視化不同領域的語義空間分佈
- 語義相似性網絡圖: 展示語義向量間的相似性關係
- 詞嵌入可視化: 詞向量的語義空間分佈
- 領域適應可視化: 展示領域適應前後的語義空間變化
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')


class SemanticSpaceVisualizer:
    """語義空間可視化器"""
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "semantic_plots"):
        """
        初始化語義空間可視化器
        
        Args:
            save_plots: 是否保存圖表
            plot_dir: 圖表保存目錄
        """
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        
        if self.save_plots:
            self.plot_dir.mkdir(exist_ok=True)
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 顏色配置
        self.colors = {
            'restaurant': '#FF6B6B',
            'laptop': '#4ECDC4', 
            'phone': '#45B7D1',
            'camera': '#96CEB4',
            'hotel': '#FFEAA7',
            'book': '#DDA0DD',
            'positive': '#2ECC71',
            'negative': '#E74C3C',
            'neutral': '#95A5A6'
        }
        
        # 方面顏色映射
        self.aspect_colors = {
            'quality': '#E74C3C',
            'price': '#3498DB', 
            'service': '#2ECC71',
            'ambiance': '#F39C12',
            'convenience': '#9B59B6'
        }
        
        # 領域名稱映射
        self.domain_names = {
            'restaurant': '餐廳',
            'laptop': '筆記本電腦',
            'phone': '手機',
            'camera': '相機',
            'hotel': '酒店',
            'book': '書籍'
        }
    
    def plot_tsne_visualization(self, embeddings: torch.Tensor,
                              labels: List[str],
                              colors: List[str] = None,
                              title: str = "t-SNE 語義空間可視化",
                              perplexity: int = 30,
                              n_iter: int = 1000) -> None:
        """
        繪製 t-SNE 降維可視化圖
        
        Args:
            embeddings: 詞嵌入向量 [n_samples, embedding_dim]
            labels: 樣本標籤
            colors: 顏色標籤
            title: 圖表標題
            perplexity: t-SNE 困惑度參數
            n_iter: 迭代次數
        """
        print("生成 t-SNE 語義空間可視化圖")
        
        # 轉換張量格式
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # 執行 t-SNE 降維
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 根據標籤分組繪製
        unique_labels = list(set(labels))
        
        if colors is None:
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        else:
            # 將顏色標籤轉換為實際顏色
            color_map = {label: self.colors.get(label, plt.cm.Set3(i/len(unique_labels))) 
                        for i, label in enumerate(unique_labels)}
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            if colors is None:
                color = colors[i]
            else:
                color = color_map[label]
            
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[color], label=label, alpha=0.7, s=50)
        
        ax.set_xlabel('t-SNE 維度 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE 維度 2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_3d_tsne_visualization(self, embeddings: torch.Tensor,
                                 labels: List[str],
                                 title: str = "3D t-SNE 語義空間可視化") -> None:
        """
        繪製 3D t-SNE 可視化圖
        
        Args:
            embeddings: 詞嵌入向量
            labels: 樣本標籤
            title: 圖表標題
        """
        print("生成 3D t-SNE 語義空間可視化圖")
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # 執行 3D t-SNE
        tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings)
        
        # 創建 3D 交互式圖表
        fig = go.Figure()
        
        unique_labels = list(set(labels))
        
        for label in unique_labels:
            mask = np.array(labels) == label
            color = self.colors.get(label, '#1f77b4')
            
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[mask, 0],
                y=embeddings_3d[mask, 1], 
                z=embeddings_3d[mask, 2],
                mode='markers',
                marker=dict(size=5, color=color, opacity=0.7),
                name=label,
                hovertemplate=f'<b>{label}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='t-SNE 維度 1',
                yaxis_title='t-SNE 維度 2',
                zaxis_title='t-SNE 維度 3'
            ),
            width=900,
            height=700
        )

        if self.save_plots:
            fig.write_image(str(self.plot_dir / f'{title.replace(" ", "_")}.png'))

        fig.show()
    
    def plot_aspect_clustering(self, embeddings: torch.Tensor,
                             aspect_labels: List[str],
                             sentiment_labels: List[str] = None,
                             title: str = "方面聚類可視化") -> None:
        """
        繪製方面聚類可視化
        
        Args:
            embeddings: 詞嵌入向量
            aspect_labels: 方面標籤
            sentiment_labels: 情感標籤（可選）
            title: 圖表標題
        """
        print("生成方面聚類可視化圖")
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # PCA 降維到 2D
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # 左圖：按方面著色
        unique_aspects = list(set(aspect_labels))
        for aspect in unique_aspects:
            mask = np.array(aspect_labels) == aspect
            color = self.aspect_colors.get(aspect, '#1f77b4')
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=color, label=aspect, alpha=0.7, s=50)
        
        ax1.set_xlabel(f'PCA 維度 1 (解釋方差: {pca.explained_variance_ratio_[0]:.2%})', 
                      fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'PCA 維度 2 (解釋方差: {pca.explained_variance_ratio_[1]:.2%})', 
                      fontsize=12, fontweight='bold')
        ax1.set_title('按方面分類', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右圖：如果有情感標籤，按情感著色
        if sentiment_labels is not None:
            unique_sentiments = list(set(sentiment_labels))
            for sentiment in unique_sentiments:
                mask = np.array(sentiment_labels) == sentiment
                color = self.colors.get(sentiment, '#1f77b4')
                ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=color, label=sentiment, alpha=0.7, s=50)
            
            ax2.set_title('按情感分類', fontsize=12, fontweight='bold')
            ax2.legend()
        else:
            # K-means 聚類
            n_clusters = len(unique_aspects)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
            for i in range(n_clusters):
                mask = cluster_labels == i
                ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i]], label=f'聚類 {i+1}', alpha=0.7, s=50)
            
            # 繪製聚類中心
            centers_2d = pca.transform(kmeans.cluster_centers_)
            ax2.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                       c='red', marker='X', s=200, label='聚類中心')
            
            ax2.set_title('K-means 聚類結果', fontsize=12, fontweight='bold')
            ax2.legend()
        
        ax2.set_xlabel(f'PCA 維度 1', fontsize=12, fontweight='bold')
        ax2.set_ylabel(f'PCA 維度 2', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cross_domain_semantic_space(self, domain_embeddings: Dict[str, torch.Tensor],
                                       title: str = "跨領域語義空間分佈") -> None:
        """
        繪製跨領域語義空間分佈圖
        
        Args:
            domain_embeddings: 領域嵌入 {domain_name: embeddings}
            title: 圖表標題
        """
        print("生成跨領域語義空間分佈圖")
        
        # 合併所有領域的嵌入
        all_embeddings = []
        all_labels = []
        
        for domain_name, embeddings in domain_embeddings.items():
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_labels.extend([domain_name] * len(embeddings))
        
        all_embeddings = np.vstack(all_embeddings)
        
        # t-SNE 降維
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # 創建交互式圖表
        fig = go.Figure()
        
        for domain_name in domain_embeddings.keys():
            mask = np.array(all_labels) == domain_name
            color = self.colors.get(domain_name, '#1f77b4')
            display_name = self.domain_names.get(domain_name, domain_name)
            
            fig.add_trace(go.Scatter(
                x=embeddings_2d[mask, 0],
                y=embeddings_2d[mask, 1],
                mode='markers',
                marker=dict(color=color, size=8, opacity=0.7),
                name=display_name,
                hovertemplate=f'<b>{display_name}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='t-SNE 維度 1',
            yaxis_title='t-SNE 維度 2',
            width=900,
            height=700,
            hovermode='closest'
        )

        if self.save_plots:
            fig.write_image(str(self.plot_dir / f'{title.replace(" ", "_")}.png'))

        fig.show()
        
        # 同時生成靜態版本
        plt.figure(figsize=(12, 8))
        
        for domain_name in domain_embeddings.keys():
            mask = np.array(all_labels) == domain_name
            color = self.colors.get(domain_name, '#1f77b4')
            display_name = self.domain_names.get(domain_name, domain_name)
            
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=color, label=display_name, alpha=0.7, s=50)
        
        plt.xlabel('t-SNE 維度 1', fontsize=12, fontweight='bold')
        plt.ylabel('t-SNE 維度 2', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}_static.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_semantic_similarity_network(self, embeddings: torch.Tensor,
                                       labels: List[str],
                                       threshold: float = 0.7,
                                       title: str = "語義相似性網絡圖") -> None:
        """
        繪製語義相似性網絡圖
        
        Args:
            embeddings: 詞嵌入向量
            labels: 標籤
            threshold: 相似度閾值
            title: 圖表標題
        """
        print("生成語義相似性網絡圖")
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # 計算相似度矩陣（余弦相似度）
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # 創建網絡圖的節點和邊
        n_nodes = len(labels)
        
        # 使用 PCA 降維來確定節點位置
        pca = PCA(n_components=2, random_state=42)
        node_positions = pca.fit_transform(embeddings)
        
        # 創建交互式網絡圖
        fig = go.Figure()
        
        # 添加邊（相似性連接）
        edge_x = []
        edge_y = []
        edge_info = []
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if similarity_matrix[i, j] > threshold:
                    edge_x.extend([node_positions[i, 0], node_positions[j, 0], None])
                    edge_y.extend([node_positions[i, 1], node_positions[j, 1], None])
                    edge_info.append(f'{labels[i]} - {labels[j]}: {similarity_matrix[i, j]:.3f}')
        
        # 繪製邊
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='lightgray'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # 添加節點
        node_colors = [self.colors.get(label.split('_')[0] if '_' in label else label, '#1f77b4') 
                      for label in labels]
        
        fig.add_trace(go.Scatter(
            x=node_positions[:, 0],
            y=node_positions[:, 1],
            mode='markers+text',
            marker=dict(size=15, color=node_colors, 
                       line=dict(width=2, color='black')),
            text=labels,
            textposition="middle center",
            textfont=dict(size=8, color='white'),
            hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=900,
            height=700
        )

        if self.save_plots:
            fig.write_image(str(self.plot_dir / f'{title.replace(" ", "_")}.png'))

        fig.show()
    
    def plot_word_embedding_visualization(self, word_embeddings: Dict[str, torch.Tensor],
                                        title: str = "詞嵌入可視化") -> None:
        """
        繪製詞嵌入可視化
        
        Args:
            word_embeddings: 詞嵌入 {word: embedding_vector}
            title: 圖表標題
        """
        print("生成詞嵌入可視化圖")
        
        words = list(word_embeddings.keys())
        embeddings = []
        
        for word in words:
            embedding = word_embeddings[word]
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # t-SNE 降維
        tsne = TSNE(n_components=2, perplexity=min(30, len(words)-1), n_iter=1000, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 創建圖表
        plt.figure(figsize=(14, 10))
        
        # 繪製詞向量點
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            alpha=0.7, s=50, c=range(len(words)), cmap='viridis')
        
        # 添加詞標籤
        for i, word in enumerate(words):
            plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.xlabel('t-SNE 維度 1', fontsize=12, fontweight='bold')
        plt.ylabel('t-SNE 維度 2', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_domain_adaptation_visualization(self, source_embeddings: torch.Tensor,
                                           target_embeddings: torch.Tensor,
                                           adapted_embeddings: torch.Tensor = None,
                                           source_labels: List[str] = None,
                                           target_labels: List[str] = None,
                                           title: str = "領域適應可視化") -> None:
        """
        繪製領域適應前後的語義空間變化
        
        Args:
            source_embeddings: 源領域嵌入
            target_embeddings: 目標領域嵌入
            adapted_embeddings: 適應後的嵌入（可選）
            source_labels: 源領域標籤
            target_labels: 目標領域標籤
            title: 圖表標題
        """
        print("生成領域適應可視化圖")
        
        # 轉換張量格式
        if isinstance(source_embeddings, torch.Tensor):
            source_embeddings = source_embeddings.detach().cpu().numpy()
        if isinstance(target_embeddings, torch.Tensor):
            target_embeddings = target_embeddings.detach().cpu().numpy()
        if adapted_embeddings is not None and isinstance(adapted_embeddings, torch.Tensor):
            adapted_embeddings = adapted_embeddings.detach().cpu().numpy()
        
        # 合併嵌入進行統一降維
        all_embeddings = [source_embeddings, target_embeddings]
        all_labels = ['源領域'] * len(source_embeddings) + ['目標領域'] * len(target_embeddings)
        
        if adapted_embeddings is not None:
            all_embeddings.append(adapted_embeddings)
            all_labels.extend(['適應後'] * len(adapted_embeddings))
        
        combined_embeddings = np.vstack(all_embeddings)
        
        # t-SNE 降維
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        embeddings_2d = tsne.fit_transform(combined_embeddings)
        
        # 分離不同類型的嵌入
        source_2d = embeddings_2d[:len(source_embeddings)]
        target_2d = embeddings_2d[len(source_embeddings):len(source_embeddings)+len(target_embeddings)]
        
        if adapted_embeddings is not None:
            adapted_2d = embeddings_2d[len(source_embeddings)+len(target_embeddings):]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # 左圖：適應前
        ax1.scatter(source_2d[:, 0], source_2d[:, 1], 
                   c='blue', alpha=0.6, label='源領域', s=50)
        ax1.scatter(target_2d[:, 0], target_2d[:, 1], 
                   c='red', alpha=0.6, label='目標領域', s=50)
        
        ax1.set_xlabel('t-SNE 維度 1', fontsize=12, fontweight='bold')
        ax1.set_ylabel('t-SNE 維度 2', fontsize=12, fontweight='bold')
        ax1.set_title('適應前', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右圖：適應後（如果有）
        if adapted_embeddings is not None:
            ax2.scatter(source_2d[:, 0], source_2d[:, 1], 
                       c='blue', alpha=0.6, label='源領域', s=50)
            ax2.scatter(adapted_2d[:, 0], adapted_2d[:, 1], 
                       c='green', alpha=0.6, label='適應後目標領域', s=50)
            
            ax2.set_xlabel('t-SNE 維度 1', fontsize=12, fontweight='bold')
            ax2.set_ylabel('t-SNE 維度 2', fontsize=12, fontweight='bold')
            ax2.set_title('適應後', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_hierarchical_clustering(self, embeddings: torch.Tensor,
                                   labels: List[str],
                                   title: str = "層次聚類樹狀圖") -> None:
        """
        繪製層次聚類樹狀圖
        
        Args:
            embeddings: 詞嵌入向量
            labels: 標籤
            title: 圖表標題
        """
        print("生成層次聚類樹狀圖")
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # 計算距離矩陣
        distances = pdist(embeddings, metric='euclidean')
        
        # 執行層次聚類
        linkage_matrix = linkage(distances, method='ward')
        
        # 創建樹狀圖
        plt.figure(figsize=(15, 8))
        
        dendrogram(linkage_matrix, 
                  labels=labels,
                  leaf_rotation=45,
                  leaf_font_size=10)
        
        plt.xlabel('樣本', fontsize=12, fontweight='bold')
        plt.ylabel('距離', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_semantic_dashboard(self, comprehensive_data: Dict[str, Any],
                                title: str = "語義空間綜合儀表板") -> None:
        """
        創建語義空間綜合儀表板
        
        Args:
            comprehensive_data: 綜合數據
            title: 圖表標題
        """
        print("創建語義空間綜合儀表板")
        
        # 創建子圖
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                '2D語義空間分佈', '3D語義空間', '方面聚類',
                '跨領域分佈', '相似性網絡', '詞嵌入分佈',
                '領域適應效果', '聚類質量評估', '語義密度分析'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter3d"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "heatmap"}]
            ]
        )
        
        # 可以根據實際數據添加各種圖表...
        
        # 更新佈局
        fig.update_layout(
            title_text=title,
            title_x=0.5,
            title_font_size=20,
            showlegend=True,
            height=1200
        )
        
        if self.save_plots:
            fig.write_image(str(self.plot_dir / f'{title.replace(" ", "_")}.png'))
            print(f"語義空間圖表已保存至: {self.plot_dir / f'{title.replace(' ', '_')}.png'}")
        
        fig.show()
    
    def analyze_semantic_space(self, embeddings: torch.Tensor,
                             labels: List[str] = None) -> Dict[str, Any]:
        """
        分析語義空間特性
        
        Args:
            embeddings: 詞嵌入向量
            labels: 標籤
            
        Returns:
            analysis_results: 分析結果
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # 計算基本統計
        results = {
            "基本統計": {
                "樣本數量": len(embeddings),
                "嵌入維度": embeddings.shape[1],
                "平均向量長度": float(np.mean(np.linalg.norm(embeddings, axis=1))),
                "向量長度標準差": float(np.std(np.linalg.norm(embeddings, axis=1)))
            },
            "分佈特性": {},
            "聚類特性": {}
        }
        
        # 計算主成分分析
        pca = PCA()
        pca.fit(embeddings)
        
        results["分佈特性"] = {
            "前5個主成分解釋方差": pca.explained_variance_ratio_[:5].tolist(),
            "累積解釋方差": np.cumsum(pca.explained_variance_ratio_)[:10].tolist(),
            "有效維度數(95%方差)": int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)) + 1
        }
        
        # 如果有標籤，計算聚類指標
        if labels is not None:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            unique_labels = list(set(labels))
            
            if len(unique_labels) > 1:
                label_encodings = [unique_labels.index(label) for label in labels]
                
                try:
                    silhouette = silhouette_score(embeddings, label_encodings)
                    calinski = calinski_harabasz_score(embeddings, label_encodings)
                    
                    results["聚類特性"] = {
                        "輪廓係數": float(silhouette),
                        "Calinski-Harabasz指數": float(calinski),
                        "類別數量": len(unique_labels)
                    }
                except:
                    results["聚類特性"] = {"error": "無法計算聚類指標"}
        
        return results
    
    def generate_semantic_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成語義空間分析報告
        
        Args:
            analysis_results: 分析結果
            
        Returns:
            semantic_report: 語義空間分析報告
        """
        report = {
            "報告生成時間": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "分析結果": analysis_results,
            "可視化類型": [
                "t-SNE降維可視化",
                "3D語義空間", 
                "方面聚類分析",
                "跨領域語義分佈",
                "語義相似性網絡",
                "詞嵌入可視化",
                "領域適應可視化",
                "層次聚類分析",
                "語義空間儀表板"
            ],
            "保存位置": str(self.plot_dir)
        }
        
        if self.save_plots:
            with open(self.plot_dir / "semantic_analysis_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        return report