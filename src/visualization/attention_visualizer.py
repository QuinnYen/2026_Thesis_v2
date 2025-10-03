# 注意力可視化器模組
"""
注意力可視化器實現

提供以下注意力機制可視化功能：
- 注意力權重熱力圖: 展示注意力權重分佈
- 多頭注意力比較圖: 比較不同注意力頭的關注模式
- 文本-注意力對應圖: 展示文本與注意力權重的對應關係
- 注意力流向圖: 可視化注意力在序列中的流向
- 跨層注意力分析: 分析不同層級的注意力模式
- 注意力聚合可視化: 展示多種注意力機制的聚合效果
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class AttentionVisualizer:
    """注意力可視化器"""
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "attention_plots"):
        """
        初始化注意力可視化器
        
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
        
        # 自定義顏色映射
        self.attention_cmap = LinearSegmentedColormap.from_list(
            'attention', ['white', 'lightblue', 'blue', 'darkblue', 'red']
        )
        
        # 注意力機制名稱映射
        self.attention_names = {
            'self_attention': '自注意力',
            'cross_attention': '交叉注意力',
            'multi_head': '多頭注意力',
            'scaled_dot_product': '縮放點積注意力',
            'additive': '加性注意力',
            'keyword_guided': '關鍵詞導向注意力',
            'similarity_attention': '相似度注意力'
        }
    
    def plot_attention_heatmap(self, attention_weights: torch.Tensor, 
                             tokens: List[str] = None,
                             title: str = "注意力權重熱力圖") -> None:
        """
        繪製注意力權重熱力圖
        
        Args:
            attention_weights: 注意力權重矩陣 [seq_len, seq_len] 或 [batch, seq_len, seq_len]
            tokens: 標記列表
            title: 圖表標題
        """
        print("生成注意力權重熱力圖")
        
        # 處理張量維度
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        if attention_weights.ndim == 3:
            # 取第一個樣本
            attention_weights = attention_weights[0]
        elif attention_weights.ndim == 4:
            # 取第一個樣本，第一個頭
            attention_weights = attention_weights[0, 0]
        
        seq_len = attention_weights.shape[0]
        
        # 如果沒有提供標記，生成默認標記
        if tokens is None:
            tokens = [f"Token_{i}" for i in range(seq_len)]
        else:
            # 截斷或填充到正確長度
            tokens = tokens[:seq_len] + [f"Token_{i}" for i in range(len(tokens), seq_len)]
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=(max(10, seq_len * 0.5), max(8, seq_len * 0.4)))
        
        # 繪製熱力圖
        im = ax.imshow(attention_weights, cmap=self.attention_cmap, aspect='auto')
        
        # 設置刻度和標籤
        ax.set_xticks(np.arange(seq_len))
        ax.set_yticks(np.arange(seq_len))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
        
        # 添加數值標籤（僅當序列較短時）
        if seq_len <= 20:
            for i in range(seq_len):
                for j in range(seq_len):
                    text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                                 ha="center", va="center",
                                 color="white" if attention_weights[i, j] > 0.5 else "black",
                                 fontsize=8)
        
        # 設置標籤和標題
        ax.set_xlabel('Key 位置', fontsize=12, fontweight='bold')
        ax.set_ylabel('Query 位置', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # 添加顏色條
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('注意力權重', fontsize=12)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_multi_head_attention(self, multi_head_weights: torch.Tensor,
                                tokens: List[str] = None,
                                head_names: List[str] = None,
                                title: str = "多頭注意力比較") -> None:
        """
        繪製多頭注意力比較圖
        
        Args:
            multi_head_weights: 多頭注意力權重 [n_heads, seq_len, seq_len]
            tokens: 標記列表
            head_names: 注意力頭名稱
            title: 圖表標題
        """
        print("生成多頭注意力比較圖")
        
        if isinstance(multi_head_weights, torch.Tensor):
            multi_head_weights = multi_head_weights.detach().cpu().numpy()
        
        if multi_head_weights.ndim == 4:
            # 取第一個批次
            multi_head_weights = multi_head_weights[0]
        
        n_heads, seq_len, _ = multi_head_weights.shape
        
        # 處理標記
        if tokens is None:
            tokens = [f"T{i}" for i in range(seq_len)]
        else:
            tokens = tokens[:seq_len]
        
        # 處理頭名稱
        if head_names is None:
            head_names = [f"Head {i+1}" for i in range(n_heads)]
        
        # 計算子圖佈局
        cols = min(4, n_heads)
        rows = (n_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for head_idx in range(n_heads):
            row = head_idx // cols
            col = head_idx % cols
            ax = axes[head_idx] if n_heads > 1 else axes
            
            attention_weights = multi_head_weights[head_idx]
            
            # 繪製熱力圖
            im = ax.imshow(attention_weights, cmap=self.attention_cmap, aspect='auto')
            
            # 設置標籤
            if seq_len <= 10:
                ax.set_xticks(np.arange(seq_len))
                ax.set_yticks(np.arange(seq_len))
                ax.set_xticklabels(tokens, rotation=45, fontsize=8)
                ax.set_yticklabels(tokens, fontsize=8)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            
            ax.set_title(head_names[head_idx], fontsize=10, fontweight='bold')
            
            # 添加顏色條
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 隱藏多餘的子圖
        for i in range(n_heads, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attention_text_alignment(self, attention_weights: torch.Tensor,
                                    source_tokens: List[str],
                                    target_tokens: List[str],
                                    title: str = "文本-注意力對應圖") -> None:
        """
        繪製文本與注意力權重對應圖
        
        Args:
            attention_weights: 注意力權重 [target_len, source_len]
            source_tokens: 源文本標記
            target_tokens: 目標文本標記
            title: 圖表標題
        """
        print("生成文本-注意力對應圖")
        
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        target_len, source_len = attention_weights.shape
        
        # 截斷標記到合適長度
        source_tokens = source_tokens[:source_len]
        target_tokens = target_tokens[:target_len]
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                      gridspec_kw={'width_ratios': [3, 1]})
        
        # 左側：注意力權重熱力圖
        im = ax1.imshow(attention_weights, cmap=self.attention_cmap, aspect='auto')
        
        ax1.set_xticks(np.arange(source_len))
        ax1.set_yticks(np.arange(target_len))
        ax1.set_xticklabels(source_tokens, rotation=45, ha='right')
        ax1.set_yticklabels(target_tokens)
        
        ax1.set_xlabel('源文本標記', fontsize=12, fontweight='bold')
        ax1.set_ylabel('目標文本標記', fontsize=12, fontweight='bold')
        ax1.set_title('注意力權重分佈', fontsize=12, fontweight='bold')
        
        # 添加顏色條
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        
        # 右側：每個目標標記的注意力分佈
        max_attention_indices = np.argmax(attention_weights, axis=1)
        max_attention_values = np.max(attention_weights, axis=1)
        
        colors = plt.cm.viridis(max_attention_values)
        bars = ax2.barh(range(target_len), max_attention_values, color=colors)
        
        ax2.set_yticks(range(target_len))
        ax2.set_yticklabels(target_tokens)
        ax2.set_xlabel('最大注意力權重', fontsize=12, fontweight='bold')
        ax2.set_title('注意力峰值', fontsize=12, fontweight='bold')
        
        # 添加最關注的源標記標籤
        for i, (idx, val) in enumerate(zip(max_attention_indices, max_attention_values)):
            if val > 0.1:  # 只顯示顯著的注意力
                ax2.text(val + 0.01, i, source_tokens[idx][:8] + '...' if len(source_tokens[idx]) > 8 else source_tokens[idx],
                        va='center', fontsize=8)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attention_flow(self, attention_weights: torch.Tensor,
                          tokens: List[str],
                          threshold: float = 0.1,
                          title: str = "注意力流向圖") -> None:
        """
        繪製注意力流向圖
        
        Args:
            attention_weights: 注意力權重 [seq_len, seq_len]
            tokens: 標記列表
            threshold: 注意力權重閾值
            title: 圖表標題
        """
        print("生成注意力流向圖")
        
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        if attention_weights.ndim > 2:
            attention_weights = attention_weights[0]  # 取第一個樣本
        
        seq_len = attention_weights.shape[0]
        tokens = tokens[:seq_len]
        
        # 創建交互式圖表
        fig = go.Figure()
        
        # 添加節點
        node_x = []
        node_y = []
        node_text = []
        
        # 圓形佈局
        for i in range(seq_len):
            angle = 2 * np.pi * i / seq_len
            x = np.cos(angle)
            y = np.sin(angle)
            node_x.append(x)
            node_y.append(y)
            node_text.append(tokens[i])
        
        # 添加邊（注意力連接）
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j and attention_weights[i, j] > threshold:
                    edge_x.extend([node_x[i], node_x[j], None])
                    edge_y.extend([node_y[i], node_y[j], None])
                    edge_weights.append(attention_weights[i, j])
        
        # 繪製邊
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # 繪製節點
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=20, color='lightblue', 
                       line=dict(width=2, color='darkblue')),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            hoverinfo='text',
            hovertext=[f'{token}<br>位置: {i}' for i, token in enumerate(tokens)],
            showlegend=False
        ))
        
        # 設置佈局
        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=800
        )
        
        if self.save_plots:
            fig.write_image(str(self.plot_dir / f'{title.replace(" ", "_")}.png'))
        
        fig.show()
    
    def plot_attention_layer_comparison(self, layer_attentions: Dict[str, torch.Tensor],
                                      tokens: List[str] = None,
                                      title: str = "跨層注意力分析") -> None:
        """
        繪製不同層級的注意力模式比較
        
        Args:
            layer_attentions: 不同層的注意力權重 {layer_name: attention_weights}
            tokens: 標記列表
            title: 圖表標題
        """
        print("生成跨層注意力分析圖")
        
        n_layers = len(layer_attentions)
        layer_names = list(layer_attentions.keys())
        
        # 處理注意力權重
        processed_attentions = {}
        seq_len = None
        
        for layer_name, attention_weights in layer_attentions.items():
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
            
            if attention_weights.ndim > 2:
                attention_weights = attention_weights[0]  # 取第一個樣本
            if attention_weights.ndim > 2:
                attention_weights = attention_weights[0]  # 取第一個頭
            
            processed_attentions[layer_name] = attention_weights
            if seq_len is None:
                seq_len = attention_weights.shape[0]
        
        # 處理標記
        if tokens is None:
            tokens = [f"T{i}" for i in range(seq_len)]
        else:
            tokens = tokens[:seq_len]
        
        # 創建子圖
        fig, axes = plt.subplots(2, n_layers, figsize=(n_layers * 4, 8))
        if n_layers == 1:
            axes = axes.reshape(2, 1)
        
        for i, layer_name in enumerate(layer_names):
            attention_weights = processed_attentions[layer_name]
            
            # 上排：注意力熱力圖
            im1 = axes[0, i].imshow(attention_weights, cmap=self.attention_cmap, aspect='auto')
            axes[0, i].set_title(f'{layer_name} - 注意力矩陣', fontsize=10, fontweight='bold')
            
            if seq_len <= 15:
                axes[0, i].set_xticks(np.arange(seq_len))
                axes[0, i].set_yticks(np.arange(seq_len))
                axes[0, i].set_xticklabels(tokens, rotation=45, fontsize=6)
                axes[0, i].set_yticklabels(tokens, fontsize=6)
            
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # 下排：注意力分佈統計
            attention_means = np.mean(attention_weights, axis=1)
            attention_maxs = np.max(attention_weights, axis=1)
            attention_stds = np.std(attention_weights, axis=1)
            
            x_pos = np.arange(len(tokens))
            width = 0.25
            
            axes[1, i].bar(x_pos - width, attention_means, width, label='平均值', alpha=0.7)
            axes[1, i].bar(x_pos, attention_maxs, width, label='最大值', alpha=0.7)
            axes[1, i].bar(x_pos + width, attention_stds, width, label='標準差', alpha=0.7)
            
            if seq_len <= 15:
                axes[1, i].set_xticks(x_pos)
                axes[1, i].set_xticklabels(tokens, rotation=45, fontsize=6)
            
            axes[1, i].set_ylabel('注意力權重')
            axes[1, i].set_title(f'{layer_name} - 統計分析', fontsize=10, fontweight='bold')
            axes[1, i].legend(fontsize=6)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attention_aggregation(self, attention_mechanisms: Dict[str, torch.Tensor],
                                 fusion_weights: torch.Tensor = None,
                                 tokens: List[str] = None,
                                 title: str = "注意力聚合可視化") -> None:
        """
        繪製多種注意力機制的聚合效果
        
        Args:
            attention_mechanisms: 不同注意力機制的權重 {mechanism_name: attention_weights}
            fusion_weights: 融合權重
            tokens: 標記列表
            title: 圖表標題
        """
        print("生成注意力聚合可視化圖")
        
        n_mechanisms = len(attention_mechanisms)
        mechanism_names = list(attention_mechanisms.keys())
        
        # 處理注意力權重
        processed_attentions = {}
        seq_len = None
        
        for mech_name, attention_weights in attention_mechanisms.items():
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
            
            if attention_weights.ndim > 2:
                attention_weights = attention_weights[0]  # 取第一個樣本
            
            processed_attentions[mech_name] = attention_weights
            if seq_len is None:
                seq_len = attention_weights.shape[0]
        
        # 處理標記
        if tokens is None:
            tokens = [f"Token{i}" for i in range(seq_len)]
        else:
            tokens = tokens[:seq_len]
        
        # 計算融合後的注意力（如果提供融合權重）
        if fusion_weights is not None:
            if isinstance(fusion_weights, torch.Tensor):
                fusion_weights = fusion_weights.detach().cpu().numpy()
            
            fused_attention = np.zeros_like(processed_attentions[mechanism_names[0]])
            for i, mech_name in enumerate(mechanism_names):
                fused_attention += fusion_weights[i] * processed_attentions[mech_name]
            
            processed_attentions['融合注意力'] = fused_attention
            mechanism_names.append('融合注意力')
            n_mechanisms += 1
        
        # 創建子圖
        cols = min(3, n_mechanisms)
        rows = (n_mechanisms + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, mech_name in enumerate(mechanism_names):
            ax = axes[i] if n_mechanisms > 1 else axes
            attention_weights = processed_attentions[mech_name]
            
            # 繪製熱力圖
            im = ax.imshow(attention_weights, cmap=self.attention_cmap, aspect='auto')
            
            # 設置標籤
            display_name = self.attention_names.get(mech_name, mech_name)
            ax.set_title(display_name, fontsize=12, fontweight='bold')
            
            if seq_len <= 12:
                ax.set_xticks(np.arange(seq_len))
                ax.set_yticks(np.arange(seq_len))
                ax.set_xticklabels(tokens, rotation=45, fontsize=8)
                ax.set_yticklabels(tokens, fontsize=8)
            
            # 添加顏色條
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 隱藏多餘的子圖
        for i in range(n_mechanisms, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_attention_dashboard(self, comprehensive_attention_data: Dict[str, Any],
                                 title: str = "注意力機制綜合儀表板") -> None:
        """
        創建注意力機制綜合儀表板
        
        Args:
            comprehensive_attention_data: 綜合注意力數據
            title: 圖表標題
        """
        print("創建注意力機制綜合儀表板")
        
        # 創建子圖
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                '整體注意力分佈', '多頭注意力對比', '跨層注意力變化',
                '注意力聚合權重', '關鍵詞注意力', '相似度注意力',
                '注意力熵分析', '注意力稀疏度', '注意力性能指標'
            ],
            specs=[
                [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "heatmap"}, {"type": "heatmap"}],
                [{"type": "histogram"}, {"type": "scatter"}, {"type": "radar"}]
            ]
        )
        
        # 可以根據實際數據添加各種圖表...
        
        # 更新佈局
        fig.update_layout(
            title_text=title,
            title_x=0.5,
            title_font_size=20,
            showlegend=True,
            height=1000
        )
        
        if self.save_plots:
            fig.write_image(str(self.plot_dir / f'{title.replace(" ", "_")}.png'))
            print(f"注意力圖表已保存至: {self.plot_dir / f'{title.replace(' ', '_')}.png'}")
        
        fig.show()
    
    def analyze_attention_patterns(self, attention_weights: torch.Tensor,
                                 tokens: List[str] = None) -> Dict[str, Any]:
        """
        分析注意力模式
        
        Args:
            attention_weights: 注意力權重
            tokens: 標記列表
            
        Returns:
            analysis_results: 分析結果
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        if attention_weights.ndim > 2:
            attention_weights = attention_weights[0]  # 取第一個樣本
        
        seq_len = attention_weights.shape[0]
        
        # 計算各種統計指標
        results = {
            "基本統計": {
                "平均注意力": float(np.mean(attention_weights)),
                "最大注意力": float(np.max(attention_weights)),
                "最小注意力": float(np.min(attention_weights)),
                "標準差": float(np.std(attention_weights))
            },
            "注意力分佈": {
                "熵": float(-np.sum(attention_weights * np.log(attention_weights + 1e-8))),
                "基尼係數": self._calculate_gini_coefficient(attention_weights),
                "稀疏度": float(np.sum(attention_weights < 0.01) / attention_weights.size)
            },
            "注意力模式": {
                "對角線注意力比例": float(np.sum(np.diag(attention_weights)) / np.sum(attention_weights)),
                "長距離注意力比例": self._calculate_long_range_attention_ratio(attention_weights),
                "局部注意力比例": self._calculate_local_attention_ratio(attention_weights)
            }
        }
        
        # 如果提供了標記，添加標記級分析
        if tokens:
            results["標記分析"] = self._analyze_token_attention(attention_weights, tokens[:seq_len])
        
        return results
    
    def _calculate_gini_coefficient(self, attention_weights: np.ndarray) -> float:
        """計算基尼係數"""
        flattened = attention_weights.flatten()
        sorted_values = np.sort(flattened)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _calculate_long_range_attention_ratio(self, attention_weights: np.ndarray, threshold: int = 5) -> float:
        """計算長距離注意力比例"""
        seq_len = attention_weights.shape[0]
        long_range_mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)) > threshold
        return float(np.sum(attention_weights[long_range_mask]) / np.sum(attention_weights))
    
    def _calculate_local_attention_ratio(self, attention_weights: np.ndarray, window: int = 3) -> float:
        """計算局部注意力比例"""
        seq_len = attention_weights.shape[0]
        local_mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)) <= window
        return float(np.sum(attention_weights[local_mask]) / np.sum(attention_weights))
    
    def _analyze_token_attention(self, attention_weights: np.ndarray, tokens: List[str]) -> Dict[str, Any]:
        """分析標記級注意力"""
        token_attention_in = np.sum(attention_weights, axis=0)  # 接收到的注意力
        token_attention_out = np.sum(attention_weights, axis=1)  # 發出的注意力
        
        # 找出最重要的標記
        top_receivers = np.argsort(token_attention_in)[-5:][::-1]
        top_senders = np.argsort(token_attention_out)[-5:][::-1]
        
        return {
            "最受關注標記": [(tokens[i], float(token_attention_in[i])) for i in top_receivers],
            "最關注他人標記": [(tokens[i], float(token_attention_out[i])) for i in top_senders],
            "平均接收注意力": float(np.mean(token_attention_in)),
            "平均發出注意力": float(np.mean(token_attention_out))
        }
    
    def generate_attention_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成注意力分析報告
        
        Args:
            analysis_results: 分析結果
            
        Returns:
            attention_report: 注意力分析報告
        """
        report = {
            "報告生成時間": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "分析結果": analysis_results,
            "可視化類型": [
                "注意力權重熱力圖",
                "多頭注意力比較",
                "文本-注意力對應",
                "注意力流向圖",
                "跨層注意力分析",
                "注意力聚合可視化",
                "注意力綜合儀表板"
            ],
            "保存位置": str(self.plot_dir)
        }
        
        if self.save_plots:
            with open(self.plot_dir / "attention_analysis_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        return report

    def create_attention_quality_assessment_chart(self,
                                                 attention_weights: torch.Tensor,
                                                 analysis_results: Dict[str, Any],
                                                 title: str = "注意力質量評估",
                                                 save_path: Optional[Path] = None) -> go.Figure:
        """
        創建注意力質量評估圖表

        Args:
            attention_weights: 注意力權重張量
            analysis_results: 分析結果
            title: 圖表標題
            save_path: 保存路徑

        Returns:
            plotly圖表對象
        """
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "聚焦度分析", "熵分佈", "異常檢測",
                "頭間相似性", "位置偏差", "質量綜合評分"
            ),
            specs=[
                [{"type": "bar"}, {"type": "histogram"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}, {"type": "indicator"}]
            ]
        )

        # 1. 聚焦度分析
        focus_analysis = analysis_results.get('focus_analysis', {})
        head_focus_scores = focus_analysis.get('head_focus_scores', [])

        if head_focus_scores:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(head_focus_scores))),
                    y=head_focus_scores,
                    name="聚焦分數",
                    marker_color='steelblue'
                ),
                row=1, col=1
            )

        # 2. 熵分佈
        entropy_analysis = analysis_results.get('entropy_analysis', {})
        head_entropies = entropy_analysis.get('head_normalized_entropies', [])

        if head_entropies:
            fig.add_trace(
                go.Histogram(
                    x=head_entropies,
                    nbinsx=10,
                    name="熵分佈",
                    marker_color='orange'
                ),
                row=1, col=2
            )

        # 3. 異常檢測
        anomaly_detection = analysis_results.get('anomaly_detection', {})
        extreme_weights = anomaly_detection.get('extreme_weights', {})

        if extreme_weights:
            fig.add_trace(
                go.Scatter(
                    x=['極端高權重', '極端低權重'],
                    y=[extreme_weights.get('extreme_high_ratio', 0),
                       extreme_weights.get('extreme_low_ratio', 0)],
                    mode='markers',
                    marker=dict(size=15, color='red'),
                    name="異常比例"
                ),
                row=1, col=3
            )

        # 4. 頭間相似性熱力圖
        head_similarity = anomaly_detection.get('head_similarity_anomalies', {})
        similarity_matrix = head_similarity.get('similarity_matrix', [])

        if similarity_matrix:
            fig.add_trace(
                go.Heatmap(
                    z=similarity_matrix,
                    colorscale='RdYlBu',
                    name="頭間相似性"
                ),
                row=2, col=1
            )

        # 5. 位置偏差
        position_bias = anomaly_detection.get('position_bias', {})
        head_position_bias = position_bias.get('head_position_bias_scores', [])

        if head_position_bias:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(head_position_bias))),
                    y=head_position_bias,
                    name="位置偏差",
                    marker_color='green'
                ),
                row=2, col=2
            )

        # 6. 質量綜合評分
        quality_score = self._calculate_attention_quality_score(analysis_results)

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "質量分數"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=3
        )

        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )

        if save_path and self.save_plots:
            # 將 .html 擴展名替換為 .png
            png_path = save_path.replace('.html', '.png') if isinstance(save_path, str) else str(save_path).replace('.html', '.png')
            fig.write_image(png_path)

        return fig

    def _calculate_attention_quality_score(self, analysis_results: Dict[str, Any]) -> float:
        """計算注意力質量綜合評分"""
        score = 0.0
        max_score = 100.0

        # 聚焦度分數 (25分)
        focus_analysis = analysis_results.get('focus_analysis', {})
        focus_score = focus_analysis.get('overall_focus_score', 0)
        score += min(25, focus_score * 100)

        # 熵合理性 (25分)
        entropy_analysis = analysis_results.get('entropy_analysis', {})
        normalized_entropy = entropy_analysis.get('overall_normalized_entropy', 0)
        # 適中的熵值得分更高（0.4-0.7範圍）
        if 0.4 <= normalized_entropy <= 0.7:
            score += 25
        elif 0.2 <= normalized_entropy < 0.4 or 0.7 < normalized_entropy <= 0.8:
            score += 15
        else:
            score += 5

        # 異常度 (25分) - 異常越少得分越高
        anomaly_detection = analysis_results.get('anomaly_detection', {})
        extreme_weights = anomaly_detection.get('extreme_weights', {})
        extreme_ratio = extreme_weights.get('extreme_high_ratio', 0) + extreme_weights.get('extreme_low_ratio', 0)
        score += max(0, 25 - extreme_ratio * 500)  # 假設極端比例通常很小

        # 頭多樣性 (25分)
        head_comparison = analysis_results.get('head_comparison', {})
        diversity_metrics = head_comparison.get('diversity_metrics', {})
        diversity_score = diversity_metrics.get('diversity_score', 0)
        score += diversity_score * 25

        return min(max_score, max(0, score))

    def create_multi_head_comparison_chart(self,
                                         attention_weights: torch.Tensor,
                                         head_names: Optional[List[str]] = None,
                                         title: str = "多頭注意力比較",
                                         save_path: Optional[Path] = None) -> go.Figure:
        """
        創建多頭注意力比較圖

        Args:
            attention_weights: 注意力權重張量 [batch_size, num_heads, seq_len, seq_len]
            head_names: 頭名稱列表
            title: 圖表標題
            save_path: 保存路徑

        Returns:
            plotly圖表對象
        """
        with torch.no_grad():
            weights = attention_weights.detach().cpu()
            batch_size, num_heads, seq_len, _ = weights.shape

            if head_names is None:
                head_names = [f"Head {i+1}" for i in range(num_heads)]

            # 創建子圖網格
            cols = min(4, num_heads)
            rows = (num_heads + cols - 1) // cols

            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=head_names,
                specs=[[{"type": "heatmap"} for _ in range(cols)] for _ in range(rows)]
            )

            # 為每個頭創建熱力圖
            for head_idx in range(num_heads):
                row = head_idx // cols + 1
                col = head_idx % cols + 1

                # 計算該頭的平均注意力權重
                head_weights = weights[:, head_idx, :, :].mean(dim=0)

                fig.add_trace(
                    go.Heatmap(
                        z=head_weights.numpy(),
                        colorscale='Blues',
                        showscale=(head_idx == 0),  # 只在第一個圖顯示色標
                        name=head_names[head_idx]
                    ),
                    row=row, col=col
                )

            fig.update_layout(
                title=title,
                height=200 * rows,
                showlegend=False
            )

            if save_path and self.save_plots:
                # 將 .html 擴展名替換為 .png
                png_path = save_path.replace('.html', '.png') if isinstance(save_path, str) else str(save_path).replace('.html', '.png')
                fig.write_image(png_path)

            return fig

    def create_head_statistics_comparison(self,
                                        attention_weights: torch.Tensor,
                                        analysis_results: Dict[str, Any],
                                        title: str = "注意力頭統計比較",
                                        save_path: Optional[Path] = None) -> go.Figure:
        """
        創建注意力頭統計特性比較圖

        Args:
            attention_weights: 注意力權重張量
            analysis_results: 分析結果
            title: 圖表標題
            save_path: 保存路徑

        Returns:
            plotly圖表對象
        """
        num_heads = attention_weights.shape[1]

        # 創建雷達圖比較各頭特性
        fig = go.Figure()

        # 提取各頭的統計特性
        focus_scores = analysis_results.get('focus_analysis', {}).get('head_focus_scores', [])
        entropies = analysis_results.get('entropy_analysis', {}).get('head_normalized_entropies', [])
        concentration_ratios = analysis_results.get('focus_analysis', {}).get('head_concentration_ratios', [])
        effective_ranges = analysis_results.get('focus_analysis', {}).get('head_effective_ranges', [])

        # 選擇幾個代表性的頭進行比較
        selected_heads = list(range(0, min(6, num_heads), max(1, num_heads // 6)))

        for head_idx in selected_heads:
            # 構建該頭的特徵向量
            features = [
                focus_scores[head_idx] if head_idx < len(focus_scores) else 0,
                1 - entropies[head_idx] if head_idx < len(entropies) else 0,  # 反轉熵值
                concentration_ratios[head_idx] if head_idx < len(concentration_ratios) else 0,
                effective_ranges[head_idx] if head_idx < len(effective_ranges) else 0
            ]

            fig.add_trace(go.Scatterpolar(
                r=features,
                theta=['聚焦度', '確定性', '集中度', '有效範圍'],
                fill='toself',
                name=f'Head {head_idx + 1}'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title
        )

        if save_path and self.save_plots:
            # 將 .html 擴展名替換為 .png
            png_path = save_path.replace('.html', '.png') if isinstance(save_path, str) else str(save_path).replace('.html', '.png')
            fig.write_image(png_path)

        return fig

    def create_attention_token_correspondence_chart(self,
                                                  attention_weights: torch.Tensor,
                                                  tokens: List[str],
                                                  analysis_results: Dict[str, Any],
                                                  title: str = "注意力-詞彙對應關係",
                                                  save_path: Optional[Path] = None) -> go.Figure:
        """
        創建注意力-詞彙對應關係圖

        Args:
            attention_weights: 注意力權重張量
            tokens: 詞彙列表
            analysis_results: 分析結果
            title: 圖表標題
            save_path: 保存路徑

        Returns:
            plotly圖表對象
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "詞彙注意力分數", "重要詞彙分佈",
                "詞彙關聯網絡", "注意力流向"
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )

        token_analysis = analysis_results.get('token_analysis', {})

        # 1. 詞彙注意力分數
        token_scores = token_analysis.get('token_scores', {})
        if token_scores:
            top_tokens = token_analysis.get('token_scores', {}).get('top_tokens_by_attention', [])[:15]

            if top_tokens:
                tokens_list = [item[0] for item in top_tokens]
                scores_list = [item[1]['total_attention'] for item in top_tokens]

                fig.add_trace(
                    go.Bar(
                        x=tokens_list,
                        y=scores_list,
                        name="注意力分數",
                        marker_color='steelblue'
                    ),
                    row=1, col=1
                )

        # 2. 重要詞彙分佈
        important_tokens = token_analysis.get('important_tokens', {})
        important_list = important_tokens.get('important_tokens', [])

        if important_list:
            positions = [token['position'] for token in important_list]
            scores = [token['attention_score'] for token in important_list]
            token_names = [token['token'] for token in important_list]

            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=scores,
                    mode='markers+text',
                    text=token_names,
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=scores,
                        colorscale='Reds',
                        showscale=True
                    ),
                    name="重要詞彙"
                ),
                row=1, col=2
            )

        # 3. 詞彙關聯網絡 (簡化為散點圖)
        token_associations = token_analysis.get('token_associations', {})
        top_associations = token_associations.get('top_associations', [])[:10]

        if top_associations:
            from_positions = [assoc['from_position'] for assoc in top_associations]
            to_positions = [assoc['to_position'] for assoc in top_associations]
            weights_list = [assoc['attention_weight'] for assoc in top_associations]

            fig.add_trace(
                go.Scatter(
                    x=from_positions,
                    y=to_positions,
                    mode='markers',
                    marker=dict(
                        size=[w*50 for w in weights_list],
                        color=weights_list,
                        colorscale='Viridis',
                        opacity=0.7
                    ),
                    name="詞彙關聯"
                ),
                row=2, col=1
            )

        # 4. 注意力流向熱力圖
        with torch.no_grad():
            weights = attention_weights.detach().cpu()
            avg_attention = weights.mean(dim=(0, 1))  # [seq_len, seq_len]

            # 如果序列太長，進行下採樣
            if len(tokens) > 20:
                step = len(tokens) // 20
                sampled_indices = list(range(0, len(tokens), step))
                sampled_attention = avg_attention[sampled_indices][:, sampled_indices]
                sampled_tokens = [tokens[i] for i in sampled_indices]
            else:
                sampled_attention = avg_attention
                sampled_tokens = tokens

            fig.add_trace(
                go.Heatmap(
                    z=sampled_attention.numpy(),
                    x=sampled_tokens,
                    y=sampled_tokens,
                    colorscale='Blues',
                    name="注意力流向"
                ),
                row=2, col=2
            )

        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )

        # 更新x軸標籤旋轉
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=2)
        fig.update_yaxes(tickangle=45, row=2, col=2)

        if save_path and self.save_plots:
            # 將 .html 擴展名替換為 .png
            png_path = save_path.replace('.html', '.png') if isinstance(save_path, str) else str(save_path).replace('.html', '.png')
            fig.write_image(png_path)

        return fig

    def create_attention_anomaly_detection_chart(self,
                                               attention_weights: torch.Tensor,
                                               analysis_results: Dict[str, Any],
                                               title: str = "注意力異常檢測",
                                               save_path: Optional[Path] = None) -> go.Figure:
        """
        創建注意力異常檢測圖表

        Args:
            attention_weights: 注意力權重張量
            analysis_results: 分析結果
            title: 圖表標題
            save_path: 保存路徑

        Returns:
            plotly圖表對象
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "極端權重分佈", "頭間相似性分析",
                "位置偏差檢測", "均勻性偏離度"
            )
        )

        anomaly_detection = analysis_results.get('anomaly_detection', {})

        # 1. 極端權重分佈
        extreme_weights = anomaly_detection.get('extreme_weights', {})
        if extreme_weights:
            categories = ['極端高權重', '正常權重', '極端低權重']
            values = [
                extreme_weights.get('extreme_high_ratio', 0),
                1 - extreme_weights.get('extreme_high_ratio', 0) - extreme_weights.get('extreme_low_ratio', 0),
                extreme_weights.get('extreme_low_ratio', 0)
            ]

            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    marker_color=['red', 'green', 'blue'],
                    name="權重分佈"
                ),
                row=1, col=1
            )

        # 2. 頭間相似性分析
        head_similarity = anomaly_detection.get('head_similarity_anomalies', {})
        similarity_matrix = head_similarity.get('similarity_matrix', [])

        if similarity_matrix:
            fig.add_trace(
                go.Heatmap(
                    z=similarity_matrix,
                    colorscale='RdYlBu_r',
                    name="相似性矩陣"
                ),
                row=1, col=2
            )

        # 3. 位置偏差檢測
        position_bias = anomaly_detection.get('position_bias', {})
        head_position_bias = position_bias.get('head_position_bias_scores', [])

        if head_position_bias:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(head_position_bias))),
                    y=head_position_bias,
                    mode='lines+markers',
                    name="位置偏差",
                    line=dict(color='orange')
                ),
                row=2, col=1
            )

        # 4. 均勻性偏離度
        uniformity_analysis = anomaly_detection.get('uniformity_analysis', {})
        head_uniformity_scores = uniformity_analysis.get('head_uniformity_scores', [])

        if head_uniformity_scores:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(head_uniformity_scores))),
                    y=head_uniformity_scores,
                    name="KL散度",
                    marker_color='purple'
                ),
                row=2, col=2
            )

        fig.update_layout(
            title=title,
            height=700,
            showlegend=True
        )

        if save_path and self.save_plots:
            # 將 .html 擴展名替換為 .png
            png_path = save_path.replace('.html', '.png') if isinstance(save_path, str) else str(save_path).replace('.html', '.png')
            fig.write_image(png_path)

        return fig