# 結果可視化器模組
"""
結果可視化器實現

提供以下可視化功能：
- 跨領域準確率比較表: 展示不同領域對的準確率對比
- 注意力機制效能對比圖: 比較不同注意力機制的性能
- 方面對齊分數熱力圖: 可視化抽象方面對齊效果
- 錯誤率趨勢圖: 展示模型訓練過程中的錯誤率變化
- 綜合性能儀表板: 整合多種指標的儀表板
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class ResultVisualizer:
    """結果可視化器"""
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "result_plots", 
                 style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        初始化結果可視化器
        
        Args:
            save_plots: 是否保存圖表
            plot_dir: 圖表保存目錄
            style: matplotlib樣式
            figsize: 預設圖表大小
        """
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        self.style = style
        self.figsize = figsize
        
        if self.save_plots:
            self.plot_dir.mkdir(exist_ok=True)
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 設置風格
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('default')
        
        # 顏色配置
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#E74C3C', 
            'success': '#27AE60',
            'warning': '#F39C12',
            'info': '#8E44AD',
            'neutral': '#95A5A6'
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
        
        # 方面名稱映射
        self.aspect_names = {
            'quality': '品質',
            'price': '價格', 
            'service': '服務',
            'ambiance': '環境氛圍',
            'convenience': '便利性'
        }
        
    def plot_cross_domain_accuracy_comparison(self, results: Dict[str, Dict[str, float]], 
                                            title: str = "跨領域準確率比較") -> None:
        """
        繪製跨領域準確率比較表
        
        Args:
            results: 跨領域結果 {source_domain: {target_domain: accuracy}}
            title: 圖表標題
        """
        print("生成跨領域準確率比較圖")
        
        # 準備數據
        domains = list(results.keys())
        matrix_data = np.zeros((len(domains), len(domains)))
        
        for i, source in enumerate(domains):
            for j, target in enumerate(domains):
                if target in results[source]:
                    matrix_data[i, j] = results[source][target]
                else:
                    matrix_data[i, j] = np.nan
        
        # 轉換領域名稱
        display_domains = [self.domain_names.get(d, d) for d in domains]
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 繪製熱力圖
        im = ax.imshow(matrix_data, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
        
        # 設置刻度
        ax.set_xticks(np.arange(len(domains)))
        ax.set_yticks(np.arange(len(domains)))
        ax.set_xticklabels(display_domains)
        ax.set_yticklabels(display_domains)
        
        # 添加數值標籤
        for i in range(len(domains)):
            for j in range(len(domains)):
                if not np.isnan(matrix_data[i, j]):
                    text = ax.text(j, i, f'{matrix_data[i, j]:.3f}',
                                 ha="center", va="center", 
                                 color="white" if matrix_data[i, j] < 0.5 else "black",
                                 fontweight='bold')
        
        # 設置標籤和標題
        ax.set_xlabel('目標領域', fontsize=12, fontweight='bold')
        ax.set_ylabel('源領域', fontsize=12, fontweight='bold') 
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # 添加顏色條
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('準確率', fontsize=12)
        
        # 旋轉x軸標籤
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attention_mechanism_comparison(self, attention_results: Dict[str, Dict[str, float]],
                                          metrics: List[str] = ['accuracy', 'f1', 'precision', 'recall'],
                                          title: str = "注意力機制效能對比") -> None:
        """
        繪製注意力機制效能對比圖
        
        Args:
            attention_results: 注意力機制結果 {mechanism_name: {metric: score}}
            metrics: 要比較的指標
            title: 圖表標題
        """
        print("生成注意力機制效能對比圖")
        
        # 準備數據
        mechanisms = list(attention_results.keys())
        n_metrics = len(metrics)
        n_mechanisms = len(mechanisms)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 4, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            scores = [attention_results[mech].get(metric, 0) for mech in mechanisms]
            
            # 創建條形圖
            bars = axes[i].bar(range(n_mechanisms), scores, 
                             color=[self.colors['primary'], self.colors['secondary'], 
                                   self.colors['success'], self.colors['warning']][:n_mechanisms],
                             alpha=0.8)
            
            # 添加數值標籤
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{scores[j]:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 設置標籤和標題
            axes[i].set_xticks(range(n_mechanisms))
            axes[i].set_xticklabels(mechanisms, rotation=45, ha='right')
            axes[i].set_ylabel(metric.upper(), fontsize=12)
            axes[i].set_title(f'{metric.upper()} 比較', fontsize=12, fontweight='bold')
            axes[i].set_ylim(0, 1.1)
            
            # 添加網格
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_aspect_alignment_heatmap(self, alignment_scores: Dict[str, Dict[str, float]],
                                    title: str = "抽象方面對齊分數熱力圖") -> None:
        """
        繪製方面對齊分數熱力圖
        
        Args:
            alignment_scores: 對齊分數 {source_aspect: {target_aspect: score}}
            title: 圖表標題
        """
        print("生成抽象方面對齊分數熱力圖")
        
        # 準備數據
        aspects = list(alignment_scores.keys())
        matrix_data = np.zeros((len(aspects), len(aspects)))
        
        for i, source_aspect in enumerate(aspects):
            for j, target_aspect in enumerate(aspects):
                score = alignment_scores[source_aspect].get(target_aspect, 0)
                matrix_data[i, j] = score
        
        # 轉換方面名稱
        display_aspects = [self.aspect_names.get(asp, asp) for asp in aspects]
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 繪製熱力圖
        sns.heatmap(matrix_data, 
                   annot=True, 
                   fmt='.3f',
                   xticklabels=display_aspects,
                   yticklabels=display_aspects,
                   cmap='viridis',
                   cbar_kws={'label': '對齊分數'},
                   ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('目標方面', fontsize=12, fontweight='bold')
        ax.set_ylabel('源方面', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_error_rate_trends(self, training_history: Dict[str, List[float]],
                              domains: List[str] = None,
                              title: str = "錯誤率趨勢圖") -> None:
        """
        繪製錯誤率趨勢圖
        
        Args:
            training_history: 訓練歷史 {domain/model: [error_rates_by_epoch]}
            domains: 領域列表
            title: 圖表標題
        """
        print("生成錯誤率趨勢圖")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['success'], self.colors['warning'], 
                 self.colors['info'], self.colors['neutral']]
        
        for i, (label, error_rates) in enumerate(training_history.items()):
            epochs = range(1, len(error_rates) + 1)
            display_label = self.domain_names.get(label, label)
            
            ax.plot(epochs, error_rates, 
                   marker='o', 
                   linewidth=2, 
                   markersize=4,
                   color=colors[i % len(colors)],
                   label=display_label)
        
        ax.set_xlabel('訓練輪數 (Epoch)', fontsize=12, fontweight='bold')
        ax.set_ylabel('錯誤率', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_performance_dashboard(self, comprehensive_results: Dict[str, Any],
                                   title: str = "模型性能綜合儀表板") -> None:
        """
        創建綜合性能儀表板
        
        Args:
            comprehensive_results: 綜合結果數據
            title: 圖表標題
        """
        print("生成綜合性能儀表板")
        
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('領域準確率分佈', '注意力機制效能', '錯誤類型分析', '訓練收斂情況'),
            specs=[[{"type": "bar"}, {"type": "radar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # 1. 領域準確率分佈 (條形圖)
        if 'domain_accuracy' in comprehensive_results:
            domain_acc = comprehensive_results['domain_accuracy']
            domains = list(domain_acc.keys())
            accuracies = list(domain_acc.values())
            display_domains = [self.domain_names.get(d, d) for d in domains]
            
            fig.add_trace(
                go.Bar(x=display_domains, y=accuracies, name='準確率',
                      marker_color=self.colors['primary']),
                row=1, col=1
            )
        
        # 2. 注意力機制效能 (雷達圖)
        if 'attention_performance' in comprehensive_results:
            att_perf = comprehensive_results['attention_performance']
            metrics = list(att_perf.keys())
            values = list(att_perf.values())
            
            fig.add_trace(
                go.Scatterpolar(r=values, theta=metrics, fill='toself',
                              name='注意力性能', line_color=self.colors['secondary']),
                row=1, col=2
            )
        
        # 3. 錯誤類型分析 (餅圖)
        if 'error_types' in comprehensive_results:
            error_types = comprehensive_results['error_types']
            labels = list(error_types.keys())
            values = list(error_types.values())
            
            fig.add_trace(
                go.Pie(labels=labels, values=values, name="錯誤類型"),
                row=2, col=1
            )
        
        # 4. 訓練收斂情況 (散點圖)
        if 'training_convergence' in comprehensive_results:
            convergence = comprehensive_results['training_convergence']
            epochs = convergence['epochs']
            loss = convergence['loss']
            
            fig.add_trace(
                go.Scatter(x=epochs, y=loss, mode='lines+markers',
                          name='訓練損失', line_color=self.colors['success']),
                row=2, col=2
            )
        
        # 更新佈局
        fig.update_layout(
            title_text=title,
            title_x=0.5,
            title_font_size=16,
            showlegend=True,
            height=800
        )

        if self.save_plots:
            fig.write_image(str(self.plot_dir / f'{title.replace(" ", "_")}.png'))

        fig.show()
    
    def plot_model_comparison_matrix(self, model_results: Dict[str, Dict[str, float]],
                                   metrics: List[str] = ['accuracy', 'f1', 'precision', 'recall'],
                                   title: str = "模型對比矩陣") -> None:
        """
        繪製模型對比矩陣
        
        Args:
            model_results: 模型結果 {model_name: {metric: score}}
            metrics: 對比指標
            title: 圖表標題
        """
        print("生成模型對比矩陣")
        
        # 準備數據
        models = list(model_results.keys())
        data = []
        
        for model in models:
            row = [model]
            for metric in metrics:
                row.append(model_results[model].get(metric, 0))
            data.append(row)
        
        # 創建DataFrame
        columns = ['模型'] + [metric.upper() for metric in metrics]
        df = pd.DataFrame(data, columns=columns)
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=(12, len(models) * 0.8 + 2))
        
        # 隱藏軸
        ax.axis('tight')
        ax.axis('off')
        
        # 創建表格
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # 設置表格樣式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # 設置標題行樣式
        for i in range(len(columns)):
            table[(0, i)].set_facecolor(self.colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 設置數據行樣式
        for i in range(1, len(models) + 1):
            for j in range(len(columns)):
                if j == 0:  # 模型名稱列
                    table[(i, j)].set_facecolor('#f0f0f0')
                    table[(i, j)].set_text_props(weight='bold')
                else:  # 數值列
                    value = float(df.iloc[i-1, j])
                    if value >= 0.8:
                        table[(i, j)].set_facecolor('#d4edda')  # 綠色
                    elif value >= 0.7:
                        table[(i, j)].set_facecolor('#fff3cd')  # 黃色
                    else:
                        table[(i, j)].set_facecolor('#f8d7da')  # 紅色
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                              top_n: int = 15,
                              title: str = "特徵重要性分析") -> None:
        """
        繪製特徵重要性圖
        
        Args:
            feature_importance: 特徵重要性 {feature_name: importance_score}
            top_n: 顯示前N個最重要特徵
            title: 圖表標題
        """
        print("生成特徵重要性分析圖")
        
        # 排序並取前N個
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importance = zip(*sorted_features)
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
        
        # 創建水平條形圖
        bars = ax.barh(range(len(features)), importance, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(features))))
        
        # 設置標籤
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('重要性分數', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # 添加數值標籤
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance[i]:.3f}', ha='left', va='center', fontweight='bold')
        
        # 反轉y軸使最重要的特徵在頂部
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_learning_curves(self, learning_data: Dict[str, Dict[str, List[float]]],
                           title: str = "學習曲線分析") -> None:
        """
        繪製學習曲線
        
        Args:
            learning_data: 學習數據 {metric: {'train': [...], 'val': [...]}}
            title: 圖表標題
        """
        print("生成學習曲線分析圖")
        
        n_metrics = len(learning_data)
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, (metric, data) in enumerate(learning_data.items()):
            train_data = data.get('train', [])
            val_data = data.get('val', [])
            epochs = range(1, len(train_data) + 1)
            
            axes[i].plot(epochs, train_data, label='訓練', 
                        color=self.colors['primary'], linewidth=2, marker='o', markersize=3)
            if val_data:
                axes[i].plot(epochs, val_data, label='驗證', 
                           color=self.colors['secondary'], linewidth=2, marker='s', markersize=3)
            
            axes[i].set_xlabel('訓練輪數', fontsize=10)
            axes[i].set_ylabel(metric.upper(), fontsize=10)
            axes[i].set_title(f'{metric.upper()} 學習曲線', fontsize=12, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plot_dir / f'{title.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results_summary(self, results: Dict[str, Any], filename: str = "results_summary.json") -> None:
        """
        保存結果摘要
        
        Args:
            results: 結果數據
            filename: 保存文件名
        """
        if self.save_plots:
            with open(self.plot_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"結果摘要已保存至: {self.plot_dir / filename}")
    
    def create_interactive_dashboard(self, all_results: Dict[str, Any]) -> None:
        """
        創建交互式儀表板
        
        Args:
            all_results: 所有結果數據
        """
        print("創建交互式綜合儀表板")
        
        # 創建多頁面儀表板
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                '整體性能概覽', '跨領域準確率', '注意力機制對比',
                '錯誤分析', '特徵重要性', '學習曲線',
                '混淆矩陣', '方面對齊', '模型比較'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "heatmap"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "table"}]
            ]
        )
        
        # 可以根據實際數據添加各種圖表...
        
        # 更新佈局
        fig.update_layout(
            title_text="跨領域情感分析系統 - 綜合性能儀表板",
            title_x=0.5,
            title_font_size=20,
            showlegend=True,
            height=1200
        )
        
        if self.save_plots:
            fig.write_image(str(self.plot_dir / "dashboard.png"))
            print(f"儀表板圖表已保存至: {self.plot_dir / 'dashboard.png'}")
        
        fig.show()
    
    def generate_visualization_report(self) -> Dict[str, Any]:
        """
        生成可視化報告
        
        Returns:
            visualization_report: 可視化報告
        """
        report = {
            "報告生成時間": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "可視化類型": [
                "跨領域準確率比較",
                "注意力機制效能對比", 
                "抽象方面對齊熱力圖",
                "錯誤率趋勢分析",
                "綜合性能儀表板",
                "模型對比矩陣",
                "特徵重要性分析",
                "學習曲線",
                "交互式儀表板"
            ],
            "保存位置": str(self.plot_dir),
            "圖表統計": {
                "靜態圖表": "PNG格式，300 DPI高解析度",
                "數據摘要": "JSON格式，包含完整數值結果"
            }
        }
        
        if self.save_plots:
            with open(self.plot_dir / "visualization_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report