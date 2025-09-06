# 實驗報告生成器模組
"""
實驗報告生成器實現

提供以下報告生成功能：
- 自動化表格生成: 自動生成實驗結果對比表
- 圖表組合器: 將多個圖表組合成綜合報告
- PDF/HTML 報告輸出: 生成專業的實驗報告文檔
- 論文質量圖表: 生成適合學術論文的高質量圖表
- 實驗摘要生成: 自動生成實驗結果摘要
- 統計表格製作: 生成標準化的統計分析表格
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
from datetime import datetime
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# HTML模板和CSS樣式
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        {css_styles}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""

CSS_STYLES = """
body {
    font-family: 'Microsoft JhengHei', 'SimHei', Arial, sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 0;
    background-color: #f8f9fa;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: white;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}

.header {
    text-align: center;
    border-bottom: 3px solid #007bff;
    padding-bottom: 20px;
    margin-bottom: 30px;
}

.header h1 {
    color: #007bff;
    margin-bottom: 10px;
}

.section {
    margin-bottom: 40px;
}

.section h2 {
    color: #343a40;
    border-left: 4px solid #007bff;
    padding-left: 15px;
    margin-bottom: 20px;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
}

.metric-value {
    font-size: 2em;
    font-weight: bold;
    margin-bottom: 5px;
}

.metric-label {
    font-size: 0.9em;
    opacity: 0.9;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    background-color: white;
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

th {
    background-color: #007bff;
    color: white;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #f8f9fa;
}

tr:hover {
    background-color: #e9ecef;
}

.best-score {
    background-color: #d4edda !important;
    font-weight: bold;
    color: #155724;
}

.chart-container {
    margin: 20px 0;
    text-align: center;
}

.chart-container img {
    max-width: 100%;
    height: auto;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border-radius: 8px;
}

.summary-box {
    background-color: #e9ecef;
    border-left: 5px solid #007bff;
    padding: 20px;
    margin: 20px 0;
    border-radius: 0 8px 8px 0;
}

.footer {
    text-align: center;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
    color: #6c757d;
}
"""


class ReportGenerator:
    """實驗報告生成器"""
    
    def __init__(self, save_dir: str = "experiment_reports"):
        """
        初始化報告生成器
        
        Args:
            save_dir: 報告保存目錄
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 報告數據存儲
        self.report_data = {}
        
        # 圖表存儲
        self.figures = {}
        
        # 表格存儲
        self.tables = {}
    
    def add_experiment_results(self, experiment_name: str, results: Dict[str, Any]) -> None:
        """
        添加實驗結果
        
        Args:
            experiment_name: 實驗名稱
            results: 實驗結果數據
        """
        self.report_data[experiment_name] = results
        print(f"已添加實驗結果: {experiment_name}")
    
    def generate_comparison_table(self, experiments: List[str], 
                                metrics: List[str],
                                title: str = "實驗結果比較表") -> pd.DataFrame:
        """
        生成實驗結果比較表
        
        Args:
            experiments: 實驗名稱列表
            metrics: 指標列表
            title: 表格標題
            
        Returns:
            comparison_df: 比較結果DataFrame
        """
        print(f"生成比較表格: {title}")
        
        # 準備數據
        data = []
        for exp_name in experiments:
            if exp_name in self.report_data:
                row = {"實驗方法": exp_name}
                exp_results = self.report_data[exp_name]
                
                for metric in metrics:
                    # 尋找指標值（支援嵌套字典）
                    value = self._find_metric_value(exp_results, metric)
                    if value is not None:
                        if isinstance(value, float):
                            row[metric] = f"{value:.4f}"
                        else:
                            row[metric] = str(value)
                    else:
                        row[metric] = "N/A"
                
                data.append(row)
        
        # 創建DataFrame
        df = pd.DataFrame(data)
        
        # 保存表格
        self.tables[title] = df
        
        # 保存到CSV
        df.to_csv(self.save_dir / f"{title.replace(' ', '_')}.csv", 
                 index=False, encoding='utf-8-sig')
        
        return df
    
    def _find_metric_value(self, data: Dict[str, Any], metric: str) -> Optional[Union[float, str]]:
        """在嵌套字典中尋找指標值"""
        # 直接匹配
        if metric in data:
            return data[metric]
        
        # 遞歸搜索
        for key, value in data.items():
            if isinstance(value, dict):
                result = self._find_metric_value(value, metric)
                if result is not None:
                    return result
        
        # 模糊匹配（小寫）
        for key, value in data.items():
            if metric.lower() in key.lower():
                if isinstance(value, (int, float)):
                    return value
                elif isinstance(value, str):
                    try:
                        return float(value)
                    except:
                        return value
        
        return None
    
    def create_performance_summary_table(self, domain_results: Dict[str, Dict[str, float]],
                                       title: str = "跨領域性能摘要表") -> pd.DataFrame:
        """
        創建性能摘要表
        
        Args:
            domain_results: 領域結果 {domain: {metric: value}}
            title: 表格標題
            
        Returns:
            summary_df: 摘要DataFrame
        """
        print(f"生成性能摘要表: {title}")
        
        # 準備數據
        df = pd.DataFrame(domain_results).T
        
        # 添加平均值行
        if len(df) > 1:
            avg_row = df.mean()
            avg_row.name = "平均值"
            df = pd.concat([df, avg_row.to_frame().T])
        
        # 格式化數值
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].apply(lambda x: f"{x:.4f}")
        
        # 保存表格
        self.tables[title] = df
        
        # 保存到CSV
        df.to_csv(self.save_dir / f"{title.replace(' ', '_')}.csv", 
                 encoding='utf-8-sig')
        
        return df
    
    def generate_statistical_significance_table(self, comparison_results: Dict[str, Any],
                                              title: str = "統計顯著性檢驗表") -> pd.DataFrame:
        """
        生成統計顯著性檢驗表
        
        Args:
            comparison_results: 比較結果
            title: 表格標題
            
        Returns:
            significance_df: 顯著性檢驗DataFrame
        """
        print(f"生成統計顯著性檢驗表: {title}")
        
        data = []
        
        # 解析比較結果
        pairwise_comparisons = comparison_results.get("pairwise_comparisons", {})
        
        for pair_key, pair_result in pairwise_comparisons.items():
            model1, model2 = pair_key.split("_vs_")
            
            row = {
                "模型1": model1,
                "模型2": model2,
                "檢驗類型": pair_result.get("test_type", "N/A"),
                "檢驗統計量": f"{pair_result.get('test_statistic', 0):.4f}",
                "p值": f"{pair_result.get('p_value', 1):.4f}",
                "效果量(Cohen's d)": f"{pair_result.get('cohens_d', 0):.4f}",
                "顯著性": "是" if pair_result.get('significant', False) else "否",
                "效果大小": pair_result.get('effect_size_interpretation', 'N/A')
            }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 保存表格
        self.tables[title] = df
        
        # 保存到CSV
        df.to_csv(self.save_dir / f"{title.replace(' ', '_')}.csv", 
                 index=False, encoding='utf-8-sig')
        
        return df
    
    def create_attention_analysis_table(self, attention_results: Dict[str, Any],
                                      title: str = "注意力機制分析表") -> pd.DataFrame:
        """
        創建注意力機制分析表
        
        Args:
            attention_results: 注意力分析結果
            title: 表格標題
            
        Returns:
            attention_df: 注意力分析DataFrame
        """
        print(f"生成注意力機制分析表: {title}")
        
        data = []
        
        # 解析注意力結果
        for mechanism_name, analysis in attention_results.items():
            if isinstance(analysis, dict):
                row = {"注意力機制": mechanism_name}
                
                # 基本統計
                basic_stats = analysis.get("基本統計", {})
                for stat_name, stat_value in basic_stats.items():
                    if isinstance(stat_value, (int, float)):
                        row[stat_name] = f"{stat_value:.4f}"
                
                # 注意力分佈
                attention_dist = analysis.get("注意力分佈", {})
                for dist_name, dist_value in attention_dist.items():
                    if isinstance(dist_value, (int, float)):
                        row[dist_name] = f"{dist_value:.4f}"
                
                data.append(row)
        
        df = pd.DataFrame(data)
        
        # 保存表格
        self.tables[title] = df
        
        # 保存到CSV
        df.to_csv(self.save_dir / f"{title.replace(' ', '_')}.csv", 
                 index=False, encoding='utf-8-sig')
        
        return df
    
    def add_figure(self, figure_name: str, figure_path: str, caption: str = "") -> None:
        """
        添加圖表到報告
        
        Args:
            figure_name: 圖表名稱
            figure_path: 圖表文件路径
            caption: 圖表說明
        """
        self.figures[figure_name] = {
            "path": figure_path,
            "caption": caption
        }
        print(f"已添加圖表: {figure_name}")
    
    def create_academic_figure(self, data: Dict[str, Any], 
                             figure_type: str = "comparison_bar",
                             title: str = "學術圖表",
                             save_name: str = None) -> str:
        """
        創建學術質量的圖表
        
        Args:
            data: 圖表數據
            figure_type: 圖表類型
            title: 圖表標題
            save_name: 保存名稱
            
        Returns:
            figure_path: 圖表文件路径
        """
        print(f"創建學術圖表: {title}")
        
        # 設置學術風格
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if figure_type == "comparison_bar":
            # 比較條形圖
            methods = list(data.keys())
            values = list(data.values())
            
            bars = ax.bar(methods, values, alpha=0.8, color='steelblue', edgecolor='navy')
            
            # 添加數值標籤
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Performance Score')
            ax.set_title(title, pad=20)
            
        elif figure_type == "line_plot":
            # 線圖
            for label, line_data in data.items():
                ax.plot(line_data['x'], line_data['y'], marker='o', 
                       linewidth=2, markersize=6, label=label)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(title, pad=20)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        elif figure_type == "heatmap":
            # 熱力圖
            matrix_data = data['matrix']
            labels = data.get('labels', None)
            
            im = ax.imshow(matrix_data, cmap='Blues', aspect='auto')
            
            if labels:
                ax.set_xticks(np.arange(len(labels)))
                ax.set_yticks(np.arange(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
            
            # 添加數值
            for i in range(len(matrix_data)):
                for j in range(len(matrix_data[0])):
                    ax.text(j, i, f'{matrix_data[i][j]:.3f}',
                           ha="center", va="center", color="white" if matrix_data[i][j] > 0.5 else "black")
            
            ax.set_title(title, pad=20)
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        # 保存圖表
        if save_name is None:
            save_name = title.replace(" ", "_").replace("/", "_")
        
        figure_path = self.save_dir / f"{save_name}.png"
        plt.savefig(figure_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(figure_path)
    
    def generate_html_report(self, report_title: str = "跨領域情感分析實驗報告",
                           author: str = "研究者",
                           include_interactive: bool = True) -> str:
        """
        生成HTML報告
        
        Args:
            report_title: 報告標題
            author: 作者
            include_interactive: 是否包含交互式圖表
            
        Returns:
            report_path: 報告文件路径
        """
        print("生成HTML格式實驗報告")
        
        # 生成報告內容
        content_parts = []
        
        # 報告頭部
        header_html = f"""
        <div class="header">
            <h1>{report_title}</h1>
            <p><strong>作者:</strong> {author}</p>
            <p><strong>生成時間:</strong> {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        </div>
        """
        content_parts.append(header_html)
        
        # 實驗摘要
        if self.report_data:
            summary_html = self._generate_summary_section()
            content_parts.append(summary_html)
        
        # 關鍵指標
        metrics_html = self._generate_metrics_section()
        content_parts.append(metrics_html)
        
        # 實驗結果表格
        if self.tables:
            tables_html = self._generate_tables_section()
            content_parts.append(tables_html)
        
        # 圖表展示
        if self.figures:
            figures_html = self._generate_figures_section()
            content_parts.append(figures_html)
        
        # 報告尾部
        footer_html = f"""
        <div class="footer">
            <p>本報告由跨領域情感分析系統自動生成</p>
            <p>© 2026 論文實驗報告</p>
        </div>
        """
        content_parts.append(footer_html)
        
        # 組合完整HTML
        full_content = "\n".join(content_parts)
        html_content = HTML_TEMPLATE.format(
            title=report_title,
            css_styles=CSS_STYLES,
            content=full_content
        )
        
        # 保存HTML文件
        report_path = self.save_dir / f"{report_title.replace(' ', '_')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML報告已保存: {report_path}")
        return str(report_path)
    
    def _generate_summary_section(self) -> str:
        """生成摘要部分"""
        html = '<div class="section"><h2>實驗摘要</h2>'
        
        # 計算總體統計
        total_experiments = len(self.report_data)
        
        summary_html = f"""
        <div class="summary-box">
            <h3>實驗概況</h3>
            <p><strong>實驗總數:</strong> {total_experiments}</p>
            <p><strong>評估指標:</strong> 準確率、F1分數、精確率、召回率</p>
            <p><strong>跨領域設定:</strong> 多源領域到目標領域的遷移學習</p>
        </div>
        """
        
        html += summary_html + '</div>'
        return html
    
    def _generate_metrics_section(self) -> str:
        """生成關鍵指標部分"""
        html = '<div class="section"><h2>關鍵性能指標</h2>'
        
        # 計算關鍵指標
        metrics_data = []
        
        if self.report_data:
            # 計算平均準確率
            accuracies = []
            f1_scores = []
            
            for exp_name, results in self.report_data.items():
                acc = self._find_metric_value(results, 'accuracy')
                f1 = self._find_metric_value(results, 'f1')
                
                if acc is not None:
                    try:
                        accuracies.append(float(acc))
                    except:
                        pass
                
                if f1 is not None:
                    try:
                        f1_scores.append(float(f1))
                    except:
                        pass
            
            if accuracies:
                avg_acc = np.mean(accuracies)
                metrics_data.append(("平均準確率", f"{avg_acc:.1%}"))
            
            if f1_scores:
                avg_f1 = np.mean(f1_scores)
                metrics_data.append(("平均F1分數", f"{avg_f1:.3f}"))
            
            metrics_data.append(("實驗方法數量", str(len(self.report_data))))
            metrics_data.append(("生成圖表數量", str(len(self.figures))))
        
        # 生成指標卡片
        html += '<div class="metrics-grid">'
        for label, value in metrics_data:
            html += f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """
        html += '</div></div>'
        
        return html
    
    def _generate_tables_section(self) -> str:
        """生成表格部分"""
        html = '<div class="section"><h2>實驗結果表格</h2>'
        
        for table_name, df in self.tables.items():
            html += f'<h3>{table_name}</h3>'
            
            # 轉換DataFrame為HTML表格
            table_html = df.to_html(index=False, escape=False, classes='table table-striped')
            
            # 高亮最佳分數
            if '準確率' in table_html or 'accuracy' in table_html.lower() or 'f1' in table_html.lower():
                table_html = self._highlight_best_scores(table_html)
            
            html += table_html
        
        html += '</div>'
        return html
    
    def _generate_figures_section(self) -> str:
        """生成圖表部分"""
        html = '<div class="section"><h2>實驗圖表</h2>'
        
        for figure_name, figure_info in self.figures.items():
            figure_path = figure_info['path']
            caption = figure_info['caption']
            
            # 轉換圖片為base64編碼
            try:
                with open(figure_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    img_src = f"data:image/png;base64,{img_data}"
            except:
                img_src = figure_path  # 如果編碼失敗，使用原路径
            
            html += f"""
            <div class="chart-container">
                <h3>{figure_name}</h3>
                <img src="{img_src}" alt="{figure_name}">
                {f'<p>{caption}</p>' if caption else ''}
            </div>
            """
        
        html += '</div>'
        return html
    
    def _highlight_best_scores(self, table_html: str) -> str:
        """高亮表格中的最佳分數"""
        # 這裡可以實現更複雜的最佳分數高亮邏輯
        # 目前返回原始HTML
        return table_html
    
    def generate_pdf_report(self, report_title: str = "跨領域情感分析實驗報告",
                          author: str = "研究者") -> str:
        """
        生成PDF報告
        
        Args:
            report_title: 報告標題
            author: 作者
            
        Returns:
            report_path: PDF文件路径
        """
        print("生成PDF格式實驗報告")
        
        report_path = self.save_dir / f"{report_title.replace(' ', '_')}.pdf"
        
        with PdfPages(report_path) as pdf:
            # 創建封面頁
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            # 標題
            ax.text(0.5, 0.7, report_title, fontsize=24, fontweight='bold',
                   ha='center', va='center', transform=ax.transAxes)
            
            # 作者和日期
            ax.text(0.5, 0.6, f"作者: {author}", fontsize=16,
                   ha='center', va='center', transform=ax.transAxes)
            ax.text(0.5, 0.55, f"生成時間: {datetime.now().strftime('%Y年%m月%d日')}",
                   fontsize=14, ha='center', va='center', transform=ax.transAxes)
            
            # 添加一些裝飾
            ax.add_patch(Rectangle((0.1, 0.4), 0.8, 0.005, transform=ax.transAxes, 
                                 facecolor='navy'))
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 添加表格頁面
            for table_name, df in self.tables.items():
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                
                # 表格標題
                ax.text(0.5, 0.95, table_name, fontsize=16, fontweight='bold',
                       ha='center', va='top', transform=ax.transAxes)
                
                # 創建表格
                table_data = []
                headers = df.columns.tolist()
                for _, row in df.iterrows():
                    table_data.append(row.tolist())
                
                table = ax.table(cellText=table_data, colLabels=headers,
                               cellLoc='center', loc='center',
                               bbox=[0.1, 0.1, 0.8, 0.8])
                
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                
                # 設置表頭樣式
                for i in range(len(headers)):
                    table[(0, i)].set_facecolor('#4472C4')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # 添加圖表頁面
            for figure_name, figure_info in self.figures.items():
                if Path(figure_info['path']).exists():
                    try:
                        # 讀取並添加現有圖片
                        img = plt.imread(figure_info['path'])
                        fig, ax = plt.subplots(figsize=(11, 8.5))
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title(figure_name, fontsize=16, fontweight='bold', pad=20)
                        
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close()
                    except:
                        print(f"警告：無法讀取圖片 {figure_info['path']}")
        
        print(f"PDF報告已保存: {report_path}")
        return str(report_path)
    
    def create_experiment_summary(self) -> Dict[str, Any]:
        """
        創建實驗總結
        
        Returns:
            experiment_summary: 實驗總結數據
        """
        print("創建實驗總結")
        
        summary = {
            "生成時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "實驗數量": len(self.report_data),
            "圖表數量": len(self.figures),
            "表格數量": len(self.tables),
            "實驗列表": list(self.report_data.keys()),
            "性能統計": {},
            "最佳方法": {}
        }
        
        # 計算性能統計
        if self.report_data:
            all_accuracies = []
            all_f1_scores = []
            method_performance = {}
            
            for exp_name, results in self.report_data.items():
                acc = self._find_metric_value(results, 'accuracy')
                f1 = self._find_metric_value(results, 'f1')
                
                performance = {}
                if acc is not None:
                    try:
                        acc_val = float(acc)
                        all_accuracies.append(acc_val)
                        performance['accuracy'] = acc_val
                    except:
                        pass
                
                if f1 is not None:
                    try:
                        f1_val = float(f1)
                        all_f1_scores.append(f1_val)
                        performance['f1'] = f1_val
                    except:
                        pass
                
                if performance:
                    method_performance[exp_name] = performance
            
            if all_accuracies:
                summary["性能統計"]["平均準確率"] = np.mean(all_accuracies)
                summary["性能統計"]["準確率標準差"] = np.std(all_accuracies)
                summary["性能統計"]["最高準確率"] = np.max(all_accuracies)
                
                # 找到最佳準確率方法
                best_acc_method = max(method_performance.items(), 
                                    key=lambda x: x[1].get('accuracy', 0))
                summary["最佳方法"]["準確率"] = {
                    "方法": best_acc_method[0],
                    "分數": best_acc_method[1]['accuracy']
                }
            
            if all_f1_scores:
                summary["性能統計"]["平均F1分數"] = np.mean(all_f1_scores)
                summary["性能統計"]["F1分數標準差"] = np.std(all_f1_scores)
                summary["性能統計"]["最高F1分數"] = np.max(all_f1_scores)
                
                # 找到最佳F1分數方法
                best_f1_method = max(method_performance.items(),
                                   key=lambda x: x[1].get('f1', 0))
                summary["最佳方法"]["F1分數"] = {
                    "方法": best_f1_method[0],
                    "分數": best_f1_method[1]['f1']
                }
        
        # 保存總結
        summary_path = self.save_dir / "experiment_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        return summary
    
    def export_all_results(self) -> Dict[str, str]:
        """
        匯出所有結果
        
        Returns:
            export_paths: 匯出文件路径字典
        """
        print("匯出所有實驗結果")
        
        export_paths = {}
        
        # 匯出原始數據
        if self.report_data:
            data_path = self.save_dir / "raw_experiment_data.json"
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(self.report_data, f, ensure_ascii=False, indent=2, default=str)
            export_paths["原始數據"] = str(data_path)
        
        # 匯出所有表格
        if self.tables:
            tables_dir = self.save_dir / "tables"
            tables_dir.mkdir(exist_ok=True)
            
            for table_name, df in self.tables.items():
                table_path = tables_dir / f"{table_name.replace(' ', '_')}.xlsx"
                df.to_excel(table_path, index=False, engine='openpyxl')
                export_paths[f"表格_{table_name}"] = str(table_path)
        
        # 創建實驗總結
        summary = self.create_experiment_summary()
        export_paths["實驗總結"] = str(self.save_dir / "experiment_summary.json")
        
        print(f"所有結果已匯出到: {self.save_dir}")
        return export_paths