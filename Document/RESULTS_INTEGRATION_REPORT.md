# 結果目錄整合報告

## 整合概要

已成功整合專案中的所有輸出結果目錄，統一到 `results/` 目錄下。

## 整合前的目錄結構

1. `D:\Quinn_SmallHouse\2026_Thesis_v2\experiment_results` - 包含部分實驗1結果（不完整）
2. `D:\Quinn_SmallHouse\2026_Thesis_v2\experiments` - 包含舊的日誌檔案
3. `D:\Quinn_SmallHouse\2026_Thesis_v2\src\experiments` - 包含程式碼和部分日誌
4. `D:\Quinn_SmallHouse\2026_Thesis_v2\src\outputs` - 空目錄結構
5. `D:\Quinn_SmallHouse\2026_Thesis_v2\outputs` - 包含完整的系統化實驗結果

## 整合後的統一結構

```
results/
├── experiment1/                 # 實驗1：融合策略比較
│   ├── experiment1.log
│   ├── experiment1_summary.txt
│   ├── fusion_strategy_report.json
│   └── fusion_strategy_results.json
├── experiment2/                 # 實驗2：注意力機制比較
│   ├── attention_comparison_report.json
│   ├── attention_mechanism_profiles.json
│   ├── experiment2.log
│   └── experiment2_summary.txt
├── experiment3/                 # 實驗3：組合分析
│   ├── ablation_analysis.png
│   ├── baseline_results.json
│   ├── combination_results.json
│   ├── comprehensive_report.json
│   ├── experiment3.log
│   ├── final_summary_report.txt
│   └── integrated_experiment_results.json
├── logs/                        # 統一的日誌目錄
│   └── cross_domain_sentiment/
│       ├── error.log
│       ├── experiment.log
│       ├── performance.log
│       └── results.log
├── models/                      # 模型輸出目錄
├── visualizations/              # 統一的視覺化目錄
│   ├── ablation_analysis.png
│   ├── attention_quality_comparison.png
│   ├── complexity_comparison.png
│   ├── comprehensive_radar_chart.png
│   ├── comprehensive_radar.png
│   ├── computational_efficiency.png
│   ├── domain_adaptation_comparison.png
│   ├── domain_transferability.png
│   ├── improvement_analysis.png
│   └── performance_comparison.png
└── overall_systematic_experiments.json  # 總體實驗結果
```

## 重複檔案處理

### 實驗1結果檔案
- **選擇策略**: 選用 `outputs/systematic_experiments/experiment1/` 中的完整版本
- **原因**: 包含完整的實驗數據，而 `experiment_results/` 中的版本為空或不完整
- **差異**:
  - `experiment_results/fusion_strategy_results.json`: 空檔案 `{}`
  - `outputs/.../fusion_strategy_results.json`: 包含4個融合策略的完整結果

### 視覺化檔案
- 所有PNG檔案統一移動到 `results/visualizations/`
- 包含10個視覺化檔案，涵蓋性能比較、雷達圖、效率分析等

## 程式碼配置更新

已更新以下檔案中的路徑配置：

1. `src/main_controller.py`
   - `model_cache_dir`: `outputs/models` → `results/models`
   - `output_dir`: `outputs/reports` → `results`

2. `src/utils/config_manager.py`
   - `output_dir`: `outputs` → `results`
   - `model_dir`: `outputs/models` → `results/models`
   - `figure_dir`: `outputs/figures` → `results/visualizations`
   - `report_dir`: `outputs/reports` → `results`
   - `log_dir`: `experiments/logs` → `results/logs`

3. `src/utils/experiment_manager.py`
   - `results_dir`: `outputs/experiments` → `results`

4. `src/utils/logging_system.py`
   - `log_dir`: `experiments/logs` → `results/logs`

## 清理建議

建議在確認整合無誤後，刪除以下舊目錄：
- `experiment_results/` （已過時，內容不完整）
- `experiments/` （日誌已遷移）
- `outputs/` （內容已完整遷移）
- `src/outputs/` （空目錄結構）

## 驗證

整合後的結果目錄包含：
- **3個實驗**的完整結果
- **10個視覺化圖表**
- **統一的日誌系統**
- **總體實驗報告**

所有程式碼引用已更新為新的路徑結構。