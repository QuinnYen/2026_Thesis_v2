# 系統性實驗使用指南

## 概述

本系統現在已整合了完整的系統性實驗框架，包含三個主要實驗：

1. **實驗1：融合策略比較** - 比較不同特徵融合策略的效果
2. **實驗2：注意力機制比較** - 評估7種注意力機制的優劣勢  
3. **實驗3：組合效果分析** - 最佳組合與基線方法的全面對比

## 運行方式

### 基本命令格式

```bash
cd src
python main_controller.py [選項]
```

### 運行模式

#### 1. 只運行基本訓練（預設）
```bash
python main_controller.py
# 或
python main_controller.py --mode basic
```

#### 2. 只運行系統性實驗
```bash
# 運行所有系統性實驗（實驗1-3）
python main_controller.py --mode systematic

# 運行特定實驗
python main_controller.py --mode systematic --experiments 1
python main_controller.py --mode systematic --experiments 2  
python main_controller.py --mode systematic --experiments 3
python main_controller.py --mode systematic --experiments 1,2
python main_controller.py --mode systematic --experiments all
```

#### 3. 運行完整流程（基本訓練 + 系統性實驗）
```bash
# 運行所有實驗
python main_controller.py --mode both

# 運行基本訓練 + 特定系統性實驗
python main_controller.py --mode both --experiments 1,2,3
```

### 完整參數說明

```bash
python src/main_controller.py \
    --config [配置文件路徑] \
    --mode [basic|systematic|both] \
    --experiments [1|2|3|1,2|all] \
    --quick
```

**參數說明：**
- `--config`: 可選，指定配置文件路徑
  - `configs/default_config.json`: 預設配置（完整實驗）
  - `configs/quick_config.json`: 快速測試配置（縮短時間）
  - `configs/experiment_config.json`: 實驗專用配置
- `--mode`: 實驗模式
  - `basic`: 只運行基本模型訓練評估
  - `systematic`: 只運行系統性實驗
  - `both`: 運行基本訓練 + 系統性實驗
- `--experiments`: 指定要運行的系統性實驗
  - `1`: 融合策略比較實驗
  - `2`: 注意力機制比較實驗
  - `3`: 組合效果分析實驗
  - `1,2`: 運行實驗1和2
  - `all`: 運行所有實驗
- `--quick`: 快速模式（減少訓練輪數）

### 配置文件使用

新的統一配置系統位於 `configs/` 目錄：

```bash
# 使用預設配置運行完整實驗
python src/main_controller.py --config configs/default_config.json --mode systematic --experiments all

# 使用快速配置進行測試
python src/main_controller.py --config configs/quick_config.json --mode systematic --experiments 1

# 使用預設配置（不指定config參數時自動使用）
python src/main_controller.py --mode systematic --experiments all
```

## 實驗輸出

### 專案結構（重構後）
```
2026_Thesis_v2/
├── src/
│   ├── experiments/           # 實驗程式碼模組
│   ├── utils/                # 工具模組
│   ├── models/               # 模型定義
│   ├── data/                 # 數據處理
│   └── main_controller.py    # 主控制器
├── configs/                  # 統一配置管理
│   ├── default_config.json   # 預設配置
│   ├── experiment_config.json # 實驗配置
│   ├── quick_config.json     # 快速測試配置
│   └── README.md             # 配置說明
├── outputs/                  # 所有輸出結果
│   ├── systematic_experiments/
│   │   ├── experiment1/      # 實驗1結果
│   │   ├── experiment2/      # 實驗2結果
│   │   ├── experiment3/      # 實驗3結果
│   │   └── overall_systematic_experiments.json
│   ├── logs/                 # 日誌文件
│   ├── models/               # 訓練模型
│   └── reports/              # 報告文件
├── data/                     # 數據集
└── docs/                     # 項目文檔
```

### 主要輸出文件

#### 實驗1輸出
- `fusion_strategy_results.json`: 詳細的融合策略對比結果
- `fusion_strategy_report.json`: 實驗報告和分析
- `experiment1_summary.txt`: 摘要報告

#### 實驗2輸出
- `attention_mechanism_profiles.json`: 各注意力機制的詳細分析
- `attention_comparison_report.json`: 比較報告
- 可視化圖表：性能對比、複雜度分析、雷達圖等

#### 實驗3輸出
- `baseline_results.json`: 基線方法結果
- `combination_results.json`: 最佳組合結果
- `comprehensive_report.json`: 綜合分析報告
- `final_summary_report.txt`: 最終摘要
- 多張高品質可視化圖表

## 實驗結果解讀

### 關鍵指標
1. **準確率 (Accuracy)**: 整體分類正確率
2. **F1分數**: 精確度和召回率的調和平均
3. **計算複雜度**: 參數量、推理時間、記憶體使用
4. **跨領域穩定性**: 在不同領域的性能一致性
5. **統計顯著性**: 改進的統計學意義

### 實驗流程
1. **實驗1** → 找出最佳融合策略
2. **實驗2** → 使用最佳融合策略，找出最佳注意力機制
3. **實驗3** → 組合最佳策略和機制，與基線方法對比

## 使用示例

### 完整論文實驗流程
```bash
# 運行完整的系統性實驗（推薦用於論文）
python src/main_controller.py --mode systematic --experiments all

# 使用完整配置運行所有實驗
python src/main_controller.py --config configs/default_config.json --mode systematic --experiments all
```

### 快速驗證
```bash
# 快速運行實驗3以驗證最佳組合效果
python src/main_controller.py --config configs/quick_config.json --mode systematic --experiments 3

# 使用quick參數快速驗證
python src/main_controller.py --mode systematic --experiments 3 --quick
```

### 特定研究重點
```bash
# 只關注融合策略比較
python src/main_controller.py --mode systematic --experiments 1

# 只關注注意力機制比較  
python src/main_controller.py --mode systematic --experiments 2

# 使用快速配置測試特定實驗
python src/main_controller.py --config configs/quick_config.json --mode systematic --experiments 1,2
```

## 注意事項

1. **運行時間**: 完整的系統性實驗可能需要較長時間（取決於硬體配置）
2. **記憶體需求**: 確保有足夠的GPU記憶體運行多個模型
3. **依賴檢查**: 系統會自動檢查實驗框架是否正確載入
4. **結果保存**: 所有實驗結果都會自動保存到指定目錄
5. **可視化**: 實驗會自動生成多種圖表用於分析

## 故障排除

### 常見問題
1. **實驗框架導入失敗**: 檢查 `src/experiments/` 目錄下的文件是否完整
2. **CUDA記憶體不足**: 使用 `configs/quick_config.json` 配置或 `--quick` 參數
3. **配置文件找不到**: 確保配置文件位於 `configs/` 目錄下
4. **依賴缺失**: 確保安裝了所需的Python包（matplotlib, seaborn等）

### 專案結構變更說明

**✅ 已重構（推薦）**：
- 配置文件統一管理：`configs/`
- 實驗代碼模組：`src/experiments/`  
- 所有輸出：`outputs/`
- 簡潔清晰的結構

**❌ 舊結構（已清理）**：
- ~~根目錄 `/experiments/`~~ 已移除
- ~~重複的配置目錄~~ 已整合
- ~~分散的輸出位置~~ 已統一

### 錯誤處理
系統會自動處理實驗過程中的錯誤，並在日誌中記錄詳細信息。如果某個實驗失敗，其他實驗仍會繼續運行。

---

現在您可以通過命令列輕鬆運行系統性實驗並獲得完整的實驗結果！