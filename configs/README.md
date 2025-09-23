# 配置文件說明

本目錄包含專案的所有配置文件，統一管理所有設定。

## 文件結構

```
configs/
├── default_config.json     # 預設系統配置
├── experiment_config.json  # 實驗專用配置  
├── quick_config.json       # 快速測試配置
└── README.md               # 本說明文件
```

## 配置文件說明

### default_config.json
包含系統的基本配置：
- **model**: 模型相關參數（維度、層數等）
- **training**: 訓練相關參數（批次大小、學習率等）
- **data**: 資料集相關設定
- **device**: 計算設備設定
- **logging**: 日誌記錄設定
- **output**: 輸出目錄設定
- **experiments**: 實驗框架設定

### experiment_config.json
包含三個主要實驗的詳細配置：
- **experiment1**: 融合策略比較實驗
- **experiment2**: 注意力機制比較實驗
- **experiment3**: 組合效果分析實驗

### quick_config.json
用於快速測試和開發的精簡配置：
- 減少訓練輪數
- 使用較小的批次大小
- 簡化實驗參數

## 使用方式

### 在程式碼中載入配置

```python
import json

# 載入預設配置
with open('configs/default_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 載入實驗配置
with open('configs/experiment_config.json', 'r', encoding='utf-8') as f:
    exp_config = json.load(f)
```

### 命令列指定配置

```bash
# 使用預設配置
python src/main_controller.py

# 使用指定配置文件
python src/main_controller.py --config configs/quick_config.json

# 使用實驗模式
python src/main_controller.py --mode systematic --experiments all
```

## 配置優先級

1. 命令列參數（最高優先級）
2. 指定的配置文件
3. default_config.json（預設）
4. 程式碼內建預設值（最低優先級）

## 注意事項

- 所有路徑都相對於專案根目錄
- 修改配置後建議備份原始設定
- 實驗配置與預設配置可以組合使用
- 配置文件支援中文註釋和說明