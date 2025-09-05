# 跨領域情感分析系統

## 專案概述
本專案實現了一個跨領域情感分析系統，使用多種注意力機制進行面向級情感分類，並提供跨領域對齊評估功能。

## 系統架構
- **數據處理模組** (`src/data/`): 數據載入、預處理、特徵提取、跨領域對齊
- **注意力機制** (`src/attention/`): 相似度注意力、關鍵詞導向注意力、自注意力、多頭注意力
- **模型訓練** (`src/models/`): BERT編碼器、注意力組合器、特徵融合、分類器
- **評估分析** (`src/evaluation/`): 標準評估、跨領域對齊評估、統計分析、錯誤分析
- **可視化** (`src/visualization/`): 結果可視化、注意力可視化、語義空間可視化
- **工具函數** (`src/utils/`): 配置管理、日誌系統、工具函數、實驗管理

## 安裝說明

### 1. 建立虛擬環境
```bash
python -m venv venv
```

### 2. 啟動虛擬環境
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. 安裝依賴套件
```bash
pip install -r requirements.txt
```

### 4. 驗證安裝
```bash
python simple_test.py
```

## 已安裝的核心套件
- **深度學習**: PyTorch 2.8.0+cpu, Transformers 4.56.1
- **數據處理**: Pandas 2.3.2, NumPy 2.1.2, Scikit-learn 1.7.1
- **可視化**: Matplotlib, Seaborn
- **配置管理**: PyYAML, TQDM, psutil

## 目錄結構
```
project_root/
├── src/                    # 源代碼目錄
│   ├── data/              # 數據處理模組
│   ├── models/            # 模型定義
│   ├── attention/         # 注意力機制
│   ├── evaluation/        # 評估模組
│   ├── visualization/     # 可視化
│   └── utils/             # 工具函數
├── data/                  # 數據存放目錄
│   ├── raw/               # 原始數據
│   ├── processed/         # 處理後數據
│   └── cache/             # 快取文件
├── experiments/           # 實驗配置和結果
│   ├── configs/           # 配置文件
│   ├── logs/              # 日誌文件
│   └── results/           # 實驗結果
├── outputs/               # 輸出目錄
│   ├── models/            # 訓練好的模型
│   ├── figures/           # 生成的圖表
│   └── reports/           # 實驗報告
├── tests/                 # 測試代碼
├── venv/                  # 虛擬環境
├── requirements.txt       # 套件依賴
└── README.md             # 說明文件
```

## 快速開始

### 1. 配置管理
```python
from src.utils import ConfigManager

# 建立配置管理器
config_manager = ConfigManager()

# 獲取預設配置
config = config_manager.get_default_config()

# 建立新實驗配置
experiment_config = config_manager.create_experiment_config(
    "my_experiment",
    data={'batch_size': 16},
    training={'learning_rate': 1e-5}
)
```

### 2. 實驗管理
```python
from src.utils import experiment_manager

# 建立實驗系列
experiment_names = experiment_manager.create_experiment_series(
    "sentiment_analysis",
    param_space={
        'learning_rate': [1e-5, 2e-5, 5e-5],
        'batch_size': [16, 32]
    }
)

# 執行實驗
def my_experiment_function(config, logger):
    # 你的實驗代碼
    return {'accuracy': 0.85, 'f1': 0.82}

results = experiment_manager.run_experiment_batch(
    experiment_names, my_experiment_function
)
```

### 3. 日誌記錄
```python
from src.utils import logging_manager

# 獲取日誌記錄器
logger = logging_manager.get_logger("my_experiment")

# 記錄實驗開始
logger.log_experiment_start(config)

# 開始性能監控
logger.start_performance_monitoring()

# 記錄進度
logger.log_epoch_progress(epoch=0, total_epochs=10, metrics={'loss': 0.5})

# 記錄實驗結束
logger.log_experiment_end({'final_accuracy': 0.85})
```

## 開發指南

### 添加新的注意力機制
1. 在 `src/attention/` 目錄下建立新文件
2. 繼承基礎注意力類別
3. 實現 forward 方法
4. 更新 `__init__.py` 匯出

### 添加新的評估指標
1. 在 `src/evaluation/` 目錄下擴展評估器
2. 實現新的評估函數
3. 更新配置文件中的評估指標列表

## 注意事項
- 確保使用虛擬環境避免套件衝突
- 實驗配置文件使用 YAML 格式
- 所有實驗結果都會自動保存和版本控制
- 使用 `set_random_seed()` 確保實驗可重現性

## 技術支援
如遇到問題，請檢查：
1. 虛擬環境是否正確啟動
2. 套件是否完整安裝
3. 目錄結構是否完整
4. 執行 `python simple_test.py` 進行系統驗證