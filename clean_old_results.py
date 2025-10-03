#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理舊的實驗結果檔案
避免格式不相容問題
"""

import os
from pathlib import Path

def clean_old_results():
    """清理舊的實驗結果檔案"""

    results_dir = Path("results")

    if not results_dir.exists():
        print("無 results 目錄，無需清理")
        return

    # 要清理的檔案模式
    patterns_to_clean = [
        "*_report.json",
        "integrated_experiment_report.json",
        "cross_domain_sentiment_*.json"
    ]

    cleaned_files = []

    for pattern in patterns_to_clean:
        for file_path in results_dir.glob(pattern):
            print(f"清理檔案: {file_path}")
            file_path.unlink()
            cleaned_files.append(str(file_path))

    if cleaned_files:
        print(f"\n✅ 已清理 {len(cleaned_files)} 個舊檔案")
        print("現在可以重新運行實驗系統")
    else:
        print("✅ 無需清理任何檔案")

if __name__ == "__main__":
    print("="*50)
    print("清理舊的實驗結果檔案")
    print("="*50)
    clean_old_results()