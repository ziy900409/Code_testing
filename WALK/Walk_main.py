# -*- coding: utf-8 -*-
"""
Created on Mon May 19 23:08:47 2025

待新增功能
1. 計算相關係數的時候，設定起始位置，或是結束位置

@author: User
"""

# import re
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns # 可選，用於更美觀的圖表
# import numpy as np
import Walk_func as func

# --- 請修改以下設定 ---
INPUT_DIRECTORY = r'C:\Users\User\Downloads\範例檔-20250519T142424Z-1-001\範例檔\S1皮爾森相關分析範例'  # 例如: 'C:/Users/YourUser/Documents/ExcelData'
OUTPUT_DIRECTORY = r'C:\Users\User\Downloads\範例檔-20250519T142424Z-1-001\範例檔\S1_ProcessedData'      # 例如: 'C:/Users/YourUser/Documents/ProcessedData'
FILENAME_KEYWORD = 'GOLF'                     # 例如: 檔名中包含 'report' 的才處理



# 定義要繪圖的欄位及圖表類型
# x_col: X軸欄位
# y_col: Y軸欄位
# plot_type: 'scatter' (散佈圖), 'line' (折線圖)
# title_suffix: 圖表標題的後綴

# --- 設定結束 ---

# --- 如何在您的主流程中使用 ---

# 假設這是您定義的公式字典
CAL_FORMULAS = {
    "right 前後": "=(D2-D$2)+J$2",
    "right 左右": "=-(C2-C$2)+I2",
    "left 前後": "=(F2-F$2)+H$2", # 假設 D$2 已被正確解析並放入環境
    "left 左右": "=(E2-E$2)+G$2"
}
col_name_map = { 
    "D": "plate 1 COPY analog",
    "J": "right_cop_y(mm)",
    "C": "plate 1 COPX analog",
    "I": "right_cop_x(mm)",
    "F": "plate 2 COPY analog",
    "H": "left_cop_y(mm)",
    "E": "plate 2 COPX analog",
    "G": "left_cop_x(mm)",
    "O": "right 前後",      # "right 前後" 是 DataFrame 中的一個欄位名
    "J": "right_cop_y(mm)", # "right_cop_y(mm)" 是 DataFrame 中的一個欄位名
    "Q": "right 左右",      # "right 左右" 是 DataFrame 中的一個欄位名
    "I": "right_cop_x(mm)",  # "right_cop_x(mm)" 是 DataFrame 中的一個欄位名
    "B": "SI time(s)",
    "S": "left 前後",
    "U": "left 左右",
    
    }

person_corr = {
    "right 前後_cor": ["O", "J"], # 這裡的 "O", "J" 是您定義的代號
    "right 左右_cor": ["Q", "I"],  # 這裡的 "Q", "I" 是您定義的代號
    "left 前後_cor": ["S", "G"],
    "left 左右_cor": ["U", "G"]
}

# 繪圖設定字典範例
PLOTS_CONFIG = {
    "right 前後": { # 這個鍵會部分用於子圖標題或檔案名
        "x_col_key": "SI time(s)",      # 對應 col_name_map 中的 X 軸欄位代號
        "y1_col_key": "right 前後",  # 對應 col_name_map 中的 Y1 軸欄位代號
        "y2_col_key": "right_cop_y(mm)",  # 對應 col_name_map 中的 Y2 軸欄位代號
        "title": "right 前後", # 子圖的完整標題
        "xlabel": "時間 (秒)",             # X 軸標籤 (可選)
        # "ylabel": "感測器讀數"             # Y 軸標籤 (可選)
    },
    "right 左右": {
        "x_col_key": "SI time(s)",
        "y1_col_key": "right 左右",
        "y2_col_key": "right_cop_x(mm)",
        "title": "right 左右"
    },
    
    "left 前後": {
        "x_col_key": "SI time(s)",
        "y1_col_key": "left 前後",
        "y2_col_key": "left_cop_y(mm)",
        "title": "left 前後"
    },
    
    "left 左右": {
        "x_col_key": "SI time(s)",
        "y1_col_key": "left 左右",
        "y2_col_key": "left_cop_x(mm)",
        "title": "left 前後",
    }
    
}

col_name_mapping = col_name_map
plots_config = PLOTS_CONFIG




# %%

exls_file_list = func.Read_File()


