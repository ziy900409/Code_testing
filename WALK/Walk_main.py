# -*- coding: utf-8 -*-
"""
Created on Mon May 19 23:08:47 2025

待新增功能
1. 計算相關係數的時候，設定起始位置，或是結束位置

@author: User
"""
from collections import defaultdict
import pandas as pd
import os
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"D:\git\Code_testing\WALK")
# import numpy as np

import Walk_func as func



# %%

# --- 請修改以下設定 ---
INPUT_DIRECTORY = r'D:\Hsin\NTSU_lab\WALK\範例檔-20250519T142424Z-1-001\範例檔\S1皮爾森相關分析範例\\'  # 例如: 'C:/Users/YourUser/Documents/ExcelData'
OUTPUT_DIRECTORY = r'D:\Hsin\NTSU_lab\WALK\範例檔-20250519T142424Z-1-001\範例檔\S1_ProcessedData\\'      # 例如: 'C:/Users/YourUser/Documents/ProcessedData'
FILENAME_KEYWORD = ['GOLF', 'WALK', "JUMP"]                     # 例如: 檔名中包含 'report' 的才處理
FILENAME_KEYWORD = ['GOLF', 'WALK', "JUMP"]                     # 例如: 檔名中包含 'report' 的才處理

# excel_path = r"C:\Users\User\Downloads\範例檔-20250519T142424Z-1-001\範例檔\S1皮爾森相關分析範例\S1_GOLF_1 FP&SI extra v2.xlsx"

# 假設這是您定義的公式字典
CAL_FORMULAS = {
    "right 前後": "=(D2-D$2)+J$2",
    "right 左右": "=-(C2-C$2)+I2",
    "left 前後": "=(F2-F$2)+H$2", # 假設 D$2 已被正確解析並放入環境
    "left 左右": "=(E2-E$2)+G$2"
}


# 各EXCEL欄位對應名稱
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

# filter col
filter_col = {
    "right 前後",
    "right 左右",
    "left 前後",
    "left 左右",
    }

# Person correleration setting
person_corr = {
    "right 前後_cor": ["N", "J"], # 這裡的 "O", "J" 是您定義的代號
    "right 左右_cor": ["P", "I"],  # 這裡的 "Q", "I" 是您定義的代號
    "left 前後_cor": ["R", "H"],
    "left 左右_cor": ["T", "G"]
}

# 繪圖設定字典範例
# x_col: X軸欄位
# y_col: Y軸欄位
# plot_type: 'scatter' (散佈圖), 'line' (折線圖)
# title_suffix: 圖表標題的後綴
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


# %%

exls_file_list = func.Read_File(INPUT_DIRECTORY,
                                file_type=".xlsx")

all_results_data = defaultdict(dict)

for file_key in FILENAME_KEYWORD:
    for file in exls_file_list:
        file_name_with_ext = os.path.basename(file)  # → "S1_GOLF_1 FP&SI extra v2.xlsx"
        if file_key in file_name_with_ext:
            print(file)

            all_correlation_results = func.process_file(file,
                                                        CAL_FORMULAS=CAL_FORMULAS,
                                                        col_name_map=col_name_map,
                                                        person_corr=person_corr,
                                                        PLOTS_CONFIG=PLOTS_CONFIG,
                                                        OUTPUT_DIRECTORY=OUTPUT_DIRECTORY,
                                                        filter_col=filter_col)
            all_correlation_results = func.process_file(file,
                                                        CAL_FORMULAS=CAL_FORMULAS,
                                                        col_name_map=col_name_map,
                                                        person_corr=person_corr,
                                                        PLOTS_CONFIG=PLOTS_CONFIG,
                                                        OUTPUT_DIRECTORY=OUTPUT_DIRECTORY,
                                                        filter_col=filter_col,
                                                        NeedFilter=False)
            all_results_data[file_name_with_ext] = all_correlation_results



#%%


if 'all_results_data' in locals() and all_results_data:
    # 轉換資料結構以便建立 DataFrame

    # 先收集所有可能的相關係數鍵名，作為 DataFrame 的欄位
    all_corr_keys = set()
    for filename, corr_results in all_results_data.items():
        if isinstance(corr_results, dict): # 確保 corr_results 是字典
             all_corr_keys.update(corr_results.keys())
        else:
            print(f"警告：檔案 '{filename}' 的相關係數結果不是一個字典，已跳過。結果：{corr_results}")


    # 排序以確保欄位順序一致 (可選)
    sorted_corr_keys = sorted(list(all_corr_keys))

    # 準備 DataFrame 的資料列表
    data_for_df = []
    for filename, corr_results in all_results_data.items():
        row = {"檔案名稱": filename} # 使用字典的鍵作為檔案名稱
        if isinstance(corr_results, dict): # 再次檢查
            for key in sorted_corr_keys:
                row[key] = corr_results.get(key) # 從 corr_results 獲取值
        else: # 如果 corr_results 不是字典，則為此檔案名稱的所有相關係數鍵填充 NaN
            for key in sorted_corr_keys:
                row[key] = None # 或者使用 pd.NA
        data_for_df.append(row)

    # 建立 DataFrame
    df_summary = pd.DataFrame(data_for_df)

    # 重新排列欄位順序，將 '檔案名稱' 放在第一欄 (可選)
    if not df_summary.empty:
        column_order = ["檔案名稱"] + sorted_corr_keys
        # 確保 column_order 中的所有欄位都存在於 df_summary 中，以避免 KeyError
        valid_column_order = [col for col in column_order if col in df_summary.columns]
        df_summary = df_summary[valid_column_order]
    else:
        print("沒有資料可供寫入 DataFrame。")


    # 指定輸出的 Excel 檔案路徑和名稱
    # OUTPUT_DIRECTORY = "your_output_directory" # <--- 請務必修改為您的實際輸出資料夾路徑
    summary_excel_filename = "correlation_summary_from_defaultdict.xlsx"
    summary_excel_filepath = os.path.join(OUTPUT_DIRECTORY, summary_excel_filename)

    # 建立輸出資料夾 (如果不存在)
    try:
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    except OSError as e:
        print(f"建立輸出資料夾 '{OUTPUT_DIRECTORY}' 時發生錯誤: {e}")
        # 如果資料夾建立失敗，可能後續儲存也會失敗，可以選擇在此處終止或讓它嘗試儲存

    # 將 DataFrame 寫入 Excel 檔案
    if not df_summary.empty:
        try:
            df_summary.to_excel(summary_excel_filepath, index=False, engine='openpyxl')
            print(f"摘要結果已成功儲存至 Excel 檔案: {summary_excel_filepath}")
        except Exception as e:
            print(f"儲存 Excel 檔案 '{summary_excel_filepath}' 時發生錯誤: {e}")
            print("請確認您已安裝 'openpyxl' 函式庫 (pip install openpyxl)")
    else:
        print(f"由於 DataFrame 為空，未產生 Excel 檔案。")

else:
    print("錯誤: 'all_results_data' 未定義或為空，無法進行 Excel 輸出。")
    print("請確保您已將 defaultdict 物件賦值給 'all_results_data' 變數。")




















