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
# sys.path.append(r"D:\git\Code_testing\WALK")
sys.path.append(r"D:\BenQ_Project\gitgit\Code_testing\WALK")

# import numpy as np

import Walk_func as func

# %%
CONFIG = {
    "INPUT_DIRECTORY": r'D:\BenQ_Project\python\WALK\第一階段（100hz）\merge xlsx', # 例如: 'C:/Users/YourUser/Documents/ExcelData'
    "OUTPUT_DIRECTORY": r'D:\BenQ_Project\python\WALK\ProcessedData\\' ,     # 例如: 'C:/Users/YourUser/Documents/ProcessedData'
    "STAGEFILE": r'D:\BenQ_Project\python\WALK\StagingFlie',
    # 設定要抓取的檔案名稱，要計算的欄位算式，以及PERSON CORR的欄位名稱
    "TYPE":{
            # "TYPE1":{
            #         "FILENAME_KEYWORD":{'ST_WALK', "ST_RUN"},
            #         "CAL_FORMULAS":{
            #                         "right 前後": "=-(C2-C$2)+J$2",
            #                         "right 左右": "=-(D2-D$2)+I$2",
            #                         "left 前後": "=-(E2-E$2)+H$2", # 假設 D$2 已被正確解析並放入環境
            #                         "left 左右": "=(F2-F$2)+G$2",
            #                         "left_FPsi_x(mm)": "=-(G2-G$2)+E$2",
            #                         "left_FPsi_y(mm)": "=-(H2-H$2)+F$2",
            #                         "right_FPsi_x(mm)": "=-(I2-I$2)+C$2",
            #                         "right_FPsi_y(mm)": "=(J2-J$2)+D$2",
            #                        },
            #         "PERSON_CORR":{
            #                        "right 前後_cor": {"right 前後", "right_cop_y(mm)"}, # 這裡的 "O", "J" 是您定義的代號
            #                        "right 左右_cor": {"right 左右", "right_cop_x(mm)"},  # 這裡的 "Q", "I" 是您定義的代號
            #                        "left 前後_cor": {"left 前後", "left_cop_y(mm)"},
            #                        "left 左右_cor": {"left 左右", "left_copx"}
            #                    },
            #         "COL_NAME_MAP": {
            #                          "A": "FP time",
            #                          "B": "SI Time(s)",
            #                          "C": "plate 1右 COPX analog",
            #                          "D": "plate 1 COPY analog",
            #                          "E": "plate 2 左COPX analog",
            #                          "F": "plate 2 COPY analog",
            #                          "G": "left_copx",
            #                          "H": "left_cop_y(mm)",
            #                          "I": "right_cop_x(mm)",
            #                          "J": "right_cop_y(mm)",
            #                          "N": "right 前後",      # "right 前後" 是 DataFrame 中的一個欄位名
            #                          "P": "right 左右",      # "right 左右" 是 DataFrame 中的一個欄位名
            #                          "R": "left 前後",
            #                          "T": "left 左右",
            #                          },
            #         "PLOTS_CONFIG":{
            #                         "right 前後": { # 這個鍵會部分用於子圖標題或檔案名
            #                                         "x_col_key": "SI Time(s)",      # 對應 col_name_map 中的 X 軸欄位代號
            #                                         "y1_col_key": "right 前後",  # 對應 col_name_map 中的 Y1 軸欄位代號
            #                                         "y2_col_key": "right_cop_y(mm)",  # 對應 col_name_map 中的 Y2 軸欄位代號
            #                                         "title": "right 前後", # 子圖的完整標題
            #                                         "xlabel": "時間 (秒)",             # X 軸標籤 (可選)
            #                                         # "ylabel": "感測器讀數"             # Y 軸標籤 (可選)
            #                                         },
            #                         "right 左右": {
            #                                        "x_col_key": "SI Time(s)",
            #                                        "y1_col_key": "right 左右",
            #                                        "y2_col_key": "right_cop_x(mm)",
            #                                        "title": "right 左右"
            #                                        },
                        
            #                         "left 前後": {
            #                                       "x_col_key": "SI Time(s)",
            #                                       "y1_col_key": "left 前後",
            #                                       "y2_col_key": "left_cop_y(mm)",
            #                                       "title": "left 前後"
            #                                       },
                                    
            #                         "left 左右": {
            #                                       "x_col_key": "SI Time(s)",
            #                                       "y1_col_key": "left 左右",
            #                                       "y2_col_key": "left_cop_x(mm)",
            #                                       "title": "left 前後",
            #                                     },
                        
            #                         }
            #         },
            "TYPE4":{
                    "FILENAME_KEYWORD":{'ST_WALK', "ST_RUN"},
                    "CAL_FORMULAS":{
                                    "right 前後": "=-(C2-C$2)+J$2",
                                    "right 左右": "=-(D2-D$2)+I$2",
                                    "left 前後": "=-(E2-E$2)+H$2", # 假設 D$2 已被正確解析並放入環境
                                    "left 左右": "=(F2-F$2)+G$2",
                                    "left_FPsi_x(mm)": "=-(G2-G$2)+E$2",
                                    "left_FPsi_y(mm)": "=-(H2-H$2)+F$2",
                                    "right_FPsi_x(mm)": "=-(I2-I$2)+C$2",
                                    "right_FPsi_y(mm)": "=(J2-J$2)+D$2",
                                   },
                    "PERSON_CORR":{
                                   "right 前後_cor": {"right 前後", "SI right copy"}, # 這裡的 "O", "J" 是您定義的代號
                                   "right 左右_cor": {"right 左右", "SI right copx"},  # 這裡的 "Q", "I" 是您定義的代號
                                   "left 前後_cor": {"left 前後", "SI left copy"},
                                   "left 左右_cor": {"left 左右", "SI left copx"}
                               },
                    "COL_NAME_MAP": {
                                     "A": "FP frame number",
                                     "B": "FP time",
                                     "C": "SI time",
                                     "D": "FP1 COPX",
                                     "E": "FP1 COPY",
                                     "F": "FP2 COPX",
                                     "G": "FP2 COPY",
                                     "H": "SI left copx",
                                     "I": "SI left copy",
                                     "J": "SI right copx",
                                     "K": "SI right copy",
                                     "N": "right 前後",      # "right 前後" 是 DataFrame 中的一個欄位名
                                     "P": "right 左右",      # "right 左右" 是 DataFrame 中的一個欄位名
                                     "R": "left 前後",
                                     "T": "left 左右",
                                     },
                    "PLOTS_CONFIG":{
                                    "right 前後": { # 這個鍵會部分用於子圖標題或檔案名
                                                    "x_col_key": "SI time",      # 對應 col_name_map 中的 X 軸欄位代號
                                                    "y1_col_key": "right 前後",  # 對應 col_name_map 中的 Y1 軸欄位代號
                                                    "y2_col_key": "SI right copy",  # 對應 col_name_map 中的 Y2 軸欄位代號
                                                    "title": "right 前後", # 子圖的完整標題
                                                    "xlabel": "時間 (秒)",             # X 軸標籤 (可選)
                                                    # "ylabel": "感測器讀數"             # Y 軸標籤 (可選)
                                                    },
                                    "right 左右": {
                                                   "x_col_key": "SI time",
                                                   "y1_col_key": "right 左右",
                                                   "y2_col_key": "SI right copx",
                                                   "title": "right 左右"
                                                   },
                        
                                    "left 前後": {
                                                  "x_col_key": "SI time",
                                                  "y1_col_key": "left 前後",
                                                  "y2_col_key": "SI left copy",
                                                  "title": "left 前後"
                                                  },
                                    
                                    "left 左右": {
                                                  "x_col_key": "SI time",
                                                  "y1_col_key": "left 左右",
                                                  "y2_col_key": "SI left copx",
                                                  "title": "left 前後",
                                                },
                        
                                    }
                    },
            
            # "TYPE2":{
            #         "FILENAME_KEYWORD":{'R_WALK, L_WALK', 'R_CHANGE', "L_CHANGE", "L_JUMP", "H_JUMP"},
            #         "CAL_FORMULAS":{
            #                         "right 前後": "=-(F2-F$2)+J2",
            #                         "right 左右": "=(E2-E$2)+I$2",
            #                         "left 前後": "=-(D2-D$2)+H$2", # 假設 D$2 已被正確解析並放入環境
            #                         "left 左右": "=-(C2-C$2)+G$2"
            #                         },
            #         "PERSON_CORR":{
            #                         "right 前後_cor": {"right 前後", "right_cop_y(mm)"}, # 這裡的 "O", "J" 是您定義的代號
            #                         "right 左右_cor": {"right 左右", "right_cop_x(mm)"},  # 這裡的 "Q", "I" 是您定義的代號
            #                         "left 前後_cor": {"left 前後", "left_cop_y(mm)"},
            #                         "left 左右_cor": {"left 左右", "left_copx"}
            #                     },
            #         # 設定 EXCEL 中欄位編號對應名稱
            #         "COL_NAME_MAP": { 
            #                           "A": "FP time",
            #                           "B": "SI Time(s)",
            #                           "C": "plate 1右 COPX analog",
            #                           "D": "plate 1 COPY analog",
            #                           "E": "plate 2 左COPX analog",
            #                           "F": "plate 2 COPY analog",
            #                           "G": "left_copx",
            #                           "H": "left_cop_y(mm)",
            #                           "I": "right_cop_x(mm)",
            #                           "J": "right_cop_y(mm)",
            #                           "O": "right 前後",      # "right 前後" 是 DataFrame 中的一個欄位名
            #                           "Q": "right 左右",      # "right 左右" 是 DataFrame 中的一個欄位名
            #                           "S": "left 前後",
            #                           "U": "left 左右",
            #                           },
            #         "PLOTS_CONFIG":{
            #                         "right 前後": { # 這個鍵會部分用於子圖標題或檔案名
            #                                         "x_col_key": "SI Time(s)",      # 對應 col_name_map 中的 X 軸欄位代號
            #                                         "y1_col_key": "right 前後",  # 對應 col_name_map 中的 Y1 軸欄位代號
            #                                         "y2_col_key": "right_cop_y(mm)",  # 對應 col_name_map 中的 Y2 軸欄位代號
            #                                         "title": "right 前後", # 子圖的完整標題
            #                                         "xlabel": "時間 (秒)",             # X 軸標籤 (可選)
            #                                         # "ylabel": "感測器讀數"             # Y 軸標籤 (可選)
            #                                         },
            #                         "right 左右": {
            #                                        "x_col_key": "SI Time(s)",
            #                                        "y1_col_key": "right 左右",
            #                                        "y2_col_key": "right_cop_x(mm)",
            #                                        "title": "right 左右"
            #                                        },
                        
            #                         "left 前後": {
            #                                       "x_col_key": "SI Time(s)",
            #                                       "y1_col_key": "left 前後",
            #                                       "y2_col_key": "left_cop_y(mm)",
            #                                       "title": "left 前後"
            #                                       },
                                    
            #                         "left 左右": {
            #                                       "x_col_key": "SI Time(s)",
            #                                       "y1_col_key": "left 左右",
            #                                       "y2_col_key": "left_cop_x(mm)",
            #                                       "title": "left 前後",
            #                                     },
                        
            #                         }
            #         },
            # "TYPE3":{
            #         "FILENAME_KEYWORD":{'GOLF'},
            #         "CAL_FORMULAS":{
            #                         "right 前後": "=(D2-D$2)+J$2",
            #                         "right 左右": "=-(C2-C$2)+I2",
            #                         "left 前後": "=(F2-F$2)+H$2", # 假設 D$2 已被正確解析並放入環境
            #                         "left 左右": "=(E2-E$2)+G$2"
            #                         },
            #         "PERSON_CORR":{
            #                         "right 前後_cor": {"N", "J"}, # 這裡的 "O", "J" 是您定義的代號
            #                         "right 左右_cor": {"P", "I"},  # 這裡的 "Q", "I" 是您定義的代號
            #                         "left 前後_cor": {"R", "H"},
            #                         "left 左右_cor": {"T", "G"}
            #                     },
            #         # 設定 EXCEL 中欄位編號對應名稱
            #         "COL_NAME_MAP": { 
            #                           "A": "FP time",
            #                           "B": "SI Time(s)",
            #                           "C": "plate 1右 COPX analog",
            #                           "D": "plate 1 COPY analog",
            #                           "E": "plate 2 左COPX analog",
            #                           "F": "plate 2 COPY analog",
            #                           "G": "left_copx",
            #                           "H": "left_cop_y(mm)",
            #                           "I": "right_cop_x(mm)",
            #                           "J": "right_cop_y(mm)",
            #                           "O": "right 前後",      # "right 前後" 是 DataFrame 中的一個欄位名
            #                           "Q": "right 左右",      # "right 左右" 是 DataFrame 中的一個欄位名
            #                           "S": "left 前後",
            #                           "U": "left 左右",
            #                           },
            #         "PLOTS_CONFIG":{
            #                         "right 前後": { # 這個鍵會部分用於子圖標題或檔案名
            #                                         "x_col_key": "SI Time(s)",      # 對應 col_name_map 中的 X 軸欄位代號
            #                                         "y1_col_key": "right 前後",  # 對應 col_name_map 中的 Y1 軸欄位代號
            #                                         "y2_col_key": "right_cop_y(mm)",  # 對應 col_name_map 中的 Y2 軸欄位代號
            #                                         "title": "right 前後", # 子圖的完整標題
            #                                         "xlabel": "時間 (秒)",             # X 軸標籤 (可選)
            #                                         # "ylabel": "感測器讀數"             # Y 軸標籤 (可選)
            #                                         },
            #                         "right 左右": {
            #                                        "x_col_key": "SI Time(s)",
            #                                        "y1_col_key": "right 左右",
            #                                        "y2_col_key": "right_cop_x(mm)",
            #                                        "title": "right 左右"
            #                                        },
                        
            #                         "left 前後": {
            #                                       "x_col_key": "SI Time(s)",
            #                                       "y1_col_key": "left 前後",
            #                                       "y2_col_key": "left_cop_y(mm)",
            #                                       "title": "left 前後"
            #                                       },
                                    
            #                         "left 左右": {
            #                                       "x_col_key": "SI Time(s)",
            #                                       "y1_col_key": "left 左右",
            #                                       "y2_col_key": "left_cop_x(mm)",
            #                                       "title": "left 前後",
            #                                     },
            #                         }
            #         },
            },

    # 設定要濾波的欄位
    "FILTER_COL": {
                   "right 前後",
                   "right 左右",
                   "left 前後",
                   "left 左右",
                   },
    }




# %%
type_num = range(len(CONFIG.get("TYPE", None)))
for idx in CONFIG["TYPE"]:
    print(idx)
    INDI_CONGIF = CONFIG["TYPE"][idx]
    FILENAME_KEYWORD = CONFIG["TYPE"][idx].get("FILENAME_KEYWORD")
    CAL_FORMULAS = CONFIG["TYPE"][idx].get("CAL_FORMULAS")
    PERSON_CORR = CONFIG["TYPE"][idx].get("PERSON_CORR")
    COL_NAME_MAP = CONFIG["TYPE"][idx].get("COL_NAME_MAP")
    PLOTS_CONFIG = CONFIG["TYPE"][idx].get("PLOTS_CONFIG")
    STAGEFILE_FOLDER = CONFIG.get("STAGEFILE")
    
    INPUT_DIRECTORY = CONFIG.get("INPUT_DIRECTORY")
    OUTPUT_DIRECTORY = CONFIG.get("OUTPUT_DIRECTORY")
    FILTER_COL =  CONFIG.get("FILTER_COL")
    
   
    
    exls_file_list = func.Read_File(INPUT_DIRECTORY,
                                    file_type=".xlsx")
    
    exls_file_list = [f for f in exls_file_list if not os.path.basename(f).startswith("~$")]

    all_results_data = defaultdict(dict)

    for file_key in FILENAME_KEYWORD:
        
        print(file_key)
        if file_key in "GOLF_SI":
            StageFile = pd.read_excel(STAGEFILE_FOLDER + "\\GOLF_SI.xlsx")
        elif file_key in "JUMP_SI":
            StageFile = pd.read_excel(STAGEFILE_FOLDER + "\\JUMP_SI.xlsx")
        elif file_key in "L_CHANGE_SI":
            StageFile = pd.read_excel(STAGEFILE_FOLDER + "\\L_CHANGE_SI.xlsx")
        elif file_key in "L_WALK_SI":
            StageFile = pd.read_excel(STAGEFILE_FOLDER + "\\L_WALK_SI.xlsx")
        elif file_key in "R_CHANGE_SI": 
            StageFile = pd.read_excel(STAGEFILE_FOLDER + "\\R_CHANGE_SI.xlsx")
        elif file_key in "R_WALK_SI":
            StageFile = pd.read_excel(STAGEFILE_FOLDER + "\\R_WALK_SI.xlsx")
        elif file_key in "ST_RUN_SI":
            StageFile = pd.read_excel(STAGEFILE_FOLDER + "\\ST_RUN_SI.xlsx")
        elif file_key in "ST_WALK_SI":
            StageFile = pd.read_excel(STAGEFILE_FOLDER + "\\ST_WALK_SI.xlsx")
            
        for file in exls_file_list:
            print(file)
            for stage_idx in range(len(StageFile["file_name"])):
                stage_file = os.path.splitext(
                    os.path.basename(StageFile["file_name"][stage_idx]))[0]
                
                if stage_file in file:
                    print(stage_file)
                    START = {
                        "LEFT":{
                            "START": (StageFile["left_start_1"][stage_idx]),
                            "END": (StageFile["left_end_1"][stage_idx])
                                },
                        "RIGHT":{
                            "START": (StageFile["right_start_1"][stage_idx]),
                            "END": (StageFile["right_end_1"][stage_idx])
                                }
                        }
                    # print(right_end)
                    break
                
            file_name_with_ext = os.path.basename(file)  # → "S1_GOLF_1 FP&SI extra v2.xlsx"
           
            
            if file_key in file_name_with_ext:
                print(file)
                if START:
                    all_correlation_results = func.process_file(file,
                                                                CAL_FORMULAS=CAL_FORMULAS,
                                                                COL_NAME_MAP=COL_NAME_MAP,
                                                                PERSON_CORR=PERSON_CORR,
                                                                PLOTS_CONFIG=PLOTS_CONFIG,
                                                                OUTPUT_DIRECTORY=OUTPUT_DIRECTORY,
                                                                FILTER_COL=FILTER_COL,
                                                                NeedFilter=True,
                                                                START=START
                                                                )
                else:
                    all_correlation_results = func.process_file(file,
                                                                CAL_FORMULAS=CAL_FORMULAS,
                                                                COL_NAME_MAP=COL_NAME_MAP,
                                                                PERSON_CORR=PERSON_CORR,
                                                                PLOTS_CONFIG=PLOTS_CONFIG,
                                                                OUTPUT_DIRECTORY=OUTPUT_DIRECTORY,
                                                                FILTER_COL=FILTER_COL,
                                                                NeedFilter=True
                                                                )
                   
                all_results_data[file_name_with_ext] = all_correlation_results
    
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
        summary_excel_filename = str(idx) + "_correlation_summary_from_defaultdict.xlsx"
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






















