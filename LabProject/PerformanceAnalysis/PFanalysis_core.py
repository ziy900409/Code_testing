# -*- coding: utf-8 -*-
"""
Created on Sat May 24 16:41:25 2025

@author: User
"""
import sys
# 路徑改成你放自己code的資料夾
# sys.path.append(r"D:\BenQ_Project\gitgit\Code_testing\LabProject\PerformanceAnalysis")
sys.path.append(r"D:\git\Code_testing\LabProject\PerformanceAnalysis")
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import argrelextrema
# from numpy.linalg import norm
# from scipy.interpolate import interp1d

import pre_processing as pre
import calculate_func as cal
import emg_function as emg

# %%

# # === in game parameters ===
# DPI = 800
# sensitivity = 1.0
# yaw = 0.022  # CS2 預設值
# yaw = 0.07 # Valorant 靈敏度

# # === motion capture system setting ===
# # Define filter cutoffs (Hz). Use None or 0 to disable for a specific type.
# marker_freq_cutoff = 20.0
# fp_freq_cutoff = 30.0
# analog_freq_cutoff = None # Example: Don't filter general analog
# rename_markers = {'MOS1': 'M1',
#                 'MOS2': 'M2',
#                 'MOS3': 'M3',
#                 'MOS4': 'M4',
#                 'RHO': 'R.Shoulder',
#                 'RSHO': 'R.Shoulder',
#                 'RUEL': 'R.Elbow.Lat',
#                 'RUEM': 'R.Elbow.Med',
#                 'RUS': 'R.Wrist.Uln',
#                 'RRS': 'R.Wrist.Rad',
#                 'RTB1': 'R.Thumb1',
#                 'RTB2': 'R.Thumb2',
#                 'RTB3': 'R.Thumb3',
#                 'RID1': 'R.I.Finger1',
#                 'RID2': 'R.I.Finger2',
#                 'RID3': 'R.I.Finger3',
#                 'RMD1': 'R.M.Finger1',
#                 'RMD2': 'R.M.Finger2',
#                 'RMD3': 'R.M.Finger3',
#                 'RRG1': 'R.R.Finger1',                
#                 'RRG2': 'R.R.Finger2',
#                 'RLT1': 'R.P.Finger1',
#                 'RLT2': 'R.P.Finger2',                
#                 }
# # Remove prefixes
# remove_prefixes = ["S03", "MarkerSet:"] 

# # === EMG Setting ===
# down_freq = 1000
# c = 0.802
# # 帶通濾波頻率
# bandpass_cutoff = [20/0.802, 450/0.802]
# # 低通濾波頻率
# lowpass_freq = 10/c
# # 設定移動平均數與移動均方根之參數
# # 更改window length, 更改overlap length
# time_of_window = 0.1 # 窗格長度 (單位 second)
# overlap_len = 0.5 # 百分比 (%)
# # 設定 notch filter cutoff frequency

# notch_cutoff = [[59, 61],
#                 [295.5, 296.5],
#                 [369.5, 370.5],
#                 [179, 181],
#                 [299, 301],
#                 [419, 421],
#                 ]

# c3d_notch_cutoff = [[49, 51],
#                     [99.5, 100.5],
#                     [149.5, 150.5],
#                     [199.5, 200.5],
#                     [249.5, 250.5],
#                     [299.5, 300.5],
#                     [349.5, 350.5],
#                     [295, 297],
#                     [369, 371],
#                     [73, 75],
#                     [399, 401]
#                     ]

# csv_recolumns_name = {'Mini sensor 1: EMG 1': 'Extensor Carpi Radialis',
#                      'Mini sensor 2: EMG 2': 'Flexor Carpi Radialis',
#                      'Mini sensor 3: EMG 3': 'Triceps Brachii',
#                      'Quattro sensor 4: EMG.A 4': 'Extensor Carpi Ulnaris', 
#                      'Quattro sensor 4: EMG.B 4': '1st Dorsal Interosseous', 
#                      'Quattro sensor 4: EMG.C 4': 'Abductor Digiti Quinti', 
#                      'Quattro sensor 4: EMG.D 4': 'Extensor Indicis',
#                      'Avanti sensor 5: EMG 5': 'Biceps Brachii'}

# c3d_recolumns_name = {'ExtRad': 'Extensor Carpi Radialis',
#                       'FleRad': 'Flexor Carpi Radialis',
#                      'Triceps': 'Triceps Brachii',
#                       'Triceps': 'Triceps Brachii',
#                      'ExtUlnar': 'Extensor Carpi Ulnaris',
#                      'ExtUlnar': 'Extensor Carpi Ulnaris',
#                      'DorInter': '1st Dorsal Interosseous', 
#                      'AbdDigMin': 'Abductor Digiti Quinti',
#                      #' AbdDigMin.IM EMG6': 'Abductor Digiti Quinti',
#                      'ExtInd': 'Extensor Indicis',
#                      'Biceps': 'Biceps Brachii',
#                      }

# c3d_analog_cha = ["ExtRad", "FleRad",
#                   "ExtUlnar", "DorInter", "AbdDigMin", "ExtInd",
#                   "Biceps", "Triceps"
#                   ]

# muscle_name = ['Extensor Carpi Radialis', 'Flexor Carpi Radialis', 'Triceps Brachii',
#                'Extensor Carpi Ulnaris', '1st Dorsal Interosseous', 
#                'Abductor Digiti Quinti', 'Extensor Indicis', 'Biceps Brachii']

# # %%

# EMG_CONFIG = {
#     "DEFAULT_DOWNSAMPLE_FREQ": 1000,
#     "DEFAULT_BANDPASS_CUTOFF": [20, 450],
#     "DEFAULT_LOWPASS_FREQ": 6,
#     "DEFAULT_CSV_NOTCH_CUTOFF_LIST": notch_cutoff, # 假設 50Hz 工頻
#     "DEFAULT_C3D_NOTCH_CUTOFF_LIST": c3d_notch_cutoff, # 假設 60Hz 工頻
#     "DEFAULT_CSV_RECOLUMNS_NAME": csv_recolumns_name, # 範例
#     "DEFAULT_C3D_RECOLUMNS_NAME": c3d_recolumns_name, # 範例
#     "EMG_CHANNEL_IDENTIFIER": muscle_name, # 用於辨識 EMG 頻道的關鍵字
#     "REMOVE_PREFIXES": remove_prefixes
# }

# motion_config = {
#     "REMOVE_PREFIXES": remove_prefixes, # 需要移除的前綴字
#     "RENAME_MARKERS": rename_markers, # 重新命名的 Marmer name
#     "CUTOFF_FREQUENCY": marker_freq_cutoff, # 截止頻率
#     "FP_CUTOFF_FREQUENCY": fp_freq_cutoff, # 力版的截止頻率
#     "ANA_CUTOFF_FREQUENCY": analog_freq_cutoff, # Analog 截止頻率
#     "BUTTERWORTH_ORDER": 4 # Standard 4th order Butterworth

# }

# # %%
# data_path = r"C:\Users\Hsin.YH.Yang\Downloads\Dynamic measurement test c3d\S00_fatigue_01_AddEMG.c3d"
# processed_data, metadata = pre.read_c3d(
#     data_path,
#     process_forceplate=False,
#     process_analog=True,
#     prefix_to_remove = motion_config.get("REMOVE_PREFIXES", None),
#     rename_map = motion_config.get("RENAME_MARKERS", None),
#     marker_cutoff = motion_config.get("MARKER_CUTOFF_FREQUENCY", 20),
#     fp_cutoff = motion_config.get("FP_CUTOFF_FREQUENCY", 50),
#     analog_cutoff = motion_config.get("ANA_CUTOFF_FREQUENCY", None),
#     filter_order = motion_config.get("BUTTERWORTH_ORDER", 4),
# )

# %%   
def pro_main(data_path, motion_config, emg_config,
             sens=1, yaw_range = 13, pitch_range = 10):
    # data_path = r"D:\BenQ_Project\01_UR_lab\00_BQE\2025_06 Lab Opening\motion\S1_Post_Spider30_EC.c3d"
    # motion_config = MOTION_CONFIG
    # emg_config = EMG_CONFIG
    try:
        processed_data, metadata = pre.read_c3d(
            data_path,
            process_forceplate=False,
            process_analog=True,
            prefix_to_remove = motion_config.get("REMOVE_PREFIXES", None),
            rename_map = motion_config.get("RENAME_MARKERS", None),
            marker_cutoff = motion_config.get("MARKER_CUTOFF_FREQUENCY", 20),
            fp_cutoff = motion_config.get("FP_CUTOFF_FREQUENCY", 50),
            analog_cutoff = motion_config.get("ANA_CUTOFF_FREQUENCY", None),
            filter_order = motion_config.get("BUTTERWORTH_ORDER", 4),
        )
        
        print("\n--- Output Data Structure ---")
        if processed_data:
            print("Processed Data Keys:", processed_data.keys())
            if "markers" in processed_data:
                print("  Marker Keys:", list(processed_data["markers"].keys()))      
            if "FP" in processed_data:
                print("  Force Plate Keys:", list(processed_data["FP"].keys()))
            if "analog" in processed_data:
                print("  Analog Keys:", list(processed_data["analog"].keys()))
    
        print("\n--- Metadata Structure ---")
        if metadata:
            print("Metadata Keys:", metadata.keys())
            if "motion_info" in metadata:
                print("  Motion Info:", metadata["motion_info"])
            if "fp_info" in metadata:
                print("  FP Info:", metadata["fp_info"])
            if "analog_info" in metadata:
                print("  Analog Info:", metadata["analog_info"])
    
    except FileNotFoundError:
        print(f"\nError: Example C3D file not found at '{data_path}'. Please update the path.")
    except ImportError:
        print("\nError: ezc3d or scipy library not found. Please install them: pip install ezc3d scipy")
    except Exception as e:
        print(f"\nAn unexpected error occurred during example execution: {e}")
        
    # 將單位從mm轉換成視角
    df = pre.ConverUnit2Angle(processed_data, metadata,
                              marker="R.I.Finger3",
                              DPI=800,
                              sens=sens,
                              yaw=0.022)
    # 2.1. 找出每一次目標擊殺的開槍數 -> 找出Z axis local minimal
    # 2.1.1. 以滑鼠點擊次數計算，使用Z軸局部最小值，如果兩次Z軸局部最小值的視角差
    #         小於某個閾值，則視為仍在瞄準同一個目標
    final_indices, final_data = pre.find_Zaxis_min_with_baseline(
            df=df,
            use_baseline_removal=True,   # <--- 啟用基線移除
            baseline_window_length=101, # <--- 調整窗口大小試試
            baseline_polyorder=3,
            order=5,
            min_frame_gap=8,
            min_z_diff=0.2,
            z_processed_threshold=-1, # <--- 試用處理後 Z 值的門檻
            show=True,
            showVel=True
        )
    final_indices = final_data["Frame"].tolist()
    # 2.1.2. 找出完成擊殺的 frame 以及上一個視角大於閾值的視角位置
    grouped_df = pre.findZminGroup(df, final_indices)
    # 2.2. 找出從中心出發的開槍軌跡
    excldueCen_grouped_df = pre.excludeCenter(df,
                                          grouped_df,
                                          yaw_range = yaw_range,
                                          pitch_range = pitch_range,
                                          show = True)
    # 只取一槍命中的數值
    oneshot_df = excldueCen_grouped_df[excldueCen_grouped_df['Shot Count']==1]
    oneshot_df = oneshot_df[oneshot_df['Direction Quadrant'].isin(['Q2', 'Q3'])]
    # oneshot_df = excldueCen_grouped_df[excldueCen_grouped_df['Direction Quadrant'].isin(['Q1', 'Q4'])]
    
    
    # 做標準化處理
    if not excldueCen_grouped_df.empty:
        try:
            standardized_speeds = pre.standardize_group_signals(
                df=df,
                filtered_grouped_df=oneshot_df,
                signal_column_name='angle_speed_dps', # 指定要標準化的欄位
                target_length=101,                   # 指定目標長度
                start_col='NEW Frame Start',             # 指定起始幀欄位
                end_col='Frame End',                 # 指定結束幀欄位
                group_id_col='Group ID'              # 指定群組ID欄位
            )
    
        except KeyError as e:
            print(f"執行標準化時出錯：{e}")
        except ImportError:
            print("錯誤：需要安裝 scipy 庫才能執行插值。請運行 pip install scipy")
    else:
          print("沒有可供標準化的群組 (filtered_grouped_df is empty)。")
          
    # === 處理 EMG ===
    fft_results = emg.calculate_fft_for_emg(data_path, emg_config)
    print("C3D 處理結果:")
    for ch_data in fft_results.get("filename", []):
        print(f"  頻道: {fft_results.get('amplitudes').keys()}")
    
    emg_results = emg.process_emg_core(data_path, # 檔案物件的 path
                                       emg_config, # 包含所有處理參數的字典
                                       smoothing_method="lowpass", # smoothing 參數
                                       original_filename=None)
    
    return df, metadata, excldueCen_grouped_df, standardized_speeds, fft_results, emg_results

# %%

def fatigue_main(data_path, motion_config, emg_config):
    try:
        processed_data, metadata = pre.read_c3d(
            data_path,
            process_forceplate=False,
            process_analog=True,
            prefix_to_remove = motion_config.get("REMOVE_PREFIXES", None),
            rename_map = motion_config.get("RENAME_MARKERS", None),
            marker_cutoff = motion_config.get("MARKER_CUTOFF_FREQUENCY", 20),
            fp_cutoff = motion_config.get("FP_CUTOFF_FREQUENCY", 50),
            analog_cutoff = motion_config.get("ANA_CUTOFF_FREQUENCY", None),
            filter_order = motion_config.get("BUTTERWORTH_ORDER", 4),
        )
        
        print("\n--- Output Data Structure ---")
        if processed_data:
            print("Processed Data Keys:", processed_data.keys())
            if "markers" in processed_data:
                print("  Marker Keys:", list(processed_data["markers"].keys()))      
            if "FP" in processed_data:
                print("  Force Plate Keys:", list(processed_data["FP"].keys()))
            if "analog" in processed_data:
                print("  Analog Keys:", list(processed_data["analog"].keys()))
    
        print("\n--- Metadata Structure ---")
        if metadata:
            print("Metadata Keys:", metadata.keys())
            if "motion_info" in metadata:
                print("  Motion Info:", metadata["motion_info"])
            if "fp_info" in metadata:
                print("  FP Info:", metadata["fp_info"])
            if "analog_info" in metadata:
                print("  Analog Info:", metadata["analog_info"])
    
    except FileNotFoundError:
        print(f"\nError: Example C3D file not found at '{data_path}'. Please update the path.")
    except ImportError:
        print("\nError: ezc3d or scipy library not found. Please install them: pip install ezc3d scipy")
    except Exception as e:
        print(f"\nAn unexpected error occurred during example execution: {e}")
    # === 處理 EMG ===
    results_c3d = emg.calculate_fft_for_emg(data_path, emg_config)
    print("C3D 處理結果:")
    for ch_data in results_c3d.get("filename", []):
        print(f"  頻道: {results_c3d.get('amplitudes').keys()}")
    
    emg_results = emg.process_emg_core(data_path, # 檔案物件的 path
                                       emg_config, # 包含所有處理參數的字典
                                       smoothing_method="lowpass", # smoothing 參數
                                       original_filename=None)
    return results_c3d, emg_results
