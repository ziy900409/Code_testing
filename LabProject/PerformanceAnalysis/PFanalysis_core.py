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
def pro_main(data_path, motion_config, emg_config):
    
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
                              marker="R.I.Finger3"
                              )
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
                                          yaw_range = 10,
                                          pitch_range = 10,
                                          show = True)
    # 只取一槍命中的數值
    oneshot_df = excldueCen_grouped_df[excldueCen_grouped_df['Shot Count']==1]
    
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
