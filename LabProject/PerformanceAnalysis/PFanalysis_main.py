# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:38:36 2025

@author: Hsin.YH.Yang
"""

import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"D:\BenQ_Project\gitgit\Code_testing\LabProject\PerformanceAnalysis")
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import argrelextrema
# from numpy.linalg import norm
# from scipy.interpolate import interp1d

import pre_processing as pre
import calculate_func as cal


# %%

rename_markers = {'MOS1': 'M1',
                'MOS2': 'M2',
                'MOS3': 'M3',
                'MOS4': 'M4',
                'RHO': 'R.Shoulder',
                'RSHO': 'R.Shoulder',
                'RUEL': 'R.Elbow.Lat',
                'RUEM': 'R.Elbow.Med',
                'RUS': 'R.Wrist.Uln',
                'RRS': 'R.Wrist.Rad',
                'RTB1': 'R.Thumb1',
                'RTB2': 'R.Thumb2',
                'RTB3': 'R.Thumb3',
                'RID1': 'R.I.Finger1',
                'RID2': 'R.I.Finger2',
                'RID3': 'R.I.Finger3',
                'RMD1': 'R.M.Finger1',
                'RMD2': 'R.M.Finger2',
                'RMD3': 'R.M.Finger3',
                'RRG1': 'R.R.Finger1',                
                'RRG2': 'R.R.Finger2',
                'RLT1': 'R.P.Finger1',
                'RLT2': 'R.P.Finger2',                
                }
# Remove prefixes
remove_prefixes = ["S06", "MarkerSet:"]   
# === 參數設定 ===
DPI = 800
sensitivity = 1.0
yaw = 0.022  # CS2 預設值

# Define filter cutoffs (Hz). Use None or 0 to disable for a specific type.
marker_freq_cutoff = 20.0
fp_freq_cutoff = 30.0
analog_freq_cutoff = None # Example: Don't filter general analog

# %%

data_path = r"D:/BenQ_Project/01_UR_lab/2024_11 Shanghai CS Major/1. Motion/Major_weight/S06/20241206/S06_SpiderShot_S1_1.c3d"
data_path = r"D:\BenQ_Project\01_UR_lab\2025_02 Asymmetry\1.Motion\1.Vicon\S05\250318\S05_LargeFlick_I_3.c3d"
"""
2. 計算
    2.1. 找出每一次目標擊殺的開槍數 -> 找出Z axis local minimal
        2.1.1. 以滑鼠點擊次數計算，使用Z軸局部最小值，如果兩次Z軸局部最小值的視角差
                小於某個閾值，則視為仍在瞄準同一個目標
        2.1.2. 找出完成擊殺的 frame 以及上一個視角大於閾值的視角位置
                data format
            	GroupID   Frames          Shot Count   Frame Start   Frame End   Frame Span
                -------   --------------  -----------  ------------  ----------  -----------
                1       [53.0, 71.0]          1           53.0         71.0        18.0
    2.2. 找出從中心出發的開槍軌跡
    2.3. 定義開槍軌跡: 多重條件
        2.3.1. 只有速度方向往目標方向才算開始
        2.3.2. 速度達到一定閾值？ 速度與目標方向的偏差角度？
    2.4. 計算初始偏移角度
"""

try:
    processed_data, metadata = pre.read_c3d(
        data_path,
        process_forceplate=True,
        process_analog=True,
        prefix_to_remove=remove_prefixes,
        rename_map=rename_markers,
        marker_cutoff=marker_freq_cutoff,
        fp_cutoff=fp_freq_cutoff,
        analog_cutoff=analog_freq_cutoff,
        filter_order=4 # Standard 4th order Butterworth
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
df = pre.ConverUnit2Angle(processed_data, metadata)
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
                                          yaw_range = 15,
                                          pitch_range = 10,
                                          show = True)

# 做標準化處理
if not excldueCen_grouped_df.empty:
    try:
        standardized_speeds = pre.standardize_group_signals(
            df=df,
            filtered_grouped_df=excldueCen_grouped_df,
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


# --- 繪製 mean std cloud ---
# 假設 standardized_speeds1 和 standardized_speeds2 是兩個包含標準化速度信號的字典
# 假設 target_length = 101
standardized_speeds1 = standardized_speeds
standardized_speeds2 = standardized_speeds

# 示例 2: 繪製兩個數據集進行比較
if standardized_speeds1 and standardized_speeds2:
      pre.plot_standardized_signals_cloud_compare(
          signals_dict1=standardized_speeds1,
          target_length=101,
          signals_dict2=standardized_speeds2, # 提供第二個字典
          title="Compare the mean speed ± standard deviation of two data sets",
          label1='Data A',
          label2='Data B',        # 為第二個數據集提供標籤
          ylabel="Speed (°/s)",
          color_index1=0,         # 第一個用顏色 0
          color_index2=1          # 第二個用顏色 1
      )
else:
      print("至少需要一個有效的標準化信號字典才能繪圖。")
      
# %%
"""
2.2. 計算
    2.2.1. 指標
        o. (廢棄)擊殺數, 命中率？
        a. Throughput (Mouse Travel Efficiency): 
        b. Mouse Speed (°/s): 找出整段時間內的最大值 or 平均值，單位換算成視角
        c. Initial Move Angle: 初始 5 個 frame 的移動方向與最終擊殺目標位置的視角差
            修改條件: 1. 排除所有Initial Move Angle大於45度的trial
                     2. Frame Span 要大於 20
        d. Full Path Time: 
            使用 Frame Span/descriptions['motion info']['frame_rate']
        e. Reaction Time: 從這次目標擊殺到某個 frame 移動速度超過一個閾值 
            扣掉直接回中的反應時間
        i. 一槍擊殺的次數, 二槍, 三槍...
        j. 超過目標的次數， 還沒到目標就開槍的次數
        k. Mouse Travel Efficiency: idea path/real path
    2.2.2. 不同方向的計算: 全部方向綜合, 分四個方向 (四象限)
"""

# === b. Mouse Speed (°/s) ===
# === x. 量化速度 ===
# 找出整段時間內的最大值 or 平均值，單位換算成視角
# 只包含第一次射擊的視角速度
# 1. 先新增 mean_speed 列，预设值为 NaN
excldueCen_grouped_df["mean_speed"] = np.nan
excldueCen_grouped_df["max_speed"] = np.nan

for num in range(len(excldueCen_grouped_df)):
    trail_start = int(excldueCen_grouped_df["Frames"][num][0])
    trail_end = int(excldueCen_grouped_df["Frames"][num][1])
    trial_speed = df.loc[trail_start:trail_end, "speed"]
    excldueCen_grouped_df.at[num, "mean_speed"] = trial_speed.mean()
    excldueCen_grouped_df.at[num, "max_speed"] = trial_speed.max()
    

# # === c. Initial Move Angle: ===
excldueCen_grouped_df['Initial Move Angle (°)']

# mean_initial_move_angle = np.mean(excldueCen_grouped_df['Initial Move Angle (°)'])

# # === d. Full Path Time (單位 Second)===
path_time = np.mean(excldueCen_grouped_df["Frame Span"])\
    /metadata['motion_info']['frame_rate']

# === e. Reaction Time ===
# 只計算從中心出發，並且 initial move angle 小於 45 度


# === k. Mouse Travel Efficiency
# Mouse Travel Efficiency: idea path/real path


# 呼叫函數計算效率
grouped_df = cal.calculate_trajectory_efficiency(df,
                                                 excldueCen_grouped_df)

# %%
"""
執行檔案模式
1. Spider shot frist run
2. Grid shot 180 second
3. Spider shot second run 

預計產出的圖片
1. Spider shot
    1.1. mean std colud (normalize)
    1.2. muscle activity level?
    
2. Grid Shot 
    2.1. median frequency with time (幾條肌肉？)
    2.2. muscle activity level with time (幾條肌肉？)
    
3. 六角圖指標 (需要再區分類別)
    3.1. 疲勞 (median frequency, muscle activity level 的權重)
    3.2. 定位能力 (一次定位: 一次定位完成的百分率)
    3.3. 微調能力 (最高速後減速定位)
    3.4. 第一槍 miss 後，再命中目標的時間 (第二槍到最後一槍的時間)
    3.5. 移動速度 (平均速度、最高速度、權重)
    3.6. 減速能力
    
4. 表格
    
"""














































