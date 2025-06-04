# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:38:36 2025

@author: Hsin.YH.Yang
"""

import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"D:\BenQ_Project\gitgit\Code_testing\LabProject\PerformanceAnalysis")
# sys.path.append(r"D:\git\Code_testing\LabProject\PerformanceAnalysis")
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import argrelextrema
# from numpy.linalg import norm
# from scipy.interpolate import interp1d

import pre_processing as pre
import calculate_func as cal
import emg_function as emg
import PFanalysis_core as core
import plot_table as ta

# %% parameters setting

# === in game parameters ===
DPI = 800
sensitivity = 1.0
yaw = 0.022  # CS2 預設值
yaw = 0.07 # Valorant 靈敏度

# === motion capture system setting ===
# Define filter cutoffs (Hz). Use None or 0 to disable for a specific type.
marker_freq_cutoff = 20.0
fp_freq_cutoff = 30.0
analog_freq_cutoff = None # Example: Don't filter general analog
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
remove_prefixes = ["S03", "MarkerSet:"] 

# === EMG Setting ===
down_freq = 1000
c = 0.802
# 帶通濾波頻率
bandpass_cutoff = [20/0.802, 450/0.802]
# 低通濾波頻率
lowpass_freq = 10/c
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.1 # 窗格長度 (單位 second)
overlap_len = 0.5 # 百分比 (%)
# 設定 notch filter cutoff frequency

notch_cutoff = [[59, 61],
                [295.5, 296.5],
                [369.5, 370.5],
                [179, 181],
                [299, 301],
                [419, 421],
                ]

c3d_notch_cutoff = [[49, 51],
                    [99.5, 100.5],
                    [149.5, 150.5],
                    [199.5, 200.5],
                    [249.5, 250.5],
                    [299.5, 300.5],
                    [349.5, 350.5],
                    [295, 297],
                    [369, 371],
                    [73, 75],
                    [399, 401]
                    ]

csv_recolumns_name = {'Mini sensor 1: EMG 1': 'Extensor Carpi Radialis',
                     'Mini sensor 2: EMG 2': 'Flexor Carpi Radialis',
                     'Mini sensor 3: EMG 3': 'Triceps Brachii',
                     'Quattro sensor 4: EMG.A 4': 'Extensor Carpi Ulnaris', 
                     'Quattro sensor 4: EMG.B 4': '1st Dorsal Interosseous', 
                     'Quattro sensor 4: EMG.C 4': 'Abductor Digiti Quinti', 
                     'Quattro sensor 4: EMG.D 4': 'Extensor Indicis',
                     'Avanti sensor 5: EMG 5': 'Biceps Brachii'}

c3d_recolumns_name = {'ExtRad': 'Extensor Carpi Radialis',
                      'FleRad': 'Flexor Carpi Radialis',
                     'Triceps': 'Triceps Brachii',
                      'Triceps': 'Triceps Brachii',
                      'ExtUlnar': 'Extensor Carpi Ulnaris',
                      'ExtUlnar': 'Extensor Carpi Ulnaris',
                     'DorInter': '1st Dorsal Interosseous', 
                     'AbdDigMin': 'Abductor Digiti Quinti',
                     #' AbdDigMin.IM EMG6': 'Abductor Digiti Quinti',
                     'ExtInd': 'Extensor Indicis',
                     'Biceps': 'Biceps Brachii',
                     }

c3d_analog_cha = ["ExtRad", "FleRad",
                  "ExtUlnar", "DorInter", "AbdDigMin", "ExtInd",
                  "Biceps", "Triceps"
                  ]

muscle_name = ['Extensor Carpi Radialis', 'Flexor Carpi Radialis', 'Triceps Brachii',
               'Extensor Carpi Ulnaris', '1st Dorsal Interosseous', 
               'Abductor Digiti Quinti', 'Extensor Indicis', 'Biceps Brachii']

# %%

EMG_CONFIG = {
    "DEFAULT_DOWNSAMPLE_FREQ": 1000,
    "DEFAULT_BANDPASS_CUTOFF": [20, 450],
    "DEFAULT_LOWPASS_FREQ": 6,
    "DEFAULT_CSV_NOTCH_CUTOFF_LIST": notch_cutoff, # 假設 50Hz 工頻
    "DEFAULT_C3D_NOTCH_CUTOFF_LIST": c3d_notch_cutoff, # 假設 60Hz 工頻
    "DEFAULT_CSV_RECOLUMNS_NAME": csv_recolumns_name, # 範例
    "DEFAULT_C3D_RECOLUMNS_NAME": c3d_recolumns_name, # 範例
    "EMG_CHANNEL_IDENTIFIER": muscle_name, # 用於辨識 EMG 頻道的關鍵字
    "REMOVE_PREFIXES": remove_prefixes
}

MOTION_CONFIG = {
    "REMOVE_PREFIXES": remove_prefixes, # 需要移除的前綴字
    "RENAME_MARKERS": rename_markers, # 重新命名的 Marmer name
    "CUTOFF_FREQUENCY": marker_freq_cutoff, # 截止頻率
    "FP_CUTOFF_FREQUENCY": fp_freq_cutoff, # 力版的截止頻率
    "ANA_CUTOFF_FREQUENCY": analog_freq_cutoff, # Analog 截止頻率
    "BUTTERWORTH_ORDER": 4 # Standard 4th order Butterworth

}


# config = APP_CEMG_CONFIGONFIG

# %%
"""
單一滑鼠

選擇檔案模式
mouse A
1. before file: spider shot 30s
2. fatigue test: spider shot 180s
3. after file: spider shot

mouse B
1. before file: spider shot 30s
2. fatigue test: spider shot 180s
3. after file: spider shot

mouse C
1. before file: spider shot 30s
2. fatigue test: spider shot 180s
3. after file: spider shot

mouse D
1. before file: spider shot 30s
2. fatigue test: spider shot 180s
3. after file: spider shot
"""


# %%

pre_path = r"D:/BenQ_Project/01_UR_lab/2024_11 Shanghai CS Major/1. Motion/Major_weight/S06/20241206/S06_SpiderShot_S1_1.c3d"
fatigue_path = r"D:\BenQ_Project\01_UR_lab\2024_11 Shanghai CS Major\1. Motion\Major_weight\S06\20241206\S06_SpiderShot_S2_3.c3d"
pos_path = r"D:\BenQ_Project\01_UR_lab\2024_11 Shanghai CS Major\1. Motion\Major_weight\S06\20241206\S06_SpiderShot_S3_1.c3d"

# pre_path = r"D:\Hsin\BenQ\testfile\PFanalysis\mouse A\S01_SpiderShot_ZA1_3.c3d"
# fatigue_path = r"D:\Hsin\BenQ\testfile\PFanalysis\mouse A\S07_GridShot_HS_1.c3d"
# pos_path = r"D:\Hsin\BenQ\testfile\PFanalysis\mouse A\S01_SpiderShot_ZA2_1.c3d"
# pre_path = r"E:\testfile\PFanalysis\mouse A\S01_SpiderShot_ZA1_3.c3d"
# fatigue_path = r"E:\testfile\PFanalysis\mouse A\S07_GridShot_HS_1.c3d"
# pos_path = r"E:\testfile\PFanalysis\mouse A\S01_SpiderShot_ZA2_1.c3d"


# pre_path = r"C:\Users\Hsin.YH.Yang\Downloads\Dynamic measurement test c3d\S00_pos_01_AddEMG.c3d"
# fatigue_path = r"C:\Users\Hsin.YH.Yang\Downloads\Dynamic measurement test c3d\S00_fatigue_01_AddEMG.c3d"
# pos_path = r"C:\Users\Hsin.YH.Yang\Downloads\Dynamic measurement test c3d\S00_pre_01_AddEMG.c3d"


pre_df, pre_metadata, pre_excldueCen_df, pre_standardized_speeds, pre_fft_results, pre_emg_results = core.pro_main(pre_path, MOTION_CONFIG, EMG_CONFIG)
fati_results_c3d, fati_emg_results = core.fatigue_main(fatigue_path, MOTION_CONFIG, EMG_CONFIG)
pos_df, pos_metadata, pos_excldueCen_df, pos_standardized_speeds, pos_fft_results, pos_emg_results = core.pro_main(pos_path, MOTION_CONFIG, EMG_CONFIG)

# %% 單個結果
"""
單支滑鼠
fig:
    A. 時序圖
        1. 視角移動速度曲線
        2. 肌肉活化時序圖
    B. 柱狀圖
        1. 肌肉活化程度斜率
        2. 中頻率斜率
    
"""

# 將多個結果放入列表
all_fft_data = [pre_fft_results, pos_fft_results]
all_configs = [EMG_CONFIG, EMG_CONFIG]
labels = ['55g', '60g'] # 可選的自定義標籤

oneshot_df = pre_excldueCen_df[pre_excldueCen_df['Shot Count']==1]

pos_oneshot_df = pos_excldueCen_df[pos_excldueCen_df['Shot Count']==1]



# 繪製瞄準速度取線
pre.plot_standardized_signals_cloud_compare(
    datasets=[pre_standardized_speeds, pos_standardized_speeds],
    target_length=101,
    title="Comparison of Mean ± Std Dev Clouds",
    xlabel="Normalized Time (%)",
    ylabel="Signal Value (°/s or other units)",
    # title="Sine Group Mean ± Std Dev",
    labels=['55g', "60g"],
    # labels=None,                
    # color_indices=[0, 2, 4]          
      )

# 繪圖 median frequency
emg.plot_multiple_mdf_over_time(list_of_fft_results_data=[fati_results_c3d], 
                                configs=[EMG_CONFIG], 
                                max_subplot_cols=2, 
                                title_name="Muscle Fatigue Analysis",
                                dataset_labels=['fatigue'],
                                selected_keys = [
                                'Biceps.IM EMG8', 'Triceps.IM EMG9', 'DorInter_1st.IM EMG4', 'AbdDigMin.IM EMG5'
                                ]
                                )

emg.plot_multiple_emg_data_over_time(
    list_of_emg_data = [fati_emg_results],
    configs=[EMG_CONFIG],
    max_subplot_cols=2,
    title_name=None,
    dataset_labels=['fatigue'],
    y_axis_label="Averaged EMG Amplitude (AU)",
    show_trendline=True, # New parameter to control trendline plotting
    selected_keys = [
        'Biceps.IM EMG8', 'Triceps.IM EMG9', 'DorInter_1st.IM EMG4', 'AbdDigMin.IM EMG5'
        ]
)

interpolated_data = emg.process_emg_data_with_direction(oneshot_df,
                                                        pre_emg_results,
                                                        dataset_labels=None,
                                                        selected_keys = None)


interpolated_data_1 = emg.process_emg_data_with_direction(pos_oneshot_df,
                                                          pos_emg_results,
                                                            dataset_labels=None,
                                                            selected_keys = None)


# %% 繪製肌肉活化程度曲線
    
plotter_instance = emg.EMGPlotter(interpolated_data,
                                  target_length=141)

# --- 調用新的多行雲圖繪製功能 ---

# 示例 1: 只有一組肌肉 (一行，兩個子圖 Left/Right)
# muscles_to_plot_fig1 = {
#     "Arm Muscles": ['Biceps.IM EMG8', 'Triceps.IM EMG9', 'NonExistentMuscle'] # 包含一個不存在的肌肉以測試過濾
# }
# print(f"\nPlotting cloud summary for: {muscles_to_plot_fig1}")
# plotter_instance.plot_emg_summary_by_direction_with_cloud(
#     muscle_groups_to_plot=muscles_to_plot_fig1,
#     main_title="EMG Activity: Arm Muscles (X: -40 to 100)",
#     share_y_axis=True
# )

# 示例 2: 兩組肌肉 (兩行，每行兩個子圖 Left/Right)
muscles_to_plot_fig2 = {
    "Upper Limb": ['Biceps.IM EMG8', 'Triceps.IM EMG9'],
    "Hand Intrinsic": ['DorInter.IM EMG4', 'AbdDigMin.IM EMG5']
}
print(f"\nPlotting cloud summary for: {muscles_to_plot_fig2}")
plotter_instance.plot_emg_summary_by_direction_with_cloud(
    muscle_groups_to_plot=muscles_to_plot_fig2,
    main_title="EMG Activity: Upper Limb & Hand (X: -40 to 100)",
    share_y_axis=False # 嘗試 share_y_axis=False 來看看效果
)

# 示例 3: 包含空肌肉列表的行 (應跳過該行)
# muscles_to_plot_fig3 = {
#     "Valid Arm Muscles": ['Biceps.IM EMG8'],
#     "Empty Hand Group": [],
#     "Another Valid Group": ['Triceps.IM EMG9']
# }
# print(f"\nPlotting cloud summary for: {muscles_to_plot_fig3}")
# plotter_instance.plot_emg_summary_by_direction_with_cloud(
#     muscle_groups_to_plot=muscles_to_plot_fig3,
#     main_title="EMG Activity: Testing Empty Group (X: -40 to 100)",
#     # share_y_axis=True
# )
# %% 繪製柱狀圖

group1 = pre_fft_results["MedianFreq_Slope"]
group2 = pos_fft_results["MedianFreq_Slope"]

ta.plot_median_freq_slope_comparison(group1, group2,
                                     selected_keys=[
                                         "ExtRad.IM EMG1", "Triceps.IM EMG9",
                                         "Biceps.IM EMG8", 'ExtRad.IM EMG1'
                                         ],
                                     title="Median Frequency Slope Comparison",
                                     label_list=["pre", "pos"],
                                     show_values=False)

group1 = pre_emg_results["Amplitudes_Slope"]
group2 = pos_emg_results["Amplitudes_Slope"]
ta.plot_median_freq_slope_comparison(group1, group2,
                                     selected_keys=[
                                         "ExtRad.IM EMG1", "Triceps.IM EMG9",
                                         "Biceps.IM EMG8", 'ExtRad.IM EMG1'
                                         ],
                                     title="Muscle Activation Slope Comparison",
                                     ylabel="Muscle Activation Level",
                                     label_list=["pre", "pos"],
                                     show_values=False)


emg.plot_multi_raw_datasets_cloud_comparison( # 使用新的函數名
        raw_datasets_list=[interpolated_data, interpolated_data_1],
        raw_dataset_labels=["Pre", "Pos"],
        directions_to_process=["left"],
        figure_title="Cloud Comparison: Alpha vs Beta (Left/Right Stats)",
        target_length=141,
        selected_emg_channels=["ExtRad.IM EMG1", "Triceps.IM EMG9",
        "Biceps.IM EMG8", 'ExtRad.IM EMG1'],
    )
# %%

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


# --- 繪製 mean std cloud ---
# 假設 standardized_speeds1 和 standardized_speeds2 是兩個包含標準化速度信號的字典
# 假設 target_length = 101
standardized_speeds1 = standardized_speeds
standardized_speeds2 = standardized_speeds_1
standardized_speeds3 = standardized_speeds_2

# 示例 2: 繪製兩個以上數據集進行比較
if standardized_speeds1 and standardized_speeds2:
      pre.plot_standardized_signals_cloud_compare(
          datasets=[standardized_speeds1, standardized_speeds2, standardized_speeds3],
          target_length=101,
          title="Comparison of Mean ± Std Dev Clouds",
          xlabel="Normalized Time (%)",
          ylabel="Signal Value (°/s or other units)",
          # title="Sine Group Mean ± Std Dev",
          labels=['55g', '65g', "60g"],
          # labels=None,                
          # color_indices=[0, 2, 4]          
      )    

else:
      print("至少需要一個有效的標準化信號字典才能繪圖。")
      
      # 長條圖 花費時間
      
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
# === k. Mouse Travel Efficiency
# Mouse Travel Efficiency: idea path/real path

# 找出整段時間內的最大值 or 平均值，單位換算成視角
# 只包含第一次射擊的視角速度
# 1. 先新增 mean_speed 列，预设值为 NaN
# 在cal_tra_efficiency一起計算

# # === c. Initial Move Angle: ===
excldueCen_grouped_df['Initial Move Angle (°)']

# mean_initial_move_angle = np.mean(excldueCen_grouped_df['Initial Move Angle (°)'])

# # === d. Full Path Time (單位 Second)===
path_time = np.mean(excldueCen_grouped_df["Frame Span"])\
    /metadata['motion_info']['frame_rate']

# === e. Reaction Time ===
# 只計算從中心出發，並且 initial move angle 小於 45 度

# 呼叫函數計算效率
grouped_df = cal.cal_tra_efficiency(df, # 原始資料
                                    excldueCen_grouped_df)

# %%
"""
執行檔案模式
1. Spider shot frist run
2. Grid shot 180 second
3. Spider shot second run 

預計2~4支滑鼠的排版

第一頁
綜合表現？ 六角圖指標 -> 各指標相加最高分
需要的圖 
1. Spider shot * 滑鼠數量
    1.1. mean std colud (2~4支)
    
    1.1. mean std colud (疲勞前後 也是四支)
    
    or 不同滑鼠分開畫 (疲勞前後比較)
    
    1.2. muscle activity level with time (幾條肌肉？)(疲勞前後比較) 
    柱狀圖 (活化平均 mV) 四條肌肉 (qur)

2. Grid Shot * 滑鼠數量
53t
    2.1. median frequency with time (幾條肌肉？)
    2.2. muscle activity level with time (幾條肌肉？)
    



比較的表格 各滑鼠比較

各隻滑鼠的數據

預計產出的圖片

3. 總評
雷達圖


1. Spider shot * 滑鼠數量
    1.1. mean std colud (normalize)
    1.2. muscle activity level?
    
2. Grid Shot * 滑鼠數量
    2.1. median frequency with time (幾條肌肉？)
    2.2. muscle activity level with time (幾條肌肉？)
    
3. 六角圖指標 (需要再區分類別)
    3.1. 疲勞 (median frequency, muscle activity level 的權重)
    3.2. 定位能力 (一次定位: 一次定位完成的百分率)
    3.3. 微調能力 (最高速後減速定位)
    3.4. 第一槍 miss 後，再命中目標的時間 (第二槍到最後一槍的時間)
    3.5. 移動速度 (平均速度、最高速度、權重)
    3.6. 減速能力
    
3. (超過兩隻) 六角圖指標 -> 直方圖 ?
    
4. 表格
    data format
	GroupID   Items           mouse A      Mouse B       Mouse C     Mouse D
    -------   --------------  -----------  ------------  ----------  -----------
    1       [53.0, 71.0]          1           53.0         71.0        18.0
    
    4.1. 敏捷面向
        4.1.1. 
    4.2. 疲勞指標
"""


# %%







































