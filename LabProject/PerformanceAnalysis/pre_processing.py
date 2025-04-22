# %%
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 09:28:13 2025

1. 找出所有的Z軸局部最小值
    1.1.  scipy.signal.argrelextrema 找出資料中的局部極值點（最大值或最小值）
            order = 5
    1.2. 小於 (平均值 - 0.05) 的點才視為局部最小值
            small than 0.05
    1.3. 加入最小 frame 間隔條件 or 兩Z軸局部最小值差異超過閾值
            min_frame_gap = 8, min_z_diff = 0.2

2. 計算
    2.1. 找出每一次目標擊殺的開槍數
        2.1.1. 以滑鼠點擊次數計算，使用Z軸局部最小值，如果兩次Z軸局部最小值的視角差
                小於某個閾值，則視為仍在瞄準同一個目標
        2.1.2. 找出完成擊殺的 frame 以及上一個視角大於閾值的視角位置
                data format
            	GroupID   Frames          Shot Count   Frame Start   Frame End   Frame Span
                -------   --------------  -----------  ------------  ----------  -----------
                1       [53.0, 71.0]          1           53.0         71.0        18.0
    2.2. 計算
        2.2.1. 指標
            o. 擊殺數, 命中率？
            a. Throughput (Mouse Travel Efficiency): 
            b. Mouse Speed (°/s): 找出整段時間內的最大值 or 平均值，單位換算成視角
            c. Initial Move Angle: 初始 5 個 frame 的移動方向與最終擊殺目標位置的視角差
            d. Full Path Time: 
            e. Reaction Time: 從這次目標擊殺到某個 frame 移動速度超過一個閾值 
                扣掉直接回中的反應時間
            i. 一槍擊殺的次數, 二槍, 三槍...
            j. 超過目標的次數， 還沒到目標就開槍的次數
            k. Mouse Travel Efficiency: idea path/real path
        2.2.2. 不同方向的計算: 全部方向綜合, 分四個方向 (四象限)

3. 畫出圖形
    3.1. 將"手部移動距離"換算成"視角移動"
        3.1.1. 因為不知道視角的絕對位置，所以視角使用手部位置的兩個frame換算視角差，
                並且使用累計視角以可視化視角位移
    3.2. 視覺化所有移動的 V-T 圖: 分成不同象限, 不同開槍次數
        3.2.1. std cloud

@author: Hsin.YH.Yang
"""
import ezc3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from numpy.linalg import norm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用簡體黑體（通常會有）
plt.rcParams['axes.unicode_minus'] = False    # 避免座標軸負號亂碼
# import sys
# import os

# %%

vicon2cortex = {'MOS1': 'M1',
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

# %%
def read_c3d(path, forceplate=False, analog=False, prefix=False, rename=False):
# the processes including the interpolation 
    """
    input1 path of the C3D data
    inpu2 the re-sampling times (using motion data frequency to time)
    ----------
    outcome1 combine marker and fp data in a dictionary
    outcome2 the description of the data (some variables are mannual)
    
    ###
    總共分成三個區塊
    1. 處理基本資料
    2. 處理 motion data
    3. 處理 analog data
        3.1. force plate data
        3.2. EMG data
    4. 處理力版資料
    
    """
    # Interpolation: using polynomial method, order = 3 
    def interpolate_with_fallback(data):
        data = pd.DataFrame(data)
        data.replace(0, np.nan, inplace=True)
        data = data.interpolate(method='linear', axis=0)
        data.bfill(inplace=True)  
        data.ffill(inplace=True)  
        if data.isnull().values.any() or (data == 0).any().any():
            data = data.interpolate(method='polynomial', order=2, axis=0).fillna(method='bfill').fillna(method='ffill')
        return data.values  
    # read c3d file
    c = ezc3d.c3d(path, extract_forceplat_data=True)
    ## 1. deal with data information
    motion_info = c["header"]["points"]
    label = []

    # add Unit in motion information
    motion_info.update(
        {
            "UNITS": c["parameters"]["POINT"]["UNITS"]["value"],
            "LABELS": c["parameters"]["POINT"]["LABELS"]["value"],
        }
    )
    # remove prefix
    if prefix:
        for letter in prefix:
            for label in range(len(motion_info['LABELS'])):
                motion_info['LABELS'][label] = motion_info['LABELS'][label].replace(letter, "")
    # rename the markers
    if rename:
        new_strings_list = [s for s in motion_info["LABELS"]]
        for key, value in rename.items():
                    new_strings_list = [s.replace(key, value) for s in new_strings_list]
    motion_info.update(
        {
            "LABELS": new_strings_list
         }
        )
    # structing the data information
    descriptions = {
        "motion info": motion_info,
        "analog info": c["header"]["analogs"],
        "FP info": {
            "caution": "the unit is following Qualisis C3D",
            "Force_unit": "N",
            "Torque_unit": "Nm",
            "COP": "mm"
            }
        }
    ## 2.1. deal with motion data
    # change the variable type from dataframe to dictionary and change unit 
    motion_data_dict = {}
    for i, marker_name in enumerate(motion_info['LABELS']):  #label the name of the data for each variable
        # change the Unit from mm to cm
        motion_data_dict[marker_name] = np.transpose(c['data']['points'][:3, i, :]) #maker the name of each variable
    # 2.2. gap filling to marker data 
    fillgap_markers = {key: interpolate_with_fallback(value) for key, value in motion_data_dict.items()}
    # create time frame
    motion_time = np.linspace(
        0, # start
        ((c['header']['points']['last_frame'])/c['header']['points']['frame_rate']), # stop = last_frame/frame_rate
        num = (np.shape(c['data']['points'])[-1]) # num = last_frame
                              )
    fillgap_markers.update({"time": motion_time})
    ## 3.1 create force plate channel name (the ori unit Force = N; torque = Nmm; COP = mm in Qualysis C3D)
    # only if the number of force plate larger than 0
    if forceplate:
        if 'FORCE_PLATFORM' in c['parameters'] and \
            c['parameters']['FORCE_PLATFORM']['USED']['value'][0] > 0:
                FP_data_dict = {}
                for i in range(c['parameters']['FORCE_PLATFORM']['USED']['value'][0]):
                    FP_data_dict[f'PF{i+1}'] = {
                        "corner": c['parameters']['FORCE_PLATFORM']['CORNERS']['value'][:, :, i].T,
                        "force": c["data"]["platform"][i]['force'].T,
                        "moment": c["data"]["platform"][i]['moment'].T / 1000, # change the Unit from Nmm to N
                        "COP": c["data"]["platform"][i]['center_of_pressure'].T / 10 # change the Unit from mm to cm
                        }
    ## store data to dict structure
    if analog:
        analog_data_dict = {}
        for i, marker_name in enumerate(c["parameters"]["ANALOG"]["LABELS"]["value"]):  #label the name of the data for each variable
            analog_data_dict[marker_name] = np.transpose(c["data"]["analogs"][0, i, :])
    
    if forceplate and analog:
        combine_dict = {"markers": fillgap_markers,
                        "FP": FP_data_dict,
                        "analog": analog_data_dict}
    elif forceplate and not analog:
        combine_dict = {"markers": fillgap_markers,
                        "FP": FP_data_dict}
    elif not forceplate and analog:
        combine_dict = {"markers": fillgap_markers,
                        "analog": analog_data_dict}
    else:
        combine_dict = {"markers": fillgap_markers}
        
    return combine_dict, descriptions


# %%

data_path = r"D:/BenQ_Project/01_UR_lab/2024_11 Shanghai CS Major/1. Motion/Major_weight/S06/20241206/S06_SpiderShot_S1_1.c3d"

combine_dict, descriptions = read_c3d(data_path,
                                      prefix="S06", rename=vicon2cortex)

# %% analysis spider shot
"""
1. 找出所有的Z軸局部最小值
    1.1.  scipy.signal.argrelextrema 找出資料中的局部極值點（最大值或最小值）
            order = 5
    1.2. 小於 (平均值 - 0.05) 的點才視為局部最小值
            small than 0.05
    1.3. 加入最小 frame 間隔條件 or 兩Z軸局部最小值差異超過閾值
            min_frame_gap = 8, min_z_diff = 0.2
"""

# 取得 Z 軸數據
# z_values = df["Z"].values
df = combine_dict["markers"]["R.I.Finger3"]
z_values = combine_dict["markers"]["R.I.Finger3"][:, 2]

# 找到 Z 軸的局部最小值索引
order = 5  # 設定區間大小，可根據數據調整
min_frame_gap = 8
min_z_diff = 0.2  
# 計算 Z 軸的平均值
z_mean = np.mean(z_values)
# 設定閾值：小於 (平均值 - 0.05) 的點才視為局部最小值
threshold = z_mean - 0.05

local_minima_idx = argrelextrema(z_values, np.less, order=order)[0]

# 篩選符合閾值條件的局部最小值
filtered_minima_idx = [idx for idx in local_minima_idx if z_values[idx] < threshold]

# === 加入最小 frame 間隔條件 ===
# === 最終篩選邏輯 ===
final_minima_idx = []

for idx in filtered_minima_idx:
    if not final_minima_idx:
        final_minima_idx.append(idx)
        continue

    last_idx = final_minima_idx[-1]
    frame_diff = idx - last_idx

    if frame_diff >= min_frame_gap:
        final_minima_idx.append(idx)  # 相隔夠遠，直接加入
    else:
        z_diff = abs(z_values[idx] - z_values[last_idx])
        if z_diff < min_z_diff:
            # 差異太小，保留 Z 較小者
            if z_values[idx] < z_values[last_idx]:
                final_minima_idx[-1] = idx  # 替換
            # 否則不做任何處理（保留原來的）
        else:
            final_minima_idx.append(idx)  # 雖然間隔近，但差異夠大，也保留

# 輸出篩選後的局部最小值數據
# filtered_minima_data = pd.DataFrame({
#     "Frame": filtered_minima_idx,
#     "Z Value": z_values[filtered_minima_idx]
# })

# # 存成 CSV
# filtered_minima_data.to_csv("Filtered_Local_Minima.csv", index=False)

# 顯示篩選後的數據
# print(filtered_minima_data.head())

# 繪製 Z 軸數據與篩選後的局部最小值
plt.figure(figsize=(12, 5))
plt.plot(z_values, label='Z-Axis', color='b', alpha=0.7)
plt.scatter(final_minima_idx, z_values[final_minima_idx], color='r', label='Filtered Local Minima', zorder=3)
plt.axhline(threshold, color='g', linestyle='--', label=f'Threshold ({threshold:.2f})')
plt.xlabel("Frame")
plt.ylabel("Z Value")
plt.title("Filtered Local Minima of Z-Axis")
plt.legend()
plt.show()
# %%



# === 參數設定 ===
DPI = 800
sensitivity = 1.0
yaw = 0.022  # CS2 預設值
# df 計算
# === 滑鼠移動轉視角（整段軌跡） ===
delta_x_mm = df["X"].diff().fillna(0)
delta_y_mm = df["Y"].diff().fillna(0)

# 800*1*c = 20.4545
# === 4. 將滑鼠移動換算成視角角度（°）===

df["yaw_deg"] = delta_x_mm / 25.4 * DPI * sensitivity * yaw     # 水平視角變化
df["pitch_deg"] = delta_y_mm / 25.4 * DPI * sensitivity * yaw   # 垂直視角變化

df["cum_yaw_deg"] = df["yaw_deg"].cumsum()     # 累積水平視角（轉向左/右）
df["cum_pitch_deg"] = df["pitch_deg"].cumsum() # 累積垂直視角（往上/下）


# 以假設的 frame rate 240 fps (可自行調整)
sampling_rate = descriptions["motion info"]["frame_rate"]
dt = 1 / sampling_rate

df["speed"] = np.sqrt((delta_x_mm / dt)**2 + \
                      (delta_y_mm / dt)**2)   # mm/s
df["yaw_speed_dps"] = df["cum_yaw_deg"].diff().fillna(0) / dt
df["pitch_speed_dps"] = df["cum_pitch_deg"].diff().fillna(0) / dt
df["angle_speed_dps"] = np.sqrt(df["yaw_speed_dps"]**2 + \
                                df["pitch_speed_dps"]**2)


# === 對應視角的局部最小值點 ===
filtered_yaw = df.loc[final_minima_idx, "cum_yaw_deg"]
filtered_pitch = df.loc[final_minima_idx, "cum_pitch_deg"]

# === 視角軌跡圖（逆時針旋轉視角等價於畫 pitch vs -yaw）===
plt.figure(figsize=(8, 8))
plt.scatter(df["cum_pitch_deg"], -df["cum_yaw_deg"], c=df.index, cmap="viridis", alpha=0.7, s=5, label="View Angle Trajectory")
plt.scatter(filtered_pitch, -filtered_yaw, color="red", s=20, label="Final Local Minima", zorder=3)
plt.colorbar(label="Frame Index")
plt.xlabel("Pitch Angle (Vertical) °")
plt.ylabel("Yaw Angle (Horizontal, Rotated) °")
plt.title("視角軌跡轉換後的 Z 軸局部最小值分析")
plt.legend()
plt.show()

# === 匯出包含視角資料的最小值 ===
filtered_minima_data = pd.DataFrame({
    "Frame": final_minima_idx,
    "Z Value": z_values[final_minima_idx],
    "Yaw Angle (°)": df["cum_yaw_deg"].iloc[final_minima_idx].values,
    "Pitch Angle (°)": df["cum_pitch_deg"].iloc[final_minima_idx].values
})
print(filtered_minima_data)
# filtered_minima_data.to_csv("Filtered_Local_Minima_Final_ViewAngle.csv", index=False)

# %%

# === 取得局部最小值對應的視角資料 ===
filtered_yaw   = df.loc[final_minima_idx, "cum_yaw_deg"]
filtered_pitch = df.loc[final_minima_idx, "cum_pitch_deg"]

# === 繪圖：以視角軌跡繪圖，使用滑鼠速度作為顏色依據 ===
plt.figure(figsize=(8, 8))
sc = plt.scatter(df["cum_pitch_deg"], -df["cum_yaw_deg"],
                 c=df["speed"], cmap="plasma", alpha=0.7, s=5,
                 label="View Angle Trajectory")
# 標記篩選後的局部最小值
plt.scatter(filtered_pitch, -filtered_yaw, color="red", s=20,
            label="Final Local Minima", zorder=3)

# 以滑鼠速度 (mm/s) 作為 colorbar 的標示
plt.colorbar(sc, label="Mouse Speed (mm/s)")
plt.xlabel("Pitch Angle (Vertical) °")
plt.ylabel("Yaw Angle (Horizontal, Rotated) °")
plt.title("View Angle Trajectory Colored by Mouse Speed")
plt.legend()
plt.show()



# %%
"""
2. 計算
    2.1. 找出每一次目標擊殺的開槍數
        2.1.1. 以滑鼠點擊次數計算，使用Z軸局部最小值，如果兩次Z軸局部最小值的視角差
                小於某個閾值，則視為仍在瞄準同一個目標
        2.1.2. 找出完成擊殺的 frame 以及上一個視角大於閾值的視角位置
                data format
            	GroupID   Frames          Shot Count   Frame Start   Frame End   Frame Span
                -------   --------------  -----------  ------------  ----------  -----------
                1       [53.0, 71.0]          1           53.0         71.0        18.0

"""




# === 滑鼠移動轉視角（整段軌跡） ===
delta_x_mm = df["X"].diff().fillna(0)
delta_y_mm = df["Y"].diff().fillna(0)



# === 6. 計算與前/後最小值的視角差 ===
angle_diffs = []
for i in range(len(final_minima_idx)):
    idx_curr = final_minima_idx[i]

    yaw_curr = df["cum_yaw_deg"].iloc[idx_curr]
    pitch_curr = df["cum_pitch_deg"].iloc[idx_curr]

    # 計算與前一個的角度差
    if i == 0:
        diff_prev = np.nan  # 沒有前一筆
    else:
        idx_prev = final_minima_idx[i - 1]
        yaw_prev = df["cum_yaw_deg"].iloc[idx_prev]
        pitch_prev = df["cum_pitch_deg"].iloc[idx_prev]
        diff_prev = np.linalg.norm([yaw_curr - yaw_prev, pitch_curr - pitch_prev])

    # 計算與後一個的角度差
    if i == len(final_minima_idx) - 1:
        diff_next = np.nan  # 沒有下一筆
    else:
        idx_next = final_minima_idx[i + 1]
        yaw_next = df["cum_yaw_deg"].iloc[idx_next]
        pitch_next = df["cum_pitch_deg"].iloc[idx_next]
        diff_next = np.linalg.norm([yaw_curr - yaw_next, pitch_curr - pitch_next])

    angle_diffs.append({
        "Frame": idx_curr,
        "Z Value": df["Z"].iloc[idx_curr],
        "Angle_Diff_To_Prev_Minima (°)": diff_prev,
        "Angle_Diff_To_Next_Minima (°)": diff_next
    })
# === 7. 輸出結果 ===
angle_diffs_df = pd.DataFrame(angle_diffs)
angle_diffs_df.to_csv("Z_Minima_ViewAngle_Comparison.csv", index=False)
print(angle_diffs_df.head())
# %%
angle_threshold = 5  # 單位為度

kill_groups = []
current_group = [final_minima_idx[0]]

for i in range(1, len(final_minima_idx)):
    idx_prev = final_minima_idx[i - 1]
    idx_curr = final_minima_idx[i]

    # 計算視角差
    yaw_prev = df["cum_yaw_deg"].iloc[idx_prev]
    pitch_prev = df["cum_pitch_deg"].iloc[idx_prev]
    yaw_curr = df["cum_yaw_deg"].iloc[idx_curr]
    pitch_curr = df["cum_pitch_deg"].iloc[idx_curr]
    angle_diff = np.linalg.norm([yaw_curr - yaw_prev, pitch_curr - pitch_prev])
    

    if angle_diff < angle_threshold:
        current_group.append(idx_curr)
    else:
        kill_groups.append(current_group)
        current_group = [idx_curr]

# 補上最後一組
if len(current_group) > 0:
    kill_groups.append(current_group)

# 整理成表格
kill_df = pd.DataFrame({
    "GroupID": list(range(1, len(kill_groups)+1)),
    "Frames": kill_groups,
    "Shot Count": [len(g) for g in kill_groups],
    "Frame Start": [min(g) for g in kill_groups],
    "Frame End": [max(g) for g in kill_groups],
})
kill_df["Frame Span"] = kill_df["Frame End"] - kill_df["Frame Start"]



# %%


# 條件：視角變化 < 5°
threshold = 5
highlight_idx_groups = []

for i, row in angle_diffs_df.iterrows():
    if (row["Angle_Diff_To_Prev_Minima (°)"] < threshold or
        row["Angle_Diff_To_Next_Minima (°)"] < threshold):
        # 對應的是 final_minima_idx[i] 以及它的前後
        if 0 < i < len(final_minima_idx) - 1:
            group = [
                final_minima_idx[i - 1],
                final_minima_idx[i],
                final_minima_idx[i + 1]
            ]
            highlight_idx_groups.append(group)

# 將 highlight 群組展平成單一 index 集合
highlight_indices = sorted(set([idx for group in highlight_idx_groups for idx in group]))

# 取得這些 index 對應的 X, Y
highlight_x = df["X"].iloc[highlight_indices]
highlight_y = df["Y"].iloc[highlight_indices]

# 原始所有最小值點
all_minima_x = df["X"].iloc[final_minima_idx]
all_minima_y = df["Y"].iloc[final_minima_idx]

# === 繪圖 ===
plt.figure(figsize=(10, 8))
plt.scatter(df["X"], df["Y"], alpha=0.3, s=5, label="All Points")
plt.scatter(all_minima_x, all_minima_y, color='blue', s=40, label="Z Minima")

# 圈出 XY 差異小的點群
plt.scatter(highlight_x, highlight_y, facecolors='none', edgecolors='red',
            s=120, linewidths=2, label="Minima with small view angle")

plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.title("Z 最小值與小角度變化的點群標記")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()



# %%


# 將必要欄位轉為 NumPy 陣列
angle_array = angle_diffs_df[["Frame", 
                              "Angle_Diff_To_Prev_Minima (°)", 
                              "Angle_Diff_To_Next_Minima (°)"]].to_numpy()

grouped_frames = []
angle_merge_threshold = 5  # 視角差閾值
i = 0
last_frame = None  # 記錄上一組的最後一筆

while i < len(angle_array) - 1:
    current_group = []

    if last_frame is not None:
        current_group.append(last_frame)  # 接續上一組的結尾

    current_group.append(angle_array[i][0])  # 加入目前起點 frame
    j = i + 1

    while j < len(angle_array):
        angle_diff = angle_array[j][1]  # Angle_Diff_To_Prev_Minima (°)

        if pd.isna(angle_diff) or angle_diff >= angle_merge_threshold:
            break

        current_group.append(angle_array[j][0])
        j += 1

    if len(current_group) > 1:
        grouped_frames.append(current_group)
        last_frame = current_group[-1]  # 更新本輪最後 frame
    else:
        last_frame = angle_array[i][0]  # 還是更新這筆（即使沒成群）

    i = j  # 繼續從下一個起點開始

grouped_df = pd.DataFrame({
    "Group ID": list(range(1, len(grouped_frames) + 1)),
    "Frames": grouped_frames,
    "Shot Count": [len(g) - 1 for g in grouped_frames],
    "Frame Start": [min(g) for g in grouped_frames],
    "Frame End": [max(g) for g in grouped_frames],
})
grouped_df["Frame Span"] = grouped_df["Frame End"] - grouped_df["Frame Start"]

initial_angles = []

for _, row in grouped_df.iterrows():
    start_idx = int(row["Frame Start"])
    end_idx = int(row["Frame End"])

    # 若總長度不足 5，則以能取的最多 frame 計算
    move_range = df.iloc[start_idx : start_idx + 5]
    if len(move_range) < 2:
        initial_angles.append(np.nan)
        continue

    # === 1. 初始移動向量（5幀內）
    init_vec = np.array([
        move_range["X"].iloc[-1] - move_range["X"].iloc[0],
        move_range["Y"].iloc[-1] - move_range["Y"].iloc[0]
    ])

    # === 2. 目標向量（Start → End）
    goal_vec = np.array([
        df["X"].iloc[end_idx] - df["X"].iloc[start_idx],
        df["Y"].iloc[end_idx] - df["Y"].iloc[start_idx]
    ])

    # === 3. 計算夾角（°）
    if norm(init_vec) == 0 or norm(goal_vec) == 0:
        angle_deg = np.nan
    else:
        cos_theta = np.dot(init_vec, goal_vec) / (norm(init_vec) * norm(goal_vec))
        cos_theta = np.clip(cos_theta, -1, 1)  # 防止浮點誤差超出 [-1,1]
        angle_deg = np.degrees(np.arccos(cos_theta))

    initial_angles.append(angle_deg)

# === 加入回 grouped_df ===
grouped_df["Initial Move Angle (°)"] = initial_angles

# === 預備資料（視角單位）===
all_minima_deg_x = df["cum_yaw_deg"].iloc[final_minima_idx]
all_minima_deg_y = df["cum_pitch_deg"].iloc[final_minima_idx]

# === 繪圖開始 ===
plt.figure(figsize=(10, 8))

# 背景點（全視角軌跡）
plt.scatter(df["cum_yaw_deg"], df["cum_pitch_deg"], alpha=0.3, s=5, label="All Points")

# 所有最小值點
plt.scatter(all_minima_deg_x, all_minima_deg_y, color='blue', s=40, label="Z Minima")

# ✅ 使用 grouped_frames 分群畫圓（轉為視角單位）
for group in grouped_frames:
    group_x = df["cum_yaw_deg"].iloc[group]
    group_y = df["cum_pitch_deg"].iloc[group]
    plt.scatter(group_x, group_y, facecolors='none', edgecolors='red',
                s=120, linewidths=2)

# === 圖例與標籤 ===
plt.xlabel("Yaw Angle (°)")
plt.ylabel("Pitch Angle (°)")
plt.title("Z 最小值分群視覺化（以視角為單位）")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()


# %%
"""
    2.2. 計算
        2.2.1. 指標
            o. 擊殺數, 命中率？
            a. Throughput (Mouse Travel Efficiency): 
            b. Mouse Speed (°/s): 找出整段時間內的最大值 or 平均值，單位換算成視角
            c. Initial Move Angle: 初始 5 個 frame 的移動方向與最終擊殺目標位置的視角差
            d. Full Path Time: 
            e. Reaction Time: 從這次目標擊殺到某個 frame 移動速度超過一個閾值 
                扣掉直接回中的反應時間
            i. 一槍擊殺的次數, 二槍, 三槍...
            j. 超過目標的次數， 還沒到目標就開槍的次數
            k. Mouse Travel Efficiency: idea path/real path
        2.2.2. 不同方向的計算: 全部方向綜合, 分四個方向 (四象限)
"""


# === o. 擊殺數, 命中率？ ===
kill_count = len(grouped_df)
shot_count = sum(grouped_df["Shot Count"])
accuracy = kill_count/shot_count

# === b. Mouse Speed (°/s) ===

max_angle_speed = max(df["angle_speed_dps"])
# === Initial Move Angle: ===

efficiencies = []
max_speeds = []
mean_speeds = []

for group in kill_df["Frames"]:
    real = compute_real_path(group)
    ideal = compute_ideal_path(group)
    eff = ideal / real if real != 0 else np.nan
    speed_vals = df.loc[group, "angle_speed"]
    efficiencies.append(eff)
    max_speeds.append(speed_vals.max())
    mean_speeds.append(speed_vals.mean())

kill_df["Travel Efficiency"] = efficiencies
kill_df["Max Speed (°/s)"] = max_speeds
kill_df["Mean Speed (°/s)"] = mean_speeds

