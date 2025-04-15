# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 09:28:13 2025

@author: Hsin.YH.Yang
"""
import ezc3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
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
        motion_data_dict[marker_name] = np.transpose(c['data']['points'][:3, i, :]) / 10  #maker the name of each variable
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


# 讀取 CSV 文件，並指定欄位名稱
# file_path = "IndexData.csv"  # 請修改成你的文件路徑
# df = pd.read_csv(file_path, header=None, names=['X', 'Y', 'Z'])


# 取得 Z 軸數據
# z_values = df["Z"].values
df = combine_dict["markers"]["R.I.Finger3"]
z_values = combine_dict["markers"]["R.I.Finger3"][:, 2]

# 找到 Z 軸的局部最小值索引
order = 5  # 設定區間大小，可根據數據調整
min_frame_gap = 8
min_z_diff = 0.2  

local_minima_idx = argrelextrema(z_values, np.less, order=order)[0]

# 計算 Z 軸的平均值
z_mean = np.mean(z_values)

# 設定閾值：小於 (平均值 - 0.05) 的點才視為局部最小值
threshold = z_mean - 0.05

# 篩選符合閾值條件的局部最小值
filtered_minima_idx = [idx for idx in local_minima_idx if z_values[idx] < threshold]



# 繪製 Z 軸數據與篩選後的局部最小值
plt.figure(figsize=(12, 5))
plt.plot(z_values, label='Z-Axis', color='b', alpha=0.7)
plt.scatter(filtered_minima_idx, z_values[filtered_minima_idx], color='r', label='Filtered Local Minima', zorder=3)
plt.axhline(threshold, color='g', linestyle='--', label=f'Threshold ({threshold:.2f})')
plt.xlabel("Frame")
plt.ylabel("Z Value")
plt.title("Filtered Local Minima of Z-Axis")
plt.legend()
plt.show()

# === 加入最小 frame 間隔條件 ===
# final_minima_idx = []
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

# 輸出篩選後的局部最小值數據
# filtered_minima_data = pd.DataFrame({
#     "Frame": filtered_minima_idx,
#     "Z Value": z_values[filtered_minima_idx]
# })

# # 存成 CSV
# filtered_minima_data.to_csv("Filtered_Local_Minima.csv", index=False)

# 顯示篩選後的數據
# print(filtered_minima_data.head())


df = pd.DataFrame(combine_dict["markers"]["R.I.Finger3"],
                  columns = ["X", "Y", "Z"])

# === 參數設定 ===
DPI = 800
sensitivity = 1.0
yaw = 0.022  # CS2 預設值

# === 滑鼠移動轉視角（整段軌跡） ===
delta_x_mm = df["X"].diff().fillna(0)
delta_y_mm = df["Y"].diff().fillna(0)

df["yaw_deg"] = delta_x_mm / 25.4 * DPI * sensitivity * yaw     # 水平視角變化
df["pitch_deg"] = delta_y_mm / 25.4 * DPI * sensitivity * yaw   # 垂直視角變化

df["cum_yaw_deg"] = df["yaw_deg"].cumsum()     # 累積水平視角（轉向左/右）
df["cum_pitch_deg"] = df["pitch_deg"].cumsum() # 累積垂直視角（往上/下）

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
filtered_minima_data.to_csv("Filtered_Local_Minima_Final_ViewAngle.csv", index=False)

































