# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:38:37 2025

@author: Hsin.YH.Yang
"""


import ezc3d
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
# 路徑改成你放自己code的資料夾
# sys.path.append(r"D:\BenQ_Project\git\BiomechanicsTools\BiomechanicsTools\Gen")
# sys.path.append(r"D:\BenQ_Project\git\BiomechanicsTools\BiomechanicsTools\Gen")

# 檢查是否存在力板資訊
folder_path = r"D:\BenQ_Project\git\BiomechanicsTools\BiomechanicsTools\example of code\step4.5_test_joint_com_TX\\"

path = folder_path + 'golfer_sta.c3d'
def read_c3d(path, forceplate=False, analog=False, prefix=False):
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
    # multiple = 2
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
    if prefix:
        for label in range(len(motion_info['LABELS'])):
            motion_info['LABELS'][label] = motion_info['LABELS'][label].replace(prefix, "")
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
    for i, marker_name in enumerate(c['parameters']['POINT']['LABELS']['value']):  #label the name of the data for each variable
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
    if analog:
        analog_data_dict = {}
        for i, marker_name in enumerate(c["parameters"]["ANALOG"]["LABELS"]["value"]):  #label the name of the data for each variable
            analog_data_dict[marker_name] = np.transpose(c["data"]["analogs"][0, i, :])
    
    if forceplate and analog:
        combine_dict = {"marker": fillgap_markers,
                        "FP": FP_data_dict,
                        "analog": analog_data_dict}
    elif forceplate and not analog:
        combine_dict = {"marker": fillgap_markers,
                        "FP": FP_data_dict}
    elif not forceplate and analog:
        combine_dict = {"marker": fillgap_markers,
                        "analog": analog_data_dict}
    else:
        combine_dict = {"marker": fillgap_markers}
        
    return combine_dict, descriptions


# %%






path = r"D:\BenQ_Project\01_UR_lab\2024_11 Shanghai CS Major\1. Motion\Major_weight\S01\20241114\S01_SpiderShot_ZA1_1.c3d"
combine_dict, descriptions = read_c3d(path, forceplate=False, analog=False, prefix="Golfer:")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# 讀取 CSV 文件，並指定欄位名稱
file_path = "IndexData.csv"  # 請修改成你的文件路徑
df = pd.read_csv(file_path, header=None, names=['X', 'Y', 'Z'])

# 取得 Z 軸數據
z_values = df["Z"].values

# 找到 Z 軸的局部最小值索引
order = 5  # 設定區間大小，可根據數據調整
local_minima_idx = argrelextrema(z_values, np.less, order=order)[0]

# 計算 Z 軸的平均值
z_mean = np.mean(z_values)

# 設定閾值：小於 (平均值 - 0.05) 的點才視為局部最小值
threshold = z_mean - 0.05

# 篩選符合閾值條件的局部最小值
filtered_minima_idx = [idx for idx in local_minima_idx if z_values[idx] < threshold]

# 繪製 Z 軸數據與篩選後的局部最小值
plt.figure(figsize=(12, 5))
plt.plot(df.index, z_values, label='Z-Axis', color='b', alpha=0.7)
plt.scatter(filtered_minima_idx, z_values[filtered_minima_idx], color='r', label='Filtered Local Minima', zorder=3)
plt.axhline(threshold, color='g', linestyle='--', label=f'Threshold ({threshold:.2f})')
plt.xlabel("Frame")
plt.ylabel("Z Value")
plt.title("Filtered Local Minima of Z-Axis")
plt.legend()
plt.show()

# 輸出篩選後的局部最小值數據
filtered_minima_data = pd.DataFrame({
    "Frame": filtered_minima_idx,
    "Z Value": z_values[filtered_minima_idx]
})

# 存成 CSV
filtered_minima_data.to_csv("Filtered_Local_Minima.csv", index=False)

# 顯示篩選後的數據
print(filtered_minima_data.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# 讀取 CSV 文件，並指定欄位名稱
file_path = "IndexData.csv"  # 請修改成你的文件路徑
df = pd.read_csv(file_path, header=None, names=['X', 'Y', 'Z'])

# 取得 Z 軸數據
z_values = df["Z"].values

# 找到 Z 軸的局部最小值索引
order = 5  # 設定區間大小，可根據數據調整
local_minima_idx = argrelextrema(z_values, np.less, order=order)[0]

# 計算 Z 軸的平均值
z_mean = np.mean(z_values)

# 設定閾值：小於 (平均值 - 0.05) 的點才視為局部最小值
threshold = z_mean - 0.05

# 篩選符合閾值條件的局部最小值
filtered_minima_idx = [idx for idx in local_minima_idx if z_values[idx] < threshold]

# 取得局部最小值對應的 X, Y 座標
filtered_x = df.loc[filtered_minima_idx, "X"]
filtered_y = df.loc[filtered_minima_idx, "Y"]

# 逆時針旋轉 90° (交換 X, Y 軸 並取負號)
plt.figure(figsize=(8, 8))
plt.scatter(df["Y"], -df["X"], c=df.index, cmap="viridis", alpha=0.7, s=5, label="Trajectory")
plt.scatter(filtered_y, -filtered_x, color="red", s=20, label="Local Minima", zorder=3)  # 標記局部最小值
plt.colorbar(label="Frame Index")
plt.xlabel("Y Axis (Rotated)")
plt.ylabel("X Axis (Rotated)")
plt.title("Rotated 2D Trajectory of Index Finger with Local Minima (90° Counterclockwise)")
plt.legend()
plt.show()













