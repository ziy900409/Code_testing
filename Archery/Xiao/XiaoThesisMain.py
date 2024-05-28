# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:18:26 2024

目前問題 安卡期判定, 繪圖

程式結構
1. 處理c3d
    1.1. 讀取.c3d
    1.2. 找出 trigger on 時間
    1.3. 計算動作資料
    • E1: 當L.Wrist.Rad Z軸高度超過T10 Z軸高度 擷取此段資料。
    • E5: 擷取直到弓身低於T10 Z軸高度，停止擷取。
    • 資料分期點:
    • E2 舉弓頂點時間:根據全段資料，以L.Wrist.Rad Z軸判定，回傳位置峰值
    數值與對應時間點，即時運算角度後取角度峰直數值與對應時間點
    • E3 當L.Wrist.Rad Z軸高度等於L. Acromion 進行標記
    • E4 放箭時間:根據資料末端2000點判定，即時運算移動平均, R. Elbow Lat
    X軸超出前1秒數據3個標準差，判定為放箭

2. 同步EMG時間
    
@author: Hsin.Yang 05.May.2024
"""
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"E:\Hsin\git\git\Code_testing\Archery\Xiao")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import XiaoThesisMotionFunction as mot
import XiaoThesisGeneralFunction as gen
from detecta import detect_onset

# %% parameter setting 
staging_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\Archery_stage_v1_input.xlsx"
c3d_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\R01\SH1_1OK.c3d"

# %%
# read staging file
staging_file = pd.read_excel(staging_path,
                             sheet_name="R01")
# read .c3d
motion_info, motion_data, analog_info, analog_data, np_motion_data = mot.read_c3d(c3d_path)

# temp parameter
start_index = staging_file["Start_index_frame"][0]
# rename columns name
rename_columns = motion_data.columns.str.replace("2023 Archery_Rev:", "")
motion_data.columns = rename_columns
# 定義所需要的 markerset, 時間都從 Start_index_frame 開始
L_Wrist_Rad_z = motion_data.loc[start_index:, ["Frame", "L.Wrist.Rad_z"]].reset_index(drop=True)
T10_z = motion_data.loc[start_index:, ["Frame", "T10_z"]].reset_index(drop=True)
L_Acromion_z = motion_data.loc[start_index:, ["Frame", "L.Acromion_z"]].reset_index(drop=True)
R_Elbow_Lat_x = motion_data.loc[start_index:, ["Frame", "R.Epi.Lat_z"]].reset_index(drop=True)
bow_middle = motion_data.loc[start_index:, ["Frame", "Middle_z"]].reset_index(drop=True)
C7 = motion_data.loc[start_index:, ["Frame", "C7_x", "C7_y", "C7_z"]].reset_index(drop=True)
R_Finger = motion_data.loc[start_index:, ["Frame", "R.Finger_x", "R.Finger_y", "R.Finger_z"]].reset_index(drop=True)
# 抓分期時間
# 找L.Wrist.Rad Z, T10 Z, L.Acromion Z, R.Elbow Lat X

# 0. 抓 trigger onset
# analog channel: C63
triggrt_on = detect_onset(analog_data["C63"]*-1,
                          np.mean(analog_data["C63"][:100]*-1)*0.8,
                          n_above=0, n_below=0, show=True)

# 1. E1: 當L.Wrist.Rad Z軸高度超過T10 Z軸高度 擷取此段資料。
# 布林判斷L_Wrist_Rad_z["L.Wrist.Rad_z"] > T10_z["T10_z"] 的第一個 TRUE 位置
E1_idx = (L_Wrist_Rad_z["L.Wrist.Rad_z"] > T10_z["T10_z"]).idxmax()

# 2. E2: 舉弓頂點時間:根據全段資料，以L.Wrist.Rad Z軸判定，回傳位置峰值
#    數值與對應時間點，即時運算角度後取角度峰直數值與對應時間點
E2_idx = L_Wrist_Rad_z["L.Wrist.Rad_z"].idxmax()

# 3. E3: 當L.Wrist.Rad Z軸高度等於L. Acromion Z進行標記
# 找兩者相減的最小值
E3_idx = abs(L_Wrist_Rad_z["L.Wrist.Rad_z"] - L_Acromion_z["L.Acromion_z"]).idxmin()
# 找安卡期 R.Finger 最貼近 C7 or Front.Head 的時間點 


def euclidean_distance(point1, point2):
    """
    計算兩個三維點之間的歐幾里得距離
    
    參數：
    point1, point2: 列表或元組，包含三個元素表示三維座標，例如 (x, y, z)
    
    返回值：
    兩點之間的歐幾里得距離
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point1 - point2)
    return distance

E3_idx = []
for i in range(len(C7)):
    E3_idx.append(euclidean_distance(C7.loc[i, ["C7_x", "C7_y", "C7_z"]],
                                     R_Finger.loc[i, ["R.Finger_x", "R.Finger_y", "R.Finger_z"]]))
np.array(E3_idx).argmin()
# 4. E4: 放箭時間:根據資料末端2000點判定，即時運算移動平均, R. Elbow Lat X軸
#       超出前1秒數據3個標準差，判定為放箭



# 5. E5: 擷取直到弓身低於T10 Z軸高度，停止擷取。
# find bow_middle < T10_z and time begin from E2
E5_idx = (bow_middle.loc[E2_idx:, "Middle_z"] < T10_z.loc[E2_idx:, "T10_z"]).idxmax()
# 6. 舉弓角度計算
# T10, L_Wrist_Rad, L_Wrist_Rad 在 T10 平面的投影點
T10 = motion_data.loc[start_index:, ["Frame","T10_x", "T10_y", "T10_z"]]
L_Wrist_Rad = motion_data.loc[start_index:, ["Frame", "L.Wrist.Rad_z"]]































