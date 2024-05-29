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
import gc
import os
import sys
# 路徑改成你放自己code的資料夾
# sys.path.append(r"E:\Hsin\git\git\Code_testing\Archery\Xiao")
sys.path.append(r"D:\BenQ_Project\git\Code_testing\Archery\Xiao")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import XiaoThesisMotionFunction as mot
import XiaoThesisGeneralFunction as gen
from detecta import detect_onset

# %% parameter setting 
# staging_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\Archery_stage_v1_input.xlsx"
staging_path = r"D:\BenQ_Project\python\Archery\Archery_stage_v1_input.xlsx"
# c3d_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\R01\SH1_1OK.c3d"
c3d_path = r"D:\BenQ_Project\python\Archery\R01\SH1_1OK.c3d"



folder_paramter = {"data_path": r"C:/Users/angel/Documents/NTSU/data/112_Plan2_YFMSArchery/",
                   "method_subfolder": ["Method_1"], # , "Method_2"
                   "subject_subfolder": ["test1", "test2"],
                   "staging_file":[]}

data_path = r"D:\BenQ_Project\python\Archery\\"

# ------------------------------------------------------------------------
# 設定資料夾
"""
Archery --- Raw_Data ---- c3d ---- file
        -            -
        -            -
        -            ---- EMG ---- motion
        -                     -
        -                     -
        -                     ---- MVC
        -                     -
        -                     -
        -                     ---- SAVE
        -                     -
        -                     -
        -                     ---- X
        -
        -
        --- Processing_Data ---- c3d ---- data
                            -        -
                            -        -
                            -        ---- figure
                            -
                            -
                            ---- EMG

"""
# 第一層 ----------------------------------------------------------------------
RawData_folder = "\\Raw_Data\\"
processingData_folder = "\\Processing_Data\\"
# 第二層 ----------------------------------------------------------------------
# 動作資料夾名稱
c3d_folder = "\\c3d\\"
# EMG 資料夾
emg_folder = "\\EMG\\"
# 第三層 ----------------------------------------------------------------------

fig_save = "\\figure"
# 子資料夾名稱
sub_folder = "\\\\"

# MVC資料夾名稱
MVC_folder = "MVC"
# downsampling frequency
down_freq = 1000
# 抓放箭時候前後秒數
# example : [秒數*採樣頻率, 秒數*採樣頻率]
release = [5*down_freq, 1*down_freq]
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.1 # 窗格長度 (單位 second)
overlap_len = 0.5 # 百分比 (%)
# 預處理資料可修改檔名，並新增標籤，如：S2_MVC_Rep_1.16 -> S2_MVC_Rep_1.16_low
end_name = "_ed"
# 平滑處理方式 ex: lowpass, rms, moving
smoothing_method = 'rms'
# median frequency duration
duration = 1 # unit : second
processing_folder_path = data_path + processingData_folder

# ---------------------找放箭時間用----------------------------
# 設定最接近放箭位置之acc sensor的欄位編號，建議看完三軸資料再選最大的
# 可設定數字或是欄位名稱：ex: R EXTENSOR GROUP: ACC.Y 1 or 5
release_acc = 7
# 設定放箭的振幅大小值
release_peak = 1.0

# %% 路徑設置

all_rawdata_folder_path = {"motion": [], "EMG": []}
all_processing_folder_path = {"motion": [], "EMG": []}

for method in folder_paramter["method_subfolder"]:
    emg_raw_folders = gen.get_folder_paths(data_path, emg_folder, RawData_folder, method)
    motion_raw_folders = gen.get_folder_paths(data_path, c3d_folder, RawData_folder, method)
    emg_processing_folders = gen.get_folder_paths(data_path, emg_folder, processingData_folder, method)
    motion_processing_folders = gen.get_folder_paths(data_path, c3d_folder, processingData_folder, method)
    
    gen.append_paths(all_rawdata_folder_path, "EMG", data_path, emg_folder, RawData_folder, method, emg_raw_folders)
    gen.append_paths(all_rawdata_folder_path, "motion", data_path, c3d_folder, RawData_folder, method, motion_raw_folders)
    gen.append_paths(all_processing_folder_path, "EMG", data_path, emg_folder, processingData_folder, method, emg_processing_folders)
    gen.append_paths(all_processing_folder_path, "motion", data_path, c3d_folder, processingData_folder, method, motion_processing_folders)
    del emg_raw_folders, motion_raw_folders, emg_processing_folders, motion_processing_folders
gc.collect(generation=2)
    
# %%
"""
1. 將分期檔與檔案對應
2. 找分期時間
3. 繪圖
"""
folder_list = ["R01"]

for folder in folder_list:
    # read staging file
    staging_file = pd.read_excel(staging_path,
                                 sheet_name=folder)
    file_list = gen.Read_File(data_path + motion_folder + folder,
                              ".c3d",
                              subfolder=False)
    for file in file_list:
        for file_name in range(len(staging_file["Motion_filename"])):
            filepath, tempfilename = os.path.split(file)
            # filename, extension = os.path.splitext(tempfilename)
            if tempfilename == staging_file["Motion_filename"][file_name]:
                print(tempfilename)
        
    
    
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




E3_idx = []
for i in range(len(C7)):
    E3_idx.append(gen.euclidean_distance(C7.loc[i, ["C7_x", "C7_y", "C7_z"]],
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































