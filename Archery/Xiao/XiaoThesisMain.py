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
sys.path.append(r"E:\Hsin\git\git\Code_testing\Archery\Xiao")
# sys.path.append(r"D:\BenQ_Project\git\Code_testing\Archery\Xiao")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import XiaoThesisMotionFunction as mot
import XiaoThesisGeneralFunction as gen
import XiaoThesisEMGFunction as emg
from detecta import detect_onset
from scipy import signal

from datetime import datetime
# matplotlib 設定中文顯示，以及圖片字型
# mpl.rcParams['font.family'] = 'Microsoft Sans Serif'


plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False   # 步驟二（解決坐標軸負數的負號顯示問題）
font = {'family': 'serif',
        'color':  'k',
        'weight': 'bold',
        'size': 20,
        }

# 获取当前日期和时间
now = datetime.now()

# 将日期转换为指定格式
# formatted_date = datetime.now().strftime('%Y-%m-%d-%H:%M')
formatted_date = datetime.now().strftime('%Y-%m-%d-%H%M')
print("當前日期：", formatted_date)
# %% parameter setting 
staging_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\Archery_stage_v5_input.xlsx"
data_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\202406\202405\\"

# 測試組
subject_list = ["R01", "R02", "R03", "R04"]
# ------------------------------------------------------------------------
# 設定資料夾
folder_paramter = {
                  "first_layer": {
                                  "motion":["\\motion\\"],
                                  "EMG": ["\\EMG\\"],
                                  },
                  "second_layer":{
                                  "motion":["\\Raw_Data\\", "\\Processing_Data\\"],
                                  "EMG": ["\\Raw_Data\\", "\\Processing_Data\\"],
                                  },
                  "third_layer":{
                                  "motion":["Method_1"],
                                  "EMG": ["Method_1", "Method_2"],
                                  },
                  "fourth_layer":{
                                  "motion":["\\motion\\"],
                                  "EMG": ["motion", "MVC", "SAVE", "X"],
                                  }
                  }
folder_paramter["fourth_layer"]["EMG"][0]

# 第三層 ----------------------------------------------------------------------

fig_folder = "\\figure\\"
data_folder = "\\data\\"
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
# cutoff frequency
c = 0.802
lowpass_cutoff = 10/c
# median frequency duration
duration = 1 # unit : second
# processing_folder_path = data_path + processingData_folder

# ---------------------找放箭時間用--------------------------------------------
# 設定最接近放箭位置之acc sensor的欄位編號，建議看完三軸資料再選最大的
# 可設定數字或是欄位名稱：ex: R EXTENSOR GROUP: ACC.Y 1 or 5
release_acc = 7
# 設定放箭的振幅大小值
release_peak = 1.0
# trigger threshold
trigger_threshold = 0.02
# 设定阈值和窗口大小
threshold = 0.03
window_size = 5
# 設置繪圖顏色用 --------------------------------------------------------------
cmap = plt.get_cmap('Set2')
# 设置颜色
colors = [cmap(i) for i in np.linspace(0, 1, 6)]
                                    
                                    

# %% 路徑設置

all_rawdata_folder_path = {"motion": [], "EMG": []}
all_processing_folder_path = {"motion": [], "EMG": []}
# 定義 motion
for method in folder_paramter["third_layer"]["motion"]:
    # 定義 rawdata folder path
    rawdata_folder_path = data_path + folder_paramter["first_layer"]["motion"][0] + \
        folder_paramter["second_layer"]["motion"][0] + method
    # 讀取 rawdata folder path
    rawdata_folder_list = [os.path.join(rawdata_folder_path, f) \
                           for f in os.listdir(rawdata_folder_path) \
                           if not f.startswith('.') and os.path.isdir(os.path.join(rawdata_folder_path, f))]
    # 將路徑加到 all_rawdata_folder_path
    all_rawdata_folder_path["motion"].extend(rawdata_folder_list)
    # 定義 processing folder path, 改變 second layer
    processing_folder_path = data_path + folder_paramter["first_layer"]["motion"][0] + \
        folder_paramter["second_layer"]["motion"][1] + method
    processing_folder_list = [os.path.join(processing_folder_path, f) \
                              for f in os.listdir(processing_folder_path) \
                                  if not f.startswith('.') and \
                                      os.path.isdir(os.path.join(processing_folder_path, f))]
    all_processing_folder_path["motion"].extend(processing_folder_list)
    
# 定義 EMG folder path
for method in folder_paramter["third_layer"]["EMG"]:
    # 定義 rawdata folder path
    rawdata_folder_path = data_path + folder_paramter["first_layer"]["EMG"][0] + \
        folder_paramter["second_layer"]["EMG"][0] + method
    # 讀取 rawdata folder path
    rawdata_folder_list = [os.path.join(rawdata_folder_path, f) \
                           for f in os.listdir(rawdata_folder_path) \
                           if not f.startswith('.') and os.path.isdir(os.path.join(rawdata_folder_path, f))]
    # 將路徑加到 all_rawdata_folder_path
    all_rawdata_folder_path["EMG"].extend(rawdata_folder_list)
    # 定義 processing folder path, 改變 second layer
    processing_folder_path = data_path + folder_paramter["first_layer"]["EMG"][0] + \
        folder_paramter["second_layer"]["EMG"][1] + method
    processing_folder_list = [os.path.join(processing_folder_path, f) \
                              for f in os.listdir(processing_folder_path) \
                                  if not f.startswith('.') and \
                                      os.path.isdir(os.path.join(processing_folder_path, f))]
    all_processing_folder_path["EMG"].extend(processing_folder_list)
        

gc.collect(generation=2)

# %% 找放箭時間
"""
2. 找放箭時間
 2.1. 需至 function code 找參數設定，修改範例如下 :
     2.1.1. release_acc = 5
     可設定數字或是欄位名稱：ex: R EXTENSOR GROUP: ACC.Y 1 or 5
     2.1.2. release_peak = 2
     設定放箭的振幅大小值
            
"""

for emg_folder in all_rawdata_folder_path["EMG"]:
    emg.find_release_time(emg_folder + "\\" + folder_paramter["fourth_layer"]["EMG"][0],
                          emg_folder.replace("EMG", "motion").replace("Raw_Data", "Processing_Data") + "\\figure\\release")
        
gc.collect(generation=2)


# %%
"""
1. 將分期檔與檔案對應
2. 找分期時間
3. 繪圖
"""


all_algorithm = pd.DataFrame({},
                             columns = ["Subject", "Motion_filename", "EMG_filename",
                                        "Bow_Angle_Peak", "Bow_Angle_Peak_Frame",
                                        "Bow_Angle_Peak_Time[s]", "Bow_Height_Peak[mm]",
                                        "Bow_Height_Peak_Frame", "Bow_Height_Peak_Time[s]",
                                        "Bow_Height_Peak_Norm", "Anchor_Threadshold[mm]",
                                        "Anchor_Frame", "Anchor_Time[s]", "Release_Threadshold[mm]", 
                                        "Release_Frame", "Release_Time[s]",
                                        "E1 frame", "E3-1 frame", "E5 frame"])

for subject in subject_list:
    for motion_folder in all_rawdata_folder_path["motion"]:
        for emg_folder in all_rawdata_folder_path["EMG"]:
            if subject in motion_folder and subject in emg_folder:
                print(motion_folder)
                print(emg_folder)
                # read staging file
                staging_file = pd.read_excel(staging_path,
                                             sheet_name=subject)
                motion_list = gen.Read_File(motion_folder,
                                            ".c3d",
                                            subfolder=False)
                emg_list = gen.Read_File(emg_folder,
                                         ".csv",
                                         subfolder=True)
                # 設定存檔路徑
                fig_save = motion_folder.replace("Raw_Data", "Processing_Data") + fig_folder
                data_save = motion_folder.replace("Raw_Data", "Processing_Data") + data_folder
                # 從分期檔來找檔案
                for idx in range(len(staging_file["Motion_filename"])):
                    for motion_file in motion_list:
                        for emg_file in emg_list:
                            if staging_file["Motion_filename"][idx] in motion_file \
                                and staging_file["EMG_filename"][idx] in emg_file \
                                    and staging_file["Note"][idx] == "V":
                                    print(motion_file)
                                    print(emg_file)
                                    filepath, tempfilename = os.path.split(motion_file)
                                    filename, extension = os.path.splitext(tempfilename)
                                    # read .c3d
                                    motion_info, motion_data, analog_info, analog_data, np_motion_data = mot.read_c3d(motion_file)
                                    # read .csv
                                    emg_filepath, emg_tempfilename = os.path.split(emg_file)
                                    Extensor_ACC = pd.read_csv(emg_file).iloc[:, [release_acc-1, release_acc]]
                                    true_data_len = len(Extensor_ACC.iloc[:, 1]) - (Extensor_ACC.iloc[:, 1][::-1] != 0).argmax(axis=0)
                                    Extensor_ACC = Extensor_ACC.iloc[:true_data_len, :]
                                    # preprocessing EMG data
                                    # processing_data, bandpass_filtered = emg.EMG_processing(emg_file, smoothing=smoothing_method)
                                    # rename columns name
                                    rename_columns = motion_data.columns.str.replace("2023 Archery_Rev:", "")
                                    motion_data.columns = rename_columns
                                    # filting motion data
                                    lowpass_sos = signal.butter(2, lowpass_cutoff,  btype='low', fs=motion_info['frame_rate'], output='sos')
                                    filted_motion = pd.DataFrame(np.empty(np.shape(motion_data)),
                                                                 columns = motion_data.columns)
                                    filted_motion.iloc[:, 0] = motion_data.iloc[:, 0]
                                    for i in range(np.shape(motion_data)[1]-1):
                                        filted_motion.iloc[:, i+1] = signal.sosfiltfilt(lowpass_sos,
                                                                                        motion_data.iloc[:, i+1].values)
                                    # temp parameter
                                    # 定義開始時間
                                    start_index = staging_file["Start_index_frame"][idx]
                                    # 定義結束時間
                                    if staging_file["End_index_frame"][idx] == '-':
                                        end_index = len(motion_data)
                                    else: 
                                        end_index = staging_file["End_index_frame"][idx]
                                    # 定義基本參數
                                    motion_sampling_rate = motion_info["frame_rate"]
                                    # emg_sample_rate = 1 / (bandpass_filtered.iloc[1, 0] - bandpass_filtered.iloc[0, 0])
                                    acc_sample_rate = 1 / (Extensor_ACC.iloc[1, 0] - Extensor_ACC.iloc[0, 0])
                                    # 定義所需要的 markerset, 時間都從 Start_index_frame 開始
                                    
                                    C7 = filted_motion.loc[start_index:end_index,
                                                           ["Frame", "C7_x", "C7_y", "C7_z"]].reset_index(drop=True)
                                    T10 = filted_motion.loc[start_index:end_index,
                                                            ["Frame", "T10_x", "T10_y", "T10_z"]].reset_index(drop=True)
                                    L_Acromion = filted_motion.loc[start_index:end_index,
                                                                   ["Frame", "L.Acromion_x", "L.Acromion_y", "L.Acromion_z"]].reset_index(drop=True)
                                    L_Wrist_Rad = filted_motion.loc[start_index:end_index,
                                                                    ["Frame", "L.Wrist.Rad_x", "L.Wrist.Rad_y", "L.Wrist.Rad_z"]].reset_index(drop=True)
                                    R_Elbow_Lat = filted_motion.loc[start_index:end_index,
                                                                    ["Frame", "R.Epi.Lat_x", "R.Epi.Lat_y", "R.Epi.Lat_z"]].reset_index(drop=True)
                                    R_Wrist_Rad = filted_motion.loc[start_index:end_index,
                                                                    ["Frame", "R.Wrist.Rad_x", "R.Wrist.Rad_y", "R.Wrist.Rad_z"]].reset_index(drop=True)
                                    R_Finger = filted_motion.loc[start_index:end_index,
                                                                 ["Frame", "R.Finger_x", "R.Finger_y", "R.Finger_z"]].reset_index(drop=True)
                                    bow_middle = filted_motion.loc[start_index:end_index,
                                                                   ["Frame", "Middle_x", "Middle_y", "Middle_z"]].reset_index(drop=True)
                                    # 抓分期時間
                                    # 找L.Wrist.Rad Z, T10 Z, L.Acromion Z, R.Elbow Lat X
                                    
                                    # 0. 抓 trigger onset, release time ----------------------------------------------------------
                                    # analog channel: C63
                                    """
                                    當前日期： 2024-06-06-2315 改到這裡
                                    不知為何以下檔案放箭時間不對
                                    "R02_SHL_Rep_4.16.csv"
                                    
                                    
                                    """
                                    triggrt_on = detect_onset(analog_data.loc[1000:, "C63"]*-1,
                                                              np.mean(analog_data["C63"][10000:10100]*-1) + trigger_threshold,
                                                              n_above=0, n_below=0, show=True)
                                    # find time of arrow release
                                    peaks, _ = signal.find_peaks(Extensor_ACC.iloc[:, 1]*-1, height = release_peak)
                                    # 0.1. 換算 EMG 時間
                                    # emg_start_index = round((start_index - (triggrt_on[0, 0]/analog_info["frame_rate"])) \
                                    #     / motion_sampling_rate * emg_sample_rate)
                                    # emg_end_index = round((end_index - (triggrt_on[0, 0]/analog_info["frame_rate"])) \
                                    #     / motion_sampling_rate * emg_sample_rate)
                                    # 0.2. 換算 ACC to motion 時間
                                    motion_release_frame = round(peaks[0] / acc_sample_rate * motion_sampling_rate + \
                                                                 ((triggrt_on[0, 0] + 1000) / analog_info["frame_rate"] * motion_sampling_rate))
                                    
                                    # 1. E1: 當L.Wrist.Rad Z軸高度超過T10 Z軸高度 擷取此段資料。
                                    # 布林判斷L_Wrist_Rad_z["L.Wrist.Rad_z"] > T10_z["T10_z"] 的第一個 TRUE 位置
                                    E1_idx = (L_Wrist_Rad["L.Wrist.Rad_z"] > T10["T10_z"]).idxmax() + start_index
                                    
                                    # 2. E2: 舉弓頂點時間:根據全段資料，以L.Wrist.Rad Z軸判定，回傳位置峰值
                                    #    數值與對應時間點，即時運算角度後取角度峰直數值與對應時間點
                                    E2_idx = L_Wrist_Rad["L.Wrist.Rad_z"].idxmax() + start_index
                                    
                                    # 3. E3-1, E3-2
                                    # E3-1 : 引弓中段，寫在欄位 3-1_Draw_half_frame、3-1_Time[s]
                                    E3_1_idx = (filted_motion.loc[E1_idx:, "R.Wrist.Rad_x"] < filted_motion.loc[E1_idx:, "T10_x"]).idxmax()

                                    # E3-2 : 固定，寫在欄位 3-2_Anchor_frame、3-2_Time[s]
                                    # 找兩者相減的最小值
                                    # E3_idx_v1 = abs(L_Wrist_Rad["L.Wrist.Rad_z"] - L_Acromion["L.Acromion_z"]).idxmin() + start_index
                                    # 找安卡期 R.Wrist.Rad 最貼近 C7 or Front.Head 的時間點
                                    # 應該用變化量來計算，設定變化量的閾值
                                    # 修改從 E1 idx 開始找，避免受試者靜止不動的情形
                                    E3_2_cal = []
                                    for i in range(len(filted_motion.loc[E1_idx:, "C7_x"])):
                                        E3_2_cal.append(gen.euclidean_distance(filted_motion.loc[E1_idx + i, ["C7_x", "C7_y", "C7_z"]],
                                                                               filted_motion.loc[E1_idx + i, ["R.Wrist.Rad_x", "R.Wrist.Rad_y", "R.Wrist.Rad_z"]]))
                                    E3_2_diff = abs(np.array(E3_2_cal)[1:] - np.array(E3_2_cal)[:-1])

                                    E3_2_idx = gen.find_index(E3_2_diff, threshold, window_size) + E1_idx
                                    # 4. E4: 放箭時間:根據 Extensor_acc ，往前抓0.3~1.3秒, R. Elbow Lat X 軸
                                    #       超出前1秒數據3個標準差，判定為放箭
                                    E4_idx = detect_onset(-filted_motion.loc[motion_release_frame-225:end_index, "R.Epi.Lat_x"].values,
                                                          np.mean(-filted_motion.loc[motion_release_frame-125:motion_release_frame-75,
                                                                                     "R.Epi.Lat_x"].values) + \
                                                              np.std(-filted_motion.loc[motion_release_frame-225:motion_release_frame-75,
                                                                                         "R.Epi.Lat_x"].values)*3,
                                                          n_above=0, n_below=0, show=True)
                                    E4_idx = E4_idx[0, 0] + motion_release_frame - 225
                                    # 5. E5: 擷取直到弓身低於 T10 Z 軸高度，停止擷取。
                                    # find bow_middle < T10_z and time begin from E2
                                    E5_idx = np.argmax(filted_motion.loc[E2_idx:, "Middle_z"] < filted_motion.loc[E2_idx:, "T10_z"]) + E2_idx
                                    a = bow_middle.loc[E2_idx:, "Middle_z"].values < T10.loc[E2_idx:, "T10_z"].values
                                    # 6. 舉弓角度計算 --------------------------------------------------------------------------------
                                    # 定義L. Wrist. Rad 在T10橫斷面上的投影點
                                    L_Wrist_Rad_project = pd.DataFrame({"x": filted_motion.loc[E1_idx:E5_idx, "L.Wrist.Rad_x"].values,
                                                                        "y": filted_motion.loc[E1_idx:E5_idx, "L.Wrist.Rad_y"].values,
                                                                        "z": filted_motion.loc[E1_idx:E5_idx, "T10_z"].values})
                                    # 計算夾角
                                    mAG = mot.included_angle(filted_motion.loc[E1_idx:E5_idx, ["L.Wrist.Rad_x", "L.Wrist.Rad_y", "L.Wrist.Rad_z"]].values,
                                                             filted_motion.loc[E1_idx:E5_idx, ["T10_x", "T10_y", "T10_z"]].values,
                                                             L_Wrist_Rad_project.loc[:, ["x", "y", "z"]].values)
                                    # 6.1. 輸出資料 filted_motion, angle -----------------------------------------------------------
                                    filted_motion.iloc[E1_idx:E5_idx, :].to_excel(data_save + filename + "_trim.xlsx")
                                    pd.DataFrame(mAG).to_excel(data_save + filename + "_angle.xlsx")
                                    temp_output = pd.DataFrame({"Subject": [subject],
                                                                "Motion_filename": tempfilename,
                                                                "EMG_filename": emg_tempfilename,
                                                                "Bow_Angle_Peak":np.max(mAG),
                                                                "Bow_Angle_Peak_Frame": (np.argmax(mAG) + E1_idx),
                                                                "Bow_Angle_Peak_Time[s]": filted_motion.loc[(np.argmax(mAG) + E1_idx), 'Frame'],
                                                                "Bow_Height_Peak[mm]": filted_motion.loc[E2_idx, "L.Wrist.Rad_z"],
                                                                "Bow_Height_Peak_Frame": E2_idx,
                                                                "Bow_Height_Peak_Time[s]": filted_motion.loc[E2_idx, 'Frame'],
                                                                "Bow_Height_Peak_Norm": 0,
                                                                "Anchor_Threadshold[mm]": 0,
                                                                "Anchor_Frame": E3_2_idx,
                                                                "Anchor_Time[s]": filted_motion.loc[E3_2_idx, 'Frame'],
                                                                "Release_Threadshold[mm]": 0, 
                                                                "Release_Frame": E4_idx,
                                                                "Release_Time[s]": filted_motion.loc[E4_idx, 'Frame'],
                                                                "E1 frame": E1_idx,
                                                                "E3-1 frame": E3_1_idx,
                                                                "E5 frame": E5_idx})
                                    all_algorithm = pd.concat([all_algorithm, temp_output])
                                    # 7. 繪圖 -------------------------------------------------
                                    # 7.1. 繪製: 資料經平滑、按事件 1、5 剪裁，標記事件2、3原時間點之資料
                                    labels = ["R.Wrist.Rad", "R.Elbow.Lat", "L.Acromin", "L.Wrist.Rad", "T10", "Middle"]
                                    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex='col')
                                    
                                    # 繪製第一個子圖 X 軸: R.Wrist.Rad, R.Elbow.Lat, L.Acromin, L.Wrist.Rad, T10, Middle 
                                    # 繪製 motion data
                                    axes[0].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # R.Wrist.Rad
                                                 filted_motion.loc[E1_idx:E5_idx, "R.Wrist.Rad_x"].values,
                                                 color=colors[0], label = "R.Wrist.Rad")
                                    axes[0].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # R.Epi.Lat
                                                 filted_motion.loc[E1_idx:E5_idx, "R.Epi.Lat_x"].values,
                                                 color=colors[1], label = "R.Epi.Lat")
                                    axes[0].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # L.Acromion
                                                 filted_motion.loc[E1_idx:E5_idx, "L.Acromion_x"].values,
                                                 color=colors[2], label = "L.Acromion")
                                    axes[0].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # L.Wrist.Rad
                                                 filted_motion.loc[E1_idx:E5_idx, "L.Wrist.Rad_x"].values,
                                                 color=colors[3], label = "L.Wrist.Rad")
                                    axes[0].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # T10
                                                 filted_motion.loc[E1_idx:E5_idx, "T10_x"].values,
                                                 color=colors[4], label = "T10")
                                    axes[0].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # Middle
                                                 filted_motion.loc[E1_idx:E5_idx, "Middle_x"].values,
                                                 color=colors[5], label = "Middle")
                                    # E1 劃分期線
                                    axes[0].axvline(filted_motion.loc[E1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[0].text(filted_motion.loc[E1_idx, 'Frame'],
                                                 axes[0].get_ylim()[1], f"E1:{filted_motion.loc[E1_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E2 劃分期線
                                    axes[0].axvline(filted_motion.loc[E2_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[0].text(filted_motion.loc[E2_idx, 'Frame'],
                                                 axes[0].get_ylim()[1], f"E2:{filted_motion.loc[E2_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-1 劃分期線
                                    axes[0].axvline(filted_motion.loc[E3_1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[0].text(filted_motion.loc[E3_1_idx, 'Frame'],
                                                 axes[0].get_ylim()[1], f"E3-1:{filted_motion.loc[E3_1_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-2 劃分期線
                                    axes[0].axvline(filted_motion.loc[E3_2_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[0].text(filted_motion.loc[E3_2_idx, 'Frame'] + 0.9,
                                                 axes[0].get_ylim()[1], f"E3-2:{filted_motion.loc[E3_2_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E4 劃分期線
                                    axes[0].axvline(filted_motion.loc[E4_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[0].text(filted_motion.loc[E4_idx, 'Frame']-1,
                                                 axes[0].get_ylim()[1], f"E4:{filted_motion.loc[E4_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E5 劃分期線
                                    axes[0].axvline(filted_motion.loc[E5_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[0].text(filted_motion.loc[E5_idx, 'Frame'],
                                                 axes[0].get_ylim()[1],  f"E5:{filted_motion.loc[E5_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # 子圖資訊設定，坐標軸
                                    axes[0].set_ylabel('X 軸 data', fontsize = 14)  # 设置子图标题
                                    # 繪製第二個子圖 Y 軸 ---------------------------
                                    # 繪製 motion data
                                    axes[1].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # R.Wrist.Rad
                                                 filted_motion.loc[E1_idx:E5_idx, "R.Wrist.Rad_y"].values,
                                                 color=colors[0], label = "R.Wrist.Rad")
                                    axes[1].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # R.Epi.Lat
                                                 filted_motion.loc[E1_idx:E5_idx, "R.Epi.Lat_y"].values,
                                                 color=colors[1], label = "R.Epi.Lat")
                                    axes[1].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # L.Acromion
                                                 filted_motion.loc[E1_idx:E5_idx, "L.Acromion_y"].values,
                                                 color=colors[2], label = "L.Acromion")
                                    axes[1].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # L.Wrist.Rad
                                                 filted_motion.loc[E1_idx:E5_idx, "L.Wrist.Rad_y"].values,
                                                 color=colors[3], label = "L.Wrist.Rad")
                                    axes[1].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # T10
                                                 filted_motion.loc[E1_idx:E5_idx, "T10_y"].values,
                                                 color=colors[4], label = "T10")
                                    axes[1].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # Middle
                                                 filted_motion.loc[E1_idx:E5_idx, "Middle_y"].values,
                                                 color=colors[5], label = "Middle")
                                    # E1 劃分期線
                                    axes[1].axvline(filted_motion.loc[E1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[1].text(filted_motion.loc[E1_idx, 'Frame'],
                                                 axes[1].get_ylim()[1], f"E1:{filted_motion.loc[E1_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E2 劃分期線
                                    axes[1].axvline(filted_motion.loc[E2_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[1].text(filted_motion.loc[E2_idx, 'Frame'],
                                                 axes[1].get_ylim()[1], f"E2:{filted_motion.loc[E2_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-1 劃分期線
                                    axes[1].axvline(filted_motion.loc[E3_1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[1].text(filted_motion.loc[E3_1_idx, 'Frame'],
                                                 axes[1].get_ylim()[1], f"E3-1:{filted_motion.loc[E3_1_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-2 劃分期線
                                    axes[1].axvline(filted_motion.loc[E3_2_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[1].text(filted_motion.loc[E3_2_idx, 'Frame'] + 0.9,
                                                 axes[1].get_ylim()[1], f"E3-2:{filted_motion.loc[E3_2_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E4 劃分期線
                                    axes[1].axvline(filted_motion.loc[E4_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[1].text(filted_motion.loc[E4_idx, 'Frame']-1,
                                                 axes[1].get_ylim()[1], f"E4:{filted_motion.loc[E4_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E5 劃分期線
                                    axes[1].axvline(filted_motion.loc[E5_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[1].text(filted_motion.loc[E5_idx, 'Frame'],
                                                 axes[1].get_ylim()[1],  f"E5:{filted_motion.loc[E5_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    axes[1].set_xlim(filted_motion.loc[E1_idx, 'Frame'],
                                                     filted_motion.loc[E5_idx, 'Frame'])
                                    # 子圖資訊設定，坐標軸
                                    axes[1].set_ylabel('Y 軸 data', fontsize = 14)# 设置子图标题
                                    # 繪製第三個子圖 Z 軸 --------------------------------------------------
                                    # 繪製 motion data
                                    axes[2].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # R.Wrist.Rad
                                                 filted_motion.loc[E1_idx:E5_idx, "R.Wrist.Rad_z"].values,
                                                 color=colors[0], label = "R.Wrist.Rad")
                                    axes[2].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # R.Epi.Lat
                                                 filted_motion.loc[E1_idx:E5_idx, "R.Epi.Lat_z"].values,
                                                 color=colors[1], label = "R.Epi.Lat")
                                    axes[2].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # L.Acromion
                                                 filted_motion.loc[E1_idx:E5_idx, "L.Acromion_z"].values,
                                                 color=colors[2], label = "L.Acromion")
                                    axes[2].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # L.Wrist.Rad
                                                 filted_motion.loc[E1_idx:E5_idx, "L.Wrist.Rad_z"].values,
                                                 color=colors[3], label = "L.Wrist.Rad")
                                    axes[2].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # T10
                                                 filted_motion.loc[E1_idx:E5_idx, "T10_z"].values,
                                                 color=colors[4], label = "T10")
                                    axes[2].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # Middle
                                                 filted_motion.loc[E1_idx:E5_idx, "Middle_z"].values,
                                                 color=colors[5], label = "Middle")
                                    # E1 劃分期線
                                    axes[2].axvline(filted_motion.loc[E1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[2].text(filted_motion.loc[E1_idx, 'Frame'],
                                                 axes[2].get_ylim()[1], f"E1:{filted_motion.loc[E1_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E2 劃分期線
                                    axes[2].axvline(filted_motion.loc[E2_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[2].text(filted_motion.loc[E2_idx, 'Frame'],
                                                 axes[2].get_ylim()[1], f"E2:{filted_motion.loc[E2_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-1 劃分期線
                                    axes[2].axvline(filted_motion.loc[E3_1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[2].text(filted_motion.loc[E3_1_idx, 'Frame'],
                                                 axes[2].get_ylim()[1], f"E3-1:{filted_motion.loc[E3_1_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-2 劃分期線
                                    axes[2].axvline(filted_motion.loc[E3_2_idx, 'Frame'] ,
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[2].text(filted_motion.loc[E3_2_idx, 'Frame'] + 0.9,
                                                 axes[2].get_ylim()[1], f"E3-2:{filted_motion.loc[E3_2_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E4 劃分期線
                                    axes[2].axvline(filted_motion.loc[E4_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[2].text(filted_motion.loc[E4_idx, 'Frame']-1,
                                                 axes[2].get_ylim()[1], f"E4:{filted_motion.loc[E4_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E5 劃分期線
                                    axes[2].axvline(filted_motion.loc[E5_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[2].text(filted_motion.loc[E5_idx, 'Frame'],
                                                 axes[2].get_ylim()[1],  f"E5:{filted_motion.loc[E5_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    axes[2].set_xlim(filted_motion.loc[E1_idx, 'Frame'],
                                                     filted_motion.loc[E5_idx, 'Frame'])
                                    # 子圖資訊設定，坐標軸
                                    axes[2].set_ylabel('z 軸 data', fontsize = 14)  # 设置子图标题
                                    # 添加整体标题
                                    plt.suptitle(tempfilename)  # 设置整体标题
                                    # 调整子图之间的间距
                                    plt.tight_layout()
                                    plt.xlabel("time (second)", fontsize = 14)
                                    fig.text(0.09, 0.5, 'mm', va='center', rotation='vertical', fontsize=16)
                                    # plt.ylabel("mm", fontsize = 14)
                                    # 在主图外部添加图例
                                    fig.legend(labels=labels, loc='center left', bbox_to_anchor=(0.98, 0.5))

                                    # 使用 tight_layout 调整子图布局
                                    # plt.tight_layout(rect=[0, 0, 0.85, 1])
                                    # plt.legend(labels=labels, loc='lower center', ncol=3, fontsize=12)
                                    # plt.tight_layout()
                                    # 调整布局以防止重叠，并为图例腾出空间
                                    plt.tight_layout(rect=[0.1, 0.04, 1, 1])
                                    plt.savefig(fig_save + "\\angle\\" + filename + "_stage.jpg",
                                                dpi=100)
                                    # 显示图形
                                    plt.show()
                                    # 7.2. 舉弓角度畫圖 ------------------------------------------------------------------------------
                                    # -----------------------------------------------------------------------------------------------
                                    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex='col')
                                    
                                    # 繪製第一個子圖 X 軸: R.Wrist.Rad, R.Elbow.Lat, L.Acromin, L.Wrist.Rad, T10, Middle 
                                    # 繪製 motion data
                                    axes[0].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # L.Wrist.Rad
                                                 filted_motion.loc[E1_idx:E5_idx, "L.Wrist.Rad_x"].values,
                                                 color=colors[3], label = "L.Wrist.Rad")
                                    axes[0].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # T10
                                                 filted_motion.loc[E1_idx:E5_idx, "T10_x"].values,
                                                 color=colors[4], label = "T10")
                                    # E1 劃分期線
                                    axes[0].axvline(filted_motion.loc[E1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[0].text(filted_motion.loc[E1_idx, 'Frame'],
                                                 axes[0].get_ylim()[1], f"E1:{filted_motion.loc[E1_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E2 劃分期線
                                    axes[0].axvline(filted_motion.loc[E2_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[0].text(filted_motion.loc[E2_idx, 'Frame'],
                                                 axes[0].get_ylim()[1], f"E2:{filted_motion.loc[E2_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-1 劃分期線
                                    axes[0].axvline(filted_motion.loc[E3_1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[0].text(filted_motion.loc[E3_1_idx, 'Frame'],
                                                 axes[0].get_ylim()[1], f"E3-1:{filted_motion.loc[E3_1_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-2 劃分期線
                                    axes[0].axvline(filted_motion.loc[E3_2_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[0].text(filted_motion.loc[E3_2_idx, 'Frame'] + 0.9,
                                                 axes[0].get_ylim()[1], f"E3-2:{filted_motion.loc[E3_2_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E4 劃分期線
                                    axes[0].axvline(filted_motion.loc[E4_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[0].text(filted_motion.loc[E4_idx, 'Frame']-1,
                                                 axes[0].get_ylim()[1], f"E4:{filted_motion.loc[E4_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E5 劃分期線
                                    axes[0].axvline(filted_motion.loc[E5_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[0].text(filted_motion.loc[E5_idx, 'Frame'],
                                                 axes[0].get_ylim()[1],  f"E5:{filted_motion.loc[E5_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # 子圖資訊設定，坐標軸
                                    axes[0].set_ylabel('X 軸 (mm)', fontsize = 14)  # 设置子图标题
                                    axes[0].legend(loc="upper right")
                                    # 繪製第二個子圖 Y 軸 ---------------------------
                                    # 繪製 motion data
                                    axes[1].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # L.Wrist.Rad
                                                 filted_motion.loc[E1_idx:E5_idx, "L.Wrist.Rad_y"].values,
                                                 color=colors[3], label = "L.Wrist.Rad")
                                    axes[1].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # T10
                                                 filted_motion.loc[E1_idx:E5_idx, "T10_y"].values,
                                                 color=colors[4], label = "T10")
                                    # E1 劃分期線
                                    axes[1].axvline(filted_motion.loc[E1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[1].text(filted_motion.loc[E1_idx, 'Frame'],
                                                 axes[1].get_ylim()[1], f"E1:{filted_motion.loc[E1_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E2 劃分期線
                                    axes[1].axvline(filted_motion.loc[E2_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[1].text(filted_motion.loc[E2_idx, 'Frame'],
                                                 axes[1].get_ylim()[1], f"E2:{filted_motion.loc[E2_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-1 劃分期線
                                    axes[1].axvline(filted_motion.loc[E3_1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[1].text(filted_motion.loc[E3_1_idx, 'Frame'],
                                                 axes[1].get_ylim()[1], f"E3-1:{filted_motion.loc[E3_1_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-2 劃分期線
                                    axes[1].axvline(filted_motion.loc[E3_2_idx, 'Frame'] ,
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[1].text(filted_motion.loc[E3_2_idx, 'Frame'] + 0.9,
                                                 axes[1].get_ylim()[1], f"E3-2:{filted_motion.loc[E3_2_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E4 劃分期線
                                    axes[1].axvline(filted_motion.loc[E4_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[1].text(filted_motion.loc[E4_idx, 'Frame']-1,
                                                 axes[1].get_ylim()[1], f"E4:{filted_motion.loc[E4_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E5 劃分期線
                                    axes[1].axvline(filted_motion.loc[E5_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[1].text(filted_motion.loc[E5_idx, 'Frame'],
                                                 axes[1].get_ylim()[1],  f"E5:{filted_motion.loc[E5_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    axes[1].set_xlim(filted_motion.loc[E1_idx, 'Frame'],
                                                     filted_motion.loc[E5_idx, 'Frame'])
                                    # 子圖資訊設定，坐標軸
                                    axes[1].set_ylabel('Y 軸 (mm)', fontsize = 14)# 设置子图标题
                                    axes[1].legend(loc="upper right")
                                    # 繪製第三個子圖 Z 軸 ------------------------------------------------
                                    # 繪製 motion data
                                    axes[2].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # L.Wrist.Rad
                                                 filted_motion.loc[E1_idx:E5_idx, "L.Wrist.Rad_z"].values,
                                                 color=colors[3], label = "L.Wrist.Rad")
                                    axes[2].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # T10
                                                 filted_motion.loc[E1_idx:E5_idx, "T10_z"].values,
                                                 color=colors[4], label = "T10")
                                    # E1 劃分期線
                                    axes[2].axvline(filted_motion.loc[E1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[2].text(filted_motion.loc[E1_idx, 'Frame'],
                                                 axes[2].get_ylim()[1], f"E1:{filted_motion.loc[E1_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E2 劃分期線
                                    axes[2].axvline(filted_motion.loc[E2_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[2].text(filted_motion.loc[E2_idx, 'Frame'],
                                                 axes[2].get_ylim()[1], f"E2:{filted_motion.loc[E2_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-1 劃分期線
                                    axes[2].axvline(filted_motion.loc[E3_1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[2].text(filted_motion.loc[E3_1_idx, 'Frame'],
                                                 axes[2].get_ylim()[1], f"E3-1:{filted_motion.loc[E3_1_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-2 劃分期線
                                    axes[2].axvline(filted_motion.loc[E3_2_idx, 'Frame'] ,
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[2].text(filted_motion.loc[E3_2_idx, 'Frame'] + 0.9,
                                                 axes[2].get_ylim()[1], f"E3-2:{filted_motion.loc[E3_2_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E4 劃分期線
                                    axes[2].axvline(filted_motion.loc[E4_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[2].text(filted_motion.loc[E4_idx, 'Frame']-1,
                                                 axes[2].get_ylim()[1], f"E4:{filted_motion.loc[E4_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E5 劃分期線
                                    axes[2].axvline(filted_motion.loc[E5_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[2].text(filted_motion.loc[E5_idx, 'Frame'],
                                                 axes[2].get_ylim()[1],  f"E5:{filted_motion.loc[E5_idx, 'Frame']:.2f}s",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    axes[2].set_xlim(filted_motion.loc[E1_idx, 'Frame'],
                                                     filted_motion.loc[E5_idx, 'Frame'])
                                    # 子圖資訊設定，坐標軸
                                    axes[2].set_ylabel('z 軸 (mm)', fontsize = 14)  # 设置子图标题
                                    axes[2].legend(loc="upper right")
                                    # 繪製第四個子圖 ------------------------------------------------------------------
                                    axes[3].plot(filted_motion.loc[E1_idx:E5_idx, 'Frame'].values, # L.Wrist.Rad
                                                 mAG,
                                                 color=colors[3], label = "L.Wrist.Rad")
                                    # E2 劃分期線
                                    axes[3].axvline(filted_motion.loc[E2_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[3].text(filted_motion.loc[E2_idx, 'Frame'],
                                                 axes[3].get_ylim()[1], f"E2:{mAG[E2_idx-E1_idx]:.2f}$^o$",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-1 劃分期線
                                    axes[3].axvline(filted_motion.loc[E3_1_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[3].text(filted_motion.loc[E3_1_idx, 'Frame'],
                                                 axes[3].get_ylim()[1], f"E3-1:{mAG[E3_1_idx-E1_idx]:.2f}$^o$",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E3-2 劃分期線
                                    axes[3].axvline(filted_motion.loc[E3_2_idx, 'Frame'] ,
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[3].text(filted_motion.loc[E3_2_idx, 'Frame'] + 0.9,
                                                 axes[3].get_ylim()[1], f"E3-2:{mAG[E3_2_idx-E1_idx]:.2f}$^o$",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    # E4 劃分期線
                                    axes[3].axvline(filted_motion.loc[E4_idx, 'Frame'],
                                                    color='r', linestyle='--', linewidth=0.5) # trigger onset
                                    # 添加标注
                                    axes[3].text(filted_motion.loc[E4_idx, 'Frame']-1,
                                                 axes[3].get_ylim()[1], f"E4:{mAG[E4_idx-E1_idx]:.2f}$^o$",
                                                 color='r', fontsize=10, ha='center', va='bottom')
                                    axes[3].set_ylabel('舉弓角度 (deg)', fontsize = 14)
                                    # 框出最大值
                                    axes[3].plot(filted_motion.loc[(np.argmax(mAG) + E1_idx), 'Frame'],
                                                np.max(mAG), # 右腳離地時間
                                                marker = 'o', ms = 10, mec='r', mfc='none')
                                    axes[3].text(filted_motion.loc[(np.argmax(mAG) + E1_idx), 'Frame'],
                                                 np.max(mAG)-5, f"Max:{np.max(mAG):.2f}$^o$",
                                                 color='r', fontsize=12)
                                    axes[3].text(filted_motion.loc[(np.argmax(mAG) + E1_idx), 'Frame'],
                                                 np.max(mAG)-8, f"Max time:{filted_motion.loc[(np.argmax(mAG) + E1_idx), 'Frame']:.2f}s",
                                                 color='r', fontsize=12)
                                     
                                    # 添加整体标题
                                    plt.suptitle(str("舉弓角度運算: " + tempfilename))  # 设置整体标题
                                    # 调整子图之间的间距
                                    plt.tight_layout()
                                    plt.xlabel("time (second)", fontsize = 14)
                                    # 调整布局以防止重叠，并为图例腾出空间
                                    # plt.tight_layout(rect=[0.1, 0.04, 1, 1])
                                    plt.savefig(fig_save + "\\angle\\" + filename + "_release.jpg",
                                                dpi=100)
                                    # 显示图形
                                    plt.show()

all_algorithm.to_excel(data_path + "_algorithm_output_formatted_date" + ".xlsx")

# %% 資料前處理 : bandpass filter, absolute value, smoothing




























