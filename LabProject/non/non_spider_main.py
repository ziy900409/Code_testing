# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:48:56 2024

@author: Hsin.YH.Yang
"""


# %% import package
import sys
# 路徑改成你放自己code的資料夾
# sys.path.append(r"E:\Hsin\git\git\Code_testing\LabProject\function")
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\function")
import gen_function as func
import Kinematic_function as kincal
import plotFig_function as FigPlot
import EMG_function as emg

import os
import numpy as np
import pandas as pd
import signal

from detecta import detect_onset
import gc
import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy import signal
# import math
# %% 路徑設置
folder_path = r"E:\Hsin\BenQ\ZOWIE non-sym\\"
motion_folder = "1.motion\\"
emg_folder = "3.EMG\\ "
subfolder = "2.LargeFlick\\"
motion_type = ["Cortex\\", "Vicon\\"]

cortex_folder = ["S11", "S12", "S13", "S14", "S15",
                 "S16", "S17", "S18", "S19", #"S20",
                 "S21"]

vicon_folder = ["S03", " S04"]

RawData_folder = ""
processingData_folder = "4.process_data"

motion_folder_path = folder_path + motion_folder
emg_folder_path = folder_path + emg_folder

# results_save_path = r"E:\Hsin\BenQ\ZOWIE non-sym\4.process_data\\"

stage_file_path = r"E:\Hsin\BenQ\ZOWIE non-sym\ZowieNonSymmetry_StagingFile_20240929.xlsx"
# 取得所有 motion data folder list
# 去除有“.“開頭的檔案 and 只獲得資料夾路徑，排除其他可能的檔案格式
motion_folder_list = []
for sub in motion_type:
    motion_folder_list = motion_folder_list + \
        [sub + f for f in os.listdir(motion_folder_path + sub) if not f.startswith('.') \
         and os.path.isdir(os.path.join((motion_folder_path + sub), f))]
# 取得所有 processing data folder list
processing_folder_path = folder_path + "\\" + processingData_folder + "\\"
processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.') \
                          and os.path.isdir(os.path.join(processing_folder_path, f))]


# %% parameter setting
# analog data begin threshold
ana_threshold = 4
# joint angular threshold
# threshold equal the proportion of the maximum value 
joint_threshold = 0.5

# %% cortex version

# 在不同的受試者資料夾下執行
for folder_name in cortex_folder:
    # 讀資料夾下的 c3d file
    c3d_list = func.Read_File(folder_path + motion_folder + motion_type[0] + folder_name, ".c3d")
    # 1.1.1. 讀 all 分期檔
    excel_sheetr_name = folder_name.replace("Cortex\\", "").replace("Vicon\\", "")
    all_table = pd.read_excel(stage_file_path,
                              sheet_name=excel_sheetr_name)
    # 第一次loop先計算tpose的問題
    for num in range(len(c3d_list)):
        for i in range(len(all_table['Motion_File_C3D'])):
            if c3d_list[num].split('\\')[-1].replace(".c3d", "") == all_table['Motion_File_C3D'][i]:
                # 1.2.0. ---------使用tpose計算 手肘內上髁 Virtual marker 之位置 --------------
                if "tpose" in c3d_list[num].lower() and all_table["Task"][i] == "Tpose_elbow":
                    print(c3d_list[num])
                    motion_info, motion_data, analog_info, FP_data, np_motion_data = func.read_c3d(c3d_list[num])
                    # 1. 設定輸入計算 Virtual marker 參數 : 手肘內上髁, 外上髁, UA1, UA3
                    R_Elbow_Med = motion_data.loc[:, "R.Elbow.Med_x":"R.Elbow.Med_z"].dropna(axis=0)
                    R_Elbow_Lat = motion_data.loc[:, "R.Elbow.Lat_x":"R.Elbow.Lat_z"].dropna(axis=0)
                    UA1 = motion_data.loc[:, "UA1_x":"UA1_z"].dropna(axis=0)
                    UA3 = motion_data.loc[:, "UA3_x":"UA3_z"].dropna(axis=0)
                    # 2. 避免數量中出現NAN，請造成不同變數見長短不一致，因此找出最短數列的 index
                    ind_frame = min([np.shape(R_Elbow_Med)[0], np.shape(R_Elbow_Lat)[0], np.shape(UA1)[0], np.shape(UA3)[0]])
                    for ii in [R_Elbow_Med, R_Elbow_Lat, UA1, UA3]:
                        if ind_frame == np.shape(ii)[0]:
                            ind_frame = ii.index
                            break
                    # 3. 計算手肘內上髁在 LCS 之位置
                    p1_all = pd.DataFrame(np.zeros([len(ind_frame), 3]))
                    for frame in ind_frame:
                        p1_all.iloc[frame :] = (kincal.transformation_matrix(R_Elbow_Lat.iloc[frame, :].values, UA1.iloc[frame, :].values, UA3.iloc[frame, :].values,
                                                                            R_Elbow_Med.iloc[frame, :].values, np.array([0, 0, 0]),
                                                                            rotation='GCStoLCS'))
                    # 4. 清除不需要的變數
                    del motion_info, motion_data, analog_info, FP_data, R_Elbow_Med, UA1, UA3, np_motion_data
                    gc.collect()
                if "tpose" in c3d_list[num].lower() and all_table["Task"][i] == "Tpose_hand":
                    print(c3d_list[num])
                    index = 1
                    static_ArmCoord, static_ForearmCoord, static_HandCoord = kincal.arm_natural_pos(c3d_list[num], p1_all, index)

# %% 將 .c3d convert to .trc
'''
未來須修正地方 : 分成是否要套入opensim之格式
1. truncate data :
    1.1. 讀分期檔 :
        1.1.1. 讀 all 分期檔
        1.1.2. read_csv Fitts merge file
        1.1.3. 讀所有 path of motion file 留下所有 Fitts law 檔名
    1.2. read c3d

    1.2. judge analog data : find +5V position， 前後裁切，不需保留完整資料
        1.2.0. 使用tpose計算 手肘內上髁 Virtual marker 之位置 
        1.2.1. 找 Analog data 的峰值 
        1.2.2. Fitts : 使用staging data 的數值做
        1.2.3. Spyder, Blink :
    
2. 寫入資料 :
    2.1. 設定與寫入標頭檔 (前六行)
    2.2. 寫入資料
'''
motion_info, motion_data, analog_info, FP_data, np_motion_data = func.read_c3d(r"E:\Hsin\BenQ\ZOWIE non-sym\1.motion\Vicon\S03\S03_DorInter_1st_MVC.c3d")
    
# 在不同的受試者資料夾下執行
for folder_name in motion_folder_list:
    # 讀資料夾下的 c3d file
    c3d_list = func.Read_File(folder_path + motion_folder + folder_name, ".c3d")
    # 1.1.1. 讀 all 分期檔
    excel_sheetr_name = folder_name.replace("Cortex\\", "").replace("Vicon\\", "")
    all_table = pd.read_excel(stage_file_path,
                              sheet_name=excel_sheetr_name)
    # 第一次loop先計算tpose的問題
    for num in range(len(c3d_list)):
        for i in range(len(all_table['c3d'])):
            if c3d_list[num].split('\\')[-1] == all_table['c3d'][i]:
                # 1.2.0. ---------使用tpose計算 手肘內上髁 Virtual marker 之位置 --------------
                if "tpose" in c3d_list[num].lower() and all_table["Task"][i] == "Tpose_elbow":
                    print(c3d_list[num])
                    motion_info, motion_data, analog_info, FP_data, np_motion_data = func.read_c3d(c3d_list[num])
                    # 1. 設定輸入計算 Virtual marker 參數 : 手肘內上髁, 外上髁, UA1, UA3
                    R_Elbow_Med = motion_data.loc[:, "EC2 Wight_Elbow:R.Elbow.Med_x":"EC2 Wight_Elbow:R.Elbow.Med_z"].dropna(axis=0)
                    R_Elbow_Lat = motion_data.loc[:, "EC2 Wight_Elbow:R.Elbow.Lat_x":"EC2 Wight_Elbow:R.Elbow.Lat_z"].dropna(axis=0)
                    UA1 = motion_data.loc[:, "EC2 Wight_Elbow:UA1_x":"EC2 Wight_Elbow:UA1_z"].dropna(axis=0)
                    UA3 = motion_data.loc[:, "EC2 Wight_Elbow:UA3_x":"EC2 Wight_Elbow:UA3_z"].dropna(axis=0)
                    # 2. 避免數量中出現NAN，請造成不同變數見長短不一致，因此找出最短數列的 index
                    ind_frame = min([np.shape(R_Elbow_Med)[0], np.shape(R_Elbow_Lat)[0], np.shape(UA1)[0], np.shape(UA3)[0]])
                    for i in [R_Elbow_Med, R_Elbow_Lat, UA1, UA3]:
                        if ind_frame == np.shape(i)[0]:
                            ind_frame = i.index
                            break
                    # 3. 計算手肘內上髁在 LCS 之位置
                    p1_all = pd.DataFrame(np.zeros([len(ind_frame), 3]))
                    for frame in ind_frame:
                        p1_all.iloc[frame :] = (func.transformation_matrix(R_Elbow_Lat.iloc[frame, :].values, UA1.iloc[frame, :].values, UA3.iloc[frame, :].values,
                                                                            R_Elbow_Med.iloc[frame, :].values, np.array([0, 0, 0]),
                                                                            rotation='GCStoLCS'))
                    # 4. 清除不需要的變數
                    del motion_info, motion_data, analog_info, FP_data, R_Elbow_Med, UA1, UA3, np_motion_data
                    gc.collect()
    # 第二次loop計算motion
    for num in range(len(c3d_list)):
        for i in range(len(all_table['c3d'])):
            if c3d_list[num].split('\\')[-1] == all_table['c3d'][i]:
                # 1.2.1. ---------找到Analog data 中 trigger 的起始時間-----------------------------
                if "Fitts" in c3d_list[num] or "Blink" in c3d_list[num] or "Spider" in c3d_list[num]:
                    print(c3d_list[num])
                    # 1. read c3d file
                    motion_info, motion_data, analog_info, analog_data, np_motion_data = func.read_c3d(c3d_list[num])
                    # 2. find peak with threshold (please parameter setting)
                    peaks, _ = find_peaks(analog_data.loc[:, "trigger1"], height=ana_threshold)
                    # 繪出 analog data 的起始時間
                    plt.plot(analog_data.loc[:, "Frame"], analog_data.loc[:, "trigger1"], label='Signal')
                    plt.plot(analog_data.loc[peaks, "Frame"], analog_data.loc[peaks, "trigger1"], 'ro', label='Peaks')
                    plt.legend()
                    plt.show()
                    # 3. 找出 analog, motion 兩個時間最接近的 frame, 並定義 start index
                    for ii, x in enumerate(motion_data.loc[:, "Frame"]):
                        # 每個 frame 減去 peak value，以找到最接近列表的索引
                        if abs(motion_data.loc[ii, "Frame"] - analog_data.loc[peaks, "Frame"].values) == \
                            min(abs(motion_data.loc[:, "Frame"] - analog_data.loc[peaks, "Frame"].values)):
                            # 定義 motion data 的起始索引
                            motion_start_ind = ii
                            print("Analog", analog_data.loc[peaks, "Frame"])
                            print("Frame", ii)
                            break
                    # 1.2.2. ---------Fitts law data 處理-----------------------------
                    # 計算開始與結束的索引位置
                    # 判斷是否為 Fitts data 並且 Fitts end 有數值
                    if "Fitts" in all_table['c3d'][i] and all_table['Fitts_end'][num]:
                        print(all_table['c3d'][i])
                        for ii, x in enumerate(motion_data.loc[:, "Frame"]):
                            # 利用 staging file 判斷截止時間
                            # 結束時間 = Fitts所花時間 + analog 開始時間
                            Fitts_end_time = (all_table['Fitts_end'][i] / 1000 + motion_start_ind/int(motion_info["frame_rate"]))
                            # 每個 frame 減去 end value，以找到最接近列表的索引
                            if abs(motion_data.loc[ii, "Frame"] - Fitts_end_time) == \
                                min(abs(motion_data.loc[:, "Frame"] - Fitts_end_time)):
                                # print(all_table['Fitts_end'][i])
                                # 定義 motion data 的結束索引
                                motion_end_ind = ii
                                print("end", ii)
                                break
                    # 1.2.3. ---------Blink 與 Spyder 資料處理-----------------------------
                    # 計算開始與結束的索引位置
                    elif "Blink" in all_table['c3d'][i] or "Spider" in all_table['c3d'][i]:
                        print(0)
                        # 開始索引從analog begin後加三秒
                        task_start_ind = int(motion_start_ind + motion_info["frame_rate"]*5)
                        # 開始索引從 motion start 加 58 秒 (多截斷一些數值，避免動作上的誤差)
                        task_end_ind = int(motion_start_ind + motion_info["frame_rate"]*(3+55))
                    # 1.2.4. ---------truncate motion data--------------------------------
                    trun_motion = np_motion_data[:, task_start_ind:task_end_ind, :]

                    # 建立手肘內上髁的資料貯存位置
                    V_R_Elbow_Med = np.zeros(shape=(1, np.shape(trun_motion)[1], np.shape(trun_motion)[2]))
                    # 找出以下三個字串的索引值
                    target_strings = ["EC2 Wight:R.Elbow.Lat", "EC2 Wight:UA1"
                                      , "EC2 Wight:UA3", "EC2 Wight:R.Shoulder"]
                    indices = []
                    for target_str in target_strings:
                        try:
                            index = motion_info["LABELS"].index(target_str)
                            indices.append(index)
                        except ValueError:
                            indices.append(None)
                    # 回算手肘內上髁在 GCS 之位置
                    for frame in range(np.shape(trun_motion)[1]):
                        V_R_Elbow_Med[0, frame, :] = func.transformation_matrix(trun_motion[indices[0], frame, :], # EC2 Wight:R.Elbow.Lat
                                                                                trun_motion[indices[1], frame, :], # EC2 Wight:UA1
                                                                                trun_motion[indices[2], frame, :], # EC2 Wight:UA3
                                                                                p1_all.iloc[5, :].values, np.array([0, 0, 0]),
                                                                                rotation='LCStoGCS')
                    # 合併 motion data and virtual R.Elbow.Med data
                    new_trun_motion = np.concatenate((trun_motion, V_R_Elbow_Med), axis=0)
                    # motion_info 新增 R.Elbow.Med 的標籤
                    # motion_info, motion_data, analog_info, analog_data, np_motion_data = read_c3d(r"E:\Motion Analysis\U3 Research\S01\S01_1VS1_1.c3d")
                    motion_info['LABELS'].append("EC2 Wight:R.Elbow.Med")
                    # 去掉LABELS中關於Cortex Marker set 之資訊 -> 去掉 EC2 Wight:
                    for label in range(len(motion_info['LABELS'])):
                        motion_info['LABELS'][label] = motion_info['LABELS'][label].replace("EC2 Wight:", "")
                    # 去除掉 Virtual marker 的 LABELS 與 motion data
                    new_labels = []
                    np_labels = []
                    for key, item in enumerate(motion_info['LABELS']):
                        # print(key, item)
                        # 要去除的特定字符 : V_marker
                        if "V_" not in item:
                            new_labels.append(item)
                            np_labels.append(key) # add time and Frame
                    motion_info['LABELS'] = new_labels
                    # 重新定義 motion data
                    # new_motion_data = motion_data.iloc[:, new_ind]
                    new_np_motion_data = new_trun_motion[np_labels, :, :]
                    # 2023.08.11. 不再修正檔案為opensim可用之格式
                    # # 進opensim，所有marker以R.shoulder做平移
                    # dis_np_motion_data = np.empty(shape=np.shape(new_np_motion_data))
                    # for iii in range(np.shape(new_np_motion_data)[0]):
                    #     dis_np_motion_data[iii, :, :] = new_np_motion_data[iii, :, :] - (new_np_motion_data[indices[3], :, :]-np.array([57.73, -56.75, 15.2]))
                    # # 低通濾波 butterworth filter
                    bandpass_filtered = np.empty(shape=np.shape(new_np_motion_data))
                    bandpass_sos = signal.butter(2, 20/0.802,  btype='lowpass', fs=motion_info["frame_rate"], output='sos')
                    for iii in range(np.shape(new_np_motion_data)[0]):
                        for iiii in range(np.shape(new_np_motion_data)[2]):
                            bandpass_filtered[iii, :, iiii] = signal.sosfiltfilt(bandpass_sos,
                                                                                  new_np_motion_data[iii, :, iiii])
                    # 將副檔名從.c3d 改成 .trc
                    trc_name = all_table['c3d'][i].split('.')[0] + '.trc'
                    # 資料貯存路徑 + folder name
                    func.c3d_to_trc(str(motion_path + folder_name + "\\" + trc_name),
                                    trc_name, new_np_motion_data, motion_info)
                    # 清除不必要的空間
                    gc.collect()
                  
# %% 計算 spider

'''
Spider
1. 找肘與腕關節角度的峰值，無論方向，都找出該峰值點位對應EMG的前後55ms，並計算平均數與最大值
main function
未來須修正地方 : 分成是否要套入opensim之格式
1. truncate data :
    1.1. 讀分期檔 :
        1.1.1. 讀 all 分期檔
        1.1.2. read_csv Fitts merge file
        1.1.3. 讀所有 path of motion file 留下所有 Fitts law 檔名
    1.2. read c3d

    1.2. judge analog data : find +5V position， 前後裁切，不需保留完整資料
        1.2.0. 使用tpose計算 手肘內上髁 Virtual marker 之位置 
        1.2.1. 找 Analog data 的峰值 
        1.2.2. Fitts : 使用staging data 的數值做
        1.2.3. Spyder, Blink :
    
2. 寫入資料 :
    2.1. 設定與寫入標頭檔 (前六行)
    2.2. 寫入資料
'''


# 在不同的受試者資料夾下執行
for folder_name in motion_folder_list:
    # 1.1.1. 讀 all 分期檔
    all_table = pd.read_excel(r"E:\BenQ_Project\U3\07_EMGrecording\U3-research_staging_v1.xlsx",
                              sheet_name=folder_name)
    Fitts_table = pd.read_csv(Fitts_path + folder_name + "\\" + folder_name + ".csv")
    c3d_list = func.Read_File(motion_path + folder_name, ".c3d")
    # 創建資料貯存空間
    all_motion_data_table = pd.DataFrame({}, columns = ['檔名', '位置', 'method',
                                             'Add-平均', 'Add-最大值',
                                             'Abd-平均', 'Abd-最大值',
                                             'Pro-平均', 'Pro-最大值',
                                             'Sup-平均', 'Sup-最大值',
                                             'Flex-平均', 'Flex-最大值',
                                             'Ext-平均', 'Ext-最大值'])

    all_emg_data_table = pd.DataFrame({}, columns = ['檔名', 'direction', 'axis', 'type',
                                                 'Dor1st-mean', 'Dor1st-max', 'Dor3rd-mean', 'Dor3rd-max',
                                                 'abduciton-mean', 'abduciton-max', 'indict-mean', 'indict-max',
                                                 'ExtR-mean', 'ExtR-max', 'ExtU-mean', 'ExtU-max',
                                                 'FlexR-mean', 'FlexR-max', 'Biceps-mean', 'Biceps-max'])

    all_median_freq_table = pd.DataFrame({}, columns=(["filename", "columns_num"] + [str(i) for i in range(47)]))
    hand_angle_table = pd.DataFrame({}, columns=["filename","CMP1",
                                                 "CMP2", "PIP2",
                                                 "CMP3",
                                                 "CMP4", "CMP5"])
    # ------第一次loop先計算tpose的問題------------------------------------------------
    for num in range(len(c3d_list)):
        for i in range(len(all_table['c3d'])):
            if c3d_list[num].split('\\')[-1] == all_table['c3d'][i]:
                # 1.2.0. ---------使用tpose計算 手肘內上髁 Virtual marker 之位置 --------------
                if "tpose" in c3d_list[num].lower() and all_table["mouse"][i] == "elbow":
                    print(c3d_list[num])
                    p1_all = pro.V_Elbow_cal(c3d_list[num])

                # 1.2.0. ---------使用tpose計算手部的自然關節角度
                elif "tpose" in c3d_list[num].lower() and all_table["mouse"][i] == "hand":
                    print(c3d_list[num])
                    index = all_table["motion"][i] - 1
                    static_ArmCoord, static_ForearmCoord, static_HandCoord = pro.arm_natural_pos(c3d_list[num], p1_all, index)
    # 讀取MVC最大值
    MVC_value = pd.read_excel(r"E:\BenQ_Project\U3\07_EMGrecording\processing_data\\" + folder_name + '\\' + folder_name + '_all_MVC.xlsx')
    MVC_value = MVC_value.iloc[-1, 2:]
    # ------第二次loop計算motion-----------------------------------------------------------------
    for num in range(len(c3d_list)):
        for i in range(len(all_table['c3d'])):
            if c3d_list[num].split('\\')[-1] == all_table['c3d'][i]:
                # 1.2.1. ---------找到Analog data 中 trigger 的起始時間-----------------------------
                if "Spider" in c3d_list[num]:
                    print(c3d_list[num])
                    # print(all_table['EMG'][i])
                    # 1. read c3d file
                    motion_info, motion_data, analog_info, analog_data, np_motion_data = func.read_c3d(c3d_list[num])
                    # 2. find peak with threshold (please parameter setting)
                    peaks = detect_onset(analog_data.loc[:, "trigger1"], np.std(analog_data.loc[:, "trigger1"][:18]),
                                         n_above=10, n_below=0, show=True)
                    # 3. 找出 analog, motion 兩個時間最接近的 frame, 並定義 start index
                    for ii, x in enumerate(motion_data.loc[:, "Frame"]):
                        # 每個 frame 減去 peak value，以找到最接近列表的索引
                        if abs(motion_data.loc[ii, "Frame"] - analog_data.loc[peaks[0][0], "Frame"]) == \
                            min(abs(motion_data.loc[:, "Frame"] - analog_data.loc[peaks[0][0], "Frame"])):
                            # 定義 motion data 的起始索引
                            motion_start_ind = ii
                            print("Analog", analog_data.loc[peaks[0][0], "Frame"])
                            print("Frame", ii)
                            break
                    # 開始索引從analog begin後加 8 秒
                    task_start_ind = int(motion_start_ind + motion_info["frame_rate"]*8)
                    # 開始索引從 motion start 加 52 秒 (多截斷一些數值，避免動作上的誤差)
                    task_end_ind = int(motion_start_ind + motion_info["frame_rate"]*(8+52))
                    # 1.2.4. ---------truncate motion data and EMG data--------------------------------
                    # 初步處理資料 
                    # motion 去除不必要的Virtual marker, 裁切, 計算坐標軸旋轉矩陣
                    # EMG 前處理, 計算iMVC, 裁切
                    trun_motion = np_motion_data[:, task_start_ind:task_end_ind, :]
                    # --------------------EMG 資料處理------------------------
                    emg_data = pd.read_csv(str(emg_folder_path + folder_name + "\\motion\\" + all_table['EMG'][i]),
                                           encoding='UTF-8')
                    processing_data, bandpass_filtered_data = emg.EMG_processing(emg_data, smoothing="lowpass")
                    # --------------------計算 iMVC---------------------------
                    emg_iMVC = pd.DataFrame(np.zeros(np.shape(processing_data)),
                                            columns=processing_data.columns)
                    emg_iMVC.iloc[:, 0] = processing_data.iloc[:, 0].values
                    emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[:, 1:].values),
                                                     MVC_value.values)*100
                    # 裁切 EMG data
                    # 因為採樣頻率是motion data的十倍, 並且從trigger開始八秒之後才算
                    trun_emg_iMVC = emg_iMVC.iloc[int(motion_info["frame_rate"]*10*8):int(motion_info["frame_rate"]*10*(8+52)) ,:].reset_index(drop=True)
                    # 計算大臂, 小臂, 手掌隨時間變化的坐標系
                    # motion_info: new
                    ArmCoord, ForearmCoord, HandCoord, new_motion_info, new_motion_data = pro.UpperExtremty_coord(trun_motion, motion_info, p1_all)
                    # 刪除不需要的變數
                    del motion_info, motion_data, analog_data, np_motion_data, processing_data
                    # --------------------------------------------------------------------
                    # 計算手指的關節角度: 只計算食、中指以及大拇指.
                    tem_hand_angle_table = pro.finger_angle_cal(c3d_list[num], new_motion_data, new_motion_info)
                    hand_angle_table = pd.concat([hand_angle_table, tem_hand_angle_table],
                                                 ignore_index=True)
                    # 計算關節的旋轉矩陣
                    ElbowRot = cal.joint_angle_rot(ArmCoord, ForearmCoord, OffsetRotP=static_ArmCoord, OffsetRotD=static_ForearmCoord)
                    WristRot = cal.joint_angle_rot(ForearmCoord, HandCoord, OffsetRotP=static_ForearmCoord, OffsetRotD=static_HandCoord)
                    ElbowEuler = cal.Rot2EulerAngle(ElbowRot, "zyx")
                    WristEuler = cal.Rot2EulerAngle(WristRot, "zxy")
                    # 使用旋轉矩陣轉換成毆拉參數後再計算角速度與角加速度
                    Elbow_AngVel, Elbow_AngAcc = cal.Rot2LocalAngularEP(ElbowRot, 180, place="joint", unit="degree")
                    Wrist_AngVel, Wrist_AngAcc = cal.Rot2LocalAngularEP(WristRot, 180, place="joint", unit="degree")
                    # --------------------計算 median frequency---------------
                    tap_emg_median_table = emg.median_frquency(emg_data.iloc[int(new_motion_info["frame_rate"]*10*8):int(new_motion_info["frame_rate"]*10*(8+52)), :],
                                                                1, results_save_path, c3d_list[num])
                    
                    # 合併資料矩陣
                    all_median_freq_table = pd.concat([all_median_freq_table, tap_emg_median_table])
                    # -----------------找角速度與角加速度的峰值,並抓取索引值---------
                    vel_motion_table, vel_emg_table = FigPlot.plot_arm_angular(c3d_list[num], Elbow_AngVel, Wrist_AngVel, trun_emg_iMVC,
                                                              joint_threshold, cal_method='vel')
                    acc_motion_table, acc_emg_table = FigPlot.plot_arm_angular(c3d_list[num], Elbow_AngAcc, Wrist_AngAcc, trun_emg_iMVC,
                                                              joint_threshold, cal_method='acc')
                    # 合併統計資料 table
                    # motion
                    all_motion_data_table = pd.concat([all_motion_data_table, vel_motion_table])
                    all_motion_data_table = pd.concat([all_motion_data_table, acc_motion_table])
                    # emg
                    all_emg_data_table = pd.concat([all_emg_data_table, vel_emg_table])
                    all_emg_data_table = pd.concat([all_emg_data_table, acc_emg_table])
                    gc.collect()
    # 將資料寫進excel
    motion_file_name = r"E:\BenQ_Project\U3\09_Results\\" + folder_name + "_Spider_motion_table.xlsx"
    emg_file_name = r"E:\BenQ_Project\U3\09_Results\\" + folder_name + "_Spider_emg_table.xlsx"
    MedFreq_file_name = r"E:\BenQ_Project\U3\09_Results\\" + folder_name + "_Spider_MedFreq_table.xlsx"
    hand_angle_table_name =  r"E:\BenQ_Project\U3\09_Results\\" + folder_name + "_Spider_FingerAngle_table.xlsx"
    # 將資料匯出至excel
    # pd.DataFrame(all_motion_data_table).to_excel(motion_file_name, sheet_name='Sheet1', index=False, header=True)
    # pd.DataFrame(all_emg_data_table).to_excel(emg_file_name, sheet_name='Sheet1', index=False, header=True)
    # pd.DataFrame(all_median_freq_table).to_excel(MedFreq_file_name, sheet_name='Sheet1', index=False, header=True)
    pd.DataFrame(hand_angle_table).to_excel(hand_angle_table_name, sheet_name='Sheet1', index=False, header=True)
