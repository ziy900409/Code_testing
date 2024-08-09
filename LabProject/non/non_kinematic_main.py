# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:26:06 2024

@author: Hsin.YH.Yang
"""

# %% import package
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"C:\Users\Public\BenQ\myPyCode\U3")
import U3_Kinematic_function as func
import U3_Kinematic_calculate as cal
import U3_Kinematic_PlotFigure as FigPlot
import U3_Kinematic_processing as pro

import os
import numpy as np
import pandas as pd

from detecta import detect_onset
import gc
import matplotlib.pyplot as plt

# %% 路徑設置
motion_path = r"E:\Motion Analysis\U3 Research\\"
Fitts_path = r"E:\BenQ_Project\U3\08_Fitts\\"


RawData_folder = ""
processingData_folder = ""

motion_folder_path = motion_path + RawData_folder + "\\" 
emg_folder_path = r"E:\BenQ_Project\U3\07_EMGrecording\raw_data\\"

results_save_path = r"E:\BenQ_Project\U3\09_Results\\"
# 取得所有 motion data folder list
# 去除有“.“開頭的檔案 and 只獲得資料夾路徑，排除其他可能的檔案格式
motion_folder_list  = [f for f in os.listdir(motion_folder_path) if not f.startswith('.') \
                       and os.path.isdir(os.path.join(motion_folder_path, f))]
# 取得所有 processing data folder list
processing_folder_path = motion_path + "\\" + processingData_folder + "\\"
processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.') \
                          and os.path.isdir(os.path.join(processing_folder_path, f))]

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

