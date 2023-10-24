# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:56:34 2023

@author: Hsin.YH.Yang
"""

# %%
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"C:\Users\Public\BenQ\myPyCode\U3")
import U3_Kinematic_function as func
import U3_Kinematic_calculate as cal

import numpy as np
import pandas as pd
import gc
from scipy import signal


# %% 計算內上髁在LCS的位置
def V_Elbow_cal(c3d_path):
    motion_info, motion_data, analog_info, FP_data, np_motion_data = func.read_c3d(c3d_path)
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
    return p1_all
# %% 計算橈側內髁之位置
def V_Elbow_cal_1(c3d_path):
    motion_info, motion_data, analog_info, FP_data, np_motion_data = func.read_c3d(c3d_path)
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
    return p1_all

# %% 使用tpose計算手部的自然關節角度
def arm_natural_pos(c3d_path, p1_all, index):
    motion_info, motion_data, analog_info, FP_data, np_motion_data = func.read_c3d(c3d_path)
    V_R_Elbow_Med = np.zeros(shape=(3))
    # 計算虛擬手肘內上髁位置
    V_R_Elbow_Med[:] = func.transformation_matrix(motion_data.loc[index, "EC2 Wight:R.Elbow.Lat_x":"EC2 Wight:R.Elbow.Lat_z"].values, # EC2 Wight:R.Elbow.Lat
                                                  motion_data.loc[index, "EC2 Wight:UA1_x":"EC2 Wight:UA1_z"].values, # EC2 Wight:UA1
                                                  motion_data.loc[index, "EC2 Wight:UA3_x":"EC2 Wight:UA3_z"].values, # EC2 Wight:UA3
                                                  p1_all.iloc[5, :].values, np.array([0, 0, 0]),
                                                  rotation='LCStoGCS')
    # 定義手部支段坐標系
    static_ArmCoord = np.empty(shape=(3, 3))
    static_ForearmCoord = np.empty(shape=(3, 3))
    static_HandCoord = np.empty(shape=(3, 3))
    # 定義人體自然角度坐標系, tpose 手部自然放置角度
    static_ArmCoord[:, :] = cal.DefCoordArm(motion_data.loc[index, "EC2 Wight:R.Shoulder_x":"EC2 Wight:R.Shoulder_z"],
                                            motion_data.loc[index, "EC2 Wight:R.Elbow.Lat_x":"EC2 Wight:R.Elbow.Lat_z"],
                                            V_R_Elbow_Med[:])
    static_ForearmCoord[:, :] = cal.DefCoordForearm(motion_data.loc[index, "EC2 Wight:R.Elbow.Lat_x":"EC2 Wight:R.Elbow.Lat_z"],
                                                    V_R_Elbow_Med[:],
                                                    motion_data.loc[index, "EC2 Wight:R.Wrist.Uln_x":"EC2 Wight:R.Wrist.Uln_z"],
                                                    motion_data.loc[index, "EC2 Wight:R.Wrist.Rad_x":"EC2 Wight:R.Wrist.Rad_z"])
    static_HandCoord[:, :] = cal.DefCoordHand(motion_data.loc[index, "EC2 Wight:R.Wrist.Uln_x":"EC2 Wight:R.Wrist.Uln_z"],
                                              motion_data.loc[index, "EC2 Wight:R.Wrist.Rad_x":"EC2 Wight:R.Wrist.Rad_z"],
                                              motion_data.loc[index, "EC2 Wight:R.M.Finger1_x":"EC2 Wight:R.M.Finger1_z"])
    # 清除不需要的變數
    del motion_info, motion_data, analog_info, FP_data, np_motion_data, index
    gc.collect()
    return static_ArmCoord, static_ForearmCoord, static_HandCoord
# %% 計算大臂, 小臂, 手掌隨時間變化的坐標系

def UpperExtremty_coord(trun_motion, motion_info, p1_all):
    # 1.2.5. ---------計算手肘內上髁之位置----------------------------------
    # 建立手肘內上髁的資料貯存位置
    V_R_Elbow_Med = np.zeros(shape=(1, np.shape(trun_motion)[1], np.shape(trun_motion)[2]))
    # 找出以下三個字串的索引值
    target_strings = ["EC2 Wight:R.Elbow.Lat", "EC2 Wight:UA1"
                      , "EC2 Wight:UA3", "EC2 Wight:R.Shoulder",
                      "EC2 Wight:R.Wrist.Uln", "EC2 Wight:R.Wrist.Rad",
                      "EC2 Wight:R.M.Finger1"]
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
    new_np_motion_data = new_trun_motion[np_labels, :, :]
    # # 低通濾波 butterworth filter
    bandpass_filtered = np.empty(shape=np.shape(new_np_motion_data))
    bandpass_sos = signal.butter(2, 6/0.802,  btype='lowpass', fs=motion_info["frame_rate"], output='sos')
    for iii in range(np.shape(new_np_motion_data)[0]):
        for iiii in range(np.shape(new_np_motion_data)[2]):
            bandpass_filtered[iii, :, iiii] = signal.sosfiltfilt(bandpass_sos,
                                                                 new_np_motion_data[iii, :, iiii])
    
    # 開始計算運動學資料
    ArmCoord = np.empty(shape=(3, 3, np.shape(bandpass_filtered)[1]))
    ForearmCoord = np.empty(shape=(3, 3, np.shape(bandpass_filtered)[1]))
    HandCoord = np.empty(shape=(3, 3, np.shape(bandpass_filtered)[1]))
    # 對每個 Frame 定義坐標系
    for i in range(np.shape(bandpass_filtered)[1]):
        ArmCoord[:, :, i] = cal.DefCoordArm(bandpass_filtered[indices[3], i, :], # R.Shoulder
                                            bandpass_filtered[indices[0], i, :], # R.Elbow.Lat
                                            bandpass_filtered[-1, i, :]) # R.Elbow.Med
        ForearmCoord[:, :, i] = cal.DefCoordForearm(bandpass_filtered[indices[0], i, :], # R.Elbow.Lat
                                                    bandpass_filtered[-1, i, :], # R.Elbow.Med
                                                    bandpass_filtered[indices[4], i, :], # R.Wrist.Uln
                                                    bandpass_filtered[indices[5], i, :]) # R.Wrist.Rad
        HandCoord[:, :, i] = cal.DefCoordHand(bandpass_filtered[indices[4], i, :], # R.Wrist.Uln
                                              bandpass_filtered[indices[5], i, :], # R.Wrist.Rad
                                              bandpass_filtered[indices[6], i, :]) # R.M.Finger1
    return ArmCoord, ForearmCoord, HandCoord, motion_info, bandpass_filtered
# %% 計算手指關節角度
def finger_angle_cal(file_name, motion_data, motion_info):
    # 建立要尋找的motion data label
    target_strings = ["R.Wrist.Rad", "R.Wrist.Uln",
                      "R.Thumb1", "R.Thumb2",
                      "R.I.Finger1", "R.I.Finger2", "R.I.Finger3",
                      "R.M.Finger1", "R.M.Finger2",
                      "R.R.Finger1", "R.R.Finger2",
                      "R.P.Finger1", "R.P.Finger2"]
    # 找出指定motion data label的索引
    indices = []
    for target_str in target_strings:
        try:
            index = motion_info["LABELS"].index(target_str)
            indices.append(index)
        except ValueError:
            indices.append(None)
    # 建立暫存的矩陣
    hand_angle_table = pd.DataFrame({}, columns=["filename", "CMP1",
                                                 "CMP2", "PIP2",
                                                 "CMP3", "PIP3",
                                                 "CMP4", "CMP5"])
            


    wrist_cen = (motion_data[indices[0], :, :] + motion_data[indices[1], :, :])/2
    CMP1 = cal.included_angle(wrist_cen,                         # wrist
                              motion_data[indices[2], :, :], # R.Thumb1
                              motion_data[indices[3], :, :]) # R.Thumb2
    CMP2 = cal.included_angle(wrist_cen,                         # wrist
                              motion_data[indices[4], :, :], # R.I.Finger1
                              motion_data[indices[5], :, :]) # R.I.Finger2
    PIP2 = cal.included_angle(motion_data[indices[4], :, :], # R.I.Finger1
                              motion_data[indices[5], :, :], # R.I.Finger2
                              motion_data[indices[6], :, :]) # R.I.Finger3
    CMP3 = cal.included_angle(wrist_cen,                         # wrist
                              motion_data[indices[7], :, :], # R.M.Finger1
                              motion_data[indices[8], :, :]) # R.M.Finger2
    CMP4 = cal.included_angle(wrist_cen,                         # wrist
                              motion_data[indices[9], :, :], # R.R.Finger1
                              motion_data[indices[10], :, :]) # R.R.Finger2
    CMP5 = cal.included_angle(wrist_cen,                          # wrist
                              motion_data[indices[11], :, :], # R.P.Finger1
                              motion_data[indices[12], :, :]) # R.P.Finger2
    hand_angle_table = pd.concat([hand_angle_table,
                                  pd.DataFrame({"filename":file_name,
                                                "CMP1":np.mean(CMP1),
                                                "CMP2":np.mean(CMP2),
                                                "PIP2":np.mean(PIP2),
                                                "CMP3":np.mean(CMP3),
                                                "CMP4":np.mean(CMP4),
                                                "CMP5":np.mean(CMP5)
                                                },index=[0])],
                                 ignore_index=True)
    return hand_angle_table














