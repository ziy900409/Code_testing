# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 08:45:10 2024

@author: Hsin.YH.Yang
"""
import os
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
from scipy import signal

from detecta import detect_onset
import gc
import time
import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy import signal
# import math
from datetime import datetime
# 获取当前日期和时间
now = datetime.now()
# 将日期转换为指定格式
formatted_date = now.strftime("%Y-%m-%d")
# %% 路徑設置
# folder_path = r"E:\Hsin\BenQ\ZOWIE non-sym\\"
folder_path = r"D:\BenQ_Project\01_UR_lab\2024_07 non-symmetry\\"
motion_folder = "1.motion\\"
emg_folder = "2.EMG\\"
subfolder = "2.LargeFlick\\"
motion_type = ["Cortex\\", "Vicon\\"]

# cortex_folder = ["S11", "S12", "S13",
#                  "S14", "S15",
#                  "S16", "S17", "S18", 
#                  "S19",
#                  "S20",
#                  "S21"]

vicon_folder = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09",
                # "S10", "S11", "S12",
                ]

RawData_folder = ""
processingData_folder = "4.process_data\\"
save_place = "2.LargeFlick\\"

motion_folder_path = folder_path + motion_folder
emg_folder_path = folder_path + emg_folder

# results_save_path = r"E:\Hsin\BenQ\ZOWIE non-sym\4.process_data\\"

# stage_file_path = r"E:\Hsin\BenQ\ZOWIE non-sym\ZowieNonSymmetry_StagingFile_20240930.xlsx"
stage_file_path = r"D:\BenQ_Project\01_UR_lab\2024_07 non-symmetry\ZowieNonSymmetry_StagingFile_20240930.xlsx"
all_mouse_name = ['_EC2_', '_ECN1_', '_ECN2_', '_ECO_', '_HS_']
muscle_name = ['Extensor Carpi Radialis', 'Flexor Carpi Radialis', 'Triceps Brachii',
               'Extensor Carpi Ulnaris', '1st Dorsal Interosseous', 
               'Abductor Digiti Quinti', 'Extensor Indicis', 'Biceps Brachii']
# 替換相異欄位名稱
# c3d_recolumns_cortex = {'EC2 Wight:R.M.Finger1_x': 'R.M.Finger1_x',# 中指掌指關節
#                       'EC2 Wight:R.M.Finger1_y': 'R.M.Finger1_y',
#                       'EC2 Wight:R.M.Finger1_z': 'R.M.Finger1_z',
                      
#                       }



# new_data = {key.replace('EC2 Wight:', ''): value for key, value in data.items()}
# new_data = {key.replace('Mouse:', ''): value for key, value in data.items()}

# 定義大拇指角度所需 markerset: 拇指第一指關節、拇指掌指關節、手腕橈側莖狀突
thumb_marker = ['R.Wrist.Rad_x', 'R.Wrist.Rad_y','R.Wrist.Rad_z', # 手腕橈側莖狀突
                'R.Wrist.Rad_x', 'R.Wrist.Rad_y', 'R.Wrist.Rad_z', # 手腕橈側莖狀突
                'R.Thumb1_x', 'R.Thumb1_y', 'R.Thumb1_z', # 拇指掌指關節
                'R.Thumb2_x', 'R.Thumb2_y', 'R.Thumb2_z' # 拇指第一指關節
                ]
ring_marker = ['R.Wrist.Uln_x', 'R.Wrist.Uln_y', 'R.Wrist.Uln_z', # 手腕尺側莖狀突
               'R.Wrist.Rad_x', 'R.Wrist.Rad_y', 'R.Wrist.Rad_z', # 手腕橈側莖狀突
               'R.R.Finger1_x', 'R.R.Finger1_y', 'R.R.Finger1_z', #小指掌指關節
               'R.R.Finger2_x', 'R.R.Finger2_y', 'R.R.Finger2_z' # 小指第一指關節
               ]

little_marker = ['R.Wrist.Uln_x', 'R.Wrist.Uln_y', 'R.Wrist.Uln_z', # 手腕尺側莖狀突
                 'R.Wrist.Rad_x', 'R.Wrist.Rad_y', 'R.Wrist.Rad_z', # 手腕橈側莖狀突
                 'R.P.Finger1_x', 'R.P.Finger1_y', 'R.P.Finger1_z', #小指掌指關節
                 'R.P.Finger2_x', 'R.P.Finger2_y', 'R.P.Finger2_z' # 小指第一指關節
                 ]
wrist_marker = ['R.M.Finger1_x', 'R.M.Finger1_y', 'R.M.Finger1_z', # 中指掌指關節
                'R.Wrist.Uln_x', 'R.Wrist.Uln_y', 'R.Wrist.Uln_z', # 手腕尺側莖狀突
                'R.Wrist.Rad_x', 'R.Wrist.Rad_y', 'R.Wrist.Rad_z', # 手腕橈側莖狀突
                'R.Elbow.Lat_x', 'R.Elbow.Lat_y','R.Elbow.Lat_z' # 手肘外上髁
                ]
# 定義滑鼠marker為 M2, M3, M4 的平均值
mouse_marker = ['M2_x', 'M2_y','M2_z', #  mouse2
                'M3_x', 'M3_y', 'M3_z', # mouse3
                'M4_x', 'M4_y', 'M4_z' # mouse4
                ]



# parameter setting
smoothing = 'lowpass'
end_name = "_ed"
c = 0.802
lowpass_cutoff = 10/c
duration = 1
# %%
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
# all_hand_include_angle = pd.DataFrame({}, columns = ['mouse'])
# all_slope_data = pd.DataFrame({}, columns = ['subject', 'data_name', 'mouse'] + muscle_name)
# all_imvc_data = pd.DataFrame({}, columns = ['subject', 'data_name', 'mouse', 'method'] + muscle_name)
# # 創建資料貯存空間
# all_motion_angle_table = pd.DataFrame({}, columns = ['mouse'])
# hand_angle_table = pd.DataFrame({}, columns=["filename","CMP1",
#                                              "CMP2", "PIP2",
#                                              "CMP3",
#                                              "CMP4", "CMP5"])
# # 在不同的受試者資料夾下執行
# for folder_name in cortex_folder:
#     # 讀資料夾下的 c3d file
#     c3d_list = func.Read_File(folder_path + motion_folder + motion_type[0] + folder_name, ".c3d")
#     # 1.1.1. 讀 all 分期檔
#     excel_sheetr_name = folder_name.replace("Cortex\\", "").replace("Vicon\\", "")
#     all_table = pd.read_excel(stage_file_path,
#                               sheet_name=excel_sheetr_name)
    
#     # 第一次loop先計算tpose的問題
#     for num in range(len(c3d_list)):
#         for i in range(len(all_table['Motion_File_C3D'])):
#             if c3d_list[num].split('\\')[-1].replace(".c3d", "") == all_table['Motion_File_C3D'][i]:
#                 # 1.2.0. ---------使用tpose計算 手肘內上髁 Virtual marker 之位置 --------------
#                 if "tpose" in c3d_list[num].lower() and all_table["Task"][i] == "Tpose_elbow":
#                     print(c3d_list[num])
#                     motion_info, motion_data, analog_info, FP_data, np_motion_data = func.read_c3d(c3d_list[num])
#                     # 1. 設定輸入計算 Virtual marker 參數 : 手肘內上髁, 外上髁, UA1, UA3
#                     R_Elbow_Med = motion_data.loc[:, "R.Elbow.Med_x":"R.Elbow.Med_z"].dropna(axis=0)
#                     R_Elbow_Lat = motion_data.loc[:, "R.Elbow.Lat_x":"R.Elbow.Lat_z"].dropna(axis=0)
#                     UA1 = motion_data.loc[:, "UA1_x":"UA1_z"].dropna(axis=0)
#                     UA3 = motion_data.loc[:, "UA3_x":"UA3_z"].dropna(axis=0)
#                     # 2. 避免數量中出現NAN，請造成不同變數見長短不一致，因此找出最短數列的 index
#                     ind_frame = min([np.shape(R_Elbow_Med)[0], np.shape(R_Elbow_Lat)[0], np.shape(UA1)[0], np.shape(UA3)[0]])
#                     for ii in [R_Elbow_Med, R_Elbow_Lat, UA1, UA3]:
#                         if ind_frame == np.shape(ii)[0]:
#                             ind_frame = ii.index
#                             break
#                     # 3. 計算手肘內上髁在 LCS 之位置
#                     p1_all = pd.DataFrame(np.zeros([len(ind_frame), 3]))
#                     for frame in ind_frame:
#                         p1_all.iloc[frame :] = (kincal.transformation_matrix(R_Elbow_Lat.iloc[frame, :].values, UA1.iloc[frame, :].values, UA3.iloc[frame, :].values,
#                                                                             R_Elbow_Med.iloc[frame, :].values, np.array([0, 0, 0]),
#                                                                         rotation='GCStoLCS'))
#                     # 4. 清除不需要的變數
#                     del motion_info, motion_data, analog_info, FP_data, R_Elbow_Med, UA1, UA3, np_motion_data
#                     gc.collect()
#                 if "tpose" in c3d_list[num].lower() and all_table["Task"][i] == "Tpose_hand":
#                     print(c3d_list[num])
#                     index = 1
#                     static_ArmCoord, static_ForearmCoord, static_HandCoord = kincal.arm_natural_pos(c3d_list[num], p1_all, index)
    
#     # 讀取all MVC data
#     MVC_value = pd.read_excel(processing_folder_path + '\\' + folder_name + '\\2.emg\\' + folder_name + '_all_MVC.xlsx')
#     MVC_value = MVC_value.iloc[-1, 2:]
#     # 讀取分期檔
#     # stage_file = pd.read_excel(stage_file_path, sheet_name=folder_name)
#     # 第二次loop計算Gridshot的問題
#     imvc_data = pd.DataFrame({}, columns = ['subject', 'data_name', 'mouse', 'method'] + muscle_name)
#     gridshot_c3d_list = [file for file in c3d_list if 'LargeFlick' in file]
#     for num in range(len(gridshot_c3d_list)):
#         for i in range(len(all_table['Motion_File_C3D'])):
#             if gridshot_c3d_list[num].split('\\')[-1].replace(".c3d", "") == all_table['Motion_File_C3D'][i] \
#                 and np.isnan(all_table['start'][i]) == False:
#                 print(gridshot_c3d_list[num])
#                 print(all_table['EMG_File'][i])

#                 # 0. 處理檔名問題------------------------------------------------------
#                 save_name, extension = os.path.splitext(gridshot_c3d_list[num].split('\\', -1)[-1])
#                 motion_svae_path = folder_path + processingData_folder + folder_name + "\\" + \
#                     "1.motion\\" + save_place

#                 print(motion_svae_path + save_name + "_MedFreq.xlsx")
#                 # 將檔名拆開
#                 # 定義受試者編號以及滑鼠名稱
#                 trial_info = save_name.split("_")
#                 trial_info = {"file_name": save_name,
#                               "subject": trial_info[0],
#                               "task": trial_info[1],
#                               "mouse": trial_info[2],
#                               "trial_num": trial_info[3]}
#                 # 1. 讀取資料與初步處理-------------------------------------------------
#                 # 讀取 c3d data
#                 motion_info, motion_data, analog_info, analog_data, np_motion_data = func.read_c3d(gridshot_c3d_list[num])
#                 motion_data = motion_data.fillna(0)
#                 # 重新命名欄位名稱
#                 # motion_data.rename(columns=c3d_recolumns_cortex, inplace=True)
#                 # 計算時間，時間從trigger on後三秒開始
#                 onset_analog = detect_onset(analog_data.loc[:, 'trigger1'],
#                                             3*np.std(analog_data.loc[:50, 'trigger1']),
#                                             n_above=10, n_below=0, show=True)
#                 oneset_idx = onset_analog[0, 0]
#                 # 設定開始索引
#                 task_start = int(all_table['start'][i] + motion_info['frame_rate'])
#                 # 結束索引是開始後加15秒
#                 task_end = int(task_start + motion_info['frame_rate']*25)
                
#                 # 設定 sampling rate
#                 sampling_time = 1/motion_info['frame_rate']
#                 # 前處理 motion data, 進行低通濾波
#                 lowpass_sos = signal.butter(2, lowpass_cutoff,  btype='low', fs=motion_info['frame_rate'], output='sos')
#                 trun_motion_pd = motion_data.iloc[task_start:task_end, :]
#                 filted_motion = pd.DataFrame(np.empty(np.shape(trun_motion_pd)),
#                                               columns = motion_data.columns)
#                 filted_motion.iloc[:, 0] = motion_data.iloc[:, 0]
#                 for iii in range(np.shape(motion_data)[1]-1):
#                     filted_motion.iloc[:, iii+1] = signal.sosfiltfilt(lowpass_sos,
#                                                                       trun_motion_pd.iloc[:, iii+1].values)
                
                
#                 # motion 去除不必要的Virtual marker, 裁切, 計算坐標軸旋轉矩陣
#                 trun_motion_np = np_motion_data[:, task_start:task_end, :]
#                 # 2. 計算資料 95% confidence ellipse -----------------------------------------------------------
#                 # 計算中指掌指關節座標點與 mouse 的距離
#                 # 平均 mouse M2, M3的距離, 相加除以2，再扣除R.M.Finger1
#                 mouse_mean = ((filted_motion.loc[task_start:task_end, ['M2_x', 'M2_y','M2_z']].values + \
#                                filted_motion.loc[task_start:task_end, ['M3_x', 'M3_y', 'M3_z']].values) / 2) -\
#                                 filted_motion.loc[task_start:task_end, ['R.M.Finger1_x', 'R.M.Finger1_y', 'R.M.Finger1_z']].values
#                 # 單獨拉出 x, y 軸位置, 並且裁掉前後十秒
#                 COPxy = mouse_mean[:, :2] -\
#                         mouse_mean[0, :2]
#                 # 計算 95% confidence ellipse
#                 Area95, fig = func.conf95_ellipse(COPxy, save_name)
#                 # 儲存圖片
#                 fig.savefig(motion_svae_path + save_name + "_conf95_ellipse.jpg")
#                 #2.1. 計算手掌與桌面的夾角------------------------------------------------------------
#                 # 創建一的Y軸向量，長度
#                 R_M_Finger1 = filted_motion.loc[:, ['R.M.Finger1_x', 'R.M.Finger1_y', 'R.M.Finger1_z']].\
#                     iloc[task_start:task_end, :].values
#                 R_M_Finger2 = filted_motion.loc[:, ['R.M.Finger2_x', 'R.M.Finger2_y', 'R.M.Finger2_z']].\
#                     iloc[task_start:task_end, :].values
#                 R_R_Finger1 = filted_motion.loc[:, ['R.R.Finger1_x', 'R.R.Finger1_y', 'R.R.Finger1_z']].\
#                     iloc[task_start:task_end, :].values                
#                 R_R_Finger2 = filted_motion.loc[:, ['R.R.Finger2_x', 'R.R.Finger2_y', 'R.R.Finger2_z']].\
#                     iloc[task_start:task_end, :].values
#                 R_P_Finger1 = filted_motion.loc[:, ['R.P.Finger1_x', 'R.P.Finger1_y', 'R.P.Finger1_z']].\
#                     iloc[task_start:task_end, :].values                
#                 R_P_Finger2 = filted_motion.loc[:, ['R.P.Finger2_x', 'R.P.Finger2_y', 'R.P.Finger2_z']].\
#                     iloc[task_start:task_end, :].values
#                 # matrix = np.tile([0, 1, 0], (len(R_I_Finger1), 1))
#                 MvsR_angle = kincal.included_angle(R_M_Finger1,
#                                                    R_M_Finger2,                                      
#                                                    R_R_Finger1,
#                                                    R_R_Finger2)
#                 RvsP_angle = kincal.included_angle(R_R_Finger1,
#                                                    R_R_Finger2,                                      
#                                                    R_P_Finger1,
#                                                    R_P_Finger2) 

#                 # # 計算 mouse_mean 與 R.M.Finger1 的距離
#                 # dis_mouse_MFinger = np.sqrt((mouse_mean[:, 0] - filted_motion.loc[oneset_idx:, 'R.M.Finger1_x'])**2 + \
#                 #                             (mouse_mean[:, 0] - filted_motion.loc[oneset_idx:, 'R.M.Finger1_y'])**2 + \
#                 #                             (mouse_mean[:, 0] - filted_motion.loc[oneset_idx:, 'R.M.Finger1_z'])**2)
        
#                 # 2.1. 儲存資料-----------------------------------------------------------
#                 # 將資料儲存至矩陣
#                 hand_include_angle = pd.DataFrame({"file_name": save_name,
#                                                   "subject": trial_info["subject"],
#                                                   "task": trial_info["task"],
#                                                   "mouse": trial_info["mouse"],
#                                                   "Area95": Area95,
#                                                   "MvsR_angle_mean": np.mean(MvsR_angle),
#                                                   "MvsR_angle_min": np.min(MvsR_angle),
#                                                   "MvsR_angle_max": np.max(MvsR_angle),
#                                                   "MvsR_angle_std": np.std(MvsR_angle),
#                                                   "RvsP_angle_mean": np.mean(RvsP_angle),
#                                                   "RvsP_angle_min": np.min(RvsP_angle),
#                                                   "RvsP_angle_max": np.max(RvsP_angle),
#                                                   "RvsP_angle_std": np.std(RvsP_angle),
#                                                   },
#                                                   index=[0])
#                 all_hand_include_angle = pd.concat([hand_include_angle, all_hand_include_angle],
#                                                    ignore_index=True)
        
#                 # 2.2. 計算上肢的尤拉角-------------------------------------------------------
#                 # 計算大臂, 小臂, 手掌隨時間變化的坐標系
#                 # motion_info: new
#                 ArmCoord, ForearmCoord, HandCoord, new_motion_info, new_motion_data = kincal.UpperExtremty_coord(trun_motion_np, motion_info, p1_all)
#                 # 刪除不需要的變數
#                 del motion_data, analog_data, np_motion_data
#                 # --------------------------------------------------------------------
#                 # 計算手指的關節角度: 只計算食、中指以及大拇指.
#                 tem_hand_angle_table = kincal.finger_angle_cal(c3d_list[num], new_motion_data, new_motion_info)
#                 hand_angle_table = pd.concat([hand_angle_table, tem_hand_angle_table],
#                                              ignore_index=True)
#                 # 計算關節的旋轉矩陣
#                 ElbowRot = kincal.joint_angle_rot(ArmCoord, ForearmCoord, OffsetRotP=static_ArmCoord, OffsetRotD=static_ForearmCoord)
#                 WristRot = kincal.joint_angle_rot(ForearmCoord, HandCoord, OffsetRotP=static_ForearmCoord, OffsetRotD=static_HandCoord)
#                 ElbowEuler = kincal.Rot2EulerAngle(ElbowRot, "zyx")
#                 WristEuler = kincal.Rot2EulerAngle(WristRot, "zxy")
#                 # 使用旋轉矩陣轉換成毆拉參數後再計算角速度與角加速度
#                 Elbow_AngVel, Elbow_AngAcc = kincal.Rot2LocalAngularEP(ElbowRot, 180, place="joint", unit="degree")
#                 Wrist_AngVel, Wrist_AngAcc = kincal.Rot2LocalAngularEP(WristRot, 180, place="joint", unit="degree")
                
#                 tep_motion_angle_table = pd.DataFrame({'檔名': save_name,
#                                                        'mouse': trial_info["mouse"],
#                                                        # mean
#                                                        'Elbow:Add-Abd平均': np.mean(ElbowEuler[:, 2]), #
#                                                        'Elbow:Pro-Sup平均': np.mean(ElbowEuler[:, 1]),
#                                                        'Elbow:Flex-Ext平均': np.mean(ElbowEuler[:, 0]),
#                                                        'Hand:Add-Abd平均': np.mean(WristEuler[:, 1]), #
#                                                        'Hand:Pro-Sup平均': np.mean(WristEuler[:, 2]),
#                                                        'Hand:Flex-Ext平均': np.mean(WristEuler[:, 0]),
#                                                        # max                                                
#                                                        'Elbow:Add-Abd最大值': np.max(ElbowEuler[:, 2]),
#                                                        'Elbow:Pro-Sup最大值': np.max(ElbowEuler[:, 1]),
#                                                        'Elbow:Flex-Ext最大值': np.max(ElbowEuler[:, 0]),
#                                                        'Hand:Add-Abd最大值': np.max(WristEuler[:, 1]),
#                                                        'Hand:Pro-Sup最大值': np.max(WristEuler[:, 2]),
#                                                        'Hand:Flex-Ext最大值': np.max(WristEuler[:, 0]),
                                               
                                                       
                                                       
#                                                        },
#                                                        index=[0])
#                 #合併統計資料 table
#                 # motion
#                 all_motion_angle_table = pd.concat([all_motion_angle_table, tep_motion_angle_table],
#                                                    ignore_index=True)
#                 gc.collect()
  
            
#                 # 3. EMG 疲勞分析 -----------------------------------------------------------
#                 # 定義及設定存檔路徑
#                 emg_path = emg_folder_path + folder_name + "\\" + all_table['EMG_File'][i]
#                 emg_save_path = folder_path + processingData_folder + folder_name + "\\" + \
#                     "2.emg\\1.motion\\" + save_place
#                 # 前處理EMG data
#                 processing_data, bandpass_filtered_data = emg.EMG_processing(emg_path, smoothing=smoothing)
#                 # 計算採樣頻率
#                 emg_fs = 1 / (processing_data.iloc[1, 0] - processing_data.iloc[0, 0])
#                 # EMG start index
#                 emg_start = int((task_start - oneset_idx/10)*emg_fs/motion_info['frame_rate'])
#                 # EMG end
#                 emg_end = int((task_end - oneset_idx/10)*emg_fs/motion_info['frame_rate'])
#                 # truncut data
#                 bandpass_filtered_data = bandpass_filtered_data.iloc[emg_start:emg_end, :]
#                 processing_data = processing_data.iloc[emg_start:emg_end, :]
#                 # 畫 bandpass 後之資料圖
#                 emg.plot_plot(bandpass_filtered_data, str(emg_save_path),
#                               save_name, "_Bandpass")
#                 # 畫smoothing 後之資料圖
#                 emg.plot_plot(processing_data, str(emg_save_path),
#                               save_name, str("_" + smoothing ))
#                 # 畫 FFT analysis 的圖
#                 emg.Fourier_plot(emg_path,
#                                 (emg_save_path),
#                                 save_name)
#                 emg.Fourier_plot(emg_path,
#                                 (emg_save_path),
#                                 save_name,
#                                 notch=True)
#                 # writting data in worksheet
#                 file_name = emg_save_path + save_name + end_name + ".xlsx"
#                 pd.DataFrame(processing_data).to_excel(emg_save_path + save_name + "_lowpass.xlsx",
#                                                         sheet_name='Sheet1', index=False, header=True)
#                 # 計算 iMVC
#                 emg_iMVC = pd.DataFrame(np.zeros(np.shape(processing_data)),
#                                         columns=processing_data.columns)
#                 emg_iMVC.iloc[:, 0] = processing_data.iloc[:, 0].values
#                 emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[:, 1:].values),
#                                                   MVC_value.values)*100
                    
#                 pd.DataFrame(emg_iMVC).to_excel(emg_save_path + save_name + "_iMVC.xlsx",
#                                                 sheet_name='Sheet1', index=False, header=True)
#                 # 進行中頻率分析
#                 med_freq_data, slope_data = emg.median_frquency(emg_path,
#                                                                 duration, emg_save_path, save_name,
#                                                                 begin=[emg_start, emg_end])
#                 # 儲存斜率的資料，並合併成一個資料表
#                 slope_data['mouse'] = [trial_info['mouse']]
#                 slope_data['data_name'] = [save_name]
#                 slope_data['subject'] = [trial_info["subject"]]
#                 all_slope_data = pd.concat([all_slope_data, slope_data])
#                 # 儲存 iMVC data
#                 for muscle in muscle_name:
#                     imvc_data.loc[0, 'method'] = 'mean'
#                     imvc_data.loc[0, 'subject'] = trial_info["subject"]
#                     imvc_data.loc[0, 'data_name'] = save_name
#                     imvc_data.loc[0, 'mouse'] = trial_info['mouse']
#                     imvc_data.loc[0, muscle] = np.mean(emg_iMVC.loc[:, muscle])
#                     imvc_data.loc[1, 'method'] = 'max'
#                     imvc_data.loc[1, 'data_name'] = save_name
#                     imvc_data.loc[1, 'mouse'] = trial_info['mouse']
#                     imvc_data.loc[1, 'subject'] = trial_info["subject"]
#                     imvc_data.loc[1, muscle] = np.max(emg_iMVC.loc[:, muscle])
#                     imvc_data.loc[2, 'method'] = 'min'
#                     imvc_data.loc[2, 'data_name'] = save_name
#                     imvc_data.loc[2, 'mouse'] = trial_info['mouse']
#                     imvc_data.loc[2, 'subject'] = trial_info["subject"]
#                     imvc_data.loc[2, muscle] = np.min(emg_iMVC.loc[:, muscle])
                
#                 # imvc_data['mouse'].replace(np.NaN, trial_info['mouse'], inplace=True)
#                 # imvc_data['data_name'].replace(np.NaN, save_name, inplace=True)
#                 # imvc_data['subject'].replace(np.NaN, trial_info["subject"], inplace=True)
#                 all_imvc_data = pd.concat([all_imvc_data, imvc_data])
#      # 將尤拉角資料寫進excel
#      # 使用 ExcelWriter 寫入 Excel 文件
# with pd.ExcelWriter(folder_path + "6.Statistic\\" + save_place + "All_LargeFlick_motion_table" + formatted_date + ".xlsx",
#                     engine='openpyxl') as writer:
#     all_motion_angle_table.to_excel(writer, sheet_name='arm_motion', index=False, header=True)
#     all_hand_include_angle.to_excel(writer, sheet_name='FingerIncludeAngle', index=False, header=True)
#     hand_angle_table.to_excel(writer, sheet_name='FingerAngle', index=False, header=True)
#     all_slope_data.to_excel(writer, sheet_name='MedFreq', index=False, header=True)
#     all_imvc_data.to_excel(writer, sheet_name='iMVC_cal', index=False, header=True)
    
# print("輸出檔案： ", folder_path + "6.Statistic\\" + save_place + "All_LargeFlick_motion_table" + formatted_date + ".xlsx")
    
        
                    


# %% Vicon version
# %% EMG preprocessing 
# for Vicon format

tic = time.process_time()
for folder_name in vicon_folder:
    tic = time.process_time()
    MVC_folder_path = folder_path + motion_folder + motion_type[1] + folder_name
    csv_list = func.Read_File(MVC_folder_path, ".c3d")
    MVC_file_list = [file for file in csv_list if 'MVC' in file]
    
    # c3d_file_path = gen.Read_File(MVC_folder_path, ".c3d")
    # MVC_list = csv_MVC_list + c3d_file_path

    fig_save_path = folder_path + processingData_folder + folder_name + "\\2.emg\\2.MVC\\"
    print("Now processing MVC data in " + folder_name)
    for MVC_path in MVC_file_list:
        print(MVC_path)
        # data = pd.read_csv(MVC_path, encoding='UTF-8')
        data_save_path = folder_path + processingData_folder + folder_name + "\\2.emg\\2.MVC\\" 
        # 將檔名拆開 deal with filename and add extension with _ed
        filepath, tempfilename = os.path.split(MVC_path)
        filename, extension = os.path.splitext(tempfilename)
        # 畫圖
        # 前處理EMG data
        processing_data, bandpass_filtered_data = emg.EMG_processing(MVC_path, smoothing="lowpass")

        # 畫 bandpass 後之資料圖
        emg.plot_plot(bandpass_filtered_data, str(fig_save_path),
                     filename, "_Bandpass")
        # 畫smoothing 後之資料圖
        emg.plot_plot(processing_data, str(fig_save_path),
                     filename, str("_" + smoothing))
        # 畫 FFT analysis 的圖
        emg.Fourier_plot(MVC_path,
                        (fig_save_path),
                        filename)
        emg.Fourier_plot(MVC_path,
                        (fig_save_path),
                        filename,
                        notch=True)

        # writting data in worksheet
        file_name = fig_save_path + filename + end_name + ".xlsx"
        pd.DataFrame(processing_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    toc = time.process_time()
    print("Total Time:",toc-tic)  
# 找最大值
for folder_name in vicon_folder:
    print("To find the maximum value of all of MVC data in: " + folder_name)
    tic = time.process_time()
    emg.Find_MVC_max(folder_path + processingData_folder + folder_name + "\\2.emg\\2.MVC\\",
                     folder_path + processingData_folder + folder_name + "\\2.emg\\")
    toc = time.process_time()
    print("Total Time:",toc-tic)
toc = time.process_time()
print("MVC Data Total Time Spent: ",toc-tic)
gc.collect()

# %% motion processing for vicon format
all_hand_include_angle = pd.DataFrame({}, columns = ['mouse'])
all_slope_data = pd.DataFrame({}, columns = ['subject', 'data_name', 'mouse'] + muscle_name)
all_imvc_data = pd.DataFrame({}, columns = ['subject', 'data_name', 'mouse', 'method'] + muscle_name)
# 創建資料貯存空間
all_motion_angle_table = pd.DataFrame({}, columns = ['mouse'])
hand_angle_table = pd.DataFrame({}, columns=["filename","CMP1",
                                             "CMP2", "PIP2",
                                             "CMP3",
                                             "CMP4", "CMP5"])
# 在不同的受試者資料夾下執行
for folder_name in vicon_folder:
    # 讀資料夾下的 c3d file
    c3d_list = func.Read_File(folder_path + motion_folder + motion_type[1] + folder_name, ".c3d")
    # 1.1.1. 讀 all 分期檔
    excel_sheetr_name = folder_name.replace("Cortex\\", "").replace("Vicon\\", "")
    all_table = pd.read_excel(stage_file_path,
                              sheet_name=excel_sheetr_name)
    # 第一次loop先計算tpose的問題
    for num in range(len(c3d_list)):
        # 1.2.0. ---------使用tpose計算 手肘內上髁 Virtual marker 之位置 --------------
        if "tpose_elbow" in c3d_list[num].lower():
            print(c3d_list[num])
            motion_info, motion_data, analog_info, FP_data, np_motion_data = func.read_c3d(c3d_list[num], method='vicon')
            # 0. 取代欄位名稱
            motion_data.rename(columns=lambda x: x.replace('elbow:', ''), inplace=True)
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
        if "tpose_hand" in c3d_list[num].lower():
            print(c3d_list[num])
            index = 1
            static_ArmCoord, static_ForearmCoord, static_HandCoord = kincal.arm_natural_pos(c3d_list[num], p1_all, index,
                                                                                            method='vicon',
                                                                                            replace=folder_name)
    
    # 讀取all MVC data
    MVC_value = pd.read_excel(processing_folder_path + '\\' + folder_name + '\\2.emg\\' + folder_name + '_all_MVC.xlsx')
    MVC_value = MVC_value.iloc[-1, 2:]
    # 讀取分期檔
    # stage_file = pd.read_excel(stage_file_path, sheet_name=folder_name)
    # 第二次loop計算Gridshot的問題
    imvc_data = pd.DataFrame({}, columns = ['subject', 'data_name', 'mouse', 'method'] + muscle_name)
    gridshot_c3d_list = [file for file in c3d_list if 'LargeFlick' in file]
    for num in range(len(gridshot_c3d_list)):
        print(gridshot_c3d_list[num])
        # 0. 處理檔名問題------------------------------------------------------
        save_name, extension = os.path.splitext(gridshot_c3d_list[num].split('\\', -1)[-1])
        motion_svae_path = folder_path + processingData_folder + folder_name + "\\" + \
                    "1.motion\\" + save_place

        print(motion_svae_path + save_name + "_MedFreq.xlsx")
        # 將檔名拆開
        # filepath, tempfilename = os.path.split(gridshot_c3d_list[num])
        # filename, extension = os.path.splitext(tempfilename)
        # 定義受試者編號以及滑鼠名稱
        trial_info = save_name.split("_")
        
        trial_info = {"file_name": save_name,
                      "subject": trial_info[0],
                      "task": trial_info[1],
                      "mouse": trial_info[2],
                      "trial_num": trial_info[3]}
        # 1. 讀取資料與初步處理-------------------------------------------------
        # 讀取 c3d data
        motion_info, motion_data, analog_info, analog_data, np_motion_data = func.read_c3d(gridshot_c3d_list[num],
                                                                                           method='vicon')
        # 取代欄位名稱
        motion_data.rename(columns=lambda x: x.replace(str(folder_name + ':'), ''), inplace=True)
        motion_data.rename(columns=lambda x: x.replace('mouse:', ''), inplace=True)
        motion_data = motion_data.fillna(0)
        # 取代 LABELS 中的 subject name
        new_columns_list = [s.replace(str(folder_name + ':'), '') for s in motion_info["LABELS"]]
        new_columns_list = [s.replace('mouse:', '') for s in new_columns_list]
        motion_info["LABELS"] = new_columns_list
        # 重新命名欄位名稱
        # motion_data = {key.replace('EC2 Wight:', ''): value for key, value in motion_data.items()}
        # motion_data.rename(columns=c3d_recolumns_cortex, inplace=True)
        # 計算時間，時間從trigger on後三秒開始
        # onset_analog = detect_onset(analog_data.loc[:, 'trigger1'],
        #                             3*np.std(analog_data.loc[:50, 'trigger1']),
        #                             n_above=10, n_below=0, show=True)
        # oneset_idx = int(motion_info['frame_rate'])
        # 設定開始索引
        task_start = int(motion_info['frame_rate'])
        # 結束索引是開始後加15秒
        task_end = int(motion_info['frame_rate']*25)
        
        # 設定 sampling rate
        sampling_time = 1/motion_info['frame_rate']
        # 前處理 motion data, 進行低通濾波
        lowpass_sos = signal.butter(2, lowpass_cutoff,  btype='low', fs=motion_info['frame_rate'], output='sos')
        trun_motion_pd = motion_data.iloc[task_start:task_end, :]
        filted_motion = pd.DataFrame(np.empty(np.shape(trun_motion_pd)),
                                      columns = motion_data.columns)
        filted_motion.iloc[:, 0] = motion_data.iloc[:, 0]
        for iii in range(np.shape(motion_data)[1]-1):
            filted_motion.iloc[:, iii+1] = signal.sosfiltfilt(lowpass_sos,
                                                              trun_motion_pd.iloc[:, iii+1].values)
        # motion 去除不必要的Virtual marker, 裁切, 計算坐標軸旋轉矩陣
        trun_motion_np = np_motion_data[:, task_start:task_end, :]
        # 2. 計算資料 95% confidence ellipse -----------------------------------------------------------
        # 計算中指掌指關節座標點與 mouse 的距離
        # 平均 mouse M2, M3的距離, 相加除以2，再扣除R.M.Finger1
        mouse_mean = ((filted_motion.loc[task_start:task_end, ['M2_x', 'M2_y','M2_z']].values + \
                       filted_motion.loc[task_start:task_end, ['M3_x', 'M3_y', 'M3_z']].values) / 2) -\
                        filted_motion.loc[task_start:task_end, ['R.M.Finger1_x', 'R.M.Finger1_y', 'R.M.Finger1_z']].values
        # 單獨拉出 x, y 軸位置
        COPxy = mouse_mean[:, :2] - mouse_mean[0, :2]
        # 計算 95% confidence ellipse
        Area95, fig = func.conf95_ellipse(COPxy, save_name)
        # 儲存圖片
        fig.savefig(motion_svae_path + save_name + "_conf95_ellipse.jpg")                 

        # # 計算 mouse_mean 與 R.M.Finger1 的距離
        # dis_mouse_MFinger = np.sqrt((mouse_mean[:, 0] - filted_motion.loc[oneset_idx:, 'R.M.Finger1_x'])**2 + \
        #                             (mouse_mean[:, 0] - filted_motion.loc[oneset_idx:, 'R.M.Finger1_y'])**2 + \
        #                             (mouse_mean[:, 0] - filted_motion.loc[oneset_idx:, 'R.M.Finger1_z'])**2)
    
        # 2.1. 儲存資料-----------------------------------------------------------
        # 將資料儲存至矩陣
        add_data = pd.DataFrame({"file_name": save_name,
                                 "subject": trial_info["subject"],
                                 "task": trial_info["task"],
                                 "mouse": trial_info["mouse"],
                                 "Area95": Area95,
                                 },
                                index=[0])
        #2.1. 計算手掌與桌面的夾角------------------------------------------------------------
        # 創建一的Y軸向量，長度
        R_M_Finger1 = filted_motion.loc[:, ['R.M.Finger1_x', 'R.M.Finger1_y', 'R.M.Finger1_z']].\
            iloc[task_start:task_end, :].values
        R_M_Finger2 = filted_motion.loc[:, ['R.M.Finger2_x', 'R.M.Finger2_y', 'R.M.Finger2_z']].\
            iloc[task_start:task_end, :].values
        R_R_Finger1 = filted_motion.loc[:, ['R.R.Finger1_x', 'R.R.Finger1_y', 'R.R.Finger1_z']].\
            iloc[task_start:task_end, :].values                
        R_R_Finger2 = filted_motion.loc[:, ['R.R.Finger2_x', 'R.R.Finger2_y', 'R.R.Finger2_z']].\
            iloc[task_start:task_end, :].values
        R_P_Finger1 = filted_motion.loc[:, ['R.P.Finger1_x', 'R.P.Finger1_y', 'R.P.Finger1_z']].\
            iloc[task_start:task_end, :].values                
        R_P_Finger2 = filted_motion.loc[:, ['R.P.Finger2_x', 'R.P.Finger2_y', 'R.P.Finger2_z']].\
            iloc[task_start:task_end, :].values
        # matrix = np.tile([0, 1, 0], (len(R_I_Finger1), 1))
        MvsR_angle = kincal.included_angle(R_M_Finger1,
                                           R_M_Finger2,                                      
                                           R_R_Finger1,
                                           R_R_Finger2)
        RvsP_angle = kincal.included_angle(R_R_Finger1,
                                           R_R_Finger2,                                      
                                           R_P_Finger1,
                                           R_P_Finger2) 

        # # 計算 mouse_mean 與 R.M.Finger1 的距離
        # dis_mouse_MFinger = np.sqrt((mouse_mean[:, 0] - filted_motion.loc[oneset_idx:, 'R.M.Finger1_x'])**2 + \
        #                             (mouse_mean[:, 0] - filted_motion.loc[oneset_idx:, 'R.M.Finger1_y'])**2 + \
        #                             (mouse_mean[:, 0] - filted_motion.loc[oneset_idx:, 'R.M.Finger1_z'])**2)

        # 2.1. 儲存資料-----------------------------------------------------------
        # 將資料儲存至矩陣
        hand_include_angle = pd.DataFrame({"file_name": save_name,
                                          "subject": trial_info["subject"],
                                          "task": trial_info["task"],
                                          "mouse": trial_info["mouse"],
                                          "Area95": Area95,
                                          "MvsR_angle_mean": np.mean(MvsR_angle),
                                          "MvsR_angle_min": np.min(MvsR_angle),
                                          "MvsR_angle_max": np.max(MvsR_angle),
                                          "MvsR_angle_std": np.std(MvsR_angle),
                                          "RvsP_angle_mean": np.mean(RvsP_angle),
                                          "RvsP_angle_min": np.min(RvsP_angle),
                                          "RvsP_angle_max": np.max(RvsP_angle),
                                          "RvsP_angle_std": np.std(RvsP_angle),
                                          },
                                          index=[0])
        all_hand_include_angle = pd.concat([hand_include_angle, all_hand_include_angle],
                                           ignore_index=True)

        # 2.2. 計算上肢的尤拉角-------------------------------------------------------
        # 計算大臂, 小臂, 手掌隨時間變化的坐標系
        # motion_info: new
        ArmCoord, ForearmCoord, HandCoord, new_motion_info, new_motion_data = kincal.UpperExtremty_coord(trun_motion_np, motion_info, p1_all)
        # 刪除不需要的變數
        del motion_data, analog_data, np_motion_data
        # --------------------------------------------------------------------
        # 計算手指的關節角度: 只計算食、中指以及大拇指.
        tem_hand_angle_table = kincal.finger_angle_cal(c3d_list[num], new_motion_data, new_motion_info)
        hand_angle_table = pd.concat([hand_angle_table, tem_hand_angle_table],
                                     ignore_index=True)
        # 計算關節的旋轉矩陣
        ElbowRot = kincal.joint_angle_rot(ArmCoord, ForearmCoord, OffsetRotP=static_ArmCoord, OffsetRotD=static_ForearmCoord)
        WristRot = kincal.joint_angle_rot(ForearmCoord, HandCoord, OffsetRotP=static_ForearmCoord, OffsetRotD=static_HandCoord)
        ElbowEuler = kincal.Rot2EulerAngle(ElbowRot, "zyx")
        WristEuler = kincal.Rot2EulerAngle(WristRot, "zxy")
        # 使用旋轉矩陣轉換成毆拉參數後再計算角速度與角加速度
        Elbow_AngVel, Elbow_AngAcc = kincal.Rot2LocalAngularEP(ElbowRot, 180, place="joint", unit="degree")
        Wrist_AngVel, Wrist_AngAcc = kincal.Rot2LocalAngularEP(WristRot, 180, place="joint", unit="degree")
        
        tep_motion_angle_table = pd.DataFrame({'檔名': save_name,
                                               'mouse': trial_info["mouse"],
                                               # mean
                                               'Elbow:Add-Abd平均': np.mean(ElbowEuler[:, 2]), #
                                               'Elbow:Pro-Sup平均': np.mean(ElbowEuler[:, 1]),
                                               'Elbow:Flex-Ext平均': np.mean(ElbowEuler[:, 0]),
                                               'Hand:Add-Abd平均': np.mean(WristEuler[:, 1]), #
                                               'Hand:Pro-Sup平均': np.mean(WristEuler[:, 2]),
                                               'Hand:Flex-Ext平均': np.mean(WristEuler[:, 0]),
                                               # max                                                
                                               'Elbow:Add-Abd最大值': np.max(ElbowEuler[:, 2]),
                                               'Elbow:Pro-Sup最大值': np.max(ElbowEuler[:, 1]),
                                               'Elbow:Flex-Ext最大值': np.max(ElbowEuler[:, 0]),
                                               'Hand:Add-Abd最大值': np.max(WristEuler[:, 1]),
                                               'Hand:Pro-Sup最大值': np.max(WristEuler[:, 2]),
                                               'Hand:Flex-Ext最大值': np.max(WristEuler[:, 0]),
                                               },
                                               index=[0])
        #合併統計資料 table
        # motion
        all_motion_angle_table = pd.concat([all_motion_angle_table, tep_motion_angle_table],
                                           ignore_index=True)
        gc.collect()


        # 3. 疲勞分析 -----------------------------------------------------------
        # 定義及設定存檔路徑
        emg_path = gridshot_c3d_list[num]
        emg_save_path = folder_path + processingData_folder + folder_name + "\\" + \
                "2.emg\\1.motion\\" + save_place
        # 前處理EMG data
        processing_data, bandpass_filtered_data = emg.EMG_processing(emg_path, smoothing=smoothing)
        # 畫 bandpass 後之資料圖
        emg.plot_plot(bandpass_filtered_data, str(emg_save_path),
                      save_name, "_Bandpass")
        # 畫smoothing 後之資料圖
        emg.plot_plot(processing_data, str(emg_save_path),
                      save_name, str(smoothing + "_"))
        # writting data in worksheet
        file_name = emg_save_path + save_name + end_name + ".xlsx"
        pd.DataFrame(processing_data).to_excel(emg_save_path + save_name + "_lowpass.xlsx",
                                               sheet_name='Sheet1', index=False, header=True)
        # 計算 iMVC
        emg_iMVC = pd.DataFrame(np.zeros(np.shape(processing_data)),
                                columns=processing_data.columns)
        emg_iMVC.iloc[:, 0] = processing_data.iloc[:, 0].values
        emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[:, 1:].values),
                                         MVC_value.values)*100
            
        pd.DataFrame(emg_iMVC).to_excel(emg_save_path + save_name + "_iMVC.xlsx",
                                        sheet_name='Sheet1', index=False, header=True)
        # 進行中頻率分析
        med_freq_data, slope_data = emg.median_frquency(emg_path,
                                                        duration, emg_save_path, save_name,
                                                        begin=[0, motion_info['frame_rate']*15*10])
        # 儲存斜率的資料，並合併成一個資料表
        slope_data['mouse'] = [trial_info['mouse']]
        slope_data['data_name'] = [save_name]
        slope_data['subject'] = [trial_info["subject"]]
        all_slope_data = pd.concat([all_slope_data, slope_data])
        # 儲存 iMVC data
        for muscle in emg_iMVC.columns:
            imvc_data.loc[0, 'method'] = 'mean'
            imvc_data.loc[0, 'subject'] = trial_info["subject"]
            imvc_data.loc[0, muscle] = np.mean(emg_iMVC.loc[:, muscle])
            imvc_data.loc[1, 'method'] = 'max'
            imvc_data.loc[1, 'subject'] = trial_info["subject"]
            imvc_data.loc[1, muscle] = np.max(emg_iMVC.loc[:, muscle])
            imvc_data.loc[2, 'method'] = 'min'
            imvc_data.loc[2, 'subject'] = trial_info["subject"]
            imvc_data.loc[2, muscle] = np.min(emg_iMVC.loc[:, muscle])
        
        # imvc_data['mouse'].replace(np.NaN, trial_info['mouse'], inplace=True)
        # imvc_data['data_name'].replace(np.NaN, save_name, inplace=True)
        # imvc_data['subject'].replace(np.NaN, trial_info["subject"], inplace=True)
        all_imvc_data = pd.concat([all_imvc_data, imvc_data])
            
      
# 儲存檔案  
with pd.ExcelWriter(folder_path + "6.Statistic\\" + save_place + "All_LargeFlick_vicon_table" + formatted_date + ".xlsx",
                    engine='openpyxl') as writer:
    all_motion_angle_table.to_excel(writer, sheet_name='arm_motion', index=False, header=True)
    all_hand_include_angle.to_excel(writer, sheet_name='FingerIncludeAngle', index=False, header=True)
    hand_angle_table.to_excel(writer, sheet_name='FingerAngle', index=False, header=True)
    all_slope_data.to_excel(writer, sheet_name='MedFreq', index=False, header=True)
    all_imvc_data.to_excel(writer, sheet_name='iMVC_cal', index=False, header=True)
        





















