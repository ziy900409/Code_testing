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
import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy import signal
# import math
# %% 路徑設置
# folder_path = r"E:\Hsin\BenQ\ZOWIE non-sym\\"
folder_path = r"D:\BenQ_Project\01_UR_lab\2024_07 non-symmetry\\"
motion_folder = "1.motion\\"
emg_folder = "3.EMG\\ "
subfolder = "2.LargeFlick\\"
motion_type = ["Cortex\\", "Vicon\\"]

cortex_folder = ["S11", "S12", "S13", "S14", "S15",
                 "S16", "S17", "S18", "S19", "S20",
                 "S21"]

vicon_folder = ["S03", " S04"]

RawData_folder = ""
processingData_folder = "4.process_data\\"
save_place = "4.GridShot\\"

motion_folder_path = folder_path + motion_folder
emg_folder_path = folder_path + emg_folder

# results_save_path = r"E:\Hsin\BenQ\ZOWIE non-sym\4.process_data\\"

# stage_file_path = r"E:\Hsin\BenQ\ZOWIE non-sym\ZowieNonSymmetry_StagingFile_20240929.xlsx"
stage_file_path = r"D:\BenQ_Project\01_UR_lab\2024_07 non-symmetry\ZowieNonSymmetry_StagingFile_20240929.xlsx"
all_mouse_name = ['_EC2_', '_ECN1_', '_ECN2_', '_ECO_', '_HS_']
muscle_name = ['Extensor carpi radialis', 'Flexor Carpi Radialis', 'Triceps',
               'Extensor carpi ulnaris', '1st. dorsal interosseous', 
               'Abductor digiti quinti', 'Extensor Indicis', 'Biceps']
# 替換相異欄位名稱
c3d_recolumns_cortex = {'EC2 Wight:R.M.Finger1_x': 'R.M.Finger1_x',# 中指掌指關節
                      'EC2 Wight:R.M.Finger1_y': 'R.M.Finger1_y',
                      'EC2 Wight:R.M.Finger1_z': 'R.M.Finger1_z',
                      'Mouse:M2_x': 'M2_x',#  mouse2
                      'Mouse:M2_y': 'M2_y',
                      'Mouse:M2_z': 'M2_z', 
                      'Mouse:M3_x': 'M3_x',
                      'Mouse:M3_y': 'M3_y',
                      'Mouse:M3_z': 'M3_z', # mouse3
                      'Mouse:M4_x': 'M4_x',
                      'Mouse:M4_y': 'M4_y',
                      'Mouse:M4_z': 'M4_z'
                      }
# 定義滑鼠marker為 M2, M3, M4 的平均值
mouse_marker = ['M2_x', 'M2_y','M2_z', #  mouse2
                'M3_x', 'M3_y', 'M3_z', # mouse3
                'M4_x', 'M4_y', 'M4_z' # mouse4
                ]

# parameter setting
smoothing = 'lowpass'
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
all_dis_data = pd.DataFrame({}, columns = ['mouse'])
all_slope_data = pd.DataFrame({}, columns = ['data_name', 'mouse']+ muscle_name)
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
    
    # 讀取all MVC data
    MVC_value = pd.read_excel(processing_folder_path + '\\' + folder_name + '\\2.emg\\' + folder_name + '_all_MVC.xlsx')
    MVC_value = MVC_value.iloc[-1, 2:]
    # 讀取分期檔
    stage_file = pd.read_excel(stage_file_path, sheet_name=folder_name)
    # 第二次loop計算Gridshot的問題
    grid_shot_file_list = [file for file in c3d_list if 'GridShot' in file]
    for num in range(len(grid_shot_file_list)):
        print(grid_shot_file_list[num])
        # 0. 處理檔名問題------------------------------------------------------
        save_name, extension = os.path.splitext(grid_shot_file_list[num].split('\\', -1)[-1])
        fig_save_path = folder_path + processingData_folder + folder_name + "\\" + \
            "motion\\" + save_place
        data_svae_path = folder_path + processingData_folder + folder_name + "\\" + \
            "motion\\" + save_place
        print(data_svae_path + save_name + "_MedFreq.xlsx")
        # 將檔名拆開
        filepath, tempfilename = os.path.split(grid_shot_file_list[num])
        filename, extension = os.path.splitext(tempfilename)
        # 定義受試者編號以及滑鼠名稱
        trial_info = filename.split("_")
        
        trial_info = {"file_name": filename,
                      "subject": trial_info[0],
                      "task": trial_info[1],
                      "mouse": trial_info[2],
                      "trial_num": trial_info[3]}
        # 1. 讀取資料與初步處理-------------------------------------------------
        # 讀取 c3d data
        motion_info, motion_data, analog_info, analog_data, np_motion_data = func.read_c3d(grid_shot_file_list[num])
        motion_data = motion_data.fillna(0)
        # 重新命名欄位名稱
        # motion_data.rename(columns=c3d_recolumns_cortex, inplace=True)
        # 計算時間，時間從trigger on後三秒開始
        onset_analog = detect_onset(analog_data.loc[:, 'trigger1'],
                                    np.std(analog_data.loc[:50, 'trigger1']),
                                    n_above=10, n_below=0, show=True)
        oneset_idx = onset_analog[0, 0]
        # 設定 sampling rate
        sampling_time = 1/motion_info['frame_rate']
        # 前處理 motion data, 進行低通濾波
        lowpass_sos = signal.butter(2, lowpass_cutoff,  btype='low', fs=motion_info['frame_rate'], output='sos')
        filted_motion = pd.DataFrame(np.empty(np.shape(motion_data)),
                                     columns = motion_data.columns)
        filted_motion.iloc[:, 0] = motion_data.iloc[:, 0]
        for iii in range(np.shape(motion_data)[1]-1):
            filted_motion.iloc[:, iii+1] = signal.sosfiltfilt(lowpass_sos,
                                                              motion_data.iloc[:, iii+1].values)
        # 2. 計算資料 95% confidence ellipse -----------------------------------------------------------
        # 計算中指掌指關節座標點與 mouse 的距離
        # 平均 mouse M2, M3的距離, 相加除以2，再扣除R.M.Finger1
        mouse_mean = ((filted_motion.loc[oneset_idx:, ['M2_x', 'M2_y','M2_z']].values + \
                    filted_motion.loc[oneset_idx:, ['M3_x', 'M3_y', 'M3_z']].values) / 2) -\
                    filted_motion.loc[oneset_idx:, ['R.M.Finger1_x', 'R.M.Finger1_y', 'R.M.Finger1_z']].values
            
        COPxy = mouse_mean[:, :2] - mouse_mean[0, :2]
        Area95, fig = func.conf95_ellipse(COPxy)
        fig.savefig()                 

        # # 計算 mouse_mean 與 R.M.Finger1 的距離
        # dis_mouse_MFinger = np.sqrt((mouse_mean[:, 0] - filted_motion.loc[oneset_idx:, 'R.M.Finger1_x'])**2 + \
        #                             (mouse_mean[:, 0] - filted_motion.loc[oneset_idx:, 'R.M.Finger1_y'])**2 + \
        #                             (mouse_mean[:, 0] - filted_motion.loc[oneset_idx:, 'R.M.Finger1_z'])**2)
        
        # 3. 儲存資料-----------------------------------------------------------
        # 將資料儲存至矩陣
        # add_data = pd.DataFrame({"file_name": filename,
        #                          "subject": trial_info[0],
        #                          "task": trial_info[1],
        #                          "mouse": trial_info[2],
        #                          "min_dis": np.min(dis_mouse_MFinger),
        #                          "max_dis": np.max(dis_mouse_MFinger),
        #                          'std_dis': np.std(dis_mouse_MFinger),
        #                          "mean_dis": np.mean(dis_mouse_MFinger)
        #                          },
        #                         index=[0])
        # all_dis_data = pd.concat([add_data, all_dis_data],
        #                          ignore_index=True)
        
        # 疲勞分析
        # 處理 .csv 檔案
        if ".csv" in grid_shot_file_list[ii]:
            print(data_svae_path + save_name + "_MedFreq.xlsx")
            # 處理檔名及定義路徑
            
            
            # 找分期檔中的檔名
            for iii in range(np.shape(stage_file)[0]):
                if save_name in str(stage_file.loc[iii, 'EMG_File']):
                    # print(stage_file.loc[iii, 'EMG_File'])
                    mouse_name = stage_file.loc[iii, 'Mouse']
                    break
            # 將檔名加上滑鼠名稱
            save_name = save_name + "_" + mouse_name
        elif ".c3d" in grid_shot_file_list[ii]:
            save_name = save_name
        # 前處理EMG data
        processing_data, bandpass_filtered_data = emg.EMG_processing(grid_shot_file_list[ii], smoothing=smoothing)
        # 畫 bandpass 後之資料圖
        emg.plot_plot(bandpass_filtered_data, str(fig_svae_path),
                      filename, "_Bandpass")
        # 畫smoothing 後之資料圖
        emg.plot_plot(processing_data, str(fig_svae_path),
                      filename, str(smoothing + "_"))
        # writting data in worksheet
        file_name = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + MVC_folder + "\\data\\" + filename + end_name + ".xlsx"
        pd.DataFrame(processing_data).to_excel(data_svae_path + save_name + "_lowpass.xlsx",
                                               sheet_name='Sheet1', index=False, header=True)
        # 計算 iMVC
        emg_iMVC = pd.DataFrame(np.zeros(np.shape(processing_data)),
                                columns=processing_data.columns)
        emg_iMVC.iloc[:, 0] = processing_data.iloc[:, 0].values
        emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[:, 1:].values),
                                         MVC_value.values)*100
        
        pd.DataFrame(emg_iMVC).to_excel(data_svae_path + save_name + "_iMVC.xlsx",
                                        sheet_name='Sheet1', index=False, header=True)
        # 進行中頻率分析
        med_freq_data, slope_data = emg.median_frquency(grid_shot_file_list[ii],
                                                        duration, fig_svae_path, save_name)
        # 儲存斜率的資料，並合併成一個資料表
        slope_data['mouse'] = [mouse_name]
        slope_data['data_name'] = [save_name]
        all_slope_data = pd.concat([all_slope_data, slope_data])
        
        
        
        
        
        
        
all_slope_data.to_excel(static_folder + "MedFreq_Static.xlsx")
        
        
        
path = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\5. Statistics\\"
file_name = "GridShot_distance_statistic_" + formatted_date + ".xlsx"
all_dis_data.to_excel(path + file_name,
                      sheet_name='Sheet1', index=False, header=True)
        
                    






















