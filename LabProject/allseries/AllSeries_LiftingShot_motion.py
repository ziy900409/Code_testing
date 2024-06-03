# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 08:34:44 2024

@author: Hsin.YH.Yang
"""

import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\allseries")
import AllSeries_general_func_20240327 as gen
# import AllSeries_emg_func_20240327 as emg
from detecta import detect_onset
from scipy import signal
from datetime import datetime
# import math

import numpy as np
import pandas as pd
import os
# import matplotlib.pyplot as plt
 # %% parameter setting
smoothing = 'lowpass'
c = 0.802
lowpass_cutoff = 10/c



# 获取当前日期和时间
now = datetime.now()

# 将日期转换为指定格式
formatted_date = now.strftime("%Y-%m-%d")

# 输出格式化后的日期
print("当前日期：", formatted_date)
# path setting
emg_data_path = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\2. EMG\\"
emg_RawData_folder = "raw_data\\"
emg_processingData_folder = "processing_data\\"
static_folder = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\5. Statics\\"
MVC_folder = "MVC\\"
# %% 定義關節座標資料
# 替換相異欄位名稱
c3d_recolumns_cortex = {'EC2 Wight:R.M.Finger1_x': 'R.M.Finger1_x',# 中指掌指關節
                      'EC2 Wight:R.M.Finger1_y': 'R.M.Finger1_y',
                      'EC2 Wight:R.M.Finger1_z': 'R.M.Finger1_z',
                      'EC2 Wight:R.Wrist.Uln_x': 'R.Wrist.Uln_x', # 手腕尺側莖狀突
                      'EC2 Wight:R.Wrist.Uln_y': 'R.Wrist.Uln_y',
                      'EC2 Wight:R.Wrist.Uln_z': 'R.Wrist.Uln_z',
                      'EC2 Wight:R.Wrist.Rad_x': 'R.Wrist.Rad_x', # 手腕橈側莖狀突
                      'EC2 Wight:R.Wrist.Rad_y': 'R.Wrist.Rad_y',
                      'EC2 Wight:R.Wrist.Rad_z': 'R.Wrist.Rad_z',
                      'EC2 Wight:R.Thumb1_x': 'R.Thumb1_x', # 拇指掌指關節
                      'EC2 Wight:R.Thumb1_y': 'R.Thumb1_y',
                      'EC2 Wight:R.Thumb1_z': 'R.Thumb1_z',
                      'EC2 Wight:R.Thumb2_x': 'R.Thumb2_x', # 拇指第一指關節
                      'EC2 Wight:R.Thumb2_y': 'R.Thumb2_y',
                      'EC2 Wight:R.Thumb2_z': 'R.Thumb2_z',
                      'EC2 Wight:R.R.Finger1_x': 'R.R.Finger1_x', # 無名指掌指關節
                      'EC2 Wight:R.R.Finger1_y': 'R.R.Finger1_y',
                      'EC2 Wight:R.R.Finger1_z': 'R.R.Finger1_z',
                      'EC2 Wight:R.R.Finger2_x': 'R.R.Finger2_x', # 無名指第一指關節
                      'EC2 Wight:R.R.Finger2_y': 'R.R.Finger2_y',
                      'EC2 Wight:R.R.Finger2_z': 'R.R.Finger2_z',
                      'EC2 Wight:R.P.Finger1_x': 'R.P.Finger1_x', # 小指掌指關節
                      'EC2 Wight:R.P.Finger1_y': 'R.P.Finger1_y',
                      'EC2 Wight:R.P.Finger1_z': 'R.P.Finger1_z',
                      'EC2 Wight:R.P.Finger2_x': 'R.P.Finger2_x', # 小指第一指關節
                      'EC2 Wight:R.P.Finger2_y': 'R.P.Finger2_y',
                      'EC2 Wight:R.P.Finger2_z': 'R.P.Finger2_z',
                      'EC2 Wight:R.Elbow.Lat_x':'R.Elbow.Lat_x', # 手肘外上髁
                      'EC2 Wight:R.Elbow.Lat_y':'R.Elbow.Lat_y',
                      'EC2 Wight:R.Elbow.Lat_z':'R.Elbow.Lat_z'
                      }
c3d_recolumns_vicon = {'RMD1_x': 'R.M.Finger1_x', # 中指掌指關節
                       'RMD1_y': 'R.M.Finger1_y',
                       'RMD1_z': 'R.M.Finger1_z',
                       'RUS_x': 'R.Wrist.Uln_x', # 手腕尺側莖狀突
                       'RUS_y': 'R.Wrist.Uln_y',
                       'RUS_z': 'R.Wrist.Uln_z',
                       'RRS_x': 'R.Wrist.Rad_x', # 手腕橈側莖狀突
                       'RRS_y': 'R.Wrist.Rad_y',
                       'RRS_z': 'R.Wrist.Rad_z',
                       'RTB1_x': 'R.Thumb1_x', # 拇指掌指關節
                       'RTB1_y': 'R.Thumb1_y',
                       'RTB1_z': 'R.Thumb1_z',
                       'RTB2_x': 'R.Thumb2_x', # 拇指第一指關節
                       'RTB2_y': 'R.Thumb2_y',
                       'RTB2_z': 'R.Thumb2_z',
                       'RRG1_x': 'R.R.Finger1_x', # 無名指掌指關節
                       'RRG1_y': 'R.R.Finger1_y',
                       'RRG1_z': 'R.R.Finger1_z',
                       'RRG2_x': 'R.R.Finger2_x', # 無名指第一指關節
                       'RRG2_y': 'R.R.Finger2_y',
                       'RRG2_z': 'R.R.Finger2_z',
                       'RLT1_x': 'R.P.Finger1_x', #小指掌指關節
                       'RLT1_y': 'R.P.Finger1_y',
                       'RLT1_z': 'R.P.Finger1_z',
                       'RLT2_x': 'R.P.Finger2_x', # 小指第一指關節
                       'RLT2_y': 'R.P.Finger2_y',
                       'RLT2_z': 'R.P.Finger2_z',
                       'RUEL_x': 'R.Elbow.Lat_x', # 手肘外上髁
                       'RUEL_y': 'R.Elbow.Lat_y',
                       'RUEL_z': 'R.Elbow.Lat_z'}
# 定義大拇指角度所需 markerset: 拇指第一指關節、拇指掌指關節、手腕橈側莖狀突
thumb_marker = ['R.Wrist.Rad_x', 'R.Wrist.Rad_y','R.Wrist.Rad_z', # 手腕橈側莖狀突
                'R.Thumb1_x', 'R.Thumb1_y', 'R.Thumb1_z', # 拇指掌指關節
                'R.Thumb2_x', 'R.Thumb2_y', 'R.Thumb2_z' # 拇指第一指關節
                ]
little_marker = ['R.Wrist.Uln_x', 'R.Wrist.Uln_y', 'R.Wrist.Uln_z', # 手腕尺側莖狀突
                 'R.P.Finger1_x', 'R.P.Finger1_y', 'R.P.Finger1_z', #小指掌指關節
                 'R.P.Finger2_x', 'R.P.Finger2_y', 'R.P.Finger2_z' # 小指第一指關節
                 ]
wrist_marker = ['R.M.Finger1_x', 'R.M.Finger1_y', 'R.M.Finger1_z', # 中指掌指關節
                'R.Wrist.Uln_x', 'R.Wrist.Uln_y', 'R.Wrist.Uln_z', # 手腕尺側莖狀突
                'R.Wrist.Rad_x', 'R.Wrist.Rad_y', 'R.Wrist.Rad_z', # 手腕橈側莖狀突
                'R.Elbow.Lat_x', 'R.Elbow.Lat_y','R.Elbow.Lat_z' # 手肘外上髁
                ]
# %% 設置資料夾路徑
rowdata_folder = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\1. Motion Analysis\raw_data\\"

rowdata_folder_path = rowdata_folder
# 去除有“.“開頭的檔案 and 只獲得資料夾路徑，排除其他可能的檔案格式
rowdata_folder_list  = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.') \
                       and os.path.isdir(os.path.join(rowdata_folder_path, f))]
processing_folder_path = emg_data_path + "\\" + emg_processingData_folder + "\\" 
# 去除有“.“開頭的檔案 and 只獲得資料夾路徑，排除其他可能的檔案格式
processing_folder_list  = [f for f in os.listdir(processing_folder_path) if not f.startswith('.') \
                        and os.path.isdir(os.path.join(processing_folder_path, f))]
    
# %%
'''
加入讀取分期檔，避免檔案誤讀的問題
'''
c3d_list = gen.Read_File(r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\1. Motion Analysis\raw_data",
                         ".c3d", subfolder=True)
fig_sace_path = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\1. Motion Analysis\processing_data\\"
liftingShot_list = []
for i in range(len(c3d_list)):
    if "LiftingShot" in c3d_list[i]:
        liftingShot_list.append(c3d_list[i])

data_store = pd.DataFrame({},
                          columns=['file_name', "direction", "max_value",
                                   'Thumb angle', 'Little finger angle', 'Wrist angle'])


for folder in rowdata_folder_list:
    print(0)
    # 讀資料夾下的 c3d data
    c3d_list = gen.Read_File(rowdata_folder + folder,
                             ".c3d", subfolder=False)
    # motion_info, motion_data, analog_info, analog_data, np_motion_data = gen.read_c3d(r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\1. Motion Analysis\raw_data\S5\S5_LiftingShotL_EC_1.c3d")
    
    # 讀分期檔
    if folder not in ["S5", "S6", "S7", "S8"]:
        staging_file = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\ZowieAllSeries_StagingFile_20240326.xlsx",
                                     sheet_name=folder)
    
        # 找出存在 staging file 中的 LiftingShot
        liftingShot_list = []
        # emg_list = []
        for c3d_name in range(len(staging_file['Motion_File_C3D'])):
            for i in range(len(c3d_list)):
                # print(c3d_name)
                if pd.isna(staging_file['Motion_File_C3D'][c3d_name]) != 1 \
                    and staging_file['Motion_File_C3D'][c3d_name] in c3d_list[i] \
                        and "LiftingShot" in c3d_list[i]:
                    liftingShot_list.append(c3d_list[i])
                    # emg_list.append(staging_file['EMG_File'][c3d_name])
    else:
        liftingShot_list = []
        for i in range(len(c3d_list)):
            # print(c3d_name)
            if "LiftingShot" in c3d_list[i]:
                liftingShot_list.append(c3d_list[i])
        
    for file_path in range(len(liftingShot_list)):  
        print(liftingShot_list[file_path])
        # 將檔名拆分
        filepath, tempfilename = os.path.split(liftingShot_list[file_path])
        filename, extension = os.path.splitext(tempfilename)
        # 讀取 c3d data
        motion_info, motion_data, analog_info, analog_data, np_motion_data = gen.read_c3d(liftingShot_list[file_path])
        motion_data = motion_data.fillna(0)
        # 確定檔案是屬於 cortex or vicon, 使用 columns name 確認
        if 'EC2 Wight:R.M.Finger1_x' in motion_data.columns:
            print(0)
            motion_data.rename(columns=c3d_recolumns_cortex, inplace=True)
            # 計算時間，時間從trigger on後三秒開始
            onset_analog = detect_onset(analog_data.loc[:, 'trigger1'],
                                        np.std(analog_data.loc[:50, 'trigger1']),
                                        n_above=10, n_below=0, show=True)
            motion_start = int((onset_analog[0, 0] + 3*analog_info['frame_rate']) / 10)
        elif 'RMD1_x' in motion_data.columns:
            print(1)
            motion_data.rename(columns=c3d_recolumns_vicon, inplace=True)
            motion_start = int(4 * motion_info['frame_rate'])
        

        # 計算 motion start time
        
        # 設定 sampling rate
        sampling_time = 1/motion_info['frame_rate']
        # filting motion data
        lowpass_sos = signal.butter(2, lowpass_cutoff,  btype='low', fs=motion_info['frame_rate'], output='sos')
        filted_motion = pd.DataFrame(np.empty(np.shape(motion_data)),
                                     columns = motion_data.columns)
        filted_motion.iloc[:, 0] = motion_data.iloc[:, 0]
        for i in range(np.shape(motion_data)[1]-1):
            filted_motion.iloc[:, i+1] = signal.sosfiltfilt(lowpass_sos,
                                                    motion_data.iloc[:, i+1].values)
        # 計算合速度: 'EC2 Wight:R.M.Finger1_x', 'EC2 Wight:R.M.Finger1_y', 'EC2 Wight:R.M.Finger1_z'
        vel_RM_finger = pd.DataFrame(np.empty([np.shape(motion_data)[0]-1, 1]))
        
        
        for i in range(np.shape(motion_data)[0]-1):
            vel_RM_finger.iloc[i, 0] = np.sqrt(
                ((filted_motion.loc[i+1, 'R.M.Finger1_x'] - \
                 filted_motion.loc[i, 'R.M.Finger1_x'])/sampling_time)**2 + \
                ((filted_motion.loc[i+1, 'R.M.Finger1_y'] - \
                 filted_motion.loc[i, 'R.M.Finger1_y'])/sampling_time)**2 + \
                ((filted_motion.loc[i+1, 'R.M.Finger1_z'] - \
                 filted_motion.loc[i, 'R.M.Finger1_z'])/sampling_time)**2 \
                    )
        
        # 尋找和速度最大值
        # 先 detect onset，再找最大值
        onset_lifting = detect_onset(vel_RM_finger.iloc[motion_start:motion_start+int(motion_info['frame_rate']*4), 0].values,
                                     1.2*np.mean(vel_RM_finger.iloc[motion_start:motion_start+18, 0].values),
                                     n_above=10, n_below=0, show=True)
        
        # 限定只能找 motion start 四秒以內的區間
        if len(vel_RM_finger) > int(motion_start + motion_info['frame_rate']*4):
            first_max_values = vel_RM_finger.iloc[motion_start:motion_start+int(motion_info['frame_rate']*4), 0].max()
            first_max_idx = vel_RM_finger.iloc[motion_start:motion_start+int(motion_info['frame_rate']*4), 0].idxmax()
        else:
            first_max_values = vel_RM_finger.iloc[motion_start:, 0].max()
            first_max_idx = vel_RM_finger.iloc[motion_start:, 0].idxmax()
        # 判斷向左向右
        if "LiftingShotL" in liftingShot_list[file_path]:
            vel_direc = "L"
        elif "LiftingShotR" in liftingShot_list[file_path]:
            vel_direc = "R"
        else:
            vel_direc = "non"
        
        # 定義拇指夾角: 拇指第一指關節、拇指掌指關節、手腕橈側莖狀突

        cord_wrist = (filted_motion.loc[:, ['R.Wrist.Uln_x', 'R.Wrist.Uln_y', 'R.Wrist.Uln_z']].values + \
            filted_motion.loc[:, ['R.Wrist.Rad_x', 'R.Wrist.Rad_y', 'R.Wrist.Rad_z']].values)/2
            
        
        # 創建資料儲存空間
        angle_thumb = pd.DataFrame(np.empty(np.shape(filted_motion.iloc[:, 0])))
        angle_little = pd.DataFrame(np.empty(np.shape(filted_motion.iloc[:, 0])))
        angle_wrist = pd.DataFrame(np.empty(np.shape(filted_motion.iloc[:, 0])))
        # 計算大拇指夾角
        angle_thumb = gen.included_angle(filted_motion.loc[:, ['R.Wrist.Rad_x', 'R.Wrist.Rad_y','R.Wrist.Rad_z']],
                                         filted_motion.loc[:, ['R.Thumb1_x', 'R.Thumb1_y', 'R.Thumb1_z']],
                                         filted_motion.loc[:, ['R.Thumb2_x', 'R.Thumb2_y', 'R.Thumb2_z']]
                                         )
        # 計算無名指夾角
        angle_ring = gen.included_angle(cord_wrist,
                                        filted_motion.loc[:, ['R.R.Finger1_x', 'R.R.Finger1_y', 'R.R.Finger1_z']],
                                        filted_motion.loc[:, ['R.R.Finger2_x', 'R.R.Finger2_y', 'R.R.Finger2_z']]
                                        )

        # 計算小拇指夾角
        angle_little = gen.included_angle(filted_motion.loc[:, ['R.Wrist.Uln_x', 'R.Wrist.Uln_y', 'R.Wrist.Uln_z']],
                                         filted_motion.loc[:, ['R.P.Finger1_x', 'R.P.Finger1_y', 'R.P.Finger1_z']],
                                         filted_motion.loc[:, ['R.P.Finger2_x', 'R.P.Finger2_y', 'R.P.Finger2_z']]
                                         )
        # 計算手腕關節夾角
        angle_wrist = gen.included_angle(filted_motion.loc[:, ['R.M.Finger1_x', 'R.M.Finger1_y', 'R.M.Finger1_z']],
                                         cord_wrist,
                                         filted_motion.loc[:, ['R.Elbow.Lat_x', 'R.Elbow.Lat_y','R.Elbow.Lat_z']]
                                         )
        # 將資料儲存至矩陣
        add_data = pd.DataFrame({"file_name":filename,
                                 "direction":vel_direc,
                                 "max_value": first_max_values,
                                 'Thumb angle': angle_thumb[first_max_idx],
                                 'Ring angle': angle_ring[first_max_idx],
                                 'Little finger angle': angle_little[first_max_idx],
                                 'Wrist angle': angle_wrist[first_max_idx]},
                                index=[0])
        data_store = pd.concat([add_data, data_store], ignore_index=True)

path = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\5. Statistics\\"
file_name = "LiftingShot_angle_data_" + formatted_date + ".xlsx"
data_store.to_excel(path + file_name,
                    sheet_name='Sheet1', index=False, header=True)
        















