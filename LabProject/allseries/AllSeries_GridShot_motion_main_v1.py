# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:40:12 2024

計算中指與滑鼠的直線距離

@author: Hsin.YH.Yang
"""

import pandas as pd
import numpy as np
import time
import os
import sys
from scipy import signal
# 路徑改成你放自己code的資料夾
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\allseries")
import AllSeries_emg_func_20240327 as emg
import AllSeries_general_func_20240327 as gen
import gc
from detecta import detect_onset
# %% parameter setting
# path setting
data_path = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\1. Motion Analysis\\"
RawData_folder = "raw_data\\"
processingData_folder = "processing_data\\"
static_folder = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\5. Statics\\"
MVC_folder = "MVC\\"
sub_folder = ""
fig_save = "figure\\"
end_name = "_ed"
smoothing = 'lowpass'
# parameter setting
smoothing = 'lowpass'
c = 0.802
lowpass_cutoff = 10/c
duration = 1
all_mouse_name = ['_Gpro_', '_U_', '_ZA_', '_S_', '_FK_', '_EC_']
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

# %% 欲儲存之資料



# %% 讀取資料夾路徑

rowdata_folder_path = data_path + RawData_folder + "\\" + sub_folder
# 去除有“.“開頭的檔案
rowdata_folder_list  = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.')]
processing_folder_path = data_path + "\\" + processingData_folder + "\\" + sub_folder
processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.')]
# read staging file
staging_file_path = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\ZowieAllSeries_StagingFile_20240326.xlsx"
# %% 分析疲勞
tic = time.process_time()
# 建立 slope data 要儲存之位置
all_slope_data = pd.DataFrame({}, columns = ['data_name', 'mouse']+ muscle_name)
for i in range(len(rowdata_folder_list)):
    print(0)
    c3d_file_path = gen.Read_File(data_path + RawData_folder +  rowdata_folder_list[i],
                               ".c3d")
    all_file_path = c3d_file_path
    grid_shot_file_list = [file for file in all_file_path if 'GridShot' in file]

    for ii in range(len(grid_shot_file_list)):
        # 處理檔名問題
        save_name, extension = os.path.splitext(grid_shot_file_list[ii].split('\\', -1)[-1])
        fig_svae_path = data_path + processingData_folder + rowdata_folder_list[i] + "\\" + \
            "FatigueAnalysis\\figure\\"
        data_svae_path = data_path + processingData_folder + rowdata_folder_list[i] + "\\" + \
            "FatigueAnalysis\\data\\"
        print(data_svae_path + save_name + "_MedFreq.xlsx")
        # 將檔名拆開
        filepath, tempfilename = os.path.split(grid_shot_file_list[ii])
        filename, extension = os.path.splitext(tempfilename)
        # 讀取 c3d data
        motion_info, motion_data, analog_info, analog_data, np_motion_data = gen.read_c3d(grid_shot_file_list[ii])
        motion_data = motion_data.fillna(0)
        # 重新命名欄位名稱
        motion_data.rename(columns=c3d_recolumns_cortex, inplace=True)
        # 計算時間，時間從trigger on後三秒開始
        onset_analog = detect_onset(analog_data.loc[:, 'trigger1'],
                                    np.std(analog_data.loc[:50, 'trigger1']),
                                    n_above=10, n_below=0, show=True)
        # 設定 sampling rate
        sampling_time = 1/motion_info['frame_rate']
        # 前處理 motion data, 進行低通濾波
        lowpass_sos = signal.butter(2, lowpass_cutoff,  btype='low', fs=motion_info['frame_rate'], output='sos')
        filted_motion = pd.DataFrame(np.empty(np.shape(motion_data)),
                                     columns = motion_data.columns)
        filted_motion.iloc[:, 0] = motion_data.iloc[:, 0]
        for i in range(np.shape(motion_data)[1]-1):
            filted_motion.iloc[:, i+1] = signal.sosfiltfilt(lowpass_sos,
                                                    motion_data.iloc[:, i+1].values)
        # 計算中指掌指關節座標點與 mouse 的距離
        
        
        
        
        
        
        
        
        
        
        
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

        

        # 繪製傅立葉轉換圖，包含要不要做 notch filter
        # emg.Fourier_plot(grid_shot_file_list[ii], fig_svae_path, save_name)
        # emg.Fourier_plot(grid_shot_file_list[ii], fig_svae_path, save_name, notch=True)
        # 將檔案儲存成.xlsx
        # med_freq_data.to_excel(data_svae_path + save_name + "_MedFreq.xlsx")
        print(data_svae_path + save_name + "_MedFreq.xlsx")