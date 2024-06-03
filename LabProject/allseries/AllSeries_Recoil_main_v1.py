# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:40:02 2024

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
# import AllSeries_emg_func_20240327 as emg
import AllSeries_general_func_20240327 as gen
import AllSeries_emg_func_20240327 as emg
# import gc
from detecta import detect_onset
from datetime import datetime
import matplotlib.pyplot as plt
# 获取当前日期和时间
now = datetime.now()

# 将日期转换为指定格式
formatted_date = now.strftime("%Y-%m-%d")

# 输出格式化后的日期
print("当前日期：", formatted_date)
# %%
data_path = r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\\"
RawData_folder = "raw_data\\"
processingData_folder = "processing_data\\"
static_folder = r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\\"
motion_folder = "1. Motion\\"
emg_forder = "2. EMG\\"
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
all_mouse_name = ['Gpro', 'ZOWIE_U', 'ZOWIE_ZA', 'ZOWIE_S', 'ZOWIE_FK', 'ZOWIE_EC']
muscle_name = ['Extensor carpi radialis', 'Flexor Carpi Radialis', 'Triceps',
               'Extensor carpi ulnaris', '1st. dorsal interosseous', 
               'Abductor digiti quinti', 'Extensor Indicis', 'Biceps']
# 替換相異欄位名稱
c3d_recolumns_cortex = {
                        'EC2 Wight:R.I.Finger3_x': 'R.I.Finger3_x',# 食指遠端關節
                        'EC2 Wight:R.I.Finger3_y': 'R.I.Finger3_y',
                        'EC2 Wight:R.I.Finger3_z': 'R.I.Finger3_z',
                        'EC2 Wight:R.M.Finger1_x': 'R.M.Finger1_x',# 中指掌指關節
                        'EC2 Wight:R.M.Finger1_y': 'R.M.Finger1_y',
                        'EC2 Wight:R.M.Finger1_z': 'R.M.Finger1_z'
                      }
# 定義滑鼠marker為 M2, M3, M4 的平均值
mouse_marker = [
                'M2_x', 'M2_y','M2_z', #  mouse2
                'M3_x', 'M3_y', 'M3_z', # mouse3
                'M4_x', 'M4_y', 'M4_z' # mouse4
                ]
# %%

def generate_mouse_dict(new_name="Mouse"):
    mouse_dict = {}
    for i in range(1, 5):
        mouse_dict[f'{new_name}:M{i}_x'] = f'M{i}_x'
        mouse_dict[f'{new_name}:M{i}_y'] = f'M{i}_y'
        mouse_dict[f'{new_name}:M{i}_z'] = f'M{i}_z'
    return mouse_dict


# %%
rowdata_folder_path = data_path + motion_folder + RawData_folder + "\\" + sub_folder
# 去除有“.“開頭的檔案
rowdata_folder_list  = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.')]
processing_folder_path = data_path + motion_folder + "\\" + processingData_folder + "\\" + sub_folder
processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.')]
# read staging file
staging_file_path = r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\ZowieAllSeriesRedo_StagingFile_20240531.xlsx"

# %% 分析壓槍
"""
1.1. 檔名定義, 任務基本資訊定義
1.2. 讀取 c3d file and EMG file 以及更新資料欄位名稱
1.3. 定義基本參數, motion and EMG data's samlping rate, find trigger onset time
1.4. motion data filting 

"""
tic = time.process_time()
# 建立 slope data 要儲存之位置
all_dis_data = pd.DataFrame({}, columns = ['mouse'])
# 0. 依序讀取所有的 rowdata folder 下的資料 ------------------------------------
for folder in range(len(rowdata_folder_list)):
    # 讀資料夾下的 c3d data
    c3d_list = gen.Read_File(data_path + motion_folder + RawData_folder + rowdata_folder_list[folder],
                             ".c3d", subfolder=False)
    csv_list = gen.Read_File(data_path + emg_forder + RawData_folder + rowdata_folder_list[folder],
                             ".csv", subfolder=False)
    # 讀分期檔
    staging_file = pd.read_excel(staging_file_path,
                                 sheet_name = rowdata_folder_list[folder])

    # 找出存在 staging file 中的 Recoil
    motion_list = {"motion": [],
                   "emg": []}
    for index in range(len(staging_file)):
        for i in range(len(c3d_list)):
            if pd.isna(staging_file['Motion_File_C3D'][index]) != 1 \
                and staging_file['Motion_File_C3D'][index] in c3d_list[i] \
                    and "Recoil" in c3d_list[i]:
                motion_list["motion"].append(c3d_list[i])
        for i in range(len(csv_list)):
            if pd.isna(staging_file['EMG_File'][index]) != 1 \
                and staging_file['EMG_File'][index] in csv_list[i] \
                    and "Recoil" in csv_list[i]:
                motion_list["emg"].append(csv_list[i])
    # 1. 依序處理所讀取的 motion 以及 emg file
    for ii in range(len(motion_list["motion"])):
        filepath, tempfilename = os.path.split(motion_list["motion"][ii])
        filename, extension = os.path.splitext(tempfilename)
        for iii in range(len(staging_file['Motion_File_C3D'])):
            if tempfilename == staging_file['Motion_File_C3D'][iii]:
                print(motion_list["motion"][ii])
                # 1. 基本資料處理------------------------------------------------------
                # 1.1. 檔名定義, 任務基本資訊定義 ---------------------------------------
                # save_name, extension = os.path.splitext(motion_list["motion"][ii].split('\\', -1)[-1])
                fig_svae_path = data_path + motion_folder + processingData_folder + \
                                rowdata_folder_list[folder] + "\\" + filename + ".jpg"
                # data_svae_path = data_path + processingData_folder + motion_list["motion"][ii] + "\\" + \
                #                 "FatigueAnalysis\\data\\"
                # 定義受試者編號以及滑鼠名稱
                trial_info = filename.split("_")
                
                # 1.2. 讀取 c3d 以及 emg file -----------------------------------------
                motion_info, motion_data, analog_info, analog_data, np_motion_data = gen.read_c3d(motion_list["motion"][ii])
                processing_data, bandpass_filtered_data = emg.EMG_processing(motion_list["emg"][ii], smoothing=smoothing)
                # 處理欄位名稱問題
                for mouse in all_mouse_name:
                    if trial_info[2] in mouse:
                        mouse_dict = generate_mouse_dict(new_name=mouse)
                # 合併人體 marker set 以及滑鼠 marker set
                c3d_recolumns_cortex.update(mouse_dict) 
                # 更新 motion data's columns name
                motion_data.rename(columns=c3d_recolumns_cortex, inplace=True)
                
                # 1.3. 定義基本參數, sampling rate -------------------------------------
                motion_sampling_rate = 1/motion_info['frame_rate']
                emg_smaple_rate = int(1 / (bandpass_filtered_data.iloc[1, 0] - bandpass_filtered_data.iloc[0, 0]))
                # find the time of trigger on, 找出大於前 500 frame 平均的三個標準差
                pa_start_onset = detect_onset(analog_data['trigger1'], # 將資料轉成一維型態
                                              np.std(analog_data['trigger1'][:500])*3,
                                              n_above=10, n_below=2, show=True)
                motion_onset = int(pa_start_onset[0, 0]/10 + 240)
                # 1.4. 基本資料處理, filting motion data
                lowpass_sos = signal.butter(2, lowpass_cutoff,  btype='low', fs=motion_info['frame_rate'], output='sos')
                filted_motion = pd.DataFrame(np.empty(np.shape(motion_data)),
                                             columns = motion_data.columns)
                filted_motion.iloc[:, 0] = motion_data.iloc[:, 0]
                for i in range(np.shape(motion_data)[1]-1):
                    filted_motion.iloc[:, i+1] = signal.sosfiltfilt(lowpass_sos,
                                                                    motion_data.iloc[:, i+1].values)
        
                # 2. 資料處理 ----------------------------------------------------------
                # 2.1. 找出食指按壓的時間點
                """
                R.I.Finger3
                1. 從 trigger on 後開始三秒，計算 50 frame 的平均，
                2. 找開始 "小於" 及 "大於" 該平均的時間、閾值設定 10%
                3. 計算食指 x 軸負向的速度
                4. 讀分期檔，開始時間再加 600 frame
                """
                recoil_begin = int(staging_file['Recoil_begin'][iii])
# %%
                # 繪圖
                # 创建一个包含四个子图的图形，并指定子图的布局
                fig, axes = plt.subplots(4, 1, figsize=(8, 10))  
                # 绘制第一个子图
                axes[0].plot(filted_motion.loc[:, 'Frame'].values,
                             filted_motion.loc[:, 'R.I.Finger3_x'].values,
                             color='blue')  # 假设 data1 是一个 Series 或 DataFrame
                axes[0].axvline(motion_onset/motion_info['frame_rate'], color='r', linestyle='--') # trigger onset
                axes[0].axvline((recoil_begin)/motion_info['frame_rate'], # R.I.Finger down
                                color='c', linestyle='--') 
                axes[0].axvline((recoil_begin + 575)/motion_info['frame_rate'], # R.I.Finger up
                                color='c', linestyle='--')
                axes[0].set_xlim(0, analog_data['Frame'].iloc[-1])
                axes[0].set_title('R.I.Finger3_x')  # 设置子图标题
                # 绘制第二个子图
                axes[1].plot(filted_motion.loc[:, 'Frame'].values,
                             filted_motion.loc[:, 'R.I.Finger3_y'].values,
                             color='blue')  # 假设 data2 是一个 Series 或 DataFrame
                axes[1].axvline((recoil_begin)/motion_info['frame_rate'], color='r', linestyle='--') # trigger onset
                axes[1].axvline((recoil_begin + 575)/motion_info['frame_rate'], # R.I.Finger down
                                color='c', linestyle='--') 
                axes[1].set_xlim(0, analog_data['Frame'].iloc[-1])
                axes[1].set_title('R.I.Finger3_y')  # 设置子图标题
                # 绘制第三个子图
                axes[2].plot(filted_motion.loc[:, 'Frame'].values,
                             filted_motion.loc[:, 'R.I.Finger3_z'].values,
                             color='blue')  # 假设 data2 是一个 Series 或 DataFrame
                axes[2].axvline((recoil_begin)/motion_info['frame_rate'], color='r', linestyle='--') # trigger onset
                axes[2].axvline((recoil_begin + 575)/motion_info['frame_rate'], # R.I.Finger down
                                color='c', linestyle='--') 
                axes[2].set_xlim(0, analog_data['Frame'].iloc[-1])
                axes[2].set_title('R.I.Finger3_z')  # 设置子图标题
                # 绘制第四个子图
                axes[3].plot(analog_data['Frame'].values,
                             analog_data['trigger1'],
                             color='blue')  # 假设 data2 是一个 Series 或 DataFrame)  # 假设 data2 是一个 Series 或 DataFrame
                axes[3].axvline(pa_start_onset[0, 0]/analog_info['frame_rate'], # trigger onset
                                color='r', linestyle='--') 
                axes[3].set_xlim(0, analog_data['Frame'].iloc[-1])
                axes[3].set_title('Analog')  # 设置子图标题
                # 添加整体标题
                fig.suptitle(filename)  # 设置整体标题
                # 调整子图之间的间距
                plt.tight_layout()
                plt.savefig(fig_svae_path,
                            dpi=100)
                # 显示图形
                plt.show()
    
        
        




        





















