# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:49:22 2024

@author: Hsin.YH.Yang
"""

import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\4. code")
import AllSeries_general_func_20240327 as gen
import AllSeries_emg_func_20240327 as emg
from detecta import detect_onset
from scipy import signal
from datetime import datetime
import math

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# %% parameter setting
smoothing = 'lowpass'
c = 0.802
lowpass_cutoff = 10/c

c3d_recolumns_cortex = {'EC2 Wight:R.M.Finger1_x': 'R.M.Finger1_x',
                      'EC2 Wight:R.M.Finger1_y': 'R.M.Finger1_y',
                      'EC2 Wight:R.M.Finger1_z': 'R.M.Finger1_z' }
c3d_recolumns_vicon = {'RMD1_x': 'R.M.Finger1_x',
                        'RMD1_y': 'R.M.Finger1_y',
                        'RMD1_z': 'R.M.Finger1_z'}

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
                                   'Extensor carpi radialis_RMS', 'Flexor Carpi Radialis_RMS', 'Triceps_RMS',
                                   'Extensor carpi ulnaris_RMS', '1st. dorsal interosseous_RMS',
                                   'Abductor digiti quinti_RMS', 'Extensor Indicis_RMS', 'Biceps_RMS'])

for folder in rowdata_folder_list:
    print(0)
    # 讀資料夾下的 c3d data
    c3d_list = gen.Read_File(rowdata_folder + folder,
                             ".c3d", subfolder=False)
    motion_info, motion_data, analog_info, analog_data, np_motion_data = gen.read_c3d(r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\1. Motion Analysis\raw_data\S5\S5_LiftingShotL_EC_1.c3d")
    
    # 讀分期檔
    if folder not in ["S5", "S6", "S7"]:
        staging_file = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\ZowieAllSeries_StagingFile_20240326.xlsx",
                                     sheet_name=folder)
    
        # 找出存在 staging file 中的 LiftingShot
        liftingShot_list = []
        emg_list = []
        for c3d_name in range(len(staging_file['Motion_File_C3D'])):
            for i in range(len(c3d_list)):
                # print(c3d_name)
                if pd.isna(staging_file['Motion_File_C3D'][c3d_name]) != 1 \
                    and staging_file['Motion_File_C3D'][c3d_name] in c3d_list[i] \
                        and "LiftingShot" in c3d_list[i]:
                    liftingShot_list.append(c3d_list[i])
                    emg_list.append(staging_file['EMG_File'][c3d_name])
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

        
        # 创建一个包含两个子图的图形，并指定子图的布局
        fig, axes = plt.subplots(4, 1, figsize=(12, 6))  # 2 行 1 列，每个子图的大小为 (8, 6)
        
        # 绘制第一个子图
        axes[0].plot(filted_motion.loc[:, 'Frame'].values,
                      filted_motion.loc[:, 'R.M.Finger1_x'].values,
                      color='blue')  # 假设 data1 是一个 Series 或 DataFrame
        axes[0].plot(filted_motion.loc[first_max_idx+1, 'Frame'],
                      filted_motion.loc[first_max_idx+1, 'R.M.Finger1_x'],
                      "rx") 
        axes[0].set_title('R.M.Finger1_x')  # 设置子图标题
        # 绘制第二个子图
        axes[1].plot(filted_motion.loc[:, 'Frame'].values,
                      filted_motion.loc[:, 'R.M.Finger1_y'].values,
                      color='blue')  # 假设 data2 是一个 Series 或 DataFrame
        axes[1].plot(filted_motion.loc[first_max_idx+1, 'Frame'],
                      filted_motion.loc[first_max_idx+1, 'R.M.Finger1_y'],
                      "rx") 
        axes[1].set_title('R.M.Finger1_y')  # 设置子图标题
        # 绘制第三个子图
        axes[2].plot(filted_motion.loc[:, 'Frame'].values,
                      filted_motion.loc[:, 'R.M.Finger1_z'].values,
                      color='blue')  # 假设 data2 是一个 Series 或 DataFrame
        axes[2].plot(filted_motion.loc[first_max_idx+1, 'Frame'],
                      filted_motion.loc[first_max_idx+1, 'R.M.Finger1_z'],
                      "rx") 
        axes[2].set_title('R.M.Finger1_y')  # 设置子图标题
        # 绘制第四个子图
        axes[3].plot(filted_motion.loc[1:, 'Frame'].values,
                      vel_RM_finger.iloc[:, 0].values,
                      color='blue')  # 假设 data2 是一个 Series 或 DataFrame
        axes[3].plot(filted_motion.loc[first_max_idx+1, 'Frame'],
                      first_max_values,
                      "rx")  # 假设 data2 是一个 Series 或 DataFrame
        axes[3].set_title('combine velocity')  # 设置子图标题
        # 添加整体标题
        fig.suptitle(filename)  # 设置整体标题
        # 调整子图之间的间距
        plt.tight_layout()
        # plt.savefig(str(fig_sace_path + filename + ".jpg"),
        #             dpi=100)
        # 显示图形
        plt.show()
        # ---------------處理 EMG-------------------------------
        # 1. 讀取 MVC MAX
        MVC_value = pd.read_excel(processing_folder_path + '\\' + folder + '\\' + \
                                  folder + '_all_MVC.xlsx')
        MVC_value = MVC_value.iloc[-1, 2:]
        # 設置圖片儲存路徑
        fig_svae_path = emg_data_path + emg_processingData_folder + folder + "\\" + \
                        "LiftingShot\\figure\\"
        data_svae_path = emg_data_path + emg_processingData_folder + folder + "\\" + \
                        "LiftingShot\\data\\"
        
        # 2. 確認讀檔格式為 .csv or .c3d
        if folder not in ["S5", "S6", "S7"]:
            emg_path = emg_data_path + emg_RawData_folder + '\\' + folder + '\\' + emg_list[file_path]
            save_name, extension = os.path.splitext(emg_list[file_path].split('\\', -1)[-1])
            # 處理EMG資料
            processing_data, bandpass_filtered_data = emg.EMG_processing(emg_path, smoothing=smoothing)
            emg_smaple_rate = int(1 / (bandpass_filtered_data.iloc[1, 0] - bandpass_filtered_data.iloc[0, 0]))
            # 找 trigger on
            pa_start_onset = detect_onset(analog_data['trigger1'], # 將資料轉成一維型態
                                          np.std(analog_data['trigger1'][:500])*3,
                                          n_above=10, n_below=2, show=True)
            # 找最大值時間，要把 moton, analog, emg frame rate 對在一起
            emg_max_idx = int(first_max_idx/motion_info['frame_rate']*emg_smaple_rate - \
                pa_start_onset[0][0]/analog_info['frame_rate']*emg_smaple_rate)
            print(np.shape(processing_data)[0], emg_max_idx)        
        else:
            # 設定儲存檔名
            save_name, extension = os.path.splitext(liftingShot_list[file_path].split('\\', -1)[-1])
            # 處理 EMG 資料
            processing_data, bandpass_filtered_data = emg.EMG_processing(liftingShot_list[file_path], smoothing=smoothing)
            emg_smaple_rate = int(1 / (bandpass_filtered_data.iloc[1, 0] - bandpass_filtered_data.iloc[0, 0]))
            emg_max_idx = int(first_max_idx/motion_info['frame_rate']*emg_smaple_rate)
            print(np.shape(processing_data)[0], emg_max_idx)
        # 3. 畫 bandpass 後之資料圖
        emg.plot_plot(bandpass_filtered_data, str(fig_svae_path),
                      filename, "_Bandpass")
        
        # save = savepath + '\\' + filter_type + filename + ".jpg"
        n = int(math.ceil((np.shape(processing_data)[1] - 1) /2))
        fig, axs = plt.subplots(n, 2, figsize = (2*n+1,12), sharex='col')
        for i in range(np.shape(processing_data)[1]-1):
            x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
            # 設定子圖之參數
            axs[x, y].plot(processing_data.iloc[:, 0], processing_data.iloc[:, i+1])
            axs[x, y].set_title(processing_data.columns[i+1], fontsize=16)
            # 在X軸指定的範圍上加上色塊
            axs[x, y].axvspan(emg_max_idx/emg_smaple_rate - 0.2,
                              emg_max_idx/emg_smaple_rate + 0.05,
                              color='red', alpha=0.3)  # alpha是色塊的透明度
            # 設定科學符號 : 小數點後幾位數
            axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
        # 設定整張圖片之參數
        plt.suptitle(filename + str(smoothing + "_"), fontsize = 16)
        plt.tight_layout()
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("time (second)", fontsize = 14)
        plt.ylabel("Voltage (V)", fontsize = 14)
        plt.savefig(str(fig_svae_path + '\\' + "lowpass_" + filename + ".jpg"),
                    dpi=200, bbox_inches = "tight")
        plt.show()
        
        # 計算 iMVC
        emg_iMVC = pd.DataFrame(np.zeros(np.shape(processing_data)),
                                columns=processing_data.columns)
        emg_iMVC.iloc[:, 0] = processing_data.iloc[:, 0].values
        emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[:, 1:].values),
                                         MVC_value.values)*100
        pd.DataFrame(emg_iMVC).to_excel(data_svae_path + save_name + "_iMVC.xlsx",
                                        sheet_name='Sheet1', index=False, header=True)
        # 找到最大值時間，往前 200ms，往後 50ms 做 RMS
        rms_data = pd.DataFrame(np.zeros([1, np.shape(emg_iMVC)[1]-1]),
                                columns=emg_iMVC.columns[1:])
        for column in range(np.shape(emg_iMVC)[1]-1):
            rms_data.iloc[:, column] = np.sqrt(np.sum((emg_iMVC.iloc[int(emg_max_idx-emg_smaple_rate*0.2):\
                                                                     int(emg_max_idx+emg_smaple_rate*0.05), column+1])**2)
                                      /len(emg_iMVC.iloc[int(emg_max_idx-emg_smaple_rate*0.2):\
                                                         int(emg_max_idx+emg_smaple_rate*0.05), column+1]))
        # 將資料儲存至矩陣
        add_data = pd.DataFrame({"file_name":filename,
                                 "direction":vel_direc,
                                 "max_value": first_max_values,
                                 'Extensor carpi radialis_RMS':rms_data.iloc[:, 0],
                                 'Flexor Carpi Radialis_RMS':rms_data.iloc[:, 1],
                                 'Triceps_RMS':rms_data.iloc[:, 2],
                                 'Extensor carpi ulnaris_RMS':rms_data.iloc[:, 3],
                                 '1st. dorsal interosseous_RMS':rms_data.iloc[:, 4],
                                 'Abductor digiti quinti_RMS':rms_data.iloc[:, 5],
                                 'Extensor Indicis_RMS':rms_data.iloc[:, 6],
                                 'Biceps_RMS':rms_data.iloc[:, 7]},
                                index=[0])
        data_store = pd.concat([add_data, data_store], ignore_index=True)

path = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\5. Statistics\\"
file_name = "LiftingShot_data_" + formatted_date + ".xlsx"
# data_store.to_excel(path + file_name,
#                     sheet_name='Sheet1', index=False, header=True)

# %% statistic

    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    










