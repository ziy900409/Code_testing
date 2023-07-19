# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:16:28 2023

流程：
1. 設定資料夾路徑與程式存放路徑
2. 先抓 release time
3. 預處理MVC
    3.1 檢查MVC統計資料
4. 處理motion

@author: Hsin.YH.Yang
"""
# %% import library
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"C:\Users\Public\BenQ\myPyCode\NPL")
# 將read_c3d function 加進現有的工作環境中

import ArcheryFunction_20230609 as af
import os
import time
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import gc

# %% 設定自己的資料路徑
# 資料路徑
data_path = r"E:\python\EMG_Data\Shooting_data_20230617\202305 Shooting"
# 設定資料夾
RawData_folder = "\\Raw_Data"
processingData_folder = "\\Processing_Data"
fig_save = "\\figure"
# 子資料夾名稱
sub_folder = "\\"
# 動作資料夾名稱
motion_folder = "motion"
# MVC資料夾名稱
MVC_folder = "MVC"
# 給定預抓分期資料的欄位名稱或標號：例如：[R EXTENSOR GROUP: ACC.Y 1] or [5]
release_staging_column = '5'
# downsampling frequency
down_freq = 1000
# 抓放箭時候前後秒數
# example : [秒數*採樣頻率, 秒數*採樣頻率]
release = [3*down_freq, 0*down_freq]
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.1 # 窗格長度 (單位 second)
overlap_len = 0.5 # 百分比 (%)
# 預處理資料可修改檔名，並新增標籤，如：S2_MVC_Rep_1.16 -> S2_MVC_Rep_1.16_low
end_name = "_ed"
# 平滑處理方式 ex: lowpass, rms, moving
smoothing = 'rms'
# median frequency duration
duration = 1 # unit : second
# 設定分期檔路徑
staging_file_path = r"E:\python\EMG_Data\Shooting_data_20230617\202305 Shooting\Shooting_staging_20230714.xlsx"

# %% 路徑設置

rowdata_folder_path = data_path + RawData_folder + "\\" + sub_folder
# 去除有“.“開頭的檔案
rowdata_folder_list  = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.')]
processing_folder_path = data_path + "\\" + processingData_folder + "\\" + sub_folder
processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.')]

# %% 資料前處理 : bandpass filter, absolute value, smoothing, trunkcut data
# 處理MVC data
tic = time.process_time()
for i in range(len(rowdata_folder_list)):
    tic = time.process_time()
    MVC_folder_path = rowdata_folder_path + "\\" + rowdata_folder_list[i] + "\\" + MVC_folder
    MVC_list = af.Read_File(MVC_folder_path, ".csv")
    fig_save_path = processing_folder_path + "\\" + rowdata_folder_list[i] + fig_save
    print("Now processing MVC data in " + rowdata_folder_list[i])
    for MVC_path in MVC_list:
        print(MVC_path)
        data = pd.read_csv(MVC_path, encoding='UTF-8')
        moving_data, rms_data, lowpass_filtered_data, bandpass_filtered_data = af.EMG_processing(data)
        data_save_path = processing_folder_path + '\\' + rowdata_folder_list[i] + "\\data\\" + MVC_folder
        # deal with filename and add extension with _ed
        filepath, tempfilename = os.path.split(MVC_path)
        filename, extension = os.path.splitext(tempfilename)
        # 畫圖
        af.plot_plot(bandpass_filtered_data, fig_save_path,
                     filename, "Bandpass_")
        # 畫 FFT analysis 的圖
        af.Fourier_plot(data,
                     (fig_save_path + "\\FFT\\MVC"),
                     filename)
        # rewrite file name
        file_name = data_save_path + '\\' + filename + end_name + '.xlsx'
        # writting data in worksheet
        if smoothing == 'lowpass':
            af.plot_plot(lowpass_filtered_data, fig_save_path,
                         filename, "lowpass_")
            pd.DataFrame(lowpass_filtered_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
        elif smoothing == 'rms':
            af.plot_plot(rms_data, fig_save_path,
                         filename, "rms_")
            pd.DataFrame(rms_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
        elif smoothing == 'moving':
            af.plot_plot(moving_data, fig_save_path,
                         filename, "moving_")
            pd.DataFrame(moving_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    toc = time.process_time()
    print("Total Time:",toc-tic)  
# 找最大值
for i in range(len(rowdata_folder_list)):
    print("To fing the maximum value of all of MVC data in: " + rowdata_folder_list[i])
    tic = time.process_time()
    af.Find_MVC_max(processing_folder_path + '\\' + rowdata_folder_list[i] + "\\data\\" + MVC_folder,
                 processing_folder_path + '\\' + rowdata_folder_list[i])
    toc = time.process_time()
    print("Total Time:",toc-tic)
toc = time.process_time()
print("MVC Data Total Time Spent: ",toc-tic)
# %% 將各肌肉不同 MVC 試驗的圖畫在一起
## 因為得到的圖都已經是綠波且平滑處理過後，所以需另外處理
# tic = time.process_time()
# for i in range(len(processing_folder_list)):
#     tic = time.process_time()
#     MVC_folder_path = processing_folder_path + "\\" + processing_folder_list[i] + "\\data\\" + MVC_folder
#     MVC_list = af.Read_File(MVC_folder_path, ".xlsx")
#     fig_save_path = processing_folder_path + "\\" + processing_folder_list[i] + fig_save
#     print("Now processing MVC data in " + rowdata_folder_list[i])
#     for MVC_path in MVC_list:
#         raw_data = pd.read_excel()
        
        


# %% 找放箭時間
tic = time.process_time()
for i in range(len(rowdata_folder_list)):
    tic = time.process_time()
    motion_folder_path = rowdata_folder_path + "\\" + rowdata_folder_list[i] + "\\" + motion_folder
    af.find_release_time(rowdata_folder_path + '\\' + rowdata_folder_list[i] + "\\" + motion_folder,
                         processing_folder_path + '\\' + rowdata_folder_list[i])
    print("圖片存檔路徑: ", processing_folder_path + '\\' + rowdata_folder_list[i])
toc = time.process_time()
print("Release Time Total Time Spent: ",toc-tic)
# %% 資料前處理 : bandpass filter, absolute value, smoothing, trunkcut data
''' 
處理shooting data
# ----------------------------------------------------------------------------
# 1. 取出所有Raw資料夾
# 2. 獲得Raw folder路徑下的“射箭”資料夾，並讀取所有.cvs file
# 3. 讀取processing folder路徑下的ReleaseTiming，並讀取檔案
# 4. 依序前處理“射箭”資料夾下的檔案
# 4.1 bandpass filting
# 4.2 trunkcut data by release time
# 4.3 依切割檔案計算moving average
# 4.4 輸出moving average to excel file
------------------------------------------------------------------------------
'''
tic = time.process_time()
for i in range(len(rowdata_folder_list)):
    # print(rowdata_folder_list[i])
    # 預處理shooting data
    # for mac version replace "\\" by '/'
    Shooting_path = rowdata_folder_path + '\\' + rowdata_folder_list[i] + "\\" + motion_folder
    Shooting_list = af.Read_File(Shooting_path, '.csv')
    # 儲存照片路徑
    fig_save_path = processing_folder_path + "\\" + rowdata_folder_list[i] + fig_save
    # 讀取staging file
    staging_data = pd.read_excel(processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + rowdata_folder_list[i] + '_ReleaseTiming.xlsx')
    # 讀取all MVC data
    MVC_value = pd.read_excel(processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + rowdata_folder_list[i] + '_all_MVC.xlsx')
    MVC_value = MVC_value.iloc[-1, 2:]
    
    for ii in range(len(Shooting_list)):
        for iii in range(len(staging_data['FileName'])):
            # for mac version replace "\\" by '/'
            if Shooting_list[ii].split('\\')[-1] == staging_data['FileName'][iii].split('\\')[-1]:
                # 印出漂亮的說明
                x = PrettyTable()
                x.field_names = ["平滑方法", "folder", "shooting_file", "Staging_file"]
                x.add_row([smoothing, rowdata_folder_list[i], Shooting_list[ii].split('\\')[-1], staging_data['FileName'][iii].split('\\')[-1]])
                print(x)

                # 寫資料進excel
                filepath, tempfilename = os.path.split(Shooting_list[ii])
                filename, extension = os.path.splitext(tempfilename)
                # rewrite file name
                file_name = processing_folder_path + '\\' + rowdata_folder_list[i] + "\\data\\" + motion_folder + '\\' + filename + "_iMVC" + end_name + ".xlsx"
                data = pd.read_csv(Shooting_list[ii], encoding='UTF-8')
                if staging_data['Time Frame(降1000Hz)'][iii] != "Nan":
                    # pre-processing data
                    moving_data, rms_data, lowpass_filtered_data, bandpass_filtered_data = af.EMG_processing(data)
                    # 畫 FFT analysis 的圖
                    af.Fourier_plot(data,
                                 (fig_save_path + "\\FFT\\motion"),
                                 filename)
                    # get release time
                    release_idx = int(staging_data['Time Frame(降1000Hz)'][iii])
                    release_samp_freq = 1/(lowpass_filtered_data.iloc[1, 0] - lowpass_filtered_data.iloc[0, 0])
                    # 去做條件判斷要輸出何種資料
                    if smoothing == 'lowpass':
                        ## 擷取 EMG data
                        # 計算MVC值
                        emg_iMVC = pd.DataFrame(np.zeros([release[0]+release[1], np.shape(lowpass_filtered_data)[1]]),
                                                     columns=lowpass_filtered_data.columns)
                        emg_iMVC.iloc[:, 0] = lowpass_filtered_data.iloc[release_idx-release[0]:release_idx+release[1], 0].values
                        emg_iMVC.iloc[:, 1:] = np.divide(abs(lowpass_filtered_data.iloc[release_idx-release[0]:release_idx+release[1], 1:].values),
                                                              MVC_value.values)*100
                    elif smoothing == 'rms':
                        start_idx = np.abs(rms_data.iloc[:, 0] - (release_idx - release[0])/release_samp_freq).argmin()
                        end_idx = int(start_idx + (release[0] + release[1])/down_freq *  np.ceil(1 / (rms_data.iloc[1, 0] - rms_data.iloc[0, 0])))
                        print(end_idx - start_idx)
                        rms_data = rms_data.iloc[start_idx:end_idx, :]
                        emg_iMVC = pd.DataFrame(np.zeros(np.shape(rms_data)),
                                                columns=rms_data.columns)
                        emg_iMVC.iloc[:, 0] = rms_data.values
                        emg_iMVC.iloc[:, 1:] = np.divide(abs(rms_data.iloc[:, 1:].values),
                                                         MVC_value.values)*100
                    elif smoothing == 'moving':
                        start_idx = np.abs(moving_data.iloc[:, 0] - (release_idx - release[0])/release_samp_freq).argmin()
                        end_idx = int(start_idx + (release[0] + release[1])/down_freq *  np.ceil(1 / (moving_data.iloc[1, 0] - moving_data.iloc[0, 0])))
                        moving_data = moving_data.iloc[start_idx:end_idx, :]
                        emg_iMVC = pd.DataFrame(np.zeros(np.shape(moving_data)),
                                                columns=moving_data.columns)
                        emg_iMVC.iloc[:, 0] = moving_data.values
                        emg_iMVC.iloc[:, 1:] = np.divide(abs(moving_data.iloc[:, 1:].values),
                                                         MVC_value.values)*100
                    # writting data in worksheet
                    pd.DataFrame(emg_iMVC).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
                else: # 通常為 fatigue file
                    # pre-processing data
                    moving_data, rms_data, lowpass_filtered_data, bandpass_filtered_data = af.EMG_processing(data)
                    # 計算MVC值
                    shooting_iMVC = pd.DataFrame(np.zeros([np.shape(lowpass_filtered_data)[0], np.shape(lowpass_filtered_data)[1]]),
                                                 columns=lowpass_filtered_data.columns)
                    shooting_iMVC.iloc[:, 0] = lowpass_filtered_data.iloc[:, 0]
                    shooting_iMVC.iloc[:, 1:] = np.divide(lowpass_filtered_data.iloc[:, 1:], MVC_value.values)*100
                    pd.DataFrame(lowpass_filtered_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
                    af.median_frquency(data, duration, fig_save_path, filename)

toc = time.process_time()
print("Motion Data Total Time Spent: ",toc-tic)
# %% 若已有分期檔
# 資料前處理 : bandpass filter, absolute value, smoothing, trunkcut data
''' 
處理shooting data
# ----------------------------------------------------------------------------
# 1. 取出所有Raw資料夾
# 2. 獲得Raw folder路徑下的“射箭”資料夾，並讀取所有.cvs file
# 3. 讀取processing folder路徑下的ReleaseTiming，並讀取檔案
# 4. 依序前處理“射箭”資料夾下的檔案
# 4.1 bandpass filting
# 4.2 trunkcut data by release time
# 4.3 依切割檔案計算moving average
# 4.4 輸出moving average to excel file
------------------------------------------------------------------------------
'''
tic = time.process_time()
# 讀取分期檔
staging_file = pd.read_excel(staging_file_path)
# 處理資料
for i in range(len(rowdata_folder_list)):
    # print(rowdata_folder_list[i])
    # 預處理shooting data
    # for mac version replace "\\" by '/'
    Shooting_path = rowdata_folder_path + '\\' + rowdata_folder_list[i] + "\\" + motion_folder
    Shooting_list = af.Read_File(Shooting_path, '.csv')
    # 儲存照片路徑
    fig_save_path = processing_folder_path + "\\" + rowdata_folder_list[i] + fig_save
    # 讀取all MVC data
    MVC_value = pd.read_excel(processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + rowdata_folder_list[i] + '_all_MVC.xlsx')
    MVC_value = MVC_value.iloc[-1, 2:]
    
    for ii in range(len(Shooting_list)):
        for iii in range(len(staging_file['EMG_FileName'])):
            # for mac version replace "\\" by '/'
            if Shooting_list[ii].split('\\')[-1] == staging_file['EMG_FileName'][iii].split('\\')[-1]:
                # 印出漂亮的說明
                x = PrettyTable()
                x.field_names = ["平滑方法", "folder", "shooting_file", "Staging_file"]
                x.add_row([smoothing, rowdata_folder_list[i], Shooting_list[ii].split('\\')[-1], staging_file['EMG_FileName'][iii].split('\\')[-1]])
                print(x)

                # 寫資料進excel
                filepath, tempfilename = os.path.split(Shooting_list[ii])
                filename, extension = os.path.splitext(tempfilename)
                # rewrite file name
                file_name = processing_folder_path + '\\' + rowdata_folder_list[i] + "\\data\\" + motion_folder + '\\' + filename + "_iMVC" + end_name + ".xlsx"
                data = pd.read_csv(Shooting_list[ii], encoding='UTF-8')
                if staging_file['Shoot (frames)'][iii] != "Nan":
                    # pre-processing data
                    moving_data, rms_data, lowpass_filtered_data, bandpass_filtered_data = af.EMG_processing(data)
                    # 畫 FFT analysis 的圖
                    af.Fourier_plot(data,
                                 (fig_save_path + "\\FFT\\motion"),
                                 filename)
                    # get release time
                    release_idx = int((staging_file['Shoot (frames)'][iii] - staging_file['Trigger Strat (frames)'][iii])*5)
                    release_samp_freq = 1/(lowpass_filtered_data.iloc[1, 0] - lowpass_filtered_data.iloc[0, 0])
                    # 去做條件判斷要輸出何種資料
                    if smoothing == 'lowpass':
                        ## 擷取 EMG data
                        # 計算MVC值
                        emg_iMVC = pd.DataFrame(np.zeros([release[0]+release[1], np.shape(lowpass_filtered_data)[1]]),
                                                     columns=lowpass_filtered_data.columns)
                        emg_iMVC.iloc[:, 0] = lowpass_filtered_data.iloc[release_idx-release[0]:release_idx+release[1], 0].values
                        emg_iMVC.iloc[:, 1:] = np.divide(abs(lowpass_filtered_data.iloc[release_idx-release[0]:release_idx+release[1], 1:].values),
                                                              MVC_value.values)*100
                    elif smoothing == 'rms':
                        start_idx = np.abs(rms_data.iloc[:, 0] - (release_idx - release[0])/release_samp_freq).argmin()
                        end_idx = int(start_idx + (release[0] + release[1])/down_freq *  np.ceil(1 / (rms_data.iloc[1, 0] - rms_data.iloc[0, 0])))
                        print(end_idx - start_idx)
                        rms_data = rms_data.iloc[start_idx:end_idx, :]
                        emg_iMVC = pd.DataFrame(np.zeros(np.shape(rms_data)),
                                                columns=rms_data.columns)
                        emg_iMVC.iloc[:, 0] = rms_data.values
                        emg_iMVC.iloc[:, 1:] = np.divide(abs(rms_data.iloc[:, 1:].values),
                                                         MVC_value.values)*100
                    elif smoothing == 'moving':
                        start_idx = np.abs(moving_data.iloc[:, 0] - (release_idx - release[0])/release_samp_freq).argmin()
                        end_idx = int(start_idx + (release[0] + release[1])/down_freq *  np.ceil(1 / (moving_data.iloc[1, 0] - moving_data.iloc[0, 0])))
                        moving_data = moving_data.iloc[start_idx:end_idx, :]
                        emg_iMVC = pd.DataFrame(np.zeros(np.shape(moving_data)),
                                                columns=moving_data.columns)
                        emg_iMVC.iloc[:, 0] = moving_data.values
                        emg_iMVC.iloc[:, 1:] = np.divide(abs(moving_data.iloc[:, 1:].values),
                                                         MVC_value.values)*100
                    # writting data in worksheet
                    pd.DataFrame(emg_iMVC).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
                else: # 通常為 fatigue file
                    # pre-processing data
                    moving_data, rms_data, lowpass_filtered_data, bandpass_filtered_data = af.EMG_processing(data)
                    # 計算MVC值
                    shooting_iMVC = pd.DataFrame(np.zeros([np.shape(lowpass_filtered_data)[0], np.shape(lowpass_filtered_data)[1]]),
                                                 columns=lowpass_filtered_data.columns)
                    shooting_iMVC.iloc[:, 0] = lowpass_filtered_data.iloc[:, 0]
                    shooting_iMVC.iloc[:, 1:] = np.divide(lowpass_filtered_data.iloc[:, 1:], MVC_value.values)*100
                    pd.DataFrame(lowpass_filtered_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
                    af.median_frquency(data, duration, fig_save_path, filename)

toc = time.process_time()
print("Motion Data Total Time Spent: ",toc-tic)
# %% 畫 Mean std cloud 圖                    
'''
1. 列出檔案夾路徑，並設定讀取motion資料夾
2. 給 mean_std_cloud function 繪圖
'''
tic = time.process_time()
for i in range(len(processing_folder_list)):
    motion_folder_path = processing_folder_path + "\\" + processing_folder_list[i] + "\\data\\motion"
    save_path =  processing_folder_path + "\\" + processing_folder_list[i] + "\\figure\\std_cloud"
    print("圖片存檔路徑: ", save_path)
    af.mean_std_cloud(motion_folder_path, save_path, processing_folder_list[i], smoothing, release)
toc = time.process_time()
print("Total Time Spent: ",toc-tic)
        
# %% wavelet analysis

shot_list = []
for i in range(len(rowdata_folder_list)):
    # print(rowdata_folder_list[i])
    # 預處理shooting data
    # for mac version replace "\\" by '/'
    Shooting_path = rowdata_folder_path + '\\' + rowdata_folder_list[i] + "\\" + motion_folder
    Shooting_list = af.Read_File(Shooting_path, '.csv')
    shot_list = shot_list + Shooting_list









