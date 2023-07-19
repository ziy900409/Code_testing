# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 08:46:27 2023

@author: Hsin.YH.Yang
"""
# %% import library
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"C:\Users\Public\BenQ\myPyCode\NPL")
import Dart_Function as df
import pandas as pd
import numpy as np
import os
import time
import gc
from IPython import get_ipython;   
# get_ipython().magic('reset -sf')
from prettytable import PrettyTable
import matplotlib.pyplot as plt

# %% 參數設定
# 設定條件
save_path = r"E:\python\Lin"
# 找資料夾
folder_path = r"E:\python\Lin\\"
# 設定資料夾
RawData_folder = "\\Raw_Data"
processingData_folder = "\\Processing_Data"
# 動作資料夾名稱
motion_folder = "motion"
# MVC資料夾名稱
MVC_folder = "MVC"
# ANC 資料夾名稱
anc_folder = "動作"
# figure 資料夾名稱
fig_save = "figure"
# 檔名修改
end_name = "_ed"
# 平滑處理方式 ex: lowpass, rms, moving
smoothing = 'lowpass'
# 分期點設定
staging_start = "加速起點"
staging_end = "釋放鏢"
# %% 路徑設置

rowdata_folder_path = folder_path + RawData_folder + "\\" 
# 去除有“.“開頭的檔案
rowdata_folder_list  = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.')]
processing_folder_path = folder_path + "\\" + processingData_folder + "\\" 
processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.')]

# %% 找MVC最大值
# 處理MVC data
tic = time.process_time()

for iii in range(len(rowdata_folder_list)):
    print(rowdata_folder_list[iii])
    MVC_folder_path = folder_path + RawData_folder + "\\" + rowdata_folder_list[iii] + "\\EMG\\" + MVC_folder
    MVC_list = df.Read_File(MVC_folder_path, ".csv")
    fig_save_path = processing_folder_path + "\\" + rowdata_folder_list[iii] + "\\" + fig_save + "\\MVC"
    print("Now processing MVC data in " + rowdata_folder_list[iii])
    for MVC_path in MVC_list:
        print(MVC_path)
        data = pd.read_csv(MVC_path,
                           encoding='UTF-8')
        moving_data, rms_data, lowpass_filtered_data, bandpass_filtered_data = df.EMG_processing(data)
        data_save_path = processing_folder_path + '\\' + rowdata_folder_list[iii] + "\\" + MVC_folder
        # deal with filename and add extension with _ed
        filepath, tempfilename = os.path.split(MVC_path)
        filename, extension = os.path.splitext(tempfilename)
        # 畫圖
        df.plot_plot(bandpass_filtered_data, fig_save_path,
                     filename, "Bandpass_")
        df.plot_plot(lowpass_filtered_data, fig_save_path,
                     filename, "lowpass_")
        df.plot_plot(rms_data, fig_save_path,
                     filename, "rms_")
        df.plot_plot(moving_data, fig_save_path,
                     filename, "moving_")
        # 畫 FFT analysis 的圖
        df.Fourier_plot(bandpass_filtered_data,
                     (fig_save_path + "\\FFT"),
                     filename)
        # rewrite file name
        file_name = data_save_path + '\\' + filename + end_name + '.xlsx'
        # writting data in worksheet
        if smoothing == 'lowpass':
            pd.DataFrame(lowpass_filtered_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
        elif smoothing == 'rms':
            pd.DataFrame(rms_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
        elif smoothing == 'moving':
            pd.DataFrame(moving_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    toc = time.process_time()
    print("Total Time:",toc-tic)  

# 找最大值
for i in range(len(rowdata_folder_list)):
    print("To fing the maximum value of all of MVC data in: " + rowdata_folder_list[i])
    tic = time.process_time()
    df.Find_MVC_max(processing_folder_path + '\\' + rowdata_folder_list[i] + "\\" + MVC_folder,
                 processing_folder_path + '\\' + rowdata_folder_list[i])
    toc = time.process_time()
    print("Total Time:",toc-tic)
toc = time.process_time()
print("Total Time Spent: ",toc-tic)

# %% 處理 motion
## 讀分期檔



for iii in range(len(rowdata_folder_list)):
    print(rowdata_folder_list[iii])
    #  先處理 .ANC file
    anc_folder_path = folder_path + RawData_folder + "\\" + rowdata_folder_list[iii] + "\\" + anc_folder
    anc_file_list = df.Read_File(anc_folder_path, ".anc")
    # 讀取all MVC data
    MVC_value = pd.read_excel(processing_folder_path + '\\' + rowdata_folder_list[iii] + '\\' + rowdata_folder_list[iii] + '_all_MVC.xlsx')
    MVC_value = MVC_value.iloc[-1, 2:]
    # 讀 motion 檔案
    Staging_file = pd.read_excel(r"E:/python/Dart/DartsStagingFile.xlsx",
                                 skiprows=1,
                                 sheet_name=rowdata_folder_list[iii])
    for i in range(len(anc_file_list)):
        # 依序找檔案
        filepath, tempfilename = os.path.split(anc_file_list[i])
        list_filename, extension = os.path.splitext(tempfilename)
        print(tempfilename)
        for ii in range(len(Staging_file["筆數"])):
            # 分期檔噢.anc檔名相同
            if list_filename == Staging_file["筆數"][ii]:
                # 印出漂亮的說明
                x = PrettyTable()
                x.field_names = ["file list", "staging file", "EMG file"]
                x.add_row([list_filename, Staging_file["筆數"][ii], Staging_file["EMGFileName"][ii]])
                print(x)
                # 設定資料貯存路徑
                data_save_path = processing_folder_path + '\\' + rowdata_folder_list[iii] + "\\" + motion_folder
                fig_save_path = processing_folder_path + "\\" + rowdata_folder_list[iii] + "\\" + fig_save + "\\motion"
                # 判讀.anc 時間
                anc_data = pd.read_csv(anc_file_list[i],
                                   encoding='UTF-8',
                                   skiprows=8,
                                   delim_whitespace=True)
                # truncate first two rows
                anc_data = anc_data.truncate(before=2).astype(float)
                # reset index and drop orgin index
                anc_data = anc_data.reset_index(drop=True)
                # find signal of trigger
                idx = anc_data[anc_data['emg'] < -20].index[0]
                # 畫 .ANC 的圖
                plt.figure()
                plt.plot(anc_data['Name'], anc_data['emg'])
                plt.title(list_filename, fontsize = 12)
                # 設定科學符號 : 小數點後幾位數
                plt.ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                plt.plot(anc_data['Name'][idx], anc_data['emg'][idx], marker = "x", markersize=10)
                plt.annotate(anc_data['Name'][idx], xy = (0, 0), fontsize = 16, color='b')
                plt.savefig(str(fig_save_path + '\\' + list_filename + "_ReleaseTiming.jpg"),
                            dpi=100)
                # caculate the sampling rate
                sampling_rate = 1 / np.mean(anc_data['Name'][1:11].values - anc_data['Name'][0:10].values)
                # 計算秒數
                trigger = idx / sampling_rate
                # EMG 資料前處理
                data = pd.read_csv(folder_path + RawData_folder + "\\" + rowdata_folder_list[iii] +
                                   "\\EMG\\motion\\" + Staging_file["EMGFileName"][ii] + ".csv",
                                   encoding='UTF-8')
                moving_data, rms_data, lowpass_filtered_data, bandpass_filtered_data = df.EMG_processing(data)

                ## 擷取時間
                start_time = (Staging_file[staging_start][ii] -1) / 200 - trigger
                end_time = Staging_file[staging_end][ii] / 200 - trigger
                ## 繪圖
                df.plot_plot(bandpass_filtered_data, fig_save_path,
                             Staging_file["EMGFileName"][ii],
                             "Bandpass_", [start_time, end_time])
                df.plot_plot(lowpass_filtered_data, fig_save_path,
                             Staging_file["EMGFileName"][ii],
                             "lowpass_", [start_time, end_time])
                df.plot_plot(rms_data, fig_save_path,
                             Staging_file["EMGFileName"][ii],
                             "rms_", [start_time, end_time])
                df.plot_plot(moving_data, fig_save_path,
                             Staging_file["EMGFileName"][ii],
                             "moving_", [start_time, end_time])
                # 畫 FFT analysis 的圖
                df.Fourier_plot(data,
                                (fig_save_path + "\\FFT"),
                                Staging_file["EMGFileName"][ii])
                    # 寫入資料位置
                save_file_name = data_save_path + "\\" + Staging_file["EMGFileName"][ii] + "_ed.xlsx"
                if smoothing == 'lowpass':
                    ## 擷取 EMG data
                    lowpass_filtered_data = lowpass_filtered_data.iloc[int(start_time*1000):int(end_time*1000), :]
                    # 計算 iMVC
                    emg_iMVC = pd.DataFrame(np.zeros(np.shape(lowpass_filtered_data)),
                                            columns=lowpass_filtered_data.columns)
                    emg_iMVC.iloc[:, 0] = lowpass_filtered_data.values
                    emg_iMVC.iloc[:, 1:] = np.divide(abs(lowpass_filtered_data.iloc[:, 1:].values),
                                                          MVC_value.values)*100
                elif smoothing == 'rms':
                    start_idx = np.abs(rms_data.iloc[:, 0] - start_time).argmin()
                    end_idx = np.abs(rms_data.iloc[:, 0] - end_time).argmin()
                    rms_data = rms_data.iloc[start_idx:end_idx, :]
                    emg_iMVC = pd.DataFrame(np.zeros(np.shape(rms_data)),
                                            columns=rms_data.columns)
                    emg_iMVC.iloc[:, 0] = rms_data.values
                    emg_iMVC.iloc[:, 1:] = np.divide(abs(rms_data.iloc[:, 1:].values),
                                                     MVC_value.values)*100
                elif smoothing == 'moving':
                    start_idx = np.abs(moving_data.iloc[:, 0] - start_time).argmin()
                    end_idx = np.abs(moving_data.iloc[:, 0] - end_time).argmin()
                    moving_data = moving_data.iloc[start_idx:end_idx, :]
                    emg_iMVC = pd.DataFrame(np.zeros(np.shape(moving_data)),
                                            columns=moving_data.columns)
                    emg_iMVC.iloc[:, 0] = moving_data.values
                    emg_iMVC.iloc[:, 1:] = np.divide(abs(moving_data.iloc[:, 1:].values),
                                                     MVC_value.values)*100            
                
                # writting data in worksheet
                pd.DataFrame(emg_iMVC).to_excel(save_file_name, sheet_name='Sheet1', index=False, header=True)
                
