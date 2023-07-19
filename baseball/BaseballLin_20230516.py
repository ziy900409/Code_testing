# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:44:25 2023

@author: Hsin.YH.Yang
"""

# %% import library
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"C:\Users\Public\BenQ\myPyCode\NPL")
# 將read_c3d function 加進現有的工作環境中
import BaseballFunction_20230516 as af
import os
import time
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import gc
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math
from decimal import Decimal

# %% 設定自己的資料路徑
# 資料路徑
data_path = r"E:\python\Lin\\"
# 設定資料夾
RawData_folder = "\\Raw_Data"
processingData_folder = "Processing_Data"
fig_save = "\\figure"
# 子資料夾名稱
sub_folder = ""
# 動作資料夾名稱
motion_folder = "motion"
# MVC資料夾名稱
MVC_folder = "MVC"
# ANC 資料夾名稱
anc_folder = "動作"
# 給定預抓分期資料的欄位名稱或標號：例如：[R EXTENSOR GROUP: ACC.Y 1] or [5]
release_staging_column = '5'
# downsampling frequency
down_freq = 1000
# 抓放箭時候前後秒數
# # example : [秒數*採樣頻率, 秒數*採樣頻率]
# release = [2*down_freq, 1*down_freq]
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.1 # 窗格長度 (單位 second)
overlap_len = 0.5 # 百分比 (%)
# 預處理資料可修改檔名，並新增標籤，如：S2_MVC_Rep_1.16 -> S2_MVC_Rep_1.16_low
end_name = "_ed"
# 平滑處理方式 ex: lowpass, rms, moving
smoothing = 'lowpass'
# # median frequency duration
# duration = 1 # unit : second

# %% 路徑設置

rowdata_folder_path = data_path + RawData_folder + "\\" + sub_folder
# 去除有“.“開頭的檔案
rowdata_folder_list  = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.')]
processing_folder_path = data_path + "\\" + processingData_folder + "\\" + sub_folder
processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.')]
# %%

a = af.Read_File(r"E:\python\Lin\Raw_Data\S18\motion",
                 ".csv")

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
gc.collect(generation=2)

# %% 處理 motion
## 讀分期檔
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

anc_time = pd.DataFrame({"file_name": [],
                         "idx": []})
for iii in range(len(rowdata_folder_list)):
    # print(rowdata_folder_list[iii])
    # #  先處理 motion file
    motion_folder_path = data_path + RawData_folder + "\\" + rowdata_folder_list[iii] + "\\" + motion_folder
    motion_file_list = af.Read_File(motion_folder_path, ".csv")
    # 讀取all MVC data
    MVC_value = pd.read_excel(processing_folder_path + '\\' + rowdata_folder_list[iii] + '\\' + rowdata_folder_list[iii] + '_all_MVC.xlsx')
    MVC_value = MVC_value.iloc[-1, 2:]
    # 讀取 .anc data list
    anc_folder_path = data_path + RawData_folder + "\\" + rowdata_folder_list[iii] + "\\" + anc_folder
    anc_file_list = af.Read_File(anc_folder_path, ".anc")
    Staging_file = pd.read_excel(r"E:/python/Lin/motion分期.xlsx", sheet_name=rowdata_folder_list[iii])
    for anc_path in anc_file_list:
        # 處理 .ANC 檔案
        # 判讀.anc 時間
        filepath, tempfilename = os.path.split(anc_path)
        list_filename, extension = os.path.splitext(tempfilename)
        anc_data = pd.read_csv(anc_path,
                               encoding='UTF-8',
                               skiprows=8,
                               delim_whitespace=True)
        # truncate first two rows
        anc_data = anc_data.truncate(before=2).astype(float)
        # reset index and drop orgin index
        anc_data = anc_data.reset_index(drop=True)
        # find signal of trigger
        idx = anc_data[(anc_data['trigger'] - np.mean(anc_data['trigger'][1:11].values)) < -100].index[0]
        anc_time.loc[len(anc_time.index)] = [list_filename, idx]
        
    # 讀 motion 檔案
    for motion_path in motion_file_list:
        filepath, tempfilename = os.path.split(motion_path)
        list_filename, extension = os.path.splitext(tempfilename)
        for emg_name in range(len(Staging_file['EMG檔案'])):
            if list_filename == Staging_file['EMG檔案'][emg_name]:
                x = PrettyTable()
                x.field_names = ["file list", "staging file", "EMG file"]
                x.add_row([list_filename, Staging_file["Subject"][emg_name], Staging_file["EMG檔案"][emg_name]])
                print(x)
                # 設定資料儲存路徑
                data_save_path = processing_folder_path + '\\' + rowdata_folder_list[iii] + "\\data\\" + motion_folder
                fig_save_path = processing_folder_path + "\\" + rowdata_folder_list[iii] + "\\" + fig_save

                # 讀取資料與資料前處理
                data = pd.read_csv(data_path + RawData_folder + "\\" + rowdata_folder_list[iii] +
                                   "\\motion\\" + Staging_file["EMG檔案"][emg_name] + ".csv",
                                   encoding='UTF-8')
                moving_data, rms_data, lowpass_filtered_data, bandpass_filtered_data = af.EMG_processing(data)                
                # 畫 bandpass filter 的圖
                save = fig_save_path + '\\' + "Bandpass_" + Staging_file["EMG檔案"][emg_name] + ".jpg"
                n = int(math.ceil((np.shape(bandpass_filtered_data)[1] - 1) /2))
                plt.figure(figsize=(2*n+1,10))
                fig, axs = plt.subplots(n, 2, figsize = (10,12), sharex='col')
                for i in range(np.shape(bandpass_filtered_data)[1]-1):
                    x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
                    # 設定子圖之參數
                    axs[x, y].plot(bandpass_filtered_data.iloc[:, 0], bandpass_filtered_data.iloc[:, i+1])
                    axs[x, y].set_title(bandpass_filtered_data.columns[i+1], fontsize=16)
                    # 設定科學符號 : 小數點後幾位數
                    axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                    axs[x, y].axvline((Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                    axs[x, y].axvline((Staging_file['foot contact'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                    axs[x, y].axvline((Staging_file['shoulder external rotation'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                    axs[x, y].axvline((Staging_file['release'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                    a_t = Decimal((Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")
                    axs[x, y].annotate(a_t,
                                       xy = (0, max(bandpass_filtered_data.iloc[:, i+1])), fontsize = 10, color='b')
                    axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                # 設定整張圖片之參數
                plt.suptitle(Staging_file["EMG檔案"][emg_name] + "Bandpass_", fontsize = 16)
                plt.tight_layout()
                fig.add_subplot(111, frameon=False)
                # hide tick and tick label of the big axes
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.grid(False)
                plt.xlabel("time (second)", fontsize = 14)
                plt.ylabel("Voltage (V)", fontsize = 14)
                plt.savefig(save, dpi=200, bbox_inches = "tight")
                plt.show()
                
                # 畫 FFT analysis 的圖
                af.Fourier_plot(data,
                                (fig_save_path + "\\FFT\\motion"),
                                Staging_file["EMG檔案"][emg_name])

                # 寫入資料位置
                save_file_name = data_save_path + "\\" + Staging_file["EMG檔案"][emg_name] + "_ed.xlsx"
                if smoothing == 'lowpass':
                    # af.plot_plot(lowpass_filtered_data, fig_save_path,
                    #              Staging_file["EMG檔案"][emg_name],
                    #              "lowpass_")
                    # 繪圖確認用
                    save = fig_save_path + '\\' + "lowpass_" + Staging_file["EMG檔案"][emg_name] + ".jpg"
                    n = int(math.ceil((np.shape(lowpass_filtered_data)[1] - 1) /2))
                    plt.figure(figsize=(2*n+1,10))
                    fig, axs = plt.subplots(n, 2, figsize = (10,12), sharex='col')
                    for i in range(np.shape(lowpass_filtered_data)[1]-1):
                        x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
                        # 設定子圖之參數
                        axs[x, y].plot(lowpass_filtered_data.iloc[:, 0], lowpass_filtered_data.iloc[:, i+1])
                        axs[x, y].set_title(lowpass_filtered_data.columns[i+1], fontsize=16)
                        # 設定科學符號 : 小數點後幾位數
                        axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                        axs[x, y].axvline((Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                        axs[x, y].axvline((Staging_file['foot contact'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                        axs[x, y].axvline((Staging_file['shoulder external rotation'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                        axs[x, y].axvline((Staging_file['release'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                        a_t = Decimal((Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")
                        axs[x, y].annotate(a_t,
                                           xy = (0, max(lowpass_filtered_data.iloc[:, i+1])), fontsize = 10, color='b')
                        axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                    # 設定整張圖片之參數
                    plt.suptitle(Staging_file["EMG檔案"][emg_name] + "_lowpass", fontsize = 16)
                    plt.tight_layout()
                    fig.add_subplot(111, frameon=False)
                    # hide tick and tick label of the big axes
                    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                    plt.grid(False)
                    plt.xlabel("time (second)", fontsize = 14)
                    plt.ylabel("Voltage (V)", fontsize = 14)
                    plt.savefig(save, dpi=200, bbox_inches = "tight")
                    plt.show()
                    # 使用秒數找尋最接近時間的引數
                    kneetop_idx = np.abs(lowpass_filtered_data.iloc[:, 0] - (Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    foot_contact_idx = np.abs(lowpass_filtered_data.iloc[:, 0] \
                                              - (Staging_file['foot contact'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    shoulder_ER_idx = np.abs(lowpass_filtered_data.iloc[:, 0] \
                                              - (Staging_file['shoulder external rotation'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    release_idx = np.abs(lowpass_filtered_data.iloc[:, 0] - (Staging_file['release'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    # 計算 iMVC
                    emg_iMVC = pd.DataFrame(np.zeros(np.shape(lowpass_filtered_data)),
                                            columns=rms_data.columns)
                    emg_iMVC.iloc[:, 0] = lowpass_filtered_data.values
                    emg_iMVC.iloc[:, 1:] = np.divide(abs(lowpass_filtered_data.iloc[:, 1:].values),
                                                          MVC_value.values)*100
                elif smoothing == 'rms':
                    af.plot_plot(rms_data, fig_save_path,
                                 Staging_file["EMG檔案"][emg_name],
                                 "rms_")
                    # 使用秒數找尋最接近時間的引數
                    kneetop_idx = np.abs(rms_data.iloc[:, 0] - (Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    foot_contact_idx = np.abs(rms_data.iloc[:, 0] \
                                              - (Staging_file['foot contact'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    shoulder_ER_idx = np.abs(rms_data.iloc[:, 0] \
                                              - (Staging_file['shoulder external rotation'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    release_idx = np.abs(rms_data.iloc[:, 0] - (Staging_file['release'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    
                    emg_iMVC = pd.DataFrame(np.zeros(np.shape(rms_data)),
                                            columns=rms_data.columns)
                    emg_iMVC.iloc[:, 0] = rms_data.values
                    emg_iMVC.iloc[:, 1:] = np.divide(abs(rms_data.iloc[:, 1:].values),
                                                     MVC_value.values)*100
                elif smoothing == 'moving':
                    af.plot_plot(moving_data, fig_save_path,
                                 Staging_file["EMG檔案"][emg_name],
                                 "moving_")
                    # 使用秒數找尋最接近時間的引數
                    kneetop_idx = np.abs(moving_data.iloc[:, 0] - (Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    foot_contact_idx = np.abs(moving_data.iloc[:, 0] \
                                              - (Staging_file['foot contact'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    shoulder_ER_idx = np.abs(moving_data.iloc[:, 0] \
                                              - (Staging_file['shoulder external rotation'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    release_idx = np.abs(moving_data.iloc[:, 0] - (Staging_file['release'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    
                    emg_iMVC.iloc[:, 0] = moving_data.values
                    emg_iMVC.iloc[:, 1:] = np.divide(abs(moving_data.iloc[:, 1:].values),
                                                     MVC_value.values)*100
                # 將資料寫進不同 EXCEL 分頁
                with pd.ExcelWriter(save_file_name) as Writer:
                    emg_iMVC.iloc[kneetop_idx:foot_contact_idx].to_excel(Writer, sheet_name="Stage1", index=False)
                    emg_iMVC.iloc[foot_contact_idx:shoulder_ER_idx].to_excel(Writer, sheet_name="Stage2", index=False)
                    emg_iMVC.iloc[shoulder_ER_idx:release_idx].to_excel(Writer, sheet_name="Stage3", index=False)

gc.collect(generation=2)                    
                    
# %% 直接用衛宣判斷的trigger作圖
# 處理 motion
## 讀分期檔
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

anc_time = pd.DataFrame({"file_name": [],
                         "idx": []})
for iii in range(len(rowdata_folder_list)):
    # print(rowdata_folder_list[iii])
    # #  先處理 motion file
    motion_folder_path = data_path + RawData_folder + "\\" + rowdata_folder_list[iii] + "\\" + motion_folder
    motion_file_list = af.Read_File(motion_folder_path, ".csv")
    # 讀取all MVC data
    MVC_value = pd.read_excel(processing_folder_path + '\\' + rowdata_folder_list[iii] + '\\' + rowdata_folder_list[iii] + '_all_MVC.xlsx')
    MVC_value = MVC_value.iloc[-1, 2:]
    # 讀取 .anc data list
    anc_folder_path = data_path + RawData_folder + "\\" + rowdata_folder_list[iii] + "\\" + anc_folder
    anc_file_list = af.Read_File(anc_folder_path, ".anc")
    Staging_file = pd.read_excel(r"E:/python/Lin/motion分期.xlsx", sheet_name=rowdata_folder_list[iii])
    for anc_path in anc_file_list:
        # 處理 .ANC 檔案
        # 判讀.anc 時間
        filepath, tempfilename = os.path.split(anc_path)
        list_filename, extension = os.path.splitext(tempfilename)
        anc_data = pd.read_csv(anc_path,
                               encoding='UTF-8',
                               skiprows=8,
                               delim_whitespace=True)
        # truncate first two rows
        anc_data = anc_data.truncate(before=2).astype(float)
        # reset index and drop orgin index
        anc_data = anc_data.reset_index(drop=True)
        # find signal of trigger
        idx = anc_data[(anc_data['trigger'] - np.mean(anc_data['trigger'][1:11].values)) < -100].index[0]
        anc_time.loc[len(anc_time.index)] = [list_filename, idx]
        
    # 讀 motion 檔案
    for motion_path in motion_file_list:
        filepath, tempfilename = os.path.split(motion_path)
        list_filename, extension = os.path.splitext(tempfilename)
        for emg_name in range(len(Staging_file['EMG檔案'])):
            if list_filename == Staging_file['EMG檔案'][emg_name]:
                x = PrettyTable()
                x.field_names = ["file list", "staging file", "EMG file"]
                x.add_row([list_filename, Staging_file["Subject"][emg_name], Staging_file["EMG檔案"][emg_name]])
                print(x)
                # 設定資料儲存路徑
                data_save_path = processing_folder_path + '\\' + rowdata_folder_list[iii] + "\\data\\" + motion_folder
                fig_save_path = processing_folder_path + "\\" + rowdata_folder_list[iii] + "\\" + fig_save

                # 讀取資料與資料前處理
                data = pd.read_csv(data_path + RawData_folder + "\\" + rowdata_folder_list[iii] +
                                   "\\motion\\" + Staging_file["EMG檔案"][emg_name] + ".csv",
                                   encoding='UTF-8')
                moving_data, rms_data, lowpass_filtered_data, bandpass_filtered_data = af.EMG_processing(data)                
                # 畫 bandpass filter 的圖
                save = fig_save_path + '\\' + "Bandpass_" + Staging_file["EMG檔案"][emg_name] + ".jpg"
                n = int(math.ceil((np.shape(bandpass_filtered_data)[1] - 1) /2))
                plt.figure(figsize=(2*n+1,10))
                fig, axs = plt.subplots(n, 2, figsize = (10,12), sharex='col')
                for i in range(np.shape(bandpass_filtered_data)[1]-1):
                    x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
                    # 設定子圖之參數
                    axs[x, y].plot(bandpass_filtered_data.iloc[:, 0], bandpass_filtered_data.iloc[:, i+1])
                    axs[x, y].set_title(bandpass_filtered_data.columns[i+1], fontsize=16)
                    # 設定科學符號 : 小數點後幾位數
                    axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                    axs[x, y].axvline((Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                    axs[x, y].axvline((Staging_file['foot contact'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                    axs[x, y].axvline((Staging_file['shoulder external rotation'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                    axs[x, y].axvline((Staging_file['release'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                    a_t = Decimal((Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")
                    axs[x, y].annotate(a_t,
                                       xy = (0, max(bandpass_filtered_data.iloc[:, i+1])), fontsize = 10, color='b')
                    axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                # 設定整張圖片之參數
                plt.suptitle(Staging_file["EMG檔案"][emg_name] + "Bandpass_", fontsize = 16)
                plt.tight_layout()
                fig.add_subplot(111, frameon=False)
                # hide tick and tick label of the big axes
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.grid(False)
                plt.xlabel("time (second)", fontsize = 14)
                plt.ylabel("Voltage (V)", fontsize = 14)
                plt.savefig(save, dpi=200, bbox_inches = "tight")
                plt.show()
                
                # 畫 FFT analysis 的圖
                af.Fourier_plot(data,
                                (fig_save_path + "\\FFT\\motion"),
                                Staging_file["EMG檔案"][emg_name])

                # 寫入資料位置
                save_file_name = data_save_path + "\\" + Staging_file["EMG檔案"][emg_name] + "_ed.xlsx"
                if smoothing == 'lowpass':
                    # af.plot_plot(lowpass_filtered_data, fig_save_path,
                    #              Staging_file["EMG檔案"][emg_name],
                    #              "lowpass_")
                    # 繪圖確認用
                    save = fig_save_path + '\\' + "lowpass_" + Staging_file["EMG檔案"][emg_name] + ".jpg"
                    n = int(math.ceil((np.shape(lowpass_filtered_data)[1] - 1) /2))
                    plt.figure(figsize=(2*n+1,10))
                    fig, axs = plt.subplots(n, 2, figsize = (10,12), sharex='col')
                    for i in range(np.shape(lowpass_filtered_data)[1]-1):
                        x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
                        # 設定子圖之參數
                        axs[x, y].plot(lowpass_filtered_data.iloc[:, 0], lowpass_filtered_data.iloc[:, i+1])
                        axs[x, y].set_title(lowpass_filtered_data.columns[i+1], fontsize=16)
                        # 設定科學符號 : 小數點後幾位數
                        axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                        axs[x, y].axvline((Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                        axs[x, y].axvline((Staging_file['foot contact'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                        axs[x, y].axvline((Staging_file['shoulder external rotation'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                        axs[x, y].axvline((Staging_file['release'][emg_name] - Staging_file['trigger'][emg_name])/240, color='r')
                        a_t = Decimal((Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")
                        axs[x, y].annotate(a_t,
                                           xy = (0, max(lowpass_filtered_data.iloc[:, i+1])), fontsize = 10, color='b')
                        axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                    # 設定整張圖片之參數
                    plt.suptitle(Staging_file["EMG檔案"][emg_name] + "_lowpass", fontsize = 16)
                    plt.tight_layout()
                    fig.add_subplot(111, frameon=False)
                    # hide tick and tick label of the big axes
                    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                    plt.grid(False)
                    plt.xlabel("time (second)", fontsize = 14)
                    plt.ylabel("Voltage (V)", fontsize = 14)
                    plt.savefig(save, dpi=200, bbox_inches = "tight")
                    plt.show()
                    # 使用秒數找尋最接近時間的引數
                    kneetop_idx = np.abs(lowpass_filtered_data.iloc[:, 0] - (Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    foot_contact_idx = np.abs(lowpass_filtered_data.iloc[:, 0] \
                                              - (Staging_file['foot contact'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    shoulder_ER_idx = np.abs(lowpass_filtered_data.iloc[:, 0] \
                                              - (Staging_file['shoulder external rotation'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    release_idx = np.abs(lowpass_filtered_data.iloc[:, 0] - (Staging_file['release'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    # 計算 iMVC
                    emg_iMVC = pd.DataFrame(np.zeros(np.shape(lowpass_filtered_data)),
                                            columns=rms_data.columns)
                    emg_iMVC.iloc[:, 0] = lowpass_filtered_data.values
                    emg_iMVC.iloc[:, 1:] = np.divide(abs(lowpass_filtered_data.iloc[:, 1:].values),
                                                          MVC_value.values)*100
                elif smoothing == 'rms':
                    af.plot_plot(rms_data, fig_save_path,
                                 Staging_file["EMG檔案"][emg_name],
                                 "rms_")
                    # 使用秒數找尋最接近時間的引數
                    kneetop_idx = np.abs(rms_data.iloc[:, 0] - (Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    foot_contact_idx = np.abs(rms_data.iloc[:, 0] \
                                              - (Staging_file['foot contact'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    shoulder_ER_idx = np.abs(rms_data.iloc[:, 0] \
                                              - (Staging_file['shoulder external rotation'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    release_idx = np.abs(rms_data.iloc[:, 0] - (Staging_file['release'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    
                    emg_iMVC = pd.DataFrame(np.zeros(np.shape(rms_data)),
                                            columns=rms_data.columns)
                    emg_iMVC.iloc[:, 0] = rms_data.values
                    emg_iMVC.iloc[:, 1:] = np.divide(abs(rms_data.iloc[:, 1:].values),
                                                     MVC_value.values)*100
                elif smoothing == 'moving':
                    af.plot_plot(moving_data, fig_save_path,
                                 Staging_file["EMG檔案"][emg_name],
                                 "moving_")
                    # 使用秒數找尋最接近時間的引數
                    kneetop_idx = np.abs(moving_data.iloc[:, 0] - (Staging_file['Kneetop'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    foot_contact_idx = np.abs(moving_data.iloc[:, 0] \
                                              - (Staging_file['foot contact'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    shoulder_ER_idx = np.abs(moving_data.iloc[:, 0] \
                                              - (Staging_file['shoulder external rotation'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    release_idx = np.abs(moving_data.iloc[:, 0] - (Staging_file['release'][emg_name] - Staging_file['trigger'][emg_name])/240).argmin() 
                    
                    emg_iMVC.iloc[:, 0] = moving_data.values
                    emg_iMVC.iloc[:, 1:] = np.divide(abs(moving_data.iloc[:, 1:].values),
                                                     MVC_value.values)*100
                # 將資料寫進不同 EXCEL 分頁
                with pd.ExcelWriter(save_file_name) as Writer:
                    emg_iMVC.iloc[kneetop_idx:foot_contact_idx].to_excel(Writer, sheet_name="Stage1", index=False)
                    emg_iMVC.iloc[foot_contact_idx:shoulder_ER_idx].to_excel(Writer, sheet_name="Stage2", index=False)
                    emg_iMVC.iloc[shoulder_ER_idx:release_idx].to_excel(Writer, sheet_name="Stage3", index=False)

gc.collect(generation=2)             

# %% 將數值標準化

musclename = ["time", "EMG_1", "EMG_2", "EMG_A4", "EMG_B4", "EMG_C4", "EMG_D4"]
for sheetname in ["Stage1", "Stage2", "Stage3"]:
    save_file_name = r"E:\python\Lin" + "\\" + sheetname + "_T2.xlsx"
    muscle1 = np.zeros((7,2000, len(rowdata_folder_list)))
    sub_num = []
    headername = []
    for iii in range(len(rowdata_folder_list)):
        motion_folder_path = processing_folder_path + '\\' + rowdata_folder_list[iii] + "\\data\\" + motion_folder
        motion_file_list = af.Read_File(motion_folder_path, ".xlsx")
        for file_name in motion_file_list:
            # print(file_name)
            if "T2" in file_name:
                data = pd.read_excel(file_name, sheet_name=sheetname)
        # 設定資料儲存位置
        # 將資料輸進矩陣
        for ii in range(np.shape(muscle1)[0]):
            muscle1[ii,:np.shape(data)[0], iii] = data.iloc[:, ii].values
        # 找出每個分期時間最長的 trail
        sub_num.append(np.count_nonzero(muscle1[0,:, iii]))
        headername.append(rowdata_folder_list[iii])
    # 將所有資料用 Cubic 內插到最長的時間區段  
    for ii in range(np.shape(muscle1)[0] -1):
        for iiii in range(len(rowdata_folder_list)):
            x = muscle1[0,:sub_num[iiii], iiii]
            y = muscle1[ii+1,:sub_num[iiii], iiii]
            # 創造內插函數
            cs = CubicSpline(x, y)    
            ny = np.linspace(x[0], x[-1], max(sub_num))
            muscle1[ii+1,:max(sub_num), iiii] = cs(ny)
    muscle1[muscle1 == 0] = np.nan          
    with pd.ExcelWriter(save_file_name) as Writer:
        for i in range(np.shape(muscle1)[0]):
            pd.DataFrame(muscle1[i, :, :], columns=headername).dropna().to_excel(Writer, sheet_name=musclename[i], index=False)
        
# %%
data1 = pd.read_excel(r"C:\Users\hsin.yh.yang\Downloads\S1_Backhand_cont_Rep_6.5.xlsx",
                     sheet_name='第一球')
data2 = pd.read_excel(r"C:\Users\hsin.yh.yang\Downloads\S1_Backhand_cont_Rep_6.5.xlsx",
                     sheet_name='第二球')
data3 = pd.read_excel(r"C:\Users\hsin.yh.yang\Downloads\S1_Backhand_cont_Rep_6.5.xlsx",
                     sheet_name='第三球')

sum_max = max([np.shape(data1)[0], np.shape(data2)[0], np.shape(data3)[0]])

pro_data1 = pd.DataFrame(np.zeros([sum_max, np.shape(data1)[1]]))
for i in range(np.shape(data1)[1]):
    x = data3.iloc[:, 0]
    y = data3.iloc[:, i]
    # 創造內插函數
    cs = CubicSpline(x, y)    
    ny = np.linspace(x[0], x.tail(1).values, sum_max)
    pro_data1.iloc[:, i] = cs(ny)
    
    
a = np.linspace(0, 0.485, 1210)
                
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    

