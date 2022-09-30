#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 11:43:18 2022

使用力版F1Y去做擊球步態的分期
1. 先使用6 Hz lowpass filter 預處理資料
2. 使用 detecta onset 來做步態的分期
https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/DetectOnset.ipynb
3. 將結果繪圖，以人工判定結果好壞
4. 輸出檔案

@author: Hsin Yang
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import gc
from detecta import detect_onset
# 顯示輸入中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False


# read file list
# for MAC version
def Read_File(x, y, subfolder='None'):
    
    # if subfolder = True, the function will run with subfolder
    folder_path = x
    data_type = y
    csv_file_list = []
    
    if subfolder:
        file_list_1 = []
        for dirPath, dirNames, fileNames in os.walk(x):
            # file_list = os.walk(folder_name)
            file_list_1.append(dirPath)
        # need to change here [1:]
        for ii in file_list_1[1:]:
            file_list = os.listdir(ii)
            for iii in file_list:
                if os.path.splitext(iii)[1] == data_type:
                    file_list_name = ii + '//' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(x)                
        for i in folder_list:
            if os.path.splitext(i)[1] == data_type:
                file_list_name = folder_path + "//" + i
                csv_file_list.append(file_list_name)                
        
    return csv_file_list

# -------code starting------------
# -------code starting------------
# setting raw data path and read file list
RawData_folder = '/Users/hui/Documents/NTSU/ForcePlate_Lin/TennisForcePlateRawData/RawData/3m/'
RawData_list = Read_File(RawData_folder, '.anc', subfolder=False)
# setting placement for saving picture
svae_path = '/Users/hui/Documents/NTSU/ForcePlate_Lin/TennisForcePlateRawData/StagingPicture/3m/'
# excel save
excel_save = '/Users/hui/Documents/NTSU/ForcePlate_Lin/TennisForcePlateRawData/StagingPicture/123.xlsx'
# parameter setting
SampleRate = 2400
# setting data saving place
save_ForcePlate_data = pd.DataFrame({
               
                                    })

for data_path in RawData_list:
    print(data_path)
    ForcePlate_data = pd.read_csv(data_path,
                                  delimiter = '	',
                                  skiprows=8)
    # 分割檔名
    file_path_split = data_path.split('//', -1)[-1]
    file_name = file_path_split.split('.', -1)[0]
    # 檢查用
    print(data_path)
    # 只取出第二塊力版Fz的值
    Force_z = ForcePlate_data['F1Y'][2:]
    # ------linear envelop analysis-----------                          
    # ------lowpass filter parameter that the user must modify for your experiment        
    lowpass_sos = signal.butter(2, 6, btype='low', fs=SampleRate, output='sos')        
    lowpass_filtered = signal.sosfiltfilt(lowpass_sos, np.transpose(Force_z.values))
    # 取出特定時間的資料
    # 使用detect_onset定義進出力板時間
    loading_time = detect_onset(- lowpass_filtered, 50, n_above=10, n_below=0, show=True)
    cut_force_z = pd.DataFrame(Force_z.iloc[loading_time[0][0]:loading_time[0][1]])
    # time
    time = np.linspace(0, len(Force_z), len(Force_z))
    # 繪圖確認力版資料
    plt.figure(1)
    plt.plot(time, Force_z, 'b', label='原始資料')
    plt.axvline(x=loading_time[0][0], c='r', ls='--', lw=1)
    plt.axvline(x=loading_time[0][1], c='r', ls='--', lw=1)
    plt.axvline(x=loading_time[1][0], c='r', ls='--', lw=1)
    plt.axvline(x=loading_time[1][1], c='r', ls='--', lw=1)
    plt.title(file_name)
    plt.legend()
    plt.savefig(svae_path  +'//' +  file_name + '.jpg',
                dpi=300)
    add_ForcePlate_data = pd.DataFrame({
                                'file_name': [file_path_split],
                                'low_file_name': ['lowpass_' + file_name],
                                '進力版時間1': [loading_time[0][0]],
                                '出力版時間1': [loading_time[0][1]],
                                '進力版時間2': [loading_time[1][0]],
                                '出力版時間2': [loading_time[1][1]],
                                })
    # 合併資料
    save_ForcePlate_data = pd.concat([save_ForcePlate_data, add_ForcePlate_data], ignore_index=True)
pd.DataFrame(save_ForcePlate_data).to_excel(excel_save, sheet_name='Sheet1', index=False, header=True)











