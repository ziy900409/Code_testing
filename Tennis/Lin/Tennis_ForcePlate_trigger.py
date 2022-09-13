# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:25:03 2022

處理大鈞EMG資料並與力版資料同步
流程：
1. 力版資料濾波 lowpass 6 Hz
2. 找進出力板瞬間
3. 找EMG trigger時間

待改善問題：
1. 迴圈占用記憶體資源過大
  1.1 可能解法： 迴圈資料註記問題，需設多重迴圈，避免單一迴圈過大，造成迴圈標記不能從記憶體中釋出
  1.2 可能解法： 使用gc.collett()做記憶體釋放

@author: Hsin Yang
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import gc
# 顯示輸入中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
from detecta import detect_onset

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
                    file_list_name = ii + '\\' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(x)                
        for i in folder_list:
            if os.path.splitext(i)[1] == data_type:
                file_list_name = folder_path + "\\" + i
                csv_file_list.append(file_list_name)                
        
    return csv_file_list


# -------------------code staring----------------------------
# -------------------code staring----------------------------
# Read data path
Parent_Folder = r"D:\NTSU\TenLab\LinData\tennis EMG+force plate\force plate"
All_Data_Path = Read_File(Parent_Folder, '.anc', subfolder=False)
# 設定資料儲存位置
save_path = r"D:\NTSU\TenLab\LinData\tennis EMG+force plate\processing\forceplate_processing"
# 圖片存檔處
picture_save_path = r"D:\NTSU\TenLab\LinData\tennis EMG+force plate\processing\forceplate_processing\picture"
# 設定參數
SampleRate = 2400 # 力版採樣頻率
# 創建資料儲存位置
save_ForcePlate_data = pd.DataFrame({
               
                                    })
# 迴圈執行所有檔案
for Data_path in All_Data_Path:
    ForcePlate_data = pd.read_csv(Data_path,
                              delimiter = '	',
                              skiprows=8)
    # 分割檔名
    file_path_split = Data_path.split('\\', -1)[-1]
    file_name = file_path_split.split('.', -1)[0]
    # 檢查用
    print(Data_path)
    # 只取出第二塊力版Fz的值
    Force_z = ForcePlate_data['F1Z'][2:]
    # trigger data
    Trigger_data = ForcePlate_data['trigger'][2:]
    # 找進出力版的時間點    
    # # 找依序數來第一個大於200的值
    # forward_non_zero = (Force_z.values > 200).argmax(axis=0)
    # # 找倒敘數來第一個大於200的值
    # res_data = Force_z.iloc[::-1] # 先反轉矩陣，再用argmax，返回第一個boolean最大值
    # backward_non_zero = (res_data.values > 200).argmax(axis=0)
    # 找出trigger時間，使用平均訊號的0.8來找
    trigger_time = (Trigger_data.values < np.mean(Trigger_data)*0.7).argmax(axis=0)

    # ------linear envelop analysis-----------                          
    # ------lowpass filter parameter that the user must modify for your experiment        
    lowpass_sos = signal.butter(2, 6, btype='low', fs=SampleRate, output='sos')        
    lowpass_filtered = signal.sosfiltfilt(lowpass_sos, np.transpose(Force_z.values))
    # 取出特定時間的資料
    # 使用detect_onset定義進出力板時間
    loading_time = detect_onset(lowpass_filtered, 10, n_above=300, n_below=300, show=True)
    cut_force_z = pd.DataFrame(Force_z.iloc[loading_time[0][0]:loading_time[0][1]])
    # time
    time = np.linspace(0, len(Force_z), len(Force_z))
    # 繪圖確認力版資料
    plt.figure(1)
    plt.plot(time, Force_z, 'b', label='原始資料')
    plt.axvline(x=loading_time[0][0], c='r', ls='--', lw=1)
    plt.axvline(x=loading_time[0][1], c='r', ls='--', lw=1)
    plt.title(Data_path)
    plt.legend()
    # plt.savefig(picture_save_path +'//' +  file_name + '.jpg',
    #             dpi=300)
    plt.figure(2)
    plt.plot(time, Trigger_data, 'b', label='原始資料')
    plt.axvline(x=trigger_time, c='r', ls='--', lw=1)
    plt.title(Data_path)
    plt.legend()
    plt.savefig(picture_save_path +'//' +  file_name + '_trigger.jpg', dpi=300)
    plt.show()
    # 將資料寫到excel
    pd.DataFrame(lowpass_filtered).to_excel(save_path + '//lowpass_' + file_name + '.xlsx',
                                            sheet_name='Sheet1', index=False, header=True)
    # 計算資料
    # 最大值
    max_force_z = np.max(lowpass_filtered)
    # 最小值
    min_force_z = np.min(lowpass_filtered)
    # 平均
    mean_force_z = np.mean(lowpass_filtered)
    # 新增資料
    add_ForcePlate_data = pd.DataFrame({
                                    'file_name': [file_path_split],
                                    'low_file_name': ['lowpass_' + file_name],
                                    '進力版時間': [loading_time[0][0]],
                                    '出力版時間': [loading_time[0][1]],
                                    'trigger時間': [trigger_time],
                                    '最大值max': [max_force_z],
                                    '最小值min': [min_force_z],
                                    '平均值mean': [mean_force_z]
                                                        })
    # 合併資料
    save_ForcePlate_data = pd.concat([save_ForcePlate_data, add_ForcePlate_data], ignore_index=True)

gc.collect()
