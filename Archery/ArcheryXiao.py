# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:19:26 2023

@author: Hsin.YH.Yang
"""
# %% import library
import os
import pandas as pd
import numpy as np
from scipy import signal, interpolate
from pandas import DataFrame
import torch.nn # upsample
# %% 設置參數位置
'''
使用順序
0.5 採樣頻率已自動計算，不需自己定義
** 由於不同代EMG sensor的採樣頻率不同，因此在初步處理時，先down sampling to 1000Hz
1. 依照個人需求更改資料處理欄位，更改位置為
    ex : EMG_data = data.iloc[:, [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]]
2. 更改bandpass filter之參數
    ex : bandpass_sos = signal.butter(2, [8/0.802, 450/0.802],  btype='bandpass', fs=Fs, output='sos')
3. 確定smoothing method並修改參數
    3.1 moving_data, rms_data
        window_width = 0.1 # 窗格長度 (單位 second)
        overlap_len = 0 # 百分比 (%)
    3.2 lowpass_filtered_data
        lowpass_sos = signal.butter(2, 6/0.802, btype='low', fs=Fs, output='sos')
4. 修改release time之時間，預設放箭前2.5秒至放箭後0.5秒

'''
# 分析欄位編號
num_columns = [1, 15, 29, 37, 45, 53, 61, 69, 77, 85]
# 帶通濾波頻率
bandpass_freq = [8/0.802, 400/0.802]
# 低通濾波頻率
lowpass_freq = 6/0.802
# downsampling frequency
down_freq = 1000
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.1 # 窗格長度 (單位 second)
overlap_len = 0 # 百分比 (%)
# %% EMG data processing
def EMG_processing(cvs_file_list, release_time=None):
    '''
    Parameters
    ----------
    cvs_file_list : str
        給予欲處理資料之路徑
    release_time : int, optional
        給定分期檔路徑
    Returns
    -------
    moving_data : pandas.DataFrame
        回給移動平均數
    rms_data : pandas.DataFrame
        回給移動均方根
    lowpass_filtered_data : pandas.DataFrame
        回給低通濾波
    
    程式邏輯：
    1. 預處理：
        1.1. 計算各sensor之採樣頻率與資料長度，最後預估downsample之資料長度，並使用最小值
        1.2 計算各sensor之採樣截止時間，並做平均
        1.3 創建資料貯存之位置： bandpass, lowpass, rms, moving mean
    2. 濾波： 先濾波，再降採樣
        2.1 依各sensor之採樣頻率分開濾波
        2.2 降採樣
    3. 插入時間軸
            
    '''
    data = pd.read_csv(r"C:/Users/Public/BenQ/python/EMG_Data/Raw_Data/Method_1/S02/S2_Preview_Rep_1.0.csv",encoding='UTF-8')
    data = pd.read_csv(cvs_file_list,encoding='UTF-8')
    # 1.  -------------前處理---------------------------
    # 1.1.-------------計算所有sensor之採樣頻率----------
    Fs = []
    # 計算各採樣頻率與計算downsample所需的位點數，並取最小的位點數
    data_len = []
    all_stop_time = []
    for col in range(len(num_columns)):
        # print(col)
        data_time = data.iloc[:,num_columns[col]-1]
        # 採樣頻率計算取前十個時間點做差值平均
        freq = int(1/np.mean(np.array(data_time[2:11])-np.array(data_time[1:10])))
        data_len.append((data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))
        Fs.append(int((len(data_time) - (data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))/freq*1000))
        # 取截止時間
        all_stop_time.append(data.iloc[(len(data_time) - (data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0)),
                                       num_columns[col]-1])
    # 1.2.-------------計算平均截止時間------------------
    mean_stop_time = np.mean(all_stop_time)

    # 1.3.-------------創建儲存EMG data的矩陣------------
    # bandpass filter used in signal
    bandpass_filtered_data = pd.DataFrame(np.zeros([min(Fs), len(num_columns)]),
                            columns=data.iloc[:, num_columns].columns)
    lowpass_filtered_data = pd.DataFrame(np.zeros([min(Fs), len(num_columns)]),
                            columns=data.iloc[:, num_columns].columns)
    # 設定moving mean的矩陣大小、欄位名稱
    window_width = int(time_of_window*np.floor(down_freq))
    moving_data = pd.DataFrame(np.zeros([int((np.shape(bandpass_filtered_data)[0] - window_width)/  ((1-overlap_len)*window_width)) + 1,
                                         np.shape(bandpass_filtered_data)[1]]),
                               columns=data.iloc[:, num_columns].columns)
    # 設定Root mean square的矩陣大小、欄位名稱
    rms_data = pd.DataFrame(np.zeros([int((np.shape(bandpass_filtered_data)[0] - window_width)/  ((1-overlap_len)*window_width)) + 1,
                                      np.shape(bandpass_filtered_data)[1]]),
                            columns=data.iloc[:, num_columns].columns)
    # 2.2 -------------分不同sensor處理各自的採樣頻率----
    for col in range(len(num_columns)):
        # 取採樣時間的前十個採樣點計算採樣頻率
        sample_freq = int(1/np.mean(np.array(data.iloc[2:11, (num_columns[col] - 1)]) 
                           - np.array(data.iloc[1:10, (num_columns[col] - 1)])))
        
        # 計算Bandpass filter
        bandpass_sos = signal.butter(2, bandpass_freq,  btype='bandpass', fs=sample_freq, output='sos')
        bandpass_filtered = signal.sosfiltfilt(bandpass_sos,
                                               data.iloc[:(len(data_time) - data_len[col]), num_columns[col]])
        
        # 取絕對值，將訊號翻正
        abs_data = abs(bandpass_filtered)
        # ------linear envelop analysis-----------                          
        # ------lowpass filter parameter that the user must modify for your experiment        
        lowpass_sos = signal.butter(2, lowpass_freq, btype='low', fs=1000, output='sos')        
        lowpass_filtered = signal.sosfiltfilt(lowpass_sos, abs_data)
        
        # 2.3.------resample data to 1000Hz-----------
        # 降取樣資料，並將資料儲存在矩陣當中
        bandpass_filtered = signal.resample(bandpass_filtered, min(Fs))
        bandpass_filtered_data.iloc[:, col] = bandpass_filtered
        abs_data = signal.resample(abs_data, min(Fs))
        lowpass_filtered = signal.resample(lowpass_filtered, min(Fs))
        lowpass_filtered_data.iloc[:, col] = lowpass_filtered
        # -------Data smoothing. Compute Moving mean
        # window width = window length(second)*sampling rate
        
        for ii in range(np.shape(moving_data)[0]):
            data_location = int(ii*(1-overlap_len)*window_width)
            # print(data_location, data_location+window_width_rms)
            moving_data.iloc[int(ii), col] = (np.sum((abs_data[data_location:data_location+window_width])**2)
                                          /window_width)
            
        # -------Data smoothing. Compute RMS
        # The user should change window length and overlap length that suit for your experiment design
        # window width = window length(second)*sampling rate
        for ii in range(np.shape(rms_data)[0]):
            data_location = int(ii*(1-overlap_len)*window_width)
            # print(data_location, data_location+window_width_rms)
            rms_data.iloc[int(ii), col] = np.sqrt(np.sum((abs_data[data_location:data_location+window_width])**2)
                                          /window_width)
                
    # 3. -------------插入時間軸-------------------
    lowpass_time_index = np.linspace(0, mean_stop_time, np.shape(lowpass_filtered_data)[0])
    lowpass_filtered_data.insert(0, 'time', lowpass_time_index)
    # 定義moving average的時間
    moving_time_index = np.linspace(0, mean_stop_time, np.shape(moving_data)[0])
    moving_time_index = moving_time_index.astype(int)
    time_1 = pd.DataFrame(data.iloc[moving_time_index, 0], index = None).reset_index(drop=True)
    moving_data = pd.concat([time_1, moving_data], axis = 1, ignore_index=False)
    # 定義RMS DATA的時間.
    rms_time_index = np.linspace(0, mean_stop_time, np.shape(rms_data)[0])
    rms_time_index = rms_time_index.astype(int)
    time_2 = pd.DataFrame(data.iloc[rms_time_index, 0], index = None).reset_index(drop=True)
    rms_data = pd.concat([time_2, pd.DataFrame(rms_data)], axis = 1, ignore_index=False)
    
    return moving_data, rms_data, lowpass_filtered_data

