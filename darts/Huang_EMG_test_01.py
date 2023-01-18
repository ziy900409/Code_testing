# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:34:07 2022

@author: Hsin Yang
"""

import pandas as pd
import numpy as np
from scipy import signal
# from detecta import detect_onset
import matplotlib.pyplot as plt
# from detecta import detect_cusum

# EMG pre-processing

def EMG_processing(cvs_file_list):
 
    data = pd.read_csv(r"D:\NTSU\ChenDissertationDataProcessing\EMG_Data\RawData\S01\射箭\機械式\S01_Plot_and_Store_Rep_3.4 機械1.csv", encoding='UTF-8')
    # to define data time
    data_time = data.iloc[:,0]
    Fs = int(1/(data_time[2]-data_time[1])) # sampling frequency
    data = pd.DataFrame(data)
    # need to change column name or using column number
    EMG_data = data.iloc[:, [1, 9, 17, 25]]
    # exchange data type to float64
    EMG_data = [pd.to_numeric(EMG_data.iloc[:, i], errors = 'coerce') 
                for i in range(np.shape(EMG_data)[1])]
    EMG_data = pd.DataFrame(np.transpose(EMG_data),
                            columns=data.iloc[:, [1, 9, 17, 25]].columns)
    # bandpass filter use in signal
    # please Winter (2009) pages 36-41
    # 帶通濾波頻率
    bandpass_sos = signal.butter(2, [20/0.802, 500/0.802],  btype='bandpass', fs=Fs, output='sos')
    
    bandpass_filtered_data = np.zeros(np.shape(EMG_data))
    for i in range(np.shape(EMG_data)[1]):
        # print(i)
        # using dual filter to processing data to avoid time delay
        bandpass_filtered = signal.sosfiltfilt(bandpass_sos, EMG_data.iloc[:,i])
        bandpass_filtered_data[:, i] = bandpass_filtered 
    
    # caculate absolute value to rectifiy EMG signal
    bandpass_filtered_data = abs(bandpass_filtered_data)     
    # -------Data smoothing. Compute Moving mean
    # without overlap
    # 更改window length
    # window width = window length(second)//time period(second)
    window_width = int(0.1/(1/np.floor(Fs)))
    moving_data = np.zeros([int(np.shape(bandpass_filtered_data)[0] / window_width),
                            np.shape(bandpass_filtered_data)[1]])
    for i in range(np.shape(moving_data)[1]):
        for ii in range(np.shape(moving_data)[0]):
            moving_data[int(ii), i] = (np.sum(bandpass_filtered_data[ii*(ii+1):(ii+window_width)*(ii+1), i]) 
                                                  /window_width)
    # -------Data smoothing. Compute RMS
    # The user should change window length and overlap length that suit for your experiment design
    # window width = window length(second)//time period(second)
    window_width = int(0.05/(1/np.floor(Fs))) #width of the window for computing RMS
    overlap_len = 0.5 # 百分比
    rms_data = np.zeros([int(np.shape(bandpass_filtered_data)[0] / (window_width + (1-overlap_len)* window_width)),
                            np.shape(bandpass_filtered_data)[1]])
    for i in range(np.shape(rms_data)[1]):
        for ii in range(np.shape(rms_data)[0]):
            data_location = int(ii*(1-overlap_len)*window_width)
            rms_data[int(ii), i] = (np.sum(bandpass_filtered_data[data_location:data_location+window_width, i])
                               /window_width)
    # 定義資料型態與欄位名稱
    moving_data = pd.DataFrame(moving_data, columns=EMG_data.columns)
    # 定義moving average的時間
    moving_time_index = np.linspace(0, np.shape(data_time)[0]-1, np.shape(moving_data)[0])
    moving_time_index = moving_time_index.astype(int)
    time_1 = pd.DataFrame(data.iloc[moving_time_index, 0], index = None).reset_index(drop=True)
    moving_data = pd.concat([time_1, moving_data], axis = 1, ignore_index=False)
    # ------linear envelop analysis-----------                          
    # ------lowpass filter parameter that the user must modify for your experiment        
    lowpass_sos = signal.butter(2, 6, btype='low', fs=Fs, output='sos')        
    lowpass_filtered_data = np.zeros(np.shape(bandpass_filtered_data))
    for i in range(np.shape(bandpass_filtered_data)[1]):
        lowpass_filtered = signal.sosfiltfilt(lowpass_sos, bandpass_filtered_data[:,i])
        lowpass_filtered_data[:, i] = lowpass_filtered
    # add columns name to data frame
    bandpass_filtered_data = pd.DataFrame(bandpass_filtered_data, columns=EMG_data.columns)

    lowpass_filtered_data = pd.DataFrame(lowpass_filtered_data, columns=EMG_data.columns)
    # insert time data in the DataFrame
    lowpass_filtered_data.insert(0, 'time', data_time)
    bandpass_filtered_data.insert(0, 'time', data_time)    
    return bandpass_filtered_data, moving_data, rms_data, lowpass_filtered_data

# EMG onset detect


bandpass_filtered_data, moving_data, rms_data, lowpass_filtered_data = EMG_processing(r"D:\NTSU\ChenDissertationDataProcessing\EMG_Data\RawData\S01\射箭\機械式\S01_Plot_and_Store_Rep_3.4 機械1.csv")

# loading_time = detect_onset(lowpass_filtered_data.iloc[:, 1], np.mean(lowpass_filtered_data.iloc[0:50, 1]), n_above=300, n_below=300, show=True)
# ta, tai, taf, amp = detect_cusum(lowpass_filtered_data.iloc[:, 1], np.mean(lowpass_filtered_data.iloc[0:50, 1]), .05, True, True)

fig, ax1 = plt.subplots(1, 1, figsize=(9, 4))
ax1.plot(rms_data[:, 0],  rms_data[:, 1], 'r.-', linewidth=2, label = 'raw data')
# ax1.plot(lowpass_filtered_data.iloc[12000:15000,0], lowpass_filtered_data.iloc[12000:15000,1], 'b.-', linewidth=2, label = 'filter @ 5 Hz')
ax1.legend(frameon=False, fontsize=14)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Amplitude")
plt.show()