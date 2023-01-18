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
 
    data = pd.read_csv(cvs_file_list, encoding='UTF-8')
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
    abs_data = abs(bandpass_filtered_data)     
    # -------Data smoothing. Compute Moving mean
    # without overlap
    # 更改window length
    # window width = window length(second)//time period(second)
    window_width_moving = int(0.1/(1/np.floor(Fs)))
    moving_data = np.zeros([int(np.shape(abs_data)[0] / window_width_moving),
                            np.shape(abs_data)[1]])
    for i in range(np.shape(moving_data)[1]):
        for ii in range(np.shape(moving_data)[0]):
            moving_data[int(ii), i] = (np.sum(abs_data[(ii*window_width_moving+1):(window_width_moving)*(ii+1), i]) 
                                                      /window_width_moving)
        
        # for ii in range(np.shape(moving_data)[0]):
        #     print((ii*window_width_moving+1),(window_width_moving)*(ii+1))
        #     moving_data[int(ii), i] = (np.sum(abs_data[(ii*window_width_moving+1):(window_width_moving)*(ii+1), i]) 
        #                                               /window_width_moving)
    # -------Data smoothing. Compute RMS
    # The user should change window length and overlap length that suit for your experiment design
    # window width = window length(second)//time period(second)
    window_width_rms = int(0.05/(1/np.floor(Fs))) #width of the window for computing RMS
    overlap_len = 0.5 # 百分比
    rms_data = np.zeros([int((np.shape(bandpass_filtered)[0] - window_width_rms)/  ((1-overlap_len)*window_width_rms)) + 1,
                            np.shape(abs_data)[1]])
    for i in range(np.shape(rms_data)[1]):
        for ii in range(np.shape(rms_data)[0]):
            data_location = int(ii*(1-overlap_len)*window_width_rms)
            print(data_location, data_location+window_width_rms)
            rms_data[int(ii), i] = np.sqrt(np.sum((abs_data[data_location:data_location+window_width_rms, i])**2)
                                      /window_width_rms)
            
    # rms_data = np.zeros([int((np.shape(bandpass_filtered)[0] - window_width_rms)/  ((1-overlap_len)*window_width_rms)) + 1])

    # for ii in range(np.shape(rms_data)[0]):
    #     data_location = int(ii*(1-overlap_len)*window_width_rms)
    #     print(data_location, data_location+window_width_rms)
    #     rms_data[int(ii)] = np.sqrt(np.sum((abs_data[data_location:data_location+window_width_rms])**2)
    #                               /window_width_rms)
            # data_location = int(ii*(1-overlap_len)*window_width_rms)
            # print(data_location)
            # rms_data[int(ii), i] = (np.sum(bandpass_filtered_data[data_location:data_location+window_width_rms, i])
            #                    /window_width_rms)
    # 定義資料型態與欄位名稱
    moving_data = pd.DataFrame(moving_data, columns=EMG_data.columns)
    rms_data = pd.DataFrame(rms_data, columns=EMG_data.columns)
    # 定義moving average的時間
    moving_time_index = np.linspace(0, np.shape(data_time)[0]-1, np.shape(moving_data)[0])
    moving_time_index = moving_time_index.astype(int)
    time_1 = pd.DataFrame(data.iloc[moving_time_index, 0], index = None).reset_index(drop=True)
    moving_data = pd.concat([time_1, moving_data], axis = 1, ignore_index=False)
    # 定義RMS DATA的時間.
    rms_time_index = np.linspace(0, np.shape(data_time)[0]-1, np.shape(rms_data)[0])
    rms_time_index = rms_time_index.astype(int)
    time_2 = pd.DataFrame(data.iloc[rms_time_index, 0], index = None).reset_index(drop=True)
    rms_data = pd.concat([time_2, pd.DataFrame(rms_data)], axis = 1, ignore_index=False)
    # ------linear envelop analysis-----------                          
    # ------lowpass filter parameter that the user must modify for your experiment        
    lowpass_sos = signal.butter(2, 6, btype='low', fs=Fs, output='sos')        
    lowpass_filtered_data = np.zeros(np.shape(abs_data))
    for i in range(np.shape(abs_data)[1]):
        lowpass_filtered = signal.sosfiltfilt(lowpass_sos, abs_data[:,i])
        lowpass_filtered_data[:, i] = lowpass_filtered
    # add columns name to data frame
    bandpass_filtered_data = pd.DataFrame(bandpass_filtered_data, columns=EMG_data.columns)

    lowpass_filtered_data = pd.DataFrame(lowpass_filtered_data, columns=EMG_data.columns)
    # insert time data in the DataFrame
    lowpass_filtered_data.insert(0, 'time', data_time)

    return moving_data, rms_data, lowpass_filtered_data

# EMG onset detect


EMG_data,bandpass_filtered_data, moving_data, rms_data, lowpass_filtered_data = EMG_processing(r"D:\NTSU\ChenDissertationDataProcessing\EMG_Data\RawData\S01\射箭\機械式\S01_Plot_and_Store_Rep_3.4 機械1.csv")
data_time = pd.read_csv(r"D:\NTSU\ChenDissertationDataProcessing\EMG_Data\RawData\S01\射箭\機械式\S01_Plot_and_Store_Rep_3.4 機械1.csv", encoding='UTF-8').iloc[:,0]
# loading_time = detect_onset(lowpass_filtered_data.iloc[:, 1], np.mean(lowpass_filtered_data.iloc[0:50, 1]), n_above=300, n_below=300, show=True)
# ta, tai, taf, amp = detect_cusum(lowpass_filtered_data.iloc[:, 1], np.mean(lowpass_filtered_data.iloc[0:50, 1]), .05, True, True)

# 畫圖

fig, axs = plt.subplots(5, dpi=300)
axs[0].plot(data_time, EMG_data.iloc[:, 0])
axs[0].set_ylabel("Raw Data")

axs[1].plot(data_time, bandpass_filtered_data.iloc[:, 0])
axs[1].set_ylabel("Band PAss")

axs[2].plot(moving_data.iloc[:, 0], moving_data.iloc[:, 1])
axs[2].set_ylabel("moving_data")

axs[3].plot(rms_data.iloc[:, 0], rms_data.iloc[:, 1])
axs[3].set_ylabel("rms_data")
axs[3].set_xlabel("Time (second)")

axs[4].plot(data_time, lowpass_filtered_data.iloc[:, 1])
axs[4].set_ylabel("lowpass_filtered_data")
axs[4].set_xlabel("Time (second)")
fig.legend()
plt.subplots_adjust(
                    left=0.125,
                    bottom=-0.51,
                    top=0.9,
                    right=1.3,
                    wspace=0.2,
                    hspace=0.2)
fig.suptitle('Data Processing in EMG', fontsize = 12)
plt.legend()
plt.show()
