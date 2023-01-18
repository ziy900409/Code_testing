# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:21:54 2022

@author: user
"""

import pandas as pd #處理資料結構用
import numpy as np #矩陣運算
import matplotlib.pyplot as plt #繪圖
from scipy import signal # 訊號處理: 濾波...等


file_path = r'E:/ChenDissertationData/EMG_Data/RawData/S01/射箭/機械式/S01_Plot_and_Store_Rep_3.4 機械1.csv'
data = pd.read_csv(file_path,
                   encoding='UTF-8')
# 選取資料位點
data_time = data.iloc[:, 0]
EMG_data = data.iloc[:, 1]
# determin the data length
# 計算資料長度
N = int(np.prod(data_time.shape))#length of the array
N2 = 2**(N.bit_length()-1) #last power of 2
# calculate the sampling rate
# 計算取樣頻率
Fs = 1/(data_time[1]-data_time[0])  #sample rate (Hz)
# convert sampling rate to period
# 計算取樣週期
T = 1/Fs;
print("# Samples length:",N)
print("# Sampling rate:",Fs)

# 帶通率波
bandpass_sos = signal.butter(2, [20/0.802, 500/0.802],  btype='bandpass', fs=Fs, output='sos')
bandpass_filtered = signal.sosfiltfilt(bandpass_sos, EMG_data)

# 取絕對值
abs_data = abs(bandpass_filtered)

# 設定低通濾波器參數
lowpass_sos = signal.butter(2, 6/0.802, btype='low', fs=Fs, output='sos')
# lowpass_filtered_1 = signal.sosfilt(lowpass_sos, abs_data_1)
lowpass_filtered_2 = signal.sosfiltfilt(lowpass_sos, abs_data)

window_width_moving = int(0.1/(1/np.floor(Fs)))
moving_data = np.zeros([int(np.shape(abs_data)[0] / window_width_moving)])

for ii in range(np.shape(moving_data)[0]):
    print((ii*window_width_moving+1),(window_width_moving)*(ii+1))
    moving_data[int(ii)] = (np.sum(abs_data[(ii*window_width_moving+1):(window_width_moving)*(ii+1)]) 
                                              /window_width_moving)



# -------Data smoothing. Compute RMS
# The user should change window length and overlap length that suit for your experiment design
# window width = window length(second)//time period(second)
window_width_rms = int(0.05/(1/np.floor(Fs))) #width of the window for computing RMS
overlap_len = 0.5 # 百分比
rms_data = np.zeros([int((np.shape(bandpass_filtered)[0] - window_width_rms)/  ((1-overlap_len)*window_width_rms)) + 1])

for ii in range(np.shape(rms_data)[0]):
    data_location = int(ii*(1-overlap_len)*window_width_rms)
    print(data_location, data_location+window_width_rms)
    rms_data[int(ii)] = np.sqrt(np.sum((abs_data[data_location:data_location+window_width_rms])**2)
                              /window_width_rms)
# 定義資料型態與欄位名稱
moving_data = pd.DataFrame(moving_data, columns=pd.DataFrame(EMG_data).columns)
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


fig, axs = plt.subplots(4, sharex = True, dpi=300)
axs[0].plot(data_time, EMG_data)
axs[0].set_ylabel("Raw Data", fontsize = 16)

axs[1].plot(data_time, bandpass_filtered)
axs[1].set_ylabel("Band Pass", fontsize = 16)

axs[2].plot(data_time, abs_data)
axs[2].set_ylabel("Rectified", fontsize = 16)

axs[3].plot(moving_data.iloc[:, 0], moving_data.iloc[:, 1])
axs[3].set_ylabel("Moving mean", fontsize = 16)
axs[3].set_xlabel("Time (second)", fontsize = 16)

fig.legend()
plt.subplots_adjust(
                    left=0.125,
                    bottom=-0.51,
                    top=0.9,
                    right=1.3,
                    wspace=0.2,
                    hspace=0.2)
fig.suptitle('Data Processing in EMG', fontsize = 20)
plt.legend()
plt.show()