# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:55:45 2023

@author: drink
"""

from scipy.signal import butter,filtfilt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

example_data = pd.read_csv(r"D:\NTSU\NTSUCourse\BiomechanicAnalysis\123123\123123.trc", delimiter='\t' ,skiprows=3, encoding='UTF-8')

c7 = example_data.iloc[1:, 2:5].astype(float)
a1 = np.array(c7)
# Filter requirements.
T = 5.0         # Sample Period
Fs = 360.0       # sample rate, Hz
cutoff = 10      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * Fs  # Nyquist Frequency
order = 4       # sin wave can be approx represented as quadratic
n = int(T * Fs) # total number of samples

lowpass_sos = signal.butter(2, 6, btype='low', fs=Fs, output='sos') 
lowpass_filtered_data = np.zeros(np.shape(c7))
for i in range(np.shape(c7)[1]):
    lowpass_filtered = signal.sosfiltfilt(lowpass_sos, c7.iloc[:,i])
    lowpass_filtered_data[:, i] = lowpass_filtered
    
# Filter the data, and plot both the original and filtered signals.
# y = butter_lowpass_filter(a1, cutoff, fs, order)


Fs = 360

lowpass_sos = signal.butter(2, 6, btype='low', fs=Fs, output='sos') 
lowpass_filtered_data = np.zeros(np.shape(abs_data))
for i in range(np.shape(abs_data)[1]):
    lowpass_filtered = signal.sosfiltfilt(lowpass_sos, abs_data[:,i])
    lowpass_filtered_data[:, i] = lowpass_filtered
    
    
def StoG(sta_frame, dynamic_frame, p, sta_local_o, dy_local_o):

    z_first = np.matmul(sta_frame, (p - sta_local_o).T)
    a = time_dynamic.shape[0]
    dy_point_global = np.zeros([a, 3])  # 初始化 z_third
    for i in range(a):
        
        z_second = np.matmul(dynamic_frame[i, :, :].T, z_first)
        dy_point_global[i, :] = z_second.T + dy_local_o[i, :]
    return dy_point_global




















