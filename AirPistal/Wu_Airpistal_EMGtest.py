# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 20:14:22 2022

@author: Hsin Yang
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from pandas import DataFrame
import time


# -----------------------Reading all of data path----------------------------
# using a recursive loop to traverse each folder
# and find the file extension has .csv
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

# ---------------------EMG data processing--------------------------------
def EMG_processing(cvs_file_path):
 
    data = pd.read_csv(cvs_file_path,encoding='UTF-8')
    # to define data time
    data_time = data.iloc[:,0]
    Fs = 1/(data_time[2]-data_time[1]) # sampling frequency
    data = pd.DataFrame(data)
    # need to change column name or using column number
    EMG_data = data.iloc[:, np.arange(1,88,8)]
    # exchange data type to float64
    EMG_data = [pd.to_numeric(EMG_data.iloc[:, i], errors = 'coerce') 
                for i in range(np.shape(EMG_data)[1])]
    EMG_data = pd.DataFrame(np.transpose(EMG_data),
                            columns=data.iloc[:, np.arange(1,88,8)].columns)
    # bandpass filter use in signal
    bandpass_sos = signal.butter(2, [20, 500],  btype='bandpass', fs=Fs, output='sos')
    
    bandpass_filtered_data = np.zeros(np.shape(EMG_data))
    for i in range(np.shape(EMG_data)[1]):
        # print(i)
        # using dual filter to processing data to avoid time delay
        bandpass_filtered = signal.sosfiltfilt(bandpass_sos, EMG_data.iloc[:,i])
        bandpass_filtered_data[:, i] = bandpass_filtered 
    
    # caculate absolute value to rectifiy EMG signal
    bandpass_filtered_data = abs(bandpass_filtered_data)     
    # -------Data smoothing. Compute RMS
    # The user should change window length and overlap length that suit for your experiment design
    # window width = window length(second)//time period(second)
    window_width = int(0.04265/(1/np.floor(Fs))) #width of the window for computing RMS
    rms_data = np.zeros(np.shape(bandpass_filtered_data))
    for i in range(np.shape(rms_data)[1]):
        for ii in range(np.shape(rms_data)[0]-window_width):
            data_location = ii+(window_width/2)
            rms_data[int(data_location), i] = (np.sum(bandpass_filtered_data[ii:ii+window_width, i])
                               /window_width)
    # ------linear envelop analysis-----------                          
    # ------lowpass filter parameter that the user must modify for your experiment        
    lowpass_sos = signal.butter(2, 6, btype='low', fs=Fs, output='sos')        
    lowpass_filtered_data = np.zeros(np.shape(bandpass_filtered_data))
    for i in range(np.shape(rms_data)[1]):
        lowpass_filtered = signal.sosfiltfilt(lowpass_sos, bandpass_filtered_data[:,i])
        lowpass_filtered_data[:, i] = lowpass_filtered
    # add columns name to data frame
    bandpass_filtered_data = pd.DataFrame(bandpass_filtered_data, columns=EMG_data.columns)
    rms_data = pd.DataFrame(rms_data, columns=EMG_data.columns)
    lowpass_filtered_data = pd.DataFrame(lowpass_filtered_data, columns=EMG_data.columns)
    # insert time data in the DataFrame
    lowpass_filtered_data.insert(0, 'time', data_time)
    rms_data.insert(0, 'time', data_time)
    bandpass_filtered_data.insert(0, 'time', data_time)    
    return bandpass_filtered_data, rms_data, lowpass_filtered_data


#  --------------------writting data to a excel file------------------------ 
def Excel_writting(file_path, data_save_path, data):
    # deal with filename and add extension with _ed
    filepath, tempfilename = os.path.split(file_path)
    filename, extension = os.path.splitext(tempfilename)
    # rewrite file name
    file_name = data_save_path + '\\' + filename + '_ed'
    file_name = file_name.replace('.', '_') + '.xlsx'
    # writting data in worksheet
    DataFrame(data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    
def Find_MVC_max(MVC_folder, MVC_save_path):
    # MVC_folder = r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\Processing\MVC'
    # MVC_save_path = r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\Processing'
    MVC_file_list = os.listdir(MVC_folder)
    MVC_data = pd.read_excel(MVC_folder + '\\' + MVC_file_list[0])
    find_max_all = []
    Columns_name = MVC_data.columns
    Columns_name = Columns_name.insert(0, 'FileName')
    find_max_all = pd.DataFrame(find_max_all, columns=Columns_name)
    for i in MVC_file_list:
        MVC_file_path = MVC_folder + '\\' + i
        MVC_data = pd.read_excel(MVC_file_path)
        find_max = MVC_data.max(axis=0)
        find_max = pd.DataFrame(find_max)
        find_max = np.transpose(find_max)
        find_max.insert(0, 'FileName', i)
        find_max_all = find_max_all.append(find_max)
    # find maximum value from each file
    MVC_max = find_max_all.max(axis=0)
    MVC_max[0] = 'Max value'
    MVC_max = pd.DataFrame(MVC_max)
    MVC_max = np.transpose(MVC_max)
    find_max_all = find_max_all.append(MVC_max)
    # writting data to EXCEl file
    find_max_name = MVC_save_path + '\\' + 'S1_MVC.xlsx'
    DataFrame(find_max_all).to_excel(find_max_name, sheet_name='Sheet1', index=False, header=True)    
# ------------code start-----------------
# ------------code start-----------------

parent_folder_path = r"D:\NTSU\TenLab\Shooting\Wu_EMG\Shooting_EMG_Data\RawData"
save_data_path = r"D:\NTSU\TenLab\Shooting\Wu_EMG\Shooting_EMG_Data\ProcessingData"
folder_list = os.listdir(parent_folder_path)

for forder_name in folder_list:
    folder_list_path = parent_folder_path + '\\' + forder_name
    file_list = Read_File(folder_list_path, '.csv', subfolder=True)
    tic = time.process_time()
    for file_path in file_list:
        print(file_path)
        bandpass_filtered_data, rms_data, lowpass_filtered_data = EMG_processing(file_path)
        save_data_path_1 = save_data_path + '\\' + forder_name
        Excel_writting(file_path, save_data_path_1, lowpass_filtered_data)
    Find_MVC_max(save_data_path + '\\' + forder_name, save_data_path + '\\' + forder_name)
    print(forder_name)

    toc = time.process_time()
    print("Total Time:",toc-tic)
