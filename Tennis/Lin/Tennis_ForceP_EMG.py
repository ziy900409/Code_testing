# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:55:41 2022
網球EMG資料分析，利用第一塊Force Plate做資料分期。
流程：
1. EMG 預處理：濾波bandpass 20-500Hz -> ads -> lowpass 6 Hz
2. 找MVC最大值
3. 標準化 EMG of motion / MVC max
@author: Hsin Yang
"""
import os
import pandas as pd
import numpy as np
from scipy import signal
import time
import math

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
def EMG_processing(cvs_file_list):
 
    data = pd.read_csv(cvs_file_list,encoding='UTF-8')
    # to define data time
    data_time = data.iloc[:,0]
    Fs = 1/(data_time[2]-data_time[1]) # sampling frequency
    data = pd.DataFrame(data)
    # need to change column name or using column number
    EMG_data = data.iloc[:, [1, 15, 29, 43, 57, 65, 73, 81, 89, 97, 105]]
    # 1 9 17 25 33 41 49
    # exchange data type to float64
    EMG_data = [pd.to_numeric(EMG_data.iloc[:, i], errors = 'coerce') 
                for i in range(np.shape(EMG_data)[1])]
    EMG_data = pd.DataFrame(np.transpose(EMG_data),
                            columns=data.iloc[:, [1, 15, 29, 43, 57, 65, 73, 81, 89, 97, 105]].columns)
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

# ------------to find maximum MVC value-----------------------------

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
    find_max_name = MVC_save_path + '\\' + MVC_save_path.split('\\')[-1] + '_MVC_2.xlsx'
    pd.DataFrame(find_max_all).to_excel(find_max_name, sheet_name='Sheet1', index=False, header=True)

## code start
# MVC path
MVC_path = r'D:\NTSU\TenLab\LinData\tennis EMG+force plate\All in\MVC'
MVC_folder_list = os.listdir(MVC_path)

# 處理MVC
for MVC_folder in MVC_folder_list:
    path = MVC_path + '\\' + MVC_folder + '\\RawData'
    MVC_list = Read_File(path, '.csv', subfolder=False)
    for data_path in MVC_list:
        print(data_path)
        tic = time.process_time()
        bandpass_filtered_data, rms_data, lowpass_filtered_data = EMG_processing(data_path)
        # 寫資料近excel
        filepath, tempfilename = os.path.split(data_path)
        filename, extension = os.path.splitext(tempfilename)
        # rewrite file name
        file_name = MVC_path + '\\' + MVC_folder + '\\AfterFiltered\\' + filename + '_lowpass' + '.xlsx'
        # writting data in worksheet
        pd.DataFrame(lowpass_filtered_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
        toc = time.process_time()
        print("Total Time:",toc-tic)
    Find_path = MVC_path + '\\' + MVC_folder + '\\AfterFiltered'
    MVC_save_path =r'D:\NTSU\TenLab\LinData\tennis EMG+force plate\All in\MVC' + '\\' + MVC_folder 
    Find_MVC_max(Find_path, MVC_save_path)
    

# 濾波EMG of motion，並擷取特定區段
Save_motion_path = r'D:\NTSU\TenLab\LinData\tennis EMG+force plate\All in\motion'
motion_folder = r'D:\NTSU\TenLab\LinData\tennis EMG+force plate\All in\motion'
motion_folder_list = os.listdir(motion_folder)
# staging data
staging_data = pd.read_excel(r"D:\NTSU\TenLab\LinData\tennis EMG+force plate\Tennis_Staging_Lin_Lin_20220912.xlsx")
for folder in motion_folder_list:
    motion_folder_path = motion_folder + '\\' + folder
    motion_file_list = Read_File(motion_folder_path, '.csv', subfolder=False)
    for path in range(len(motion_file_list)):
        for num in range(len(staging_data['EMG_file_name'])):
            if motion_file_list[path] == staging_data['EMG_file_name'][num]:
                tic2 = time.process_time()
                print(motion_file_list[path])
                print(staging_data['EMG_file_name'][num])
                bandpass_filtered_data, rms_data, lowpass_filtered_data = EMG_processing(motion_file_list[path])
                # 寫資料近excel
                filepath, tempfilename = os.path.split(motion_file_list[path])
                filename, extension = os.path.splitext(tempfilename)
                # rewrite file name
                file_name = Save_motion_path + '\\' + folder + '\\AfterFiltered\\' + filename + '_lowpass' + '.xlsx'
                if math.isnan(staging_data['進力版時間'][num]) != True:
                    # writting data in worksheet
                    start_time = int((staging_data['進力版時間'][num] - staging_data['trigger時間'][num]) / 2400 * 2000)
                    end_time = int((staging_data['出力版時間'][num]- staging_data['trigger時間'][num]) / 2400 * 2000)
                    print('starting time is: ', start_time)
                    print('ending time is: ', end_time)
                    # load EMG data
                    # trunkcate specific period
                    shooting_data = lowpass_filtered_data.iloc[start_time:end_time, 1:]
                    pd.DataFrame(shooting_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
                    toc2 = time.process_time()
                    print("Total Time:",toc2-tic2)
                    
# 設定資料夾位置
MVC_max_path = r'D:\NTSU\TenLab\LinData\tennis EMG+force plate\All in\MVC'                    
motion_folder_path = r'D:\NTSU\TenLab\LinData\tennis EMG+force plate\All in\motion'
save_file_path = r'D:\NTSU\TenLab\LinData\tennis EMG+force plate\All in\motion'
# 資料夾清單
MVC_max_folder_list = os.listdir(MVC_max_path)
motion_folder_list = os.listdir(motion_folder_path)

for i in range(len(motion_folder_list)):
    for ii in range(len(MVC_max_folder_list)):
        if motion_folder_list[i] == MVC_max_folder_list[ii]:
            print(motion_folder_list[i])
            # 讀取MVC數據
            MVC_path = MVC_max_path + '\\' + MVC_max_folder_list[ii] + '\\' + MVC_max_folder_list[ii] + '_MVC_2.xlsx'
            MVC_value = pd.read_excel(MVC_path)
            MVC_value = MVC_value.iloc[-1, 1:]
            MVC_value = MVC_value.iloc[1:]
            # 讀取EMG of motion data
            motion_path = motion_folder_path + '\\' + motion_folder_list[i] + '\\AfterFiltered'
            motion_list = Read_File(motion_path, '.xlsx', subfolder=False)
            for iii in motion_list:
                print(iii)
                moation_data =  pd.read_excel(iii)
                shooting_iMVC = np.divide(moation_data, MVC_value)*100
                # 將資料寫進excel
                filepath, tempfilename = os.path.split(iii)
                save_iMVC_name = save_file_path + '\\' + motion_folder_list[i] + '\\iMVC\\' + 'iMVC_' + tempfilename
                pd.DataFrame(shooting_iMVC).to_excel(save_iMVC_name, sheet_name='Sheet1', index=False, header=True)

