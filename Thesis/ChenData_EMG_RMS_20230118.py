# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 12:55:04 2022
For Dr. Chen's EMG data proccessing
@author: drink
"""

import os
import pandas as pd
import numpy as np
from scipy import signal
from pandas import DataFrame
import time

# from numba import autojit

# @autojit

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
                    # replace "\\" to '/', due to MAC version
                    file_list_name = ii + '\\' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(x)                
        for i in folder_list:
            if os.path.splitext(i)[1] == data_type:
                # replace "\\" to '/', due to MAC version
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
    EMG_data = data.iloc[:, [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]]
    # exchange data type to float64
    EMG_data = [pd.to_numeric(EMG_data.iloc[:, i], errors = 'coerce') 
                for i in range(np.shape(EMG_data)[1])]
    EMG_data = pd.DataFrame(np.transpose(EMG_data),
                            columns=data.iloc[:, [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]].columns)
    # bandpass filter use in signal
    bandpass_sos = signal.butter(2, [8/0.802, 450/0.802],  btype='bandpass', fs=Fs, output='sos')
    
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
        
    # -------Data smoothing. Compute RMS
    # The user should change window length and overlap length that suit for your experiment design
    # window width = window length(second)//time period(second)
    window_width_rms = int(0.05/(1/np.floor(Fs))) #width of the window for computing RMS
    overlap_len = 0.99 # 百分比
    rms_data = np.zeros([int((np.shape(bandpass_filtered)[0] - window_width_rms)/  ((1-overlap_len)*window_width_rms)) + 1,
                            np.shape(abs_data)[1]])
    for i in range(np.shape(rms_data)[1]):
        for ii in range(np.shape(rms_data)[0]):
            data_location = int(ii*(1-overlap_len)*window_width_rms)
            # print(data_location, data_location+window_width_rms)
            rms_data[int(ii), i] = np.sqrt(np.sum((abs_data[data_location:data_location+window_width_rms, i])**2)
                                      /window_width_rms)
            
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
    
#  --------------------writting data to a excel file------------------------ 
def Excel_writting(file_path, data_save_path, data):
    # deal with filename and add extension with _ed
    filepath, tempfilename = os.path.split(file_path)
    filename, extension = os.path.splitext(tempfilename)
    # rewrite file name
    file_name = data_save_path + '\\' + filename + '_RMS'
    file_name = file_name.replace('.', '_') + '.xlsx'
    # writting data in worksheet
    DataFrame(data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    
# ------------to find maximum MVC value-----------------------------

def Find_MVC_max(MVC_folder, MVC_save_path):
    # MVC_folder = r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\Processing\MVC'
    # MVC_save_path = r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\Processing'
    MVC_file_list = os.listdir(MVC_folder)
    MVC_data = pd.read_excel(MVC_folder + '\\' + MVC_file_list[0], engine='openpyxl')
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
        # find_max_all = find_max_all.append(find_max)
        find_max_all = pd.concat([find_max_all, find_max], axis=0, ignore_index=True)
    # find maximum value from each file
    MVC_max = find_max_all.max(axis=0)
    MVC_max[0] = 'Max value'
    MVC_max = pd.DataFrame(MVC_max)
    MVC_max = np.transpose(MVC_max)
    # find_max_all = find_max_all.append(MVC_max)
    find_max_all = pd.concat([find_max_all, MVC_max], axis=0, ignore_index=True)
    # writting data to EXCEl file
    find_max_name = MVC_save_path + '\\' + MVC_save_path.split('\\')[-1] + '_MVC_move.xlsx'
    DataFrame(find_max_all).to_excel(find_max_name, sheet_name='Sheet1', index=False, header=True)

## -----iMVC calculate------------
## it has something wrong in time insert and definition

def iMVC_calculate(MVC_file, shooting_folder, fatigue_folder, save_file_path, staging_file):
    # to obtain file name from the parent folder and make a list
    shooting_file_list = os.listdir(shooting_folder)
    fatigue_file_list = os.listdir(fatigue_folder)
    # to define staging data
    staging_data = pd.read_excel(staging_file)
    # to define MVC value
    MVC_value = pd.read_excel(MVC_file)
    MVC_value = MVC_value.iloc[-1, 1:]
    MVC_value = MVC_value.iloc[1:]
    
    for i in range(len(shooting_file_list)):
        # loading shooting EMG value
        shooting_file_name = shooting_folder + '\\' + shooting_file_list[i]
        # load EMG data
        shooting_data = pd.read_excel(shooting_file_name)
        # trunkcate specific period
        shooting_EMG = shooting_data.iloc[:, 1:]    
        # calculate iMVC data
        shooting_iMVC = np.divide(shooting_EMG, MVC_value)*100
        shooting_iMVC.insert(0, 'time', shooting_data.iloc[:,0])
        # writting iMVC data in a EXCEL
        save_iMVC_name = save_file_path + '\\' + 'iMVC_' + shooting_file_list[i]
        DataFrame(shooting_iMVC).to_excel(save_iMVC_name, sheet_name='Sheet1', index=False, header=True)

    
    for i in range(len(fatigue_file_list)):
        # loading shooting EMG value
        fatigue_file_name = fatigue_folder + '\\' + fatigue_file_list[i]
        # load EMG data
        fatigue_data = pd.read_excel(fatigue_file_name)
        # trunkcate specific period
        fatigue_EMG = fatigue_data.iloc[:, 1:]    
        # calculate iMVC data
        fatigue_iMVC = np.divide(fatigue_EMG, MVC_value)*100
        fatigue_iMVC.insert(0, 'time', fatigue_data.iloc[:,0])
        # writting iMVC data in a EXCEL
        save_iMVC_name = save_file_path + '\\' + 'iMVC_' + fatigue_file_list[i]
        DataFrame(fatigue_iMVC).to_excel(save_iMVC_name, sheet_name='Sheet1', index=False, header=True)
  
        


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# -----------------------code staring-----------------------------------------
# -----------------------loop code starting-----------------------------------
rowdata_folder_path = r"D:\NTSU\ChenDissertationDataProcessing\EMG_Data\RawData"
rowdata_folder_list = os.listdir(rowdata_folder_path)
# 去除有“.“開頭的檔案
rowdata_folder_list  = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.')]
processing_folder_path = r"D:\NTSU\ChenDissertationDataProcessing\EMG_Data\ProcessingData\RMS"

# 處理MVC data
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 資料前處理
for i in range(len(rowdata_folder_list)):
    tic2 = time.process_time()
    print(rowdata_folder_list[i])
    # 預處理MVC data
    MVC_path = rowdata_folder_path + '\\' + rowdata_folder_list[i] + '\\MVC'
    MVC_list = Read_File(MVC_path, '.csv', subfolder=False)
    for ii in MVC_list:
        tic = time.process_time()
        print(ii)
        moving_data, rms_data, lowpass_filtered_data = EMG_processing(ii)
        data_save_path = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\MVC'
        Excel_writting(ii, data_save_path , rms_data)
        toc = time.process_time()
        print("Total Time:",toc-tic)
# 找最大值


for i in range(len(rowdata_folder_list)):
    print(rowdata_folder_list[i])
    tic = time.process_time()
    Find_MVC_max(processing_folder_path + '\\' + rowdata_folder_list[i] + '\\MVC',
                  processing_folder_path + '\\' + rowdata_folder_list[i])
    toc = time.process_time()
    print("Total Time:",toc-tic)

    
# 處理shooting data
# ----------------------------------------------------------------------------
# 1. 取出所有Raw資料夾
# 2. 獲得Raw folder路徑下的“射箭”資料夾，並讀取所有.cvs file
# 3. 讀取processing folder路徑下的ReleaseTiming，並讀取檔案
# 4. 依序前處理“射箭”資料夾下的檔案
# 4.1 bandpass filting
# 4.2 trunkcut data by release time
# 4.3 依切割檔案計算moving average
# 4.4 輸出moving average to excel file
# ----------------------------------------------------------------------------

for i in range(len(rowdata_folder_list)):
    tic2 = time.process_time()
    print(rowdata_folder_list[i])
    # 預處理shooting data
    # for mac version replace "\\" by '/'
    Shooting_path = rowdata_folder_path + '\\' + rowdata_folder_list[i] + '\\射箭'
    Shooting_list = Read_File(Shooting_path, '.csv', subfolder=True)
    # 讀取staging file
    staging_file = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + rowdata_folder_list[i] + '_ReleaseTiming_1.xlsx'
    
    staging_data = pd.read_excel(staging_file)
    for ii in range(len(Shooting_list)):
        for iii in range(len(staging_data['FileName'])):
            # for mac version replace "\\" by '/'
            if Shooting_list[ii].split('\\')[-1] == staging_data['FileName_org'][iii].split('\\')[-1]:
                print('shooting_file: ', Shooting_list[ii])
                print('Staging_file: ', staging_data['FileName_rms'][iii].split('\\')[-1])
                data = pd.read_csv(Shooting_list[ii], encoding='UTF-8')
                # to define data time
                data_time = data.iloc[:,0]
                Fs = 1/(data_time[2]-data_time[1]) # sampling frequency
                data = pd.DataFrame(data)
                # need to change column name or using column number
                EMG_data = data.iloc[:, [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]]
                # exchange data type to float64
                EMG_data = [pd.to_numeric(EMG_data.iloc[:, i], errors = 'coerce') 
                            for i in range(np.shape(EMG_data)[1])]
                # define columns name
                ## useless, because in bandpass processing columns name will be eliminate!!
                EMG_data = pd.DataFrame(np.transpose(EMG_data),
                                        columns=data.iloc[:, [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]].columns)
                # bandpass filter use in signal
                bandpass_sos = signal.butter(2, [8/0.802, 450/0.802],  btype='bandpass', fs=Fs, output='sos')
                bandpass_filtered_data = np.zeros(np.shape(EMG_data))
                for columns in range(np.shape(EMG_data)[1]):
                    # print(i)
                    # using dual filter to processing data to avoid time delay
                    bandpass_filtered = signal.sosfiltfilt(bandpass_sos, EMG_data.iloc[:,columns])
                    bandpass_filtered_data[:, columns] = bandpass_filtered 
                
                # caculate absolute value to rectifiy EMG signal
                abs_data = abs(bandpass_filtered_data)
                # 定義分期時間
                release_time = int(staging_data['Time Frame'][iii])
                print(release_time)
                shooting_EMG = abs_data[release_time - 5000:release_time + 1100, :]
                # -------Data smoothing. Compute RMS
                # The user should change window length and overlap length that suit for your experiment design
                # window width = window length(second)//time period(second)
                window_width_rms = int(0.05/(1/np.floor(Fs))) #width of the window for computing RMS
                overlap_len = 0.99 # 百分比
                rms_data = np.zeros([int((np.shape(shooting_EMG)[0] - window_width_rms)/  ((1-overlap_len)*window_width_rms)) + 1,
                                        np.shape(shooting_EMG)[1]])
                for rows in range(np.shape(rms_data)[1]):
                    for columns in range(np.shape(rms_data)[0]):
                        data_location = int(columns*(1-overlap_len)*window_width_rms)
                        # print(data_location, data_location+window_width_rms)
                        rms_data[int(columns), rows] = np.sqrt(np.sum((shooting_EMG[data_location:data_location+window_width_rms, rows])**2)
                                                  /window_width_rms)
                # 定義資料型態與欄位名稱
                rms_data = pd.DataFrame(rms_data, columns=EMG_data.columns)
                # 定義RMS DATA的時間.
                rms_time_index = np.linspace(0, np.shape(data_time)[0]-1, np.shape(rms_data)[0])
                rms_time_index = rms_time_index.astype(int)
                time_2 = pd.DataFrame(data.iloc[rms_time_index, 0], index = None).reset_index(drop=True)
                rms_data = pd.concat([time_2, pd.DataFrame(rms_data)], axis = 1, ignore_index=False)                
                # 寫資料進excel
                filepath, tempfilename = os.path.split(Shooting_list[ii])
                filename, extension = os.path.splitext(tempfilename)
                # rewrite file name
                file_name = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + filename + '_move.xlsx'
                # writting data in worksheet
                pd.DataFrame(rms_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)

        
        
# 處理iMVC caculate
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
tic1 = time.process_time()
for i in range(len(rowdata_folder_list)):
    tic2 = time.process_time()
    print(rowdata_folder_list[i])


    # calculate iMVC data
    # define data path
    rhythm_file = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + '韻律'
    mechanic_file = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + '機械'
    MVC_file = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + rowdata_folder_list[i] + '_MVC_move.xlsx'
    save_file_path = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\iMVC'
    staging_file = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + rowdata_folder_list[i] + '_ReleaseTiming.xlsx'
    # # read data
    # rhythm_file_list = Read_File(rhythm_file, '.xlsx', subfolder=False)
    # mechanic_file_list = Read_File(mechanic_file, '.xlsx', subfolder=False)
    # iMVC function
    iMVC_calculate(MVC_file, rhythm_file, mechanic_file, save_file_path, staging_file)
    toc2 = time.process_time()
    print("Total Time:",toc2-tic2)
toc1 = time.process_time()
print("Total Time:",toc1-tic1)
