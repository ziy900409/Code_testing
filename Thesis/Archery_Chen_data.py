
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 12:53:32 2022
For Dr. Chen's EMG data proccessing
@author: Hsin Yang, 20220424
"""

import os
import pandas as pd
import numpy as np
from scipy import signal
from pandas import DataFrame
import time
import matplotlib.pyplot as plt 

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
    EMG_data = data.iloc[:, [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]]
    # exchange data type to float64
    EMG_data = [pd.to_numeric(EMG_data.iloc[:, i], errors = 'coerce') 
                for i in range(np.shape(EMG_data)[1])]
    EMG_data = pd.DataFrame(np.transpose(EMG_data),
                            columns=data.iloc[:, [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]].columns)
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
    window_width = int(0.05/(1/np.floor(Fs))) #width of the window for computing RMS
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
    file_name = data_save_path + '\\' + filename + '_RMS'
    file_name = file_name.replace('.', '_') + '.xlsx'
    # writting data in worksheet
    DataFrame(data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    
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
    find_max_name = MVC_save_path + '\\' + MVC_save_path.split('\\')[-1] + '_MVC_rms.xlsx'
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
        for ii in range(len(staging_data['FileName'])):
            if shooting_file_list[i] == staging_data['FileName_rms'][ii].split('\\')[-1]:
                print('shooting_file: ', shooting_file_list[i])
                print('Staging_file: ', staging_data['FileName_rms'][ii].split('\\')[-1])
                
                # loading shooting EMG value
                shooting_file_name = shooting_folder + '\\' + shooting_file_list[i]
                # define release time
                release_time = int(staging_data['Time Frame'][ii])
                print(release_time)
                # load EMG data
                shooting_data = pd.read_excel(shooting_file_name)
                # trunkcate specific period
                shooting_EMG = shooting_data.iloc[release_time - 5000:release_time + 1000, 1:]    
                # calculate iMVC data
                shooting_iMVC = np.divide(shooting_EMG, MVC_value)*100
                shooting_iMVC.insert(0, 'time', shooting_data.iloc[:,0])
                # writting iMVC data in a EXCEL
                save_iMVC_name = save_file_path + '\\' + 'iMVC_' + shooting_file_list[i]
                DataFrame(shooting_iMVC).to_excel(save_iMVC_name, sheet_name='Sheet1', index=False, header=True)
    
    for i in range(len(fatigue_file_list)):
            for ii in range(len(staging_data['FileName'])):
                if fatigue_file_list[i] == staging_data['FileName_rms'][ii].split('\\')[-1]:
                    print('fatigue_file: ', fatigue_file_list[i])
                    print('Staging_file: ', staging_data['FileName_rms'][ii].split('\\')[-1])
                    # loading shooting EMG value
                    fatigue_file_name = fatigue_folder + '\\' + fatigue_file_list[i]
                    # define release time
                    release_time = int(staging_data['Time Frame'][ii])
                    print(release_time)
                    # load EMG data
                    fatigue_data = pd.read_excel(fatigue_file_name)
                    # trunkcate specific period
                    fatigue_EMG = fatigue_data.iloc[release_time - 5000:release_time + 1000, 1:]    
                    # calculate iMVC data
                    fatigue_iMVC = np.divide(fatigue_EMG, MVC_value)*100
                    fatigue_iMVC.insert(0, 'time', fatigue_data.iloc[:,0])
                    # writting iMVC data in a EXCEL
                    save_iMVC_name = save_file_path + '\\' + 'iMVC_' + fatigue_file_list[i]
                    DataFrame(fatigue_iMVC).to_excel(save_iMVC_name, sheet_name='Sheet1', index=False, header=True)
        
        
## ------to find release timing---------

def find_release_time(folder_path, save_path):
    file_list_path = folder_path
    file_list = Read_File(file_list_path, '.csv', subfolder=True)
    release_timing_list = pd.DataFrame(columns = ['FileName', 'Time Frame'])
    for ii in range(len(file_list)):
        if os.path.splitext(file_list[ii])[1] == ".csv":
            # File_Name_List.append(ii)
            print(file_list[ii])
            File_Name = file_list[ii]
            data = pd.read_csv(File_Name)
            # to find R EXTENSOR CARPI RADIALIS: ACC X data [43]
            Extensor_ACC = data.iloc[:, 75]
            release_timing = data.iloc[:, 0]
            release_timing = pd.DataFrame(release_timing)
            release_timing = release_timing.fillna(0)
            # there should change with different subject
            peaks, _ = signal.find_peaks(Extensor_ACC*-1, height=3.5)
            # Because ACC sampling frequency is 148Hz and EMG is 2000Hz
            release_index = np.argmin(abs(release_timing - (peaks[0]/148)))
            # to create DataFrame that easy to write data in a excel
            # ii = ii.replace('.', '_')
            release_time_number = pd.DataFrame([File_Name, release_index])
            release_time_number = np.transpose(release_time_number)
            release_time_number.columns = ['FileName', 'Time Frame']
            release_timing_list = release_timing_list.append(release_time_number)
    # writting data to a excel file
    
    save_iMVC_name = save_path + '\\' + folder_path.split('\\')[-1] + '_ReleaseTiming.xlsx' 
    DataFrame(release_timing_list).to_excel(save_iMVC_name, sheet_name='Sheet1', index=False, header=True)
    return release_timing_list


# ---------------code staring------------------------------
# -----------------------loop code starting--------------------
rowdata_folder_path = r"D:\NTSU\ChenDissertationDataProcessing\EMG_Data\RawData"
rowdata_folder_list = os.listdir(rowdata_folder_path)
processing_folder_path = r"D:\NTSU\ChenDissertationDataProcessing\EMG_Data\ProcessingData\RMS"

tic1 = time.process_time()
for i in range(len(rowdata_folder_list)):
    tic2 = time.process_time()
    print(rowdata_folder_list[i])
    # 預處理MVC data
    MVC_path = rowdata_folder_path + '\\' + rowdata_folder_list[i] + '\\MVC'
    MVC_list = Read_File(MVC_path, '.csv', subfolder=False)
    for ii in MVC_list:
        tic = time.process_time()
        print(ii)
        bandpass_filtered_data, rms_data, lowpass_filtered_data = EMG_processing(ii)
        data_save_path = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\MVC'
        Excel_writting(ii, data_save_path , rms_data)
        toc = time.process_time()
        print("Total Time:",toc-tic)
    # 預處理shooting data
    Shooting_path = rowdata_folder_path + '\\' + rowdata_folder_list[i] + '\\射箭'
    Shooting_list = Read_File(Shooting_path, '.csv', subfolder=True)
    for ii in Shooting_list:
        print(ii)
        tic = time.process_time()
        bandpass_filtered_data, rms_data, lowpass_filtered_data = EMG_processing(ii)
        data_save_path = processing_folder_path + '\\' + rowdata_folder_list[i]
        Excel_writting(ii, data_save_path, rms_data)
        # 寫資料近excel
        filepath, tempfilename = os.path.split(ii)
        filename, extension = os.path.splitext(tempfilename)
        # rewrite file name
        file_name = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\射箭\\' + filename + '_RMS.xlsx'
        # writting data in worksheet
        pd.DataFrame(rms_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
        toc = time.process_time()
        print("Total Time:",toc-tic)

    # # Find MVC max (已完成)
    # Find_MVC_max(processing_folder_path + '\\' + rowdata_folder_list[i] + '\\MVC',
    #               processing_folder_path + '\\' + rowdata_folder_list[i])    

    # calculate iMVC data
    # define data path
    rhythm_file = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + '韻律'
    mechanic_file = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + '機械'
    MVC_file = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + rowdata_folder_list[i] + '_MVC_rms.xlsx'
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
MVC_folder_list = Read_File(r'D:\NTSU\TenLab\ChenThesisData\EMG_Data\ProcessingData',
                            '',
                            subfolder = False)
for ii in range(len(MVC_folder_list)):
    print(ii)
    tic = time.process_time()
    MVC_folder = MVC_folder_list[ii] + '\\MVC'
    Find_MVC_max(MVC_folder, MVC_folder_list[ii])
    toc = time.process_time()
    print("Total Time:",toc-tic)

release_folder = r'D:\NTSU\ChenThesisData\EMG_Data\RawData'
release_folder_list = os.listdir(release_folder)
save_place = r'D:\NTSU\ChenThesisData\EMG_Data\ProcessingData'
for ii in range(len(release_folder_list)):
    print(ii)
    tic = time.process_time()
    find_release_time(release_folder + '\\' + release_folder_list[ii], save_place)
    toc = time.process_time()
    print("Total Time:",toc-tic)


