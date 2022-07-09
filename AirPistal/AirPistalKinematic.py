# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 11:34:58 2022

@author: Hsin Yang
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
# 自動讀檔用，可判斷路徑下所有檔名，不管有沒有子資料夾
# 可針對不同副檔名的資料作判讀
def Read_File(x, y, subfolder='None'):
    '''
    This function will return file path
    
    Parameters
    ----------
    x : str
        file path
    y : str
        file type
        example: ".csv"
    subfolder : boolean
        if subfolder = True, the function will run with subfolder
    
    Returns
    -------
    csv_file_list : list
    '''
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

# -------------------------code staring---------------------------------
filename = r"D:\NTSU\TenLab\Shooting\AirPistol\TRC_Raw\S1\S1_1.trc"

# file = open(filename, 'rb')

# content = file.read().decode()
# content = content.split(os.linesep)
# data_frame = int(float(content[2].split('\t')[0]))

# def _append_per_label_data(self, markers, data):
#     for index, marker_data in enumerate(data):
#         self[markers[index]] += [marker_data]

# file.close()
# read staging file
# 設定分期檔的路徑
staging_file_path = r"D:\NTSU\TenLab\Shooting\AirPistol\Shooting Staging_Motion EMG Finger_0709.xlsx"
staging_file_data = pd.read_excel(staging_file_path)
# read data
# 設定範例資料檔用，可用隨意一種檔案做使用
data_path = r"D:\NTSU\TenLab\Shooting\AirPistol\TRC_Raw\S1\S1_1.trc"
data_path = r"D:\NTSU\TenLab\Shooting\AirPistol\TRC_Raw\S4\S4_1.trc"
example_data = pd.read_csv(data_path, header = None, delimiter='	', skiprows=5, encoding='UTF-8', on_bad_lines='skip')

# 確認資料夾數量
folder_path = r'D:\NTSU\TenLab\Shooting\AirPistol\TRC_Raw'
folder_number = os.listdir(folder_path)
# 確認資料路徑
file_path = r'D:\NTSU\TenLab\Shooting\AirPistol\TRC_Raw'
file_list = Read_File(file_path, '.trc', subfolder=True)
# 設定資料參數
Fs = 240

# ------------------先取出gun2資料-------------------------
# 依資料夾順序

for i in folder_number:
    # read file list in the folder
    file_list = Read_File(folder_path + '\\' + i,
                          '.trc', subfolder=False)
    # create column name
    file_name = []
    for name in file_list:
        file_name.append(name.split('\\')[-1])
        
    gun_x = np.zeros([360, np.shape(file_list)[0]])
    gun_y = np.zeros([360, np.shape(file_list)[0]])
    gun_z = np.zeros([360, np.shape(file_list)[0]])
    for ii in range(len(file_list)):
        for iii in range(len(staging_file_data['file_path'])):
            # to judge staging file file path is equal to file list
            if file_list[ii] == staging_file_data['file_path'][iii]:
                print(ii)
                print(iii)
                print(staging_file_data['file_path'][iii])
                print(file_list[ii])
                # read data
                shooting_data = pd.read_csv(file_list[ii], header = None, delimiter='\t', skiprows=5, encoding='UTF-8', on_bad_lines='skip')
                # using staging file to extract data
                # 利用分期檔抓時間點
                start_frame = int(staging_file_data['Shooting(motion)'][iii])
                # end_frame = int(staging_file_data['釋放鏢'][ii])
                extract_data = shooting_data.iloc[start_frame-240:start_frame+120, 128:131]
                gun_x[:, ii] = extract_data.iloc[:, 0]
                gun_y[:, ii] = extract_data.iloc[:, 1]
                gun_z[:, ii] = extract_data.iloc[:, 2]
    # --------------------------Smoothing-----------------------------------------                          
    # ------lowpass filter parameter that the user must modify for your experiment        
    lowpass_sos = signal.butter(2, 16, btype='low', fs=Fs, output='sos')
    # create a matrix to save data        
    lowpass_filtered_data_x = np.zeros([360, np.shape(file_list)[0]])
    lowpass_filtered_data_y = np.zeros([360, np.shape(file_list)[0]])
    lowpass_filtered_data_z = np.zeros([360, np.shape(file_list)[0]])
    for iiii in range(np.shape(lowpass_filtered_data_x)[1]):
        # coordinate X
        lowpass_filtered_x = signal.sosfiltfilt(lowpass_sos, gun_x[:,iiii])
        lowpass_filtered_data_x[:, iiii] = lowpass_filtered_x
        # coordinate Y
        lowpass_filtered_y = signal.sosfiltfilt(lowpass_sos, gun_y[:,iiii])
        lowpass_filtered_data_y[:, iiii] = lowpass_filtered_y
        # coordinate Z
        lowpass_filtered_z = signal.sosfiltfilt(lowpass_sos, gun_z[:,iiii])
        lowpass_filtered_data_z[:, iiii] = lowpass_filtered_z
        # add columns name to data frame
    
    # redefine the data structure
    lowpass_filtered_data_x = pd.DataFrame(lowpass_filtered_data_x, columns = file_name)
    lowpass_filtered_data_y = pd.DataFrame(lowpass_filtered_data_y, columns = file_name)
    lowpass_filtered_data_z = pd.DataFrame(lowpass_filtered_data_z, columns = file_name)
    
    # writing data to the excel
    file_name = file_path + '\\' + i + '_output.xlsx'
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    lowpass_filtered_data_x.to_excel(writer, sheet_name='x', index=False, header=True)
    lowpass_filtered_data_y.to_excel(writer, sheet_name='y', index=False, header=True)
    lowpass_filtered_data_z.to_excel(writer, sheet_name='z', index=False, header=True)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


# --------------------------draw the figure-----------------------------------------                          
# Data_time = shooting_data.iloc[start_frame-240:start_frame+120, 1]
# # plot the signal after filter
# plt.figure(1)
# plt.plot(Data_time, extract_data.iloc[:,0], linewidth = 0.5, label = 'Original')
# plt.plot(Data_time, lowpass_filtered_data_x.iloc[:, -1], linewidth = 0.5, label = 'Bandpassfilter')
# plt.title('After 16 Hz low-pass filter')
# plt.xlabel('Time [seconds]')
# plt.legend()
# plt.tight_layout()
# plt.grid()
