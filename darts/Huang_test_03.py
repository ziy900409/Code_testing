# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:05:11 2022
增加創資料夾的功能
增加濾波功能

@author: Hsin Yang
"""
import os
import pandas as pd
from scipy import signal
import numpy as np
# 設定步驟
## 讀取所有資料"路徑"

# -----------------------Reading all of data path----------------------------
# using a recursive loop to traverse each folder
# and find the file extension has .csv
def Read_File(FilePath, y, subfolder='None'):
    """
    Parameters
    ----------
    FilePath : string
        資料夾路徑.
    y : string
        副檔名.
    subfolder : boolean, optional
        是否有子資料夾. The default is 'None'.

    Returns
    -------
    csv_file_list : list
        所有資料路徑.

    """
    # if subfolder = True, the function will run with subfolder
    folder_path = FilePath
    data_type = y
    csv_file_list = []
    
    if subfolder:
        file_list_1 = []
        for dirPath, dirNames, fileNames in os.walk(FilePath):
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
        folder_list = os.listdir(FilePath)                
        for i in folder_list:
            if os.path.splitext(i)[1] == data_type:
                file_list_name = folder_path + "\\" + i
                csv_file_list.append(file_list_name)                
        
    return csv_file_list

# code starting

# file_list = Read_File(r'F:\HsinYang\NTSU\TenLab\Dart\test',
#               '.xlsx',
#               subfolder=True)

## 讀取路徑下的"資料"
# raw_data = pd.read_excel(r"F:/HsinYang/NTSU/TenLab/Dart/test/S4/S4_S2R1.xlsx",
#                          skiprows=4)

## 讀分期檔
Staging_file = pd.read_excel(r"F:/HsinYang/NTSU/TenLab/Dart/S4.xlsx",
                         skiprows=1)
## 擷取時間
Staging_file["FileName"]

start_time = Staging_file["加速起點"] -1
end_time = Staging_file["釋放鏢"]
## 擷取raw data
# raw_data.iloc[start_time:end_time, [1,2,3,8,9,10]]
## 寫入excel

# 設定條件
save_path = r"F:\HsinYang\NTSU\TenLab\Dart\test\processing"
# 找資料夾
folder_path = r"F:\HsinYang\NTSU\TenLab\Dart\test\raw"
folder_list = os.listdir(folder_path)
for iii in range(len(folder_list)):
    print(folder_list[iii])
    file_list = Read_File(folder_path + "\\" + folder_list[iii],
                  '.xlsx',
                  subfolder=False)
    print(file_list)
    for i in range(len(file_list)):
        # 依序找檔案
        filepath, tempfilename = os.path.split(file_list[i])
        list_filename, extension = os.path.splitext(tempfilename)
        for ii in range(len(Staging_file["FileName"])):
            # 對分期檔
            filepath, tempfilename = os.path.split(Staging_file["FileName"][ii])
            staging_filename, extension = os.path.splitext(tempfilename)
            if list_filename == staging_filename:
                print("file list: ", file_list[i])
                print("staging file:", Staging_file["FileName"][ii])
                ## 擷取raw data
                ## 讀取路徑下的"資料"
                raw_data = pd.read_excel(file_list[i],
                                         skiprows=4)
                ## 擷取時間
                start_time = Staging_file["加速起點"][ii] -1
                end_time = Staging_file["釋放鏢"][ii]
                ## 擷取raw data
                cut_data = raw_data.iloc[start_time:end_time, [1,2,3,8,9,10]]
                bandpass_sos = signal.butter(2, 50/0.802,  btype='lowpass', fs=200, output='sos')
                bandpass_filtered_data = np.zeros(np.shape(cut_data))
                for i in range(np.shape(cut_data)[1]):
                    # print(i)
                    # using dual filter to processing data to avoid time delay
                    bandpass_filtered = signal.sosfiltfilt(bandpass_sos, cut_data.iloc[:,i])
                    bandpass_filtered_data[:, i] = bandpass_filtered 
                # 寫入資料位置
                save_file_name = save_path +"\\" + folder_list[iii] + "\\" + staging_filename + "_ed.xlsx"
                ## 寫入excel
                pd.DataFrame(cut_data).to_excel(save_file_name, sheet_name='Sheet1', index=True, header=True)
                