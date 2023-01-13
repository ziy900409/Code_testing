# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:32:12 2022

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


### read file
# 定義資料夾位置
folder_path = r'F:\HsinYang\NTSU\TenLab\Shooting\Wu_EMG\shooting data\一般'
sub_folder_name = os.listdir(folder_path)
for folder_name in sub_folder_name:
    print(folder_name)
    all_data_path = Read_File((folder_path + '\\' + folder_name + '\\iMVC'),
                          '.xlsx',
                          subfolder=False)
    # 去掉.xlsx的暫存檔
    all_data_path = [s for s in all_data_path if "~$" not in s]
    # create empty matrix
    all_max_index = pd.DataFrame({})
    all_max_values = pd.DataFrame({})
    all_max_time = pd.DataFrame({})
    for data_path in all_data_path:        
        # read path
        shooting_data = pd.read_excel(data_path, skiprows=1)
        # find max index
        max_index = pd.DataFrame(shooting_data.idxmax()).T

        # find max value
        max_values = pd.DataFrame(shooting_data.max()).T

        # locate the time of the maximum values
        max_time = pd.DataFrame(columns=shooting_data.columns)
        for columns_num in range(np.size(max_index)):
            # 將個欄位賦予時間值
            max_time.loc[-1, shooting_data.columns[columns_num]] = shooting_data.iloc[max_index.iloc[0, columns_num], 0]
        # insert data path 
        max_index.insert(0, "data_path", data_path)
        # insert data path
        max_values.insert(0, "data_path", data_path)
        # insert data path
        max_time.insert(0, "data_path", data_path)
        # concat data
        all_max_index = pd.concat([all_max_index, max_index], ignore_index=True)
        all_max_values = pd.concat([all_max_values, max_values], ignore_index=True)
        all_max_time = pd.concat([all_max_time, max_time], ignore_index=True)
    # writting data to EXCEL
    excel_save = r'F:\HsinYang\NTSU\TenLab\Shooting\Wu_EMG\shooting data\iMVC_max_index' + '\\' + folder_name + '_MaxTime.xlsx'
    with pd.ExcelWriter(excel_save) as writer:
    # period all
        all_max_index.to_excel(writer, sheet_name='all_max_index')
        all_max_values.to_excel(writer, sheet_name='all_max_values')
        all_max_time.to_excel(writer, sheet_name='all_max_time')
# find max and location
# find time

































