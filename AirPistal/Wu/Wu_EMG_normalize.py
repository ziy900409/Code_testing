#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:16:29 2022

@author: Hsin Yang
"""
import os
import pandas as pd
import numpy as np
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

# 設定工作路徑
MVC_folder = 'MVC'
processing_data_folder = 'RR'
# 獲取子資料集
MVC_file_list = os.listdir(MVC_folder)
processing_data_folder = os.listdir(processing_data_folder)

for data_folder in processing_data_folder:
    # 處理MVC資料
    MVC_file＿path = MVC_folder + '//' + data_folder + '_MVC.xlsx'
    MVC_value = pd.read_excel(MVC_file＿path)
    MVC_value = MVC_value.iloc[-1, 1:]
    MVC_value = MVC_value.iloc[1:]
    # 讀取motion data list
    processing_data_list = Read_File(processing_data_folder + '//' + data_folder,
                                     '.xlsx', subfolder=False)
    
    for motion_list in processing_data_list:
        motion_data = pd.read_excel(motion_list)
        # 標準化EMG
        normalize_data = np.divide(motion_data, MVC_value) * 100
        # 處理存擋路徑
        filepath, tempfilename = os.path.split(motion_list)
        save_iMVC_name = processing_data_folder + '\\' + data_folder + '\\iMVC\\' + 'iMVC_' + tempfilename
        # 將資料寫進excel
        pd.DataFrame(normalize_data).to_excel(save_iMVC_name, sheet_name='Sheet1', index=False, header=True)
    
