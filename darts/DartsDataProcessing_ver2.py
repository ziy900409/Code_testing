# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:56:09 2022
version 1.2
 
Test for Huang's Dart data

@author: Hsin Yang, 20220530
"""

import os
import numpy as np
import pandas as pd

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

# read staging file
# 設定分期檔的路徑
staging_file_path = r'D:\NTSU\TenLab\HuangTest\Joint kinematics\DartsStagingFile.xlsx'
staging_file_data = pd.read_excel(staging_file_path, skiprows=1, sheet_name="S3")
# read data
# 設定範例資料檔用，可用隨意一種檔案做使用
data_path = r'D:\NTSU\TenLab\HuangTest\Joint kinematics\S3\S3_S1_R3.data'
example_data = pd.read_csv(data_path, delimiter='\t', skiprows=2, encoding='UTF-8')

# read file list
# setting file's folder
# 設定目標資料夾
folder_path = r'D:\NTSU\TenLab\HuangTest\Joint kinematics\S3'
file_list = Read_File(folder_path, '.data', subfolder=False)

# Determine the file name
# 預先創建貯存資料的矩陣
calculate_data_1 = np.zeros([np.shape(file_list)[0], np.shape(example_data)[1]+1])
calculate_data_2 = np.zeros([np.shape(file_list)[0], np.shape(example_data)[1]+1])
calculate_data_3 = np.zeros([np.shape(file_list)[0], np.shape(example_data)[1]+1])
# 設定欄位名稱
columns_name = example_data.columns
columns_name = columns_name.insert(0, 'FileName')
calculate_data_1 = pd.DataFrame(calculate_data_1, columns=columns_name)
calculate_data_2 = pd.DataFrame(calculate_data_2, columns=columns_name)
calculate_data_3 = pd.DataFrame(calculate_data_3, columns=columns_name)
spe_claculate_data = pd.DataFrame(calculate_data_1, columns=columns_name)
max_calculate_data = pd.DataFrame(calculate_data_2, columns=columns_name)
min_calculate_data = pd.DataFrame(calculate_data_3, columns=columns_name)

for i in range(len(file_list)):
    for ii in range(len(staging_file_data['FileName'])):
        if file_list[i] == staging_file_data['FileName'][ii]:
            print(i)
            print(ii)
            print(staging_file_data['FileName'][ii])
            print(file_list[i])
            # read data
            dart_data = pd.read_csv(file_list[i], delimiter='\t', skiprows=3, encoding='UTF-8')
            # using staging file to extract data
            # 利用分期檔抓時間點
            start_frame = int(staging_file_data['加速起點'][ii])
            end_frame = int(staging_file_data['釋放鏢'][ii])
            extract_data = dart_data.iloc[start_frame-1:end_frame, :]
            # convert str to float data type
            # 轉換資料格式 從字串變浮點數
            # extract_data.iloc[:, :] = extract_data.iloc[:, :].astype(float)
            # 將資料的0轉換成NaN
            extract_data[extract_data==0] = np.nan
            # 計算最大值
            max_extract_data = pd.DataFrame(extract_data.max())
            # 計算最小值
            min_extract_data = pd.DataFrame(extract_data.min())
            # 抓特定時間點
            spe_extract_data = dart_data.iloc[end_frame-1, :]
            # assign data to claculate matrix
            # assign file name to first column
            
            spe_claculate_data.iloc[i, 0] = file_list[i]
            max_calculate_data.iloc[i, 0] = file_list[i]
            min_calculate_data.iloc[i, 0] = file_list[i]
            
            spe_claculate_data.iloc[i, 1:] = pd.DataFrame(spe_extract_data).T
            max_calculate_data.iloc[i, 1:] = max_extract_data.T
            min_calculate_data.iloc[i, 1:] = min_extract_data.T
            
# write data to excel
# 將資料寫進EXCEL，可修改檔案名稱
file_name = r'D:\NTSU\TenLab\HuangTest\Joint kinematics\S4_output.xlsx'
# DataFrame(calculate_data).to_excel(file_name, sheet_name='ReleaseTime', index=False, header=True)
# DataFrame(max_calculate_data).to_excel(file_name, sheet_name='Maximum', index=False, header=True)
# DataFrame(min_calculate_data).to_excel(file_name, sheet_name='Minimum', index=False, header=True)

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

# Write each dataframe to a different worksheet.
spe_claculate_data.to_excel(writer, sheet_name='ReleaseTime', index=False, header=True)
max_calculate_data.to_excel(writer, sheet_name='Maximum', index=False, header=True)
min_calculate_data.to_excel(writer, sheet_name='Minimum', index=False, header=True)

# Close the Pandas Excel writer and output the Excel file.
writer.save()
