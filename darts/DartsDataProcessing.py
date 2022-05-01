# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:28:24 2022

@author: Hsin Yang, 20220428
"""

import os
import pandas as pd
import numpy as np
from pandas import DataFrame

def Read_File(x, subfolder='None'):
    # if subfolder = True, the function will run with subfolder
    folder_path = x
    csv_file_list = []
    
    if subfolder:
        file_list_1 = []
        for dirPath, dirNames, fileNames in os.walk(x):
            # file_list = os.walk(folder_name)
            file_list_1.append(dirPath)
                
        for ii in file_list_1:
            file_list = os.listdir(ii)
            for iii in file_list:
                if os.path.splitext(iii)[1] == ".ts":
                    file_list_name = ii + '\\' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(x)                
        for i in folder_list:
            if os.path.splitext(i)[1] == ".ts":
                file_list_name = folder_path + "\\" + i
                csv_file_list.append(i)                
        
    return csv_file_list

# -------------------------code staring---------------------------------

# read staging file
staging_file_path = r'D:\NTSU\TenLab\test\DartsStagingFile.xlsx'
staging_file_data = pd.read_excel(staging_file_path, sheet_name="S3", skiprows=1)
# read data
data_path = r'D:\NTSU\TenLab\test\data\S3\SET1 ROUND_3 ok chen 20HZ.ts'
example_data = pd.read_csv(data_path, delimiter='\t' ,skiprows=4, encoding='UTF-8')

# read file list
# setting file's folder
folder_path = r'D:\NTSU\TenLab\test\data\S3'
file_list = Read_File(folder_path, subfolder=True)

# Determine the file name
extract_data = np.zeros([np.shape(file_list)[0], 14])
columns_name = example_data.columns[0:13]
columns_name = columns_name.insert(0, 'FileName')
extract_data = pd.DataFrame(extract_data, columns=columns_name)
for i in range(len(file_list)):
    for ii in range(len(staging_file_data.iloc[:len(file_list), 1])):
        if file_list[i] == staging_file_data.iloc[:len(file_list), 1][ii]:
            darts_data = pd.read_csv(file_list[i], delimiter='\t' ,skiprows=4, encoding='UTF-8')
            # using staging file to define data location
            extract_data.loc[i, 1:] = darts_data.iloc[int(staging_file_data.iloc[ii, 4])-1, :]
            extract_data.iloc[i, 0] = file_list[i]
# write data to excel
file_name = r'D:\NTSU\TenLab\test\output\S3_output.xlsx'
DataFrame(extract_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
