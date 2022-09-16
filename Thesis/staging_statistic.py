# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:24:25 2022
分三個時期，進行統計
@author: drink
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import spm1d

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

    
# read folder
all_folder_path = r'D:\NTSU\ChenDissertationDataProcessing\EMG_Data\ProcessingData\RMS\pattern2_rms'
folder_list = os.listdir(all_folder_path)

staging_data = pd.read_excel(r"D:\NTSU\Thesis\EMG_data_20220915.xlsx",
                             sheet_name = 'staging')
# 創立所有資料集之貯存位置
all_mean_aim = pd.DataFrame({})
all_mean_expansion = pd.DataFrame({})
all_mean_follow = pd.DataFrame({})
# 個人平均之貯存位置
group_mean_aim = pd.DataFrame({})
group_mean_expansion = pd.DataFrame({})
group_mean_follow = pd.DataFrame({})

for folder in range(len(folder_list)):
    print(folder_list[folder])
    folder_path = all_folder_path + '\\' + folder_list[folder] + '\iMVC\韻律'
    file_list = Read_File(folder_path, '.xlsx', subfolder=False)
    individual_mean_aim = pd.DataFrame({})
    individual_mean_expansion = pd.DataFrame({})
    individual_mean_follow = pd.DataFrame({})
    for file in file_list:
        filepath, tempfilename = os.path.split(file)
        filename, extension = os.path.splitext(tempfilename)
        for trail in range(len(staging_data['Trail'])):
            if filename == staging_data['Trail'][trail]:
                print(staging_data['expansion_time'][trail])
                data = pd.read_excel(file)
                # 取出資料
                aim_data = data.iloc[:staging_data['expansion_time'][trail], :]
                expansion_data = data.iloc[staging_data['expansion_time'][trail]:5000, :]
                follow_data = data.iloc[5000:, :]
                # 取平均值
                mean_aim_data = np.transpose(pd.DataFrame(np.mean(aim_data, axis=0)))
                mean_expansion_data = np.transpose(pd.DataFrame(np.mean(expansion_data, axis=0)))
                mean_follow_data = np.transpose(pd.DataFrame(np.mean(follow_data, axis=0)))
                # 插入釋放時間
                mean_aim_data.insert(0, 'expansion', staging_data['expansion_time'][trail])
                mean_expansion_data.insert(0, 'expansion', staging_data['expansion_time'][trail])
                mean_follow_data.insert(0, 'expansion', staging_data['expansion_time'][trail])
                # 加入檔名
                mean_aim_data.insert(0, 'filename', filename)
                mean_expansion_data.insert(0, 'filename', filename)
                mean_follow_data.insert(0, 'filename', filename)
                # 合併矩陣
                individual_mean_aim = pd.concat([individual_mean_aim, mean_aim_data], axis = 0)
                individual_mean_expansion = pd.concat([individual_mean_expansion, mean_expansion_data], axis = 0)
                individual_mean_follow = pd.concat([individual_mean_follow, mean_follow_data], axis = 0)
    # 合併矩陣
    all_mean_aim = pd.concat([all_mean_aim, individual_mean_aim], axis = 0, ignore_index=True)
    # all_mean_aim = all_mean_aim.reset_index(inplace=True)
    all_mean_expansion = pd.concat([all_mean_expansion, individual_mean_expansion], axis = 0, ignore_index=True)
    # all_mean_expansion = all_mean_expansion.reset_index(inplace=True)
    all_mean_follow = pd.concat([all_mean_follow, individual_mean_follow], axis = 0, ignore_index=True)
    # all_mean_follow = all_mean_follow.reset_index(inplace=True)
    # 計算個人平均數
    mean_individual_aim = np.transpose(pd.DataFrame(np.mean(individual_mean_aim.iloc[1:, :], axis = 0)))
    mean_individual_expansion = np.transpose(pd.DataFrame(np.mean(individual_mean_expansion.iloc[1:, :], axis = 0)))
    mean_individual_follow = np.transpose(pd.DataFrame(np.mean(individual_mean_follow.iloc[1:, :], axis = 0)))
    # 插入資料夾名稱
    mean_individual_aim.insert(0, 'folder', folder_list[folder])
    mean_individual_expansion.insert(0, 'folder', folder_list[folder])
    mean_individual_follow.insert(0, 'folder', folder_list[folder])
    # 合併矩陣
    group_mean_aim = pd.concat([group_mean_aim, mean_individual_aim], axis = 0)
    group_mean_expansion = pd.concat([group_mean_expansion, mean_individual_expansion], axis = 0)
    group_mean_follow = pd.concat([group_mean_follow, mean_individual_follow], axis = 0)











