# -*- coding: utf-8 -*-
"""
Author: Hsin Yang, 2022. 11.07

purpose: to capture specific range of interest in motion data. Using Force Plate data to define staging period.

"""
import pandas as pd
import numpy as np
import os

data_path = '/Users/hui/Documents/NTSU/ForcePlate_Lin/motion/S10_clay_3m_1.data'
example_data = pd.read_csv(data_path, delimiter='\t', skiprows=2, encoding='UTF-8')

# read data list
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
                    file_list_name = ii + '//' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(x)                
        for i in folder_list:
            if os.path.splitext(i)[1] == data_type:
                file_list_name = folder_path + "//" + i
                csv_file_list.append(file_list_name)                
        
    return csv_file_list

## --------------------code starting---------------------
## ------------------------------------------------------
## ------------------------------------------------------
# judge staging file name and motion file
# 分期檔路徑設定
staging_file = pd.read_excel('/Users/hui/Documents/NTSU/ForcePlate_Lin/Tennis_Staging_3m_Lin_20221017.xlsx')
# 讀取資料檔案路徑
data_list = Read_File('/Users/hui/Documents/NTSU/ForcePlate_Lin/motion/',
                      '.data',
                      subfolder=False)
# 資料存放位置
data_save_folder = '/Users/hui/Documents/NTSU/ForcePlate_Lin/staging_motion/'
# create a space to store data
all_max_period_all = pd.DataFrame({})
all_min_period_all = pd.DataFrame({})
all_max_period_1 = pd.DataFrame({})
all_min_period_1 = pd.DataFrame({})
all_max_period_2 = pd.DataFrame({})
all_min_period_2 = pd.DataFrame({})
all_ROM_period_all = pd.DataFrame({})
all_ROM_period_1 = pd.DataFrame({})
all_ROM_period_2 = pd.DataFrame({})
all_max_air = pd.DataFrame({})
all_min_air = pd.DataFrame({})
all_ROM_air = pd.DataFrame({})
time_period = pd.DataFrame({})

# read data list
for data_path in data_list:
    file_path_split = data_path.split('/', -1)[-1]
    data_name = file_path_split.split('.', -1)[0]
    # 讀分期檔中存取的欄位名稱
    for staging_path in range(len(staging_file['file_name'])):
        staging_name = staging_file['file_name'][staging_path].split('.', -1)[0]
        # make sure data name is same with staging name
        # 確定motion與分期檔中的檔名相同
        if data_name == staging_name:
            print(data_name)
            print(staging_name)
            # make sure staging period is not NAN
            # 確定分期時間點中沒有NAN
            if np.isnan(staging_file['進力版時間1'][staging_path]) != 1:
                # define staging period
                # 定義分期時間點
                start_1 = int(staging_file['進力版時間1'][staging_path] / 10)
                end_1 = int(staging_file['出力版時間1'][staging_path] / 10)
                start_2 = int(staging_file['進力版時間2'][staging_path] / 10)
                end_2 = int(staging_file['出力版時間2'][staging_path] / 10)
                # read data
                data = pd.read_csv(data_path, delimiter='\t', names = example_data.columns, skiprows=4, encoding='UTF-8')
                # motion data
                # trunkcut specifical time period
                # 切割時間點
                motion_data = data.iloc[start_1:end_2, :].astype('float64')
                # 增加個時間點資料
                add_time = pd.DataFrame({
                    'file_path':[data_path],
                    'file_name':[data_name],
                    'start_1':[start_1],
                    'end_1':[end_1],
                    'staet_2':[start_2],
                    'end_2':[end_2]
                    })
                # 合併時間資料
                time_period = pd.concat([time_period, add_time], ignore_index=True)
                # period all
                # maximum period all
                # .astype('float64') 將所有欄位轉成float64 避免有欄位被當成string
                add_max_period_all = pd.DataFrame(data.iloc[start_1:end_2, :].astype('float64').max()).T #最大
                # minimum period all
                add_min_period_all = pd.DataFrame(data.iloc[start_1:end_2, :].astype('float64').min()).T # 最小
                # ROM period all
                add_ROM_period_all = add_max_period_all - add_min_period_all
                # 插入檔名
                add_max_period_all.insert(0, 'file_name', data_path) #插入檔名
                add_min_period_all.insert(0, 'file_name', data_path)
                add_ROM_period_all.insert(0, 'file_name', data_path)
                # 合併矩陣
                all_max_period_all = pd.concat([all_max_period_all, add_max_period_all], ignore_index=True) #合併矩陣
                all_min_period_all = pd.concat([all_min_period_all, add_min_period_all], ignore_index=True)
                all_ROM_period_all = pd.concat([all_ROM_period_all, add_ROM_period_all], ignore_index=True)
                # period 1
                # maximum period 1
                add_max_period_1 = pd.DataFrame(data.iloc[start_1:end_1, :].astype('float64').max()).T
                # minimum period 1
                add_min_period_1 = pd.DataFrame(data.iloc[start_1:end_1, :].astype('float64').min()).T
                # ROM period 1
                add_ROM_period_1 = add_max_period_1 - add_min_period_1
                # 插入檔名
                add_max_period_1.insert(0, 'file_name', data_path)
                add_min_period_1.insert(0, 'file_name', data_path)
                add_ROM_period_1.insert(0, 'file_name', data_path)
                # 合併矩陣
                all_max_period_1 = pd.concat([all_max_period_1, add_max_period_1], ignore_index=True)
                all_min_period_1 = pd.concat([all_min_period_1, add_min_period_1], ignore_index=True)
                all_ROM_period_1 = pd.concat([all_ROM_period_1, add_ROM_period_1], ignore_index=True)
                # period 2
                # maximum period 2
                add_max_period_2 = pd.DataFrame(data.iloc[start_2:end_2, :].astype('float64').max()).T
                # minimum period 2
                add_min_period_2 = pd.DataFrame(data.iloc[start_2:end_2, :].astype('float64').min()).T
                # ROM period 2
                add_ROM_period_2 = add_max_period_2 - add_min_period_2
                # 插入檔名
                add_max_period_2.insert(0, 'file_name', data_path)
                add_min_period_2.insert(0, 'file_name', data_path)
                add_ROM_period_2.insert(0, 'file_name', data_path)
                # 合併矩陣
                all_max_period_2 = pd.concat([all_max_period_2, add_max_period_2], ignore_index=True)
                all_min_period_2 = pd.concat([all_min_period_2, add_min_period_2], ignore_index=True)
                all_ROM_period_2 = pd.concat([all_ROM_period_2, add_ROM_period_2], ignore_index=True)
                # period air
                add_max_air = pd.DataFrame(data.iloc[end_1:start_2, :].astype('float64').max()).T
                add_min_air = pd.DataFrame(data.iloc[end_1:start_2, :].astype('float64').min()).T
                add_ROM_air = add_max_air - add_min_air
                # 插入檔名
                add_max_air.insert(0, 'file_name', data_path)
                add_min_air.insert(0, 'file_name', data_path)
                add_ROM_air.insert(0, 'file_name', data_path)
                # 合併矩陣
                all_max_air = pd.concat([all_max_air, add_max_air], ignore_index=True)
                all_min_air = pd.concat([all_min_air, add_min_air], ignore_index=True)
                all_ROM_air = pd.concat([all_ROM_air, add_ROM_air], ignore_index=True)
                # save motion data
                # 將資料存到EXCEL
                pd.DataFrame(motion_data).to_excel((data_save_folder + 'cut_' + data_name + '.xlsx'),
                                                    sheet_name='Sheet1', index=False, header=True)
# 存取最大最小值之excel
excel_save = '/Users/hui/Documents/NTSU/ForcePlate_Lin/motion_3m.xlsx'
with pd.ExcelWriter(excel_save) as writer:
    # period all
    all_max_period_all.to_excel(writer, sheet_name='all_period_max')
    all_min_period_all.to_excel(writer, sheet_name='all_period_min')
    all_ROM_period_all.to_excel(writer, sheet_name='all_period_ROM')
    # period 1
    all_max_period_1.to_excel(writer, sheet_name='step_1_max')
    all_min_period_1.to_excel(writer, sheet_name='step_1_min')
    all_ROM_period_1.to_excel(writer, sheet_name='step_1_ROM')
    # period 2
    all_max_period_2.to_excel(writer, sheet_name='step_2_max')
    all_min_period_2.to_excel(writer, sheet_name='step_2_min')
    all_ROM_period_2.to_excel(writer, sheet_name='step_2_ROM')
    # air
    all_max_air.to_excel(writer, sheet_name='air_max')
    all_min_air.to_excel(writer, sheet_name='air_min')
    all_ROM_air.to_excel(writer, sheet_name='air_ROM')
    time_period.to_excel(writer, sheet_name='time_data')
