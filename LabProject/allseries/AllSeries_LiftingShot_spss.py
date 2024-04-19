# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:14:27 2024

@author: Hsin.YH.Yang
"""
# %% import library
import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append(r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\4. code")

import AllSeries_general_func_20240327 as gen
# %% parameter setting
emg_muscle = ['Extensor carpi radialis_RMS', 'Flexor Carpi Radialis_RMS',
              'Triceps_RMS', 'Extensor carpi ulnaris_RMS',
              '1st. dorsal interosseous_RMS', 'Abductor digiti quinti_RMS',
              'Extensor Indicis_RMS', 'Biceps_RMS']
mouse_name = ['EC', 'FK', 'S', 'U', 'ZA', 'Gpro']

# 獲取當前的日期及時間
now = datetime.now()

# 將日期轉換為指定格式
formatted_date = now.strftime("%Y-%m-%d")

# 輸出格式化後的日期
print("当前日期：", formatted_date)

# %% read data
raw_data = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\5. Statistics\LiftingShot_data_2024-04-15_1.xlsx",
                         sheet_name="Sheet1")

mouse_group = raw_data.loc[:, 'file_name'].str.split("_")
mouse_element = pd.DataFrame(mouse_group.str[2])
subject = pd.DataFrame(mouse_group.str[0])

raw_data["mouse"] = mouse_element
raw_data["subject"] = subject

emg_data = raw_data.loc[:, emg_muscle]
right_group = raw_data[raw_data['direction'] == 'R']
right_group = right_group.loc[:, emg_muscle]
left_group = raw_data[raw_data['direction'] == 'L']
left_group = left_group.loc[:, emg_muscle]


# remove_left = gen.removeoutliers(left_group)

# remove_left.isna().sum()
subject_number = raw_data['subject'].value_counts().index

# 找到同肌肉名稱的 columns 的 index
index_dict = {col: raw_data.columns.get_loc(col) for col in emg_muscle}
# 創建資料儲存的地方
emg_data_table = pd.DataFrame({}, columns = list(['subject','direction', 'mouse'] + emg_muscle)
                              )
for sub in subject_number:
    for direction in ['R', 'L']:
        temp_idx = []
        for i in range(len(raw_data)):    
            if sub == raw_data["subject"][i] and \
                direction == raw_data['direction'][i]:
                print(raw_data["subject"][i], raw_data['direction'][i])
                temp_idx.append(i)
        if np.size(raw_data.loc[temp_idx, emg_muscle]) > 0:
            subject_data = raw_data.loc[temp_idx, :].reset_index(drop=True)
            remove_data = gen.iqr_removeoutlier(raw_data.loc[temp_idx, emg_muscle])
            subject_data.loc[:, emg_muscle] = remove_data.values
            remove_data.isna().sum()
        for mouse in mouse_name:
            temp_mouse_data = pd.DataFrame(columns = subject_data.columns)
            for ii in range(np.shape(subject_data)[0]):
                if mouse == subject_data['mouse'][ii]:
                    print(subject_data['mouse'][ii])
                    add_data = pd.DataFrame([subject_data.iloc[ii, :].values],
                                            columns = subject_data.columns)
                    temp_mouse_data = pd.concat([temp_mouse_data, add_data],
                                                ignore_index=True)
            mean_temp = pd.DataFrame([np.mean(temp_mouse_data.loc[:, emg_muscle], axis=0)],
                                              columns = emg_muscle)
            mean_temp.insert(0, 'subject', sub)
            mean_temp.insert(1, 'direction', direction)
            mean_temp.insert(2, 'mouse', mouse)
            emg_data_table = pd.concat([emg_data_table, mean_temp])
                    
file_name = "LiftingShot_statistic_auto_" + formatted_date + ".xlsx"
path = r'D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\5. Statistics\\'
emg_data_table.to_excel(path + file_name,
                    sheet_name='Sheet1', index=False, header=True)                  
                


















