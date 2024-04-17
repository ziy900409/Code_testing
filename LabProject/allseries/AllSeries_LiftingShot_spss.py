# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:14:27 2024

@author: Hsin.YH.Yang
"""
# %% import library
import pandas as pd
import sys
sys.path.append(r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\4. code")

import AllSeries_general_func_20240327 as gen
# %% parameter setting
emg_muscle = ['Extensor carpi radialis_RMS', 'Flexor Carpi Radialis_RMS',
              'Triceps_RMS', 'Extensor carpi ulnaris_RMS',
              '1st. dorsal interosseous_RMS', 'Abductor digiti quinti_RMS',
              'Extensor Indicis_RMS', 'Biceps_RMS']

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


remove_left = gen.removeoutliers(left_group)

remove_left.isna().sum()
subject_number = raw_data['subject'].value_counts().index

# 找到同肌肉名稱的 columns 的 index
index_dict = {col: raw_data.columns.get_loc(col) for col in emg_muscle}
# 創建資料儲存的地方

for subject in subject_number:
    for direction in ['R', 'L']:
        temp_idx = []
        for i in range(len(raw_data)):    
            if subject == raw_data["subject"][i] and \
                direction == raw_data['direction'][i]:
                print(raw_data["subject"][i], raw_data['direction'][i])
                temp_idx.append(i)
        subject_data = raw_data.loc[temp_idx, emg_muscle]
        remove_data = gen.removeoutliers(subject_data)
        remove_data.isna().sum()
                


















