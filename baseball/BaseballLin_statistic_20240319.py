# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:04:47 2024

@author: Hsin.YH.Yang
"""
# %%
import pandas as pd
import os
import sys
sys.path.append(r"D:\BenQ_Project\python\Lin\PythonCode")
# 將read_c3d function 加進現有的工作環境中
import BaseballFunction_20230516 as af
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# %% 0. parameter estting
folder_path = r"D:\BenQ_Project\python\Lin\Processing_Data"
data_sheet = ["Stage2", "Stage3"]

# %% 1. read staging file
staging_file = pd.read_excel(r"D:\BenQ_Project\python\Lin\motion分期肌電用_20240317.xlsx",
                             sheet_name='memo2')
folder_file_list = os.listdir(folder_path)
all_file_list = []

for folder in folder_file_list: 
    file_list = af.Read_File(folder_path + "\\" + folder + "\data\motion",
                             ".xlsx",
                             subfolder=False)
    all_file_list.extend(file_list)

ture_file_path = []
for file_name in staging_file["EMG_File"]:
    # print(file_name)
    for file_path in all_file_list:
        # print(file_path)
        if file_name in file_path:
            print(file_path)
            ture_file_path.append(file_path)
            
# %% truncate data
'''
1. 讀取資料，分別為 stage2, stage3
2. 內插成101個點
'''


muscle1 = pd.DataFrame()
muscle2 = pd.DataFrame()
muscle3 = pd.DataFrame()
muscle4 = pd.DataFrame()
muscle5 = pd.DataFrame()
muscle6 = pd.DataFrame()
muscle_data = np.zeros([len(ture_file_path), 202, 6])

for file in range(len(ture_file_path)):
    for sheet in data_sheet:
        emg_data = pd.read_excel(ture_file_path[file], sheet_name=sheet)
        for i in range(np.shape(emg_data)[1] - 1): # 減去時間欄位
            # 內插函數
            x = emg_data.iloc[:, 0] # time
            y = emg_data.iloc[:, i+1]
            f = interp1d(x, y, kind='cubic')
            x_new = np.linspace(emg_data.iloc[0, 0], emg_data.iloc[-1, 0], 101)
            y_new = f(x_new)
            if sheet == "Stage2":
                muscle_data[file, :101, i] = y_new
            elif sheet == "Stage3":
                muscle_data[file, 101:, i] = y_new

arr_muscle_data = np.zeros([6, 202, len(ture_file_path)])
for i in range(np.shape(arr_muscle_data)[0]):
    for ii in range(np.shape(arr_muscle_data)[2]):
        arr_muscle_data = 
















