# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 10:53:27 2024

@author: Hsin.YH.Yang
"""

import gc
import os
import sys
# 路徑改成你放自己code的資料夾
# sys.path.append(r"E:\Hsin\git\git\Code_testing\Archery\Xiao")
sys.path.append(r"D:\BenQ_Project\git\Code_testing\Archery\Xiao")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import XiaoThesisMotionFunction as mot
import XiaoThesisGeneralFunction as gen
import XiaoThesisEMGFunction as emg
from detecta import detect_onset
from scipy import signal
import time

from datetime import datetime
# matplotlib 設定中文顯示，以及圖片字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False   # 步驟二（解決坐標軸負數的負號顯示問題）
font = {'family': 'serif',
        'color':  'k',
        'weight': 'bold',
        'size': 20,
        }

# 获取当前日期和时间
now = datetime.now()

# 将日期转换为指定格式
# formatted_date = datetime.now().strftime('%Y-%m-%d-%H:%M')
formatted_date = datetime.now().strftime('%Y-%m-%d-%H%M')
print("當前日期：", formatted_date)
# %% parameter setting 
# staging_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\Archery_stage_v5_input.xlsx"
staging_path = r"D:\BenQ_Project\python\Archery\202405\Archery_stage_v5_input.xlsx"
data_path = r"D:\BenQ_Project\python\Archery\202405\202405\202405\\"

# 測試組
subject_list = ["R01", "R02", "R03", "R04"]
# ------------------------------------------------------------------------
# 設定資料夾
folder_paramter = {
                  "first_layer": {
                                  "motion":["\\motion\\"],
                                  "EMG": ["\\EMG\\"],
                                  },
                  "second_layer":{
                                  "motion":["\\Raw_Data\\", "\\Processing_Data\\"],
                                  "EMG": ["\\Raw_Data\\", "\\Processing_Data\\"],
                                  },
                  "third_layer":{
                                  "motion":["Method_1"],
                                  "EMG": ["Method_1", "Method_2"],
                                  },
                  "fourth_layer":{
                                  "motion":["\\motion\\"],
                                  "EMG": ["motion", "MVC", "SAVE", "X"],
                                  }
                  }
folder_paramter["fourth_layer"]["EMG"][0]

# 第三層 ----------------------------------------------------------------------

fig_folder = "\\figure\\"
data_folder = "\\data\\"
MVC_folder = "\\MVC\\"
# downsampling frequency
down_freq = 1000
# 抓放箭時候前後秒數
# example : [秒數*採樣頻率, 秒數*採樣頻率]
release = [5*down_freq, 1*down_freq]
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.1 # 窗格長度 (單位 second)
overlap_len = 0.5 # 百分比 (%)
# 預處理資料可修改檔名，並新增標籤，如：S2_MVC_Rep_1.16 -> S2_MVC_Rep_1.16_low
end_name = "_ed"
# 平滑處理方式 ex: lowpass, rms, moving
smoothing_method = 'rms'
# cutoff frequency
c = 0.802
lowpass_cutoff = 10/c
# median frequency duration
duration = 1 # unit : second
# processing_folder_path = data_path + processingData_folder

# ---------------------找放箭時間用--------------------------------------------
# 設定最接近放箭位置之acc sensor的欄位編號，建議看完三軸資料再選最大的
# 可設定數字或是欄位名稱：ex: R EXTENSOR GROUP: ACC.Y 1 or 5
release_acc = 7
# 設定放箭的振幅大小值
release_peak = 1.0
# trigger threshold
trigger_threshold = 0.02
# 设定阈值和窗口大小
threshold = 0.03
window_size = 5
# 設置繪圖顏色用 --------------------------------------------------------------
cmap = plt.get_cmap('Set2')
# 设置颜色
colors = [cmap(i) for i in np.linspace(0, 1, 6)]
                                    
                                    

# %% 路徑設置

all_rawdata_folder_path = {"motion": [], "EMG": []}
all_processing_folder_path = {"motion": [], "EMG": []}
# 定義 motion
for method in folder_paramter["third_layer"]["motion"]:
    # 定義 rawdata folder path
    rawdata_folder_path = data_path + folder_paramter["first_layer"]["motion"][0] + \
        folder_paramter["second_layer"]["motion"][0] + method
    # 讀取 rawdata folder path
    rawdata_folder_list = [os.path.join(rawdata_folder_path, f) \
                           for f in os.listdir(rawdata_folder_path) \
                           if not f.startswith('.') and os.path.isdir(os.path.join(rawdata_folder_path, f))]
    # 將路徑加到 all_rawdata_folder_path
    all_rawdata_folder_path["motion"].extend(rawdata_folder_list)
    # 定義 processing folder path, 改變 second layer
    processing_folder_path = data_path + folder_paramter["first_layer"]["motion"][0] + \
        folder_paramter["second_layer"]["motion"][1] + method
    processing_folder_list = [os.path.join(processing_folder_path, f) \
                              for f in os.listdir(processing_folder_path) \
                                  if not f.startswith('.') and \
                                      os.path.isdir(os.path.join(processing_folder_path, f))]
    all_processing_folder_path["motion"].extend(processing_folder_list)
    
# 定義 EMG folder path
for method in folder_paramter["third_layer"]["EMG"]:
    # 定義 rawdata folder path
    rawdata_folder_path = data_path + folder_paramter["first_layer"]["EMG"][0] + \
        folder_paramter["second_layer"]["EMG"][0] + method
    # 讀取 rawdata folder path
    rawdata_folder_list = [os.path.join(rawdata_folder_path, f) \
                           for f in os.listdir(rawdata_folder_path) \
                           if not f.startswith('.') and os.path.isdir(os.path.join(rawdata_folder_path, f))]
    # 將路徑加到 all_rawdata_folder_path
    all_rawdata_folder_path["EMG"].extend(rawdata_folder_list)
    # 定義 processing folder path, 改變 second layer
    processing_folder_path = data_path + folder_paramter["first_layer"]["EMG"][0] + \
        folder_paramter["second_layer"]["EMG"][1] + method
    processing_folder_list = [os.path.join(processing_folder_path, f) \
                              for f in os.listdir(processing_folder_path) \
                                  if not f.startswith('.') and \
                                      os.path.isdir(os.path.join(processing_folder_path, f))]
    all_processing_folder_path["EMG"].extend(processing_folder_list)
        

gc.collect(generation=2)

# %%
"""
3. 資料前處理: 
    3.1. 需至 function code 修改設定參數
        3.1.1. down_freq = 1800
        # downsampling frequency
        3.1.2. bandpass_cutoff = [8/0.802, 450/0.802]
        # 帶通濾波頻率
        3.1.3. lowpass_freq = 20/0.802
        # 低通濾波頻率
        3.1.4. time_of_window = 0.1 # 窗格長度 (單位 second)
        # 設定移動平均數與移動均方根之參數
        3.1.5. overlap_len = 0.5 # 百分比 (%)
        # 更改window length, 更改overlap length
        
    3.2. 資料處理順序
        3.2.1. bandpsss filter, smoothing data.
        3.2.2. 將處理後之 MVC data 存成 .xlsx 檔案.
        3.2.3. motion data 僅繪圖，資料貯存在 motion data 裁切的部分
"""
# 處理MVC data

for i in range(len(all_rawdata_folder_path["EMG"])):
    tic = time.process_time()
    
    MVC_folder_path = all_rawdata_folder_path["EMG"][i] + "\\" + MVC_folder
    MVC_list = gen.Read_File(MVC_folder_path, ".csv")
    fig_save_path = all_rawdata_folder_path["EMG"][i].replace("Raw_Data", "Processing_Data") \
                + "\\" + fig_folder
    print("Now processing MVC data in " + all_rawdata_folder_path["EMG"][i] + "\\")
    for MVC_path in MVC_list:
        print(MVC_path)
        # 讀取資料
        raw_data = pd.read_csv(MVC_path)
        # EMG data 前處理
        processing_data, bandpass_filtered_data = emg.EMG_processing(raw_data, smoothing=smoothing_method)
        # 將檔名拆開
        filepath, tempfilename = os.path.split(MVC_path)
        filename, extension = os.path.splitext(tempfilename)
        # 畫 FFT analysis 的圖
        emg.Fourier_plot(MVC_path,
                        (fig_save_path + "\\FFT\\MVC"),
                        filename,
                        notch=False)
        emg.Fourier_plot(MVC_path,
                        (fig_save_path + "\\FFT\\MVC"),
                        filename,
                        notch=True)
        # 畫 bandpass 後之資料圖
        emg.plot_plot(bandpass_filtered_data, str(fig_save_path + "\\processing\\bandpass\\" + MVC_folder),
                     filename, "Bandpass_")
        # 畫smoothing 後之資料圖
        emg.plot_plot(processing_data, str(fig_save_path + "\\processing\\smoothing\\" + MVC_folder),
                     filename, str(smoothing_method + "_"))
        # writting data in worksheet
        file_name =  all_rawdata_folder_path["EMG"][i].replace("Raw_Data", "Processing_Data")\
                + data_folder + MVC_folder + '\\' + filename + end_name + ".xlsx"
        pd.DataFrame(processing_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    toc = time.process_time()
    print("Total Time:",toc-tic)  
gc.collect(generation=2)

# %% 找 MVC 最大值
"""
4. 
"""
tic = time.process_time()
for i in range(len(all_processing_folder_path["EMG"])):
    print("To find the maximum value of all of MVC data in: " + \
          all_processing_folder_path["EMG"][i].split("\\")[-1])
    
    emg.Find_MVC_max(all_processing_folder_path["EMG"][i] + data_folder + MVC_folder,
                     all_processing_folder_path["EMG"][i] + "\\")
toc = time.process_time()
print("MVC Data Total Time Spent: ",toc-tic)      
gc.collect(generation=2)



































