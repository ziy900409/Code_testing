# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:54:27 2023
流程：
1. 設定資料夾路徑與程式存放路徑
2. 先抓 release time
3. 預處理MVC
    3.1 檢查MVC統計資料
4. 處理motion

@author: Hsin.YH.Yang
"""

# %% import library
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"C:\Users\Public\BenQ\myPyCode\U3")
# 將read_c3d function 加進現有的工作環境中

import U3_EMG_function_v1 as af
import os
import time
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import gc
# %% 設定自己的資料路徑
# 資料路徑
data_path = r"E:\BenQ_Project\U3\07_EMGrecording"
# 設定資料夾
RawData_folder = "\\raw_data"
processingData_folder = "\\processing_data"
fig_save = "\\figure"
# 子資料夾名稱
sub_folder = ""
# 動作資料夾名稱
motion_folder = "motion"
# MVC資料夾名稱
MVC_folder = "MVC"
# downsampling frequency
down_freq = 1000
# 抓放箭時候前後秒數
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.1 # 窗格長度 (單位 second)
overlap_len = 0.5 # 百分比 (%)
# 預處理資料可修改檔名，並新增標籤，如：S2_MVC_Rep_1.16 -> S2_MVC_Rep_1.16_low
end_name = "_ed"
# 平滑處理方式 ex: lowpass, rms, moving
smoothing = 'lowpass'
# median frequency duration
duration = 1 # unit : second

# %% 路徑設置

rowdata_folder_path = data_path + RawData_folder + "\\" + sub_folder
# 去除有“.“開頭的檔案
rowdata_folder_list  = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.')]
processing_folder_path = data_path + "\\" + processingData_folder + "\\" + sub_folder
processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.')]
# %% 1. 清除所有 processing data 下.xlsx 與 .jpg 檔案
"""
1. 刪除檔案 (本動作刪除之檔案皆無法復原)
    1.1. 刪除processing data 資料夾下所有 .xlsx 與 .jpg 檔案
    1.2. 僅需執行一次

"""
def print_warning_banner():
    print("**************************************************")
    print("*                                                *")
    print("*     警告：這是一個警告標語！                     *")
    print("*     執行將會刪除資料夾下所有 .xlsx 與 .jpg 文件  *")
    print("*     此步驟無法回復所刪除之檔案                   *")
    print("*                                                *")
    print("**************************************************")
tic = time.process_time()    
print_warning_banner()
user_input = input("是否繼續執行刪除文件？(Y/N): ").strip().upper()
if user_input == "Y":
    # 在这里写下后续的代码
    print("繼續執行後續代碼...")

    # 先清除所有 processing data 下所有的資料    
    for i in range(len(processing_folder_list)):
        # 創建儲存資料夾路徑
        folder_list = []
        # 列出所有 processing data 下之所有資料夾
        for dirPath, dirNames, fileNames in os.walk(str(processing_folder_path + "\\" + processing_folder_list[i])):
            if os.path.isdir(dirPath):
                folder_list.append(dirPath)
        for ii in folder_list:
        # 清除所有 .xlsx 檔案
            print(ii)
            af.remove_file(ii, ".xlsx")
            af.remove_file(ii, ".xlsx")

elif user_input == "N":
    print("取消執行後續。")
else:
    print("無效輸入，請输入 Y 或 N。")
toc = time.process_time()
print("刪除檔案總共花費時間: ",toc-tic)

# %% 資料前處理 : bandpass filter, absolute value, smoothing, trunkcut data
# 處理MVC data
tic = time.process_time()
for i in range(len(rowdata_folder_list)):
    tic = time.process_time()
    MVC_folder_path = rowdata_folder_path + "\\" + rowdata_folder_list[i] + "\\" + MVC_folder
    MVC_list = af.Read_File(MVC_folder_path, ".csv")
    fig_save_path = processing_folder_path + "\\" + rowdata_folder_list[i] + fig_save
    print("Now processing MVC data in " + rowdata_folder_list[i])
    for MVC_path in MVC_list:
        print(MVC_path)
        data = pd.read_csv(MVC_path, encoding='UTF-8')
        data_save_path = processing_folder_path + '\\' + rowdata_folder_list[i] + "\\data\\" + MVC_folder
        # deal with filename and add extension with _ed
        filepath, tempfilename = os.path.split(MVC_path)
        filename, extension = os.path.splitext(tempfilename)
        # 畫圖
        # 前處理EMG data
        processing_data, bandpass_filtered_data = af.EMG_processing(data, smoothing="lowpass")
        # 將檔名拆開
        filepath, tempfilename = os.path.split(MVC_path)
        filename, extension = os.path.splitext(tempfilename)
        # 畫 FFT analysis 的圖
        af.Fourier_plot(data,
                        (fig_save_path + "\\FFT\\MVC"),
                        filename)
        # 畫 bandpass 後之資料圖
        af.plot_plot(bandpass_filtered_data, str(fig_save_path + "\\processing\\smoothing\\" + MVC_folder),
                     filename, "Bandpass_")
        # 畫smoothing 後之資料圖
        af.plot_plot(processing_data, str(fig_save_path + "\\processing\\smoothing\\" + MVC_folder),
                     filename, str(smoothing + "_"))
        # writting data in worksheet
        file_name = processing_folder_path + '\\' + rowdata_folder_list[i] + "\\data\\" + MVC_folder + '\\' + filename + end_name + ".xlsx"
        pd.DataFrame(processing_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    toc = time.process_time()
    print("Total Time:",toc-tic)  
# 找最大值
for i in range(len(rowdata_folder_list)):
    print("To fing the maximum value of all of MVC data in: " + rowdata_folder_list[i])
    tic = time.process_time()
    af.Find_MVC_max(processing_folder_path + '\\' + rowdata_folder_list[i] + "\\data\\" + MVC_folder,
                 processing_folder_path + '\\' + rowdata_folder_list[i])
    toc = time.process_time()
    print("Total Time:",toc-tic)
toc = time.process_time()
print("MVC Data Total Time Spent: ",toc-tic)

# %% 資料前處理 : bandpass filter, absolute value, smoothing, trunkcut data
''' 
處理shooting data
# ----------------------------------------------------------------------------
# 1. 取出所有Raw資料夾
# 2. 獲得Raw folder路徑下的“射箭”資料夾，並讀取所有.cvs file
# 3. 讀取processing folder路徑下的ReleaseTiming，並讀取檔案
# 4. 依序前處理“射箭”資料夾下的檔案
# 4.1 bandpass filting
# 4.2 trunkcut data by release time
# 4.3 依切割檔案計算moving average
# 4.4 輸出moving average to excel file
------------------------------------------------------------------------------
'''
tic = time.process_time()
for i in range(len(rowdata_folder_list)):
    # print(rowdata_folder_list[i])
    # 預處理shooting data
    # for mac version replace "\\" by '/'
    Shooting_path = rowdata_folder_path + '\\' + rowdata_folder_list[i] + "\\" + motion_folder
    Shooting_list = af.Read_File(Shooting_path, '.csv')
    # 儲存照片路徑
    fig_save_path = processing_folder_path + "\\" + rowdata_folder_list[i] + fig_save
    # 讀取all MVC data
    MVC_value = pd.read_excel(processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + rowdata_folder_list[i] + '_all_MVC.xlsx')
    MVC_value = MVC_value.iloc[-1, 2:]
    
    for ii in range(len(Shooting_list)):
        # 印出漂亮的說明
        x = PrettyTable()
        x.field_names = ["平滑方法", "folder", "shooting_file"]
        x.add_row([smoothing, rowdata_folder_list[i], Shooting_list[ii].split('\\')[-1]])
        print(x)
        # 讀取資料
        data = pd.read_csv(Shooting_list[ii], encoding='UTF-8')
        # EMG data 前處理
        processing_data, bandpass_filtered_data = af.EMG_processing(data, smoothing="lowpass")
        # 設定 EMG data 資料儲存路徑
        # 將檔名拆開
        filepath, tempfilename = os.path.split(Shooting_list[ii])
        filename, extension = os.path.splitext(tempfilename)
        # 畫 FFT analysis 的圖
        af.Fourier_plot(data,
                        (fig_save_path + "\\FFT\\motion"),
                        filename)
        # 畫 bandpass 後之資料圖
        af.plot_plot(bandpass_filtered_data, str(fig_save_path + "\\processing\\bandpass\\" + motion_folder),
                     filename, "Bandpass_")
        # 計算 iMVC
        emg_iMVC = pd.DataFrame(np.zeros(np.shape(processing_data)),
                                columns=processing_data.columns)
        emg_iMVC.iloc[:, 0] = processing_data.iloc[:, 0].values
        emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[:, 1:].values),
                                         MVC_value.values)*100

        # 畫前處理後之資料圖
        af.plot_plot(processing_data, str(fig_save_path + "\\processing\\smoothing\\" + motion_folder),
                     filename, str(smoothing + "_"))
    
        # writting data in worksheet
        pd.DataFrame(emg_iMVC).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
toc = time.process_time()
print("Motion Data Total Time Spent: ",toc-tic)


# %%

x = af.Read_File(r"E:\Motion Analysis\U3 Research\S01",
              '.trc', subfolder=False)





















