# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:56:19 2024

1. read data
    1.0. read staging data
    1.1. force plate data
    1.2. motion data
    1.3. EMG data
2. data pre-processing
    2.1. EMG data
    2.2. (?) force plate data
3. draw picture
    3.1. force plate, motion, EMG
    3.2. annotation staging phase
    
@author: Hsin.Yang 10.04.2024
"""
# %% import library
import os
import gc
import time
import numpy
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"E:\Hsin\git\git\Code_testing\baseball\kao")
# sys.path.append(r"D:\BenQ_Project\git\Code_testing\baseball\kao")
import Kao_Function as func

import math
import pandas as pd
import numpy as np
from detecta import detect_onset
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from scipy import signal
from scipy.integrate import cumtrapz, trapz
import logging
from datetime import datetime

# data path
computer_path = r"E:\Hsin\NTSU_lab\data\\"
# computer_path = r"D:\BenQ_Project\python\Kao\\"
data_path = computer_path + "EMG\\"
rawData_folder = "raw_data"
processingData_folder = "processing_data"
MVC_folder = "MVC"
motion_folder = "motion"
fig_save = "figure"
end_name = "_ed"
# parameter setting
smoothing_method = 'lowpass'
samplingRate_motion = 250
# 讀取分期檔
StagingFile_Exist = pd.read_excel(computer_path + "Kao_StagingFile_20240415.xlsx",
                                  sheet_name="工作表2")
# 定義圖片儲存路徑
motion_fig_save = computer_path + "motion_processing\\force_figure\\"
force_data_save = computer_path + "motion_processing\\force_data\\"
emg_figure_save = computer_path + "motion_processing\\EMG\\"
# 定義圖片儲存路徑
folder_path = r"E:\Hsin\NTSU_lab\data\\"
# folder_path = r"D:\BenQ_Project\python\Kao\\"
# 获取当前日期和时间
now = datetime.now()
# 将日期转换为指定格式
formatted_date = now.strftime("%Y-%m-%d-%H%M")

# 输出格式化后的日期
print("当前日期：", formatted_date)
# %% EMG data preprocessing
# 路徑設置
all_rawdata_folder_path = []
all_processing_folder_path = []



rowdata_folder_path = data_path + rawData_folder 
# 去除有“.“開頭的檔案 and 只獲得資料夾路徑，排除其他可能的檔案格式
rowdata_folder_list = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.') \
                       and os.path.isdir(os.path.join(rowdata_folder_path, f))]
        
for i in range(len(rowdata_folder_list)):
    all_rawdata_folder_path.append((data_path + rawData_folder + "\\" \
                                    + "\\" + rowdata_folder_list[i]))
    
    
processing_folder_path = data_path + "\\" + processingData_folder + "\\"
# 去除有“.“開頭的檔案 and 只獲得資料夾路徑，排除其他可能的檔案格式
processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.') \
                          and os.path.isdir(os.path.join(processing_folder_path, f))]
        
for ii in range(len(processing_folder_list)):
    all_processing_folder_path.append((data_path + processingData_folder + "\\" \
                                           + "\\" + processing_folder_list[ii]))
        
del rowdata_folder_list, processing_folder_list
gc.collect(generation=2)

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

    # 先清除所有 processing data 下 MVC 所有的資料    
    for i in range(len(all_processing_folder_path)):
        # 創建儲存資料夾路徑
        folder_list = []
        # 列出所有 processing data 下之所有資料夾
        for dirPath, dirNames, fileNames in os.walk(all_processing_folder_path[i]):
            if os.path.isdir(dirPath):
                folder_list.append(dirPath)
        for ii in folder_list:
        # 清除所有 .xlsx 檔案
            print(ii)
            func.remove_file(ii, ".xlsx")
            func.remove_file(ii, ".jpg")

elif user_input == "N":
    print("取消執行後續。")
else:
    print("無效輸入，請输入 Y 或 N")
toc = time.process_time()
print("刪除檔案總共花費時間: ",toc-tic)

# %% 資料前處理 : bandpass filter, absolute value, smoothing
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

for i in range(len(all_rawdata_folder_path)):
    tic = time.process_time()
    
    MVC_folder_path = all_rawdata_folder_path[i] +  "\\" + MVC_folder
    MVC_list = func.Read_File(MVC_folder_path, ".csv")
    fig_save_path = all_rawdata_folder_path[i].replace(rawData_folder, processingData_folder) \
        + "\\" + fig_save
    print("Now processing MVC data in " + all_rawdata_folder_path[i] + "\\")
    for MVC_path in MVC_list:
        print(MVC_path)
        # 讀取資料
        data = pd.read_csv(MVC_path, encoding='UTF-8')
        # EMG data 前處理
        processing_data, bandpass_filtered_data = func.EMG_processing(MVC_path, smoothing=smoothing_method)
        # 將檔名拆開
        filepath, tempfilename = os.path.split(MVC_path)
        filename, extension = os.path.splitext(tempfilename)
        # 畫 FFT analysis 的圖
        func.Fourier_plot(data,
                        (fig_save_path + "\\FFT\\MVC"),
                        filename)
        # 畫 bandpass 後之資料圖
        func.plot_plot(bandpass_filtered_data, str(fig_save_path + "\\processing\\smoothing\\" + MVC_folder),
                     filename, "Bandpass_")
        # 畫smoothing 後之資料圖
        func.plot_plot(processing_data, str(fig_save_path + "\\processing\\smoothing\\" + MVC_folder),
                     filename, str(smoothing_method + "_"))
        # writting data in worksheet
        file_name =  all_rawdata_folder_path[i].replace(rawData_folder, processingData_folder)\
            + "\\data\\" + MVC_folder + '\\' + filename + end_name + ".xlsx"
        pd.DataFrame(processing_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    
toc = time.process_time()
print("Total Time:",toc-tic)  
gc.collect(generation=2)
        
# 找 MVC 最大值
"""
4. 
"""
for i in range(len(all_processing_folder_path)):
    
    print("To find the maximum value of all of MVC data in: " + all_processing_folder_path[i].split("\\")[-1])
    tic = time.process_time()
    func.Find_MVC_max(all_processing_folder_path[i] + "\\data\\" + MVC_folder,
                      all_processing_folder_path[i])
    toc = time.process_time()
    print("Total Time:",toc-tic)
toc = time.process_time()
print("MVC Data Total Time Spent: ",toc-tic)      
gc.collect(generation=2)


# %% 找出分期檔所在路徑
'''
1. 流程
 1.1. 讀取 staging file，獲得所需要的 EMG、.anc、.force 檔案名稱
 1.2. 獲得時間標的
     1.2.1. trigger off、左腳離地、右腳離地
     1.2.2. 獲取 trigger on、啟動
2. 輸出資料
    2.0. 檔名
    2.1. 啟動時間
    2.2. 左腳最大峰值時間
    2.3. 左腳最大峰值
    2.4. 右腳最大峰值時間
    2.5. 右腳最大峰值
3. 繪圖
    3.1. 5*2個子圖，分別為左、右腳力版，以及各自的 EMG data
    3.2. 時間從 trigger off 到右腳離地 
'''


# 1. read staging file
# StagingFile_Exist = pd.read_excel(r"D:\BenQ_Project\python\Kao\Kao_StagingFile.xlsx",
#                              sheet_name="工作表2")
# 2. read EMG data
# 獲得所有格式 data 的路徑

anc_file_list = func.Read_File(folder_path, '.anc', subfolder=True)
force_file_list = func.Read_File(folder_path, '.forces', subfolder=True)
emg_file_list = func.Read_File(folder_path, '.csv', subfolder=True)
true_anc_file = []
true_force_file = []
true_emg_file = []

for file_name in range(np.shape(StagingFile_Exist)[0]):
    print(StagingFile_Exist.loc[file_name, 'MotionFileName'])
    for emg_path in emg_file_list:
        # print(file_path)
        if StagingFile_Exist.loc[file_name, 'EMG_Name'] in emg_path:
            print(emg_path)
            true_emg_file.append(emg_path)
    for anc_path in anc_file_list:
        print(anc_path)
        if StagingFile_Exist.loc[file_name, '.anc'] in anc_path and 'GRF' in anc_path:
            true_anc_file.append(anc_path)
            print(anc_path)
    print(str(StagingFile_Exist.loc[file_name, '.anc'] + ".forces"))
    for force_path in force_file_list:
        # 
        if str(StagingFile_Exist.loc[file_name, '.anc'] + ".forces") in force_path and 'GRF' in force_path:
            # print(str(StagingFile_Exist.loc[file_name, 'MotionFileName'] + ".forces"))
            true_force_file.append(force_path)
# %% 找力版時間

# 定義資料儲存位置
data_table = pd.DataFrame({}, columns = ['filename', 'order', '左腳離地時間', '左腳最大值時間',
                                         '左腳最大值', '左腳啟動時間', '左腳啟動力量',
                                         'Left_RFD', '',
                                         '右腳離地時間', '右腳最大值時間', '右腳最大值', 
                                          '右腳發力時間_g', '右腳發力時間_r', '右腳發力值_g', '右腳發力值_r',
                                         'Right_RFD_g', 'Right_RFD_r', 'Right_DRFD_g', 'Right_DRFD_r',
                                         'onset time']

                          )
emg_data_table = pd.DataFrame({}, columns = ['task','trial', 'time', 'L RECTUS FEMORIS: EMG 1', 'L VASTUS LATERALIS: EMG 2',
                                             'L BICEPS FEMORIS: EMG 3', 'L SEMITENDINOSUS: EMG 4',
                                             'L GASTROCNEMIUS MEDIAL HEAD: EMG 5', 'R RECTUS FEMORIS: EMG 6',
                                             'R VASTUS LATERALIS: EMG 7', 'R BICEPS FEMORIS: EMG 8',
                                             'R SEMITENDINOSUS: EMG 9', 'R GASTROCNEMIUS MEDIAL HEAD: EMG 10']
                              )

for file_name in range(np.shape(StagingFile_Exist)[0]):
    # 1. 找出 anc, force, emg 的檔案路徑
    for emg_path in emg_file_list:
        if StagingFile_Exist.loc[file_name, 'EMG_Name'] in emg_path:
            print(emg_path)
            read_emg = emg_path
            break
    for anc_path in anc_file_list:
        if StagingFile_Exist.loc[file_name, '.anc'] in anc_path and 'GRF' in anc_path:
            read_anc = anc_path
            print(anc_path)
            break
    for force_path in force_file_list:
        if str(StagingFile_Exist.loc[file_name, '.anc'] + ".forces") in force_path and 'GRF' in force_path:
            print(force_path)
            read_force = force_path
            break
    # -------------------------------------------------------------------------
    # 2. read .anc and .force data 
    analog_data = pd.read_csv(read_anc,
                              skiprows=8, delimiter = '	', low_memory=False).iloc[2:, :]
    forcePlate_data = pd.read_csv(read_force,
                                  skiprows=4, delimiter = '	')

    # 2.2. 設定左右腳力版
    rightLeg_FP2 = forcePlate_data.loc[:, ['#Sample', 'FX2', 'FY2', 'FZ2']]
    leftLeg_FP1 = forcePlate_data.loc[:, ['#Sample', 'FX1', 'FY1', 'FZ1']]
    # -------------------------------------------------------------------------
    # 3. 找出分期檔的時間，以及其他要處理的時間點truncate data
    time_TriggerOff = StagingFile_Exist.loc[file_name, '起點trigger off']
    time_start = StagingFile_Exist.loc[file_name, '啟動'] # 變成抓左腳力版變化
    time_LeftOff = StagingFile_Exist.loc[file_name, '左腳離地']
    time_RightOff = StagingFile_Exist.loc[file_name, '右腳離地']
    # 抓 trigger on 的時間
    onset_analog = detect_onset(analog_data.loc[:, ['C63']].values.reshape(-1),
                                np.mean(analog_data.loc[:50, ['C63']].values)*1.1,
                                n_above=10, n_below=10, show=True)
    if not np.isnan(time_TriggerOff):
        time_TriggerOff = int(time_TriggerOff)
        # time_TriggerOff = onset_analog[0, 1]
        # 先找最大衝力時間點，再往前推
        
        # 3.1. 計算所需資料
        # 抓反應時間: time_start - trigger off 
        # reaction_time = time_start - time_TriggerOff  
        # 先濾波力板資料
        
        left_lowpass = pd.DataFrame(np.zeros(np.shape(leftLeg_FP1)),
                                    columns = leftLeg_FP1.columns)
        right_lowpass = pd.DataFrame(np.zeros(np.shape(rightLeg_FP2)),
                                    columns = rightLeg_FP2.columns)
        # 定義濾波參數
        lowpass_sos = signal.butter(2, 50, btype='low', fs=samplingRate_motion, output='sos')
        for i in range(np.shape(left_lowpass)[1]-1):    
            left_lowpass.iloc[:, i+1] = signal.sosfiltfilt(lowpass_sos,
                                                           np.transpose(leftLeg_FP1.iloc[:, i+1].values))
            right_lowpass.iloc[:, i+1] = signal.sosfiltfilt(lowpass_sos,
                                                           np.transpose(rightLeg_FP2.iloc[:, i+1].values))
        # 定義時間區段
        left_lowpass['#Sample'] = leftLeg_FP1.iloc[:, 0]
        right_lowpass['#Sample'] = rightLeg_FP2.iloc[:, 0]
        
        # 計算力版三軸合力
        combin_left = np.sqrt((left_lowpass.iloc[:, 1].values)**2 + \
                              (left_lowpass.iloc[:, 2].values)**2 + \
                              (left_lowpass.iloc[:, 3].values)**2)
        combin_right = np.sqrt((right_lowpass.iloc[:, 1].values)**2 + \
                              (right_lowpass.iloc[:, 2].values)**2 + \
                              (right_lowpass.iloc[:, 3].values)**2)
        
        # 從啟動開始，往前推0.3秒，找左腳力板變化正負10%
        # 因為不是所有人都有time start 所以改為 trigger off
        # analog time 為 samplingRate_motion*10
        ana_time_start = int(time_TriggerOff*10-samplingRate_motion*10*0.3)
        pa_start_onset = detect_onset(combin_left[time_TriggerOff*10:], # 將資料轉成一維型態
                                      np.mean(combin_left[ana_time_start-100:ana_time_start])*1.1,
                                      n_above=10, n_below=2, show=True)
        # 找第二峰值
        sec_left_onset = detect_onset(combin_left[time_TriggerOff*10:]*-1, # 將資料轉成一維型態
                                      np.mean(combin_left[ana_time_start-100:ana_time_start])*-0.9,
                                      n_above=10, n_below=2, show=True)
        sec_left_max = (combin_left[time_TriggerOff*10 + sec_left_onset[0, 0]: \
                                   time_TriggerOff*10 + sec_left_onset[0, 1]]*-1).argmax() + sec_left_onset[0, 0]
        # 找到左腳離地時間
        leftLeg_off = np.where(combin_left[ana_time_start:] < 10)[0][0]
        # 抓取左腳"最大峰值"及"峰值時間"
        left_max = combin_left[ana_time_start:].max()
        left_max_time = combin_left[ana_time_start:].argmax() + ana_time_start + 1 # 因為 python index 會少 1
        # 使用左腳合力，找到動作開始時間
        if sec_left_max > pa_start_onset[0, 0] and sec_left_max - pa_start_onset[0, 0] < 500:
            time_start = sec_left_max + time_TriggerOff*10 
        else:
            time_start = pa_start_onset[0, 0] + time_TriggerOff*10
        # 找右腳發力時間
        # 先把太小的數字都替換成 0
        combin_right[combin_right < 10**-20] = 0
        # 先找 onset 的時間段，再找最大值
        right_start_onset = detect_onset(combin_right[time_TriggerOff*10:]*-1, # 將資料轉成一維型態
                                      np.mean(combin_right[ana_time_start-100:ana_time_start])*1.1*-1,
                                      n_above=10, n_below=2, show=True)
        first_right_max_value = np.max((combin_right[time_TriggerOff*10:time_TriggerOff*10+right_start_onset[0, 1]]*-1))
        # 使用斜率判定
        slope_right = combin_right[time_TriggerOff*10+1:time_TriggerOff*10+right_start_onset[0, 1]] - \
            combin_right[time_TriggerOff*10:time_TriggerOff*10+right_start_onset[0, 1]-1]
        lowpass_sos = signal.butter(2, 6, btype='low', fs=samplingRate_motion, output='sos')
            
        slope_right = signal.sosfiltfilt(lowpass_sos, slope_right)
        slope_idx = np.argmax(slope_right > 2) + time_TriggerOff*10 
                                                              
        plt.plot(slope_right)
        # 如果有找到右腳離開力板的時間，則去找最後離開時間，不然就直接去找該段時間內的最小值
        if first_right_max_value == 0:
            first_right_max_idx = np.where((combin_right[time_TriggerOff*10:time_TriggerOff*10+right_start_onset[0, 1]]*-1) \
                                           == first_right_max_value)[0]
            first_right_max_idx = first_right_max_idx[-1] + time_TriggerOff*10
        else:
            first_right_max_idx = time_TriggerOff*10 + \
                (combin_right[time_TriggerOff*10:time_TriggerOff*10+right_start_onset[0, 1]]*-1).argmax()
        

        # 抓取右腳"最大峰值"及"峰值時間"
        right_max = combin_right[first_right_max_idx:].max()
        right_time = combin_right[first_right_max_idx:].argmax() + first_right_max_idx
        # 找到右腳離地時間，由於有右腳離地的情形，所以時間開始從最大值以後開始算
        rightLeg_off = np.where(combin_right[right_time:] < 10)[0][0]
        # 繪一張合力圖
        x_data = pd.Series(analog_data.loc[:, 'Name'])
        y_data = pd.Series(analog_data.loc[:, 'C63'])
        # force plate 繪圖
        fig, axs = plt.subplots(3, 1, figsize=(7, 8))
        # 子圖一
        # 只畫動作開始前1秒，到右腳離地後0.5秒
        axs[0].plot(left_lowpass.iloc[time_start-2500:rightLeg_off + right_time+1250, 0],
                    combin_left[time_start-2500:rightLeg_off + right_time+1250])
        axs[0].axvline(time_TriggerOff*10, color='red', linestyle='--', linewidth=0.5) # trigger off
        axs[0].plot(leftLeg_off + ana_time_start, combin_left[leftLeg_off + ana_time_start], # 找左腳離地時間
                    marker = 'o', ms = 10, mec='c', mfc='none')
        axs[0].plot(time_start, combin_left[time_start], # 找啟動時間點
                    marker = '*', color = 'r')
        axs[0].plot(left_max_time, combin_left[left_max_time], # 找最大值
                    marker = 'o', ms = 10, mec='b', mfc='none')
        axs[0].set_title('ForcePlare 1: Left Leg')
        axs[0].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
        # 子圖二
        # 只畫動作開始前1秒，到右腳離地後0.5秒
        axs[1].plot(right_lowpass.iloc[time_start-2500:rightLeg_off + right_time+1250, 0],
                    combin_right[time_start-2500:rightLeg_off + right_time+1250])
        axs[1].axvline(time_TriggerOff*10, color='red', linestyle='--', linewidth=0.5) # trigger off
        axs[1].plot(first_right_max_idx, combin_right[first_right_max_idx], # 右腳發力時間
                    marker = 'o', ms = 10, mec='lime', mfc='none')
        axs[1].plot(slope_idx, combin_right[slope_idx], # 右腳發力時間
                    marker = 'o', ms = 10, mec='r', mfc='none')
        axs[1].plot(rightLeg_off + right_time, combin_right[rightLeg_off + right_time], # 右腳離地時間
                    marker = 'o', ms = 10, mec='c', mfc='none')
        axs[1].plot(right_time, combin_right[right_time], # 右腳力版最大值
                    marker = 'o', ms = 10, mec='b', mfc='none')
        axs[1].set_title('ForcePlare 2: Right Leg')
        axs[1].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
        # 子圖三
        axs[2].plot(right_lowpass.iloc[:, 0], y_data)
        if np.size(onset_analog) > 0:
            axs[2].axvline(onset_analog[0, 0], color='red', linestyle='--', linewidth=0.5) # trigger off
        else:
            axs[2].annotate('Can not find onset', xy = (0, 0), fontsize = 16, color='r')
        axs[2].axvline(time_TriggerOff*10, color='red', linestyle='--', linewidth=0.5) # trigger off
        axs[2].set_title('Trigger C63')
        # 設定科學符號 : 小數點後幾位數
        axs[2].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
        # 
        plt.suptitle(str("Force Plate: " + StagingFile_Exist.loc[file_name, '.anc']), fontsize = 16)
        plt.tight_layout()
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.grid(False)
        plt.xlabel("time (second)", fontsize = 14)
        plt.ylabel("Force (N)", fontsize = 14)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.savefig(str(motion_fig_save + StagingFile_Exist.loc[file_name, '.anc'] + "_PF.jpg"),
                    dpi=200, bbox_inches = "tight")
        plt.show()
        # 輸出力版資料
        left_data = pd.DataFrame({'#Sample':left_lowpass.iloc[time_start:leftLeg_off + ana_time_start, 0],
                                  'combine_left':combin_left[time_start:leftLeg_off + ana_time_start]})
        right_data_g = pd.DataFrame({'#Sample':right_lowpass.iloc[first_right_max_idx:rightLeg_off + right_time, 0],
                                     'combine_right_g':combin_right[first_right_max_idx:rightLeg_off + right_time]})
        right_data_r = pd.DataFrame({'#Sample':right_lowpass.iloc[slope_idx:rightLeg_off + right_time, 0],
                                     'combine_right_g':combin_right[slope_idx:rightLeg_off + right_time]})
        # 將 force plate data 資料寫進 EXCEL
        # if not os.path.exists(force_data_save):
        #     os.makedirs(force_data_save)
        save_file_name = os.path.join(force_data_save + StagingFile_Exist.loc[file_name, '.anc'] + "_PF.xlsx")
        
        with pd.ExcelWriter(save_file_name) as Writer:
            left_data.to_excel(Writer, sheet_name="Stage1", index=False)
            right_data_g.to_excel(Writer, sheet_name="Stage2_g", index=False)
            right_data_r.to_excel(Writer, sheet_name="Stage2_r", index=False)
        
        # ---------------處理 EMG data--------------------------------------
        # 2.1. read EMG data and pre-processing data
        '''
        1. 需要輸出 FFT，檢查資料
        '''
        # 處理EMG data 必須要有 onset data
        if np.size(onset_analog) > 0:
            # 1. EMG 前處理 ---------------------------------------------------
            EMG_data = pd.read_csv(read_emg)
            processing_data, bandpass_filtered_data = func.EMG_processing(read_emg, smoothing='lowpass')
            # 計算iMVC
            # 讀取 all MVC data
            parent_dir = os.path.dirname(read_emg.replace("raw_data", "processing_data")).replace("motion", "")
            MVC_value = pd.read_excel(parent_dir + parent_dir.split("\\")[-2] + '_all_MVC.xlsx')
            # 只取 all MVC data 數字部分
            MVC_value = MVC_value.iloc[-1, 2:]
            # 計算 iMVC，分別為 processing data and bandpass data
            bandpass_iMVC = pd.DataFrame(np.empty(np.shape(bandpass_filtered_data)),
                                         columns=bandpass_filtered_data.columns)
            # 取得時間
            bandpass_iMVC.iloc[:, 0] = bandpass_filtered_data.iloc[:, 0].values
            # 除以 MVC 最大值
            bandpass_iMVC.iloc[:, 1:] = np.divide(abs(bandpass_filtered_data.iloc[:, 1:].values),
                                                  MVC_value.values)*100
            # processing data
            processing_iMVC = pd.DataFrame(np.empty(np.shape(processing_data)),
                                           columns=processing_data.columns)
            # 取得時間
            processing_iMVC.iloc[:, 0] = processing_data.iloc[:, 0].values
            # 除以 MVC 最大值
            processing_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[:, 1:].values),
                                                    MVC_value.values)*100
            # EMG 與 motion 時間換算
            motion_start = int((time_start - onset_analog[0, 0])/2500*2000)
            leftLeave = int((leftLeg_off + ana_time_start - onset_analog[0, 0])/2500*2000)
            right_start_g = int((first_right_max_idx - onset_analog[0, 0])/2500*2000)
            right_start_r = int((slope_idx - onset_analog[0, 0])/2500*2000)
            rightLeave = int((rightLeg_off + right_time - onset_analog[0, 0])/2500*2000)
            
            # 2. EMG data 計算 -------------------------------------------------
            # 2.1. 計算積分面積
            temp_stage1_Atrap = pd.DataFrame([1/2000*trapz(bandpass_iMVC.iloc[motion_start:leftLeave, :], axis=0)],
                                             columns=bandpass_iMVC.columns[:])
            temp_stage2_g_Atrap = pd.DataFrame([1/2000*trapz(bandpass_iMVC.iloc[right_start_g:rightLeave, :], axis=0)], # 右腳啟動綠色
                                               columns=bandpass_iMVC.columns[:])
            temp_stage2_r_Atrap = pd.DataFrame([1/2000*trapz(bandpass_iMVC.iloc[right_start_r:rightLeave, :], axis=0)], #右腳啟動紅色
                                               columns=bandpass_iMVC.columns[:])
            # 2024.05.23 新增計算推蹬前期以及推蹬後期
            stage1_mid = motion_start + int((leftLeave - motion_start)/2)
            stage2_mid_g = right_start_g + int((rightLeave - right_start_g)/2)
            # stage2_mid_r = right_start_r + int((rightLeave - right_start_r)/2)
            # 計算積分面積
            # stage 1
            temp_stage1_first_Atrap = pd.DataFrame([1/2000*trapz(bandpass_iMVC.iloc[motion_start:stage1_mid, :], axis=0)],
                                                   columns=bandpass_iMVC.columns[:])
            temp_stage1_second_Atrap = pd.DataFrame([1/2000*trapz(bandpass_iMVC.iloc[stage1_mid:leftLeave, :], axis=0)],
                                                   columns=bandpass_iMVC.columns[:])
            # stage 2 green
            temp_stage2_g_first_Atrap = pd.DataFrame([1/2000*trapz(bandpass_iMVC.iloc[right_start_g:stage2_mid_g, :], axis=0)], # 右腳啟動綠色
                                                     columns=bandpass_iMVC.columns[:])
            temp_stage2_g_second_Atrap = pd.DataFrame([1/2000*trapz(bandpass_iMVC.iloc[stage2_mid_g:rightLeave, :], axis=0)], # 右腳啟動綠色
                                                      columns=bandpass_iMVC.columns[:])
            # stage 2 red
            # temp_stage2_r_first_Atrap = pd.DataFrame([1/2000*trapz(bandpass_iMVC.iloc[right_start_r:stage2_mid_r, :], axis=0)], #右腳啟動紅色
            #                                          columns=bandpass_iMVC.columns[:])
            # temp_stage2_r_second_Atrap = pd.DataFrame([1/2000*trapz(bandpass_iMVC.iloc[stage2_mid_r:rightLeave, :], axis=0)], #右腳啟動紅色
            #                                           columns=bandpass_iMVC.columns[:])
            # 資料儲存, 插入 task 名稱以做區隔
            temp_stage1_Atrap.insert(0, 'task', 'stage1 intergated')
            temp_stage1_Atrap.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            temp_stage2_g_Atrap.insert(0, 'task', 'stage2 intergated_g')
            temp_stage2_g_Atrap.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            # temp_stage2_r_Atrap.insert(0, 'task', 'stage2 intergated_r')
            # temp_stage2_r_Atrap.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            # 新增計算
            temp_stage1_first_Atrap.insert(0, 'task', 'stage1 intergated first half')
            temp_stage1_first_Atrap.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            temp_stage1_second_Atrap.insert(0, 'task', 'stage1 intergated second half')
            temp_stage1_second_Atrap.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            temp_stage2_g_first_Atrap.insert(0, 'task', 'stage2 intergated_g first half')
            temp_stage2_g_first_Atrap.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            temp_stage2_g_second_Atrap.insert(0, 'task', 'stage2 intergated_g second half')
            temp_stage2_g_second_Atrap.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            # temp_stage2_r_first_Atrap.insert(0, 'task', 'stage2 intergated_r first half')
            # temp_stage2_r_first_Atrap.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            # temp_stage2_r_second_Atrap.insert(0, 'task', 'stage2 intergated_r second half')
            # temp_stage2_r_second_Atrap.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            # 2.2. 計算移動平均
            moving_process_iMVC = pd.DataFrame(np.empty(np.shape(processing_iMVC)), # 創建資料儲存位置
                                               columns = processing_iMVC.columns)
            moving_process_iMVC.iloc[:, 0] = processing_iMVC.iloc[:, 0] # 定義時間
            for i in range(np.shape(processing_iMVC)[1]-1):
                moving_process_iMVC.iloc[:, i+1] = processing_iMVC.iloc[:, i+1].rolling(int(0.05*2000)).mean()
            # 2.3. 找到兩個stage的最大值
            temp_stage1_max = pd.DataFrame([moving_process_iMVC.iloc[motion_start:leftLeave, :].max()],
                                           columns = moving_process_iMVC.columns)
            temp_stage2_max = pd.DataFrame([moving_process_iMVC.iloc[right_start_g:rightLeave, :].max()],
                                           columns = moving_process_iMVC.columns)
            # 2024.05.23 新增計算推蹬前期以及推蹬後期
            temp_stage1_first_max = pd.DataFrame([moving_process_iMVC.iloc[motion_start:stage1_mid, :].max()],
                                                 columns = moving_process_iMVC.columns)
            temp_stage1_second_max = pd.DataFrame([moving_process_iMVC.iloc[stage1_mid:leftLeave, :].max()],
                                                  columns = moving_process_iMVC.columns)
            temp_stage2_first_max = pd.DataFrame([moving_process_iMVC.iloc[right_start_g:stage2_mid_g, :].max()],
                                                 columns = moving_process_iMVC.columns)
            temp_stage2_second_max = pd.DataFrame([moving_process_iMVC.iloc[stage2_mid_g:rightLeave, :].max()],
                                                  columns = moving_process_iMVC.columns)
            # 資料儲存, 插入 task 名稱以做區隔
            temp_stage1_max.insert(0, 'task', 'stage1 max')
            temp_stage1_max.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            temp_stage2_max.insert(0, 'task', 'stage2 max')
            temp_stage2_max.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            # 新增資料
            temp_stage1_first_max.insert(0, 'task', 'stage1 max first half')
            temp_stage1_first_max.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            temp_stage1_second_max.insert(0, 'task', 'stage1 max second half')
            temp_stage1_second_max.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            temp_stage2_first_max.insert(0, 'task', 'stage2 max first half')
            temp_stage2_first_max.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            temp_stage2_second_max.insert(0, 'task', 'stage2 max second half')
            temp_stage2_second_max.insert(1, 'trial', StagingFile_Exist.loc[file_name, 'EMG_Name'])
            # 合併計算資料
            add_emg_statics = pd.concat([temp_stage1_Atrap, temp_stage2_g_Atrap,
                                         temp_stage1_first_Atrap, temp_stage1_second_Atrap,
                                         temp_stage2_g_first_Atrap, temp_stage2_g_second_Atrap,
                                         temp_stage1_max, temp_stage2_max,
                                         temp_stage1_first_max, temp_stage1_second_max,
                                         temp_stage2_first_max, temp_stage2_second_max])
            # 3. 將資料寫進 EXCEL -----------------------------------------------
            save_file_name = os.path.dirname(read_emg.replace("raw_data", "processing_data").replace("motion", "data\\motion")) + \
                "\\" + StagingFile_Exist.loc[file_name, 'EMG_Name'] + "_ed.xlsx"
            with pd.ExcelWriter(save_file_name) as Writer:
                processing_iMVC.iloc[motion_start:leftLeave].to_excel(Writer, sheet_name="Stage1", index=False)
                processing_iMVC.iloc[right_start_g:rightLeave].to_excel(Writer, sheet_name="Stage2_g", index=False)
                processing_iMVC.iloc[right_start_r:rightLeave].to_excel(Writer, sheet_name="Stage2_r", index=False)
            
            # 4. EMG 繪圖 --------------------------------------------------------
            # 設置資料儲存路徑 JPG
            filepath_fig = os.path.dirname(read_emg.replace("raw_data", "processing_data")\
                                           .replace("motion", r"figure\processing\smoothing\motion"))
            save_fig = filepath_fig + "\\" + StagingFile_Exist.loc[file_name, 'EMG_Name'] + "_Bandpass.jpg"
            # 設定子圖數量
            n = int(math.ceil((np.shape(bandpass_filtered_data)[1] - 1) /2)) + 1
            fig, axs = plt.subplots(n, 2, figsize = ((2*n+1,10)), sharex=False)
            for i in range(np.shape(bandpass_filtered_data)[1]+1):
                xx, yy = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
                # print(i-1*yy)
                # print(xx ,yy)
                if xx == 0 and yy == 0:
                    # 繪製左腳力版圖
                    axs[xx, yy].plot(left_lowpass.iloc[time_start-2500:rightLeg_off + right_time+1250, 0],
                                     combin_left[time_start-2500:rightLeg_off + right_time+1250])
                    axs[xx, yy].axvline(time_TriggerOff*10, color='red', linestyle='--') # trigger off
                    axs[xx, yy].plot(leftLeg_off + ana_time_start, combin_left[leftLeg_off + ana_time_start], # 找左腳離地時間
                                     marker = 'o', ms = 10, mec='c', mfc='none')
                    axs[xx, yy].plot(time_start, combin_left[time_start], # 找啟動時間點
                                     marker = '*', color = 'r')
                    axs[xx, yy].plot(left_max_time, combin_left[left_max_time], # 找最大值
                                     marker = 'o', ms = 10, mec='b', mfc='none')
                    axs[xx, yy].set_title('ForcePlare 1: Left Leg', fontsize=16)
                    axs[xx, yy].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                elif xx == 0 and yy == 1:
                    # 子圖二
                    axs[xx, yy].plot(right_lowpass.iloc[time_start-2500:rightLeg_off + right_time+1250, 0],
                                     combin_right[time_start-2500:rightLeg_off + right_time+1250])
                    axs[xx, yy].axvline(time_TriggerOff*10, color='red', linestyle='--') # trigger off
                    axs[xx, yy].plot(first_right_max_idx, combin_right[first_right_max_idx], # 右腳發力時間
                                marker = 'o', ms = 10, mec='lime', mfc='none')
                    axs[xx, yy].plot(rightLeg_off + right_time, combin_right[rightLeg_off + right_time], # 右腳離地時間
                                marker = 'o', ms = 10, mec='c', mfc='none')
                    axs[xx, yy].plot(right_time, combin_right[right_time], # 右腳力版最大值
                                marker = 'o', ms = 10, mec='b', mfc='none')
                    axs[xx, yy].set_title('ForcePlare 2: Right Leg', fontsize=16)
                    axs[xx, yy].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                else:
                    # 設定子圖之參數
                    # EMG 繪圖從 啟動前一秒開始畫，到右腳離地後0.5秒
                    axs[xx, yy].plot(processing_iMVC.iloc[motion_start-2000:rightLeave+1000, 0],
                                     processing_iMVC.iloc[motion_start-2000:rightLeave+1000, i-1*yy])
                    axs[xx, yy].set_title(processing_iMVC.columns[i-1*yy], fontsize=16)
                    # 設定科學符號 : 小數點後幾位數
                    axs[xx, yy].axvline(motion_start/2000, color='r', linestyle='--')
                    axs[xx, yy].axvline(leftLeave/2000, color='r', linestyle='--')
                    axs[xx, yy].axvline(right_start_g/2000, color='c', linestyle='--')
                    axs[xx, yy].axvline(rightLeave/2000, color='c', linestyle='--')
                    axs[xx, yy].axvline(stage1_mid/2000, color='lightcoral', linestyle='--')
                    axs[xx, yy].axvline(stage2_mid_g/2000, color='lightskyblue', linestyle='--')

                    """
                    2024.05.23 目前改到這裡
                    """
                    # 2024.05.23 新增, 將每個階段找到的最大值圈起來
                    if len(processing_iMVC.iloc[motion_start:stage1_mid, i-1*yy]) != 0:
                        axs[xx, yy].plot(processing_iMVC.iloc[processing_iMVC.iloc[motion_start:stage1_mid, i-1*yy].idxmax(), 0],
                                         processing_iMVC.iloc[motion_start:stage1_mid, i-1*yy].max(), # 右腳力版最大值
                                         marker = 'o', ms = 10, mec='b', mfc='none')
                        axs[xx, yy].plot(processing_iMVC.iloc[processing_iMVC.iloc[stage1_mid:leftLeave, i-1*yy].idxmax(), 0],
                                         processing_iMVC.iloc[stage1_mid:leftLeave, i-1*yy].max(), # 右腳力版最大值
                                         marker = 'o', ms = 10, mec='b', mfc='none')
                        axs[xx, yy].plot(processing_iMVC.iloc[processing_iMVC.iloc[right_start_g:stage2_mid_g, i-1*yy].idxmax(), 0],
                                         processing_iMVC.iloc[right_start_g:stage2_mid_g, i-1*yy].max(), # 右腳力版最大值
                                         marker = 'o', ms = 10, mec='b', mfc='none')
                        axs[xx, yy].plot(processing_iMVC.iloc[processing_iMVC.iloc[stage2_mid_g:rightLeave, i-1*yy].idxmax(), 0],
                                         processing_iMVC.iloc[stage2_mid_g:rightLeave, i-1*yy].max(), # 右腳力版最大值
                                         marker = 'o', ms = 10, mec='b', mfc='none')
                        
                    axs[xx, yy].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
            # 設定整張圖片之參數
            # plt.suptitle(StagingFile_Exist.loc[file_name, '.anc'] + "_Bandpass", fontsize = 16)
            plt.suptitle(StagingFile_Exist.loc[file_name, ".anc"] + '\n' 
                         + StagingFile_Exist.loc[file_name, "EMG_Name"], fontsize=16)
            plt.tight_layout()
            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.grid(False)
            plt.xlabel("time (second)", fontsize = 14)
            plt.ylabel("Muscle Activation (%)", fontsize = 14)
            plt.savefig(str(emg_figure_save + StagingFile_Exist.loc[file_name, '.anc'] + "_emg.jpg"),
                            dpi=200, bbox_inches = "tight")
            plt.show()
        # -----------------儲存資料-------------------------------
        # 儲存 EMG table data
        emg_data_table = pd.concat([emg_data_table, add_emg_statics],
                                   ignore_index=True)
        if np.size(onset_analog) > 0:
            onset = onset_analog[0, 0]
        else:
            onset = np.nan
        # RFD = (left force max - left start force)/(left_max_time - time_start)/2500
        data_table = pd.concat([data_table, pd.DataFrame({'filename' : StagingFile_Exist.loc[file_name, '.anc'],
                                                          'order' : StagingFile_Exist.loc[file_name, 'order'],
                                                          '左腳離地時間': [leftLeg_off + ana_time_start],
                                                          '左腳最大值時間': [left_max_time],
                                                          '左腳最大值': [combin_left[left_max_time]],
                                                          '左腳啟動時間': [time_start],
                                                          '左腳啟動力量': [combin_left[time_start]],
                                                          'Left_RFD': [(combin_left[left_max_time] - combin_left[time_start])/ \
                                                                       ((left_max_time - time_start)/2500)],
                                                        'Left_DRFD': [(combin_left[int(time_start+2500*0.1)] - combin_right[time_start])/ \
                                                                     ((2500*0.1)/2500)],
                                                          '右腳離地時間': [rightLeg_off + right_time],
                                                          '右腳最大值': [combin_right[right_time]],
                                                          '右腳最大值時間':[right_time],
                                                        '右腳發力時間_g':[first_right_max_idx],
                                                        '右腳發力時間_r':[slope_idx],
                                                        '右腳發力值_g':[combin_right[first_right_max_idx]],
                                                        '右腳發力值_r':[combin_right[slope_idx]],
                                                        'Right_RFD_g':[(combin_right[right_time] - combin_right[first_right_max_idx])/ \
                                                                     ((right_time - first_right_max_idx)/2500)],
                                                        'Right_RFD_r':[(combin_right[right_time] - combin_right[slope_idx])/ \
                                                                        ((right_time - slope_idx)/2500)],
                                                        'Right_DRFD_g':[(combin_right[int(first_right_max_idx+2500*0.1)] - combin_right[first_right_max_idx])/ \
                                                                     ((2500*0.1)/2500)],
                                                        'Right_DRFD_r':[(combin_right[int(slope_idx+2500*0.1)] - combin_right[slope_idx])/ \
                                                                       ((2500*0.1)/2500)],
                                                        'onset time': [onset]
                                                          }, index=[0])],
                               ignore_index=True)
            
# 將資料輸出成 EXCEL TABLE
data_table.to_excel(computer_path + "motion_statistic_" + formatted_date + ".xlsx")
emg_data_table.to_excel(computer_path + "emg_data_statistic_" + formatted_date + ".xlsx")




# %% 計算 iMVC : trunkcut data and caculate iMVC value
''' 
處理shooting data
# ----------------------------------------------------------------------------
1. 取出所有Raw資料夾
2. 獲得 Raw folder -> motion -> ReadFile ".csv"
3. 讀取 processing folder 路徑下的 SXX_all_MVC
4. pre-processing emg data
5. 找 staging file 中對應的檔案名稱
    5.1 擷取時間點
    5.2 切割資料
6. 輸出資料
------------------------------------------------------------------------------
'''
# 配置日志记录
# logging.basicConfig(filename=r'D:\BenQ_Project\python\Kao\processing_data\example.log',
#                     level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




















