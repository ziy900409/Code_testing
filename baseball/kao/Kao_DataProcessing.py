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
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"D:\BenQ_Project\python\Kao\code")
import Kao_Function as func

import math
import pandas as pd
import numpy as np
from detecta import detect_onset
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from scipy import signal
import logging

# data path
data_path = r"D:\BenQ_Project\python\Kao\EMG\\"
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
StagingFile_Exist = pd.read_excel(r"D:\BenQ_Project\python\Kao\Kao_StagingFile.xlsx",
                                  sheet_name="工作表4")
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
        processing_data, bandpass_filtered_data = func.EMG_processing(data, smoothing=smoothing_method)
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
    
    
    # 預處理shooting data
    # for mac version replace "\\" by '/'
    Shooting_path = all_rawdata_folder_path[i] + "\\" + motion_folder
    Shooting_list = func.Read_File(Shooting_path, '.csv')
    for ii in range(len(Shooting_list)):
        # 印出說明
        x = PrettyTable()
        x.field_names = ["平滑方法", "folder", "shooting_file"]
        x.add_row([smoothing_method, all_rawdata_folder_path[i].split("\\")[-1],
                   Shooting_list[ii].split('\\')[-1]])
        print(x)
        # 讀取資料
        data = pd.read_csv(Shooting_list[ii], encoding='UTF-8')
        # EMG data 前處理
        processing_data, bandpass_filtered_data = func.EMG_processing(data, smoothing="lowpass")
        # 設定 EMG data 資料儲存路徑
        # 將檔名拆開
        filepath, tempfilename = os.path.split(Shooting_list[ii])
        filename, extension = os.path.splitext(tempfilename)
        # 畫 FFT analysis 的圖
        func.Fourier_plot(data,
                        (fig_save_path + "\\FFT\\motion"),
                        filename)
        # 畫 bandpass 後之資料圖
        func.plot_plot(bandpass_filtered_data, str(fig_save_path + "\\processing\\bandpass\\" + motion_folder),
                     filename, "Bandpass_")
        # 畫前處理後之資料圖
        func.plot_plot(processing_data, str(fig_save_path + "\\processing\\smoothing\\" + motion_folder),
                     filename, str("_" + smoothing_method))
toc = time.process_time()
print("Total Time:",toc-tic)  
gc.collect(generation=2)
        
# %% 找 MVC 最大值
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
logging.basicConfig(filename=r'D:\BenQ_Project\python\Kao\processing_data\example.log',
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 记录信息
tic = time.process_time()

# a = []
# 開始處理 motion 資料
for i in range(len(all_rawdata_folder_path)):
    
    # 讀取路徑下所有的 shooting motion file
    emg_folder_path = all_rawdata_folder_path[i] + "\\" + motion_folder
    anc_folder_path = all_rawdata_folder_path[i].replace("EMG\\\\raw_data", "GRF")
    emg_list = func.Read_File(emg_folder_path, '.csv')
    anc_list = func.Read_File(anc_folder_path, '.anc')
    # 設定儲存照片路徑
    fig_save_path = all_rawdata_folder_path[i].replace(rawData_folder, "Processing_Data") \
         + "\\" + fig_save        
    # 讀取 all MVC data
    MVC_value = pd.read_excel(all_rawdata_folder_path[i].replace(rawData_folder, processingData_folder) \
                              + '\\' + all_rawdata_folder_path[i].split("\\")[-1] \
                                  + '_all_MVC.xlsx')
    # 只取 all MVC data 數字部分
    MVC_value = MVC_value.iloc[-1, 2:]
    # ---------------------------------------------------------------------------------------
    for motion_file in range(len(emg_list)):
        filepath, tempfilename = os.path.split(emg_list[motion_file])
        filename, extension = os.path.splitext(tempfilename)
        emg_path = ""
        for iii in range(len(StagingFile_Exist['EMG_Name'])):
            if StagingFile_Exist.loc[iii, 'EMG_Name'] in emg_list[motion_file]:
                emg_path = emg_list[motion_file]
                print(iii, emg_list[motion_file])
                break
        if len(emg_path) != 0:
            for anc_path in range(len(anc_list)):
                if StagingFile_Exist['.anc'][iii] in anc_list[anc_path]:
                    read_anc = anc_list[anc_path]
                    print(anc_path, anc_list[anc_path])
                    break
        if len(emg_path) != 0 and len(read_anc) != 0:
            # 設置資料儲存路徑 EXCEL
            filepath_1 = filepath.replace(rawData_folder, processingData_folder).replace("motion", "data\\motion")
            save_file_name = filepath_1 + "\\" + filename + "_ed.xlsx"
            # 設置資料儲存路徑 JPG
            filepath_fig = filepath_1.replace("data\\motion", r"figure\processing\bandpass\motion\\")
            save_fig = filepath_fig + "\\" + filename + "_Bandpass.jpg"
            # 讀取 analog data and detect onset
            analog_data = pd.read_csv(anc_list[anc_path],
                                      skiprows=8, delimiter = '	', low_memory=False).iloc[2:, :]
            
            onset_analog = detect_onset(analog_data.loc[:, ['C63']].values.reshape(-1),
                                        np.mean(analog_data.loc[:50, ['C63']].values)*1.1,
                                        n_above=10, n_below=2, show=True)

                
            if len(onset_analog) == 0:
                logging.info('找不到 trigger on: ', read_anc)
                logging.warning('找不到 trigger on: ', read_anc)
            else:
                # 繪製 onset 的時間
                plt.figure()
                plt.plot(analog_data.loc[:, ['C63']].values.reshape(-1))
                plt.axvline(onset_analog[0, 0], color='r')
                plt.title(anc_list[anc_path])
                plt.savefig(filepath_fig + "\\" + filename + "_onset.jpg",
                            dpi=200, bbox_inches = "tight")
                # 定義分期時間點
                triggerOff = StagingFile_Exist.loc[iii, '起點trigger off'] - int(onset_analog[0, 0]/10)
                bodyStart = StagingFile_Exist.loc[iii, '啟動'] - int(onset_analog[0, 0]/10)
                leftLegLeave = StagingFile_Exist.loc[iii, '左腳離地'] - int(onset_analog[0, 0]/10)
                rightLegLeave = StagingFile_Exist.loc[iii, '右腳離地'] - int(onset_analog[0, 0]/10)
                # 讀取 EMG data
                emg_data = pd.read_csv(emg_path)
                processing_data, bandpass_filtered_data = func.EMG_processing(emg_list[motion_file],
                                                                              smoothing=smoothing_method)
                # 計算 iMVC
                emg_iMVC = pd.DataFrame(np.empty([rightLegLeave - triggerOff, np.shape(processing_data)[1]]),
                                        columns=processing_data.columns)
                emg_iMVC.iloc[:, 0] = processing_data.iloc[triggerOff:rightLegLeave, 0].values
                emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[triggerOff:rightLegLeave, 1:].values),
                                                 MVC_value.values)*100
                # writting data in worksheet

                print(save_file_name)
                save_file_name = save_file_name.replace(rawData_folder, processingData_folder)
                with pd.ExcelWriter(save_file_name) as Writer:
                    emg_iMVC.iloc[triggerOff:bodyStart].to_excel(Writer, sheet_name="Stage1", index=False)
                    emg_iMVC.iloc[bodyStart:leftLegLeave].to_excel(Writer, sheet_name="Stage2", index=False)
                    emg_iMVC.iloc[leftLegLeave:rightLegLeave].to_excel(Writer, sheet_name="Stage3", index=False)
                
                # 畫 bandpass filter 的圖

                n = int(math.ceil((np.shape(bandpass_filtered_data)[1] - 1) /2)) # add force plate data
        
                fig, axs = plt.subplots(n, 2, figsize = ((2*n+1,10)), sharex='col')
                for i in range(np.shape(bandpass_filtered_data)[1]-1):
                    xx, yy = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
        
                    # 設定子圖之參數
                    axs[xx, yy].plot(bandpass_filtered_data.iloc[:, 0], bandpass_filtered_data.iloc[:, i+1])
                    axs[xx, yy].set_title(bandpass_filtered_data.columns[i+1], fontsize=16)
                    # 設定科學符號 : 小數點後幾位數
                    axs[xx, yy].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                    axs[xx, yy].axvline(triggerOff/250, color='r')
                    axs[xx, yy].axvline(bodyStart/250, color='r')
                    axs[xx, yy].axvline(leftLegLeave/250, color='r')
                    axs[xx, yy].axvline(rightLegLeave/250, color='r')
                    # a_t = Decimal((StagingFile_Exist['Kneetop'][emg_name] - StagingFile_Exist['trigger'][emg_name])/250).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP")
                    # axs[x, y].annotate(a_t,
                    #                    xy = (0, max(bandpass_filtered_data.iloc[:, i+1])), fontsize = 10, color='b')
                    axs[xx, yy].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                # 設定整張圖片之參數
                plt.suptitle(filename + "Bandpass_", fontsize = 16)
                plt.tight_layout()
                fig.add_subplot(111, frameon=False)
                # hide tick and tick label of the big axes
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.grid(False)
                plt.xlabel("time (second)", fontsize = 14)
                plt.ylabel("Voltage (V)", fontsize = 14)
                plt.savefig(save_fig, dpi=200, bbox_inches = "tight")
                plt.show()



toc = time.process_time()
print("Motion Data Total Time Spent: ",toc-tic)
gc.collect(generation=2)
 # %% read data
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
folder_path = r"D:\BenQ_Project\python\Kao\\"

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
# 定義圖片儲存路徑
motion_fig_save = r"D:\BenQ_Project\python\Kao\motion_processing\\"
# 定義資料儲存位置
data_table = pd.DataFrame({'filename' : [],
                           'order' : [],
                           '左腳離地時間': [],
                           '左腳最大值': [],
                           '左腳最大值時間': [],
                           '右腳離地時間': [],
                           '右腳最大值': [],
                           '右腳最大值時間':[],
                           '啟動時間': [],
                           '啟動左腳力量': [],
                           'Left_RFD': []
                                       })

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
    # 2.1. read EMG data and pre-processing data
    '''
    EMG 尚未處理
    '''
    # EMG_data = pd.read_csv(read_emg)
    # processing_data, bandpass_filtered_data = func.EMG_processing(EMG_data, smoothing='lowpass')
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
        left_lowpass.iloc[:, 0] = leftLeg_FP1.iloc[:, 0]
        right_lowpass.iloc[:, 0] = rightLeg_FP2.iloc[:, 0]
        
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
        # ne_start_onset = detect_onset(combin_left[ana_time_start:], # 將資料轉成一維型態
        #                               np.mean(combin_left[ana_time_start-100:ana_time_start])*0.9,
        #                               n_above=10, n_below=2, show=True)
        pa_start_onset = detect_onset(combin_left[time_TriggerOff*10:], # 將資料轉成一維型態
                                      np.mean(combin_left[ana_time_start-100:ana_time_start])*1.1,
                                      n_above=10, n_below=2, show=True)
        # 找到左腳離地時間
        leftLeg_off = np.where(combin_left[ana_time_start:] < 10)[0][0]
        # 抓取左腳"最大峰值"及"峰值時間"
        left_max = combin_left[ana_time_start:].max()
        left_max_time = combin_left[ana_time_start:].argmax() + ana_time_start + 1 # 因為 python index 會少 1
        # 使用左腳合力，找到動作開始時間
        time_start = pa_start_onset[0, 0] + time_TriggerOff*10
        # 抓取右腳"最大峰值"及"峰值時間"
        right_max = combin_right.max()
        right_time = combin_right.argmax() + 1
        # 找到右腳離地時間，由於有右腳離地的情形，所以時間開始從最大值以後開始算
        rightLeg_off = np.where(combin_right[right_time:] < 10)[0][0]
        # 繪一張合力圖
        x_data = pd.Series(analog_data.loc[:, 'Name'])
        y_data = pd.Series(analog_data.loc[:, 'C63'])
        fig, axs = plt.subplots(3, 1, figsize=(7, 8))
        # 子圖一
        axs[0].plot(left_lowpass.iloc[:, 0],
                    combin_left)
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
        axs[1].plot(right_lowpass.iloc[:, 0],
                    combin_right)
        axs[1].axvline(time_TriggerOff*10, color='red', linestyle='--', linewidth=0.5) # trigger off
        axs[1].plot(rightLeg_off + right_time, combin_right[rightLeg_off + right_time], # 右腳離地時間
                    marker = 'o', ms = 10, mec='c', mfc='none')
        axs[1].plot(right_time, combin_right[right_time], # 右腳力版最大值
                    marker = 'o', ms = 10, mec='b', mfc='none')
        axs[1].set_title('ForcePlare 2: Right Leg')
        axs[1].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
        # 子圖三
        axs[2].plot(right_lowpass.iloc[:, 0], y_data)
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
        plt.ylabel("Frequency (Hz)", fontsize = 14)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # plt.savefig(str(motion_fig_save + StagingFile_Exist.loc[file_name, '.anc'] + "_PF.jpg"),
        #             dpi=200, bbox_inches = "tight")
        # 儲存資料
        # RFD = (left force max - left start force)/(left_max_time - time_start)/2500
        data_table = pd.concat([data_table, pd.DataFrame({'filename' : StagingFile_Exist.loc[file_name, '.anc'],
                                                          'order' : StagingFile_Exist.loc[file_name, 'order'],
                                                          '左腳離地時間': [leftLeg_off + ana_time_start],
                                                          '左腳最大值': [combin_left[left_max_time]],
                                                          '左腳最大值時間': [left_max_time],
                                                          '右腳離地時間': [rightLeg_off + right_time],
                                                          '右腳最大值': [combin_right[right_time]],
                                                          '右腳最大值時間':[right_time],
                                                          '啟動時間': [time_start],
                                                          '啟動左腳力量': [combin_right[time_start]],
                                                          'Left_RFD': [(combin_left[left_max_time] - combin_right[time_start])/ \
                                                                       ((left_max_time - time_start)/2500)],
                                                          }, index=[0])],
                               ignore_index=True)
    

data_table.to_excel(r"D:\BenQ_Project\python\Kao\motion_statistic.xlsx")
    
    

    

    # # 3.2. 計算繪圖所需時間
    # # 重新繪製 force plate 的時間
    # time_ForcePlate = np.linspace(0, (time_RightOff - time_TriggerOff)*1/samplingRate_motion, (time_RightOff - time_TriggerOff)*10)
    # # EMG data processing
    # # EMG 時間換算 (motion - (TriggerOn /10)) * 8
    # # (EMG samppling rate: 2000; motion sampling rate: 250)
    # # 擷取時間為 time_start -> time_RightOff
    # truck_EMG = processing_data.iloc[(time_TriggerOff-int(onset_analog[0,0]/10))*8:(time_RightOff-int(onset_analog[0,0]/10))*8+1, :]
    # time_EMG = np.linspace(0, (time_RightOff - time_TriggerOff)*1/samplingRate_motion,
    #                        np.shape(truck_EMG)[0]) # 以 truck_EMG 的長度為時間切分步數

    # # 5. 畫圖
    # # 創建一個6*2的子圖佈局
    # fig, axs = plt.subplots(6, 2, figsize=(12, 18))
    # for i in range(2):
    #     for j in range(6):
    #         index = i*5 + j # 因為第一欄是time
    #         # print(index)
    #         if j == 0 and i == 0:
    #             x = time_ForcePlate
    #             # 由於力板的採樣時間為 motion system 的十倍
    #             y = leftLeg_FP1.iloc[(time_TriggerOff)*10:(time_RightOff)*10, 1]
    #             axs[j, i].set_title('ForcePlare 2: Left Leg')
                
    #         elif j ==0 and i == 1:
    #             x = time_ForcePlate
    #             # 由於力板的採樣時間為 motion system 的十倍
    #             y = rightLeg_FP2.iloc[(time_TriggerOff)*10:(time_RightOff)*10, 1]
    #             axs[j, i].set_title('ForcePlare 1: Right Leg')
                
    #         else:
    #             print(index)
    #             x = time_EMG
    #             y = truck_EMG.iloc[:, index]
    #             axs[j, i].set_title(truck_EMG.columns[index])
               
    #         axs[j, i].plot(x, y, label=f'Subplot {index+1}')
    #         axs[j, i].axvline((time_start-time_TriggerOff)/250, color='red', linestyle='--', linewidth=0.5)
    #         axs[j, i].axvline((time_LeftOff-time_TriggerOff)/250, color='red', linestyle='--', linewidth=0.5)
    #         axs[j, i].axvline((time_RightOff-time_TriggerOff)/250, color='red', linestyle='--', linewidth=0.5)
    #         # axs[j, i].set_title(f'Subplot {index+1}')
    #         axs[j, i].legend()
    # # 調整子圖之間的距離
    # plt.tight_layout()
    # plt.show()



# %%
    
    # if np.shape(analog_data)[0]/2500 >= np.shape(bandpass_filtered_data)[0]/2000:
    #     analog_time = analog_data.loc[int(onset_analog[0, 0]):\
    #                                   int(onset_analog[0, 0]) + int(np.shape(bandpass_filtered_data)[0]/2000*2500),
    #                                   'Name']
    # else:
    #     analog_time = analog_data.loc[int(onset_analog[0, 0]):,'Name']    
    # n = int(math.ceil((np.shape(bandpass_filtered_data)[1] - 1) /2)) + 1 # add force plate data

    # fig, axs = plt.subplots(n, 2, figsize=(2*n+1, 10), sharex='col')
    # for i in range(2):
    #     for j in range(6):
    #         index = i*5 + j # 因為第一欄是time
    #         # print(index)
    #         if j == 0 and i == 0:
    #             x = analog_time
    #             # 由於力板的採樣時間為 motion system 的十倍
    #             y = analog_data.loc[int(onset_analog[0, 0]):
    #                                 int(onset_analog[0, 0]) + len(analog_time), 'F1Z']
    #             # axs[j, i].set_title('ForcePlare 2: Left Leg')
    #             print("left", len(x), len(y))
    #         elif j ==0 and i == 1:
    #             x = analog_time
    #             # 由於力板的採樣時間為 motion system 的十倍
    #             y = analog_data.loc[int(onset_analog[0, 0]):
    #                                 int(onset_analog[0, 0]) + len(analog_time), 'F2Z']
    #             print("right", len(x), len(y))
    #             # axs[j, i].set_title('ForcePlare 1: Right Leg')
                
    #         else:
    #             print(index)
    #             x = bandpass_filtered_data.iloc[:, 0]
    #             y = bandpass_filtered_data.iloc[:, index]
    #             print('emg', len(x), len(y))
    #             # axs[j, i].set_title(bandpass_filtered_data.columns[index], fontsize=16)
    #         # print(x, y)
    #         # print(len(x), len(y))
    #         plt.plot(x, y)
    #         plt.show()
               
    #         axs[j, i].plot(x, y, label=f'Subplot {index+1}')
    #         axs[j, i].ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))
    #         axs[j, i].axvline(triggerOff/250, color='r')
    #         axs[j, i].axvline(bodyStart/250, color='r')
    #         axs[j, i].axvline(leftLegLeave/250, color='r')
    #         axs[j, i].axvline(rightLegLeave/250, color='r')
    #         # axs[j, i].set_title(f'Subplot {index+1}')
    #         axs[j, i].legend()
    # # 調整子圖之間的距離
    # plt.tight_layout()
    # plt.show()

        # 讀取資料
        
              
                
        # ---------------------------------------------------------------------------         
        # if release_idx != "Nan":
        #     # pre-processing data
            
        #     # get release time
               
        #     # release_samp_freq = int(1/(processing_data.iloc[1, 0] - processing_data.iloc[0, 0]))
        #     # 去做條件判斷要輸出何種資料
        #     if smoothing_method == 'lowpass':
        #         ## 擷取 EMG data
        #         # 計算MVC值
        #         emg_iMVC = pd.DataFrame(np.empty([release[0]+release[1], np.shape(processing_data)[1]]),
        #                                 columns=processing_data.columns)
        #         emg_iMVC.iloc[:, 0] = processing_data.iloc[release_idx-release[0]:release_idx+release[1], 0].values
        #         emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[release_idx-release[0]:release_idx+release[1], 1:].values),
        #                                          MVC_value.values)*100
        #     elif smoothing_method == 'rms' or smoothing_method == 'moving':
        #         # 找出最接近秒數的索引值
        #         start_idx = np.abs(processing_data.iloc[:, 0] - (release_idx - release[0])/down_freq).argmin()
        #         # # 由於 python 取數字需多 +1
        #         end_idx = np.abs(processing_data.iloc[:, 0] - (release_idx + release[1])/down_freq).argmin()
        #         print(processing_data.loc[start_idx, "time"], processing_data.loc[end_idx, "time"])
        #         if (release_idx + release[1])/down_freq > processing_data.loc[processing_data.index[-1], "time"]:
        #             warnings.warn("時間開始位置不一", RuntimeWarning)
        #             print("原始數據短於設定擊發後時間，請減少擊發後時間")
                    
        #         # Sep 13 2023.  修正 end_inx 會有不一致的情形, 但是總訊號會少 1 frame
        #         while int(end_idx - start_idx) > \
        #             int((sum(release) - down_freq * time_of_window) / (down_freq*time_of_window*(1-overlap_len))) + 1:
        #             end_idx = end_idx - 1
        #         while int(end_idx - start_idx) < \
        #             int((sum(release) - down_freq * time_of_window) / (down_freq*time_of_window*(1-overlap_len))) + 1:
        #             end_idx = end_idx + 1
        
        #         rms_data = processing_data.iloc[start_idx:end_idx, :].reset_index(drop=True)
    
        #         emg_iMVC = pd.DataFrame(np.zeros(np.shape(rms_data)),
        #                                 columns=processing_data.columns)
        #         emg_iMVC.iloc[:, 0] = rms_data.iloc[:, 0].values
        #         # 加絕對值，以避免數值趨近 0 時，會出現負數問題
        #         emg_iMVC.iloc[:, 1:] = np.divide(abs(rms_data.iloc[:, 1:].values),
        #                                          MVC_value.values)*100





# %%



















