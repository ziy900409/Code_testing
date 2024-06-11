# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 10:53:27 2024

@author: Hsin.YH.Yang
"""

import gc
import os
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"E:\Hsin\git\git\Code_testing\Archery\Xiao")
# sys.path.append(r"D:\BenQ_Project\git\Code_testing\Archery\Xiao")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import XiaoThesisMotionFunction as mot
import XiaoThesisGeneralFunction as gen
import XiaoThesisEMGFunction as emg
# from detecta import detect_onset
# from scipy import signal

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
staging_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\Archery_stage_v5_input.xlsx"
data_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\202406\202405"
shooting_staging_file = r"E:\Hsin\NTSU_lab\Archery\Xiao\202406\202405\_algorithm_output_formatted_date.xlsx"

# staging_path = r"D:\BenQ_Project\python\Archery\202405\Archery_stage_v5_input.xlsx"
# data_path = r"D:\BenQ_Project\python\Archery\202405\202405\202405\\"
# shooting_staging_file = r"E:\Hsin\NTSU_lab\Archery\Xiao\202406\202405\_algorithm_output_formatted_date.xlsx"

# 測試組
subject_list = ["R01"]
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
folder_paramter["first_layer"]["motion"][0]
time_period =  ["E1-E2", "E2-E3-1", "E3-1-E3-2",
                "E3-2-E4", "E4-E5"]

# 第三層 ----------------------------------------------------------------------

fig_folder = "\\figure\\"
data_folder = "\\data\\"
MVC_folder = "\\MVC\\"
motion_folder = folder_paramter["first_layer"]["motion"][0]
# downsampling frequency
down_freq = 2000
# 抓放箭時候前後秒數
motion_fs = 250
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.2 # 窗格長度 (單位 second)
overlap_len = 0.98 # 百分比 (%)
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
# 設置繪圖參數 --------------------------------------------------------------

muscle_group = {"release": ['R EXT: EMG 1', 'R TRI : EMG 2', 'R FLX: EMG 3', 'R BI: EMG 4'],
                "right": ['R UT: EMG 5', 'R LT: EMG 6', 'R INF: EMG 7 (IM)', 'R LAT: EMG 8', 'R PD: EMG 13'],
                "left": ['L UT: EMG 9', 'L LT: EMG 10', 'L INF: EMG 11', 'L LAT: EMG 12', 'L MD: EMG 14']}

# 设置颜色
cmap = plt.get_cmap('Set2')
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

# %% MVC 資料前處理
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


# %% 計算 iMVC : trunkcut data and caculate iMVC value
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

# 開始處理 motion 資料
for subject in subject_list:
    for i in range(len(all_rawdata_folder_path["EMG"])):
        if subject in all_rawdata_folder_path["EMG"][i]:
            print(all_rawdata_folder_path["EMG"][i])
            # 讀取路徑下所有的 shooting motion file
            Shooting_path = all_rawdata_folder_path["EMG"][i] + "\\" + motion_folder
            Shooting_list = gen.Read_File(Shooting_path, '.csv')
            # 設定儲存照片路徑
            fig_save_path = all_rawdata_folder_path["EMG"][i].replace("Raw_Data", "Processing_Data") \
                + "\\" + fig_folder        
            # 讀取 all MVC data
            MVC_value = pd.read_excel(all_rawdata_folder_path["EMG"][i].replace("Raw_Data", "Processing_Data") \
                                     + '\\' + all_rawdata_folder_path["EMG"][i].split("\\")[-1] \
                                     + '_all_MVC.xlsx')
            # 只取 all MVC data 數字部分
            MVC_value = MVC_value.iloc[-1, 2:]
            # 確認資料夾路徑下是否有 staging file
            subject = all_rawdata_folder_path["EMG"][i].split("\\")[-1]
            staging_data = pd.read_excel(shooting_staging_file,
                                         # data_path + "_algorithm_output_formatted_date" + ".xlsx"
                                         sheet_name=subject)
            for ii in range(len(Shooting_list)):
                for iii in range(len(staging_data['EMG_filename'])):
                    # for mac version replace "\\" by '/'
                    if Shooting_list[ii].split('\\')[-1] == staging_data['EMG_filename'][iii].split('\\')[-1]:
                        print(staging_data['EMG_filename'][iii])
                        # 1. 讀取資料及資料前處理 ------------------------------
                        data = pd.read_csv(Shooting_list[ii], encoding='UTF-8')
                        processing_data, bandpass_filtered_data = emg.EMG_processing(data,
                                                                                     smoothing=smoothing_method,
                                                                                     notch=True)
                        # 擷取檔名
                        filepath, tempfilename = os.path.split(Shooting_list[ii])
                        filename, extension = os.path.splitext(tempfilename)
                        # save file name
                        save_file =  all_rawdata_folder_path["EMG"][i].replace("Raw_Data", "Processing_Data")\
                            + data_folder + '\\' + motion_folder + '\\' + filename + end_name + ".xlsx"
                        # EMG sampling rate
                        emg_fs = 1 / (data.iloc[1, 0] - data.iloc[0, 0])
                        # 2. 前處理資料繪圖 ----------------------------------------------------------
                        # 畫 FFT analysis 的圖
                        emg.Fourier_plot(Shooting_list[ii],
                                        (fig_save_path + "\\FFT\\motion"),
                                        filename,
                                        notch=False)
                        emg.Fourier_plot(Shooting_list[ii],
                                        (fig_save_path + "\\FFT\\motion"),
                                        filename,
                                        notch=True)
                        # 畫 bandpass 後之資料圖
                        emg.plot_plot(bandpass_filtered_data,
                                      str(fig_save_path + "\\processing\\bandpass\\" + motion_folder),
                                     filename, "Bandpass_")
                        # 畫前處理後之資料圖
                        emg.plot_plot(processing_data,
                                      str(fig_save_path + "\\processing\\smoothing\\" + motion_folder),
                                     filename, str("_" + smoothing_method))
                        
                        # 3. 定義分期時間，staging file 的時間為 motion，需轉為 EMG -----------------------------
                        E1_idx = int((staging_data["E1 frame"][iii] - staging_data["trigger"][iii]) \
                                     / motion_fs * emg_fs)
                        E2_idx = int((staging_data["Bow_Height_Peak_Frame"][iii] - staging_data["trigger"][iii])\
                                     / motion_fs * emg_fs)
                        E3_1_idx = int((staging_data["E3-1 frame"][iii] - staging_data["trigger"][iii])\
                                       / motion_fs * emg_fs)
                        E3_2_idx = int((staging_data["Anchor_Frame"][iii] - staging_data["trigger"][iii])\
                                       / motion_fs * emg_fs)
                        E4_idx = int((staging_data["Release_Frame"][iii] - staging_data["trigger"][iii])\
                                     / motion_fs * emg_fs) # release_idx
                        E5_idx = int((staging_data["E5 frame"][iii] - staging_data["trigger"][iii])\
                                     / motion_fs * emg_fs)
                        
                        
                        
                        # 去做條件判斷要輸出何種資料
                        if smoothing_method == 'lowpass':
                            ## 擷取 EMG data
                            # 計算MVC值
                            emg_iMVC = pd.DataFrame(np.empty([E5_idx-E1_idx, np.shape(processing_data)[1]]),
                                                    columns=processing_data.columns)
                            emg_iMVC.iloc[:, 0] = processing_data.iloc[E1_idx:E5_idx, 0].values
                            emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[E1_idx:E5_idx, 1:].values),
                                                             MVC_value.values)*100
                        elif smoothing_method == 'rms' or smoothing_method == 'moving':
                            moving_E1_idx = np.abs(processing_data.iloc[:, 0] - (E1_idx)/down_freq).argmin()
                            moving_E2_idx = np.abs(processing_data.iloc[:, 0] - (E2_idx)/down_freq).argmin()
                            moving_E3_1_idx = np.abs(processing_data.iloc[:, 0] - (E3_1_idx)/down_freq).argmin()
                            moving_E3_2_idx = np.abs(processing_data.iloc[:, 0] - (E3_2_idx)/down_freq).argmin()
                            moving_E4_idx = np.abs(processing_data.iloc[:, 0] - (E4_idx)/down_freq).argmin()
                            moving_E5_idx = np.abs(processing_data.iloc[:, 0] - (E5_idx)/down_freq).argmin()
                            
                            # iMVC
                            emg_iMVC = pd.DataFrame(np.zeros(np.shape(processing_data)),
                                                    columns=processing_data.columns)
                            emg_iMVC.iloc[:, 0] = processing_data.iloc[:, 0].values
                            # 加絕對值，以避免數值趨近 0 時，會出現負數問題
                            emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[:, 1:].values),
                                                             MVC_value.values)*100
                        print(save_file)
                        # writting data in worksheet
                        pd.DataFrame(emg_iMVC).to_excel(save_file, sheet_name='Sheet1',
                                                        index=False, header=True)
                        with pd.ExcelWriter(save_file) as Writer:
                            emg_iMVC.iloc[moving_E1_idx:moving_E2_idx, :].to_excel(Writer, sheet_name="E1-E2", index=False)
                            emg_iMVC.iloc[moving_E2_idx:moving_E3_1_idx, :].to_excel(Writer, sheet_name="E2-E3-1", index=False)
                            emg_iMVC.iloc[moving_E3_1_idx:moving_E3_2_idx, :].to_excel(Writer, sheet_name="E3-1-E3-2", index=False)
                            emg_iMVC.iloc[moving_E3_2_idx:moving_E4_idx, :].to_excel(Writer, sheet_name="E3-2-E4", index=False)
                            emg_iMVC.iloc[moving_E4_idx:moving_E5_idx, :].to_excel(Writer, sheet_name="E4-E5", index=False)
                try:
                    if staging_data in globals() or staging_data in locals():
                        del staging_data
                except TypeError:
                    print('The variable does not exist')
    
toc = time.process_time()
print("Motion Data Total Time Spent: ",toc-tic)
gc.collect(generation=2)


# %% 畫 Mean std cloud 圖                    
'''
1. 列出檔案夾路徑，並設定讀取motion資料夾
2. 給 mean_std_cloud function 繪圖
'''
tic = time.process_time()
# 開始處理 motion 資料
for subject in subject_list:
    for i in range(len(all_rawdata_folder_path["EMG"])):
        if subject in all_rawdata_folder_path["EMG"][i]:
            print(all_rawdata_folder_path["EMG"][i])
            temp_motion_folder = all_rawdata_folder_path["EMG"][i].replace("Raw_Data", "Processing_Data") + data_folder + motion_folder
            fig_save_1 = all_rawdata_folder_path["EMG"][i].replace("Raw_Data", "Processing_Data") + "\\figure\\Cut1_figure"
            fig_save_2 = all_rawdata_folder_path["EMG"][i].replace("Raw_Data", "Processing_Data") + "\\figure\\Cut2_figure"
            for muscle in muscle_group:
                # print(muscle)
                emg.compare_mean_std_cloud(temp_motion_folder,
                                          fig_save_1,
                                           str(subject + "_SH1 vs SHM_" + muscle),
                                           "rms",
                                           compare_name = ["SH1", "SHM"],
                                           muscle_name = muscle_group[muscle])
                
                emg.compare_mean_std_cloud(temp_motion_folder,
                                           fig_save_1,
                                           str(subject + "_SHH vs SHM vs SHH_" + muscle),
                                           "rms",
                                           compare_name = ["SHH", "SHM", "SHL"],
                                           muscle_name = muscle_group[muscle])
                emg.compare_mean_std_cloud(temp_motion_folder,
                                           fig_save_2,
                                           str(subject + "_SH1 vs SHH vs SHM vs SHH_" + muscle),
                                           "rms",
                                           compare_name = ["SH1", "SHH", "SHM", "SHL"],
                                           muscle_name = muscle_group[muscle])
toc = time.process_time()
print("Total Time Spent: ",toc-tic)
gc.collect(generation=2)




        
  






























