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
# import XiaoThesisMotionFunction as mot
import XiaoThesisGeneralFunction as gen
import XiaoThesisEMGFunction as emg
# from detecta import detect_onset
# from scipy import signal
from scipy.interpolate import interp1d
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
            staging_data = pd.read_excel(data_path + "_algorithm_output_formatted_date" + ".xlsx",
                                              sheet_name=subject)
        
            # --------------------------------------------------------------------------------------
            for ii in range(len(Shooting_list)):
                for iii in range(len(staging_data['EMG_filename'])):
                    # for mac version replace "\\" by '/'
                    if Shooting_list[ii].split('\\')[-1] == staging_data['EMG_filename'][iii].split('\\')[-1]:
                        print(staging_data['EMG_filename'][iii])
                        # 讀取資料
                        data = pd.read_csv(Shooting_list[ii], encoding='UTF-8')
                        # 擷取檔名
                        filepath, tempfilename = os.path.split(Shooting_list[ii])
                        filename, extension = os.path.splitext(tempfilename)
                        # save file name
                        save_file =  all_rawdata_folder_path["EMG"][i].replace("Raw_Data", "Processing_Data")\
                            + data_folder + '\\' + motion_folder + '\\' + filename + end_name + ".xlsx"
                        # EMG sampling rate
                        emg_fs = 1 / (data.iloc[1, 0] - data.iloc[0, 0])
                        
                        # 定義分期時間，staging file 的時間為 motion，需轉為 EMG
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
                        
                        processing_data, bandpass_filtered_data = emg.EMG_processing(data, smoothing=smoothing_method)
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

# %%
def Read_File(file_path, file_type, subfolder=None):
    '''
    Parameters
    ----------
    x : str
        給予欲讀取資料之路徑.
    y : str
        給定欲讀取資料之副檔名.
    subfolder : boolean, optional
        是否子資料夾一起讀取. The default is 'None'.

    Returns
    -------
    csv_file_list : list
        回給所有路徑下的資料絕對路徑.

    '''
    # if subfolder = True, the function will run with subfolder

    csv_file_list = []
    
    if subfolder:
        file_list_1 = []
        for dirPath, dirNames, fileNames in os.walk(file_path):
            # file_list = os.walk(folder_name)
            file_list_1.append(dirPath)
        # need to change here [1:]
        for ii in file_list_1[1:]:
            file_list = os.listdir(ii)
            for iii in file_list:
                if os.path.splitext(iii)[1] == file_type:
                    # replace "\\" to '/', due to MAC version
                    file_list_name = ii + '\\' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(file_path)                
        for i in folder_list:
            if os.path.splitext(i)[1] == file_type:
                # replace "\\" to '/', due to MAC version
                file_list_name = file_path + "\\" + i
                csv_file_list.append(file_list_name)
    # 排除可能會擷取到暫存檔的問題，例如：~$test1_C06_SH1_Rep_2.2_iMVC_ed.xlsx                
    csv_file_list = [file for file in csv_file_list if not "~$" in file]
    return csv_file_list

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
            Shooting_path = all_rawdata_folder_path["EMG"][i] + "\\" + motion_folder
            Shooting_list = gen.Read_File(Shooting_path, '.csv')
            # 設定儲存照片路徑
            fig_save_path = all_rawdata_folder_path["EMG"][i].replace("Raw_Data", "Processing_Data") \
                + "\\" + fig_folder
            motion_folder_path = all_rawdata_folder_path["EMG"][i].replace("Raw_Data", "Processing_Data")\
                + data_folder + motion_folder
            save_path =  all_processing_folder_path[i]
            print("圖片存檔路徑: ", save_path)
            emg.compare_mean_std_cloud(motion_folder_path,
                                      all_processing_folder_path[i] + "\\" + folder_paramter["subject_subfolder"][1] +"\\data\\motion",
                                      save_path, all_processing_folder_path[i].split("\\")[-1],
                                      smoothing_method,
                                      self_oreder=False)
toc = time.process_time()
print("Total Time Spent: ",toc-tic)
gc.collect(generation=2)




# %%
def compare_mean_std_cloud(data_path, savepath, filename, smoothing,
                           compare_name = ["SH1", "SHM"],
                           muscle_name = ["R EXT: EMG 1", "R TRI : EMG 2", "R FLX: EMG 3",
                                          "R BI: EMG 4", "R UT: EMG 5", "R LT: EMG 6"],
                           self_oreder=False):
    '''


    Parameters
    ----------
    v1_data_path : str
        第一個要比較數據的資料位置.
    v2_data_path : str
        DESCRIPTION.
    savepath : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.
    smoothing : TYPE
        DESCRIPTION.
    release : TYPE
        DESCRIPTION.
    self_oreder : dict, optional
        order_mapping = {'R EXT': 1, 'R FLX': 2, 'R UT': 3,
                         'R LT': 4, 'R LAT': 5, 'R PD': 6,
                         'L LT': 7, 'L MD': 8}.
        The default is False.

    Returns
    -------
    None.

    '''
    # 創造資料儲存位置
    time_ratio = {"E1-E2": 1,
                  "E2-E3-1": 1,
                  "E3-1-E3-2": 0.5,
                  "E3-2-E4": 3,
                  "E4-E5": 0.2}
    total_time = 0
    for ratio in time_ratio.keys():
        total_time += time_ratio[ratio]
    
    data_path = r'D:\\BenQ_Project\\python\\Archery\\202405\\202405\\202405\\\\\\EMG\\\\Processing_Data\\Method_1\\R01\\data\\\\motion\\'
    
    # data_path = r'D:\python\EMG_Data\To HSIN\EMG\Processing_Data\Method_1\C06\\'
    # v1_data_path = data_path + 'test1\\data\\motion'
    # v2_data_path = data_path + 'test2\\data\\motion'
    # 找出所有資料夾下的 .xlsx 檔案
    data_list = Read_File(data_path, ".xlsx", subfolder=False)
    compare_data = {key: [] for key in compare_name}
    for i in range(len(compare_name)):
        for ii in range(len(data_list)):
            if compare_name[i] in data_list[ii]:
                compare_data[compare_name[i]].append(data_list[ii])
                
    data_1 = np.empty([len(muscle_name),
                       int(total_time*10*2),
                       len(compare_data["SH1"])])
    # muscle name * time length * subject number
    if len(compare_data) == 2:
        data_1 = np.empty([len(muscle_name),
                           int(total_time*10*2),
                           len(compare_data.keys()[0])])
        data_2 = np.empty([len(muscle_name),
                           int(total_time*10*2),
                           len(compare_data[1])])
    elif len(compare_data) == 3:
        data_1 = np.empty([len(muscle_name),
                           int(total_time*10*2),
                           len(compare_data[0])])
        data_2 = np.empty([len(muscle_name),
                           int(total_time*10*2),
                           len(compare_data[1])])
        data_3 = np.empty([len(muscle_name),
                           int(total_time*10*2),
                           len(compare_data[2])])
    
    
    for i in range(len(compare_name)): # 處理不同condition之間的比較
        for ii in range(len(compare_data[compare_name[i]])): # 共有多少筆資料
            if i == 0:
                print(compare_data[compare_name[i]][ii])
                time_idx = 0
                for period in time_ratio.keys():
                    print(period)
                    raw_data = pd.read_excel(compare_data[compare_name[i]][ii],
                                             sheet_name=period)
                    for muscle in range(len(muscle_name)):
                        # 定義分期時間比例
                        time_period = int(time_ratio[period]*10*2)
                        # print(time_period)
                        # 使用 cubic 將資料內插
                        x = raw_data.iloc[:, 0] # time
                        y = raw_data.loc[:, muscle_name[muscle]]
                        
                        f = interp1d(x, y, kind='cubic')
                        x_new = np.linspace(raw_data.iloc[0, 0], raw_data.iloc[-1, 0],
                                            time_period)
                        y_new = f(x_new)
                        # print(y_new)
                        data_1[muscle, time_idx:time_idx + time_period, ii] = y_new
                    print(time_idx, time_idx + time_period)
                    time_idx = time_idx + time_period
                    
                    
    
    
    v2_file_list = Read_File(v2_data_path, ".xlsx", subfolder=False)
    # 排除可能會擷取到暫存檔的問題，例如：~$test1_C06_SH1_Rep_2.2_iMVC_ed.xlsx
    v1_file_list = [file for file in v1_file_list if not "~$" in file]
    v2_file_list = [file for file in v2_file_list if not "~$" in file]
    # 取得資料欄位名稱，並置換掉 :EMG
    v1_data_cloumns = list(pd.read_excel(v1_file_list[0]).columns)
    v2_data_cloumns = list(pd.read_excel(v2_file_list[0]).columns)
    # 去掉時間欄位
    for i in ['time']:
        v1_data_cloumns.remove(i)
        v2_data_cloumns.remove(i)
    

    # 初始化一個空列表，用來存放相同字串的位置
    v1_data_cloumns = remove_specific_string_from_list(v1_data_cloumns)
    v2_data_cloumns = remove_specific_string_from_list(v2_data_cloumns)
    
    common_elements_positions = []
    
    # 使用迴圈逐一比較兩個列表中的元素
    for item1 in v1_data_cloumns:
        if item1 in v2_data_cloumns:
            # 找到相同的字串，取得在兩個列表中的位置
            position1 = v1_data_cloumns.index(item1)
            position2 = v2_data_cloumns.index(item1)
            
            # 將位置資訊加入到列表中
            common_elements_positions.append((item1, position1, position2))
    

    # 說明兩組資料各幾筆

    # read example data
    example_data = pd.read_excel(v1_file_list[0])

    # create multi-dimension matrix
    type1_dict = np.zeros(((np.shape(example_data)[1] - 1), # muscle name without time
                           (np.shape(example_data)[0]), # time length
                           len(v1_file_list)))                 # subject number
    type2_dict = np.zeros(((np.shape(example_data)[1] - 1), # muscle name without time
                           (np.shape(example_data)[0]), # time length
                           len(v2_file_list)))                 # subject number
    if not self_oreder:
    # 將資料逐步放入預備好的矩陣
        for ii in range(len(v1_file_list)):
            # read data
            type1_data = pd.read_excel(v1_file_list[ii])
            for iii in range(len(common_elements_positions)): # exclude time
                type1_dict[iii, :, ii] = type1_data.iloc[:, common_elements_positions[iii][1]+1]
        
        for ii in range(len(v2_file_list)):
            type2_data = pd.read_excel(v2_file_list[ii])
            for iii in range(len(common_elements_positions)): # exclude time
                type2_dict[iii, :, ii] = type2_data.iloc[:, common_elements_positions[iii][2]+1]
        # 設定圖片 tilte
        data_title = common_elements_positions
    else:
        # 給定編排方式
        # order_mapping = {'R EXT': 1, 'R FLX': 2, 'R UT': 3, 'R LT': 4, 'R LAT': 5, 'R PD': 6, 'L LT': 7, 'L MD': 8}
        # 使用 sorted 函數進行排序，根據映射方式提供的排序順序
        sorted_data = sorted(common_elements_positions, key=lambda x: self_oreder[x[0]])
        # 設定圖片 tilte
        data_title = sorted_data[:len(self_oreder)]
        for ii in range(len(v1_file_list)):
            # read data
            type1_data = pd.read_excel(v1_file_list[ii])
            for iii in range(len(data_title)): # exclude time
                type1_dict[iii, :, ii] = type1_data.iloc[:, sorted_data[iii][1]+1]
        
        for ii in range(len(v2_file_list)):
            type2_data = pd.read_excel(v2_file_list[ii])
            for iii in range(len(data_title)): # exclude time
                type2_dict[iii, :, ii] = type2_data.iloc[:, sorted_data[iii][2]+1]


    # 設定圖片大小
    # 畫第一條線
    save = savepath + "\\mean_std_" + filename + ".jpg"
    n = int(math.ceil((np.shape(type2_dict)[0]) /2))
    # 設置圖片大小
    plt.figure(figsize=(2*n+1,10))
    # 設定繪圖格式與字體
    # plt.style.use('seaborn-white')
    # 顯示輸入中文
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    palette = plt.get_cmap('Set1')
    fig, axs = plt.subplots(n, 2, figsize = (10,12), sharex='col')
    
    for i in range(len(data_title)):
        # 確定繪圖順序與位置
        x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
        color = palette(0) # 設定顏色
        iters = list(np.linspace(-release[0], release[1], 
                                 len(type1_dict[0, :, 0])))
        # 設定計算資料
        avg1 = np.mean(type1_dict[i, :, :], axis=1) # 計算平均
        std1 = np.std(type1_dict[i, :, :], axis=1) # 計算標準差
        r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
        r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
        axs[x, y].plot(iters, avg1, color=color, label='before', linewidth=3)
        axs[x, y].fill_between(iters, r1, r2, color=color, alpha=0.2)
        # 找所有數值的最大值，方便畫括弧用
        yy = max(r2)
        # 畫第二條線
        color = palette(1) # 設定顏色
        avg2 = np.mean(type2_dict[i, :, :], axis=1) # 計畫平均
        std2 = np.std(type2_dict[i, :, :], axis=1) # 計算標準差
        r1 = list(map(lambda x: x[0]-x[1], zip(avg2, std2))) # 畫一個標準差以內的線
        r2 = list(map(lambda x: x[0]+x[1], zip(avg2, std2)))
        # 找所有數值的最大值，方便畫括弧用
        yy = max([yy, max(r2)])
        axs[x, y].plot(iters, avg2, color=color, label='after', linewidth=3) # 畫平均線
        axs[x, y].fill_between(iters, r1, r2, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊
        # 圖片的格式設定
        axs[x, y].set_title(data_title[i][0], fontsize=12)
        axs[x, y].legend(loc="upper left") # 圖例位置
        axs[x, y].grid(True, linestyle='-.')
        # 畫放箭時間
        axs[x, y].set_xlim(-(release[0]), release[1])
        axs[x, y].axvline(x=0, color = 'darkslategray', linewidth=1, linestyle = '--')

        # 畫花括號
        curlyBrace(fig, axs[x, y], [shooting_time["stage1"][0], yy], [shooting_time["stage1"][1], yy],
                   0.05, bool_auto=True, str_text="", color=shooting_time["stage1"][2],
                   lw=2, int_line_num=1, fontdict=font)
        curlyBrace(fig, axs[x, y], [shooting_time["stage2"][0], yy], [shooting_time["stage2"][1], yy],
                   0.05, bool_auto=True, str_text="", color=shooting_time["stage2"][2],
                   lw=2, int_line_num=1, fontdict=font)
        curlyBrace(fig, axs[x, y], [shooting_time["stage3"][0], yy], [shooting_time["stage3"][1], yy],
                   0.05, bool_auto=True, str_text="", color=shooting_time["stage3"][2],
                   lw=2, int_line_num=1, fontdict=font)
        curlyBrace(fig, axs[x, y], [shooting_time["stage4"][0], yy], [shooting_time["stage4"][1], yy],
                   0.05, bool_auto=True, str_text="", color=shooting_time["stage4"][2],
                   lw=2, int_line_num=1, fontdict=font)

        
    plt.suptitle(str("mean std cloud: " + filename), fontsize=16)
    plt.tight_layout()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("time (second)", fontsize = 14)
    plt.ylabel("muscle activation (%)", fontsize = 14)
    plt.savefig(save, dpi=200, bbox_inches = "tight")
    plt.show()
































