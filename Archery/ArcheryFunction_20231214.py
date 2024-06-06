# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:19:26 2023

新增功能
December 14 2023.
    1. function compare_mean_std_cloud. (新增)
        1.1. 繪圖比較功能
        1.2. 可以自定義繪圖順序，使用方式參考 function discribe
November 02 2023.
    1. function EMG_processing.
        1.1. 修正 resample 問題，新增變數 resample length以統一降取樣的資料長度
September 12 2023.
    1. function EMG_processing.
        1.1. 新增找各不同代 Sensor 時間長短不一的問題，並將所有不同 Sensor 的資料長短都裁成
            一致的長短，然後再做後續的資料處理
        1.2. 修正找 all_stop_time 的功能, 先前設定 len(data_time) 會出現長短不一致的問題
        1.3. 修正如果有 Sensor 出現斷訊問題，將使用次短的訊號作替代，直到最大與最小的差異小於 1 秒
            all_stop_time, Fs
August 03 2023.
    1. 若重新執行程式，先刪除 processing data 資料夾中的所有資料
    2. 調整 code 順序
        2.1. 找放箭時間 (或其他舉弓頂點、開始拉弓的分期)
        2.2. MVC motion資料夾一起FFT、一起bandpass、一起RMS 生成圖
        2.3. 取MVC 最大值
        2.4. motion 切割 & 除最大值
        2.5. 射箭資料分組取平均&標準差，作圖
    3. 圖形方面，有一些細節不知道容不容易
        3.1. 伸腕三頭屈腕二頭 畫一張圖，其他肌肉畫一張圖，希望能設定各自分期點&前後取的時間
        3.2. Fatigue 中位頻率每條肌肉Y軸座標軸一樣
July 13 2023.
    1. function EMG_processing, Fourier_plot, median_frquency.
        1.1. 新增自動找EMG的欄位，不用再設定相關欄位.
        1.2. 不再計算平均截止時間，改用最小值，避免 sensor 收資料的時間長度不一致的問題.
    2. function mean_std_cloud
        2.1. 修改 release 參數，統一在 main function 賦值即可.
June 06 2023.    
    1. function mean_std_cloud 新增功能
        1.1. 修正只有lowpass會有標示放箭時間的問題
        1.2. 修正時間軸錯誤問題
        1.3. 修正繪圖 Y 軸的科學記號問題
    mean std 的放箭時間點, 時間軸, 時間錯誤、不能出現科學符號、標紅線的放箭時間
    疲勞測試，畫在同一張圖
June 03 2023.
    1. 修正單數肌群會畫不出圖的問題
    解決方式 : 繪圖數量為 無條件進位(採樣肌肉數量/2)
                np.ceil(num_columns / 2)
    2. 修正中文無法顯示的問題
    3. 修正如果 sensor 斷訊出現 NAN 問題
    解決方法 :
        3.1 使用 np.where(np.isnan(a)) 找出 NAN 位置的索引值，如果"缺值"時間短於 0.1 秒，先將 NAN 填 0
        3.2 若缺值時間長於 0.1 秒，則給予警告訊息       
May 02 2023.
    1. 新增可選擇平滑處理的方式
    2. 新增 median freqency 功能
April 26 2023.
    1. 修正 EMG_processing 中 lowpass的採樣頻率問題


@author: Hsin.YH.Yang
"""
# %% import library
import os
import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.fft import fft
from scipy import signal
import math
import logging #print 警告用
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import re

# 路徑改成你放自己code的資料夾
sys.path.append(r"C:/Users/angel/Documents/NTSU/BioLearning/Python")
# 用來畫括弧的 package
from curlyBrace import curlyBrace
# matplotlib 設定中文顯示，以及圖片字型
# mpl.rcParams['font.family'] = 'Microsoft Sans Serif'
mpl.rcParams["font.sans-serif"] = ["'Microsoft Sans Serif"]
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False   # 步驟二（解決坐標軸負數的負號顯示問題）
font = {'family': 'serif',
        'color':  'k',
        'weight': 'bold',
        'size': 20,
        }
import warnings

# 開啟警告捕捉
warnings.filterwarnings('always')
# %% 設置參數位置
'''
使用順序
0.5 採樣頻率已自動計算，不需自己定義
** 由於不同代EMG sensor的採樣頻率不同，因此在初步處理時，先down sampling to 1000Hz
1. 依照個人需求更改資料處理欄位，更改位置為
    ex : num_columns = [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]
2. 更改bandpass filter之參數
    ex : bandpass_sos = signal.butter(2, [8/0.802, 450/0.802],  btype='bandpass', fs=Fs, output='sos')
3. 確定smoothing method並修改參數
    3.1 moving_data, rms_data
        window_width = 0.1 # 窗格長度 (單位 second)
        overlap_len = 0 # 百分比 (%)
    3.2 lowpass_filtered_data
        lowpass_sos = signal.butter(2, 6/0.802, btype='low', fs=Fs, output='sos')
4. 修改release time之時間，預設放箭前2.5秒至放箭後0.5秒

'''
# ---------------------找放箭時間用----------------------------
# 設定最接近放箭位置之acc sensor的欄位編號，建議看完三軸資料再選最大的
# 可設定數字或是欄位名稱：ex: R EXTENSOR GROUP: ACC.Y 1 or 5
release_acc = 17
# 設定放箭的振幅大小值
release_peak = 1.5
# ------------------------------------------------------------
# ---------------------前處理用--------------------------------
# downsampling frequency
down_freq = 1000
# 帶通濾波頻率
bandpass_cutoff = [8/0.802, 450/0.802]
# 低通濾波頻率
lowpass_freq = 6/0.802
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.1 # 窗格長度 (單位 second)
overlap_len = 0.5 # 百分比 (%)
# ------------------------------------------------------------
# ---------------------繪製疲勞前後比較圖用---------------------
# 放箭前檔名
before_fatigue = "_1_"
# 放箭後檔名
after_fatigue = "_2_"
# 預處理資料可修改檔名，並新增標籤，如：S2_MVC_Rep_1.16 -> S2_MVC_Rep_1.16_low
end_name = "_ed"
# 平滑處理方式 ex: lowpass, rms, moving
smoothing_method = 'lowpass' 
# example : [秒數*採樣頻率, 秒數*採樣頻率]
# release = [3*down_freq, 0.5*down_freq]
# 設定擊發分期時間
# 置換顏色參考網站
# https://matplotlib.org/stable/gallery/color/named_colors.html
shooting_time = dict({"stage1": [-2.0, -1.5, 'r'], # 'r' 為顏色設定，請參考網站修改成自己想要的
                      "stage2": [-1.5, -1.0, 'y'],
                      "stage3": [-1.0, 0.0, 'g'],
                      "stage4": [0.0, 1.0, 'c']})


# %% Reading all of data path
# using a recursive loop to traverse each folder
# and find the file extension has .csv
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
        
    return csv_file_list
# %% EMG data processing
def EMG_processing(raw_data, smoothing="lowpass"):
    '''
    Parameters
    ----------
    raw_data : pandas.DataFrame
        給予欲處理之資料.
    smoothing : str, optional
        設定 smoothing method,分別為 lowpass, rms, moving. The default is 'lowpass'
        
    Returns
    -------
    moving_data : pandas.DataFrame.
        回傳平滑處理後之資料
    bandpass_filtered_data  : pandas.DataFrame.
        回傳僅bandpass filting後之資料

    -------
    程式邏輯：
    1. 預處理：
        1.1. 計算各sensor之採樣頻率與資料長度，最後預估downsample之資料長度，並使用最小值
        1.2 計算各sensor之採樣截止時間，並做平均
        1.3 創建資料貯存之位置： bandpass, lowpass, rms, moving mean
    2. 濾波： 先濾波，再降採樣
        2.1 依各sensor之採樣頻率分開濾波
        2.2 降採樣
    3. 插入時間軸
            
    '''
    # raw_data = emg_data

    # 測試用
    # raw_data = pd.read_csv(r"C:\Users\Public\BenQ\myPyCode\Lab\Raw_Data\Method_2\a04\MVC\a04_MVC_R Biceps.csv",
    #                     encoding='UTF-8')
    # 1.  -------------前處理---------------------------
    # 1.1.-------------計算所有sensor之採樣頻率----------
    # 找尋EMG 訊號所在欄位 num_columns
    num_columns = []
    for i in range(len(raw_data.columns)):
        for ii in range(len(raw_data.columns[raw_data.columns.str.contains("EMG")])):
            if raw_data.columns[i] == raw_data.columns[raw_data.columns.str.contains("EMG")][ii]:
                num_columns.append(i)
    print("處理 EMG 訊號，總共", len(num_columns), "條肌肉， 分別為以下欄位")
    print(raw_data.columns[raw_data.columns.str.contains("EMG")])
    # 計算各採樣頻率與計算downsample所需的位點數，並取最小的位點數
    Fs = []
    freq = []
    data_len = []
    all_stop_time = []
    for col in range(len(num_columns)):
        # print(col)
        data_time = raw_data.iloc[:,num_columns[col]-1].dropna()
        # 採樣頻率計算取前十個時間點做差值平均
        freq.append(int(1/np.mean(np.array(data_time[2:11])-np.array(data_time[1:10]))))
        # print(freq)
        # 找到第一個 Raw data 不等於零的位置
        # 計算數列中0的數量
        '''
        找到某一數列為何計算出來會都是NAN
        有一個sensor的紀錄時間特別短，可能是儀器問題
        E:\python\EMG_Data\Shooting_data_20230617\202305 Shooting\Raw_Data\\\A2\MVC\A2_MVC_Rep_7.6_QL.csv
        '''
        data_len.append((raw_data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))
        
        Fs.append(int((len(raw_data.iloc[:, num_columns[col]]) - (raw_data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))))
        # 取截止時間
        all_stop_time.append(raw_data.iloc[(len(raw_data.iloc[:, num_columns[col]]) - (raw_data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))-1 ,
                                       num_columns[col]-1])
    # 1.2.-------------計算平均截止時間------------------
    # 丟棄NAN的值，並選擇最小值
    min_stop_time = np.min([x for x in all_stop_time if math.isnan(x) == False])
    while max(all_stop_time) - min(all_stop_time) > 1:
        print("兩 sensor 數據時間差超過 1 秒")
        print("將使用次短時間的 Sensor 作替代")
        all_stop_time.remove(min_stop_time)
        Fs.remove(min(Fs))
        min_stop_time = np.min([x for x in all_stop_time if math.isnan(x) == False])
    # 1.3.-------------創建儲存EMG data的矩陣------------
    # bandpass filter used in signal
    bandpass_filtered_data = pd.DataFrame(np.zeros([int(min(Fs)//(min(freq)/down_freq)), len(num_columns)]),
                            columns=raw_data.iloc[:, num_columns].columns)
    lowpass_filtered_data = pd.DataFrame(np.zeros([int(min(Fs)//(min(freq)/down_freq)), len(num_columns)]),
                            columns=raw_data.iloc[:, num_columns].columns)
    # 設定 moving mean 的矩陣大小、欄位名稱
    window_width = int(time_of_window*np.floor(down_freq))
    moving_data = pd.DataFrame(np.zeros([int((np.shape(bandpass_filtered_data)[0] - window_width)/  ((1-overlap_len)*window_width)) + 1,
                                         np.shape(bandpass_filtered_data)[1]]),
                               columns=raw_data.iloc[:, num_columns].columns)
    # 設定 Root mean square 的矩陣大小、欄位名稱
    rms_data = pd.DataFrame(np.zeros([int((np.shape(bandpass_filtered_data)[0] - window_width)/  ((1-overlap_len)*window_width)) + 1,
                                      np.shape(bandpass_filtered_data)[1]]),
                            columns=raw_data.iloc[:, num_columns].columns)
    # 2.2 -------------分不同sensor處理各自的採樣頻率----
    for col in range(len(num_columns)):
        # 取採樣時間的前十個採樣點計算採樣頻率
        sample_freq = int(1/np.mean(np.array(raw_data.iloc[2:11, (num_columns[col] - 1)])
                                    - np.array(raw_data.iloc[1:10, (num_columns[col] - 1)])))
        decimation_factor = sample_freq / down_freq
        isnan = np.where(np.isnan(raw_data.iloc[:(np.shape(raw_data)[0] - data_len[col]), num_columns[col]]))
        # 預處理資料,判斷資料中是否有 nan, 並將 nan 取代為 0 
        if isnan[0].size == 0:
        # 計算Bandpass filter
            data = raw_data.iloc[:(np.shape(raw_data)[0] - data_len[col]), num_columns[col]].values
        # 設定給斷訊超過 0.1 秒的 sensor 警告
        elif isnan[0].size > 0.1*sample_freq:
            logging.warning(str(raw_data.columns[num_columns[col]] + "sensor 總訊號斷訊超過 0.1 秒，"))
            data = raw_data.iloc[:(np.shape(raw_data)[0] - data_len[col]), num_columns[col]].values
        else:
            logging.warning(str("共發現 " + str(isnan[0].size) + " 個缺值,位置為 " + str(isnan[0])))
            logging.warning("已將 NAN 換為 0")
            data = raw_data.iloc[:(np.shape(raw_data)[0] - data_len[col]), num_columns[col]].fillna(0)
        # 由於各截止時間不同，所以找出最小的截止時間，並將其他較長時間的 sensor，都截成短的
        # 找出最小的時間，並且找出所有欄位數據中最接近的索引值
        end_index = np.abs(raw_data.iloc[:, num_columns[col]-1].fillna(0) - min_stop_time).argmin()
        data = data[:end_index+1]
        # 進行 bandpass filter
        bandpass_sos = signal.butter(2, bandpass_cutoff,  btype='bandpass', fs=sample_freq, output='sos')
        bandpass_filtered = signal.sosfiltfilt(bandpass_sos, data)
        # 取絕對值，將訊號翻正
        abs_data = abs(bandpass_filtered)
        # ------linear envelop analysis-----------                          
        # ------lowpass filter parameter that the user must modify for your experiment        
        lowpass_sos = signal.butter(2, lowpass_freq, btype='low', fs=sample_freq, output='sos')        
        lowpass_filtered = signal.sosfiltfilt(lowpass_sos, abs_data)
        # 2.3.------resample data to 1000Hz-----------
        # 降採樣資料，並將資料儲存在矩陣當中
        resample_length = int(len(data)//(decimation_factor))
        bandpass_filtered = signal.resample(bandpass_filtered, resample_length)
        bandpass_filtered_data.iloc[:resample_length, col] = bandpass_filtered
        abs_data = signal.resample(abs_data, resample_length)
        lowpass_filtered = signal.resample(lowpass_filtered, resample_length)
        lowpass_filtered_data.iloc[:resample_length, col] = lowpass_filtered
        # -------Data smoothing. Compute Moving mean
        # window width = window length(second)*sampling rate
        
        for ii in range(np.shape(moving_data)[0]):
            data_location = int(ii*(1-overlap_len)*window_width)
            # print(data_location, data_location+window_width_rms)
            moving_data.iloc[int(ii), col] = (np.sum((abs_data[data_location:data_location+window_width])**2)
                                          /window_width)
            
        # -------Data smoothing. Compute RMS
        # The user should change window length and overlap length that suit for your experiment design
        # window width = window length(second)*sampling rate
        for ii in range(np.shape(rms_data)[0]):
            data_location = int(ii*(1-overlap_len)*window_width)
            # print(data_location, data_location+window_width_rms)
            rms_data.iloc[int(ii), col] = np.sqrt(np.sum((abs_data[data_location:data_location+window_width])**2)
                                          /window_width)
                
    # 3. -------------插入時間軸-------------------
    # 定義bandpass filter的時間
    bandpass_time_index = np.linspace(0, min_stop_time, np.shape(bandpass_filtered_data)[0])
    bandpass_filtered_data.insert(0, 'time', bandpass_time_index)
    # 丟棄 bandpass_filtered_data 欄位內包含 0 的行
    bandpass_filtered_data = bandpass_filtered_data.loc[~(bandpass_filtered_data==0).any(axis=1)]
    # 定義lowpass filter的時間
    lowpass_time_index = np.linspace(0, min_stop_time, np.shape(lowpass_filtered_data)[0])
    lowpass_filtered_data.insert(0, 'time', lowpass_time_index)
    # 丟棄 lowpass_filtered_data 欄位內包含 0 的行
    lowpass_filtered_data = lowpass_filtered_data.loc[~(lowpass_filtered_data==0).any(axis=1)]
    # 定義moving average的時間
    moving_time_index = np.linspace(0, min_stop_time, np.shape(moving_data)[0])
    moving_data.insert(0, 'time', moving_time_index)
    # 丟棄 moving_data 欄位內包含 0 的行
    moving_data = moving_data.loc[~(moving_data==0).any(axis=1)]
    # 定義RMS DATA的時間
    rms_time_index = np.linspace(0, min_stop_time, np.shape(rms_data)[0])
    rms_data.insert(0, 'time', rms_time_index)
    # 設定 return 參數
    if smoothing == "lowpass":
        return lowpass_filtered_data, bandpass_filtered_data
    elif smoothing == "rms":
        return rms_data, bandpass_filtered_data
    elif smoothing == "moving":
        return moving_data, bandpass_filtered_data  

# %% to find maximum MVC value

def Find_MVC_max(MVC_folder, MVC_save_path):
    # MVC_folder = r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\Processing\MVC'
    # MVC_save_path = r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\Processing'
    MVC_file_list = os.listdir(MVC_folder)
    MVC_data = pd.read_excel(MVC_folder + '\\' + MVC_file_list[0], engine='openpyxl')
    find_max_all = []
    Columns_name = MVC_data.columns
    Columns_name = Columns_name.insert(0, 'FileName')
    find_max_all = pd.DataFrame(find_max_all, columns=Columns_name)
    for i in MVC_file_list:
        MVC_file_path = MVC_folder + '\\' + i
        MVC_data = pd.read_excel(MVC_file_path)
        find_max = MVC_data.max(axis=0)
        find_max = pd.DataFrame(find_max)
        find_max = np.transpose(find_max)
        find_max.insert(0, 'FileName', i)
        # find_max_all = find_max_all.append(find_max)
        find_max_all = pd.concat([find_max_all, find_max], axis=0, ignore_index=True)
    # find maximum value from each file
    MVC_max = find_max_all.max(axis=0)
    MVC_max[0] = 'Max value'
    MVC_max = pd.DataFrame(MVC_max)
    MVC_max = np.transpose(MVC_max)
    # find_max_all = find_max_all.append(MVC_max)
    find_max_all = pd.concat([find_max_all, MVC_max], axis=0, ignore_index=True)
    # writting data to EXCEl file
    find_max_name = MVC_save_path + '\\' + MVC_save_path.split('\\')[-2] + "_" \
        + MVC_save_path.split('\\')[-1] + '_all_MVC.xlsx'
    DataFrame(find_max_all).to_excel(find_max_name, sheet_name='Sheet1', index=False, header=True)

# %% iMVC calculate

def iMVC_calculate(MVC_file, shooting_folder, save_file_path):
    '''

    Parameters
    ----------
    MVC_file : str
        給定計算完之MVC檔案路徑.
    shooting_folder : TYPE
        給定motion data的資料夾路徑.
    save_file_path : TYPE
        存檔路徑.

    Returns
    -------
    None.

    '''
    # to obtain file name from the parent folder and make a list
    shooting_file_list = os.listdir(shooting_folder)
    # to define MVC value
    MVC_value = pd.read_excel(MVC_file)
    MVC_value = MVC_value.iloc[-1, 1:]
    MVC_value = MVC_value.iloc[1:]
    for i in shooting_file_list:
        # loading shooting EMG value
        shooting_file_name = shooting_folder + '\\' + shooting_file_list[i]
        # load EMG data
        shooting_data = pd.read_excel(shooting_file_name)
        # trunkcate specific period
        shooting_EMG = shooting_data.iloc[:, :]    
        # calculate iMVC data
        shooting_iMVC = np.divide(shooting_EMG, MVC_value)*100
        shooting_iMVC.insert(0, 'time', shooting_data.iloc[:,0])
        # writting iMVC data in a EXCEL
        save_iMVC_name = save_file_path + 'iMVC_' + shooting_file_list[i]
        DataFrame(shooting_iMVC).to_excel(save_iMVC_name, sheet_name='Sheet1', index=False, header=True)
  
# %% to find the release timing

def find_release_time(folder_path, save_path, save_fig=True):
    # 讀所有.csv file
    file_list = Read_File(folder_path, '.csv')
    release_timing_list = pd.DataFrame(columns = ["FileName", "Time Frame(降1000Hz)", "Time"])
    # 繪圖用
    n = int(math.ceil(len(file_list) /2))
    plt.figure(figsize=(2*n,10))
    fig, axs = plt.subplots(n, 2, figsize = (10,12))
    for ii in range(len(file_list)):
        if os.path.splitext(file_list[ii])[1] == ".csv":
            filepath, tempfilename = os.path.split(file_list[ii])
            data = pd.read_csv(file_list[ii])
            # to find R EXTENSOR CARPI RADIALIS: ACC X data [43]
            Extensor_ACC = data.iloc[:, release_acc]
            # acc sampling rate
            acc_freq = int(1/np.mean(np.array(data.iloc[2:11, (release_acc - 1)])
                                     - np.array(data.iloc[1:10, (release_acc - 1)])))
            # there should change with different subject
            peaks, _ = signal.find_peaks(Extensor_ACC*-1, height = release_peak)
            # 繪圖用，計算子圖編號
            x, y = ii - n*math.floor(abs(ii)/n), math.floor(abs(ii)/n)
            if peaks.any():
                # Because ACC sampling frequency is 148Hz and EMG is 2000Hz
                release_time = data.iloc[peaks[0], release_acc-1]
                release_index = int((peaks[0]/acc_freq)*1000)
                # 畫圖
                axs[x, y].plot(data.iloc[:, release_acc-1], Extensor_ACC)
                axs[x, y].set_title(tempfilename, fontsize = 12)
                # 設定科學符號 : 小數點後幾位數
                axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                axs[x, y].plot(release_time,Extensor_ACC[peaks[0]], marker = "x", markersize=10)
                axs[x, y].annotate(release_time, xy = (0, 0), fontsize = 16, color='b')
            else:
                release_index = "Nan"
                release_time = "Nan"
                axs[x, y].plot(data.iloc[:, release_acc-1], Extensor_ACC)
                axs[x, y].set_title(tempfilename, fontsize = 12)
                # 設定科學符號 : 小數點後幾位數
                axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
                axs[x, y].annotate('Can not find', xy = (0, 0), fontsize = 16, color='r')
            # to create DataFrame that easy to write data in a excel
            # ii = ii.replace('.', '_')
            release_time_number = pd.DataFrame([file_list[ii], release_index, release_time])
            release_time_number = np.transpose(release_time_number)
            release_time_number.columns = ["FileName", "Time Frame(降1000Hz)", "Time"]
            release_timing_list = pd.concat([release_timing_list, release_time_number], ignore_index=True)

    # 設定整張圖片之參數
    plt.suptitle(str("release time: " + save_path.split('\\')[-1]), fontsize = 16)
    plt.tight_layout()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("time (second)", fontsize = 14)
    plt.ylabel("acc (g)", fontsize = 14)
    plt.savefig(str(save_path + '\\' + save_path.split('\\')[-2] + "_" \
                    + save_path.split('\\')[-1] + "_ReleaseTiming.jpg"),
                dpi=100)
    plt.show()
    # writting data to a excel file
    save_iMVC_name = save_path + '\\' + save_path.split('\\')[-2] + "_" \
        + save_path.split('\\')[-1] + '_ReleaseTiming.xlsx' 
    DataFrame(release_timing_list).to_excel(save_iMVC_name, sheet_name='Sheet1', index=False, header=True)
    return release_index

# %% 傅立葉轉換與畫圖
# 計算傅立葉轉換
def Fourier_plot(raw_data, savepath, filename):
    '''

    Parameters
    ----------
    data : pandas,DataFrame
        給定預計算傅立葉轉換之資料.
    savepath : str
        給定預存擋的資料夾路徑.
    filename : str
        現正運算之資料檔名.

    Returns
    -------
    None.

    '''
    # raw_data = data
    save = savepath + '\\FFT_' + filename + ".jpg"
    num_columns = []
    for i in range(len(raw_data.columns)):
        for ii in range(len(raw_data.columns[raw_data.columns.str.contains("EMG")])):
            if raw_data.columns[i] == raw_data.columns[raw_data.columns.str.contains("EMG")][ii]:
                num_columns.append(i)
    n = int(math.ceil(len(num_columns)/2))
    '''
    # 是否先濾波
    
    '''
    
    processing_data, bandpass_filtered_data = EMG_processing(raw_data, smoothing=smoothing_method)
    # due to our data type is series, therefore we need to extract value in the series
    # --------畫圖用與計算FFT----------------------
    # --------------------------------------------
    # 設定圖片大小
    plt.figure(figsize=(2*n+1,10))
    fig, axs = plt.subplots(n, 2, figsize = (10,12))
    for col in range(len(num_columns)):
        x, y = col - n*math.floor(abs(col)/n), math.floor(abs(col)/n)
        # print(col)
        # 設定資料時間
        # 採樣頻率計算 : 取前十個時間點做差值平均
        freq = int(1/np.mean(np.array(raw_data.iloc[2:11, num_columns[col]-1])-np.array(raw_data.iloc[1:10, num_columns[col]-1])))
        data_len = (np.shape(raw_data)[0] - (raw_data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))
        # convert sampling rate to period
        # 計算取樣週期
        T = 1/freq;

        # ---------------------開始計算 FFT -----------------------------------
        # 1. 先計算 bandpass filter
        isnan = np.where(np.isnan(raw_data.iloc[:data_len, num_columns[col]]))
        if isnan[0].size == 0:
        # 計算Bandpass filter
            processing_data, bandpass_filtered_data = EMG_processing(raw_data.iloc[:data_len, num_columns[col]].values,
                                                                     smoothing=smoothing_method)
            # bandpass_sos = signal.butter(1, bandpass_cutoff,  btype='bandpass', fs=freq, output='sos')
            # bandpass_filtered = signal.sosfiltfilt(bandpass_sos,
            #                                        raw_data.iloc[:data_len, num_columns[col]].values)
        # 設定給斷訊超過 0.1 秒的 sensor 警告
        elif isnan[0].size > 0.1*freq:
            logging.warning(str(raw_data.columns[num_columns[col]] + "sensor 總訊號斷訊超過 0.1 秒，"))
            processing_data, bandpass_filtered_data = EMG_processing(raw_data.iloc[:data_len, num_columns[col]].values,
                                                                     smoothing=smoothing_method)
            # bandpass_sos = signal.butter(1, bandpass_cutoff,  btype='bandpass', fs=freq, output='sos')
            # bandpass_filtered = signal.sosfiltfilt(bandpass_sos,
            #                                        raw_data.iloc[:(np.shape(raw_data)[0] - data_len[col]), num_columns[col]].values)
        else:
            logging.warning(str("共發現 " + str(isnan[0].size) + " 個缺值,位置為 " + str(isnan[0])))
            logging.warning("已將 NAN 換為 0")

            # bandpass_sos = signal.butter(1, bandpass_cutoff,  btype='bandpass', fs=freq, output='sos')
            # bandpass_filtered = signal.sosfiltfilt(bandpass_sos,
            #                                        raw_data.iloc[:data_len, num_columns[col]].fillna(0))
        # 2. 資料前處理
        # 計算資料長度
        N = int(np.prod(bandpass_filtered_data.shape[0]))#length of the array
        N2 = 2**(N.bit_length()-1) #last power of 2
        # convert sampling rate to period 
        # 計算取樣週期
        T = 1/freq;
        N = N2 #truncate array to the last power of 2
        xf = np.linspace(0.0, np.ceil(1.0/(2.0*T)), N//2)
        # print("# caculate Fast Fourier transform")
        # print("# Samples length:",N)
        # print("# Sampling rate:",freq)
        # 開始計算 FFT   
        yf = fft(bandpass_filtered_data, N)
        axs[x, y].plot(xf, 2.0/N * abs(yf[0:int(N/2)]))
        axs[x, y].set_title(raw_data.columns[num_columns[col]], fontsize = 16)
        # 設定科學符號 : 小數點後幾位數
        axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
        axs[x, y].set_xlim(0, 500)
    # 設定整張圖片之參數
    plt.suptitle(str("FFT Analysis " + filename), fontsize = 16)
    plt.tight_layout()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Frequency (Hz)", fontsize = 14)
    plt.ylabel("Power", fontsize = 14)
    # plt.savefig(save, dpi=200, bbox_inches = "tight")
    plt.show()
    

# %% 畫圖用

def plot_plot(data, savepath, filename, filter_type):
    save = savepath + '\\' + filter_type + filename + ".jpg"
    n = int(math.ceil((np.shape(data)[1] - 1) /2))
    plt.figure(figsize=(2*n+1,10))
    fig, axs = plt.subplots(n, 2, figsize = (10,12), sharex='col')
    for i in range(np.shape(data)[1]-1):
        x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
        # 設定子圖之參數
        axs[x, y].plot(data.iloc[:, 0], data.iloc[:, i+1])
        axs[x, y].set_title(data.columns[i+1], fontsize=16)
        # 設定科學符號 : 小數點後幾位數
        axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
    # 設定整張圖片之參數
    plt.suptitle(filename + filter_type, fontsize = 16)
    plt.tight_layout()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("time (second)", fontsize = 14)
    plt.ylabel("Voltage (V)", fontsize = 14)
    plt.savefig(save, dpi=200, bbox_inches = "tight")
    plt.show()
    
# %%  畫 mean std colud

# 設定顏色格式
# 參考 Qualitative: https://matplotlib.org/stable/tutorials/colors/colormaps.html

def mean_std_cloud(data_path, savepath, filename, smoothing, release):
    '''

    Parameters
    ----------
    data_path : str
        給定motion data的資料夾路徑.
    savepath : str
        存檔路徑.
    filename : str
        受試者資料夾名稱，ex: S1.
        
    Returns
    -------
    None.

    '''
    # data_path = motion_folder_path
    # filename = processing_folder_list[i]
    # release = [release[0]/down_freq, release[1]/down_freq]
    # savepath = save_path
    
    file_list = Read_File(data_path, ".xlsx", subfolder=False)
    
    type1, type2 = [], []
    # 2. 找尋特定檔案名稱
    for ii in range(len(file_list)):
        if before_fatigue in file_list[ii]:
            type1.insert(0, file_list[ii])
        if after_fatigue in file_list[ii]:
            type2.insert(0, file_list[ii])
    # 說明兩組資料各幾筆
    print("before fatigue: ", len(type1))
    print("after_fatigue: ", len(type2))
    # read example data
    example_data = pd.read_excel(type1[0])

    # create multi-dimension matrix
    type1_dict = np.zeros(((np.shape(example_data)[1] - 1), # muscle name without time
                           (np.shape(example_data)[0]), # time length
                           len(type1)))                 # subject number

    for ii in range(len(type1)):
        # read data
        type1_data = pd.read_excel(type1[ii])
        for iii in range(np.shape(example_data)[1] - 1): # exclude time
            type1_dict[iii, :, ii] = type1_data[example_data.columns[iii + 1]]
    
    type2_dict = np.zeros(((np.shape(example_data)[1] - 1), # muscle name without time
                           (np.shape(example_data)[0]), # time length
                           len(type2)))                 # subject number

    for ii in range(len(type2)):
        type2_data = pd.read_excel(type2[ii])
        for iii in range(np.shape(example_data)[1] - 1): # exclude time
            type2_dict[iii, :, ii] = type2_data[example_data.columns[iii + 1]]
    
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
    for i in range(np.shape(type2_dict)[0]):
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
        axs[x, y].set_title(example_data.columns[i+1], fontsize=12)
        axs[x, y].legend(loc="lower left") # 圖例位置
        axs[x, y].grid(True, linestyle='-.')
        # 畫放箭時間
        axs[x, y].set_xlim(-(release[0]), release[1])
        axs[x, y].axvline(x=0, color = 'darkslategray', linewidth=1, linestyle = '--')

        # 畫花括號
        curlyBrace(fig, axs[x, y], [shooting_time["stage1"][0], yy], [shooting_time["stage1"][1], yy],
                   0.05, bool_auto=True, str_text="", color=shooting_time["stage1"][2],
                   lw=2, int_line_num=1, fontdict=font)
        curlyBrace(fig, axs[x, y], [shooting_time["stage2"][0], yy], [shooting_time["stage2"][1], yy],
                   0.05*2.4, bool_auto=True, str_text="", color=shooting_time["stage2"][2],
                   lw=2, int_line_num=1, fontdict=font)
        curlyBrace(fig, axs[x, y], [shooting_time["stage3"][0], yy], [shooting_time["stage3"][1], yy],
                   0.05*6, bool_auto=True, str_text="", color=shooting_time["stage3"][2],
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
    
# %% 中頻率

def median_frquency(raw_data, duration, fig_svae_path, filename):
    """
    
    Parameters
    ----------
    raw_data : str
        給定 fatigue data 的資料夾路徑.
    duration : float
        Unit : second, 給定每次計算資料時間長度.
    fig_svae_path : str
        存檔路徑.
    filename : str
        檔案名稱.
    
    Returns
    -------
    None.
    
    程式流程 :
        1. 計算每個 sensor column 的採樣頻率
        2. bandpass data with each columns
    參考資料 :
        1. https://dsp.stackexchange.com/questions/85683/how-to-find-median-frequency-of-binned-signal-fft
    """
    save = fig_svae_path + '\\MedianFreq_' + filename + ".jpg"
    print("執行 Fatigue Analysis 檔名: ", filename)

    num_columns = []
    for i in range(len(raw_data.columns)):
        for ii in range(len(raw_data.columns[raw_data.columns.str.contains("EMG")])):
            if raw_data.columns[i] == raw_data.columns[raw_data.columns.str.contains("EMG")][ii]:
                num_columns.append(i)
    # 繪圖用
    n = int(math.ceil((len(num_columns)) /2))
    # 設定圖片大小
    plt.figure(figsize=(2*n+1,10))
    fig, axs = plt.subplots(n, 2, figsize = (10,12))
    # 計算各採樣頻率與計算downsample所需的位點數，並取最小的位點數
    for col in range(len(num_columns)):
        x, y = col - n*math.floor(abs(col)/n), math.floor(abs(col)/n)
        # print(col)
        # 設定資料時間
        # 採樣頻率計算 : 取前十個時間點做差值平均
        freq = int(1/np.mean(np.array(raw_data.iloc[2:11, num_columns[col]-1])-np.array(raw_data.iloc[1:10, num_columns[col]-1])))
        data_len = (np.shape(raw_data)[0] - (raw_data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))
        # convert sampling rate to period
        # 計算取樣週期
        T = 1/freq;
        # ---------------------開始計算 FFT -----------------------------------
        # 1. 先計算 bandpass filter
        bandpass_sos = signal.butter(2, bandpass_cutoff,  btype='bandpass', fs=freq, output='sos')
        # 計算資料長度，從後面數來，直到欄位裡面第一個"非0值"出現
        bandpass_data = signal.sosfiltfilt(bandpass_sos, raw_data.iloc[:data_len, num_columns[col]])
        # 2. 資料前處理
        # 計算資料長度
        N = int(np.prod(bandpass_data.shape[0]))#length of the array
        N2 = 2**(N.bit_length()-1) #last power of 2
        # convert sampling rate to period 
        # 計算取樣週期
        T = 1/freq;
        N = N2 #truncate array to the last power of 2
        xf = np.linspace(0.0, np.ceil(1.0/(2.0*T)), N//2)
        # 3. 每一個 duration 計算一次 FFT
        med_freq_list = []
        for i in range(int(np.ceil(len(bandpass_data)/freq))):
            # 判斷擷取資料是否在整數點
            if freq*(i+1) < len(bandpass_data):
                # 3. 計算 FFT
                # 計算資料長度
                N = int(np.prod(bandpass_data[i*freq:(i+1)*freq].shape[0]))#length of the array
                N2 = 2**(N.bit_length()-1) #last power of 2
                N = N2 #truncate array to the last power of 2
                xf = np.linspace(0.0, np.ceil(1.0/(2.0*T)), N//2)
                # print(i*freq, (i+1)*freq)
                yf = fft(bandpass_data[i*freq:(i+1)*freq])      
                # plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
                # 計算每個 duration 的 FFT 總和
                all_y = np.sum(2.0/N * np.abs(yf[:N//2]))
                med_y = 0
                for ii in range(len(yf[0:int(N/2)])):
                    med_y = med_y + 2.0/N * np.abs(yf[ii])
                    if med_y >= all_y/2:
                        med_freq_list.append(xf[ii])
                        break
            else:
                yf = fft(bandpass_data[i*freq:]) 
                all_y = np.sum(2.0/N * np.abs(yf[:N//2]))
                med_y = 0
                for ii in range(len(yf[0:int(N/2)])):
                    med_y = med_y + 2.0/N * np.abs(yf[ii])
                    if med_y >= all_y/2:
                        med_freq_list.append(xf[ii])
                        break
        # 畫中頻率圖
        axs[x, y].plot(med_freq_list)
        axs[x, y].set_title(raw_data.columns[num_columns[col]], fontsize = 16)
        # 畫水平線
        axs[x, y].axhline(y=np.mean(med_freq_list), color = 'darkslategray', linewidth=1, linestyle = '--')
        # 設定科學符號 : 小數點後幾位數
        axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
    # 設定整張圖片之參數
    plt.suptitle(str("Fatigue Analysis (Median frequency) " + filename), fontsize = 16)
    plt.tight_layout()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("time (second)", fontsize = 14)
    plt.ylabel("Frequency (Hz)", fontsize = 14)
    plt.savefig(save, dpi=200, bbox_inches = "tight")
    plt.show()
# %% 刪除指定資料夾中的特定檔案格式

def remove_file(folder_path, file_format):
    """
    Parameters
    ----------
    folder_path : str
        給定預清除檔案的資料夾路徑.
    file_format : str
        給定預清除的檔案格式.

    Returns
    -------
    None.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(file_format):
            print(filename)
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)     
# %%
def remove_specific_string_from_list(input_list):
    # 定義正規表達式模式
    pattern1 = re.compile(r': EMG ')
    pattern2 = re.compile(r'\d')
    
    # 對列表的每個元素應用處理
    result_list = []
    for item in input_list:
        # 移除特定字串 EMG1、EMG2、EMG3
        item = pattern1.sub('', item)
        
        # 移除不特定數字
        item = pattern2.sub('', item)
        
        result_list.append(item)
        
    return result_list
# %%
def compare_mean_std_cloud(v1_data_path, v2_data_path, savepath, filename,
                           smoothing, release, self_oreder=False):
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

    # data_path = r'D:\python\EMG_Data\To HSIN\EMG\Processing_Data\Method_1\C06\\'
    # v1_data_path = data_path + 'test1\\data\\motion'
    # v2_data_path = data_path + 'test2\\data\\motion'
    # 找出所有資料夾下的 .xlsx 檔案
    v1_file_list = Read_File(v1_data_path, ".xlsx", subfolder=False)
    
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


#%% wavelet analysis
# import pywt
# import numpy as np
# import matplotlib.pyplot as plt 
# import pandas as pd

# raw_data = pd.read_csv(r"E:/python/EMG_Data/Raw_Data/Method_1/S01/motion/S2_Fatigue_Rep_1.12.csv", encoding='UTF-8')

# moving_data, rms_data, lowpass_filtered_data, bandpass_filtered_data = EMG_processing(raw_data)

# # 畫原始信號
# plt.figure()
# plt.plot(bandpass_filtered_data.iloc[:,1])
# # stft 處理
# # fs:时间序列的采样频率,  nperseg:每个段的长度   noverlap:段之间重叠的点数。如果没有则noverlap=nperseg/2
# f, t, nd = signal.stft(bandpass_filtered_data.iloc[:,1], fs = 1000, window ='hann', nperseg = 15, noverlap = 5)
# f, t, Zxx = signal.stft(bandpass_filtered_data.iloc[:,3], fs=1000, nperseg=1000)
# Zxx_min, Zxx_max = -np.abs(Zxx).max(), np.abs(Zxx).max()
# plt.pcolormesh(t, f, np.abs(Zxx), vmin = 0, vmax = Zxx_max)
# plt.colorbar()
# plt.title('STFT')
# plt.ylabel('frequency')
# plt.xlabel('time')
# plt.show()


# # 連續小波轉換
# sampling_rate = 1000
# wavename = 'cgau8'
# totalscal = 256
# # 中心频率
# fc = pywt.central_frequency(wavename)
# # 计算对应频率的小波尺度
# cparam = 2 * fc * totalscal
# scales = cparam / np.arange(totalscal, 1, -1)
# [cwtmatr, frequencies] = pywt.cwt(bandpass_filtered_data.iloc[:,3], scales, wavename, 1.0 / sampling_rate)
# plt.figure(figsize=(8, 4))
# plt.subplot(211)
# t = np.arange(0, 600, 1.0)
# plt.plot(bandpass_filtered_data.iloc[:,0], bandpass_filtered_data.iloc[:,3])
# plt.xlabel(u"time(s)")
# plt.subplot(212)
# plt.contourf(bandpass_filtered_data.iloc[:,0], frequencies, abs(cwtmatr))
# plt.ylabel(u"freq(Hz)")
# plt.xlabel(u"time(s)")
# plt.subplots_adjust(hspace=0.4)
# plt.colorbar()
# plt.show()

# # 離散小波轉換
# wavename = 'db4'
# cA, cD = pywt.dwt(bandpass_filtered_data.iloc[:600,1], wavename)
# ya = pywt.idwt(cA, None, wavename,'smooth') # approximated component
# yd = pywt.idwt(None, cD, wavename,'smooth') # detailed component
# x = range(len(ya))
# plt.figure(figsize=(12,9))
# plt.subplot(311)
# plt.plot(x, bandpass_filtered_data.iloc[:600,1])
# plt.title('original signal')
# plt.subplot(312)
# plt.plot(x, ya)
# plt.title('approximated component')
# plt.subplot(313)
# plt.plot(x, yd)
# plt.title('detailed component')
# plt.tight_layout()
# plt.show()



# from matplotlib.font_manager import FontManager
# import subprocess

# mpl_fonts = set(f.name for f in FontManager().ttflist)

# print('all font list get from matplotlib.font_manager:')
# for f in sorted(mpl_fonts):
#     print('\t' + f)

            
# import numpy as np
# import matplotlib.pyplot as plt
# import pywt
# import mne
# mne.set_log_level(False)
# ######################################################连续小波变换##########
# # totalscal小波的尺度，对应频谱分析结果也就是分析几个（totalscal-1）频谱
# def TimeFrequencyCWT(data,fs,totalscal,wavelet='cgau8'):
#     # 采样数据的时间维度
#     t = np.arange(data.shape[0])/fs
#     # 中心频率
#     wcf = pywt.central_frequency(wavelet=wavelet)
#     # 计算对应频率的小波尺度
#     cparam = 2 * wcf * totalscal
#     scales = cparam/np.arange(totalscal, 1, -1)
#     # 连续小波变换
#     [cwtmatr, frequencies] = pywt.cwt(data, scales, wavelet, 1.0/fs)
#     # 绘图
#     plt.figure(figsize=(8, 4))
#     plt.subplot(211)
#     plt.plot(t, data)
#     plt.xlabel(u"time(s)")
#     plt.title(u"Time spectrum")
#     plt.subplot(212)
#     plt.contourf(t, frequencies, abs(cwtmatr))
#     plt.ylabel(u"freq(Hz)")
#     plt.xlabel(u"time(s)")
#     plt.subplots_adjust(hspace=0.4)
#     plt.show()
 
 
# if __name__ == '__main__':
#     # 读取筛选好的epoch数据
#     epochsCom = mne.read_epochs(r'F:\BaiduNetdiskDownload\BCICompetition\BCICIV_2a_gdf\Train\Fif\A02T_epo.fif')
#     dataCom = epochsCom[10].get_data()[0][0]
#     TimeFrequencyCWT(dataCom, fs=250, totalscal=10, wavelet='cgau8')

# %% https://www.jb51.net/article/209864.htm
# https://cloud.tencent.com/developer/article/1806429
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import pywt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# # 解决负号显示问题
# # 解决保存图像是负号'-'显示为方块的问题
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams.update({'text.usetex': False, 'font.family': 'serif', 'font.serif': 'cmr10', 'mathtext.fontset': 'cm'})
# font1 = {'family': 'SimHei', 'weight': 'normal', 'size': 12}
# font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
# label = {'family': 'SimHei', 'weight': 'normal', 'size': 15}
# xlsx_path = "../小波能量谱作图.xlsx"
# sheet_name = "表名"
# data_arr = pd.read_excel(xlsx_path, sheet_name=sheet_name)
# column_name = '列名'
# row = 1024
# y = data_arr[column_name][0:row]
# x = data_arr['time'][0:row]
# scale = np.arange(1, 50)
# wavelet = 'gaus1' # 'morl'  'gaus1'  小波基函数
# # 时间-尺度小波能量谱
# def time_scale_spectrum():
#     # np.arange(1, 31) 第一个参数必须 >=1     'morl'  'gaus1'
#     coefs, freqs = pywt.cwt(y, scale, wavelet)
#     scale_freqs = np.power(freqs, -1)  # 对频率freqs 取倒数变为尺度
#     fig = plt.figure(figsize=(5, 4))
#     ax = Axes3D(fig)
#     # X:time   Y:Scale   Z:Amplitude
#     X = np.arange(0, row, 1)  # [0-1023]
#     Y = scale_freqs
#     X, Y = np.meshgrid(X, Y)
#     Z = abs(coefs)
#     # 绘制三维曲面图
#     ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
#     # 设置三个坐标轴信息
#     ax.set_xlabel('$Mileage/km$', color='b', fontsize=12)
#     ax.set_ylabel('$Scale$', color='g', fontsize=12)
#     ax.set_zlabel('$Amplitude/mm$', color='r', fontsize=12)
#     plt.draw()
#     plt.show()
# # 时间小波能量谱
# def time_spectrum():
#     coefs, freqs = pywt.cwt(y, scale, wavelet)
#     coefs_pow = np.power(coefs, 2)      # 对二维数组中的数平方
#     spectrum_value = [0] * row    # len(freqs)
#     # 将二维数组按照里程叠加每个里程上的所有scale值
#     for i in range(row):
#         sum = 0
#         for j in range(len(freqs)):
#             sum += coefs_pow[j][i]
#         spectrum_value[i] = sum
#     fig = plt.figure(figsize=(7, 2))
#     line_width = 1
#     line_color = 'dodgerblue'
#     line_style = '-'
#     T1 = fig.add_subplot(1, 1, 1)
#     T1.plot(x, spectrum_value, label='模拟', linewidth=line_width, color=line_color, linestyle=line_style)
#     # T1.legend(loc='upper right', prop=font1, frameon=True)  # lower ,left
#     # 坐标轴名称
#     T1.set_xlabel('$time$', fontsize=15, fontdict=font1)  # fontdict设置子图字体
#     T1.set_ylabel('$E/mm^2$', fontsize=15, fontdict=font1)
#     # 坐标刻度值字体大小
#     T1.tick_params(labelsize=15)
#     print(spectrum_value[269])
#     plt.show()
# # 尺度小波能量谱
# def scale_spectrum():
#     coefs, freqs = pywt.cwt(y, scale, wavelet)
#     coefs_pow = np.power(coefs, 2)      # 对二维数组中的数平方
#     scale_freqs = np.power(freqs, -1)   # 对频率freqs 取倒数变为尺度
#     spectrum_value = [0] * len(freqs)    # len(freqs)
#     # 将二维数组按照里程叠加每个里程上的所有scale值
#     for i in range(len(freqs)):
#         sum = 0
#         for j in range(row):
#             sum += coefs_pow[i][j]
#         spectrum_value[i] = sum
#     fig = plt.figure(figsize=(7, 4))
#     line_width = 1
#     line_color1 = 'dodgerblue'
#     line_style1 = '-'
#     T1 = fig.add_subplot(1, 1, 1)
#     T1.plot(scale_freqs, spectrum_value, label=column_name, linewidth=line_width, color=line_color1, linestyle=line_style1)
#     # T1.legend(loc='upper right', prop=font1, frameon=True)  # lower ,left
#     # 坐标轴名称
#     T1.set_xlabel('$Scale$', fontsize=15, fontdict=font1)  # fontdict设置子图字体
#     T1.set_ylabel('$E/mm^2$', fontsize=15, fontdict=font1)
#     # 坐标刻度值字体大小
#     T1.tick_params(labelsize=15)
#     plt.show()
# # 通过调用下面三个不同的函数选择绘制能量谱
# time_scale_spectrum()
# # time_spectrum()
# # scale_spectrum()

















