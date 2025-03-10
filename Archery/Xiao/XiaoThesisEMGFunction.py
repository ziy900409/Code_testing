# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:18:26 2024


• E1: 當L.Wrist.Rad Z軸高度超過T10 Z軸高度 擷取此段資料。
• E5: 擷取直到弓身低於T10 Z軸高度，停止擷取。
• 資料分期點:
• E2 舉弓頂點時間:根據全段資料，以L.Wrist.Rad Z軸判定，回傳位置峰值
數值與對應時間點，即時運算角度後取角度峰直數值與對應時間點
• E3 當L.Wrist.Rad Z軸高度等於L. Acromion 進行標記
• E4 放箭時間:根據資料末端2000點判定，即時運算移動平均, R. Elbow Lat
X軸超出前1秒數據3個標準差，判定為放箭
    
@author: Hsin.Yang 05.May.2024
"""
# %% import libraryt
import os
import pandas as pd
import numpy as np
from scipy import signal
import ezc3d
import math
import logging #print 警告用
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"E:\Hsin\git\git\Code_testing\Archery\Xiao")
# sys.path.append(r"D:\BenQ_Project\git\Code_testing\Archery\Xiao")
import XiaoThesisGeneralFunction as gen
from scipy.interpolate import interp1d

# %%
# ---------------------前處理用--------------------------------
# downsampling frequency
down_freq = 2000
# 帶通濾波頻率
bandpass_cutoff = [20/0.802, 450/0.802]
# 低通濾波頻率
lowpass_freq = 6/0.802
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.1 # 窗格長度 (單位 second)
overlap_len = 0.5 # 百分比 (%)
# 設定 notch filter cutoff frequency
csv_notch_cutoff_1 = [59, 61]
csv_notch_cutoff_2 = [295.5, 296.5]
csv_notch_cutoff_3 = [369.5, 370.5]
csv_notch_cutoff_4 = [179, 180]

c3d_notch_cutoff_1 = [49.5, 50.5]
c3d_notch_cutoff_2 = [99.5, 100.5]
c3d_notch_cutoff_3 = [149.5, 150.5]
c3d_notch_cutoff_4 = [249.5, 250.5]
c3d_notch_cutoff_5 = [349.5, 350.5]

csv_recolumns_name = {}

c3d_recolumns_name = {}

c3d_analog_cha = []

c3d_analog_idx = []

# ---------------------找放箭時間用----------------------------
# 設定最接近放箭位置之acc sensor的欄位編號，建議看完三軸資料再選最大的
# 可設定數字或是欄位名稱：ex: R EXTENSOR GROUP: ACC.Y 1 or 5
release_acc = 17
# 設定放箭的振幅大小值
release_peak = 1.5
# 設置繪圖參數 --------------------------------------------------------------
compare_name = {"group1" :["SH1"],
                "group2": ["SHM"]}
muscle_name = ["R EXT: EMG 1", "R TRI : EMG 2", "R FLX: EMG 3",
               "R BI: EMG 4", "R UT: EMG 5", "R LT: EMG 6"]

# 創造資料儲存位置
time_ratio = {"E1-E2": 1,
              "E2-E3-1": 1,
              "E3-1-E3-2": 0.5,
              "E3-2-E4": 3,
              "E4-E5": 0.2}

total_time = 0
for ratio in time_ratio.keys():
    total_time += time_ratio[ratio]

time_length = int(total_time * 10 * 2)

# %% EMG data processing
def EMG_processing(raw_data, smoothing="lowpass", notch=False):
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
    
    @author: Hsin.Yang 05.May.2024
            
    '''
    # raw_data = pd.read_csv(raw_data_path)

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
        if notch:
            # 做 band stop filter
            notch_sos_1 = signal.butter(2, csv_notch_cutoff_1, btype='bandstop', fs=sample_freq, output='sos')
            notch_filtered_1 = signal.sosfiltfilt(notch_sos_1,
                                                  bandpass_filtered)
            notch_sos_2 = signal.butter(2, csv_notch_cutoff_2, btype='bandstop', fs=sample_freq, output='sos')
            notch_filtered_2 = signal.sosfiltfilt(notch_sos_2,
                                                  notch_filtered_1)
            notch_sos_3 = signal.butter(2, csv_notch_cutoff_3, btype='bandstop', fs=sample_freq, output='sos')
            notch_filtered_3 = signal.sosfiltfilt(notch_sos_3,
                                                  notch_filtered_2)
            notch_sos_4 = signal.butter(2, csv_notch_cutoff_4, btype='bandstop', fs=sample_freq, output='sos')
            notch_filtered = signal.sosfiltfilt(notch_sos_4,
                                                notch_filtered_3)
        else:
            notch_filtered = bandpass_filtered
        # 取絕對值，將訊號翻正
        abs_data = abs(notch_filtered)
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
    # MVC_folder = r'D:\\BenQ_Project\\python\\Archery\\202405\\202405\\202405\\\\\\EMG\\\\Processing_Data\\Method_1\\R01\\data\\\\MVC\\'
    
    # MVC_save_path = r'D:\\BenQ_Project\\python\\Archery\\202405\\202405\\202405\\\\\\EMG\\\\Processing_Data\\Method_2\\R08\\'
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
    find_max_name = MVC_save_path + '\\' + MVC_save_path.split('\\')[-2] + '_all_MVC.xlsx'
    pd.DataFrame(find_max_all).to_excel(find_max_name, sheet_name='Sheet1', index=False, header=True)

# %% 傅立葉轉換與畫圖
# 計算傅立葉轉換
def Fourier_plot(raw_data_path, savepath, filename, notch=False):
    '''
    最終修訂時間: 20240329
    
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

    1. 新增可以處理 c3d 的方法
    '''
    # raw_data_path = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\2. EMG\raw_data\S01\S01_GridShot_Rep_2.1.csv"
    
    if '.csv' in raw_data_path:
        raw_data = pd.read_csv(raw_data_path)
    elif '.c3d' in raw_data_path:
        c = ezc3d.c3d(raw_data_path)
        # 3. convert c3d analog data to DataFrame format
        raw_data = pd.DataFrame(np.transpose(c['data']['analogs'][0, c3d_analog_idx, :]),
                                columns=c3d_analog_cha)
        ## 3.3 insert time frame
        ### 3.3.1 create time frame
        analog_time = np.linspace(
            0, # start
            ((c['header']['analogs']['last_frame'])/c['header']['analogs']['frame_rate']), # stop = last_frame/frame_rate
            num = (np.shape(c['data']['analogs'])[-1]) # num = last_frame
            )
        raw_data.insert(0, 'Frame', analog_time)
    
    
    num_columns = []
    for i in range(len(raw_data.columns)):
        for ii in range(len(raw_data.columns[raw_data.columns.str.contains("EMG")])):
            if raw_data.columns[i] == raw_data.columns[raw_data.columns.str.contains("EMG")][ii]:
                num_columns.append(i)
    # 讀取資料並重新定義 columns name
    if '.csv' in raw_data_path:
        raw_data.rename(columns=csv_recolumns_name, inplace=True)
    elif '.c3d' in raw_data_path:
        raw_data.rename(columns=c3d_recolumns_name, inplace=True)
    # 定義畫圖的子圖數量
    n = int(math.ceil(len(num_columns)/2))
    # due to our data type is series, therefore we need to extract value in the series
    # --------畫圖用與計算FFT----------------------
    # --------------------------------------------
    # 設定圖片大小
    plt.figure(figsize=(2*n+1,10))
    fig, axs = plt.subplots(n, 2, figsize = (10,12))
    # 設定圖片儲存位置，確認是否有做 notch
    if notch:
        save = savepath + '\\FFT_' + filename + "_notch.jpg"
    else:
        save = savepath + '\\FFT_' + filename + ".jpg"
    # 開始畫圖
    for col in range(len(num_columns)):
        x, y = col - n*math.floor(abs(col)/n), math.floor(abs(col)/n)
        # print(col)
        # 設定資料時間
        # 採樣頻率計算 : 取前十個時間點做差值平均
        # 因為新增 c3d 故採樣頻率依 ['analogs']['frame_rate']
        if '.csv' in raw_data_path:
            freq = int(1/np.mean(np.array(raw_data.iloc[2:11, num_columns[col]-1])-np.array(raw_data.iloc[1:10, num_columns[col]-1])))
        elif '.c3d' in raw_data_path:
            freq = c['header']['analogs']['frame_rate']
        data_len = (np.shape(raw_data)[0] - (raw_data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))
        # convert sampling rate to period
        # 計算取樣週期
        T = 1/freq

        # ---------------------開始計算 FFT -----------------------------------
        # 1. 先計算 bandpass filter
        isnan = np.where(np.isnan(raw_data.iloc[:data_len, num_columns[col]]))
        if isnan[0].size == 0:
        # 計算Bandpass filter
            # b, a = signal.butter(2, 20,  btype='high', fs=freq)
            # bandpass_filtered = signal.filtfilt(b, a, raw_data.iloc[:data_len, num_columns[col]].values)
            bandpass_sos = signal.butter(2, bandpass_cutoff,  btype='bandpass', fs=freq, output='sos')
            bandpass_filtered = signal.sosfiltfilt(bandpass_sos,
                                                    raw_data.iloc[:data_len, num_columns[col]].values)

        # 設定給斷訊超過 0.1 秒的 sensor 警告
        elif isnan[0].size > 0.1*freq:
            logging.warning(str(raw_data.columns[num_columns[col]] + "sensor 總訊號斷訊超過 0.1 秒，"))
            bandpass_sos = signal.butter(2, bandpass_cutoff,  btype='bandpass', fs=freq, output='sos')
            bandpass_filtered = signal.sosfiltfilt(bandpass_sos,
                                                   raw_data.iloc[:(np.shape(raw_data)[0] - data_len[col]), num_columns[col]].values)

        else:
            logging.warning(str("共發現 " + str(isnan[0].size) + " 個缺值,位置為 " + str(isnan[0])))
            logging.warning("已將 NAN 換為 0")
            bandpass_sos = signal.butter(2, bandpass_cutoff,  btype='bandpass', fs=freq, output='sos')
            bandpass_filtered = signal.sosfiltfilt(bandpass_sos,
                                                   raw_data.iloc[:data_len, num_columns[col]].fillna(0))
        # -----------------是否需要 notch data----------------------------------
        if notch:
            # print(0)
            if '.csv' in raw_data_path:
                notch_sos_1 = signal.butter(2, csv_notch_cutoff_1, btype='bandstop', fs=freq, output='sos')
                notch_filtered_1 = signal.sosfiltfilt(notch_sos_1,
                                                    bandpass_filtered)
                notch_sos_2 = signal.butter(2, csv_notch_cutoff_2, btype='bandstop', fs=freq, output='sos')
                notch_filtered_2 = signal.sosfiltfilt(notch_sos_2,
                                                    notch_filtered_1)
                notch_sos_3 = signal.butter(2, csv_notch_cutoff_3, btype='bandstop', fs=freq, output='sos')
                notch_filtered_3 = signal.sosfiltfilt(notch_sos_3,
                                                    notch_filtered_2)
                notch_sos_4 = signal.butter(2, csv_notch_cutoff_4, btype='bandstop', fs=freq, output='sos')
                notch_filtered = signal.sosfiltfilt(notch_sos_4,
                                                    notch_filtered_3)
            elif '.c3d' in raw_data_path:
                print(0)
                notch_sos_1 = signal.butter(2, c3d_notch_cutoff_1, btype='bandstop', fs=freq, output='sos')
                notch_filtered_1 = signal.sosfiltfilt(notch_sos_1,
                                                    bandpass_filtered)
                notch_sos_2 = signal.butter(2, c3d_notch_cutoff_2, btype='bandstop', fs=freq, output='sos')
                notch_filtered_2 = signal.sosfiltfilt(notch_sos_2,
                                                    notch_filtered_1)
                notch_sos_3 = signal.butter(2, c3d_notch_cutoff_3, btype='bandstop', fs=freq, output='sos')
                notch_filtered_3 = signal.sosfiltfilt(notch_sos_3,
                                                    notch_filtered_2)
                notch_sos_4 = signal.butter(2, c3d_notch_cutoff_4, btype='bandstop', fs=freq, output='sos')
                notch_filtered_4 = signal.sosfiltfilt(notch_sos_4,
                                                    notch_filtered_3)
                notch_sos_5 = signal.butter(2, c3d_notch_cutoff_5, btype='bandstop', fs=freq, output='sos')
                notch_filtered = signal.sosfiltfilt(notch_sos_5,
                                                    notch_filtered_4)
            fft_data = notch_filtered
        else:
            # print(1)
            fft_data = bandpass_filtered
        # 2. 資料前處理
        # 計算資料長度
        N = len(fft_data)#length of the array
        # N = int(np.prod(fft_data.shape[0]))#length of the array
        N2 = 2**(N.bit_length()-1) #last power of 2
        # convert sampling rate to period 
        # 計算取樣週期
        T = 1.0/freq
        N = N2 #truncate array to the last power of 2
        xf = np.linspace(0.0, np.ceil(1.0/(2.0*T)), N//2)
        # print("# caculate Fast Fourier transform")
        # print("# Samples length:",N)
        # print("# Sampling rate:",freq)
        # 開始計算 FFT   
        yf = fft(fft_data, N)
        freqs = fftfreq(N, T) 
        axs[x, y].plot(freqs[0:int(N/2)], abs(yf[0:int(N/2)])*2/N,
                       linewidth=0.5)
        # axs[x, y].plot(xf, 2.0/N * abs(yf[0:int(N/2)]))
        axs[x, y].set_title(raw_data.columns[num_columns[col]], fontsize = 16)
        # 設定科學符號 : 小數點後幾位數
        axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
        # 標出第一、二、三大值的位置
        float_array = 2.0/N * abs(yf[0:int(N/2)])
        max_value = np.max(float_array)
        max_index = np.argmax(float_array)
        # print(xf[max_index])
        axs[x, y].plot(xf[max_index], max_value, 'o', color='red')
        slope = [xf[max_index], max_value]
        axs[x, y].annotate('{:.2f}, {:.2f}'.format(*slope), xy=(xf[max_index], max_value))
        
        # 將最大值的位置設為負無窮大，以找到第二大的值
        float_array[max_index] = float('-inf')
        second_max_value = np.max(float_array)
        second_max_index = np.argmax(float_array)
        slope = [xf[second_max_index], second_max_value]
        axs[x, y].plot(xf[second_max_index], second_max_value, 'o', color='red')
        axs[x, y].annotate('{:.2f}, {:.2f}'.format(*slope), xy=(xf[second_max_index], second_max_value))
        # print(xf[second_max_index])

        # 將第二大值的位置設為負無窮大，以找到第三大的值
        float_array[second_max_index] = float('-inf')
        third_max_value = np.max(float_array)
        third_max_index = np.argmax(float_array)
        slope = [xf[third_max_index], third_max_value]
        axs[x, y].plot(xf[third_max_index], third_max_value, 'o', color='red')
        axs[x, y].annotate('{:.2f}, {:.2f}'.format(*slope), xy=(xf[third_max_index], third_max_value))
        # print(xf[third_max_index])
        # axs[x, y].set_xlim(0, 500)
    # 設定整張圖片之參數
    plt.suptitle(str("FFT Analysis " + filename), fontsize = 16)
    plt.tight_layout()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Frequency (Hz)", fontsize = 14)
    plt.ylabel("Power", fontsize = 14)
    plt.savefig(save, dpi=200, bbox_inches = "tight")
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
# %%  to find the release timing

def find_release_time(folder_path, save_path, save_fig=True):
    # 讀所有.csv file
    file_list = gen.Read_File(folder_path, '.csv')
    release_timing_list = pd.DataFrame(columns = ["FileName", "Time Frame(降1000Hz)", "Time"])
    # 繪圖用
    n = int(math.ceil(len(file_list) /2))
    plt.figure(figsize=(2*n,10))
    fig, axs = plt.subplots(n, 2, figsize = (10,12))
    for ii in range(len(file_list)):
        if os.path.splitext(file_list[ii])[1] == ".csv":
            filepath, tempfilename = os.path.split(file_list[ii])
            data = pd.read_csv(file_list[ii])
            # data = pd.read_csv(r"C:\Users\h7058\Downloads\R11_SH1_Rep_1.5.csv")
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
    plt.suptitle(str("release time: " + save_path.split('\\')[-3]), fontsize = 16)
    plt.tight_layout()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("time (second)", fontsize = 14)
    plt.ylabel("acc (g)", fontsize = 14)
    plt.savefig(str(save_path + '\\' + save_path.split('\\')[-3] + "_ReleaseTiming.jpg"),
                dpi=100)
    plt.show()
    # writting data to a excel file
    save_iMVC_name = save_path + '\\' + save_path.split('\\')[-3] + '_ReleaseTiming.xlsx' 
    DataFrame(release_timing_list).to_excel(save_iMVC_name, sheet_name='Sheet1', index=False, header=True)
    return release_index
    
    
# save_path = r'E:\\Hsin\\NTSU_lab\\Archery\\Xiao\\202406\\202405\\\\\\motion\\\\Processing_Data\\Method_1\\R04\\figure\\release'
    
# %%
def compare_mean_std_cloud(data_path, savepath, filename, smoothing,
                           compare_name = compare_name,
                           muscle_name = muscle_name,
                           compare_group = False,
                           subfolder=False):
    
    '''

    比较多个数据集的均值和标准差，并生成相应的图表。

    Parameters
    ----------
    data_path : str
        要比较数据的文件夹路径.
    savepath : str
        保存图表的文件夹路径.
    filename : str
        保存图表的文件名.
    smoothing : bool
        是否对数据进行平滑处理.
    compare_name : list of str, optional
        需要比较的数据集名称列表，默认为 ["SH1", "SHM"].
    muscle_name : list of str, optional
        肌肉名称列表，默认为 ["R EXT: EMG 1", "R TRI : EMG 2", "R FLX: EMG 3",
        "R BI: EMG 4", "R UT: EMG 5", "R LT: EMG 6"].

    Returns

    @author: Hsin.Yang 05.May.2024
    '''

    
    # data_path = r'E:\Hsin\NTSU_lab\Archery\Xiao\202406\202405\EMG\Processing_Data\Method_1\R01\data\motion'
    data_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\202406\202405\EMG\Processing_Data"
    
    # 找出所有資料夾下的 .xlsx 檔案
    if subfolder:
        data_list = gen.Read_File(data_path, ".xlsx", subfolder=True)
    else:
        data_list = gen.Read_File(data_path, ".xlsx", subfolder=False)
    
    compare_data = {key: [] for key in compare_name}
    if compare_group:
        for key in compare_name:
            for i in compare_name[key]:
                for ii in range(len(data_list)):
                    if i in data_list[ii]:
                        compare_data[key].append(data_list[ii])
    else:
        for i in range(len(compare_name)):
            for ii in range(len(data_list)):
                if compare_name[i] in data_list[ii]:
                    compare_data[compare_name[i]].append(data_list[ii])
                
    # 假设 muscle_name 和 total_time 已经定义
    muscle_length = len(muscle_name)

    # 初始化空字典来存储数据数组
    data_arrays = {}
    # 根据 compare_data 的长度，创建相应数量的数据数组
    # muscle name * time length * subject number
    for key in compare_data:
        subject_count = len(compare_data[key])
        data_arrays[key] = np.empty([muscle_length, time_length, subject_count])

    for key in compare_data:
        for idx in range(len(compare_data[key])):
            time_idx = 0
            for period in time_ratio.keys():
                raw_data = pd.read_excel(compare_data[key][idx],
                                         sheet_name=period)
                for muscle in range(len(muscle_name)):
                    time_period = int(time_ratio[period]*10*2)
                    # print(muscle_name[muscle])
                    # print(time_period)
                    # 使用 cubic 將資料內插
                    x = raw_data.iloc[:, 0] # time
                    y = raw_data.loc[:, muscle_name[muscle]]
                    if len(x) < 4:
                        f = interp1d(x, y, kind='linear')
                    else:
                        f = interp1d(x, y, kind='cubic')
                    x_new = np.linspace(raw_data.iloc[0, 0], raw_data.iloc[-1, 0],
                                        time_period)
                    y_new = f(x_new)
                    data_arrays[key][muscle, time_idx:time_idx + time_period, idx] = y_new
                time_idx = time_idx + time_period
    # 設定圖片大小
    # 畫第一條線
    save = savepath + "\\mean_std_" + filename + ".jpg"
    n = int(math.ceil((len(muscle_name)) /2))
    # 設置圖片大小
    # plt.figure(figsize=(n+1,10))
    # 設定繪圖格式與字體
    # plt.style.use('seaborn-white')
    # 顯示輸入中文
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    palette = plt.get_cmap('Set1')
    
    fig, axs = plt.subplots(n, 2, figsize = (10, 2*n+1), sharex='col')
    i = 0 # 更改顏色用
    for key in compare_data:
        for muscle in range(len(muscle_name)):
            # 確定繪圖順序與位置
            x, y = muscle - n*math.floor(abs(muscle)/n), math.floor(abs(muscle)/n) 
            iters = list(np.linspace(0, 114, time_length))
            # 設定計算資料
            avg1 = np.mean(data_arrays[key][muscle, :, :], axis=1) # 計算平均
            std1 = np.std(data_arrays[key][muscle, :, :], axis=1) # 計算標準差
            r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
            r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
            axs[x, y].plot(iters, avg1, color=palette(i), label=key, linewidth=3)
            axs[x, y].fill_between(iters, r1, r2, color=palette(i), alpha=0.2)
            
            # 圖片的格式設定
            axs[x, y].set_title(muscle_name[muscle], fontsize=12)
            axs[x, y].legend(loc="upper left") # 圖例位置
            # axs[x, y].grid(True, linestyle='-.')
            # 畫放箭時間
            axs[x, y].set_xlim(0, time_length)
            axs[x, y].axvline(x=110, color = 'darkslategray', linewidth=1, linestyle = '--')
        i += 1
    # plt.suptitle(str("mean std cloud: " + filename), fontsize=16)
    plt.tight_layout()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("time (second)", fontsize = 14)
    plt.ylabel("muscle activation (%)", fontsize = 14)
    # plt.savefig(save, dpi=200, bbox_inches = "tight")
    plt.show()
    

        # # 畫花括號
        # curlyBrace(fig, axs[x, y], [shooting_time["stage1"][0], yy], [shooting_time["stage1"][1], yy],
        #            0.05, bool_auto=True, str_text="", color=shooting_time["stage1"][2],
        #            lw=2, int_line_num=1, fontdict=font)
        # curlyBrace(fig, axs[x, y], [shooting_time["stage2"][0], yy], [shooting_time["stage2"][1], yy],
        #            0.05, bool_auto=True, str_text="", color=shooting_time["stage2"][2],
        #            lw=2, int_line_num=1, fontdict=font)
        # curlyBrace(fig, axs[x, y], [shooting_time["stage3"][0], yy], [shooting_time["stage3"][1], yy],
        #            0.05, bool_auto=True, str_text="", color=shooting_time["stage3"][2],
        #            lw=2, int_line_num=1, fontdict=font)
        # curlyBrace(fig, axs[x, y], [shooting_time["stage4"][0], yy], [shooting_time["stage4"][1], yy],
        #            0.05, bool_auto=True, str_text="", color=shooting_time["stage4"][2],
        #            lw=2, int_line_num=1, fontdict=font)
# %%

def compare_mean_std_cloud_onecol(data_path, savepath, filename, smoothing,
                                  compare_name = compare_name,
                                  muscle_name = muscle_name,
                                  compare_group = False,
                                  subfolder=False):
    
    '''

    比较多个数据集的均值和标准差，并生成相应的图表。

    Parameters
    ----------
    data_path : str
        要比较数据的文件夹路径.
    savepath : str
        保存图表的文件夹路径.
    filename : str
        保存图表的文件名.
    smoothing : bool
        是否对数据进行平滑处理.
    compare_name : list of str, optional
        需要比较的数据集名称列表，默认为 ["SH1", "SHM"].
    muscle_name : list of str, optional
        肌肉名称列表，默认为 ["R EXT: EMG 1", "R TRI : EMG 2", "R FLX: EMG 3",
        "R BI: EMG 4", "R UT: EMG 5", "R LT: EMG 6"].

    Returns

    @author: Hsin.Yang 05.May.2024
    '''

    time_accum = {"E1-E2": 10,
                  "E2-E3-1": 20,
                  "E3-1-E3-2": 25,
                  "E3-2-E4": 55,
                  "E4-E5": 57}
    # data_path = r'E:\Hsin\NTSU_lab\Archery\Xiao\202406\202405\EMG\Processing_Data\Method_1\R01\data\motion'
    # data_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\202406\202405\EMG\Processing_Data"
    
    # 找出所有資料夾下的 .xlsx 檔案
    if subfolder:
        data_list = gen.Read_File(data_path, ".xlsx", subfolder=True)
    else:
        data_list = gen.Read_File(data_path, ".xlsx", subfolder=False)
    
    compare_data = {key: [] for key in compare_name}
    if compare_group:
        for key in compare_name:
            for i in compare_name[key]:
                for ii in range(len(data_list)):
                    if i in data_list[ii]:
                        compare_data[key].append(data_list[ii])
    else:
        for i in range(len(compare_name)):
            for ii in range(len(data_list)):
                if compare_name[i] in data_list[ii]:
                    compare_data[compare_name[i]].append(data_list[ii])
                
    # 假设 muscle_name 和 total_time 已经定义
    muscle_length = len(muscle_name)

    # 初始化空字典来存储数据数组
    data_arrays = {}
    # 根据 compare_data 的长度，创建相应数量的数据数组
    # muscle name * time length * subject number
    for key in compare_data:
        subject_count = len(compare_data[key])
        data_arrays[key] = np.empty([muscle_length, time_length, subject_count])

    for key in compare_data:
        for idx in range(len(compare_data[key])):
            time_idx = 0
            for period in time_ratio.keys():
                raw_data = pd.read_excel(compare_data[key][idx],
                                         sheet_name=period)
                for muscle in range(len(muscle_name)):
                    time_period = int(time_ratio[period]*10*2)
                    # print(muscle_name[muscle])
                    # print(time_period)
                    # 使用 cubic 將資料內插
                    x = raw_data.iloc[:, 0] # time
                    y = raw_data.loc[:, muscle_name[muscle]]
                    if len(x) < 4:
                        f = interp1d(x, y, kind='linear')
                    else:
                        f = interp1d(x, y, kind='cubic')
                    x_new = np.linspace(raw_data.iloc[0, 0], raw_data.iloc[-1, 0],
                                        time_period)
                    y_new = f(x_new)
                    data_arrays[key][muscle, time_idx:time_idx + time_period, idx] = y_new
                time_idx = time_idx + time_period
    # 設定圖片大小
    # 畫第一條線
    save = savepath + "\\mean_std_" + filename + ".jpg"
    n = len(muscle_name)
    # 設置圖片大小
    # plt.figure(figsize=(n+1,10))
    # 設定繪圖格式與字體
    # plt.style.use('seaborn-white')
    # 顯示輸入中文
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    palette = plt.get_cmap('Set1')
    
    fig, axs = plt.subplots(n, 1, figsize = (8, 2*n+1), sharex='col')
    i = 0 # 更改顏色用
    for key in compare_data:
        for muscle in range(len(muscle_name)):
            # 確定繪圖順序與位置
            # x, y = muscle - n*math.floor(abs(muscle)/n), math.floor(abs(muscle)/n) 
            iters = list(np.linspace(0, 114, time_length))
            # 設定計算資料
            avg1 = np.mean(data_arrays[key][muscle, :, :], axis=1) # 計算平均
            std1 = np.std(data_arrays[key][muscle, :, :], axis=1) # 計算標準差
            r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
            r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
            axs[muscle].plot(iters, avg1, color=palette(i), label=key, linewidth=3)
            axs[muscle].fill_between(iters, r1, r2, color=palette(i), alpha=0.2)
            
            # 圖片的格式設定
            axs[muscle].set_title(muscle_name[muscle], fontsize=12)
            axs[muscle].legend(loc="upper left") # 圖例位置
            # axs[x, y].grid(True, linestyle='-.')
            # 畫放箭時間
            axs[muscle].set_xlim(0, time_length)
            # axs[muscle].axvline(x=110, color = 'darkslategray', linewidth=1, linestyle = '--')
            
            

            for time_key in time_accum.keys():
                axs[muscle].axvline(time_accum[time_key]*2,
                                    color='darkslategray', linestyle='--', linewidth=1) # trigger onset+
        i += 1 # 更變顏色用
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

    
    
    
    
    
    
    
    
    
    