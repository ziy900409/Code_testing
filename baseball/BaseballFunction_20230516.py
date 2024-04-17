# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:44:53 2023

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
import matplotlib.pyplot as plt
import logging


# %% 設置參數位置
'''
使用順序
0.5 採樣頻率已自動計算，不需自己定義
** 由於不同代EMG sensor的採樣頻率不同，因此在初步處理時，先down sampling to 1000Hz
1. 依照個人需求更改資料處理欄位，更改位置為
    ex : EMG_data = data.iloc[:, [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]]
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
# 分析欄位編號
num_columns = [1, 3, 5, 7, 9, 11]
# 帶通濾波頻率
bandpass_cutoff = [30/0.802, 450/0.802]
# 低通濾波頻率
lowpass_freq = 6/0.802
# downsampling frequency
down_freq = 1000

# 設定最接近放箭位置之acc sensor的欄位編號，建議看完三軸資料再選最大的
# 可設定數字或是欄位名稱：ex: R EXTENSOR GROUP: ACC.Y 1 or 5
release_acc = 5
# 放箭前檔名
before_fatigue = "SH1"
# 放箭後檔名
after_fatigue = "SH2"
# 抓放箭時候前後秒數
# example : [秒數*採樣頻率, 秒數*採樣頻率]
# 兩個檔案的參數必須相同 ArcheryXiao, ArcheryFunction
release = [2*down_freq, 1*down_freq]
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.02 # 窗格長度 (單位 second)
overlap_len = 0.95 # 百分比 (%)

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
# %% EMG data processing
def EMG_processing(raw_data, smoothing="lowpass"):
    '''
    version: 20240414
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
        # 2.3.------resample data to 1000Hz-----------
        # 降採樣資料，並將資料儲存在矩陣當中
        resample_length = int(len(data)//(decimation_factor))
        bandpass_filtered = signal.resample(bandpass_filtered, resample_length)
        bandpass_filtered_data.iloc[:resample_length, col] = bandpass_filtered
        abs_data = signal.resample(abs_data, resample_length)
        # 2.2.5 data smoothing
        # ------linear envelop analysis-----------                          
        # ------lowpass filter parameter that the user must modify for your experiment        
        if smoothing == "lowpass":
            # filter
            lowpass_sos = signal.butter(2, lowpass_freq, btype='low', fs=sample_freq, output='sos')        
            lowpass_filtered = signal.sosfiltfilt(lowpass_sos, abs_data)
            # 降採樣資料，並將資料儲存在矩陣當中
            lowpass_filtered = signal.resample(lowpass_filtered, resample_length)
            lowpass_filtered_data.iloc[:resample_length, col] = lowpass_filtered
        # -------Data smoothing. Compute Moving mean
        # window width = window length(second)*sampling rate
        elif smoothing == "moving":
            for ii in range(np.shape(moving_data)[0]):
                data_location = int(ii*(1-overlap_len)*window_width)
                # print(data_location, data_location+window_width_rms)
                moving_data.iloc[int(ii), col] = (np.sum((abs_data[data_location:data_location+window_width]))
                                                  /window_width)
        # -------Data smoothing. Compute RMS
        # The user should change window length and overlap length that suit for your experiment design
        # window width = window length(second)*sampling rate
        elif smoothing == "rms":
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
    # 設定 return 參數
    if smoothing == "lowpass":
        # 定義lowpass filter的時間
        lowpass_time_index = np.linspace(0, min_stop_time, np.shape(lowpass_filtered_data)[0])
        lowpass_filtered_data.insert(0, 'time', lowpass_time_index)
        # 丟棄 lowpass_filtered_data 欄位內包含 0 的行
        lowpass_filtered_data = lowpass_filtered_data.loc[~(lowpass_filtered_data==0).any(axis=1)]
        return lowpass_filtered_data, bandpass_filtered_data
    elif smoothing == "rms":
        # 定義RMS DATA的時間
        rms_time_index = np.linspace(0, min_stop_time, np.shape(rms_data)[0])
        rms_data.insert(0, 'time', rms_time_index)
        return rms_data, bandpass_filtered_data
    elif smoothing == "moving":
        # 定義moving average的時間
        moving_time_index = np.linspace(0, min_stop_time, np.shape(moving_data)[0])
        moving_data.insert(0, 'time', moving_time_index)
        # 丟棄 moving_data 欄位內包含 0 的行
        moving_data = moving_data.loc[~(moving_data==0).any(axis=1)]
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
    find_max_name = MVC_save_path + '\\' + MVC_save_path.split('\\')[-1] + '_all_MVC.xlsx'
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
    save = savepath + '\\FFT_' + filename + ".jpg"
    n = int(math.ceil((len(num_columns) - 1) /2))
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
        # print("# caculate Fast Fourier transform")
        # print("# Samples length:",N)
        # print("# Sampling rate:",freq)
        # 開始計算 FFT   
        yf = fft(bandpass_data)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    