# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:54:30 2023

April 26 2023.
    1. 修正 EMG_processing 中 lowpass的採樣頻率問題.
May 02 2023.
    1. 新增可選擇平滑處理的方式.
    2. 新增 median freqency 功能.
June 03 2023.
    1. 修正單數肌群會畫不出圖的問題.
    解決方式 : 繪圖數量為 無條件進位(採樣肌肉數量/2)
                np.ceil(num_columns / 2)
    2. 修正中文無法顯示的問題.
    3. 修正如果 sensor 斷訊出現 NAN 問題.
    解決方法 :
        3.1 使用 np.where(np.isnan(a)) 找出 NAN 位置的索引值，如果"缺值"時間短於 0.1 秒，先將 NAN 填 0.
        3.2 若缺值時間長於 0.1 秒，則給予警告訊息.
June 06 2023.    
    1. function mean_std_cloud 新增功能.
        1.1. 修正只有lowpass會有標示放箭時間的問題.
        1.2. 修正時間軸錯誤問題.
        1.3. 修正繪圖 Y 軸的科學記號問題.
    mean std 的放箭時間點, 時間軸, 時間錯誤、不能出現科學符號、標紅線的放箭時間.
    疲勞測試，畫在同一張圖.
June 27 2023.
    1. function iMVC_calculate 修改功能.
        1.1. 修改掉需要手動回去設定EXCEL最大值的問題.
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
# matplotlib其實是不支持顯示中文的 顯示中文需要一行代碼設置字體
# mpl.rcParams['font.family'] = 'Microsoft Sans Serif'
mpl.rcParams["font.sans-serif"] = ["'Microsoft Sans Serif"]
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False   # 步驟二（解決坐標軸負數的負號顯示問題）

# %% 設置參數位置
'''
使用順序
0.5 採樣頻率已自動計算，不需自己定義
** 由於不同代EMG sensor的採樣頻率不同，因此在初步處理時，先down sampling to 1000Hz
1. 依照個人需求更改資料處理欄位，更改位置為
    ex : num_columns = [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]
    Method_1 = [1, 15, 29, 37, 45, 53, 61, 69, 77, 85], Release= 5, 值3.5 EXT Y
    Method_2 = [1, 3, 17, 25, 33, 41, 49, 57, 65, 73], Release= 19, 值2 FLX X
    Method_3 = [1, 3, 5, 7, 9, 11, 13, 15, 23, 31, 51, 55], Release= 35, 值3.0 EXT Y
    Method_4 = [1, 15, 29, 37, 45, 53, 61, 69, 77, 85], Release= 5, 值2
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
# num_columns = [1, 15, 29, 43, 45, 47, 49, 51]
# 帶通濾波頻率
bandpass_cutoff = [8/0.802, 450/0.802]
# 低通濾波頻率
lowpass_freq = 6/0.802
# downsampling frequency
down_freq = 1800
# 設定最接近放箭位置之acc sensor的欄位編號，建議看完三軸資料再選最大的
# 可設定數字或是欄位名稱：ex: R EXTENSOR GROUP: ACC.Y 1 or 5
release_acc = 5
# 設定放箭的振幅大小值
release_peak = 2
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
time_of_window = 0.1 # 窗格長度 (單位 second)
overlap_len = 0.5 # 百分比 (%)

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
    cvs_file_list : str
        給予欲處理資料之路徑.
    smoothing : str, optional
        設定 smoothing method,分別為 lowpass, rms, moving. The default is 'lowpass'
        
    Returns
    -------
    moving_data : pandas.DataFrame
        回給移動平均數.
    rms_data : pandas.DataFrame
        回給移動均方根.
    lowpass_filtered_data : pandas.DataFrame
        回給低通濾波.
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
    # raw_data = pd.read_csv(r"E:\BenQ_Project\ForBQP\S01_Blink_8_Rep_6.22.csv",
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
    # 1.3.-------------創建儲存EMG data的矩陣------------
    # bandpass filter used in signal
    bandpass_filtered_data = pd.DataFrame(np.zeros([int(min(Fs)//(min(freq)/down_freq)), len(num_columns)]),
                            columns=raw_data.iloc[:, num_columns].columns)
    lowpass_filtered_data = pd.DataFrame(np.zeros([int(min(Fs)//(min(freq)/down_freq)), len(num_columns)]),
                            columns=raw_data.iloc[:, num_columns].columns)
    # 設定moving mean的矩陣大小、欄位名稱
    window_width = int(time_of_window*np.floor(down_freq))
    moving_data = pd.DataFrame(np.zeros([int((np.shape(bandpass_filtered_data)[0] - window_width)/  ((1-overlap_len)*window_width)) + 1,
                                         np.shape(bandpass_filtered_data)[1]]),
                               columns=raw_data.iloc[:, num_columns].columns)
    # 設定Root mean square的矩陣大小、欄位名稱
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
        # 降取樣資料，並將資料儲存在矩陣當中
        bandpass_filtered = signal.resample(bandpass_filtered, int(min(Fs)//decimation_factor))
        bandpass_filtered_data.iloc[:, col] = bandpass_filtered
        abs_data = signal.resample(abs_data, int(min(Fs)//decimation_factor))
        lowpass_filtered = signal.resample(lowpass_filtered, int(min(Fs)//decimation_factor))
        lowpass_filtered_data.iloc[:, col] = lowpass_filtered
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
    # 定義lowpass filter的時間
    lowpass_time_index = np.linspace(0, min_stop_time, np.shape(lowpass_filtered_data)[0])
    lowpass_filtered_data.insert(0, 'time', lowpass_time_index)
    # 定義moving average的時間
    moving_time_index = np.linspace(0, min_stop_time, np.shape(moving_data)[0])
    moving_data.insert(0, 'time', moving_time_index)
    # 定義RMS DATA的時間.
    rms_time_index = np.linspace(0, min_stop_time, np.shape(rms_data)[0])
    rms_data.insert(0, 'time', rms_time_index)
    # 設定 return 參數
    if smoothing == "lowpass":
        return lowpass_filtered_data, bandpass_filtered_data
    elif smoothing == "rms":
        return rms_data, bandpass_filtered_data
    elif smoothing == "moving":
        return moving_data, bandpass_filtered_data  

# %% writting data to a excel file 
def Excel_writting(file_path, data_save_path, data):
    # deal with filename and add extension with _ed
    filepath, tempfilename = os.path.split(file_path)
    filename, extension = os.path.splitext(tempfilename)
    # rewrite file name
    file_name = data_save_path + '\\' + filename + '_RMS'
    file_name = file_name.replace('.', '_') + '.xlsx'
    # writting data in worksheet
    DataFrame(data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
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
    # 修改為自動判斷最大值
    MVC_value = np.max(MVC_value, axis=0)[2:]
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

def find_release_time(folder_path, save_path):
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
    plt.savefig(str(save_path + '\\' + save_path.split('\\')[-1] + "_ReleaseTiming.jpg"),
                dpi=100)
    plt.show()
    # writting data to a excel file
    save_iMVC_name = save_path + '\\' + save_path.split('\\')[-1] + '_ReleaseTiming.xlsx' 
    DataFrame(release_timing_list).to_excel(save_iMVC_name, sheet_name='Sheet1', index=False, header=True)
    # return release_timing_list       

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
    num_columns = []
    for i in range(len(raw_data.columns)):
        for ii in range(len(raw_data.columns[raw_data.columns.str.contains("EMG")])):
            if raw_data.columns[i] == raw_data.columns[raw_data.columns.str.contains("EMG")][ii]:
                num_columns.append(i)
    save = savepath + '\\FFT_' + filename + ".jpg"
    n = int(math.ceil((len(num_columns)) /2))
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
            bandpass_sos = signal.butter(1, bandpass_cutoff,  btype='bandpass', fs=freq, output='sos')
            bandpass_filtered = signal.sosfiltfilt(bandpass_sos,
                                                   raw_data.iloc[:data_len, num_columns[col]].values)
        # 設定給斷訊超過 0.1 秒的 sensor 警告
        elif isnan[0].size > 0.1*freq:
            logging.warning(str(raw_data.columns[num_columns[col]] + "sensor 總訊號斷訊超過 0.1 秒，"))
            bandpass_sos = signal.butter(1, bandpass_cutoff,  btype='bandpass', fs=freq, output='sos')
            bandpass_filtered = signal.sosfiltfilt(bandpass_sos,
                                                   raw_data.iloc[:(np.shape(raw_data)[0] - data_len[col]), num_columns[col]].values)
        else:
            logging.warning(str("共發現 " + str(isnan[0].size) + " 個缺值,位置為 " + str(isnan[0])))
            logging.warning("已將 NAN 換為 0")
            bandpass_sos = signal.butter(1, bandpass_cutoff,  btype='bandpass', fs=freq, output='sos')
            bandpass_filtered = signal.sosfiltfilt(bandpass_sos,
                                                   raw_data.iloc[:data_len, num_columns[col]].fillna(0))
        # 2. 資料前處理
        # 計算資料長度
        N = int(np.prod(bandpass_filtered.shape[0]))#length of the array
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
        yf = fft(bandpass_filtered)
        axs[x, y].plot(xf, 2.0/N * abs(yf[0:int(N/2)]))
        axs[x, y].set_title(raw_data.columns[num_columns[col]], fontsize = 16)
        # 設定科學符號 : 小數點後幾位數
        axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
        axs[x, y].set_ylim(0, 500)
    # 設定整張圖片之參數
    # plt.suptitle(str("FFT Analysis " + filename), fontsize = 16)
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

def mean_std_cloud(data_path, savepath, filename, smoothing):
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
    plt.style.use('seaborn-white')
    # 顯示輸入中文
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    palette = plt.get_cmap('Set1')
    fig, axs = plt.subplots(n, 2, figsize = (10,12), sharex='col')
    for i in range(np.shape(type2_dict)[0]):
        # 確定繪圖順序與位置
        x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
        color = palette(0) # 設定顏色
        iters = list(np.linspace(int(-release[0]/down_freq), int(release[1]/down_freq), 
                                 len(type1_dict[0, :, 0])))
        # 設定計算資料
        avg1 = np.mean(type1_dict[i, :, :], axis=1) # 計算平均
        std1 = np.std(type1_dict[i, :, :], axis=1) # 計算標準差
        r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
        r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
        axs[x, y].plot(iters, avg1, color=color, label='before', linewidth=3)
        axs[x, y].fill_between(iters, r1, r2, color=color, alpha=0.2)
        
        # 畫第二條線
        color = palette(1) # 設定顏色
        avg2 = np.mean(type2_dict[i, :, :], axis=1) # 計畫平均
        std2 = np.std(type2_dict[i, :, :], axis=1) # 計算標準差
        r1 = list(map(lambda x: x[0]-x[1], zip(avg2, std2))) # 畫一個標準差以內的線
        r2 = list(map(lambda x: x[0]+x[1], zip(avg2, std2)))
        axs[x, y].plot(iters, avg2, color=color, label='after', linewidth=3) # 畫平均線
        axs[x, y].fill_between(iters, r1, r2, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊
        # 圖片的格式設定
        axs[x, y].set_title(example_data.columns[i+1], fontsize=12)
        axs[x, y].legend() # 圖例位置
        axs[x, y].grid(True, linestyle='-.')
        # 畫放箭時間
        axs[x, y].set_xlim(iters[0], iters[-1])
        axs[x, y].axvline(x=0, color = 'darkslategray', linewidth=1, linestyle = '--')

        # axs[x, y].set_xticks(np.linspace(-(release[0]), release[1], num=len(iters)))
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
    # 創建資料處存位置
    # raw_data = emg_data.iloc[int(motion_info["frame_rate"]*10*8):int(motion_info["frame_rate"]*10*(8+52)), :]
    # raw_data = emg_data.iloc[int(new_motion_info["frame_rate"]*10*8):int(new_motion_info["frame_rate"]*10*(8+52)), :]
    median_freq_table = pd.DataFrame({}, columns=(["filename", "columns_num"] + [str(i) for i in range(52)]))
    # 尋找 EMG 所在欄位
    num_columns = []
    for i in range(len(raw_data.columns)):
        for ii in range(len(raw_data.columns[raw_data.columns.str.contains("EMG")])):
            if raw_data.columns[i] == raw_data.columns[raw_data.columns.str.contains("EMG")][ii]:
                num_columns.append(i)
    # 處理貯存檔名問題
    folder_name, tempfilename = filename.split('\\', -1)[-2], filename.split('\\', -1)[-1]
    save_name, extension = os.path.splitext(tempfilename)
    # fig_svae_path +"\\EMG_fig\\" + folder_name + "\\Spider\\" + save_name + "_EMG.jpg"
    # save = fig_svae_path + '\\MedianFreq_' + filename + ".jpg"
    print("執行 Fatigue Analysis 檔名: ", filename)
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
        # durtion length = 1 seccond
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
        # 將資料輸進矩陣
        median_freq_table = pd.concat([median_freq_table, 
                                       pd.DataFrame([([filename, num_columns[col]] + med_freq_list[:])],
                                                    columns=(["filename", "columns_num"] + [str(i) for i in range(47)]),
                                                    index=[0])])
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
    plt.savefig(str(fig_svae_path +"\\EMG_fig\\" + folder_name + "\\Spider\\" + save_name + "_MedFreq.jpg"),
                dpi=200, bbox_inches = "tight")
    plt.show()
    # return table
    return median_freq_table