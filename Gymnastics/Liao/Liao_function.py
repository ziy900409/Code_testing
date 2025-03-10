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
from scipy.stats import linregress
import time

# %%
# ---------------------前處理用--------------------------------
# downsampling frequency
down_freq = 1000
c = 0.802
# 帶通濾波頻率
bandpass_cutoff = [20/0.802, 450/0.802]
# 低通濾波頻率
lowpass_freq = 10/c
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.1 # 窗格長度 (單位 second)
overlap_len = 0.5 # 百分比 (%)
# 設定 notch filter cutoff frequency

notch_cutoff = [[59, 61],
                         [295.5, 296.5],
                         [369.5, 370.5],
                         [179, 181],
                         [299, 301],
                         [419, 421],
                        ]

c3d_notch_cutoff_list = [[49, 51],
                         [99.5, 100.5],
                         [149.5, 150.5],
                         [199.5, 200.5],
                         [249.5, 250.5],
                         [299.5, 300.5],
                         [349.5, 350.5],
                         [295, 297],
                         [369, 371],
                         [73, 75],
                         [399, 401]
                        ]

csv_recolumns_name = {'R.RA: EMG 1': 'Rectus Abdominus',
                     'R.ES: EMG 2': 'Erector Spinae',
                     'R.IL: EMG 3': 'Iliopsoas',
                     'R.GMax: EMG 4': 'Gluteus Maximus',
                     'R.RF: EMG 5': 'Rectus Femoris',
                     'R.BF: EMG 6': 'Biceps Femoris',
                     'R.TA&IO: EMG 7': 'Tranverse Abdominus & Internal Oblique',
                     'R.MF: EMG 8': 'Multifidus',}

c3d_recolumns_name = {'ExtRad': 'Extensor Carpi Radialis',
                     'FleRad': 'Flexor Carpi Radialis',
                     'Triceps': 'Triceps Brachii',
                     'Triceps': 'Triceps Brachii',
                     'ExtUlnar': 'Extensor Carpi Ulnaris',
                     'ExtUlnar': 'Extensor Carpi Ulnaris',
                     'DorInter': '1st Dorsal Interosseous', 
                     'AbdDigMin': 'Abductor Digiti Quinti',
                     #' AbdDigMin.IM EMG6': 'Abductor Digiti Quinti',
                     'ExtInd': 'Extensor Indicis',
                     'Biceps': 'Biceps Brachii',
                     }

c3d_analog_cha = ["ExtRad", "FleRad", "ExtUlnar", "DorInter", "AbdDigMin", "ExtInd",
                  "Biceps", "Triceps"]

muscle_name = ['Extensor Carpi Radialis', 'Flexor Carpi Radialis', 'Triceps Brachii',
               'Extensor Carpi Ulnaris', '1st Dorsal Interosseous', 
               'Abductor Digiti Quinti', 'Extensor Indicis', 'Biceps Brachii']
# %%

def EMG_processing(raw_data_path, smoothing="lowpass", window_width=None, overlap_len=None, down_sap=False):
    '''
    最終修訂時間: 20240329
    note:
        1. 2024.03.28
        新增可以處理 c3d 的功能
        2. moving mean, RMS 的功能尚未修正
    
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
    raw_data_path = r"E:\Hsin\NTSU_lab\Gymnastics\論文資料CSV檔\EMG\NSF1.1_1\NSF1.1_Back_Tuck_Somersault_Rep_1.5.csv"
    if '.csv' in raw_data_path:
        raw_data = pd.read_csv(raw_data_path)
    elif '.c3d' in raw_data_path:
        c = ezc3d.c3d(raw_data_path)
        # 3. convert c3d analog data to DataFrame format
        raw_data_header = c['parameters']['ANALOG']['LABELS']
        raw_header_index = []
        c3d_header_all = []
        for c3d_header in c3d_recolumns_name:
            for i in range(len(raw_data_header['value'])):
                if c3d_header in raw_data_header['value'][i]:
                    raw_header_index.append(i)
                    c3d_header_all.append(raw_data_header['value'][i])
                    
        # 3. convert c3d analog data to DataFrame format
        raw_data = pd.DataFrame(np.transpose(c['data']['analogs'][0, raw_header_index, :]),
                                columns=c3d_header_all)
        ## 3.3 insert time frame
        ### 3.3.1 create time frame
        analog_time = np.linspace(
            0, # start
            ((c['header']['analogs']['last_frame'])/c['header']['analogs']['frame_rate']), # stop = last_frame/frame_rate
            num = (np.shape(c['data']['analogs'])[-1]) # num = last_frame
                                    )
        raw_data.insert(0, 'Frame', analog_time)
    
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
    if '.csv' in raw_data_path:
        raw_data.rename(columns=csv_recolumns_name, inplace=True)
    elif '.c3d' in raw_data_path:
        raw_data.rename(columns=c3d_recolumns_name, inplace=True)
    # 只有檔案格式是 .csv 才做
    if '.csv' in raw_data_path:
        # 計算各採樣頻率與計算downsample所需的位點數，並取最小的位點數
        Fs = []
        data_len = []
        count0 = []
        all_stop_time = []
        downsample_len = []
        for col in range(len(num_columns)):
            data_time = raw_data.iloc[:,num_columns[col]-1].dropna()
            Fs.append((1/np.mean(np.array(data_time[2:11])-np.array(data_time[1:10]))))
            # 計算數列中0的數量
            count0.append((raw_data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))
            # 找到第一個 Raw data 不等於零的位置
            data_len.append(int((len(raw_data.iloc[:, num_columns[col]]) - (raw_data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))))
            # 取截止時間
            all_stop_time.append(raw_data.iloc[(len(raw_data.iloc[:, num_columns[col]]) - (raw_data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))-1 ,
                                           num_columns[col]-1])
            
            downsample_len.append(data_len[-1] / Fs[-1] * down_freq)
        # 1.2.-------------計算平均截止時間------------------
        # 丟棄NAN的值，並選擇最小值
        min_stop_time = np.min([x for x in all_stop_time if math.isnan(x) == False])
        while max(all_stop_time) - min(all_stop_time) > 1:
            print("兩 sensor 數據時間差超過 1 秒")
            print("將使用次短時間的 Sensor 作替代")
            all_stop_time.remove(min_stop_time)
            data_len.remove(min(data_len))
            min_stop_time = np.min([x for x in all_stop_time if math.isnan(x) == False])
        # data_len = min(data_len)
        Fs = min(Fs)
        downsample_len = math.floor(min(downsample_len))
        data_len = math.floor(min(data_len))
    elif '.c3d' in raw_data_path:
        Fs = c['header']['analogs']['frame_rate']
        data_len = np.shape(raw_data)[0]
        min_stop_time = c['header']['analogs']['last_frame']/Fs
        
        downsample_len = data_len / Fs * down_freq
    # 計算各採樣頻率與計算downsample所需的位點數，並取最小的位點數
    # 1.3.-------------創建儲存EMG data的矩陣------------
    if down_sap:
        data_len = downsample_len
        Fs = down_freq
        
    # bandpass filter used in signal
    bandpass_filtered_data = pd.DataFrame(np.zeros([math.floor(data_len), len(num_columns)]),
                            columns=raw_data.iloc[:, num_columns].columns)
    notch_filtered_data = pd.DataFrame(np.zeros([math.floor(data_len), len(num_columns)]),
                            columns=raw_data.iloc[:, num_columns].columns)
    lowpass_filtered_data = pd.DataFrame(np.zeros([math.floor(data_len), len(num_columns)]),
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
        if '.csv' in raw_data_path:
            # 取採樣時間的前十個採樣點計算採樣頻率
            sample_freq = 1/np.mean(np.array(raw_data.iloc[2:11, (num_columns[col] - 1)])
                                        - np.array(raw_data.iloc[1:10, (num_columns[col] - 1)]))
            # decimation_factor = sample_freq / down_freq
            # 在raw data中以最短的數據長短為標準，只取最短數據的資料找其中是否包含NAN
            if type(data_len) != int:
                indi_data_len = data_len[col]
            isnan = np.where(np.isnan(raw_data.iloc[:(np.shape(raw_data)[0] - (indi_data_len)), num_columns[col]]))
            # 預處理資料,判斷資料中是否有 nan, 並將 nan 取代為 0 
            if isnan[0].size == 0:
            # 計算Bandpass filter
                # data = raw_data.iloc[:(np.shape(raw_data)[0] - indi_data_len), num_columns[col]].values
                data = raw_data.iloc[:(np.shape(raw_data)[0]), num_columns[col]].values
            # 設定給斷訊超過 0.1 秒的 sensor 警告
            elif isnan[0].size > 0.1*sample_freq:
                logging.warning(str(raw_data.columns[num_columns[col]] + "sensor 總訊號斷訊超過 0.1 秒，"))
                # data = raw_data.iloc[:(np.shape(raw_data)[0] - indi_data_len), num_columns[col]].values
                data = raw_data.iloc[:(np.shape(raw_data)[0]), num_columns[col]].values
            else:
                logging.warning(str("共發現 " + str(isnan[0].size) + " 個缺值,位置為 " + str(isnan[0])))
                logging.warning("已將 NAN 換為 0")
                # data = raw_data.iloc[:(np.shape(raw_data)[0] - indi_data_len), num_columns[col]].fillna(0)
                data = raw_data.iloc[:(np.shape(raw_data)[0]), num_columns[col]].fillna(0)
            # 由於各截止時間不同，所以找出最小的截止時間，並將其他較長時間的 sensor，都截成短的
            # 找出最小的時間，並且找出所有欄位數據中最接近的索引值
            end_index = np.abs(raw_data.iloc[:, num_columns[col]-1].fillna(0) - min_stop_time).argmin()
            data = data[:end_index+1]
            # 進行 bandpass filter
            bandpass_sos = signal.butter(2, bandpass_cutoff,  btype='bandpass', fs=sample_freq, output='sos')
            bandpass_filtered = signal.sosfiltfilt(bandpass_sos, data)
            # 做 band stop filter
            notch_filtered = bandpass_filtered  # 起始輸入信號

            # 使用迴圈進行多次 Notch 過濾
            for cutoff in csv_notch_cutoff_list:
                notch_sos = signal.butter(2, cutoff, btype='bandstop', fs=sample_freq, output='sos')
                notch_filtered = signal.sosfiltfilt(notch_sos, notch_filtered)
            
            # 取絕對值，將訊號翻正
            abs_data = abs(notch_filtered)
            # ------linear envelop analysis-----------                          
            # ------lowpass filter parameter that the user must modify for your experiment        
            lowpass_sos = signal.butter(2, lowpass_freq, btype='low', fs=sample_freq, output='sos')        
            lowpass_filtered = signal.sosfiltfilt(lowpass_sos, abs_data)
        
        elif '.c3d' in raw_data_path:
            # decimation_factor = Fs / down_freq
            data = raw_data.iloc[:(np.shape(raw_data)[0]), num_columns[col]].values
            bandpass_sos = signal.butter(2, bandpass_cutoff,  btype='bandpass', fs=Fs, output='sos')
            bandpass_filtered = signal.sosfiltfilt(bandpass_sos, data)
            # notch filter
            # 初始化過濾數據
            notch_filtered = bandpass_filtered  # 起始輸入信號

            # 使用迴圈進行多次 Notch 過濾
            for cutoff in c3d_notch_cutoff_list:
                notch_sos = signal.butter(2, cutoff, btype='bandstop', fs=Fs, output='sos')
                notch_filtered = signal.sosfiltfilt(notch_sos, notch_filtered)
  
            # 取絕對值，將訊號翻正
            abs_data = abs(notch_filtered)
            # ------linear envelop analysis-----------                          
            # ------lowpass filter parameter that the user must modify for your experiment        
            lowpass_sos = signal.butter(2, lowpass_freq, btype='low', fs=Fs, output='sos')        
            lowpass_filtered = signal.sosfiltfilt(lowpass_sos, abs_data)
 
        
        
        # 2.3.------resample data to 1000Hz-----------
        # 降採樣資料，並將資料儲存在矩陣當中
        
        notch_filtered = signal.resample(notch_filtered, int(downsample_len))
        notch_filtered_data.iloc[:, col] = notch_filtered[:int(downsample_len)]
        bandpass_filtered = signal.resample(bandpass_filtered, int(downsample_len))
        bandpass_filtered_data.iloc[:, col] = bandpass_filtered[:int(downsample_len)]
        abs_data = signal.resample(abs_data, int(downsample_len))
        lowpass_filtered = signal.resample(lowpass_filtered, int(downsample_len))
        lowpass_filtered_data.iloc[:, col] = lowpass_filtered[:int(downsample_len)]
        # -------Data smoothing. Compute Moving mean
        # window width = window length(second)*sampling rate
        
        # 需重新修改 moving mean 以及 RMS method
        
        # for col in range(columns):  # 針對不同欄位計算
        #     for ii in range(moving_data.shape[0]):  
        #         data_location = int(ii * (1 - overlap_len) * window_width)
        #         if data_location + window_width > len(abs_data):  # 避免超出索引範圍
        #             break
        #         moving_data.iloc[ii, col] = np.mean(abs_data[data_location:data_location + window_width])  # 計算移動平均
        
        #     for ii in range(rms_data.shape[0]):  
        #         data_location = int(ii * (1 - overlap_len) * window_width)
        #         if data_location + window_width > len(abs_data):  # 避免超出索引範圍
        #             break
        #         rms_data.iloc[ii, col] = np.sqrt(np.mean(abs_data[data_location:data_location + window_width] ** 2))  # 計算 RMS
        
        
        
        # for ii in range(np.shape(moving_data)[0]):
        #     data_location = int(ii*(1-overlap_len)*window_width)
        #     # print(data_location, data_location+window_width_rms)
        #     moving_data.iloc[int(ii), col] = (np.sum((abs_data[data_location:data_location+window_width])**2)
        #                                   /window_width)
            
        # # -------Data smoothing. Compute RMS
        # # The user should change window length and overlap length that suit for your experiment design
        # # window width = window length(second)*sampling rate
        # for ii in range(np.shape(rms_data)[0]):
        #     data_location = int(ii*(1-overlap_len)*window_width)
        #     # print(data_location, data_location+window_width_rms)
        #     rms_data.iloc[int(ii), col] = np.sqrt(np.sum((abs_data[data_location:data_location+window_width])**2)
        #                                   /window_width)
        
                
    # 3. -------------插入時間軸-------------------
    # 定義bandpass filter的時間
    min_stop_time = 0 + np.shape(bandpass_filtered_data)[0] * 1/down_freq
    bandpass_time_index = np.linspace(0, min_stop_time, np.shape(bandpass_filtered_data)[0])
    bandpass_filtered_data.insert(0, 'time', bandpass_time_index)
    # 定義 notch filter的時間
    notch_time_index = np.linspace(0, min_stop_time, np.shape(notch_filtered_data)[0])
    notch_filtered_data.insert(0, 'time', notch_time_index)
    # 定義lowpass filter的時間
    lowpass_time_index = np.linspace(0, min_stop_time, np.shape(lowpass_filtered_data)[0])
    lowpass_filtered_data.insert(0, 'time', lowpass_time_index)

    # 定義moving average的時間
    # moving_time_index = np.linspace(0, min_stop_time, np.shape(moving_data)[0])
    # moving_data.insert(0, 'time', moving_time_index)
    # # 定義RMS DATA的時間
    # rms_time_index = np.linspace(0, min_stop_time, np.shape(rms_data)[0])
    # rms_data.insert(0, 'time', rms_time_index)
    # 設定 return 參數
    if smoothing == "lowpass":
        return lowpass_filtered_data, notch_filtered_data
    # elif smoothing == "rms":
    #     return rms_data, bandpass_filtered_data
    # elif smoothing == "moving":
    #     return moving_data, bandpass_filtered_data

# %%


def load_emg_data(file_path):
    """ 
    加載 CSV 或 C3D 檔案並轉換為 DataFrame
    - 若為 CSV 檔案，則讀取 EMG 相關欄位
    - 若為 C3D 檔案，則使用 `ezc3d` 解析並轉換為 DataFrame
    - 若格式不支援，則拋出錯誤
    
    參數:
    file_path (str): 檔案路徑 (必須是 CSV 或 C3D)
    
    回傳:
    raw_data (pd.DataFrame): 轉換後的 EMG 數據
    data_type (str): "csv" 或 "c3d"，表示數據類型
    """    
    # 讀取 CSV 檔案
    if file_path.endswith('.csv'):
        raw_data = pd.read_csv(file_path)  # 讀取 CSV 為 DataFrame

        num_columns = []  # 用來儲存 EMG 數據的索引
        for i in range(len(raw_data.columns)):  # 遍歷所有欄位名稱
            if "EMG" in raw_data.columns[i]:  # 如果欄位名稱包含 "EMG"
                num_columns.append(i - 1)  # 加入 EMG 前一列 (通常是時間戳記)
                num_columns.append(i)  # 加入 EMG 數據列

        raw_data = raw_data.iloc[:, num_columns]  # 只保留時間軸與 EMG 數據
        return raw_data, "csv"  # 回傳處理後的數據和類型標記
    
    # 讀取 C3D 檔案
    elif file_path.endswith('.c3d'):
        c3d_data = ezc3d.c3d(file_path)  # 使用 ezc3d 讀取 C3D 檔案

        # 取得所有訊號名稱 (包含 EMG 和其他感測數據)
        raw_data_header = c3d_data['parameters']['ANALOG']['LABELS']['value']

        # 過濾出 EMG 相關的欄位索引
        raw_header_index = [i for i, name in enumerate(raw_data_header) if "EMG" in name]
        emg_headers = [raw_data_header[i] for i in raw_header_index]  # 取得 EMG 欄位名稱

        # 轉換 C3D 的 EMG 數據為 DataFrame
        raw_data = pd.DataFrame(
            np.transpose(c3d_data['data']['analogs'][0, raw_header_index, :]),  # 轉置數據，讓 EMG 訊號成為列
            columns=emg_headers  # 設定對應的欄位名稱
        )

        # 產生時間軸
        frame_rate = c3d_data['header']['analogs']['frame_rate']  # 取得 EMG 採樣頻率
        last_frame = c3d_data['header']['analogs']['last_frame']  # 取得最後一幀的編號
        raw_data.insert(0, 'Frame', np.linspace(0, last_frame / frame_rate, num=raw_data.shape[0]))  
        # 在 DataFrame 第一欄插入 "Frame" (時間戳記)，確保時間資訊對齊 EMG 訊號

        return raw_data, "c3d"  # 回傳處理後的數據和類型標記
    
    # 若格式不支援，則拋出錯誤
    else:
        raise ValueError("不支援的檔案格式，請提供 CSV 或 C3D 檔案。")


raw_data_path = r"E:\Hsin\NTSU_lab\Gymnastics\論文資料CSV檔\EMG\NSF1.1_1\NSF1.1_Back_Tuck_Somersault_Rep_1.5.csv"
raw_data_path = r"E:\Hsin\BenQ\ZOWIE non-sym\1.motion\Vicon\S03\S03_Biceps_MVC.c3d"
raw_data, data_type = load_emg_data(raw_data_path)

def preprocess_emg_data(raw_data, data_type, down_freq=1000):
    """ 
    計算採樣頻率、對齊數據長度、進行降採樣 (downsampling)
    
    參數:
    raw_data (pd.DataFrame): 原始 EMG 數據
    data_type (str): "csv" 或 "c3d"，表示數據類型
    down_freq (int): 目標降採樣頻率 (Hz)，預設為 1000Hz
    
    回傳:
    downsample_len (int): 降採樣後的數據長度
    """

    
    if data_type == "csv":
        data_len = []  # 儲存每個 EMG 通道的有效數據長度
        count0 = []  # 儲存數據末尾 0 值的數量 (表示無效數據長度)
        all_stop_time = []  # 儲存每個 EMG 通道的數據截止時間
        downsample_len = []  # 計算降採樣後的數據長度
        Fs = []  # 儲存每個通道的原始採樣頻率 (Hz)
        
        # 遍歷每個 EMG 通道 (CSV 格式假設時間欄位與 EMG 數據交錯排列)
        for col in range(int(len(raw_data.columns)/2)): # 每 2 個欄位是一組 (時間欄 + EMG 數據)
            data_time = raw_data.iloc[:, col*2].dropna() # 取得該通道的時間欄位 (去掉 NaN)
            # 計算該通道的原始採樣頻率 Fs
            Fs.append((1/np.mean(np.array(data_time[2:11])-np.array(data_time[1:10]))))
            # # 計算數據中 0 值的數量 (表示數據末尾的無效部分)
            count0.append((raw_data.iloc[:, (col*2+1)][::-1] != 0).argmax(axis = 0)) # 對齊 column of EMG data
            # 計算該通道的有效數據長度 (去掉末尾 0 值部分)
            data_len.append(int((len(raw_data.iloc[:, (col*2+1)]) - (raw_data.iloc[:, (col*2+1)][::-1] != 0).argmax(axis = 0))))
            # 計算該通道的數據截止時間 (找到數據末尾的時間戳記)
            all_stop_time.append(raw_data.iloc[(len(raw_data.iloc[:, (col*2+1)]) - \
                                                (raw_data.iloc[:, (col*2+1)][::-1] != 0).argmax(axis = 0))-1 ,
                                                (col*2)])
            # 計算降採樣後的數據長度
            downsample_len.append(data_len[-1] / Fs[-1] * down_freq)
        # 使用最小的 Fs (確保所有通道的降採樣保持同步)
        Fs = min(Fs)
        # 使用最短的降採樣長度，確保所有通道數據對齊
        downsample_len = math.floor(min(downsample_len))
        # 1.2.-------------計算平均截止時間------------------
        # 丟棄NAN的值，並選擇最小值
        min_stop_time = np.min([x for x in all_stop_time if math.isnan(x) == False])
        # 如果最長與最短的數據時間差超過 1 秒，則刪除最長數據，確保數據同步
        while max(all_stop_time) - min(all_stop_time) > 1:
            print("兩 sensor 數據時間差超過 1 秒")
            print("將使用次短時間的 Sensor 作替代")
            all_stop_time.remove(min_stop_time) # 移除最短時間
            data_len.remove(min(data_len)) # 移除對應的數據長度
            min_stop_time = np.min([x for x in all_stop_time if math.isnan(x) == False])
    
    elif data_type == "c3d":
        Fs = raw_data.shape[0] / raw_data["Frame"].iloc[-1] 
        downsample_len = math.floor(raw_data.shape[0] / Fs * down_freq)

    return downsample_len


downsample_len = preprocess_emg_data(raw_data, data_type, down_freq=1000)

def apply_filters(raw_data, data_type, downsample_len, bandpass_cutoff, notch_cutoff=None):
    """ 應用 Bandpass、Notch 和 Lowpass 濾波器 """
    if data_type == "csv":
        data_columns = list(np.arange(1, len(raw_data.columns), 2))
    elif data_type == "c3d":
        data_columns = list(np.arange(1, len(raw_data.columns), 1))

    bandpass_filtered_data = pd.DataFrame(np.zeros([downsample_len, len(data_columns)]),
                                          columns=raw_data.iloc[:, data_columns].columns)
    abs_data = pd.DataFrame(np.zeros([downsample_len, len(data_columns)]),
                            columns=raw_data.iloc[:, data_columns].columns)
    # 針對每個欄位進行濾波
    for col in range(len(data_columns)):
        # 依照不同的檔案格式進行濾波，因為 Delsys 會因為不同的 Sensor 有不同的採樣頻率
        if data_type == "csv":
            Fs = (1/np.mean(np.array(raw_data.iloc[2:11, data_columns[col]-1]) -\
                            np.array(raw_data.iloc[1:10, data_columns[col]-1])))
        elif data_type == "c3d":
            Fs = (1/np.mean(np.array(raw_data.iloc[2:11, 0]) -\
                            np.array(raw_data.iloc[1:10, 0])))
        # 將資料中的 nan 補零
        data = raw_data.iloc[:, data_columns[col]].fillna(0).values
        # 將資料進行 bandpass filter
        bandpass_sos = signal.butter(2, bandpass_cutoff, btype='bandpass', fs=Fs, output='sos')
        bandpass_filtered = signal.sosfiltfilt(bandpass_sos, data)
        # notch filter
        if notch_cutoff:
            notch_filtered = bandpass_filtered
            for cutoff in notch_cutoff:
                notch_sos = signal.butter(2, cutoff, btype='bandstop', fs=Fs, output='sos')
                notch_filtered = signal.sosfiltfilt(notch_sos, notch_filtered)
                bandpass_filtered = notch_filtered

        # 降採樣
        bandpass_filtered_data.iloc[:, col] = signal.resample(bandpass_filtered, downsample_len)
        # 取絕對值
        abs_data.iloc[:, col] = abs(signal.resample(bandpass_filtered, downsample_len))
    
    # 定義bandpass filter的時間
    min_stop_time = 0 + np.shape(bandpass_filtered_data)[0] * 1/down_freq
    bandpass_time_index = np.linspace(0, min_stop_time, np.shape(bandpass_filtered_data)[0])
    bandpass_filtered_data.insert(0, 'time', bandpass_time_index)
    abs_data.insert(0, 'time', bandpass_time_index)

    return bandpass_filtered_data, abs_data

_, abs_data = apply_filters(raw_data, data_type, downsample_len, bandpass_cutoff, notch_cutoff=None)

def smoothing_method(filtered_data, method="moving", lowpass_cutoff=None, window_width=None, overlap_len=None):
    """ 
    計算 Lowpass, Moving Mean, RMS

    參數:
    - filtered_data (pd.DataFrame): 要處理的 EMG 數據
    - method (str): 選擇 "lowpass", "moving", "rms"
    - lowpass_cutoff (float): 低通濾波的截止頻率 (Hz)
    - window_width (float): 移動平均或 RMS 計算的窗口寬度 (秒)
    - overlap_len (float): 移動平均或 RMS 計算的窗口重疊長度 (秒)

    回傳:
    - smoothing_data (pd.DataFrame): 平滑處理後的數據
    """
    # 計算採樣頻率
    Fs = (1/np.mean(np.array(filtered_data.iloc[2:11, 0]) -\
                    np.array(filtered_data.iloc[1:10, 0])))
    if method.lower() == "lowpass":
        if lowpass_cutoff == None :
            raise ValueError("Must define the lowpass_cutoff")
        lowpass_filtered_data = pd.DataFrame(np.zeros([downsample_len, len(filtered_data.columns)]),
                                             columns=filtered_data.columns)
        for col_idx, col in enumerate(filtered_data.columns):
            if col_idx == 0:
                lowpass_filtered_data.iloc[:, col_idx] = filtered_data.iloc[:, col_idx].values
            elif col_idx != 0:
                # lowpass filter
                lowpass_sos = signal.butter(2, lowpass_freq, btype='low', fs=Fs, output='sos')
                lowpass_filtered = signal.sosfiltfilt(lowpass_sos, filtered_data.iloc[:, col_idx].values)
                lowpass_filtered_data.iloc[:, col_idx] = lowpass_filtered
        return lowpass_filtered_data
    
    elif method.lower() == "moving" or method.lower() == "rms":
        # 列出警告標示
        if not isinstance(window_width, (int, float)) or not isinstance(overlap_len, (int, float)):
            raise ValueError("window_width and overlap_len must be numbers.")
            
         # 轉換 DataFrame 為 NumPy 陣列
        filtered_array = filtered_data.to_numpy()  # 加快處理速度
        num_samples, num_columns = filtered_array.shape
        # 轉換秒數為數據點數
        window_width = max(1, int(window_width * Fs))  # 確保至少為 1
        overlap_len = max(0, int(overlap_len * Fs)) # 確保不為負數
        step = max(1, int(window_width - overlap_len))  # 確保 step 至少為 1

        num_windows = max(1, (num_samples - window_width) // step + 1)
        # 初始化結果陣列
        smoothing_array = np.zeros((num_windows, num_columns))
        
        moving_data = pd.DataFrame(np.zeros([num_windows, filtered_data.shape[1]]),
                                   columns=filtered_data.columns)
        
        # 計時開始
        # start_time = time.time()
        # for col in filtered_data.columns:
        #     for i in range(num_windows):
        #         start_idx = i * step
        #         end_idx = start_idx + window_width
        #         window = filtered_data.loc[start_idx:end_idx, col]
    
        #         if method.lower() == "moving":
        #             moving_data.loc[i, col] = np.mean(window)
        #         elif method.lower() == "rms":
        #             moving_data.loc[i, col] = np.sqrt(np.mean(window ** 2))
        # # 計時結束
        # end_time = time.time()

        # # 計算並輸出運行時間
        # elapsed_time = end_time - start_time
        # print(f"函數執行時間: {elapsed_time:.6f} 秒")
        # 計時開始
        start_time = time.time()
    
        # 使用 NumPy Sliding Window
        for col_idx in range(num_columns):
            # 取得所有窗口數據 (shape = (num_windows, window_width))
            windows = np.lib.stride_tricks.sliding_window_view(filtered_array[:, col_idx], window_shape=window_width)[::step]
    
            if method.lower() == "moving":
                smoothing_array[:, col_idx] = np.mean(windows, axis=1)
            elif method.lower() == "rms":
                smoothing_array[:, col_idx] = np.sqrt(np.mean(windows ** 2, axis=1))
    
        # 計時結束
        end_time = time.time()
    
        # 轉回 DataFrame
        smoothing_data_1 = pd.DataFrame(smoothing_array, columns=filtered_data.columns)
    
        # 輸出運行時間
        elapsed_time = end_time - start_time
        print(f"函數執行時間: {elapsed_time:.6f} 秒")
        return moving_data
    else:
        raise ValueError("Invalid method. Choose 'lowpass', 'moving', or 'rms'.")

    

smoothing_data = smoothing_method(abs_data, method="moving", window_width=0.02, overlap_len=0.019)

def EMG_processing(raw_data_path, smoothing="lowpass", window_width=None, overlap_len=None, down_sap=False):
    """
    EMG 信號處理函數：支援 CSV / C3D 格式，並提供 Lowpass、Moving Mean、RMS 選項
    """
    # 1. 加載數據
    raw_data, data_type = load_emg_data(raw_data_path)

    # 2. 計算採樣頻率與降採樣參數
    downsample_len = preprocess_emg_data(raw_data, data_type)

    # 3. 濾波處理
    bandpass_cutoff = [20, 450]
    notch_cutoff_list = [50, 100, 150]
    lowpass_freq = 10
    window_width = 0.02
    overlap_len = 0.019
    method = "lowpass"
    
    _, abs_data = apply_filters(raw_data, data_type, downsample_len, bandpass_cutoff, notch_cutoff=notch_cutoff_list)

    # 4. 根據 smoothing 進行額外處理
    if smoothing == "lowpass":
        smoothing_data = smoothing_method(abs_data, method=method, lowpass_cutoff=lowpass_freq)
        return smoothing_data
    elif smoothing in ["moving", "RMS"]:
        smoothing_data = smoothing_method(abs_data, method=method,
                                                window_width=window_width, overlap_len=overlap_len)
        return smoothing_data


# 測試使用範例：
# lowpass_filtered, notch_filtered = EMG_processing("data.csv", smoothing="lowpass")
# moving_data, bandpass_data = EMG_processing("data.csv", smoothing="moving", window_width=100, overlap_len=0.5)










































