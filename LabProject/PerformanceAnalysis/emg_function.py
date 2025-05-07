# -*- coding: utf-8 -*-
"""
Created on Wed May  7 09:34:33 2025

@author: Hsin.YH.Yang
"""
import pandas as pd
import numpy as np
from scipy import signal
import ezc3d
import math
# %%
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

c3d_notch_cutoff = [[49, 51],
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


raw_data_path = r"D:\Hsin\NTSU_lab\Baseball\Raw_Data\S03\MVC\S03_MVC_Forearm_Rep_1.0.csv"

# %%


    

# smoothing_data = smoothing_method(abs_data, method="moving", window_width=0.02, overlap_len=0.019)

def EMG_processing(raw_data_path, bandpass_cutoff=[20, 450], lowpass_freq = 6, notch_cutoff_list = [[59, 61]],
                   smoothing="lowpass", window_width=None, overlap_len=None, down_sap=False):
    """
    EMG 信號處理函數：支援 CSV / C3D 格式，並提供 Lowpass、Moving Mean、RMS 選項
    """
    
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
        # file_path = raw_data_path
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
            # 找尋EMG 訊號所在欄位 num_columns
            num_columns = []
            for i in range(len(raw_data.columns)):
                for ii in range(len(raw_data.columns[raw_data.columns.str.contains("EMG")])):
                    if raw_data.columns[i] == raw_data.columns[raw_data.columns.str.contains("EMG")][ii]:
                        num_columns.append(i)
            print("處理 EMG 訊號，總共", len(num_columns), "條肌肉， 分別為以下欄位")
            print(raw_data.columns[raw_data.columns.str.contains("EMG")])
            
            for col in range(len(num_columns)):
                data_time = raw_data.iloc[:,num_columns[col]-1].dropna()
                # 計算該通道的原始採樣頻率 Fs
                Fs.append((1/np.mean(np.array(data_time[2:11])-np.array(data_time[1:10]))))
                # # 計算數據中 0 值的數量 (表示數據末尾的無效部分)
                count0.append((raw_data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))
                # 計算該通道的有效數據長度 (去掉末尾 0 值部分)
                data_len.append(int((len(raw_data.iloc[:, num_columns[col]]) - (raw_data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))))
                # 計算該通道的數據截止時間 (找到數據末尾的時間戳記)
                all_stop_time.append(raw_data.iloc[(len(raw_data.iloc[:, num_columns[col]]) - (raw_data.iloc[:, num_columns[col]][::-1] != 0).argmax(axis = 0))-1 ,
                                               num_columns[col]-1])
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
    
    
    def apply_filters(raw_data, data_type, downsample_len, bandpass_cutoff, notch_cutoff=None):
        """ 
        應用 Bandpass、Notch 和 Lowpass 濾波器，適用於 CSV 和 C3D 格式的 EMG 信號處理
        
        參數:
        - raw_data (pd.DataFrame): 原始 EMG 數據
        - data_type (str): 數據類型 ("csv" 或 "c3d")
        - downsample_len (int): 降採樣後的數據長度
        - bandpass_cutoff (list): 帶通濾波的頻率範圍 [low, high]
        - notch_cutoff (list, optional): 陷波濾波頻率列表，如 [50, 100] (可選)
    
        回傳:
        - bandpass_filtered_data (pd.DataFrame): 帶通濾波後的數據
        - abs_data (pd.DataFrame): 取絕對值後的數據
        """
        # ------------------------ 取得需要處理的數據欄位 ------------------------
        if data_type == "csv":
            # CSV 檔案中，數據欄位是奇數索引 (假設時間欄位為偶數索引)
            data_columns = list(np.arange(1, len(raw_data.columns), 2))
        elif data_type == "c3d":
            # C3D 檔案中，所有數據欄位都要處理
            data_columns = list(np.arange(1, len(raw_data.columns), 1))
        # ------------------------ 初始化濾波後的 DataFrame ------------------------
        # 創建與 downsample_len 相同長度的 DataFrame，用來儲存濾波後的數據
        bandpass_filtered_data = pd.DataFrame(np.zeros([downsample_len, len(data_columns)]),
                                              columns=raw_data.iloc[:, data_columns].columns)
        abs_data = pd.DataFrame(np.zeros([downsample_len, len(data_columns)]),
                                columns=raw_data.iloc[:, data_columns].columns)
        # ------------------------ 針對每個 EMG 通道進行濾波 ------------------------
        for col in range(len(data_columns)):
            # 依照不同的檔案格式進行濾波，因為 Delsys 會因為不同的 Sensor 有不同的採樣頻率
            if data_type == "csv":
                Fs = (1/np.mean(np.array(raw_data.iloc[2:11, data_columns[col]-1]) -\
                                np.array(raw_data.iloc[1:10, data_columns[col]-1])))
            elif data_type == "c3d":
                Fs = (1/np.mean(np.array(raw_data.iloc[2:11, 0]) -\
                                np.array(raw_data.iloc[1:10, 0])))
            # 將資料中的 nan 補 0
            data = raw_data.iloc[:, data_columns[col]].fillna(0).values
            # ------------------------ Bandpass filter-----------------------
            bandpass_sos = signal.butter(2, bandpass_cutoff, btype='bandpass', fs=Fs, output='sos')
            bandpass_filtered = signal.sosfiltfilt(bandpass_sos, data)
            # ------------------------ Notch filter- ------------------------
            if notch_cutoff:
                notch_filtered = bandpass_filtered
                for cutoff in notch_cutoff:
                    notch_sos = signal.butter(2, cutoff, btype='bandstop', fs=Fs, output='sos')
                    notch_filtered = signal.sosfiltfilt(notch_sos, notch_filtered)
                    # 更新 bandpass_filtered 變數
                    bandpass_filtered = notch_filtered
    
            # ------------------------ 降採樣 ------------------------
            # 使用 scipy.signal.resample() 將數據降採樣到 downsample_len
            bandpass_filtered_data.iloc[:, col] = signal.resample(bandpass_filtered, downsample_len)
            # 取絕對值
            abs_data.iloc[:, col] = abs(signal.resample(bandpass_filtered, downsample_len))
        
        # ------------------------ 產生時間軸並插入 DataFrame ------------------------
        min_stop_time = 0 + np.shape(bandpass_filtered_data)[0] * 1/down_freq
        bandpass_time_index = np.linspace(0, min_stop_time, np.shape(bandpass_filtered_data)[0])
        # ------------------------ 產生時間軸並插入 DataFrame ------------------------
        bandpass_filtered_data.insert(0, 'time', bandpass_time_index)
        abs_data.insert(0, 'time', bandpass_time_index)
    
        return bandpass_filtered_data, abs_data
    
    
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
            lowpass_filtered_data = pd.DataFrame(np.zeros([np.shape(filtered_data)[0],
                                                           len(filtered_data.columns)]),
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
            step = max(1, window_width - overlap_len)
    
            # # 轉換秒數為數據點數
            # window_width = max(1, int(window_width * Fs))  # 確保至少為 1
            # overlap_len = max(0, int(overlap_len * Fs)) # 確保不為負數
            # step = max(1, int(window_width - overlap_len))  # 確保 step 至少為 1
    
            num_windows = max(1, (num_samples - window_width) // step + 1)
            # 初始化結果陣列
            smoothing_array = np.zeros((num_windows, num_columns))
            
            moving_data = pd.DataFrame(np.zeros([num_windows, filtered_data.shape[1]]),
                                       columns=filtered_data.columns)
        
            # 使用 NumPy Sliding Window
            for col_idx in range(num_columns):
                # 取得所有窗口數據 (shape = (num_windows, window_width))
                windows = np.lib.stride_tricks.sliding_window_view(filtered_array[:, col_idx], window_shape=window_width)[::step]
        
                if method.lower() == "moving":
                    smoothing_array[:, col_idx] = np.mean(windows, axis=1)
                elif method.lower() == "rms":
                    smoothing_array[:, col_idx] = np.sqrt(np.mean(windows ** 2, axis=1))
            # 轉回 DataFrame
            moving_data = pd.DataFrame(smoothing_array, columns=filtered_data.columns)
    
            return moving_data
        else:
            raise ValueError("Invalid method. Choose 'lowpass', 'moving', or 'rms'.")
    # 1. 加載數據
    raw_data, data_type = load_emg_data(raw_data_path)

    # 2. 計算採樣頻率與降採樣參數
    downsample_len = preprocess_emg_data(raw_data, data_type)

    # 3. 濾波處理
    _, abs_data = apply_filters(raw_data, data_type, downsample_len, bandpass_cutoff, notch_cutoff=notch_cutoff_list)

    # 4. 根據 smoothing 進行額外處理
    if smoothing == "lowpass":
        smoothing_data = smoothing_method(abs_data, method=smoothing, lowpass_cutoff=lowpass_freq)
        return smoothing_data
    elif smoothing in ["moving", "RMS"]:
        smoothing_data = smoothing_method(abs_data, method=smoothing,
                                          window_width=window_width, overlap_len=overlap_len)
        return smoothing_data