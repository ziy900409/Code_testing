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
import logging
import io # 用於處理記憶體中的檔案
from flask import Flask, request, jsonify
import json
import os
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

csv_recolumns_name = {'Mini sensor 1: EMG 1': 'Extensor Carpi Radialis',
                     'Mini sensor 2: EMG 2': 'Flexor Carpi Radialis',
                     'Mini sensor 3: EMG 3': 'Triceps Brachii',
                     'Quattro sensor 4: EMG.A 4': 'Extensor Carpi Ulnaris', 
                     'Quattro sensor 4: EMG.B 4': '1st Dorsal Interosseous', 
                     'Quattro sensor 4: EMG.C 4': 'Abductor Digiti Quinti', 
                     'Quattro sensor 4: EMG.D 4': 'Extensor Indicis',
                     'Avanti sensor 5: EMG 5': 'Biceps Brachii'}

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


data_file_path = r"D:\BenQ_Project\01_UR_lab\2024_11 Shanghai CS Major\1. Motion\Major_weight\S06\20241206\S06_SpiderShot_S3_1.c3d"


APP_CONFIG = {
    "DEFAULT_DOWNSAMPLE_FREQ": 1000,
    "DEFAULT_BANDPASS_CUTOFF": [20, 450],
    "DEFAULT_LOWPASS_FREQ": 6,
    "DEFAULT_CSV_NOTCH_CUTOFF_LIST": notch_cutoff, # 假設 50Hz 工頻
    "DEFAULT_C3D_NOTCH_CUTOFF_LIST": c3d_notch_cutoff, # 假設 60Hz 工頻
    "DEFAULT_CSV_RECOLUMNS_NAME": csv_recolumns_name, # 範例
    "DEFAULT_C3D_RECOLUMNS_NAME": c3d_recolumns_name, # 範例
    "EMG_CHANNEL_IDENTIFIER": muscle_name # 用於辨識 EMG 頻道的關鍵字
}

config = APP_CONFIG

# %%


    
# --- 應用程式設定 (理想情況下從設定檔載入) ---
# 這些可以作為 API 的預設參數，或允許用戶透過請求覆蓋


# 核心 EMG 處理邏輯 (從原始程式碼修改而來)
def process_emg_core(
    data_file_path, # 檔案物件的 path
    config = APP_CONFIG, # 包含所有處理參數的字典
    smoothing_method="lowpass", # smoothing 參數
    original_filename=None,
    # window_width=None, # 如果需要
    # overlap_len=None   # 如果需要
):
    if not os.path.exists(data_file_path):
        logging.error(f"檔案路徑不存在: {data_file_path}")
        raise FileNotFoundError(f"檔案路徑不存在: {data_file_path}")

    file_extension = ""
    if '.' in data_file_path:
        file_extension = '.' + data_file_path.rsplit('.', 1)[1].lower()

    if not original_filename:
        original_filename = os.path.basename(data_file_path)

    raw_data = None
    c3d_instance = None
    # 儲存根據 config key 匹配到的原始欄位索引和名稱

    # ----- 檔案讀取與初步頻道識別 -----
    if file_extension == '.csv':
        try:
            raw_data_full_csv = pd.read_csv(data_file_path) # 先完整讀取
            csv_channel_map = config.get("DEFAULT_CSV_RECOLUMNS_NAME", {})
            
            num_columns_indices = []
            emg_signal_columns = []
            for key_identifier in csv_channel_map.keys(): # key_identifier 是 config 中定義的搜索字串
                for i, actual_col_name in enumerate(raw_data_full_csv.columns):
                    if key_identifier in actual_col_name:
                        if i not in num_columns_indices:
                            num_columns_indices.append(i)
                            emg_signal_columns.append(actual_col_name)
                        # break # 假設一個 key_identifier 只對應一個最先匹配到的頻道
                                # 如果一個 key 可能匹配多個，則不應 break

            if not emg_signal_columns:
                raise ValueError(f"在 CSV 檔案 '{original_filename}' 中，根據 DEFAULT_CSV_RECOLUMNS_NAME 的 keys 未找到任何 EMG 頻道。")
            
            # 根據識別出的原始欄位來決定後續操作的 DataFrame
            # 注意：這裡的重命名邏輯與 EMG_processing 不同，EMG_processing 是先選取再重命名選取後的子集
            # 而使用者提供的 process_emg_core 是先識別，然後對整個 raw_data 進行 rename
            # 這裡我們遵循後者，先識別，然後對整個 raw_data_full_csv 進行 rename
            raw_data = raw_data_full_csv.copy() # 操作副本
            raw_data.rename(columns=csv_channel_map, inplace=True)


        except Exception as e:
            logging.error(f"讀取或初步處理 CSV 檔案 '{original_filename}' 時發生錯誤: {e}")
            raise ValueError(f"無法解析或處理 CSV 檔案 '{original_filename}': {e}")

    elif file_extension == '.c3d':
        try:
            c3d_instance = ezc3d.c3d(data_file_path)
            c3d_analog_labels_original = c3d_instance['parameters']['ANALOG']['LABELS']['value']
            c3d_channel_map = config.get("DEFAULT_C3D_RECOLUMNS_NAME", {})

            num_columns_indices = []
            emg_signal_columns = [] # 儲存原始 C3D 標籤名
            
            for key_identifier in c3d_channel_map.keys(): # key_identifier 是 config 中定義的搜索字串
                for i, actual_c3d_label in enumerate(c3d_analog_labels_original):
                    if key_identifier in actual_c3d_label:
                        if i not in num_columns_indices:
                            num_columns_indices.append(i)
                            emg_signal_columns.append(actual_c3d_label)
                        # break # 同上，取決於一個 key 是否只匹配一個

            if not num_columns_indices:
                raise ValueError(f"在 C3D 檔案 '{original_filename}' 中，根據 DEFAULT_C3D_RECOLUMNS_NAME 的 keys 未找到任何 EMG 頻道。")
            
            analog_data_subset = c3d_instance['data']['analogs'][0, num_columns_indices, :]
            # 使用原始 C3D 標籤名創建 DataFrame，然後再重命名
            raw_data_from_c3d = pd.DataFrame(np.transpose(analog_data_subset), columns=emg_signal_columns)
            
            # 進行重命名
            raw_data = raw_data_from_c3d.copy()
            raw_data.rename(columns=c3d_channel_map, inplace=True)
            
            # 插入時間軸
            analog_time = np.linspace(
                0,
                (c3d_instance['header']['analogs']['last_frame']) / c3d_instance['header']['analogs']['frame_rate'],
                num=(np.shape(c3d_instance['data']['analogs'])[-1])
            )
            raw_data.insert(0, 'Frame', analog_time)
            # 欄位重命名 (C3D)
            raw_data.rename(columns=config.get("DEFAULT_C3D_RECOLUMNS_NAME", {}), inplace=True)

        except Exception as e:
            logging.error(f"處理 C3D 檔案 '{original_filename}' 時發生錯誤: {e}")
            raise ValueError(f"無法解析或處理 C3D 檔案 '{original_filename}': {e}")
    else:
        raise ValueError(f"不支援的檔案類型 '{file_extension}'。請上傳 .csv 或 .c3d 檔案。")

    if raw_data is None or raw_data.empty:
        raise ValueError(f"資料讀取失敗或檔案 '{original_filename}' 為空或未成功轉換。")

    # ----- 逐頻道處理 -----
    bandpass_cutoff_freqs = config.get("BANDPASS_CUTOFF", [20, 450])
    perform_notch = config.get("PERFORM_NOTCH_FILTER", True)
    truncate_fft = config.get("FFT_TRUNCATE_TO_POWER_OF_2", True)
    # csv_time_column_explicit = config.get("CSV_TIME_COLUMN_NAME", None) # 明確的CSV時間欄位名
    # MDF 相關設定
    mdf_window_duration = config.get("MDF_WINDOW_DURATION", 1.0) # 秒
    # MDF 窗格的 FFT 是否截斷，與主 FFT 設定一致
    mdf_truncate_segment_fft = config.get("MDF_TRUNCATE_SEGMENT_FFT", truncate_fft)

    down_freq = config.get("DEFAULT_DOWNSAMPLE_FREQ")

    Fs_global = 0
    data_len_global = 0 # 這裡指降採樣前的長度
    min_stop_time_global = 0
    downsample_len_global = 0 # 降採樣後的統一長度
    

    if '.csv' in data_file_path:
        # ... 原碼中 CSV 的 Fs, data_len, all_stop_time, downsample_len 計算邏輯 ...
        # 注意：原碼中 data_time = raw_data.iloc[:,num_columns[col]-1].dropna()
 
        all_fs_csv = []
        all_data_len_csv = []
        all_stop_times_csv = []
        all_downsample_len_csv = []

        for emg_col_idx in num_columns_indices:
            # 原碼的 num_columns[col]-1 邏輯比較脆弱
            # 如果每個EMG頻道有獨立的時間欄，那結構會更複雜
            data_time_series = raw_data.iloc[:, emg_col_idx-1]
            # data_time_series = raw_data.iloc[:, emg_col_idx].dropna()
            if len(data_time_series) < 11:
                raise ValueError(f"時間欄 '{raw_data.columns[num_columns_indices-1]}' 的數據不足以計算取樣頻率。")

            current_fs = (1 / np.mean(np.array(data_time_series[2:11]) - np.array(data_time_series[1:10])))
            all_fs_csv.append(current_fs)
            
            emg_data_series = raw_data.iloc[:, emg_col_idx]
            # 計算 data_len (有效數據長度)
            non_zero_indices = (emg_data_series[::-1] != 0)
            # non_zero_indices = (data_time_series[::-1] != 0)
            if not non_zero_indices.any(): # 如果全是0
                 first_non_zero_from_end_pos = len(emg_data_series)
            else:
                first_non_zero_from_end_pos = non_zero_indices.argmax()

            current_data_len = int(len(emg_data_series) - first_non_zero_from_end_pos)
            all_data_len_csv.append(current_data_len)

            if current_data_len > 0:
                 current_stop_time = data_time_series.iloc[current_data_len -1]
            else: # 如果頻道全是0或空
                current_stop_time = 0 # 或者 NaN，取決於如何處理
            all_stop_times_csv.append(current_stop_time)
            
            all_downsample_len_csv.append(current_data_len / current_fs * down_freq if current_fs > 0 else 0)

        # 清理 NaN 的 stop_time (如果有的話)
        valid_stop_times = [x for x in all_stop_times_csv if not math.isnan(x)]
        if not valid_stop_times:
            raise ValueError("所有頻道的截止時間均無效。")
        
        min_stop_time_csv = np.min(valid_stop_times)
        
        Fs_global = min(all_fs_csv) if all_fs_csv else 0
        # data_len_global 應該是基於 min_stop_time 和 Fs_global 重新計算，或者取最小的有效長度
        # downsample_len_global 取最小的，並確保是整數
        downsample_len_global = math.floor(min(all_downsample_len_csv)) if all_downsample_len_csv else 0
        min_stop_time_global = min_stop_time_csv

    elif '.c3d' in data_file_path:
        Fs_global = c3d_instance['header']['analogs']['frame_rate']
        # data_len_global 是原始 c3d 數據的長度 (影格數)
        data_len_global = np.shape(c3d_instance['data']['analogs'])[-1] # 或 raw_data.shape[0] 如果 'Frame' 欄已移除
        min_stop_time_global = (c3d_instance['header']['analogs']['last_frame']) / Fs_global
        downsample_len_global = math.floor(data_len_global / Fs_global * down_freq)

    if Fs_global <= 0 or downsample_len_global <= 0:
        raise ValueError("無法計算有效的取樣頻率或降採樣長度。")

    logging.info(f"全局取樣頻率 (估計/實際): {Fs_global}, 降採樣後長度: {downsample_len_global}, 統一截止時間: {min_stop_time_global}")

    # ----- 初始化結果 DataFrame -----
    # 欄位名稱使用處理後的 EMG 欄位名
    processed_emg_columns = emg_signal_columns
    
    bandpass_filtered_data_df = pd.DataFrame(np.zeros([downsample_len_global, len(num_columns_indices)]),
                                           columns=processed_emg_columns)
    notch_filtered_data_df = pd.DataFrame(np.zeros([downsample_len_global, len(num_columns_indices)]),
                                         columns=processed_emg_columns)
    lowpass_filtered_data_df = pd.DataFrame(np.zeros([downsample_len_global, len(num_columns_indices)]),
                                           columns=processed_emg_columns)
   
    
    # ----- 2. 濾波與訊號處理 (逐頻道) -----
    bandpass_cutoff_freqs = config.get("DEFAULT_BANDPASS_CUTOFF")
    
    # channel_data_results = {}
    # 這裡的 col 應該是迭代 processed_emg_columns 的索引，或者直接迭代欄位名
    for i, emg_col_name in enumerate(processed_emg_columns):
        emg_col_original_idx = raw_data.columns.get_loc(emg_col_name) # 獲取在 raw_data 中的實際索引
        
        current_sample_freq = 0
        data_to_filter = None

        if '.csv' in data_file_path:
            # 重新計算該頻道的 sample_freq (或者使用之前計算的 Fs_global，如果假設所有頻道一致)
            # 原碼中是重新計算的
            time_series_for_fs = raw_data.iloc[:, emg_col_original_idx-1] # 再次獲取時間序列
            if len(time_series_for_fs) < 11:
                current_sample_freq = Fs_global # Fallback or raise error
            else:
                current_sample_freq = (1 / np.mean(np.array(time_series_for_fs[2:11]) - np.array(time_series_for_fs[1:10])))

            # 準備數據並處理 NaN
            # 原碼中 indi_data_len 的邏輯比較複雜，與 data_len 的更新有關
            # 簡化：直接取該欄位的數據
            series_data = raw_data.iloc[:, emg_col_original_idx].copy() # 使用 .copy() 避免 SettingWithCopyWarning
            
            nan_indices = np.where(np.isnan(series_data))[0]
            if nan_indices.size == 0:
                pass # No NaN
            elif nan_indices.size > 0.1 * current_sample_freq:
                logging.warning(f"頻道 {emg_col_name} 總訊號斷訊 (NaN) 超過 0.1 秒。已將 NaN 替換為 0。")
                series_data.fillna(0, inplace=True)
            else:
                logging.warning(f"頻道 {emg_col_name} 共發現 {nan_indices.size} 個缺值, 位置為 {nan_indices.tolist()}。已將 NaN 替換為 0。")
                series_data.fillna(0, inplace=True)
            
            data_values = series_data.values

            # 截斷數據到 min_stop_time_global
            # 需要找到 min_stop_time_global 在該頻道時間序列中的索引
            # 假設時間序列是 raw_data[time_column_name]
            non_zero_indices = (data_values[::-1] != 0).argmax()
          
            # time_points = raw_data[time_column_name].fillna(0) # 處理時間中的 NaN
            end_index_for_channel = int(len(data_values) - non_zero_indices)
            data_to_filter = data_values[:end_index_for_channel]
            
            notch_freq_list = config.get("DEFAULT_CSV_NOTCH_CUTOFF_LIST")

        elif '.c3d' in data_file_path:
            current_sample_freq = Fs_global # c3d 的 Fs 是固定的
            # c3d 資料在轉換時已處理過長度，理論上所有頻道長度一致
            data_to_filter = raw_data.iloc[:, emg_col_original_idx].values
            notch_freq_list = config.get("DEFAULT_C3D_NOTCH_CUTOFF_LIST")
        
        if data_to_filter is None or len(data_to_filter) == 0:
            logging.warning(f"頻道 {emg_col_name} 沒有數據進行濾波，跳過。")
            continue

        # --- 執行濾波 ---
        # Bandpass
        try:
            bandpass_sos = signal.butter(2, bandpass_cutoff_freqs, btype='bandpass', fs=current_sample_freq, output='sos')
            bandpassed_signal = signal.sosfiltfilt(bandpass_sos, data_to_filter)
        except ValueError as e: # 例如 fs 太低導致的 Nyquist 問題
            logging.error(f"頻道 {emg_col_name} Bandpass 濾波失敗: {e}。Fs={current_sample_freq}, Cutoff={bandpass_cutoff_freqs}")
            # 可以選擇跳過此頻道或填充預設值
            continue

        # Notch
        notched_signal = bandpassed_signal # 起始訊號
        for notch_cutoff in notch_freq_list:
            try:
                # 檢查 notch_cutoff 是否在 Nyquist 頻率內
                if any(f >= current_sample_freq / 2 for f in notch_cutoff) or any(f <= 0 for f in notch_cutoff):
                    logging.warning(f"頻道 {emg_col_name} 的 Notch 頻率 {notch_cutoff} 超出範圍 (Fs={current_sample_freq})，跳過此 Notch。")
                    continue
                if notch_cutoff[0] >= notch_cutoff[1]: # 確保 Wn[0] < Wn[1]
                    logging.warning(f"頻道 {emg_col_name} 的 Notch 頻率範圍不正確 {notch_cutoff}，跳過此 Notch。")
                    continue
                notch_sos = signal.butter(2, notch_cutoff, btype='bandstop', fs=current_sample_freq, output='sos')
                notched_signal = signal.sosfiltfilt(notch_sos, notched_signal)
            except ValueError as e:
                 logging.error(f"頻道 {emg_col_name} Notch 濾波 ({notch_cutoff}) 失敗: {e}。Fs={current_sample_freq}")
                 continue # 跳過這個壞掉的 notch
        
        # Abs
        abs_signal = np.abs(notched_signal)
        lowpass_cutoff_freq = config.get("DEFAULT_LOWPASS_FREQ")

        # Lowpass
        try:
            if lowpass_cutoff_freq >= current_sample_freq / 2 or lowpass_cutoff_freq <= 0:
                logging.warning(f"頻道 {emg_col_name} 的 Lowpass 頻率 {lowpass_cutoff_freq} 超出範圍 (Fs={current_sample_freq})，跳過 Lowpass。")
                lowpassed_signal = abs_signal # 如果跳過，則直接使用 abs_signal
            else:
                lowpass_sos = signal.butter(2, lowpass_cutoff_freq, btype='low', fs=current_sample_freq, output='sos')
                lowpassed_signal = signal.sosfiltfilt(lowpass_sos, abs_signal)
        except ValueError as e:
            logging.error(f"頻道 {emg_col_name} Lowpass 濾波失敗: {e}。Fs={current_sample_freq}, Cutoff={lowpass_cutoff_freq}")
            lowpassed_signal = abs_signal # 出錯時使用 abs_signal
        
        # --- 降採樣 ---
        # `downsample_len_global` 是目標長度
        if len(notched_signal) > 0 :
            resampled_notch = signal.resample(notched_signal, downsample_len_global)
            notch_filtered_data_df.iloc[:, i] = resampled_notch[:downsample_len_global]
        else: # 如果原始訊號為空
            notch_filtered_data_df.iloc[:, i] = np.zeros(downsample_len_global)
        
        if len(bandpassed_signal) > 0:
            resampled_bandpass = signal.resample(bandpassed_signal, downsample_len_global)
            bandpass_filtered_data_df.iloc[:, i] = resampled_bandpass[:downsample_len_global]
        else:
            bandpass_filtered_data_df.iloc[:, i] = np.zeros(downsample_len_global)

        if len(lowpassed_signal) > 0:
            resampled_lowpass = signal.resample(lowpassed_signal, downsample_len_global)
            lowpass_filtered_data_df.iloc[:, i] = resampled_lowpass[:downsample_len_global]
        else:
            lowpass_filtered_data_df.iloc[:, i] = np.zeros(downsample_len_global)


    # ----- 3. 插入時間軸 -----
    # down_freq 是降採樣後的目標頻率
    # 時間軸長度是 downsample_len_global
    final_duration = downsample_len_global / down_freq if down_freq > 0 else 0
    time_index = np.linspace(0, final_duration, downsample_len_global, endpoint=False if downsample_len_global > 0 else True) # endpoint=False 更常見

    bandpass_filtered_data_df.insert(0, 'time', time_index)
    notch_filtered_data_df.insert(0, 'time', time_index)
    lowpass_filtered_data_df.insert(0, 'time', time_index)

    # ----- 回傳 -----
    if smoothing_method == "lowpass":
        return lowpass_filtered_data_df, notch_filtered_data_df
    # elif smoothing_method == "rms":
    #     # return rms_data, bandpass_filtered_data_df (需要實作 RMS)
    # elif smoothing_method == "moving":
    #     # return moving_data, bandpass_filtered_data_df (需要實作 Moving Mean)
    else:
        logging.warning(f"不支援的平滑方法: {smoothing_method}，預設回傳 lowpass 結果。")
        return lowpass_filtered_data_df, notch_filtered_data_df
    
# %%

processed_data1, processed_data2 = process_emg_core(
            file_object_or_path, # 或者 temp_file.name for c3d if needed
            file_extension,
            processing_params,
            smoothing_method
        )
# %%

# @app.route('/process_emg_signal', methods=['POST'])
# def handle_emg_processing():
#     if 'file' not in request.files:
#         return jsonify({"error": "缺少檔案部分"}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "未選擇檔案"}), 400

#     filename = file.filename
#     file_extension = ""
#     if '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv':
#         file_extension = '.csv'
#     elif '.' in filename and filename.rsplit('.', 1)[1].lower() == 'c3d':
#         file_extension = '.c3d'
#     else:
#         return jsonify({"error": "不支援的檔案類型。請上傳 .csv 或 .c3d 檔案。"}), 400

#     try:
#         # 獲取請求中的參數或使用預設值
#         # 這裡可以從 request.form 中獲取用戶自訂的參數來覆蓋 APP_CONFIG 中的預設值
#         processing_params = APP_CONFIG.copy() # Start with defaults
#         for key in ['DEFAULT_DOWNSAMPLE_FREQ', 'DEFAULT_BANDPASS_CUTOFF', 'DEFAULT_LOWPASS_FREQ', 
#                     'DEFAULT_CSV_NOTCH_CUTOFF_LIST', 'DEFAULT_C3D_NOTCH_CUTOFF_LIST',
#                     'EMG_CHANNEL_IDENTIFIER']:
#             if request.form.get(key):
#                 try:
#                     # 需要小心轉換類型，例如列表和數值
#                     # 簡單起見，這裡假設 request.form 中的值都是字串，需要解析
#                     # 例如: processing_params[key] = json.loads(request.form.get(key))
#                     # 這裡僅作示意，實際轉換會更複雜
#                     if key in ['DEFAULT_BANDPASS_CUTOFF', 'DEFAULT_CSV_NOTCH_CUTOFF_LIST', 'DEFAULT_C3D_NOTCH_CUTOFF_LIST']:
#                         processing_params[key] = eval(request.form.get(key)) # eval 不安全，僅為示意，應使用 json.loads 或更安全的解析
#                     elif key in ['DEFAULT_DOWNSAMPLE_FREQ', 'DEFAULT_LOWPASS_FREQ']:
#                          processing_params[key] = int(request.form.get(key))
#                     else: # EMG_CHANNEL_IDENTIFIER
#                         processing_params[key] = request.form.get(key)
#                 except Exception as e:
#                     logging.warning(f"解析請求參數 {key} 失敗: {e}。將使用預設值。")
        
#         smoothing_method = request.form.get("smoothing_method", "lowpass")

#         # 將檔案內容傳遞給核心處理函式
#         # 對於 CSV，可以直接傳遞 file (它是 werkzeug.datastructures.FileStorage，是 file-like)
#         # 對於 C3D，ezc3d 可能需要檔案路徑。一種方法是將上傳的檔案暫存：
#         file_object_or_path = None
#         if file_extension == '.csv':
#             # 轉換為 BytesIO 再傳給 read_csv
#             file_bytes = io.BytesIO(file.read())
#             file_object_or_path = file_bytes
#         elif file_extension == '.c3d':
#             # 為了 ezc3d，可能需要暫存檔案 (如果它不接受 file-like object)
#             # import tempfile
#             # temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.c3d')
#             # file.save(temp_file.name)
#             # file_object_or_path = temp_file.name
#             # # 記得在處理完後刪除 temp_file.name
#             # 簡化：假設 ezc3d 可以處理 file-like object (file)
#             # 查閱 ezc3d 文件確認，如果不行，則必須用暫存檔
#             file_object_or_path = file # 直接傳遞 FileStorage 物件 (需要測試 ezc3d 是否支援)
#                                        # 或者，更安全的方式是 file.stream

#         processed_data1, processed_data2 = process_emg_core(
#             file_object_or_path, # 或者 temp_file.name for c3d if needed
#             file_extension,
#             processing_params,
#             smoothing_method
#         )
        
#         # if file_extension == '.c3d' and isinstance(file_object_or_path, str): # 如果是暫存檔案路徑
#         #    os.remove(file_object_or_path) # 清理暫存檔案

#         # 將 DataFrame 轉換為 JSON
#         # orient='records' 會產生 [{col:val}, {col:val}, ...] 的列表
#         # orient='split' 會產生 {'index': [...], 'columns': [...], 'data': [[...], [...]]}
#         result1_json = processed_data1.to_json(orient="split", double_precision=10, force_ascii=False)
#         result2_json = processed_data2.to_json(orient="split", double_precision=10, force_ascii=False)
        
#         return jsonify({
#             "message": "EMG 訊號處理成功",
#             "smoothing_method": smoothing_method,
#             "processed_smoothed_data": json.loads(result1_json), # json.loads 將字串轉回字典/列表結構
#             "processed_bandpass_notch_data": json.loads(result2_json)
#         }), 200

#     except ValueError as ve:
#         logging.error(f"處理請求時發生 Value Error: {ve}")
#         return jsonify({"error": str(ve)}), 400
#     except Exception as e:
#         logging.exception(f"處理請求時發生未預期錯誤: {e}") # logging.exception 會包含堆疊追蹤
#         # if file_extension == '.c3d' and isinstance(file_object_or_path, str) and os.path.exists(file_object_or_path):
#         #    os.remove(file_object_or_path) # 清理暫存檔案
#         return jsonify({"error": f"內部伺服器錯誤: {e}"}), 500

# if __name__ == '__main__':
#     # 啟動 Flask 應用 (僅用於本地測試)
#     # 在生產環境中，應使用 WSGI 伺服器如 Gunicorn 或 uWSGI
#     app.run(debug=True, host='0.0.0.0', port=5000)