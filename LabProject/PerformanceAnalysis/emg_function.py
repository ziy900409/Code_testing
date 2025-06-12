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
from typing import List, Dict, Optional, Any, Set, Tuple  # Added Set for type hinting
from scipy.fft import fft, fftfreq # 使用 scipy.fft

from collections import defaultdict
import matplotlib.pyplot as plt

from adjustText import adjust_text
plt.rcParams['font.sans-serif'] =  ['Roboto']  
plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號
plt.rcParams['figure.dpi'] = 150
from scipy.stats import linregress
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d # 需要導入
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


# %%

# --- 應用程式設定 (理想情況下從設定檔載入) ---
# 這些可以作為 API 的預設參數，或允許用戶透過請求覆蓋
# data_file_path = data_path
# config = EMG_CONFIG
# 核心 EMG 處理邏輯 (從原始程式碼修改而來)
def process_emg_core(
    data_file_path, # 檔案物件的 path
    config, # 包含所有處理參數的字典
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
            
            # 先識別，然後對整個 raw_data_full_csv 進行 rename
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
    # truncate_fft = config.get("FFT_TRUNCATE_TO_POWER_OF_2", True)
    # csv_time_column_explicit = config.get("CSV_TIME_COLUMN_NAME", None) # 明確的CSV時間欄位名
    # MDF 相關設定
    # mdf_window_duration = config.get("MDF_WINDOW_DURATION", 1.0) # 秒
    # # MDF 窗格的 FFT 是否截斷，與主 FFT 設定一致
    # mdf_truncate_segment_fft = config.get("MDF_TRUNCATE_SEGMENT_FFT", truncate_fft)

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
    bandpass_filtered_data_df = pd.DataFrame(np.zeros([downsample_len_global, len(num_columns_indices)]),
                                           columns=emg_signal_columns)
    notch_filtered_data_df = pd.DataFrame(np.zeros([downsample_len_global, len(num_columns_indices)]),
                                         columns=emg_signal_columns)
    lowpass_filtered_data_df = pd.DataFrame(np.zeros([downsample_len_global, len(num_columns_indices)]),
                                           columns=emg_signal_columns)
    
    # ----- 新增：初始化平均值 DataFrame -----
    averaged_data_df = None
    # 時間窗口設置與 MDF 相同
    avg_window_duration_config = config.get("MDF_WINDOW_DURATION", 1)
    num_averaged_points_global = 0

    if avg_window_duration_config > 0 and down_freq > 0:
        samples_per_avg_window_global = int(down_freq * avg_window_duration_config)
        if samples_per_avg_window_global > 0:
            num_averaged_points_global = downsample_len_global // samples_per_avg_window_global
            if num_averaged_points_global > 0:
                averaged_data_df = pd.DataFrame(np.zeros([num_averaged_points_global, len(emg_signal_columns)]),
                                                columns=emg_signal_columns)
                logging.info(f"將計算 {avg_window_duration_config}s 窗格平均值，產生 {num_averaged_points_global} 點。")
            else:
                logging.warning(f"訊號總長度不足以產生至少一個 {avg_window_duration_config}s 的平均窗格 (基於降採樣後數據)。")
        else:
            logging.warning(f"平均窗格時長 {avg_window_duration_config}s 相對於降採樣頻率 {down_freq}Hz 過短，無法定義窗格樣本數。")
   
    # ----- 2. 濾波與訊號處理 (逐頻道) -----
    bandpass_cutoff_freqs = config.get("DEFAULT_BANDPASS_CUTOFF")
    emg_results = defaultdict(dict)
    emg_results["filename"] = original_filename
    # channel_data_results = {}
    # 這裡的 col 應該是迭代 emg_signal_columns 的索引，或者直接迭代欄位名
    for i, emg_col_name in enumerate(emg_signal_columns):
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
        if perform_notch:
            abs_signal = np.abs(notched_signal)
        else:
            abs_signal = np.abs(bandpassed_signal)
            
        lowpass_cutoff_freq = config.get("DEFAULT_LOWPASS_FREQ", 6)

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
        # 新增標準化動作
        if len(lowpassed_signal) > 0:
            resampled_lowpass = signal.resample(lowpassed_signal, downsample_len_global)
            lowpass_filtered_data_df.iloc[:, i] = resampled_lowpass[:downsample_len_global]
            std_data = resampled_lowpass[:downsample_len_global] / max(resampled_lowpass[:downsample_len_global]) * 100
            emg_results["Smoothing"][emg_col_name] = std_data
        else:
            lowpass_filtered_data_df.iloc[:, i] = np.zeros(downsample_len_global)
        
        # ----- 新增：計算時間窗格平均值 -----
        if averaged_data_df is not None and num_averaged_points_global > 0:
            signal_to_average = resampled_lowpass # 此時長度為 downsample_len_global
            # current_effective_fs_for_avg = down_freq # 降採樣後的有效 Fs
            
            # samples_per_avg_window_global 已在前面計算過
            # num_averaged_points_global 也已在前面計算過 (即 num_windows)

            averaged_values_for_channel = []
            if samples_per_avg_window_global > 0 and len(signal_to_average) >= samples_per_avg_window_global :
                for win_idx in range(num_averaged_points_global): # 迭代預期數量的窗格
                    segment_start = win_idx * samples_per_avg_window_global
                    segment_end = (win_idx + 1) * samples_per_avg_window_global
                    # 確保 segment_end 不超過 signal_to_average 的長度
                    # 雖然理論上 num_averaged_points_global * samples_per_avg_window_global <= len(signal_to_average)
                    window_segment = signal_to_average[segment_start:min(segment_end, len(signal_to_average))]
                    
                    if len(window_segment) > 0:
                        # 處理窗格內可能的 NaN 值 (例如來自失敗的降採樣)
                        if np.all(np.isnan(window_segment)):
                            mean_value = np.nan
                        else:
                            mean_value = np.nanmean(window_segment) # nanmean 會忽略 NaN
                        averaged_values_for_channel.append(mean_value)
                    else:
                        # 如果因邊界條件導致窗格為空(理論上不應發生在此循環結構)
                        averaged_values_for_channel.append(np.nan) 
            else: # 訊號不足一個窗格，或窗格樣本數為0 (前面已有log)
                 averaged_values_for_channel = [np.nan] * num_averaged_points_global


            # 填充到 DataFrame，確保長度一致
            # 新增標準化的方法
            if len(averaged_values_for_channel) == num_averaged_points_global:
                averaged_data_df.iloc[:, i] = averaged_values_for_channel
                averaged_values_for_channel = averaged_values_for_channel / max(averaged_values_for_channel) * 100
                emg_results["AverageData"][emg_col_name] = averaged_values_for_channel
                time_axis = np.arange(len(averaged_values_for_channel))
                # 計算趨勢線的斜率
                slope, intercept, r_value, p_value, std_err = linregress(time_axis, averaged_values_for_channel)
            elif len(averaged_values_for_channel) < num_averaged_points_global: # 如果產生值較少
                temp_array = np.full(num_averaged_points_global, np.nan)
                temp_array[:len(averaged_values_for_channel)] = averaged_values_for_channel
                averaged_data_df.iloc[:, i] = temp_array
                temp_array = temp_array / max(temp_array) * 100
                emg_results["AverageData"][emg_col_name] = temp_array
                time_axis = np.arange(len(temp_array))
                # 計算趨勢線的斜率
                slope, intercept, r_value, p_value, std_err = linregress(time_axis, temp_array)
            else: # 如果產生值較多 (不應發生)
                averaged_data_df.iloc[:, i] = averaged_values_for_channel[:num_averaged_points_global]
        
        emg_results["SamplingRate"][emg_col_name] = current_sample_freq
        # 儲存趨勢線的斜率
        emg_results["Amplitudes_Slope"][emg_col_name] = slope
        

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
        return emg_results
    # elif smoothing_method == "rms":
    #     # return rms_data, bandpass_filtered_data_df (需要實作 RMS)
    # elif smoothing_method == "moving":
    #     # return moving_data, bandpass_filtered_data_df (需要實作 Moving Mean)
    else:
        logging.warning(f"不支援的平滑方法: {smoothing_method}，預設回傳 lowpass 結果。")
        # return lowpass_filtered_data_df, notch_filtered_data_df
    
# %%
def calculate_fft_for_emg(
    data_file_path, # 檔案的完整路徑
    config,
    original_filename=None # 原始檔案名稱，用於記錄
):
    """
    計算 EMG 數據的 FFT 並回傳頻譜數據。ˇ

    參數:
    - data_file_path (str): 原始資料檔案的完整路徑。
    - config (dict): 包含所有處理參數的字典。
    - original_filename (str, optional): 原始檔案名稱，用於記錄。

    回傳:
    - dict: 包含 FFT 結果的字典。
    """
    
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
            
            # 先識別，然後對整個 raw_data_full_csv 進行 rename
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
    truncate_fft = config.get("FFT_TRUNCATE_TO_POWER_OF_2", False)
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
    emg_signal_columns = emg_signal_columns
    
    # ----- 2. 濾波與訊號處理 (逐頻道) -----
    bandpass_cutoff_freqs = config.get("DEFAULT_BANDPASS_CUTOFF")
    
    fft_results = defaultdict(dict)
    fft_results["filename"] = original_filename
    # channel_data_results = {}
    # 這裡的 col 應該是迭代 emg_signal_columns 的索引，或者直接迭代欄位名
    for i, emg_col_name in enumerate(emg_signal_columns):
        emg_col_original_idx = raw_data.columns.get_loc(emg_col_name) # 獲取在 raw_data 中的實際索引
        
        current_sample_freq = 0
        data_to_filter = None

        if '.csv' in data_file_path:
            # 重新計算該頻道的 sample_freq 
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
        
        if perform_notch:
            fft_input_data = notched_signal.copy()
        else:
            fft_input_data = bandpassed_signal.copy()

        # --- FFT 計算 ---
        N = len(fft_input_data)
        if N == 0:
            fft_results["error"] = "Data length for FFT is zero after filtering."
            # fft_results["channels_fft_data"].append(channel_data_results)
            continue

        if truncate_fft:
            N_truncated = 2**(N.bit_length() - 1) if N > 1 else N # 避免 N=1 時 N_truncated=0
            if N_truncated > 0 and N_truncated < N : # 只在有意義截斷時才截斷
                fft_input_data = fft_input_data[:N_truncated]
                N = N_truncated
            elif N_truncated == 0 and N > 0: # N=1 的情況
                 logging.warning(f"頻道 '{emg_col_name}' ({original_filename}) 數據長度 {len(fft_input_data)} 過短，無法截斷到2的冪次方，將使用原始長度。")
            # else N_truncated == N, 不用做任何事

        T = 1.0 / current_sample_freq
        yf = fft(fft_input_data, n=N)
        xf = fftfreq(N, T)[:N // 2]
        amplitudes = (2.0 / N) * np.abs(yf[0:N // 2])
        fft_results["SamplingRate"][emg_col_name] = current_sample_freq
        fft_results["frequencies"][emg_col_name] = xf.tolist()
        fft_results["amplitudes"][emg_col_name] = amplitudes.tolist()

        # --- 找出前三大峰值 ---
        if len(amplitudes) > 0:
            amp_for_peaks = np.copy(amplitudes)
            peaks_found = []
            for _ in range(3):
                if not np.any(np.isfinite(amp_for_peaks)) or np.max(amp_for_peaks) == float('-inf') or len(amp_for_peaks)==0:
                    break
                max_idx = np.argmax(amp_for_peaks)
                if max_idx < len(xf):
                    peaks_found.append({"frequency": xf[max_idx], "amplitude": amplitudes[max_idx]})
                    amp_for_peaks[max_idx] = float('-inf')
                else: break
            fft_results["top_peaks"][emg_col_name] = peaks_found
        
        # 3. 每一個 duration 計算一次 FFT
        med_freq_list_for_channel = [] # 儲存此頻道隨時間變化的 MDF
        # ----- MDF 計算 (時域中頻數率) -----
        if mdf_window_duration > 0 and current_sample_freq > 0:
            fft_step_mdf = int(current_sample_freq * mdf_window_duration)
            if fft_step_mdf > 0 and len(fft_input_data) >= fft_step_mdf : # 確保至少有一個完整窗格
                num_windows_mdf = len(fft_input_data) // fft_step_mdf
                
                for win_idx in range(num_windows_mdf):
                    segment = fft_input_data[win_idx * fft_step_mdf : (win_idx + 1) * fft_step_mdf]
                    N_segment_orig = len(segment)
                    if N_segment_orig == 0: continue

                    N_fft_segment = N_segment_orig
                    if mdf_truncate_segment_fft: # 使用MDF特定的截斷設定或與主FFT一致
                        N_temp_segment = 2**(N_segment_orig.bit_length() - 1) if N_segment_orig > 1 else N_segment_orig
                        if N_temp_segment > 0: N_fft_segment = N_temp_segment
                    
                    segment_for_fft = segment[:N_fft_segment] if N_fft_segment < N_segment_orig else segment
                    
                    yf_segment = fft(segment_for_fft, n=N_fft_segment)
                    # T_segment is 1.0 / freq
                    xf_segment = fftfreq(N_fft_segment, 1.0 / current_sample_freq)[:N_fft_segment // 2]
                    # 計算功率譜密度 (PSD) 而不是單純的振幅譜，通常MDF基於PSD
                    # PSD_segment = (1.0 / (current_sample_freq * N_fft_segment)) * np.abs(yf_segment[0:N_fft_segment // 2])**2 # 單邊PSD
                    # 或者，如果仍用振幅譜的定義：
                    amplitudes_segment = (2.0 / N_fft_segment) * np.abs(yf_segment[0:N_fft_segment // 2])

                    if len(amplitudes_segment) == 0: # 如果窗格FFT結果為空
                        med_freq_list_for_channel.append(np.nan) # 或 0.0
                        continue

                    total_power_segment = np.sum(amplitudes_segment) # 或 np.sum(PSD_segment)
                    if total_power_segment <= 1e-9: # 避免除零或極小功率的情況
                        med_freq_list_for_channel.append(0.0) # 或 np.nan
                        continue
                        
                    cumulative_power = 0.0
                    found_mdf_for_window = False
                    for freq_idx, power_val in enumerate(amplitudes_segment): # 或 PSD_segment
                        cumulative_power += power_val
                        if cumulative_power >= total_power_segment / 2.0:
                            if freq_idx < len(xf_segment):
                                med_freq_list_for_channel.append(xf_segment[freq_idx])
                                found_mdf_for_window = True
                                break
                            else: # 索引超出 xf_segment 範圍
                                logging.warning(f"MDF: 頻道 {emg_col_name}, 窗格 {win_idx}, freq_idx 越界。")
                                med_freq_list_for_channel.append(np.nan)
                                found_mdf_for_window = True
                                break
                    if not found_mdf_for_window: # 如果循環結束仍未找到 (例如所有功率集中在最後)
                        med_freq_list_for_channel.append(xf_segment[-1] if len(xf_segment) > 0 else 0.0)
            else:
                logging.info(f"頻道 '{emg_col_name}' ({original_filename}) 資料長度不足以進行MDF計算 (窗格長度 {fft_step_mdf} vs 資料長度 {len(fft_input_data)})。")
        # 儲存 Median Frequency 的結果
        fft_results["MedianFreq"][emg_col_name] = med_freq_list_for_channel
        
        time_axis = np.arange(len(med_freq_list_for_channel))
        # 計算趨勢線的斜率
        slope, intercept, r_value, p_value, std_err = linregress(time_axis, med_freq_list_for_channel)
        fft_results["MedianFreq_Slope"][emg_col_name] = slope

    return fft_results
# %%
def plot_fft_data_output(fft_results, max_subplot_cols=2,
                         title_name=None):
    """
    接收來自 calculate_fft_for_emg_data_v2 的結果，並繪製頻譜圖。

    參數:
    - fft_results_data (dict): 包含 FFT 分析結果的字典，
                               其結構應為 calculate_fft_for_emg_data_v2 的輸出。
    - max_subplot_cols (int): 子圖每行最大欄數。
    """
    if not fft_results or "amplitudes" not in fft_results or not fft_results["amplitudes"]:
        print("沒有有效的頻道數據可以繪製。")
        return

    num_channels = len(fft_results["amplitudes"])

    if num_channels == 0:
        print("頻道數據為空，無法繪製。")
        return
    
    # 計算子圖的行數和列數
    cols = min(max_subplot_cols, num_channels)
    rows = math.ceil(num_channels / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5), squeeze=False)
    # squeeze=False 確保 axs 總是一個二維陣列，即使只有一行或一列
    if title_name:
        fig_title = f"FFT Analysis: {title_name}"
    else:
        fig_title = f"FFT Analysis: {fft_results.get('filename', '未知檔案')}"
    
    fig.suptitle(fig_title, fontsize=24)
    # 存放所有標註（避免重疊用）
    all_texts = []

    for i, channel_info in enumerate(fft_results["amplitudes"].keys()):
        print(channel_info)
        
        row_idx = i // cols
        col_idx = i % cols
        ax = axs[row_idx, col_idx]
        texts = []

        frequencies = fft_results.get("frequencies", {}).get(channel_info, None)
        amplitudes = fft_results.get("amplitudes").get(channel_info, None)
        top_peaks = fft_results.get("top_peaks", []).get(channel_info, None)
        fs_used = fft_results.get("SamplingRate").get(channel_info, None)

        ax.set_title(f"{channel_info}\n(Fs used: {fs_used if fs_used else 'N/A'} Hz)", fontsize=16)

        if frequencies is None or amplitudes is None or len(frequencies) == 0 or len(amplitudes) == 0:
            ax.text(0.5, 0.5, "數據不足", ha='center', va='center', color='gray', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # 繪製頻譜
        ax.plot(frequencies, amplitudes, linewidth=0.7, color='dodgerblue')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 3),
                            useMathText=True) # Y軸科學記號

        # 標註峰值
        for peak_idx, peak in enumerate(top_peaks):
            peak_freq = peak.get("frequency")
            peak_amp = peak.get("amplitude")
            if peak_freq is not None and peak_amp is not None:
                ax.plot(peak_freq, peak_amp, 'o', color='red', markersize=4)
                # 稍微錯開標註位置以避免重疊
                offset_y = (peak_idx % 3 -1) * (0.1 * max(amplitudes) if amplitudes else 0.01)
                text = ax.annotate(f'{peak_freq:.1f}Hz',
                                xy=(peak_freq, peak_amp),
                                xytext=(5, 5 + offset_y*100), textcoords='offset points', # 調整 xytext
                                fontsize=12, color='crimson',
                                arrowprops=dict(arrowstyle="->", color='gray', connectionstyle="arc3,rad=.2"))

                texts.append(text)
        
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        ax.tick_params(axis='both', labelsize=12)

        # 設定 X 軸範圍
        if frequencies and fs_used:
            # 通常顯示到奈奎斯特頻率，或者一個感興趣的上限
            display_max_freq = fs_used / 2
            # 如果數據中的最大頻率遠小於奈奎斯特頻率，可以適當調整
            if max(frequencies) < display_max_freq / 2 and max(frequencies) > 0:
                display_max_freq = min(display_max_freq, max(frequencies) * 1.5) # 給一些緩衝
            ax.set_xlim(0, min(display_max_freq, 500)) # 例如最多顯示到500Hz或奈奎斯特頻率
        elif frequencies:
             ax.set_xlim(0, min(max(frequencies) if len(frequencies) > 1 else 500, 500))
        else:
            ax.set_xlim(0, 500) # 預設

    # 如果有未使用的子圖，隱藏它們
    for i in range(num_channels, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        if row_idx < axs.shape[0] and col_idx < axs.shape[1]: # 確保索引在範圍內
            fig.delaxes(axs[row_idx, col_idx])
            
    plt.tight_layout(rect=[0.03, 0.03, 1.01, 0.98])  # 預留空間給 suptitle 和共用標籤
    # plt.tight_layout()
    fig.supxlabel("Frequency (Hz)", fontsize=20)
    fig.supylabel("Amplitude", fontsize=20)
    # 自動調整所有標註位置避免重疊
    adjust_text(all_texts, arrowprops=dict(arrowstyle="->", color='gray'))
    plt.show()
# %%

def plot_mdf_over_time(fft_results_data, config, 
                       max_subplot_cols=2, title_name=None):
    """
    接收來自 calculate_fft_for_emg_data_v3 的結果，並繪製 MDF 時程圖。

    參數:
    - fft_results_data (dict): 包含 FFT 分析結果及 MDF 分析的字典。
                               其結構應為 calculate_fft_for_emg_data_v3 的輸出。
    - max_subplot_cols (int): 子圖每行最大欄數。
    """
    # fft_results_data = fft_results
    if not fft_results_data or "MedianFreq" not in fft_results_data:
        print("沒有有效的 MDF 分析數據可以繪製。")
        return

    mdf_analysis_data = fft_results_data.get("MedianFreq", {})
    if not mdf_analysis_data:
        print("MDF 分析數據為空，無法繪製。")
        return

    # 篩選出實際有 MDF 數據的頻道
    channels_with_mdf = {
        name: data for name, data in mdf_analysis_data.items() if data and any(not np.isnan(x) for x in data)
    }
    
    if not channels_with_mdf:
        print("所有頻道的 MDF 數據均為空或 NaN，無法繪製。")
        return

    num_channels = len(channels_with_mdf)

    # 計算子圖的行數和列數
    cols = min(max_subplot_cols, num_channels)
    rows = math.ceil(num_channels / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4), squeeze=False)
    # squeeze=False 確保 axs 總是一個二維陣列
    if title_name:
        fig_title = f"Median Frequency (MDF) Over Time: {title_name}"
    else:
        fig_title = f"Median Frequency (MDF) Over Time: {fft_results_data.get('filename', '未知檔案')}"
    fig.suptitle(fig_title, fontsize=24, y=1 - 0.02 * rows)

    plot_idx = 0
    for channel_name, mdf_values in channels_with_mdf.items():
        row_idx = plot_idx // cols
        col_idx = plot_idx % cols
        ax = axs[row_idx, col_idx]

        
        # fs_used = None
        mdf_window_duration = None # 需要從 config 或 fft_results_data 中獲取

        # 嘗試從主 FFT 結果中找到對應頻道的 fs_used (如果有的話)
        # 並假設 MDF 的 config 參數 (如 MDF_WINDOW_DURATION) 也許可以間接得知
        # 這裡簡化，如果需要精確時間軸，MDF 計算時應同時儲存時間點
        for ch_fft_data in fft_results_data.get("channels_fft_data", []):
            if ch_fft_data.get("channel_name") == channel_name:
                # fs_used = ch_fft_data.get("sampling_frequency_used")
                mdf_window_duration = config.get("DURATION", 1)
                break
        # 嘗試從 channels_fft_data 中獲取該頻道的取樣頻率，以推算時間軸
        # 這部分是可選的，如果沒有，則 x 軸就是窗格索引
        time_axis = np.arange(0, #start
                              len(mdf_values)*mdf_window_duration, # stop
                              mdf_window_duration) # step

        ax.plot(time_axis, mdf_values, marker='o', linestyle='-', linewidth=1, markersize=3, label="MDF")
        
        # 計算趨勢線的斜率
        slope, intercept, r_value, p_value, std_err = linregress(time_axis, mdf_values)
        trendline = intercept + slope * np.array(time_axis)

        # 畫趨勢線
        ax.plot(time_axis, trendline, linewidth=1, color='red', linestyle='--',
                label='Slope: {:.2f}'.format(slope))
        # annotation_text = 'Slope: {:.2f}'.format(slope)
        # ax.annotate(annotation_text, xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=12)
        ax.legend(fontsize=10)
        
        title_str = f"{channel_name}"

        ax.set_title(title_str, fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        # 設定 X 軸範圍
        if time_axis.any():
            ax.set_xlim(0, max(time_axis)) 
        plot_idx += 1

    # 如果有未使用的子圖，隱藏它們
    for i in range(plot_idx, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        if row_idx < axs.shape[0] and col_idx < axs.shape[1]:
            fig.delaxes(axs[row_idx, col_idx])

    plt.tight_layout(rect=[0.03, 0.03, 1.01, 0.92])
    fig.supxlabel("Time (s)", fontsize=16)
    fig.supylabel("Median Frequency (Hz)", fontsize=16)
    plt.show()
# %%
def plot_multiple_mdf_over_time(list_of_fft_results_data,
                                configs,
                                max_subplot_cols=2,
                                title_name=None,
                                dataset_labels=None,
                                selected_keys: list = None):
    """
    接收一個或多個 FFT 分析結果的列表，並繪製 MDF 時程圖。
    每個頻道一個子圖，每個子圖可包含來自多個數據集的 MDF 時程線及趨勢線。

    參數:
    - list_of_fft_results_data (list or dict): 包含一個或多個 FFT 分析結果字典的列表。
                                              每個字典的結構應包含 "MedianFreq" 鍵。
    - configs (dict or list of dict): 單個配置字典 (適用於所有數據集) 或配置字典列表。
                                     每個配置字典應包含 "DURATION" (MDF 窗口持續時間)。
    - max_subplot_cols (int): 子圖每行最大欄數。
    - title_name (str, optional): 圖表的整體標題。
    - dataset_labels (list of str, optional): 每個數據集的標籤，用於圖例。
    """

    if not list_of_fft_results_data:
        print("沒有提供 FFT 結果數據。")
        return
    if not configs:
        print("沒有提供配置信息 (configs)。")
        return

    # --- 1. 輸入標準化與數據有效性檢查 ---
    if isinstance(list_of_fft_results_data, dict):
        list_of_fft_results_data = [list_of_fft_results_data]
        if dataset_labels and not isinstance(dataset_labels, list):
            dataset_labels = [dataset_labels]
    
    if isinstance(configs, dict):
        configs = [configs] * len(list_of_fft_results_data) # 複製配置給每個數據集

    if len(configs) != len(list_of_fft_results_data):
        print("警告: configs 列表長度與 list_of_fft_results_data 長度不匹配。請檢查輸入。")
        return

    # 準備數據集標籤
    if dataset_labels:
        if len(dataset_labels) != len(list_of_fft_results_data):
            print("警告: dataset_labels 數量與數據集數量不符。將使用自動標籤。")
            dataset_labels = None
    if not dataset_labels:
        dataset_labels = [fft_res.get('filename', f'Dataset {i+1}')
                          for i, fft_res in enumerate(list_of_fft_results_data)]

    # --- 2. 收集所有唯一的、包含有效 MDF 數據的頻道名稱 ---
    all_valid_channel_names = set()
    # 先收集所有數據集中所有可能的頻道
    temp_all_channel_names = set()
    for fft_results in list_of_fft_results_data:
        if "MedianFreq" in fft_results and fft_results["MedianFreq"]:
            temp_all_channel_names.update(fft_results["MedianFreq"].keys())
    
    # 對於每個潛在頻道，檢查是否至少有一個數據集包含該頻道的有效MDF數據
    for channel_name in temp_all_channel_names:
        has_valid_data_for_channel = False
        for fft_results in list_of_fft_results_data:
            mdf_values = fft_results.get("MedianFreq", {}).get(channel_name)
            if mdf_values is not None and isinstance(mdf_values, (list, np.ndarray)) and \
               len(mdf_values) > 0 and any(not np.isnan(x) for x in mdf_values if x is not None): # 檢查非空且至少有一個非NaN值
                has_valid_data_for_channel = True
                break
        if has_valid_data_for_channel:
            all_valid_channel_names.add(channel_name)

    if not all_valid_channel_names:
        print("在所有數據集中均未找到有效的 MDF 數據頻道。")
        return

    # 可以設定需要繪圖的keys()
    if selected_keys:
        sorted_channel_names = [k for k in selected_keys if k in all_valid_channel_names]
    else:
        # sorted_channel_names = sorted(valid_channels)
        sorted_channel_names = sorted(list(all_valid_channel_names))    

    num_unique_channels = len(sorted_channel_names)

    # --- 3. 子圖佈局 ---
    cols = min(max_subplot_cols, num_unique_channels)
    rows = math.ceil(num_unique_channels / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5), squeeze=False)

    # 設定整體標題
    if title_name:
        fig_title_text = f"Median Frequency (MDF) Over Time: {title_name}"
    elif len(list_of_fft_results_data) == 1:
        fig_title_text = f"Median Frequency (MDF) Over Time: {dataset_labels[0]}"
    else:
        fig_title_text = "Median Frequency (MDF) Over Time: Multiple Datasets"
    
    title_y_adjust = 0.98 # 初始y位置
    if rows > 1:
        title_y_adjust = 1 - 0.04 * (1/rows + 0.05) # 根據行數調整，避免與子圖標題太近
    elif num_unique_channels == 0 : # 雖然前面有檢查，但以防萬一
        title_y_adjust = 0.95

    fig.suptitle(fig_title_text, fontsize=20, y=title_y_adjust)


    # --- 4. 顏色和繪圖循環 ---
    prop_cycle = plt.rcParams['axes.prop_cycle']
    plot_colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])] 
                   for i in range(len(list_of_fft_results_data))]

    for i, channel_name in enumerate(sorted_channel_names):
        row_idx = i // cols
        col_idx = i % cols
        ax = axs[row_idx, col_idx]
        ax.set_title(f"{channel_name}", fontsize=16)
        
        max_time_for_subplot = 0 # 用於設定 xlim
        found_data_for_channel_in_any_dataset = False

        for dataset_idx, (fft_results, config_item) in enumerate(zip(list_of_fft_results_data, configs)):
            dataset_label = dataset_labels[dataset_idx]
            color = plot_colors[dataset_idx % len(plot_colors)]

            mdf_values_list = fft_results.get("MedianFreq", {}).get(channel_name)

            if mdf_values_list is not None and isinstance(mdf_values_list, (list, np.ndarray)) and \
               len(mdf_values_list) > 0:
                
                mdf_values = np.array([val for val in mdf_values_list if val is not None]) # 過濾掉 None
                
                # 移除 NaN 以便繪製和計算趨勢線
                valid_indices = ~np.isnan(mdf_values)
                if not np.any(valid_indices): # 如果過濾後沒有有效數據點
                    # print(f"數據集 '{dataset_label}' 的頻道 '{channel_name}' MDF 數據全為 NaN 或 None。")
                    continue

                mdf_values_clean = mdf_values[valid_indices]
                
                mdf_window_duration = config_item.get("DURATION", 1) # 預設為1秒
                if not isinstance(mdf_window_duration, (int, float)) or mdf_window_duration <= 0:
                    print(f"警告: 數據集 '{dataset_label}' 的 DURATION ({mdf_window_duration}) 無效，將使用1秒。")
                    mdf_window_duration = 1

                # 生成對應有效點的時間軸
                original_indices = np.arange(len(mdf_values))
                time_axis_full = original_indices * mdf_window_duration
                time_axis_clean = time_axis_full[valid_indices]

                if len(mdf_values_clean) < 2: # 需要至少兩個點來畫線和計算趨勢
                    # print(f"數據集 '{dataset_label}' 的頻道 '{channel_name}' 有效 MDF 數據點不足 (<2)。")
                    ax.plot(time_axis_clean, mdf_values_clean, marker='o', linestyle='', markersize=3, 
                            color=color, label=f"{dataset_label} - MDF (Data points)")
                    if time_axis_clean.any():
                         max_time_for_subplot = max(max_time_for_subplot, time_axis_clean.max())
                    found_data_for_channel_in_any_dataset = True
                    continue

                found_data_for_channel_in_any_dataset = True
                max_time_for_subplot = max(max_time_for_subplot, time_axis_clean.max())

                # 繪製 MDF 線
                ax.plot(time_axis_clean, mdf_values_clean, marker='o', linestyle='-', linewidth=1, 
                        markersize=3, color=color, alpha=0.6)

                # 計算並繪製趨勢線
                try:
                    # Scipy >= 1.6.0 支持 nan_policy='omit'. 若舊版, 前面已手動清理NaN
                    slope, intercept, r_value, p_value, std_err = linregress(time_axis_clean, mdf_values_clean)
                    trendline = intercept + slope * time_axis_clean
                    ax.plot(time_axis_clean, trendline, linewidth=3, color=color, linestyle='--',
                            label=f"{dataset_label} - Slope: {slope:.2f}", alpha=0.6)
                except ValueError as e:
                    print(f"計算頻道 '{channel_name}' 數據集 '{dataset_label}' 的趨勢線時出錯: {e}")
            # else:
                # print(f"數據集 '{dataset_label}' 中頻道 '{channel_name}' 的 MDF 數據缺失或格式不正確。")


        if not found_data_for_channel_in_any_dataset:
            ax.text(0.5, 0.5, "無有效MDF數據", ha='center', va='center', color='gray', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.tick_params(axis='both', labelsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=False)) # 允許非整數時間點
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False)) # MDF可以是浮點數
            ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
            ax.legend(fontsize=10)
            ax.tick_params(axis='both', labelsize=12)
            if max_time_for_subplot > 0 :
                ax.set_xlim(0, max_time_for_subplot) # 加一點緩衝
            else: # 如果只有單點或沒有時間軸
                ax.set_xlim(0,1) 


    # --- 5. 清理和顯示 ---
    # 隱藏未使用的子圖
    for i in range(num_unique_channels, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        if row_idx < axs.shape[0] and col_idx < axs.shape[1]:
            fig.delaxes(axs[row_idx, col_idx])

    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.98]) # 調整rect以適應suptitle
    
    if rows > 0 and cols > 0:
        try:
            fig.supxlabel("Time (s)", fontsize=20, y=0.01)
            fig.supylabel("Median Frequency (Hz)", fontsize=20, x=0.01 if cols >1 else 0.04)
        except Exception as e:
            print(f"設置 supxlabel/supylabel 時出錯: {e}")
            
    plt.show()
# %%
# list_of_averaged_data = [averaged_df, averaged_df_1, averaged_df_2]
# dataset_labels = ['55g', '65g', "60g"]
def plot_multiple_emg_data_over_time(
    list_of_emg_data,
    configs,
    max_subplot_cols=2,
    title_name=None,
    dataset_labels=None,
    y_axis_label="Averaged EMG Amplitude (AU)",
    show_trendline=True, # New parameter to control trendline plotting
    selected_keys: list = None
):
    """
    Plots time-windowed averaged EMG data from multiple datasets.

    Parameters:
    - list_of_averaged_data (list or pd.DataFrame): A list of Pandas DataFrames,
        where each DataFrame is an 'averaged_data_df' (e.g., from process_emg_core).
        Each DataFrame must have a 'time' column as the first column, and
        subsequent columns representing EMG channels with their averaged values.
        If a single DataFrame is passed, it will be treated as a single dataset.
    - max_subplot_cols (int): Maximum number of subplots per row.
    - title_name (str, optional): Overall title for the figure.
    - dataset_labels (list of str, optional): Labels for each dataset, for the legend.
        Should match the order and number of DataFrames in list_of_averaged_data.
    - y_axis_label (str): Label for the Y-axis of the subplots.
    - show_trendline (bool): If True, calculates and plots a linear trendline for each series.
    """

    # --- 1. Input Validation and Standardization ---
    if not list_of_emg_data:
        print("No averaged data provided for plotting.")
        return

    if isinstance(list_of_emg_data, dict):
        list_of_emg_data = [list_of_emg_data]
        if dataset_labels and not isinstance(dataset_labels, list):
            dataset_labels = [dataset_labels]
        elif dataset_labels and len(dataset_labels) != 1:
            print("Warning: dataset_labels count mismatch for a single DataFrame input. Ignoring labels.")
            dataset_labels = None

    # if not all(isinstance(df, dict) for df in list_of_averaged_data):
    #     print("Error: All items in list_of_averaged_data must be Pandas DataFrames.")
    #     return

    # Prepare dataset labels
    if dataset_labels:
        if len(dataset_labels) != len(list_of_emg_data):
            print("Warning: dataset_labels count does not match the number of datasets. Using default labels.")
            dataset_labels = None
    if not dataset_labels:
        dataset_labels = [f'Dataset {i+1}' for i in range(len(list_of_emg_data))]

    
    # --- 2. 收集所有唯一的、包含有效 MDF 數據的頻道名稱 ---
    # 先收集所有數據集中所有可能的頻道
    
    all_valid_channel_names = set()
    for emg_results in list_of_emg_data:
        if "AverageData" in emg_results and emg_results["AverageData"]:
            all_valid_channel_names.update(emg_results["AverageData"].keys())
       
    # 對於每個潛在頻道，檢查是否至少有一個數據集包含該頻道的有效MDF數據
    for channel_name in all_valid_channel_names:
        has_valid_data_for_channel = False
        for emg_results in list_of_emg_data:
            mdf_values = emg_results.get("AverageData", {}).get(channel_name)
            if mdf_values is not None and isinstance(mdf_values, (list, np.ndarray)) and \
               len(mdf_values) > 0 and any(not np.isnan(x) for x in mdf_values if x is not None): # 檢查非空且至少有一個非NaN值
                has_valid_data_for_channel = True
                break
        if has_valid_data_for_channel:
            all_valid_channel_names.add(channel_name)
    
    if not all_valid_channel_names:
        print("在所有數據集中均未找到有效的 MDF 數據頻道。")
        return
    
    # 可以設定需要繪圖的keys()
    if selected_keys:
        sorted_channel_names = [k for k in selected_keys if k in all_valid_channel_names]
    else:
        # sorted_channel_names = sorted(valid_channels)
        sorted_channel_names = sorted(list(all_valid_channel_names))

    
    num_unique_channels = len(sorted_channel_names)

    # --- 3. Subplot Layout Calculation ---
    cols = min(max_subplot_cols, num_unique_channels)
    rows = math.ceil(num_unique_channels / cols)

    # --- 4. Figure and Axes Creation ---
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5), squeeze=False)

    # Overall figure title
    if title_name:
        fig_title_text = f"Time-Windowed Averaged EMG: {title_name}"
    elif len(list_of_emg_data) == 1 and dataset_labels:
        fig_title_text = f"Time-Windowed Averaged EMG: {dataset_labels[0]}"
    else:
        fig_title_text = "Time-Windowed Averaged EMG: Multiple Datasets"
    
    title_y_adjust = 0.98
    if rows > 1: title_y_adjust = 1 - 0.04 * (1/rows + 0.05)
    elif num_unique_channels == 0: title_y_adjust = 0.95
    fig.suptitle(fig_title_text, fontsize=20, y=title_y_adjust)


    # --- 5. Main Plotting Loop ---
    prop_cycle = plt.rcParams['axes.prop_cycle']
    plot_colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])]
                   for i in range(len(list_of_emg_data))]

    for i, channel_name in enumerate(sorted_channel_names):
        row_idx = i // cols
        col_idx = i % cols
        ax = axs[row_idx, col_idx]
        ax.set_title(f"{channel_name}", fontsize=16)

        max_time_for_subplot = 0
        found_data_for_channel_in_any_dataset = False

        # for dataset_idx, avg_df in enumerate(list_of_emg_data):
        for dataset_idx, (emg_results, config_item) in enumerate(zip(list_of_emg_data, configs)):
            dataset_label = dataset_labels[dataset_idx]
            color = plot_colors[dataset_idx % len(plot_colors)]

            if channel_name in all_valid_channel_names:
                duration = config_item.get("DURATION", 1) 
                time_axis = np.arange(0, len(emg_results["AverageData"][channel_name]), duration)
                mdf_values_list = emg_results.get("AverageData", {}).get(channel_name)
                channel_data = np.array(emg_results["AverageData"][channel_name])

                # Remove NaNs for plotting and trendline
                valid_indices = ~np.isnan(channel_data)
                time_clean = time_axis[valid_indices]
                data_clean = channel_data[valid_indices]

                if len(data_clean) > 0:
                    found_data_for_channel_in_any_dataset = True
                    if len(time_clean) > 0:
                         max_time_for_subplot = max(max_time_for_subplot, time_clean.max())

                    # Plot the averaged data
                    ax.plot(time_clean, data_clean, marker='o', linestyle='-', linewidth=1,
                            markersize=3, color=color, alpha=0.8)

                    # Calculate and plot trendline if enabled and enough data
                    if show_trendline and len(data_clean) >= 2:
                        try:
                            slope, intercept, r_value, p_value, std_err = linregress(time_clean, data_clean)
                            trendline = intercept + slope * time_clean
                            ax.plot(time_clean, trendline, linewidth=1.5, color=color, linestyle='--',
                                    label=f"{dataset_label} - Slope: {slope:.2f}", alpha=0.6)
                        except ValueError as e:
                            print(f"Could not calculate trendline for {dataset_label}, {channel_name}: {e}")
                # else:
                #     print(f"No valid data for {dataset_label}, channel {channel_name} after NaN removal.")
            # else:
            #     print(f"Channel {channel_name} or 'time' not found in dataset {dataset_label}.")


        if not found_data_for_channel_in_any_dataset:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', color='gray', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.tick_params(axis='both', labelsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=False))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6)) # Y-axis might be float
            ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
            ax.legend(fontsize=8)
            if max_time_for_subplot > 0:
                ax.set_xlim(0, max_time_for_subplot * 1.05)
            else: # If only one point or no time extent
                ax.set_xlim(0, 1)


    # --- 6. Clean Up Unused Subplots ---
    for i in range(num_unique_channels, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        if row_idx < axs.shape[0] and col_idx < axs.shape[1]: # Check bounds
            fig.delaxes(axs[row_idx, col_idx])

    # --- 7. Add Super Labels and Adjust Layout ---
    plt.tight_layout(rect=[0.02, 0.03, 0.98, title_y_adjust - 0.03]) # Adjust rect for suptitle

    if rows > 0 and cols > 0 :
        try:
            fig.supxlabel("Time (s)", fontsize=16, y=0.01)
            fig.supylabel(y_axis_label, fontsize=16, x=0.01 if cols > 1 else 0.04)
        except Exception as e:
            print(f"Error setting sup-labels: {e}")

    # --- 8. Show Plot ---
    plt.show()
# %%

def process_emg_data_with_direction(pre_excldueCen_df,
                                    emg_results,
                                    dataset_labels=None, # 未使用
                                    selected_keys: list = None):

    # --- 2. 收集所有唯一的、包含有效 MDF 數據的頻道名稱 ---
    # 先收集所有數據集中所有可能的頻道
    all_valid_channel_names = set()
    
    # 步驟 2.1: 從 emg_results["Smoothing"] 獲取基礎頻道列表
    if "Smoothing" in emg_results and emg_results["Smoothing"] and isinstance(emg_results["Smoothing"], dict):
        all_valid_channel_names.update(emg_results["Smoothing"].keys())
    else:
        print("錯誤：emg_results 中缺少 'Smoothing' 鍵，或者其值無效。無法繼續處理。")
        return {} # 返回空字典表示處理失敗

    # 步驟 2.2: (此部分邏輯有問題，如後續分析)
    # 對於每個潛在頻道，檢查是否至少有一個數據集包含該頻道的有效MDF數據
    # --- 問題點 1 & 2 開始 ---
    # 這段邏輯看起來是想根據 emg_results["AverageData"] 來驗證頻道，但存在幾個問題。
    # 1. `break` 會導致循環提前終止，只檢查一個頻道。
    # 2. `all_valid_channel_names.add(channel_name)` 的意圖不明確，因為集合已經包含了來自 "Smoothing" 的鍵。
    # 3. 如果這個檢查的目的是過濾 `all_valid_channel_names`，則應該創建一個新的集合。
    # 鑒於後續的插值主要依賴 "Smoothing" 中的數據，此段 AverageData 檢查可能需要重寫或移除，
    # 除非它有特定且正確實現的用途。
    # 目前，我將假設主要頻道列表來自 "Smoothing"。

    # 假設我們只基於 "Smoothing" 的鍵和 "selected_keys" 來確定要處理的頻道：
    # (移除了有問題的 AverageData 檢查循環)

    if not all_valid_channel_names: # 檢查 "Smoothing" 是否提供了任何頻道
        print("在 emg_results['Smoothing'] 中未找到任何 EMG 頻道。")
        return {}
    
    # 步驟 2.3: 根據 selected_keys 過濾頻道
    if selected_keys:
        # 確保 selected_keys 中的頻道確實存在於從 "Smoothing" 獲取的頻道列表中
        processed_channel_names = [k for k in selected_keys if k in all_valid_channel_names]
        if not processed_channel_names:
            print(f"警告：提供的 selected_keys ({selected_keys}) 中的頻道均未在可用頻道中找到。將處理所有可用頻道。")
            processed_channel_names = sorted(list(all_valid_channel_names))
    else:
        processed_channel_names = sorted(list(all_valid_channel_names))
    
    if not processed_channel_names:
        print("沒有可供處理的 EMG 頻道。")
        return {}
    # --- 問題點 1 & 2 結束 ---

    pre_emg_results_smoothing = emg_results["Smoothing"] # 確保這是個字典

    # 步驟 3: 找出 pre_excldueCen_df ["Shot Count"] == 1 的欄位
    # --- 問題點 4 開始 ---
    # df_shot_one = pre_excldueCen_df # 原始程式碼缺少過濾
    # 修正：應該根據 'Shot Count' == 1 進行過濾
    if 'Shot Count' not in pre_excldueCen_df.columns:
        print("錯誤：'pre_excldueCen_df' 中缺少 'Shot Count' 欄位。")
        return {}
    df_shot_one = pre_excldueCen_df[pre_excldueCen_df['Shot Count'] == 1].copy()
    if df_shot_one.empty:
        print("在 'pre_excldueCen_df' 中沒有找到 'Shot Count' == 1 的記錄。")
        return {"right": {}, "left": {}} # 雖然沒有數據，但返回期望的結構
    # --- 問題點 4 結束 ---

    # 步驟 4: 數據提取、插值與分類
    interpolated_data = {"right": defaultdict(dict), "left": defaultdict(dict)} # 使用 defaultdict 以簡化後續賦值
    num_points_interpolated = 141 # 這是您指定的插值點數

    for index, row in df_shot_one.iterrows():
        group_id = row.get('Group ID', f"UnknownGroup_{index}") # 使用 .get() 避免 KeyError
        frame_info_val = row.get('Frames')
        direction_quadrant = row.get('Direction Quadrant')
        
        # 統一使用 group_id 作為最外層鍵，符合 EMGPlotter 的預期結構
        # data_key 將用於 target_group_dict[group_id]
        
        if not isinstance(frame_info_val, list) or len(frame_info_val) < 2:
            print(f"警告：跳過 Group ID {group_id}，因 'Frames' 格式不正確: {frame_info_val}")
            continue
        
        # 新的影格索引起始條件和計算
        # 確保 frame_info_val[0] 和 frame_info_val[1] 是數字
        try:
            start_frame = float(frame_info_val[0])
            end_frame = float(frame_info_val[1])
        except (ValueError, TypeError):
            print(f"警告：跳過 Group ID {group_id}，因 'Frames' 包含非數字值: {frame_info_val}")
            continue

        if start_frame < 20:
            print(f"警告：跳過 Group ID {group_id}，因起始影格 ({start_frame}) < 20。")
            continue
        
        emg_start_index = int((start_frame - 20) * 10)
        emg_end_index = int(end_frame * 10)

        target_group_storage_key = None # "right" or "left"
        if direction_quadrant in ['Q1', 'Q4']:
            target_group_storage_key = "right"
        elif direction_quadrant in ['Q2', 'Q3']:
            target_group_storage_key = "left"
        else:
            print(f"警告：Group ID {group_id} 的方向象限 '{direction_quadrant}' 無效。跳過此 Group。")
            continue
        
        # 創建一個臨時字典來存儲此 group_id 下所有 EMG 通道的插值結果
        # 結構: {"EMGChannelName1": array, "EMGChannelName2": array}
        current_group_interpolated_emgs = {}

        # --- 問題點 3 開始 ---
        # 應遍歷 `processed_channel_names` 而不是 `pre_emg_results_smoothing.items()`
        # 以確保 `selected_keys` 的過濾生效。
        for emg_key in processed_channel_names: # 遍歷經過篩選的頻道名稱
            if emg_key not in pre_emg_results_smoothing:
                print(f"警告：選擇的頻道 {emg_key} 未在 emg_results['Smoothing'] 中找到。跳過此頻道於 Group {group_id}。")
                continue
            emg_full_signal = pre_emg_results_smoothing[emg_key]
        # --- 問題點 3 結束 ---

            if not isinstance(emg_full_signal, np.ndarray): # 確保信號是 numpy array
                print(f"警告：頻道 {emg_key} 的數據不是 NumPy 陣列。跳過此頻道於 Group {group_id}。")
                continue

            actual_end_index = min(emg_end_index + 1, len(emg_full_signal)) # Python 切片不包含末端
            actual_start_index = min(emg_start_index, actual_end_index) # 確保 start 不超過 end
            
            segment = emg_full_signal[actual_start_index:actual_end_index]
            
            # cleaned_emg_key 未被使用，emg_key 直接作為字典鍵
            # cleaned_emg_key = emg_key.replace(' ', '_').replace('.', '') 

            interpolated_sequence = np.full(num_points_interpolated, np.nan) # 預設為 NaN

            if len(segment) < 4: # 三次樣條插值至少需要4個點
                # print(f"警告：Group {group_id}, EMG {emg_key} 的數據片段長度 ({len(segment)}) < 4。填充為 NaN。")
                pass # 減少重複的警告信息，因為已經預設為 NaN
            elif len(segment) == 0:
                # print(f"警告：Group {group_id}, EMG {emg_key} 的數據片段為空。填充為 NaN。")
                pass
            else:
                x_original = np.linspace(0, 1, num=len(segment))
                x_new = np.linspace(0, 1, num=num_points_interpolated)
                try:
                    interpolator = interp1d(x_original, segment, kind='cubic',
                                            bounds_error=False, fill_value="extrapolate")
                    interpolated_sequence = interpolator(x_new)
                except ValueError as e:
                    print(f"錯誤：Group {group_id}, EMG {emg_key} 插值失敗: {e}。片段長度: {len(segment)}。填充為 NaN。")
            
            current_group_interpolated_emgs[emg_key] = interpolated_sequence # 使用原始 emg_key
        
        # 將此 Group 的所有 EMG 插值結果存儲到對應的方向和 Group ID下
        # 結構: interpolated_data["right"]["Group_1"] = {"EMG1": array, "EMG2": array}
        if current_group_interpolated_emgs: # 僅當該 group 有處理成功的 EMG 通道時才添加
            interpolated_data[target_group_storage_key][f"Group_{group_id}"] = current_group_interpolated_emgs
            
    return interpolated_data

# # 執行處理
# processed_data_directional = process_emg_data_with_direction()

# # 輸出結果 (部分範例)
# print("\n--- Processed Interpolated Data with Direction (Sample) ---")
# for direction, data_group in processed_data_directional.items():
#     print(f"\nData for '{direction}' group:")
#     if not data_group:
#         print("  No data in this group.")
#         continue
#     count = 0
#     for key, value in data_group.items():
#         print(f"  '{key}': Array of float64, shape {value.shape}, first 3 values: {np.round(value[:3], 3)}")
#         count += 1
#         if count >= 3: # 每個方向組只印出前3個結果作為範例
#             print("  ...")
#             break
# if not any(processed_data_directional.values()):
#      print("No data was processed. Check filters and input data.")
     
# %%

# -----------------------------------------------------------------------------
# SECTION 1: CORE PLOTTING FUNCTION (plot_standardized_signals_cloud_compare)
# -----------------------------------------------------------------------------

def plot_standardized_signals_cloud_compare(
    datasets: List[Dict[str, np.ndarray]],
    target_length: int,
    title: str = "Comparison of Mean ± Std Dev Clouds",
    xlabel: str = "Normalized Time (-40% to 100%)", # MODIFIED XLABEL to reflect new range
    ylabel: str = "Signal Value",
    labels: Optional[List[str]] = None,
    color_indices: Optional[List[int]] = None
) -> None:
    """ Plots mean ± std dev for multiple datasets of standardized signals. """
    def _calculate_stats(signals_dict: Dict[str, np.ndarray], length: int) -> Optional[tuple]:
        if not signals_dict or not isinstance(signals_dict, dict): return None
        
        valid_signals = [s for s in signals_dict.values() if isinstance(s, np.ndarray) and s.ndim == 1 and s.size == length and not np.all(np.isnan(s))]
        if not valid_signals: return None

        try:
            stacked = np.stack(valid_signals, axis=0) 
            avg = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
            return avg, avg - std, avg + std, len(valid_signals)
        except Exception: 
            return None

    if not isinstance(datasets, list) or not datasets:
        print("Plotting Error: 'datasets' must be a non-empty list of signal dictionaries.")
        return
    
    num_datasets = len(datasets)
    if labels is None or len(labels) != num_datasets:
        labels = [f'Dataset {i+1}' for i in range(num_datasets)]
    
    palette = plt.get_cmap('tab10') 

    fig, ax = plt.subplots(figsize=(12, 7)) # Slightly wider for new x-axis range
    
    # MODIFICATION: Change x_axis generation and limits
    x_axis = np.linspace(-40, 100, target_length) # X-axis from -40 to 100
    
    plotted_any = False

    for i, signal_collection in enumerate(datasets):
        stats = _calculate_stats(signal_collection, target_length)
        if stats:
            avg_signal, lower_bound, upper_bound, num_trials = stats
            if color_indices and i < len(color_indices):
                 color_val = palette(color_indices[i] % palette.N)
            else: 
                 color_val = palette(i % palette.N)

            ax.plot(x_axis, avg_signal, color=color_val, label=f'{labels[i]} (n={num_trials})', linewidth=2)
            ax.fill_between(x_axis, lower_bound, upper_bound, color=color_val, alpha=0.2)
            plotted_any = True
        else:
            print(f"Plotting Info: Could not compute statistics for {labels[i]}. Skipping.")

    if plotted_any:
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # MODIFICATION: Change x-axis limits
        ax.set_xlim(-40, 100) 
        
        ax.tick_params(labelsize=11)
        plt.tight_layout()
        plt.show()
    else:
        print("Plotting Error: No data could be plotted for any dataset.")
        plt.close(fig)
    
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import linregress # Not used in this specific method, but kept if other methods need it
import math
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict

# --- 假設 plot_standardized_signals_cloud_compare (如果您的 EMGPlotter 內部有其他方法直接調用它) ---
# --- 以及 EMGPlotter 類別的先前定義 (包括 __init__, _extract_unique_emg_channels, _calculate_signal_stats) ---
# --- 都已經存在於您的程式碼環境中。為簡潔起見，這裡不再重複它們。 ---
# --- 請確保從前一個回應中複製這些定義。 ---

class EMGPlotter:
    def __init__(self, processed_data_directional: Dict[str, Dict[str, Dict[str, np.ndarray]]], target_length: int = 101):
        if not isinstance(processed_data_directional, dict):
            raise ValueError("processed_data_directional must be a dictionary.")
        self.data_directional = processed_data_directional
        self.target_length = target_length
        self._unique_emg_channels = self._extract_unique_emg_channels()
        if not self._unique_emg_channels:
            print("EMGPlotter Warning: No unique EMG channels found in the provided data.")
        else:
            print(f"EMGPlotter initialized. Found unique EMG channels: {self._unique_emg_channels}")

    def _extract_unique_emg_channels(self) -> List[str]:
        emg_channels_set: Set[str] = set()
        for direction_key, groups_data in self.data_directional.items():
            if isinstance(groups_data, dict):
                for group_id, emg_signals_dict in groups_data.items():
                    if isinstance(emg_signals_dict, dict): # Handles dict or defaultdict
                        for emg_channel_name in emg_signals_dict.keys():
                            emg_channels_set.add(emg_channel_name)
        return sorted(list(emg_channels_set))
        
    def _calculate_signal_stats(self, signals_dict: Dict[str, np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        if not signals_dict or not isinstance(signals_dict, dict):
            return None
        
        valid_signals = [
            s for s in signals_dict.values() 
            if isinstance(s, np.ndarray) and s.ndim == 1 and s.size == self.target_length and not np.all(np.isnan(s))
        ]

        if not valid_signals:
            return None

        try:
            stacked_signals = np.stack(valid_signals, axis=0)
            num_valid_signals = stacked_signals.shape[0]
            with np.errstate(all='ignore'): # Suppress warnings for slices with all NaNs
                mean_signal = np.nanmean(stacked_signals, axis=0)
                std_signal = np.nanstd(stacked_signals, axis=0)
            
            if np.all(np.isnan(mean_signal)): # If mean is all NaNs
                return None
            std_signal = np.nan_to_num(std_signal) # Replace NaN std (e.g., if only 1 trial) with 0
            lower_bound = mean_signal - std_signal
            upper_bound = mean_signal + std_signal
            return mean_signal, lower_bound, upper_bound, num_valid_signals
        except Exception as e:
            print(f"Plotter Error (_calculate_signal_stats): Could not stack/calculate stats: {e}")
            return None

    # --- 修改後的繪圖方法 ---
    def plot_emg_summary_by_direction_with_cloud(
        self,
        muscle_groups_to_plot: Dict[str, List[str]], # 新的參數格式
        main_title: str = "EMG Activity Summary (Mean ± Std Dev)",
        y_axis_label: str = "EMG Amplitude (AU)",
        x_axis_label: str = "Time (-40 to 100 units)",
        share_y_axis: bool = True # Applies to all subplots in the figure if True
    ) -> None:
        """
        根據提供的 muscle_groups_to_plot 字典，繪製多行子圖。
        每行包含兩個子圖 (Left/Right)，每張子圖上以雲圖形式顯示該行指定的多條肌肉的 EMG 活動。
        X 軸範圍為 -40 到 100。
        """
        if not muscle_groups_to_plot or not isinstance(muscle_groups_to_plot, dict):
            print("Plotter Info: 'muscle_groups_to_plot' must be a non-empty dictionary.")
            return
        
        num_rows = len(muscle_groups_to_plot)
        if num_rows == 0:
            print("Plotter Info: No muscle groups provided for plotting.")
            return

        fig, axs = plt.subplots(num_rows, 2, 
                                figsize=(18, 6 * num_rows), # 調整高度以適應行數
                                sharey=share_y_axis, 
                                squeeze=False) # squeeze=False 確保 axs 始終是 2D 陣列
        
        time_axis = np.linspace(-40, 100, self.target_length) # X-axis from -40 to 100
        palette = plt.get_cmap('tab10') # Colormap

        for row_idx, (row_title, emg_channels_for_row) in enumerate(muscle_groups_to_plot.items()):
            if not emg_channels_for_row or not isinstance(emg_channels_for_row, list):
                print(f"Plotter Warning: Skipping row '{row_title}' due to empty or invalid EMG channel list.")
                # 可以選擇隱藏這一行的子圖
                if axs.shape[0] > row_idx: # 檢查 axs 是否已創建足夠的行
                    if axs.shape[1] > 0: axs[row_idx, 0].set_visible(False)
                    if axs.shape[1] > 1: axs[row_idx, 1].set_visible(False)
                continue

            valid_emg_channels_for_row = [ch for ch in emg_channels_for_row if ch in self._unique_emg_channels]
            if not valid_emg_channels_for_row:
                print(f"Plotter Info: None of the selected EMG channels for row '{row_title}' ({emg_channels_for_row}) are valid or found. Skipping this row.")
                if axs.shape[0] > row_idx:
                    if axs.shape[1] > 0: axs[row_idx, 0].set_visible(False)
                    if axs.shape[1] > 1: axs[row_idx, 1].set_visible(False)
                continue
            
            print(f"Plotter Info: Plotting row '{row_title}' for muscles: {valid_emg_channels_for_row}")

            for col_idx, direction in enumerate(["left", "right"]):
                ax = axs[row_idx, col_idx]
                ax.set_title(f"{direction.capitalize()} Group - {row_title}", fontsize=14)
                plotted_anything_on_ax = False
                
                current_direction_data = self.data_directional.get(direction, {})

                for color_c_idx, emg_channel in enumerate(valid_emg_channels_for_row):
                    trials_for_this_emg_and_direction: Dict[str, np.ndarray] = {}
                    if isinstance(current_direction_data, dict):
                        for group_id, group_emg_data in current_direction_data.items():
                            if isinstance(group_emg_data, dict) and emg_channel in group_emg_data:
                                signal = group_emg_data[emg_channel]
                                if isinstance(signal, np.ndarray) and signal.size == self.target_length and not np.all(np.isnan(signal)):
                                    trials_for_this_emg_and_direction[group_id] = signal
                    
                    if trials_for_this_emg_and_direction:
                        stats_result = self._calculate_signal_stats(trials_for_this_emg_and_direction)
                        
                        if stats_result:
                            mean_signal, lower_bound, upper_bound, num_trials = stats_result
                            color = palette(color_c_idx % palette.N) # Color by EMG channel index within this row's list
                            
                            ax.plot(time_axis, mean_signal, color=color, label=f'{emg_channel} (n={num_trials})', linewidth=2)
                            ax.fill_between(time_axis, lower_bound, upper_bound, color=color, alpha=0.2)
                            plotted_anything_on_ax = True

                if plotted_anything_on_ax:
                    ax.legend(fontsize=9, loc='best')
                    ax.grid(True, linestyle=':', alpha=0.6)
                    ax.set_xlim(-40, 100)
                    ax.tick_params(axis='y', labelsize=10)
                    if row_idx == num_rows - 1: # X 軸標籤只顯示在最底部的子圖
                        ax.set_xlabel(x_axis_label, fontsize=12)
                        ax.tick_params(axis='x', labelsize=10)
                    else: # 隱藏非底部子圖的 X 軸刻度標籤
                        ax.tick_params(axis='x', labelbottom=False)
                    
                    if col_idx == 0: # Y 軸標籤只顯示在最左邊的子圖
                        ax.set_ylabel(y_axis_label, fontsize=12)
                    # 如果 share_y_axis=True，matplotlib 會自動處理右邊子圖的 Y 軸刻度標籤是否顯示
                    # 如果 share_y_axis=False，且 col_idx > 0，則matplotlib也會自動顯示Y軸標籤
                else:
                    ax.text(0.5, 0.5, "No data for selected channels", ha="center", va="center", transform=ax.transAxes, color="grey")
                    ax.set_xlim(-40, 100)
                    if row_idx == num_rows - 1: ax.set_xlabel(x_axis_label, fontsize=12)
                    else: ax.tick_params(axis='x', labelbottom=False)
                    if col_idx == 0: ax.set_ylabel(y_axis_label, fontsize=12)


        fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.99 if num_rows ==1 else 1.00)
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95 if num_rows >1 else 0.92]) # Adjust rect based on rows
        plt.show()

    # --- 保留您可能需要的舊方法 ---
    # def plot_time_comparison(...): ...
    # def plot_cloud_comparison_per_emg_channel(...): ... 
    # (這些方法也應該檢查並確保其X軸設定與您的需求一致，如果它們還被使用的話)

# --- 示例數據和用法 ---
if __name__ == '__main__':
    # ... (此處應有 plot_multiple_emg_data_over_time 和 plot_standardized_signals_cloud_compare 的完整定義，
    # 或者您的 EMGPlotter 類別包含所有需要的方法。為簡潔起見，這裡省略這些函數的重複粘貼。)
    # 確保您的環境中定義了 plot_standardized_signals_cloud_compare，因為 EMGPlotter 的其他方法可能依賴它。

    processed_data_directional_example = {
        "left": { 
            "Group_L1": defaultdict(lambda: np.full(101, np.nan),{
                "Biceps.IM EMG8": np.sin(np.linspace(0, np.pi*2, 101)) * 0.8 + 0.4 + np.random.rand(101)*0.15,
                "Triceps.IM EMG9": np.cos(np.linspace(0, np.pi*2, 101)) * 0.9 + 0.5 + np.random.rand(101)*0.2,
                "DorInter.IM EMG4": np.random.rand(101) * 0.5 + 0.2,
                "AbdDigMin.IM EMG5": np.random.rand(101) * 0.4 + 0.1,
            }),
             "Group_L2": defaultdict(lambda: np.full(101, np.nan),{
                "Biceps.IM EMG8": np.sin(np.linspace(0, np.pi*2, 101) -0.1) * 0.85 + 0.45 + np.random.rand(101)*0.22,
                "Triceps.IM EMG9": np.cos(np.linspace(0, np.pi*2, 101) -0.2) * 0.95 + 0.55 + np.random.rand(101)*0.28,
                "DorInter.IM EMG4": np.random.rand(101) * 0.55 + 0.22,
                "AbdDigMin.IM EMG5": np.random.rand(101) * 0.42 + 0.12,
            })
        },
        "right": {
            "Group_R1": defaultdict(lambda: np.full(101, np.nan), {
                "Biceps.IM EMG8": np.sin(np.linspace(0, np.pi*2, 101)) + 0.5 + np.random.rand(101)*0.2,
                "Triceps.IM EMG9": np.cos(np.linspace(0, np.pi*2, 101)) + 0.6 + np.random.rand(101)*0.3,
                "DorInter.IM EMG4": np.random.rand(101) * 0.6 + 0.25,
                "AbdDigMin.IM EMG5": np.random.rand(101) * 0.45 + 0.15,
            }),
            "Group_R2": defaultdict(lambda: np.full(101, np.nan),{
                "Biceps.IM EMG8": np.sin(np.linspace(0, np.pi*2, 101) + 0.2) + 0.55 + np.random.rand(101)*0.25,
                "Triceps.IM EMG9": np.cos(np.linspace(0, np.pi*2, 101) + 0.1) + 0.65 + np.random.rand(101)*0.35,
                "DorInter.IM EMG4": np.random.rand(101) * 0.62 + 0.28,
                "AbdDigMin.IM EMG5": np.random.rand(101) * 0.48 + 0.18,
            })
        }
    }

    plotter_instance = EMGPlotter(processed_data_directional_example, target_length=101)
    
    # --- 調用新的多行雲圖繪製功能 ---
    
    # 示例 1: 只有一組肌肉 (一行，兩個子圖 Left/Right)
    muscles_to_plot_fig1 = {
        "Arm Muscles": ['Biceps.IM EMG8', 'Triceps.IM EMG9', 'NonExistentMuscle'] # 包含一個不存在的肌肉以測試過濾
    }
    print(f"\nPlotting cloud summary for: {muscles_to_plot_fig1}")
    plotter_instance.plot_emg_summary_by_direction_with_cloud(
        muscle_groups_to_plot=muscles_to_plot_fig1,
        main_title="EMG Activity: Arm Muscles (X: -40 to 100)",
        share_y_axis=True
    )

    # 示例 2: 兩組肌肉 (兩行，每行兩個子圖 Left/Right)
    muscles_to_plot_fig2 = {
        "Upper Limb": ['Biceps.IM EMG8', 'Triceps.IM EMG9'],
        "Hand Intrinsic": ['DorInter.IM EMG4', 'AbdDigMin.IM EMG5']
    }
    print(f"\nPlotting cloud summary for: {muscles_to_plot_fig2}")
    plotter_instance.plot_emg_summary_by_direction_with_cloud(
        muscle_groups_to_plot=muscles_to_plot_fig2,
        main_title="EMG Activity: Upper Limb & Hand (X: -40 to 100)",
        share_y_axis=True # 嘗試 share_y_axis=False 來看看效果
    )

    # 示例 3: 包含空肌肉列表的行 (應跳過該行)
    muscles_to_plot_fig3 = {
        "Valid Arm Muscles": ['Biceps.IM EMG8'],
        "Empty Hand Group": [],
        "Another Valid Group": ['Triceps.IM EMG9']
    }
    print(f"\nPlotting cloud summary for: {muscles_to_plot_fig3}")
    plotter_instance.plot_emg_summary_by_direction_with_cloud(
        muscle_groups_to_plot=muscles_to_plot_fig3,
        main_title="EMG Activity: Testing Empty Group (X: -40 to 100)",
        share_y_axis=True
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

# %%import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import linregress # 不再需要，因為雲圖主要展示均值和標準差區域
import math
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict

# --- 輔助函數 (獨立於任何類別) ---
def _standalone_get_unique_emg_channels(
    direction_data: Dict[str, Dict[str, np.ndarray]]
) -> List[str]:
    """從單個方向的數據中提取所有唯一的 EMG 頻道名稱。"""
    emg_channels_set: Set[str] = set()
    if isinstance(direction_data, dict):
        for group_id, emg_signals_dict in direction_data.items():
            if isinstance(emg_signals_dict, dict):
                for emg_channel_name in emg_signals_dict.keys():
                    emg_channels_set.add(emg_channel_name)
    return sorted(list(emg_channels_set))

def _standalone_calculate_emg_stats_for_direction(
    direction_data: Dict[str, Dict[str, np.ndarray]], # 例如 one_dataset["left"]
    emg_channels_to_process: List[str],
    target_length: int
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]]: # EMGName -> (avg, low_bound, upp_bound, N_trials)
    """
    為指定方向數據中的每個 EMG 頻道計算統計數據 (平均值, 標準差上下限, 試驗次數)。
    """
    output_stats: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}

    for emg_channel_name in emg_channels_to_process:
        trials_for_this_emg: Dict[str, np.ndarray] = {} # Key: group_id, Value: signal_array
        if isinstance(direction_data, dict):
            for group_id, group_emg_data in direction_data.items():
                if isinstance(group_emg_data, dict) and emg_channel_name in group_emg_data:
                    signal = group_emg_data[emg_channel_name]
                    if isinstance(signal, np.ndarray) and signal.size == target_length and not np.all(np.isnan(signal)):
                        trials_for_this_emg[group_id] = signal
        
        if not trials_for_this_emg:
            # print(f"Stats Helper: No valid trials for {emg_channel_name} in given direction_data.")
            continue

        # 計算統計數據
        valid_signals_list = list(trials_for_this_emg.values())
        try:
            stacked = np.stack(valid_signals_list, axis=0) 
            avg = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
            num_trials = len(valid_signals_list)

            if np.all(np.isnan(avg)):
                # print(f"Stats Helper Warning: Mean for {emg_channel_name} is all NaN despite valid trials.")
                continue # 跳過此 EMG 頻道如果平均值全是 NaN
            
            std = np.nan_to_num(std) # 處理 std 可能為 NaN 的情況 (例如只有一個試驗)
            lower_bound = avg - std
            upper_bound = avg + std
            output_stats[emg_channel_name] = (avg, lower_bound, upper_bound, num_trials)
        except Exception as e:
            print(f"Stats Helper Error: Calculating stats for {emg_channel_name}: {e}")
            continue # 出錯則跳過此 EMG 頻道
            
    return output_stats


# --- 修改後的獨立繪圖函數 (現在繪製雲圖) ---
def plot_multi_raw_datasets_cloud_comparison( #更改了函數名以反映其繪製雲圖的功能
    raw_datasets_list: List[Dict[str, Dict[str, Dict[str, np.ndarray]]]],
    raw_dataset_labels: List[str],
    directions_to_process: List[str], # 例如 ["left", "right"]
    figure_title: str,
    target_length: int,
    selected_emg_channels: Optional[List[str]] = None,
    max_subplot_cols: int = 2,
    y_axis_label: str = "muscle activation level",
    x_axis_label: str = "Time (%)",
    color_hex_codes: Optional[List[str]] = None
    # show_trendline 參數已移除，因為雲圖的均值線已是主要趨勢
) -> None:
    """
    比較多個 'processed_data_directional' 格式的數據集，以雲圖形式展示。
    對於每個指定的 EMG 頻道，會在一個子圖上繪製所有數據集（按指定方向聚合後）的對應雲圖。
    X 軸範圍固定為 -40 到 100。
    """
    if not raw_datasets_list or not isinstance(raw_datasets_list, list):
        print("Plotting Error: 'raw_datasets_list' must be a non-empty list.")
        return
    if len(raw_datasets_list) != len(raw_dataset_labels):
        print("Plotting Error: Length of 'raw_datasets_list' and 'raw_dataset_labels' must match.")
        return
    if not directions_to_process:
        print("Plotting Error: 'directions_to_process' list cannot be empty.")
        return

    # 1. 準備 "datasets" 以存儲統計數據
    # 結構: List[{"StatsData": {emg_channel: (avg, low, upp, N)}}]
    datasets_with_stats: List[Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]]]] = []
    labels_for_stats_datasets: List[str] = []
    
    for i, raw_data_dict in enumerate(raw_datasets_list):
        raw_label_prefix = raw_dataset_labels[i]
        for direction in directions_to_process:
            data_for_one_direction = raw_data_dict.get(direction)
            if not data_for_one_direction:
                print(f"Plotting Info: Direction '{direction}' not found in dataset '{raw_label_prefix}'. Skipping.")
                continue

            emg_channels_in_this_part = _standalone_get_unique_emg_channels(data_for_one_direction)
            if not emg_channels_in_this_part:
                continue

            stats_for_all_emgs = _standalone_calculate_emg_stats_for_direction(
                data_for_one_direction,
                emg_channels_in_this_part,
                target_length
            )

            if stats_for_all_emgs:
                datasets_with_stats.append({"StatsData": stats_for_all_emgs})
                # labels_for_stats_datasets.append(f"{raw_label_prefix} - {direction.capitalize()}")
                labels_for_stats_datasets.append(f"{raw_label_prefix}")
    if not datasets_with_stats:
        print("Plotting Error: No data available for plotting after statistical aggregation.")
        return

    # 2. 確定要為哪些 EMG 頻道創建子圖
    all_present_emg_in_stats: Set[str] = set()
    for ds_with_stats in datasets_with_stats:
        all_present_emg_in_stats.update(ds_with_stats.get("StatsData", {}).keys())
    
    if not all_present_emg_in_stats:
        print("Plotting Error: No EMG channels found in any aggregated statistical datasets.")
        return

    channels_for_subplots: List[str]
    if selected_emg_channels:
        channels_for_subplots = [ch for ch in selected_emg_channels if ch in all_present_emg_in_stats]
        if not channels_for_subplots:
            print(f"Plotting Warning: None of selected channels {selected_emg_channels} have stats data. Plotting all.")
            channels_for_subplots = sorted(list(all_present_emg_in_stats))
    else:
        channels_for_subplots = sorted(list(all_present_emg_in_stats))

    if not channels_for_subplots:
        print("Plotting Error: No EMG channels to create subplots for.")
        return

    # 3. 繪圖佈局和繪製
    num_subplots = len(channels_for_subplots)
    cols = min(max_subplot_cols, num_subplots)
    rows = math.ceil(num_subplots / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4.1), dpi=150,
                            squeeze=False, sharex=True) # 增加高度
    # 🔳 在 figure 上加一個淡灰色外框
    rect = FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,  # (x, y, width, height) in figure coordinates
        boxstyle="round,pad=0.01",  # 可改成 "square" 若不想要圓角
        edgecolor="#e5e5e5", facecolor="none", linewidth=2,
        transform=fig.transFigure, clip_on=False
    )
    fig.patches.append(rect)
    time_axis = np.linspace(-40, 100, target_length)
    palette = plt.get_cmap('tab10')

    for i_subplot, emg_channel_name in enumerate(channels_for_subplots):
        ax = axs[i_subplot // cols, i_subplot % cols]
        ax.set_title(emg_channel_name, fontsize=14)
        plotted_anything_on_ax = False

        for dataset_idx, dataset_dict_with_stats in enumerate(datasets_with_stats):
            stats_map = dataset_dict_with_stats.get("StatsData", {})
            if emg_channel_name in stats_map:
                mean_signal, lower_bound, upper_bound, num_trials = stats_map[emg_channel_name]
                
                # 確保所有統計數據都是有效的 NumPy 陣列且長度正確
                if not (isinstance(mean_signal, np.ndarray) and mean_signal.size == target_length and
                        isinstance(lower_bound, np.ndarray) and lower_bound.size == target_length and
                        isinstance(upper_bound, np.ndarray) and upper_bound.size == target_length):
                    print(f"Plotting Warning: Invalid stats data for {emg_channel_name} in dataset {labels_for_stats_datasets[dataset_idx]}. Skipping.")
                    continue

                plotted_anything_on_ax = True
              
                if color_hex_codes and dataset_idx < len(color_hex_codes):
                   color = color_hex_codes[dataset_idx]
                else:
                   color = palette(dataset_idx % palette.N)
                   
                current_label = labels_for_stats_datasets[dataset_idx]
                
                # ax.plot(time_axis, mean_signal, color=color, label=f'{current_label} (n={num_trials})', linewidth=1.5)
                ax.plot(time_axis, mean_signal, color=color, linewidth=1.5)
                ax.fill_between(time_axis, lower_bound, upper_bound, color=color, alpha=0.15)
            
        if plotted_anything_on_ax:
            # ax.legend(fontsize=9, loc='best')
            # ax.grid(True, linestyle='-', alpha=0.5)
            # 橫線（畫在上下子圖之間）
            # 畫十字線分隔四張子圖
            # 橫線：Y = 中間，X 從邊界 0.01 開始到 0.99
            hline = Line2D([0.0, 1], [0.5, 0.5], transform=fig.transFigure,
                           color='#e5e5e5', linewidth=2, linestyle='-')
            
            # 直線：X = 中間，Y 從邊界 0.01 到 0.99
            vline = Line2D([0.5, 0.5], [0.00, 1], transform=fig.transFigure,
                           color='#e5e5e5', linewidth=2, linestyle='-')
            
            fig.add_artist(hline)
            fig.add_artist(vline)
            
            ax.grid(True, linestyle=(0, (10, 5)), alpha=0.5, linewidth=0.5)
            ax.set_xlim(-40, 100)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='y', labelsize=10)

            is_bottom_row = (i_subplot // cols) == rows - 1
            is_left_col = (i_subplot % cols) == 0
            ax.yaxis.tick_right()                  # 把刻度值也放右邊
            # ax.set_ylim(0, 110)              # 設定 Y 軸範圍
            ax.set_ylim(0, 105)  # 可視範圍
            ax.set_yticks(np.arange(20, 101, 20))  # 顯示 20~100，但不含 0、110
            # ax.set_yticks(np.arange(0, 101, 20))  # 設定 Y 軸刻度間距
            # 自訂 Y 軸刻度，去掉 0
            # ticks = [tick for tick in ax.get_yticks() if tick != 0]
            
            ticks = ax.get_yticks()
            labels = ["" if t == 0 else str(int(t)) for t in ticks]
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            # ax.set_yticks(ticks)
            # 移除上框線與右框線
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(2)         # 下方邊框加粗
            ax.spines['left'].set_linewidth(2)           # 左側邊框加粗
            
            ax.spines['bottom'].set_color('#959595')     # 下方邊框改為深灰色
            ax.spines['left'].set_color('#959595')     # 下方邊框改為深灰色

            if is_bottom_row:
                ax.set_xlabel(x_axis_label, fontsize=12)
                ax.tick_params(axis='x', labelsize=10, which='both', length=0)
            else:
                ax.tick_params(axis='x', which='both', length=0)
            
            # if is_left_col:
            ax.set_ylabel(y_axis_label, fontsize=16, labelpad=30, rotation=270, color="#868686")
            ax.yaxis.set_label_coords(-0.1, 0.43)  # (x, y) → y=0.0 對齊 X 軸
            ax.tick_params(axis='y', labelsize=10, labelcolor='#e5e5e5', which='both', length=0)
                 # ax.yaxis.set_label_position("right")   # 把標籤放右邊
                 
        else:
            ax.text(0.5, 0.5, "No data for this channel", ha="center", va="center", transform=ax.transAxes, color="grey")
            ax.set_xlim(-40, 100)
            if (i_subplot // cols) == rows - 1: ax.set_xlabel(x_axis_label, fontsize=12)
            if (i_subplot % cols) == 0: ax.set_ylabel(y_axis_label, fontsize=12)

    for i_ax in range(num_subplots, rows * cols):
        fig.delaxes(axs[i_ax // cols, i_ax % cols])

    # fig.suptitle(figure_title, fontsize=18, fontweight='bold', y=0.99 if rows == 1 else 1.00)
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95 if rows > 1 else 0.92])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # hspace 控制上下距離，wspace 控制左右距離

    plt.show()

# --- 示例用法 ---
if __name__ == '__main__':
    # 假設 processed_data_directional_example_v1 和 v2 已定義
    processed_data_directional_example_v1 = {
        "left": { 
            "Group_L1_v1": defaultdict(lambda: np.full(101, np.nan),{
                "Biceps.IM EMG8": np.sin(np.linspace(0, np.pi*2, 101)) * 0.8 + 1.0 + np.random.normal(0, 0.2, 101),
                "Triceps.IM EMG9": np.cos(np.linspace(0, np.pi*2, 101)) * 0.9 + 1.2 + np.random.normal(0, 0.2, 101),
            }),
            "Group_L2_v1": defaultdict(lambda: np.full(101, np.nan),{
                "Biceps.IM EMG8": np.sin(np.linspace(0, np.pi*2, 101) -0.1) * 0.82 + 1.05 + np.random.normal(0, 0.2, 101),
                 "Triceps.IM EMG9": np.cos(np.linspace(0, np.pi*2, 101)-0.1) * 0.92 + 1.25 + np.random.normal(0, 0.2, 101),
            })
        },
        "right": { 
             "Group_R1_v1": defaultdict(lambda: np.full(101, np.nan), {
                "Biceps.IM EMG8": np.sin(np.linspace(0, np.pi*2, 101)) + 1.1 + np.random.normal(0, 0.2, 101),
                "Triceps.IM EMG9": np.cos(np.linspace(0, np.pi*2, 101)) + 1.3 + np.random.normal(0, 0.2, 101),
            })
        }
    }
    processed_data_directional_example_v2 = {
        "left": { 
            "Group_L1_v2": defaultdict(lambda: np.full(101, np.nan),{
                "Biceps.IM EMG8": np.sin(np.linspace(0, np.pi*2, 101)) * 0.7 + 0.9 + np.random.normal(0, 0.25, 101),
                "Triceps.IM EMG9": np.cos(np.linspace(0, np.pi*2, 101)) * 0.8 + 1.1 + np.random.normal(0, 0.25, 101),
            })
        },
        "right": { 
             "Group_R1_v2": defaultdict(lambda: np.full(101, np.nan), {
                "Biceps.IM EMG8": np.sin(np.linspace(0, np.pi*2, 101) + 0.1) + 1.0 + np.random.normal(0, 0.25, 101),
                "Triceps.IM EMG9": np.cos(np.linspace(0, np.pi*2, 101) - 0.1) + 1.2 + np.random.normal(0, 0.25, 101),
                 "ExtInd.IM EMG6": np.random.rand(101) + 0.5 + np.random.normal(0,0.1,101) # v2 特有的頻道
            })
        }
    }
    target_signal_length = 101
    
    print(f"\n--- Plotting Cloud Comparison of Two Raw Datasets (Left and Right Stats) ---")
    plot_multi_raw_datasets_cloud_comparison( # 使用新的函數名
        raw_datasets_list=[processed_data_directional_example_v1, processed_data_directional_example_v2],
        raw_dataset_labels=["Condition Alpha", "Condition Beta"],
        directions_to_process=["left", "right"],
        figure_title="Cloud Comparison: Alpha vs Beta (Left/Right Stats)",
        target_length=target_signal_length,
        selected_emg_channels=['Biceps.IM EMG8', 'Triceps.IM EMG9', 'ExtInd.IM EMG6'],
    )

    print(f"\n--- Plotting Cloud Comparison (Only Right Stats) ---")
    plot_multi_raw_datasets_cloud_comparison( # 使用新的函數名
        raw_datasets_list=[processed_data_directional_example_v1, processed_data_directional_example_v2],
        raw_dataset_labels=["Set A", "Set B"],
        directions_to_process=["right"], # 只看 Right 的統計數據比較
        figure_title="Cloud Comparison: Set A vs Set B (Right Stats Only)",
        target_length=target_signal_length,
    )
