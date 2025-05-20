import os
import logging
import pandas as pd
import numpy as np
import ezc3d
import math
from collections import defaultdict
from signal import sosfiltfilt, butter, resample
from scipy.stats import linregress


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
            bandpass_sos = butter(2, bandpass_cutoff_freqs, btype='bandpass', fs=current_sample_freq, output='sos')
            bandpassed_signal = sosfiltfilt(bandpass_sos, data_to_filter)
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
                notch_sos = butter(2, notch_cutoff, btype='bandstop', fs=current_sample_freq, output='sos')
                notched_signal = sosfiltfilt(notch_sos, notched_signal)
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
                lowpass_sos = butter(2, lowpass_cutoff_freq, btype='low', fs=current_sample_freq, output='sos')
                lowpassed_signal = sosfiltfilt(lowpass_sos, abs_signal)
        except ValueError as e:
            logging.error(f"頻道 {emg_col_name} Lowpass 濾波失敗: {e}。Fs={current_sample_freq}, Cutoff={lowpass_cutoff_freq}")
            lowpassed_signal = abs_signal # 出錯時使用 abs_signal
        
        # --- 降採樣 ---
        # `downsample_len_global` 是目標長度
        if len(notched_signal) > 0 :
            resampled_notch = resample(notched_signal, downsample_len_global)
            notch_filtered_data_df.iloc[:, i] = resampled_notch[:downsample_len_global]
        else: # 如果原始訊號為空
            notch_filtered_data_df.iloc[:, i] = np.zeros(downsample_len_global)
        
        if len(bandpassed_signal) > 0:
            resampled_bandpass = resample(bandpassed_signal, downsample_len_global)
            bandpass_filtered_data_df.iloc[:, i] = resampled_bandpass[:downsample_len_global]
        else:
            bandpass_filtered_data_df.iloc[:, i] = np.zeros(downsample_len_global)

        if len(lowpassed_signal) > 0:
            resampled_lowpass = resample(lowpassed_signal, downsample_len_global)
            lowpass_filtered_data_df.iloc[:, i] = resampled_lowpass[:downsample_len_global]
            emg_results["Smoothing"][emg_col_name] = resampled_lowpass[:downsample_len_global]
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
            if len(averaged_values_for_channel) == num_averaged_points_global:
                averaged_data_df.iloc[:, i] = averaged_values_for_channel
                emg_results["AverageData"][emg_col_name] = averaged_values_for_channel
                time_axis = np.arange(len(averaged_values_for_channel))
                # 計算趨勢線的斜率
                slope, intercept, r_value, p_value, std_err = linregress(time_axis, averaged_values_for_channel)
            elif len(averaged_values_for_channel) < num_averaged_points_global: # 如果產生值較少
                temp_array = np.full(num_averaged_points_global, np.nan)
                temp_array[:len(averaged_values_for_channel)] = averaged_values_for_channel
                averaged_data_df.iloc[:, i] = temp_array
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