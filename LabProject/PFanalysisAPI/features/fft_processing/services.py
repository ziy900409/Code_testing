
import os
import logging
import pandas as pd
import numpy as np
import ezc3d
import math
from collections import defaultdict
from signal import sosfiltfilt, butter, resample
from scipy.stats import linregress


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