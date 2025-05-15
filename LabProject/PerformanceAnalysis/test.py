
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
 #%%
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq # 使用 scipy.fft
import ezc3d
import math
import logging
import io
import os # 用於路徑操作
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math

# (如果後端環境不確定是否有顯示，可以加上下面這行，但如果完全不繪圖則非必須)
# import matplotlib
# matplotlib.use('Agg')

# --- 日誌設定 ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 假設日誌已在 Flask app 層級設定


APP_CONFIG = {
    "DEFAULT_DOWNSAMPLE_FREQ": 1000,
    "DEFAULT_BANDPASS_CUTOFF": [20, 450],
    "DEFAULT_LOWPASS_FREQ": 6,
    "DEFAULT_CSV_NOTCH_CUTOFF_LIST": notch_cutoff, # 假設 50Hz 工頻
    "DEFAULT_C3D_NOTCH_CUTOFF_LIST": c3d_notch_cutoff, # 假設 60Hz 工頻
    "DEFAULT_CSV_RECOLUMNS_NAME": csv_recolumns_name, # 範例
    "DEFAULT_C3D_RECOLUMNS_NAME": c3d_recolumns_name, # 範例
    "EMG_CHANNEL_IDENTIFIER": muscle_name, # 用於辨識 EMG 頻道的關鍵字
    "DURATION": 1,
}



# %%

import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq # 使用 scipy.fft
import ezc3d
import math
import logging
import io
import os # 用於路徑操作
from collections import defaultdict
# raw_data_object = r"D:\Hsin\BenQ\testfile\S02_LargeFlick_Rep_9.25.csv"
# raw_data_object = r"D:\Hsin\BenQ\testfile\S06_SpiderShot_S1_3.c3d"
# data_file_path = r"D:\Hsin\BenQ\testfile\S06_SpiderShot_S1_3.c3d"

data_file_path = r"D:/BenQ_Project/01_UR_lab/2024_11 Shanghai CS Major/1. Motion/Major_weight/S06/20241206/S06_SpiderShot_S1_1.c3d"
data_path_2 = r"D:\BenQ_Project\01_UR_lab\2024_11 Shanghai CS Major\1. Motion\Major_weight\S06\20241206\S06_SpiderShot_S2_3.c3d"
data_path_1 = r"D:\BenQ_Project\01_UR_lab\2024_11 Shanghai CS Major\1. Motion\Major_weight\S06\20241206\S06_SpiderShot_S3_1.c3d"
data_file_path = r"D:\test\S21_LargeFlick_Rep_4.150.csv"
data_file_path = r"D:\BenQ_Project\01_UR_lab\2024_11 Shanghai CS Major\1. Motion\Major_weight\S06\20241206\S06_SpiderShot_S3_1.c3d"

config = APP_CONFIG
# %%


# 假設日誌已在應用程式層級設定
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    # ----- 2. 濾波與訊號處理 (逐頻道) -----
    bandpass_cutoff_freqs = config.get("DEFAULT_BANDPASS_CUTOFF")
    
    fft_results = defaultdict(dict)
    fft_results["filename"] = original_filename
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

    return fft_results

if __name__ == '__main__':
    # ----- 測試設定 -----
    # 創建一個假的 config
    APP_CONFIG_TEST = {
        "DEFAULT_CSV_RECOLUMNS_NAME": {
            "EMG1_Raw": "EMG_Sensor1_Renamed", # key 是原始 CSV 中的部分字串
            "EMG2_Raw": "EMG_Sensor2_Renamed",
            "TimeColumn": "Time_Explicit" # 假設時間欄位也被重命名或識別
        },
        "DEFAULT_C3D_RECOLUMNS_NAME": {
            "Analog.EMG.CH1": "EMG_C3D_SensorA", # key 是原始 C3D label 中的部分字串
            "Analog.EMG.CH2": "EMG_C3D_SensorB"
        },
        "EMG_CHANNEL_IDENTIFIER": "EMG_", # 用於在重命名後的欄位中最終篩選EMG頻道
        "BANDPASS_CUTOFF": [20, 250],
        "PERFORM_NOTCH_FILTER": True,
        "DEFAULT_CSV_NOTCH_CUTOFF_LIST": [[48, 52], [98, 102]],
        "DEFAULT_C3D_NOTCH_CUTOFF_LIST": [[58, 62], [118, 122]],
        "FFT_TRUNCATE_TO_POWER_OF_2": False,
        "CSV_TIME_COLUMN_NAME": "Time_Explicit", # 明確指定重命名後的時間欄位名
        "DEFAULT_FS_FALLBACK": 1000
    }
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- 創建假的 CSV 檔案 ---
    dummy_csv_content = """Timestamp,EMG1_Raw_Signal,OtherData,EMG2_Raw_Signal
0.0,0.1,abc,0.5
0.001,0.2,def,0.6
0.002,0.15,ghi,0.55
0.003,0.22,jkl,0.62
0.004,0.18,mno,0.58
0.005,0.25,pqr,0.65
0.006,0.12,stu,0.52
0.007,0.21,vwx,0.61
0.008,0.19,yz,0.59
0.009,0.23,123,0.63
0.010,0.17,456,0.57
"""
    # 替換 Timestamp 為 Time_Explicit 以匹配 config 中的 CSV_TIME_COLUMN_NAME
    dummy_csv_content_timed = dummy_csv_content.replace("Timestamp", APP_CONFIG_TEST["CSV_TIME_COLUMN_NAME"])

    temp_dir = "temp_test_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    dummy_csv_path = os.path.join(temp_dir, "dummy_emg_test.csv")
    with open(dummy_csv_path, "w") as f:
        f.write(dummy_csv_content_timed)
    
    print(f"創建了測試 CSV 檔案: {dummy_csv_path}")

    try:
        print("\n--- 測試 CSV 檔案 ---")
        # 更新 config 中的 DEFAULT_CSV_RECOLUMNS_NAME 的 key 以匹配 dummy_csv_content 中的欄位名
        APP_CONFIG_TEST_CSV = APP_CONFIG_TEST.copy()
        APP_CONFIG_TEST_CSV["DEFAULT_CSV_RECOLUMNS_NAME"] = {
            "EMG1_Raw_Signal": "EMG_Sensor1_Renamed", # 匹配 dummy_csv_content
            "EMG2_Raw_Signal": "EMG_Sensor2_Renamed",
            APP_CONFIG_TEST["CSV_TIME_COLUMN_NAME"]: APP_CONFIG_TEST["CSV_TIME_COLUMN_NAME"] # 確保時間欄位也被正確處理
        }

        results_csv = calculate_fft_for_emg(dummy_csv_path, APP_CONFIG_TEST_CSV, "dummy_emg_test.csv")
        print("CSV 處理結果:")
        # 簡化輸出，只印出頻道名和是否有錯誤
        for ch_data in results_csv.get("channels_fft_data", []):
            print(f"  頻道: {ch_data.get('channel_name')}, 錯誤: {ch_data.get('error', '無')}, Fs: {ch_data.get('sampling_frequency_used')}")
            if not ch_data.get('error'):
                 print(f"    頻率點數: {len(ch_data.get('frequencies',[]))}, 振幅點數: {len(ch_data.get('amplitudes',[]))}")
                 if ch_data.get('top_peaks'):
                     print(f"    第一個峰值頻率: {ch_data['top_peaks'][0]['frequency']:.2f} Hz")


        # 繪製 CSV 結果 (如果需要)
        # from emg_fft_plotter_test import plot_fft_results_for_testing # 假設您將繪圖函式存在此檔案
        # plot_fft_results_for_testing(results_csv)

    except Exception as e:
        print(f"CSV 測試失敗: {e}")
    finally:
        if os.path.exists(dummy_csv_path):
            os.remove(dummy_csv_path)

    # --- C3D 檔案測試需要一個實際的 C3D 檔案 ---
    # 請將 'path/to/your/test.c3d' 替換為一個有效的 C3D 檔案路徑來進行測試
    # example_c3d_path = 'path/to/your/test.c3d'
    # if os.path.exists(example_c3d_path):
    #     try:
    #         print("\n--- 測試 C3D 檔案 ---")
    #         # 確保 APP_CONFIG_TEST 中的 DEFAULT_C3D_RECOLUMNS_NAME 的 keys
    #         # 能夠匹配 example_c3d_path 檔案中的部分 analog label
    #         results_c3d = calculate_fft_for_emg_data_v2(example_c3d_path, APP_CONFIG_TEST)
    #         print("C3D 處理結果:")
    #         for ch_data in results_c3d.get("channels_fft_data", []):
    #             print(f"  頻道: {ch_data.get('channel_name')}, 錯誤: {ch_data.get('error', '無')}, Fs: {ch_data.get('sampling_frequency_used')}")
    #             if not ch_data.get('error'):
    #                  print(f"    頻率點數: {len(ch_data.get('frequencies',[]))}, 振幅點數: {len(ch_data.get('amplitudes',[]))}")

    #         # plot_fft_results_for_testing(results_c3d)
    #     except Exception as e:
    #         print(f"C3D 測試失敗: {e}")
    # else:
    #     print(f"\n未找到 C3D 測試檔案: {example_c3d_path}，跳過 C3D 測試。")

    if os.path.exists(temp_dir) and not os.listdir(temp_dir): # 如果目錄為空則刪除
        os.rmdir(temp_dir)
    elif os.path.exists(temp_dir):
        print(f"測試檔案目錄 {temp_dir} 未被完全清理。")
# %%

def plot_fft_data_from_v2_output(fft_results, max_subplot_cols=2):
    """
    接收來自 calculate_fft_for_emg_data_v2 的結果，並繪製頻譜圖。

    參數:
    - fft_results_data (dict): 包含 FFT 分析結果的字典，
                               其結構應為 calculate_fft_for_emg_data_v2 的輸出。
    - max_subplot_cols (int): 子圖每行最大欄數。
    """
    if not fft_results or "amplitude" not in fft_results or not fft_results["amplitude"]:
        print("沒有有效的頻道數據可以繪製。")
        return

    # channels_data = fft_results["amplitudes"]
    num_channels = len(fft_results["amplitudes"])

    if num_channels == 0:
        print("頻道數據為空，無法繪製。")
        return

    # 計算子圖的行數和列數
    cols = min(max_subplot_cols, num_channels)
    rows = math.ceil(num_channels / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5), squeeze=False)
    # squeeze=False 確保 axs 總是一個二維陣列，即使只有一行或一列

    fig_title = f"FFT Analysis: {fft_results.get('filename', '未知檔案')}"
    # if fft_results.get('c3d_sampling_rate_from_header'):
    #     fig_title += f" (C3D Header Fs: {fft_results['c3d_sampling_rate_from_header']:.0f} Hz)"
    fig.suptitle(fig_title, fontsize=16)

    for i, channel_info in enumerate(fft_results["amplitudes"].keys()):
        print(channel_info)
        
        row_idx = i // cols
        col_idx = i % cols
        ax = axs[row_idx, col_idx]

        # channel_name = channel_info
        frequencies = fft_results.get("frequencies", {}).get(channel_info, None)
        amplitudes = fft_results.get("amplitudes").get(channel_info, None)
        top_peaks = fft_results.get("top_peaks", []).get(channel_info, None)
        # error_msg = channel_info.get("error")
        fs_used = fft_results.get("SamplingRate").get(channel_info, None)

        ax.set_title(f"{channel_info}\n(Fs used: {fs_used if fs_used else 'N/A'} Hz)", fontsize=10)

        # if error_msg:
        #     ax.text(0.5, 0.5, f"錯誤:\n{error_msg}", ha='center', va='center', color='red', fontsize=9, wrap=True)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     continue

        if frequencies is None or amplitudes is None or len(frequencies) == 0 or len(amplitudes) == 0:
            ax.text(0.5, 0.5, "數據不足", ha='center', va='center', color='gray', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # 繪製頻譜
        ax.plot(frequencies, amplitudes, linewidth=0.7, color='dodgerblue')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 3), useMathText=True) # Y軸科學記號

        # 標註峰值
        for peak_idx, peak in enumerate(top_peaks):
            peak_freq = peak.get("frequency")
            peak_amp = peak.get("amplitude")
            if peak_freq is not None and peak_amp is not None:
                ax.plot(peak_freq, peak_amp, 'o', color='red', markersize=4)
                # 稍微錯開標註位置以避免重疊
                offset_y = (peak_idx % 3 -1) * (0.1 * max(amplitudes) if amplitudes else 0.01)
                ax.annotate(f'{peak_freq:.1f}Hz\n{peak_amp:.2e}',
                            xy=(peak_freq, peak_amp),
                            xytext=(5, 5 + offset_y*100), textcoords='offset points', # 調整 xytext
                            fontsize=7, color='crimson',
                            arrowprops=dict(arrowstyle="->", color='gray', connectionstyle="arc3,rad=.2"))


        # 設定 X 軸和 Y 軸標籤
        if row_idx == rows - 1 or (rows > 1 and i // cols == rows -2 and i % cols >= num_channels - cols ):
            ax.set_xlabel("頻率 (Hz)", fontsize=9)
        if col_idx == 0:
            ax.set_ylabel("振幅", fontsize=9)
        
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 調整佈局以容納 suptitle
    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_mdf_over_time(fft_results_data, max_subplot_cols=2):
    """
    接收來自 calculate_fft_for_emg_data_v3 的結果，並繪製 MDF 時程圖。

    參數:
    - fft_results_data (dict): 包含 FFT 分析結果及 MDF 分析的字典。
                               其結構應為 calculate_fft_for_emg_data_v3 的輸出。
    - max_subplot_cols (int): 子圖每行最大欄數。
    """
    if not fft_results_data or "median_frequency_analysis" not in fft_results_data:
        print("沒有有效的 MDF 分析數據可以繪製。")
        return

    mdf_analysis_data = fft_results_data.get("median_frequency_analysis", {})
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

    num_channels_to_plot = len(channels_with_mdf)

    # 計算子圖的行數和列數
    cols = min(max_subplot_cols, num_channels_to_plot)
    rows = math.ceil(num_channels_to_plot / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4), squeeze=False)
    # squeeze=False 確保 axs 總是一個二維陣列

    fig_title = f"Median Frequency (MDF) Over Time: {fft_results_data.get('filename', '未知檔案')}"
    fig.suptitle(fig_title, fontsize=16)

    plot_idx = 0
    for channel_name, mdf_values in channels_with_mdf.items():
        row_idx = plot_idx // cols
        col_idx = plot_idx % cols
        ax = axs[row_idx, col_idx]

        # 嘗試從 channels_fft_data 中獲取該頻道的取樣頻率，以推算時間軸
        # 這部分是可選的，如果沒有，則 x 軸就是窗格索引
        time_axis = np.arange(len(mdf_values)) # 預設為窗格索引
        fs_used = None
        mdf_window_duration = None # 需要從 config 或 fft_results_data 中獲取

        # 嘗試從主 FFT 結果中找到對應頻道的 fs_used (如果有的話)
        # 並假設 MDF 的 config 參數 (如 MDF_WINDOW_DURATION) 也許可以間接得知
        # 這裡簡化，如果需要精確時間軸，MDF 計算時應同時儲存時間點
        for ch_fft_data in fft_results_data.get("channels_fft_data", []):
            if ch_fft_data.get("channel_name") == channel_name:
                fs_used = ch_fft_data.get("sampling_frequency_used")
                # 假設 MDF_WINDOW_DURATION 存在於 config 中，且在生成 fft_results_data 時被使用
                # 這裡我們無法直接訪問 config，所以用一個預設值或讓使用者自行調整
                # 為了更精確的時間軸，`calculate_fft_for_emg_data_v3` 應考慮也輸出每個MDF點對應的時間
                # 這裡我們只用窗格索引作為X軸，或者如果能拿到 window duration，可以計算時間
                break
        
        # 如果能拿到窗格時長，可以計算時間軸 (這裡假設可以從某處獲取，例如 config)
        # mdf_window_duration_from_config = 1.0 # 假設值，應與計算MDF時一致
        # time_axis = np.arange(len(mdf_values)) * mdf_window_duration_from_config


        ax.plot(time_axis, mdf_values, marker='o', linestyle='-', linewidth=1, markersize=3, label="MDF")
        
        title_str = f"{channel_name}"
        if fs_used:
            title_str += f"\n(Fs used: {fs_used:.0f} Hz)"
        ax.set_title(title_str, fontsize=10)
        
        # 顯示平均 MDF (排除 NaN)
        valid_mdf_values = [mdf for mdf in mdf_values if not np.isnan(mdf)]
        if valid_mdf_values:
            mean_mdf = np.mean(valid_mdf_values)
            ax.axhline(mean_mdf, color='red', linestyle='--', linewidth=0.8, label=f'平均 MDF: {mean_mdf:.2f} Hz')
            ax.legend(fontsize=8)

        if row_idx == rows - 1 or (rows > 1 and plot_idx // cols == rows -2 and plot_idx % cols >= num_channels_to_plot - cols ):
            ax.set_xlabel("時間窗格索引", fontsize=9) # 或 "時間 (秒)" 如果有 mdf_window_duration
        if col_idx == 0:
            ax.set_ylabel("中頻數率 (Hz)", fontsize=9)
        
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        plot_idx += 1

    # 如果有未使用的子圖，隱藏它們
    for i in range(plot_idx, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        if row_idx < axs.shape[0] and col_idx < axs.shape[1]:
            fig.delaxes(axs[row_idx, col_idx])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# if __name__ == '__main__':
#     # --- 產生一個模擬的 fft_results_data (包含 MDF) 以供測試 ---
#     def generate_dummy_mdf_channel_data_for_plotter(channel_name, num_windows=20, base_mdf=60, fs=1000):
#         # 模擬 MDF 隨時間有些波動或下降趨勢 (疲勞)
#         mdf_trend = base_mdf - np.linspace(0, 15, num_windows) * (np.random.rand() * 0.5 + 0.5)
#         mdf_noise = (np.random.rand(num_windows) - 0.5) * 8
#         mdf_values = mdf_trend + mdf_noise
#         mdf_values = np.clip(mdf_values, 20, 150) # 限制在合理範圍
#         # 隨機插入一些 NaN
#         if num_windows > 5:
#             nan_indices = np.random.choice(num_windows, size=num_windows // 5, replace=False)
#             mdf_values[nan_indices] = np.nan
        
#         # 同時生成一些假的 channels_fft_data 結構，以便獲取 fs_used
#         dummy_main_fft_data = {
#             "channel_name": channel_name,
#             "sampling_frequency_used": fs,
#             "frequencies": [1.0,2.0], # 簡化
#             "amplitudes": [0.1,0.1], # 簡化
#             "top_peaks": [],
#             "error": None
#         }
#         return mdf_values.tolist(), dummy_main_fft_data

#     mdf_data_for_plot = {}
#     main_fft_data_for_plot = []

#     ch1_mdf, ch1_main_fft = generate_dummy_mdf_channel_data_for_plotter("EMG_TA_R_MDF", num_windows=15, base_mdf=75, fs=2000)
#     mdf_data_for_plot["EMG_TA_R_MDF"] = ch1_mdf
#     main_fft_data_for_plot.append(ch1_main_fft)

#     ch2_mdf, ch2_main_fft = generate_dummy_mdf_channel_data_for_plotter("EMG_GAS_R_MDF", num_windows=25, base_mdf=60, fs=1000)
#     mdf_data_for_plot["EMG_GAS_R_MDF"] = ch2_mdf
#     main_fft_data_for_plot.append(ch2_main_fft)
    
#     ch3_mdf, ch3_main_fft = generate_dummy_mdf_channel_data_for_plotter("EMG_VL_L_MDF_Short", num_windows=5, base_mdf=90, fs=1000)
#     mdf_data_for_plot["EMG_VL_L_MDF_Short"] = ch3_mdf
#     main_fft_data_for_plot.append(ch3_main_fft)

#     mdf_data_for_plot["EMG_NoValidMDF"] = [np.nan, np.nan, np.nan] # 測試全為 NaN 的情況
#     main_fft_data_for_plot.append({ "channel_name": "EMG_NoValidMDF", "sampling_frequency_used": 1000, "error": None})


#     dummy_results_with_mdf = {
#         "filename": "Dummy_MDF_Plot_Test.c3d",
#         "c3d_sampling_rate_from_header": 2000.0, # 假設
#         "channels_fft_data": main_fft_data_for_plot, # 主頻譜數據
#         "median_frequency_analysis": mdf_data_for_plot # MDF 時程數據
#     }

#     print("正在繪製模擬的 MDF 時程圖...")
#     plot_mdf_over_time(dummy_results_with_mdf, max_subplot_cols=2)

#     # 測試單一頻道MDF
#     single_ch_mdf, single_ch_main_fft = generate_dummy_mdf_channel_data_for_plotter("EMG_Biceps_MDF", num_windows=10, base_mdf=70)
#     dummy_single_mdf_results = {
#          "filename": "Single_Channel_MDF.csv",
#          "channels_fft_data": [single_ch_main_fft],
#          "median_frequency_analysis": {
#              "EMG_Biceps_MDF": single_ch_mdf
#          }
#     }
#     print("正在繪製模擬的單一頻道 MDF 時程圖...")
#     plot_mdf_over_time(dummy_single_mdf_results)


