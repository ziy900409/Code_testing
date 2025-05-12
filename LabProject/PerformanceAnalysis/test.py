
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
    "EMG_CHANNEL_IDENTIFIER": muscle_name # 用於辨識 EMG 頻道的關鍵字
}

# %%
config = APP_CONFIG
def calculate_fft_for_emg_data(
    raw_data_object,
    input_file_extension,
    config,
    original_filename=None
):
    """
    計算 EMG 數據的 FFT 並回傳頻譜數據。

    參數:
    - raw_data_object: 檔案物件 (如 io.BytesIO) 或 c3d 檔案路徑 (ezc3d 可能需要路徑)。
    - input_file_extension: '.csv' 或 '.c3d'。
    - config: 包含所有處理參數的字典。
    - original_filename: 原始檔案名稱，用於記錄。

    回傳:
    - dict: 包含 FFT 結果的字典。
    """
    raw_data = None
    c3d_instance = None # 用於儲存 c3d 物件以便後續提取 header info

    # ----- 檔案讀取與初步轉換 -----
    if '.csv' in raw_data_object:
        try:
            # 假設 raw_data_object 是一個 file-like object
            raw_data = pd.read_csv(raw_data_object)
            # 使用 config 中的 csv_recolumns_name 來篩選和準備重命名
            # 原碼中的 csv_recolumns_name 是直接使用的，這裡從 config 傳入
            # 注意：原碼中 csv_recolumns_name 的使用方式 (迴圈其鍵) 與其後面的 rename 邏輯可能需要仔細對應
            target_csv_labels_map = config.get("DEFAULT_CSV_RECOLUMNS_NAME", {})
            num_columns_indices = []
            emg_signal_columns = []
            # 方法1: 如果 c3d_recolumns_name 的鍵是 c3d 檔案中「期望找到的部分字串」
            for desired_key_part in target_csv_labels_map.keys():
                for i, actual_label in enumerate(raw_data.columns):
                    if desired_key_part in actual_label:
                        if i not in num_columns_indices: # 避免重複添加
                            num_columns_indices.append(i)
                            emg_signal_columns.append(actual_label) # 原始 c3d 標籤
                        break # 找到一個就跳出內層迴圈 (假設一個 desired_key_part 對應一個頻道)
            if not num_columns_indices:
                raise ValueError("在 C3D 檔案中找不到符合條件的 EMG 頻道。")
            # ----- 1. 前處理 (續) -----
            # # 識別 EMG 頻道 (現在使用 config 中的 IDENTIFIER)
            # emg_cols_bool = raw_data.columns.str.contains(config.get("EMG_CHANNEL_IDENTIFIER", "EMG"))
            # emg_signal_columns = raw_data.columns[emg_cols_bool]
            # num_columns_indices = [raw_data.columns.get_loc(col) for col in emg_signal_columns] # 獲取索引
        except Exception as e:
            logging.error(f"讀取 CSV 檔案時發生錯誤: {e}")
            raise ValueError(f"無法解析 CSV 檔案: {e}")
    elif '.c3d' in raw_data_object:
        try:
            # ezc3d 通常需要檔案路徑，但可以嘗試傳遞 file-like object
            # 如果不行，則需先將上傳的檔案暫存到臨時位置再傳遞路徑
            # 這裡假設 ezc3d 可以處理類似檔案的物件，或已在 API 層處理成路徑
            c = ezc3d.c3d(raw_data_object) # 這行可能需要調整
            
            # --- c3d 轉 DataFrame (與原碼類似，但使用 config 中的參數) ---
            raw_data_header_labels = c['parameters']['ANALOG']['LABELS']['value']
            num_columns_indices = []
            emg_signal_columns = []
            
            # 使用 config 中的 c3d_recolumns_name 來篩選和準備重命名
            # 原碼中的 c3d_recolumns_name 是直接使用的，這裡從 config 傳入
            # 注意：原碼中 c3d_recolumns_name 的使用方式 (迴圈其鍵) 與其後面的 rename 邏輯可能需要仔細對應
            target_c3d_labels_map = config.get("DEFAULT_C3D_RECOLUMNS_NAME", {})
            
            # 方法1: 如果 c3d_recolumns_name 的鍵是 c3d 檔案中「期望找到的部分字串」
            for desired_key_part in target_c3d_labels_map.keys():
                for i, actual_label in enumerate(raw_data_header_labels):
                    if desired_key_part in actual_label:
                        if i not in num_columns_indices: # 避免重複添加
                            num_columns_indices.append(i)
                            emg_signal_columns.append(actual_label) # 原始 c3d 標籤
                        break # 找到一個就跳出內層迴圈 (假設一個 desired_key_part 對應一個頻道)
            if not num_columns_indices:
                raise ValueError("在 C3D 檔案中找不到符合條件的 EMG 頻道。")

            analog_data = c['data']['analogs'][0, num_columns_indices, :]
            raw_data = pd.DataFrame(np.transpose(analog_data), columns=emg_signal_columns)
            
            # 插入時間軸
            analog_time = np.linspace(
                0,
                (c['header']['analogs']['last_frame']) / c['header']['analogs']['frame_rate'],
                num=(np.shape(c['data']['analogs'])[-1])
            )
            raw_data.insert(0, 'Frame', analog_time) # 初始時間欄位

        except Exception as e:
            logging.error(f"處理 C3D 檔案時發生錯誤: {e}")
            raise ValueError(f"無法解析或處理 C3D 檔案: {e}")
    else:
        raise ValueError("不支援的檔案類型。請上傳 .csv 或 .c3d 檔案。")

    if raw_data is None or raw_data.empty:
        raise ValueError("資料讀取失敗或檔案為空。")

   

    if not num_columns_indices:
        logging.error("找不到任何 EMG 訊號欄位。")
        raise ValueError("找不到任何 EMG 訊號欄位。請檢查欄位名稱是否包含指定的識別符。")
    
    logging.info(f"處理 EMG 訊號，總共 {len(num_columns_indices)} 條肌肉，分別為以下欄位: {emg_signal_columns}")
    if not emg_signal_columns:
        logging.error(f"在檔案 '{original_filename}' 中找不到任何 EMG 訊號欄位。")
        raise ValueError(f"在檔案 '{original_filename}' 中找不到任何 EMG 訊號欄位。請檢查欄位名稱或 EMG_CHANNEL_IDENTIFIER。")
    # 欄位重命名 (CSV)
    if '.csv' in raw_data_object:
        raw_data.rename(columns=config.get("DEFAULT_CSV_RECOLUMNS_NAME", {}), inplace=True)
    elif '.c3d' in raw_data_object:
        raw_data.rename(columns=config.get("DEFAULT_C3D_RECOLUMNS_NAME", {}), inplace=True)


    # ----- 準備回傳結果結構 -----
    fft_results = {
        "filename": original_filename,
        "c3d_sampling_rate_from_header": c3d_instance['header']['analogs']['frame_rate'] if c3d_instance else None,
        "channels_fft_data": []
    }

    # ----- 逐頻道處理 -----
    bandpass_cutoff_freqs = config.get("BANDPASS_CUTOFF") # 提供預設值
    perform_notch = config.get("PERFORM_NOTCH_FILTER")
    truncate_fft = config.get("FFT_TRUNCATE_TO_POWER_OF_2") # 默認截斷行為

    for emg_col_name in config["EMG_CHANNEL_IDENTIFIER"]:
        channel_data_results = {
            "channel_name": emg_col_name,
            "sampling_frequency_used": None,
            "frequencies": [],
            "amplitudes": [],
            "top_peaks": []
        }
        
        # --- 計算取樣頻率 (freq) ---
        freq = 0
        if '.csv' in raw_data_object:
            
            data_time_series = raw_data.iloc[:, num_columns_indices-1]
            # 原碼的 num_columns[col]-1 邏輯比較脆弱
            # 如果每個EMG頻道有獨立的時間欄，那結構會更複雜
            data_time_series = raw_data.iloc[:, num_columns_indices-1]
            # data_time_series = raw_data.iloc[:, emg_col_idx].dropna()
            if len(data_time_series) < 11:
                raise ValueError(f"時間欄 '{raw_data.columns[num_columns_indices-1]}' 的數據不足以計算取樣頻率。")
   
            if len(data_time_series) >= 11:
                try:
                    freq = int(1 / np.mean(np.array(data_time_series[2:11]) - np.array(data_time_series[1:10])))
                except ZeroDivisionError:
                    logging.warning(f"頻道 '{emg_col_name}' ({original_filename}) 計算 Fs 時發生除零錯誤 (時間差為0)。")
                    freq = 0 # 或其他處理
            else:
                logging.warning(f"頻道 '{emg_col_name}' ({original_filename}) 時間序列數據不足11點，無法精確計算 Fs。")
                # 可以嘗試使用 c3d_sampling_rate_from_header (如果它是csv但有這個資訊) 或一個預設 Fs
                freq = config.get("DEFAULT_FS_FALLBACK", 2000) # 需要一個預設值
        elif input_file_extension == '.c3d' and c3d_instance:
            freq = c3d_instance['header']['analogs']['frame_rate']
        
        if freq <= 0:
            logging.error(f"頻道 '{emg_col_name}' ({original_filename}) 計算得到的取樣頻率無效 ({freq} Hz)。")
            channel_data_results["error"] = f"Invalid sampling frequency: {freq} Hz."
            fft_results["channels_fft_data"].append(channel_data_results)
            continue
        channel_data_results["sampling_frequency_used"] = freq

        # --- 獲取並處理該頻道數據 ---
        # 計算有效數據長度 (排除末尾的0) - 原碼中的 data_len
        current_emg_series = raw_data[emg_col_name]
        non_zero_indices = (current_emg_series.iloc[::-1] != 0)
        if not non_zero_indices.any():
            effective_data_len = 0
        else:
            effective_data_len = len(current_emg_series) - non_zero_indices.argmax()

        if effective_data_len == 0:
            logging.warning(f"頻道 '{emg_col_name}' ({original_filename}) 沒有有效數據。")
            channel_data_results["error"] = "No effective data in channel."
            fft_results["channels_fft_data"].append(channel_data_results)
            continue

        channel_raw_values = current_emg_series.iloc[:effective_data_len].copy()

        nan_indices = np.where(np.isnan(channel_raw_values))[0]
        if nan_indices.size > 0:
            if nan_indices.size > 0.1 * freq:
                logging.warning(f"頻道 '{emg_col_name}' ({original_filename}) NaN 數據超過0.1秒，已補0。")
            else:
                logging.warning(f"頻道 '{emg_col_name}' ({original_filename}) 發現 {nan_indices.size} 個NaN，已補0。位置: {nan_indices.tolist()}")
            channel_raw_values.fillna(0, inplace=True)
        
        data_for_filter = channel_raw_values.values

        # --- 訊號濾波 ---
        try:
            # Bandpass filter
            bandpass_sos = signal.butter(4, bandpass_cutoff_freqs, btype='bandpass', fs=freq, output='sos')
            filtered_signal = signal.sosfiltfilt(bandpass_sos, data_for_filter)

            # Notch filter (optional)
            if perform_notch:
                notch_freq_list = config.get("CSV_NOTCH_CUTOFF_LIST" if input_file_extension == '.csv' else "C3D_NOTCH_CUTOFF_LIST", [])
                for notch_cutoff in notch_freq_list:
                    if any(f >= freq / 2 for f in notch_cutoff) or any(f <= 0 for f in notch_cutoff) or notch_cutoff[0] >= notch_cutoff[1]:
                        logging.warning(f"頻道 '{emg_col_name}' ({original_filename}) Notch 頻率 {notch_cutoff} 無效 (Fs={freq})，跳過。")
                        continue
                    notch_sos = signal.butter(2, notch_cutoff, btype='bandstop', fs=freq, output='sos')
                    filtered_signal = signal.sosfiltfilt(notch_sos, filtered_signal)
            
            fft_input_data = filtered_signal
        except Exception as filter_err:
            logging.error(f"頻道 '{emg_col_name}' ({original_filename}) 濾波失敗: {filter_err}")
            channel_data_results["error"] = f"Filtering error: {filter_err}"
            fft_results["channels_fft_data"].append(channel_data_results)
            continue

        # --- FFT 計算 ---
        N = len(fft_input_data)
        if N == 0:
            channel_data_results["error"] = "Data length for FFT is zero after filtering."
            fft_results["channels_fft_data"].append(channel_data_results)
            continue

        if truncate_fft:
            N_truncated = 2**(N.bit_length() - 1) if N > 0 else 0
            if N_truncated > 0:
                # 截斷數據 (這會丟失尾部數據)
                fft_input_data = fft_input_data[:N_truncated]
                N = N_truncated
            else: # 如果 N 太小 (e.g. N=1), N_truncated 可能為0
                logging.warning(f"頻道 '{emg_col_name}' ({original_filename}) 數據長度 {len(fft_input_data)} 過短，無法截斷到2的冪次方，將使用原始長度。")
                # 不截斷

        T = 1.0 / freq
        
        # 計算 FFT
        yf = fft(fft_input_data, n=N) # 使用 scipy.fft.fft
        xf = fftfreq(N, T)[:N // 2]   # 使用 scipy.fft.fftfreq，並取正頻率部分
        
        # 計算振幅 (歸一化)
        amplitudes = (2.0 / N) * np.abs(yf[0:N // 2])
        
        channel_data_results["frequencies"] = xf.tolist()
        channel_data_results["amplitudes"] = amplitudes.tolist()

        # --- 找出前三大峰值 ---
        if len(amplitudes) > 0:
            # 複製一份振幅數據用於尋找峰值，避免修改原始數據
            amp_for_peaks = np.copy(amplitudes)
            peaks_found = []
            for _ in range(3): # 尋找三個峰值
                if not np.any(np.isfinite(amp_for_peaks)) or np.max(amp_for_peaks) == float('-inf'):
                    break # 如果沒有有效值或都已標記為-inf
                
                max_idx = np.argmax(amp_for_peaks)
                if max_idx < len(xf): # 確保索引有效
                    peak_freq = xf[max_idx]
                    peak_amp = amplitudes[max_idx] # 從原始amplitudes取值，因為amp_for_peaks會被修改
                    peaks_found.append({"frequency": peak_freq, "amplitude": peak_amp})
                    amp_for_peaks[max_idx] = float('-inf') # 標記已找到的峰值
                else: # 索引超出xf範圍，不太可能發生，但作為保護
                    break
            channel_data_results["top_peaks"] = peaks_found
        
        fft_results["channels_fft_data"].append(channel_data_results)

    return fft_results

# ----- Flask API 路由範例 (與前一個 EMG_processing 類似) -----
# from flask import Flask, request, jsonify
# import os # 如果需要處理暫存檔案
#
# app = Flask(__name__)
#
# # 假設 APP_CONFIG 類似於 EMG_processing 中的定義
# APP_CONFIG_FFT = {
#     "EMG_CHANNEL_IDENTIFIER": "EMG",
#     "CSV_RECOLUMNS_NAME": {"EMG_RAW1": "EMG_Sensor1"},
#     "C3D_RECOLUMNS_NAME": {"AnaEMG1": "EMG_Sensor1", "Right VL": "EMG_RVL"},
#     "BANDPASS_CUTOFF": [20, 450],
#     "PERFORM_NOTCH_FILTER": True,
#     "CSV_NOTCH_CUTOFF_LIST": [[48, 52], [98, 102]], # 50Hz
#     "C3D_NOTCH_CUTOFF_LIST": [[58, 62], [118, 122]], # 60Hz
#     "FFT_TRUNCATE_TO_POWER_OF_2": False, # 示例：不截斷
#     "CSV_TIME_COLUMN_NAME": "Time", # CSV 中時間欄位的名稱
#     "DEFAULT_FS_FALLBACK": 1000 # 當無法計算Fs時的備用值
# }
#
# @app.route('/analyze_emg_fft', methods=['POST'])
# def handle_emg_fft_analysis():
#     if 'file' not in request.files:
#         return jsonify({"error": "缺少檔案部分"}), 400
#     
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "未選擇檔案"}), 400
#
#     original_filename = file.filename
#     file_extension = ""
#     if '.' in original_filename and original_filename.rsplit('.', 1)[1].lower() == 'csv':
#         file_extension = '.csv'
#     elif '.' in original_filename and original_filename.rsplit('.', 1)[1].lower() == 'c3d':
#         file_extension = '.c3d'
#     else:
#         return jsonify({"error": "不支援的檔案類型。請上傳 .csv 或 .c3d 檔案。"}), 400
#
#     try:
#         # 獲取請求中的參數或使用預設值 (這裡簡化，實際中可能需要更複雜的參數解析)
#         processing_config = APP_CONFIG_FFT.copy() # 從預設開始
#         # 可以遍歷 request.form 來更新 processing_config 中的值
#         # 例如: processing_config["PERFORM_NOTCH_FILTER"] = request.form.get("perform_notch", "true").lower() == "true"
#
#         file_object_or_path = None
#         temp_file_to_delete = None
#
#         if file_extension == '.csv':
#             file_object_or_path = io.BytesIO(file.read())
#         elif file_extension == '.c3d':
#             # ezc3d 可能需要檔案路徑
#             import tempfile
#             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.c3d')
#             file.save(temp_file.name)
#             file_object_or_path = temp_file.name
#             temp_file_to_delete = temp_file.name
#
#         fft_data_results = calculate_fft_for_emg_data(
#             file_object_or_path,
#             file_extension,
#             processing_config,
#             original_filename
#         )
#
#         if temp_file_to_delete:
#             try:
#                 os.remove(temp_file_to_delete)
#             except Exception as e_remove:
#                 logging.error(f"刪除暫存檔 {temp_file_to_delete} 失敗: {e_remove}")
#
#         return jsonify({
#             "message": "EMG FFT 分析成功",
#             "data": fft_data_results
#         }), 200
#
#     except ValueError as ve:
#         logging.error(f"FFT 分析請求時發生 Value Error ({original_filename}): {ve}")
#         if temp_file_to_delete and os.path.exists(temp_file_to_delete): # 清理
#             try: os.remove(temp_file_to_delete)
#             except: pass
#         return jsonify({"error": str(ve)}), 400
#     except Exception as e:
#         logging.exception(f"FFT 分析請求時發生未預期錯誤 ({original_filename}): {e}")
#         if temp_file_to_delete and os.path.exists(temp_file_to_delete): # 清理
#             try: os.remove(temp_file_to_delete)
#             except: pass
#         return jsonify({"error": f"內部伺服器錯誤: {e}"}), 500
#
# if __name__ == '__main__':
#     # 為了測試 calculate_fft_for_emg_data，可以模擬檔案和 config
#     # 例如:
#     # config_test = APP_CONFIG_FFT.copy()
#     # # 準備一個模擬的 csv_file_object (io.BytesIO) 或 c3d_file_path
#     # # dummy_csv_content = "Frame,EMG_Sensor1\n0.0,1\n0.001,2\n0.002,3" # ... 更多數據
#     # # csv_file_object = io.BytesIO(dummy_csv_content.encode())
#     # # results = calculate_fft_for_emg_data(csv_file_object, '.csv', config_test, "dummy.csv")
#     # # print(results)
#
#     # 若要執行 Flask app:
#     # logging.basicConfig(level=logging.INFO) # 設定日誌
#     # app.run(debug=True, host='0.0.0.0', port=5001) # 使用不同端口以避免與前一個衝突


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
raw_data_object = r"D:\Hsin\BenQ\testfile\S02_LargeFlick_Rep_9.25.csv"
raw_data_object = r"D:\Hsin\BenQ\testfile\S06_SpiderShot_S1_3.c3d"
data_file_path = r"D:\Hsin\BenQ\testfile\S06_SpiderShot_S1_3.c3d"

data_file_path = r"D:/BenQ_Project/01_UR_lab/2024_11 Shanghai CS Major/1. Motion/Major_weight/S06/20241206/S06_SpiderShot_S1_1.c3d"
data_path_2 = r"D:\BenQ_Project\01_UR_lab\2024_11 Shanghai CS Major\1. Motion\Major_weight\S06\20241206\S06_SpiderShot_S2_3.c3d"
data_path_1 = r"D:\BenQ_Project\01_UR_lab\2024_11 Shanghai CS Major\1. Motion\Major_weight\S06\20241206\S06_SpiderShot_S3_1.c3d"
csv_path = r"D:\test\S21_LargeFlick_Rep_4.150.csv"
# %%


# 假設日誌已在應用程式層級設定
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
config = APP_CONFIG
def calculate_fft_for_emg_data_v2(
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
    initial_selected_indices = [] # 原始 DataFrame 中的索引
    initial_selected_column_names = [] # 原始 DataFrame 中的欄位名

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
            
            initial_selected_indices = num_columns_indices
            initial_selected_column_names = emg_signal_columns
            
            # 根據識別出的原始欄位來決定後續操作的 DataFrame
            # 注意：這裡的重命名邏輯與 EMG_processing 不同，EMG_processing 是先選取再重命名選取後的子集
            # 而使用者提供的 process_emg_core 是先識別，然後對整個 raw_data 進行 rename
            # 這裡我們遵循後者，先識別，然後對整個 raw_data_full_csv 進行 rename
            raw_data = raw_data_full_csv.copy() # 操作副本
            raw_data.rename(columns=csv_channel_map, inplace=True)
            
            # 更新 initial_selected_column_names 為重命名後的名稱 (如果它們被重命名了)
            # 這裡需要一個映射關係，或者直接使用重命名後的 raw_data.columns 進行後續的 EMG 頻道篩選
            # 為了簡化，後續的 EMG 頻道篩選將基於重命名後的 raw_data 和 config 中的 EMG_CHANNEL_IDENTIFIER

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

            initial_selected_indices = num_columns_indices
            # initial_selected_column_names 儲存的是原始 C3D 標籤名
            initial_selected_column_names = emg_signal_columns
            
            analog_data_subset = c3d_instance['data']['analogs'][0, initial_selected_indices, :]
            # 使用原始 C3D 標籤名創建 DataFrame，然後再重命名
            raw_data_from_c3d = pd.DataFrame(np.transpose(analog_data_subset), columns=initial_selected_column_names)
            
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

    # # ----- EMG 頻道最終確定 (基於重命名後的 raw_data) -----
    # # 現在 raw_data 的欄位名已經是 (可能) 重命名後的了
    # # 我們需要根據 config 中的 EMG_CHANNEL_IDENTIFIER 來最終確定哪些欄位是我們要處理的 EMG 頻道
    # emg_identifier_str = config.get("DEFAULT_C3D_RECOLUMNS_NAME", {}) # 通用識別符
    
    # # 從重命名後的 raw_data.columns 中篩選
    # final_emg_column_names =  [col for col in raw_data.columns \
    #                            if any(key in col for key in emg_identifier_str.keys())]
    # # final_emg_column_names = config.get("DEFAULT_CSV_RECOLUMNS_NAME", {})
    
    # if not final_emg_column_names:
    #     logging.error(f"在檔案 '{original_filename}' (重命名後) 中找不到任何包含 '{emg_identifier_str}' 的 EMG 訊號欄位。")
    #     raise ValueError(f"在檔案 '{original_filename}' (重命名後) 中找不到任何包含 '{emg_identifier_str}' 的 EMG 訊號欄位。")
    
    # logging.info(f"檔案 '{original_filename}': 最終處理 {len(final_emg_column_names)} 個 EMG 頻道: {final_emg_column_names}")

    # ----- 準備回傳結果結構 -----
    fft_results = {
        "filename": original_filename,
        "c3d_sampling_rate_from_header": c3d_instance['header']['analogs']['frame_rate'] if c3d_instance else None,
        "channels_fft_data": []
    }

    # ----- 逐頻道處理 -----
    bandpass_cutoff_freqs = config.get("BANDPASS_CUTOFF", [20, 450])
    perform_notch = config.get("PERFORM_NOTCH_FILTER", False)
    truncate_fft = config.get("FFT_TRUNCATE_TO_POWER_OF_2", True)
    csv_time_column_explicit = config.get("CSV_TIME_COLUMN_NAME", None) # 明確的CSV時間欄位名

    down_freq = config.get("DEFAULT_DOWNSAMPLE_FREQ")

    Fs_global = 0
    data_len_global = 0 # 這裡指降採樣前的長度
    min_stop_time_global = 0
    downsample_len_global = 0 # 降採樣後的統一長度

    if '.csv' in raw_data_object:
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
        
        # 原碼中同步化 sensor 時間的邏輯
        # while max(valid_stop_times) - min(valid_stop_times) > 1:
        #     logging.info("兩 sensor 數據時間差超過 1 秒，將使用次短時間的 Sensor 作替代。")
        #     # 這個移除邏輯比較複雜，需要小心處理對應的 Fs, data_len
        #     # 簡化：直接使用 min_stop_time，並在濾波時截斷
        #     break # 暫時跳過複雜的移除邏輯
        
        Fs_global = min(all_fs_csv) if all_fs_csv else 0
        # data_len_global 應該是基於 min_stop_time 和 Fs_global 重新計算，或者取最小的有效長度
        # downsample_len_global 取最小的，並確保是整數
        downsample_len_global = math.floor(min(all_downsample_len_csv)) if all_downsample_len_csv else 0
        min_stop_time_global = min_stop_time_csv

    elif '.c3d' in raw_data_object:
        Fs_global = c['header']['analogs']['frame_rate']
        # data_len_global 是原始 c3d 數據的長度 (影格數)
        data_len_global = np.shape(c['data']['analogs'])[-1] # 或 raw_data.shape[0] 如果 'Frame' 欄已移除
        min_stop_time_global = (c['header']['analogs']['last_frame']) / Fs_global
        downsample_len_global = math.floor(data_len_global / Fs_global * down_freq)

    if Fs_global <= 0 or downsample_len_global <= 0:
        raise ValueError("無法計算有效的取樣頻率或降採樣長度。")

    logging.info(f"全局取樣頻率 (估計/實際): {Fs_global}, 降採樣後長度: {downsample_len_global}, 統一截止時間: {min_stop_time_global}")

    # ----- 初始化結果 DataFrame -----
    # 欄位名稱使用處理後的 EMG 欄位名
    processed_emg_columns = raw_data.columns[num_columns_indices].tolist()

    bandpass_filtered_data_df = pd.DataFrame(np.zeros([downsample_len_global, len(num_columns_indices)]),
                                           columns=processed_emg_columns)
    notch_filtered_data_df = pd.DataFrame(np.zeros([downsample_len_global, len(num_columns_indices)]),
                                         columns=processed_emg_columns)
    
    
    # ----- 2. 濾波與訊號處理 (逐頻道) -----
    bandpass_cutoff_freqs = config.get("DEFAULT_BANDPASS_CUTOFF")

    
    # 這裡的 col 應該是迭代 processed_emg_columns 的索引，或者直接迭代欄位名
    for i, emg_col_name in enumerate(processed_emg_columns):
        emg_col_original_idx = raw_data.columns.get_loc(emg_col_name) # 獲取在 raw_data 中的實際索引
        
        current_sample_freq = 0
        data_to_filter = None

        if '.csv' in raw_data_object:
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

        elif '.c3d' in raw_data_object:
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
            
        fft_input_data = notched_signal.copy()
        

        # --- FFT 計算 ---
        N = len(fft_input_data)
        if N == 0:
            channel_data_results["error"] = "Data length for FFT is zero after filtering."
            fft_results["channels_fft_data"].append(channel_data_results)
            continue

        if truncate_fft:
            N_truncated = 2**(N.bit_length() - 1) if N > 1 else N # 避免 N=1 時 N_truncated=0
            if N_truncated > 0 and N_truncated < N : # 只在有意義截斷時才截斷
                fft_input_data = fft_input_data[:N_truncated]
                N = N_truncated
            elif N_truncated == 0 and N > 0: # N=1 的情況
                 logging.warning(f"頻道 '{emg_col_name_final}' ({original_filename}) 數據長度 {len(fft_input_data)} 過短，無法截斷到2的冪次方，將使用原始長度。")
            # else N_truncated == N, 不用做任何事

        T = 1.0 / freq
        yf = fft(fft_input_data, n=N)
        xf = fftfreq(N, T)[:N // 2]
        amplitudes = (2.0 / N) * np.abs(yf[0:N // 2])
        
        channel_data_results["frequencies"] = xf.tolist()
        channel_data_results["amplitudes"] = amplitudes.tolist()

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
            channel_data_results["top_peaks"] = peaks_found
        
        fft_results["channels_fft_data"].append(channel_data_results)

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

        results_csv = calculate_fft_for_emg_data_v2(dummy_csv_path, APP_CONFIG_TEST_CSV, "dummy_emg_test.csv")
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

