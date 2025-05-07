import pandas as pd
import numpy as np
from scipy import signal
import math
import ezc3d
import logging # For warnings
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
try:
    import ezc3d
except ImportError:
    ezc3d = None

def load_emg_data(file_path, time_column_csv=None, known_fs_csv=None, c3d_channel_keywords=None):
    """
    加載 CSV 或 C3D 檔案中的 EMG 數據。

    參數:
    file_path (str): 數據檔案的路徑 (支援 .csv 或 .c3d)。
    time_column_csv (str, optional): 對於 CSV 檔案，指定包含時間戳記的欄位名稱。
                                     若為 None，則嘗試自動檢測 'Time', 'time', 'Frame', 'frame'。
    known_fs_csv (float, optional): 對於 CSV 檔案，如果已知原始採樣頻率 (Hz)，可在此指定。
                                   若為 None，則會從時間欄位估算。
    c3d_channel_keywords (list of str, optional): 對於 C3D 檔案，提供一個字串列表。
                                                 程式將選取那些標籤名稱包含列表中任一字串的類比通道。
                                                 如果為 None 或空列表，則會先嘗試查找含 "EMG" 的通道。

    回傳:
    processed_df (pd.DataFrame): 包含 EMG 數據的 DataFrame。第一欄為 'time'，後續欄位為各 EMG 通道。
                                 若加載失敗或格式不支援，則返回 None。
    original_fs (float): 偵測到或提供的原始採樣頻率 (Hz)。若無法確定，則返回 None。
    emg_channel_names (list): EMG 數據欄位的名稱列表。
    data_type (str): "csv" 或 "c3d"，表示數據類型。

    拋出:
    ValueError: 如果檔案格式不支援，或 C3D 檔案加載需要 ezc3d 但未安裝，或找不到任何可處理的通道。
    FileNotFoundError: 如果檔案路徑不存在。
    """
    emg_channel_names = []
    original_fs = None

    if not isinstance(file_path, str):
        raise ValueError("檔案路徑必須是字串。")

    try:
        if file_path.lower().endswith('.csv'):
            data_type = "csv"
            raw_data = pd.read_csv(file_path)
            
            # 尋找時間欄位
            if time_column_csv and time_column_csv in raw_data.columns:
                time_col_name = time_column_csv
            else:
                possible_time_cols = ['Time', 'time', 'Frame', 'frame']
                time_col_name = next((col for col in possible_time_cols if col in raw_data.columns), None)
                if not time_col_name and raw_data.shape[1] > 0 :
                    if pd.api.types.is_numeric_dtype(raw_data.iloc[:, 0]):
                         print(f"警告: CSV檔案中未找到明確的時間欄位名稱 ('Time', 'time', 'Frame', 'frame')，且未指定 time_column_csv。將假設第一欄 '{raw_data.columns[0]}' 為時間欄位。")
                         time_col_name = raw_data.columns[0]
                    else:
                        raise ValueError("CSV檔案中找不到可用的時間欄位，請透過 'time_column_csv' 參數指定。")
                elif not time_col_name and raw_data.shape[1] == 0:
                     raise ValueError("CSV檔案為空或無法識別時間欄位。")

            time_series = raw_data[time_col_name].copy()

            if known_fs_csv:
                original_fs = float(known_fs_csv)
            elif pd.api.types.is_numeric_dtype(time_series) and len(time_series) > 1:
                valid_time_series = time_series.dropna()
                if len(valid_time_series) > 1:
                    mean_diff = np.mean(np.diff(valid_time_series))
                    if mean_diff > 0:
                        original_fs = 1.0 / mean_diff
                    else:
                        print(f"警告: 從CSV時間欄位 '{time_col_name}' 計算得到的平均時間差非正值，無法估算採樣頻率。")
                else:
                    print(f"警告: CSV時間欄位 '{time_col_name}' 的有效數據點不足以估算採樣頻率。")
            else:
                print(f"警告: CSV時間欄位 '{time_col_name}' 非數值或數據點不足，無法估算採樣頻率。")

            emg_columns_data = []
            for col in raw_data.columns:
                if "EMG" in col.upper() and col != time_col_name:
                    emg_channel_names.append(col)
                    emg_columns_data.append(raw_data[col])
            
            if not emg_channel_names:
                print("警告: CSV檔案中未找到欄位名稱包含 'EMG' 的欄位。將嘗試把所有非時間的數值欄位視為 EMG 數據。")
                for col in raw_data.columns:
                    if col != time_col_name and pd.api.types.is_numeric_dtype(raw_data[col]):
                        emg_channel_names.append(col)
                        emg_columns_data.append(raw_data[col])
                if not emg_channel_names:
                    raise ValueError("CSV檔案中未能識別出任何 EMG 數據欄位。")

            processed_df = pd.concat([time_series] + emg_columns_data, axis=1)
            processed_df.columns = ['time'] + emg_channel_names
            processed_df['time'] = pd.to_numeric(processed_df['time'], errors='coerce')

        elif file_path.lower().endswith('.c3d'):
            data_type = "c3d"
            if ezc3d is None:
                raise ImportError("處理 C3D 檔案需要 'ezc3d' 函式庫，但該函式庫未安裝。請執行 'pip install ezc3d' 安裝。")
            
            c3d_data = ezc3d.c3d(file_path)
            
            analog_labels = c3d_data['parameters']['ANALOG']['LABELS']['value']
            selected_channels_tuples = [] # 儲存 (索引, 標籤名稱)

            # 階段 1: 嘗試使用 c3d_channel_keywords
            if c3d_channel_keywords and isinstance(c3d_channel_keywords, list) and len(c3d_channel_keywords) > 0:
                print(f"訊息: 使用提供的 'c3d_channel_keywords' ({c3d_channel_keywords}) 來篩選C3D通道。")
                temp_selected_by_keywords = []
                processed_indices_keywords = set() # 確保每個通道只被添加一次
                for i, label_name in enumerate(analog_labels):
                    if i in processed_indices_keywords:
                        continue
                    for pattern in c3d_channel_keywords:
                        if pattern in label_name: # 檢查模式是否存在於標籤名稱中
                            temp_selected_by_keywords.append((i, label_name))
                            processed_indices_keywords.add(i)
                            break # 此標籤已匹配，移至下一個標籤
                if temp_selected_by_keywords:
                    selected_channels_tuples = temp_selected_by_keywords
            
            # 階段 2: 如果關鍵字未產生結果 (或未提供)，嘗試 "EMG"
            if not selected_channels_tuples:
                print("訊息: 未通過 'c3d_channel_keywords' 找到通道，或該參數未提供/為空。嘗試查找標籤中包含 'EMG' 的類比通道。")
                temp_selected_by_emg = []
                # 此處假設如果 selected_channels_tuples 為空，則之前沒有通道被選中
                for i, label_name in enumerate(analog_labels):
                    if "EMG" in label_name.upper():
                        temp_selected_by_emg.append((i, label_name))
                if temp_selected_by_emg:
                    selected_channels_tuples = temp_selected_by_emg
            
            # 階段 3: 如果仍然沒有通道，嘗試所有類比通道
            if not selected_channels_tuples:
                print("警告: C3D檔案中未找到符合指定關鍵字或 'EMG' 的類比通道。將嘗試加載所有類比通道。")
                if len(analog_labels) > 0:
                    selected_channels_tuples = list(enumerate(analog_labels))
                # 如果 analog_labels 為空, selected_channels_tuples 仍為空
            
            # 階段 4: 最終檢查
            if not selected_channels_tuples:
                raise ValueError("C3D檔案中未找到任何可處理的類比通道數據 (analog_labels 可能為空或無匹配項)。")

            raw_header_index = [item[0] for item in selected_channels_tuples]
            emg_channel_names = [item[1] for item in selected_channels_tuples]
            
            emg_data_array = c3d_data['data']['analogs'][0, raw_header_index, :]
            emg_df_c3d = pd.DataFrame(emg_data_array.T, columns=emg_channel_names)
            
            original_fs = float(c3d_data['parameters']['ANALOG']['RATE']['value'][0])
            num_frames = emg_df_c3d.shape[0]
            time_series = pd.Series(np.linspace(0, (num_frames - 1) / original_fs, num=num_frames), name='time')
            
            processed_df = pd.concat([time_series, emg_df_c3d], axis=1)

        else:
            raise ValueError(f"不支援的檔案格式: {file_path}。請提供 CSV 或 C3D 檔案。")

        if original_fs is None or original_fs <=0:
            print(f"警告: 未能成功確定 '{file_path}' 的有效原始採樣頻率。後續處理可能出錯。")

        return processed_df, original_fs, emg_channel_names, data_type

    except FileNotFoundError:
        raise FileNotFoundError(f"錯誤: 找不到檔案 {file_path}")
    except Exception as e:
        print(f"加載數據時發生錯誤 ({file_path}): {e}")
        raise

# %%
data_path = r"D:\BenQ_Project\01_UR_lab\2025_02 Asymmetry\1.Motion\1.Vicon\S06\250318\S06_GridShot_I_1.c3d"

csv_data_path = r"D:\BenQ_Project\01_UR_lab\2025_02 Asymmetry\3.EMG\S10\S02_LargeFlick_Rep_8.24.csv"
# %%

# %%
# 嘗試導入 ezc3d，如果失敗則在需要時拋出錯誤




# 嘗試導入 ezc3d，如果失敗則在需要時拋出錯誤
try:
    import ezc3d
except ImportError:
    ezc3d = None

# 設定日誌記錄器
logging.basicConfig(level=logging.INFO) # 可以調整為 logging.WARNING 等
logger = logging.getLogger(__name__)

def EMG_Process_Combined(
    raw_data_path,
    target_down_freq=1000,
    bandpass_cutoff=[20, 450],
    envelope_lowpass_freq=6,
    notch_cutoff_list=[[59, 61]],
    time_column_csv=None,
    # known_fs_csv=None,
    c3d_select_keywords=None, # 用於C3D通道選擇的關鍵字列表
    channel_rename_map=None # 用於重命名通道的字典 {'old_name': 'new_name'}
):
    """
    整合的 EMG 信號處理函數。
    執行數據加載、預處理、濾波、包絡提取和降採樣。

    程式邏輯：
    1. 數據加載 (CSV/C3D):
        - 自動檢測或使用指定的時間欄位 (CSV)。
        - 使用關鍵字 (C3D) 或 "EMG" 標識 (CSV/C3D) 選擇 EMG 通道。
        - 估算或使用已知的原始採樣頻率。
        - 可選：重命名通道。
    2. 預處理：
        - CSV: 計算各通道的實際採樣頻率、有效數據長度、截止時間。
               根據最短的有效截止時間 (`min_stop_time`) 調整數據。
        - C3D: 使用頭部訊息獲取採樣頻率和數據長度。
        - 計算降採樣後的目標數據長度 (`downsample_len`)。
    3. 逐通道濾波與處理 (在原始採樣率下進行，然後降採樣)：
        - 處理 NaN 值 (填充為0並警告)。
        - CSV: 根據 `min_stop_time` 截斷各通道數據。
        - 應用帶通濾波。
        - (儲存一份僅帶通濾波的結果，用於後續降採樣)
        - 應用陷波濾波。
        - 取絕對值。
        - 應用低通濾波創建包絡。
    4. 降採樣：
        - 將處理後的包絡信號和僅帶通濾波的信號降採樣到 `target_down_freq`。
    5. 插入時間軸並返回結果。

    參數:
    raw_data_path (str): 原始數據檔案路徑 (.csv 或 .c3d)。
    target_down_freq (float): 目標降採樣頻率 (Hz)。預設 1000 Hz。
    bandpass_cutoff (list): 帶通濾波截止頻率 [low, high] (Hz)。預設 [20, 450]。
    envelope_lowpass_freq (float): 包絡提取用的低通濾波截止頻率 (Hz)。預設 6 Hz。
    notch_cutoff_list (list of lists): 陷波濾波頻率列表 [[low1, high1], ...]。預設 [[59, 61]]。
    time_column_csv (str, optional): CSV 檔案的時間欄位名稱。
    known_fs_csv (float, optional): CSV 檔案的已知原始採樣頻率 (Hz)。
    c3d_select_keywords (list of str, optional): C3D 檔案中用於選擇通道的關鍵字列表。
                                                若為 None，則嘗試 "EMG"，再嘗試所有通道。
    channel_rename_map (dict, optional): 用於重命名通道的字典，格式為 {'原始名稱': '新名稱'}。
                                         適用於 CSV 加載後的欄位名或 C3D 的原始標籤名。

    回傳:
    final_envelope_df (pd.DataFrame): 包含時間軸和降採樣後 EMG 包絡的 DataFrame。
    bandpass_only_df (pd.DataFrame): 包含時間軸和降採樣後僅帶通濾波的 EMG 信號的 DataFrame。
                                     如果處理失敗或無有效數據，可能返回 None 或部分空的 DataFrame。
    """
    logger.info(f"開始處理 EMG 數據: {raw_data_path}")
    # --- 1. 數據加載 ---
    # raw_data_path = csv_data_path
    raw_df = None
    original_fs_dict = {} # 對於CSV，可能每個通道Fs不同
    emg_channel_names_loaded = []
    data_type = ""
    time_col_actual_name = 'time' # 預期處理後的內部時間欄位名

    if raw_data_path.lower().endswith('.csv'):
        data_type = "csv"
        try:
            temp_raw_df = pd.read_csv(raw_data_path)
            
            # 確定時間欄位
            # if time_column_csv and time_column_csv in temp_raw_df.columns:
            #     time_col_actual_name = time_column_csv
            # else:
            #     possible_time_cols = ['Time', 'time', 'Frame', 'frame', 'X[s]']
            #     time_col_actual_name = next((col for col in possible_time_cols if col in temp_raw_df.columns), None)
            #     if not time_col_actual_name and temp_raw_df.shape[1] > 0:
            #         if pd.api.types.is_numeric_dtype(temp_raw_df.iloc[:, 0]):
            #             time_col_actual_name = temp_raw_df.columns[0]
            #             logger.warning(f"CSV: 未找到明確時間欄位，假設第一欄 '{time_col_actual_name}' 為時間。")
            #         else:
            #             raise ValueError("CSV: 找不到時間欄位，請用 'time_column_csv' 指定。")
            #     elif not time_col_actual_name:
            #          raise ValueError("CSV: 檔案為空或無法識別時間欄位。")
            
            # 識別 EMG 欄位 (名稱包含 "EMG"，且非時間欄位)
            for col in temp_raw_df.columns:
                if col.upper().startswith("EMG") and col != time_col_actual_name: # 修改為 startswith("EMG") 更靈活
                    emg_channel_names_loaded.append(col)
            
            if not emg_channel_names_loaded: # 如果沒有 "EMG" 開頭的，嘗試包含 "EMG"
                 for col in temp_raw_df.columns:
                    if "EMG" in col.upper() and col != time_col_actual_name:
                        emg_channel_names_loaded.append(col)

            if not emg_channel_names_loaded:
                logger.warning("CSV: 未找到 'EMG' 相關欄位，將嘗試所有非時間的數值欄位。")
                for col in temp_raw_df.columns:
                    if col != time_col_actual_name and pd.api.types.is_numeric_dtype(temp_raw_df[col]):
                        emg_channel_names_loaded.append(col)
                if not emg_channel_names_loaded:
                    raise ValueError("CSV: 未能識別任何 EMG 數據欄位。")
            
            emg_channel_names_loaded_withtime = []
            cols = temp_raw_df.columns  # 原始欄位 Index

            for col in emg_channel_names_loaded:
                idx = cols.get_loc(col)         # 找到 col 在欄位裡的索引
                if idx == 0:
                    prev_name = None           # 第 0 個欄位本身就沒有前一個
                else:
                    prev_name = cols[idx - 1]  # 前一個欄位的字串名稱
                    emg_channel_names_loaded_withtime.append(prev_name)
                    emg_channel_names_loaded_withtime.append(col)

            # 組合 DataFrame，時間欄位統一命名為 'time'
            raw_df_cols = [temp_raw_df[col] for col in emg_channel_names_loaded_withtime]
            raw_df = pd.concat(raw_df_cols, axis=1)

            for col in raw_df.columns:
                raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
            """
            改到這裡
            """

            # 估算或使用已知的採樣頻率 (CSV 可能每個通道不同，但這裡先估算一個整體的，後續可細化)
            # if known_fs_csv:
            #     # 如果提供了 known_fs_csv，假設所有通道都是這個 Fs
            #     for ch_name in emg_channel_names_loaded:
            #         original_fs_dict[ch_name] = float(known_fs_csv)
            # else: # 否則，將在預處理階段為每個通道估算 Fs
            #     pass # Fs 將在下面計算

        except Exception as e:
            logger.error(f"加載 CSV 檔案 '{raw_data_path}' 失敗: {e}")
            return None, None

    elif raw_data_path.lower().endswith('.c3d'):
        data_type = "c3d"
        if ezc3d is None:
            raise ImportError("處理 C3D 檔案需要 'ezc3d' 函式庫。請執行 'pip install ezc3d'。")
        try:
            c3d = ezc3d.c3d(raw_data_path)
            all_analog_labels = c3d['parameters']['ANALOG']['LABELS']
            selected_tuples = [] # (index, original_label)

            if c3d_select_keywords and isinstance(c3d_select_keywords, list) and len(c3d_select_keywords) > 0:
                processed_indices = set()
                for keyword in c3d_select_keywords:
                    for i, label in enumerate(all_analog_labels):
                        if keyword in label and i not in processed_indices:
                            selected_tuples.append((i, label))
                            processed_indices.add(i)
            
            if not selected_tuples: # 未提供關鍵字或未匹配到，嘗試 "EMG"
                logger.info("C3D: 未通過關鍵字選擇通道，嘗試查找含 'EMG' 的通道。")
                for i, label in enumerate(all_analog_labels):
                    if "EMG" in label.upper():
                        selected_tuples.append((i, label))
            
            if not selected_tuples: # 仍未匹配到，嘗試所有通道
                logger.warning("C3D: 未找到 'EMG' 通道，嘗試加載所有類比通道。")
                if len(all_analog_labels) > 0:
                    selected_tuples = list(enumerate(all_analog_labels))
            
            if not selected_tuples:
                raise ValueError("C3D: 未找到任何可處理的類比通道。")

            selected_indices = [item[0] for item in selected_tuples]
            emg_channel_names_loaded = [item[1] for item in selected_tuples]

            analog_data_array = c3d['data']['analogs'][0, selected_indices, :]
            emg_df_c3d = pd.DataFrame(analog_data_array.T, columns=emg_channel_names_loaded)
            
            # C3D 的所有類比通道通常有相同的採樣率
            fs_c3d = float(c3d['parameters']['ANALOG']['RATE']['value'][0])
            for ch_name in emg_channel_names_loaded:
                original_fs_dict[ch_name] = fs_c3d
            
            num_frames = emg_df_c3d.shape[0]
            time_series_c3d = pd.Series(np.linspace(0, (num_frames - 1) / fs_c3d, num=num_frames), name='time')
            raw_df = pd.concat([time_series_c3d, emg_df_c3d], axis=1)

        except Exception as e:
            logger.error(f"加載 C3D 檔案 '{raw_data_path}' 失敗: {e}")
            return None, None
    else:
        raise ValueError(f"不支援的檔案格式: {raw_data_path}。請提供 CSV 或 C3D。")

    if raw_df is None or raw_df.empty:
        logger.error("數據加載後 DataFrame 為空。")
        return None, None
    # channel_rename_map = csv_recolumns_name
    # 可選：重命名通道
    if channel_rename_map and isinstance(channel_rename_map, dict):
        # 更新 emg_channel_names_loaded 列表以反映重命名
        current_columns = list(raw_df.columns)
        new_emg_channel_names = []
        for old_name in emg_channel_names_loaded:
            new_emg_channel_names.append(channel_rename_map.get(old_name, old_name))
        
        raw_df.rename(columns=channel_rename_map, inplace=True)
        emg_channel_names_loaded = new_emg_channel_names # 更新列表
        logger.info(f"通道已重命名。新 EMG 通道名稱: {emg_channel_names_loaded}")


    logger.info(f"數據加載完成。偵測到 EMG 通道: {emg_channel_names_loaded}")

    # --- 2. 預處理 ---
    
    
    # 確定一個用於計算總體 downsample_len 的 Fs (對於C3D是固定的，對於CSV取最小值或平均值)
    overall_fs_for_downsample_len_calc = None

    if data_type == "csv":
        min_stop_time_csv = float('inf')
        csv_channel_data_lengths = {} # 儲存CSV各通道的原始有效長度
        csv_channel_stop_times = {}   # 儲存CSV各通道的截止時間
        
        temp_fs_values = []
        for ch_idx, emg_col_name in enumerate(emg_channel_names_loaded):
            # CSV: 計算每個通道的 Fs (如果 known_fs_csv 未提供)
            # if not known_fs_csv:
                # 假設時間欄位是 'time'，EMG數據欄位是 emg_col_name
                # 原始代碼中，時間欄位是 EMG 欄位索引 - 1。這裡我們有統一的 'time' 欄。
            idx = raw_df.columns.get_loc(emg_col_name) - 1
            time_data_for_fs = raw_df.iloc[:, idx].dropna()
            if len(time_data_for_fs) > 10 : # 需要足夠點來估算
                # 使用前10個差異的平均值
                fs_est = 1.0 / np.mean(np.diff(time_data_for_fs.iloc[1:11])) # 從1開始避免0索引
                if fs_est > 0:
                    original_fs_dict[emg_col_name] = fs_est
                    temp_fs_values.append(fs_est)
                else:
                    logger.warning(f"CSV: 通道 {emg_col_name} Fs 估算失敗 (時間差非正)，將嘗試使用其他通道的Fs。")
            else: # 數據點不足
                logger.warning(f"CSV: 通道 {emg_col_name} 時間數據不足以估算 Fs。")


            # 計算有效數據長度和截止時間 (僅針對CSV的尾部0值處理)
            channel_data_series = raw_df[emg_col_name]
            non_zero_indices = np.where(channel_data_series.fillna(0).values != 0)[0] # fillna(0) 以處理 NaN
            if len(non_zero_indices) > 0:
                valid_len = non_zero_indices[-1] + 1
                csv_channel_data_lengths[emg_col_name] = valid_len
                if valid_len > 0:
                    stop_time_val = raw_df.iloc[:, idx].iloc[valid_len - 1]
                    if pd.notna(stop_time_val):
                        csv_channel_stop_times[emg_col_name] = stop_time_val
                        min_stop_time_csv = min(min_stop_time_csv, stop_time_val)
            else: # 通道全為0或NaN
                csv_channel_data_lengths[emg_col_name] = 0
                logger.warning(f"CSV: 通道 {emg_col_name} 無有效數據 (全為0或NaN)。")
        
        if not temp_fs_values: # 如果所有通道Fs估算失敗且無已知Fs
            raise ValueError("CSV: 無法確定任何通道的採樣頻率。")
        elif not temp_fs_values: # 使用估算Fs的最小值
            overall_fs_for_downsample_len_calc = min(temp_fs_values)
            # 將估算失敗的通道的Fs也設為這個最小值
            for ch_name in emg_channel_names_loaded:
                if ch_name not in original_fs_dict or original_fs_dict[ch_name] <=0:
                    original_fs_dict[ch_name] = overall_fs_for_downsample_len_calc
                    logger.info(f"CSV: 通道 {ch_name} Fs 設為估算的最小 Fs: {overall_fs_for_downsample_len_calc:.2f} Hz")
        
        
        if min_stop_time_csv == float('inf'): # 如果沒有任何有效的 stop time
            if raw_df.shape[0] > 0:
                min_stop_time_csv = raw_df.iloc[-1:, 0] # 使用數據的最後時間點
                logger.warning("CSV: 未能確定有效的 min_stop_time，將使用數據的總時長。")
            else:
                logger.error("CSV: 數據為空，無法確定 min_stop_time。")
                return None, None
        logger.info(f"CSV: 所有通道將對齊到最小截止時間: {min_stop_time_csv:.3f} s")

    elif data_type == "c3d":
        # 對於 C3D，所有通道 Fs 相同，且通常無尾部0問題
        if emg_channel_names_loaded: # 確保列表非空
            overall_fs_for_downsample_len_calc = original_fs_dict[emg_channel_names_loaded[0]]
        else:
            logger.error("C3D: 加載後 EMG 通道列表為空。")
            return None, None

    if overall_fs_for_downsample_len_calc is None or overall_fs_for_downsample_len_calc <= 0:
        logger.error(f"無法確定有效的整體採樣頻率 ({overall_fs_for_downsample_len_calc})。")
        return None, None

    # 計算降採樣目標長度
    # 使用 raw_df 的總長度（對於C3D）或與 min_stop_time_csv 對應的長度（對於CSV）
    if data_type == "csv":
        # 找到 min_stop_time_csv 在 'time' 列中的索引，或最接近的索引
        if raw_df.empty or raw_df['time'].empty:
             max_initial_len = 0
        else:
            time_diff = np.abs(raw_df['time'] - min_stop_time_csv)
            if time_diff.empty: # 如果 raw_df['time'] 是空的
                max_initial_len = 0
            else:
                max_initial_len = time_diff.idxmin() + 1 if not time_diff.empty else 0

    else: # C3D
        max_initial_len = raw_df.shape[0]
    
    if max_initial_len == 0:
        logger.warning("用於計算降採樣長度的初始數據長度為0。")
        downsample_len = 0
    else:
        downsample_len = math.floor(max_initial_len / overall_fs_for_downsample_len_calc * target_down_freq)

    if downsample_len <= 0:
        logger.warning(f"計算得到的降採樣目標長度為 {downsample_len}。可能無輸出數據。")
        # 根據需求，這裡可以決定是否繼續或返回
        # return None, None # 如果不希望處理長度為0的情況

    logger.info(f"整體 Fs 用於降採樣長度計算: {overall_fs_for_downsample_len_calc:.2f} Hz. 降採樣目標長度: {downsample_len} 點.")

    # 初始化結果 DataFrame
    # 列名將是 emg_channel_names_loaded
    final_envelope_list = []
    bandpass_only_list = []

    # --- 3. 逐通道濾波與處理 ---
    for emg_col in emg_channel_names_loaded:
        channel_fs = original_fs_dict.get(emg_col)
        if channel_fs is None or channel_fs <= 0:
            logger.warning(f"通道 {emg_col} 的採樣頻率無效 ({channel_fs})，跳過此通道。")
            if downsample_len > 0: # 保持 DataFrame 結構一致性
                 final_envelope_list.append(pd.Series(np.zeros(downsample_len), name=emg_col))
                 bandpass_only_list.append(pd.Series(np.zeros(downsample_len), name=emg_col))
            else: # 如果 downsample_len 也是0
                 final_envelope_list.append(pd.Series(dtype=float, name=emg_col))
                 bandpass_only_list.append(pd.Series(dtype=float, name=emg_col))
            continue

        # 獲取原始數據
        data_series = raw_df[emg_col].copy()

        # 處理 NaN
        nan_indices = np.where(np.isnan(data_series))[0]
        if len(nan_indices) > 0:
            logger.warning(f"通道 {emg_col}: 發現 {len(nan_indices)} 個 NaN 值，位置: {nan_indices[:5]}... 已用 0 填充。")
            if len(nan_indices) > 0.1 * channel_fs: # 斷訊超過0.1秒
                logger.warning(f"通道 {emg_col}: NaN 值數量超過0.1秒的數據量。")
            data_series.fillna(0, inplace=True)
        
        data_values = data_series.values

        # CSV: 根據 min_stop_time 截斷數據
        if data_type == "csv":
            # 找到 min_stop_time_csv 在該通道時間軸上的索引
            # (假設 raw_df['time'] 是所有通道共用的時間軸)
            if not raw_df['time'].empty:
                end_index_for_channel = (np.abs(raw_df['time'] - min_stop_time_csv)).idxmin()
                # 確保 end_index_for_channel 不超過 data_values 的長度
                end_index_for_channel = min(end_index_for_channel, len(data_values) - 1)
                if end_index_for_channel >= 0 :
                    data_values = data_values[:end_index_for_channel + 1]
                else: # 如果 end_index < 0 (例如時間序列為空或min_stop_time_csv無效)
                    data_values = np.array([]) # 空數據
            else: # 時間序列為空
                data_values = np.array([])


        if len(data_values) == 0:
            logger.warning(f"通道 {emg_col} 在預處理後數據長度為0，跳過濾波。")
            if downsample_len > 0:
                 final_envelope_list.append(pd.Series(np.zeros(downsample_len), name=emg_col))
                 bandpass_only_list.append(pd.Series(np.zeros(downsample_len), name=emg_col))
            else:
                 final_envelope_list.append(pd.Series(dtype=float, name=emg_col))
                 bandpass_only_list.append(pd.Series(dtype=float, name=emg_col))
            continue
            
        # 3.1 帶通濾波
        bp_sos = signal.butter(2, bandpass_cutoff, btype='bandpass', fs=channel_fs, output='sos')
        signal_after_bandpass = signal.sosfiltfilt(bp_sos, data_values)
        
        # 儲存僅帶通濾波的結果 (用於後續降採樣)
        resampled_bandpass_only_signal = np.array([])
        if downsample_len > 0 and len(signal_after_bandpass) > 0:
            resampled_bandpass_only_signal = signal.resample(signal_after_bandpass, downsample_len)
        elif downsample_len > 0 and len(signal_after_bandpass) == 0: # 輸入為空，輸出補零
            resampled_bandpass_only_signal = np.zeros(downsample_len)
        bandpass_only_list.append(pd.Series(resampled_bandpass_only_signal, name=emg_col))

        # 3.2 陷波濾波 (在帶通濾波後的信號上進行)
        signal_after_notch = signal_after_bandpass
        if notch_cutoff_list:
            for notch_range in notch_cutoff_list:
                if not (isinstance(notch_range, list) and len(notch_range) == 2):
                    logger.warning(f"陷波濾波範圍 {notch_range} 格式不正確，已跳過。")
                    continue
                try:
                    notch_sos = signal.butter(2, notch_range, btype='bandstop', fs=channel_fs, output='sos')
                    signal_after_notch = signal.sosfiltfilt(notch_sos, signal_after_notch)
                except ValueError as ve: # 例如截止頻率超出奈奎斯特頻率
                    logger.warning(f"通道 {emg_col}: 應用陷波濾波 {notch_range} 失敗: {ve} (Fs={channel_fs}). 跳過此陷波。")


        # 3.3 取絕對值
        signal_abs = np.abs(signal_after_notch)

        # 3.4 低通濾波創建包絡
        env_lp_sos = signal.butter(2, envelope_lowpass_freq, btype='low', fs=channel_fs, output='sos')
        envelope_signal = signal.sosfiltfilt(env_lp_sos, signal_abs)

        # --- 4. 降採樣包絡 ---
        resampled_envelope = np.array([])
        if downsample_len > 0 and len(envelope_signal) > 0:
            resampled_envelope = signal.resample(envelope_signal, downsample_len)
        elif downsample_len > 0 and len(envelope_signal) == 0:
            resampled_envelope = np.zeros(downsample_len)
        final_envelope_list.append(pd.Series(resampled_envelope, name=emg_col))

    # --- 5. 組合結果並插入時間軸 ---
    final_envelope_df = pd.DataFrame()
    bandpass_only_df = pd.DataFrame()

    if final_envelope_list:
        # 檢查是否所有 Series 都為空或長度不一致 (理論上應與 downsample_len 一致)
        # 這裡假設如果 downsample_len > 0，則列表中的 Series 長度都應為 downsample_len
        # 如果 downsample_len = 0，則列表中的 Series 都應為空
        final_envelope_df = pd.concat(final_envelope_list, axis=1)
    else: # 如果 emg_channel_names_loaded 為空或所有通道處理失敗
        logger.warning("沒有 EMG 通道數據被處理成包絡。")
        # 創建一個空的 DataFrame 但保留欄位名，如果 downsample_len > 0
        if downsample_len > 0 and emg_channel_names_loaded:
            final_envelope_df = pd.DataFrame(np.zeros((downsample_len, len(emg_channel_names_loaded))), columns=emg_channel_names_loaded)
        else: # 否則完全空
            final_envelope_df = pd.DataFrame(columns=emg_channel_names_loaded)


    if bandpass_only_list:
        bandpass_only_df = pd.concat(bandpass_only_list, axis=1)
    else:
        logger.warning("沒有 EMG 通道數據被處理成僅帶通濾波。")
        if downsample_len > 0 and emg_channel_names_loaded:
             bandpass_only_df = pd.DataFrame(np.zeros((downsample_len, len(emg_channel_names_loaded))), columns=emg_channel_names_loaded)
        else:
             bandpass_only_df = pd.DataFrame(columns=emg_channel_names_loaded)


    # 創建新的時間軸
    new_time_axis_values = np.array([])
    if downsample_len > 0 and target_down_freq > 0:
        new_time_axis_values = np.linspace(0, (downsample_len - 1) / target_down_freq, num=downsample_len)
    
    time_series_for_output = pd.Series(new_time_axis_values, name='time')

    # 將時間軸插入到結果 DataFrame 的第一列
    # 確保即使 EMG 數據為空，時間軸也能正確插入
    if not final_envelope_df.empty:
        final_envelope_df.insert(0, 'time', time_series_for_output)
    elif len(time_series_for_output) > 0 : # 如果 EMG 數據為空但有時間軸
        final_envelope_df = pd.DataFrame({'time': time_series_for_output})
        for col_name in emg_channel_names_loaded: # 添加空的 EMG 列
            final_envelope_df[col_name] = pd.Series(dtype=float)


    if not bandpass_only_df.empty:
        bandpass_only_df.insert(0, 'time', time_series_for_output)
    elif len(time_series_for_output) > 0:
        bandpass_only_df = pd.DataFrame({'time': time_series_for_output})
        for col_name in emg_channel_names_loaded:
            bandpass_only_df[col_name] = pd.Series(dtype=float)


    logger.info("EMG 核心處理完成。")
    return final_envelope_df, bandpass_only_df

# %%
final_envelope_df, bandpass_only_df = EMG_Process_Combined(csv_data_path,
                                                           target_down_freq=1000)
# %%
# --- 使用範例 ---
if __name__ == '__main__':
    # 建立一個假的 CSV 檔案來測試
    dummy_csv_data = {
        'Timestamp': np.linspace(0, 5, 10000), # 2kHz 採樣率，5秒數據
        'EMG_Signal_A': np.random.rand(10000) - 0.5 + np.sin(np.linspace(0, 10, 10000) * 2 * np.pi * 1) * 0.2,
        'EMG_B': np.random.rand(10000) - 0.5 + np.cos(np.linspace(0, 10, 10000) * 2 * np.pi * 2) * 0.3,
        'AuxInput': np.random.rand(10000) 
    }
    # 在 EMG_B 的末尾加入一些0，模擬無效數據
    dummy_csv_data['EMG_B'][-500:] = 0 
    dummy_df = pd.DataFrame(dummy_csv_data)
    dummy_csv_path = 'dummy_emg_combined_sample.csv'
    dummy_df.to_csv(dummy_csv_path, index=False)
    
    logger.info("\n--- 測試 CSV 檔案處理 (EMG_Process_Combined) ---")
    
    # 測試1: 基本CSV處理
    envelope_data_csv, bp_data_csv = EMG_Process_Combined(
        raw_data_path=dummy_csv_path,
        target_down_freq=500,
        bandpass_cutoff=[30, 400],
        envelope_lowpass_freq=8,
        notch_cutoff_list=[[58, 62], [118, 122]],
        time_column_csv='Timestamp',
        # known_fs_csv=2000 # 可以選擇提供或讓程式估算
        channel_rename_map={'EMG_Signal_A': 'Vastus_R', 'EMG_B': 'Tibialis_L'}
    )
    if envelope_data_csv is not None:
        logger.info("CSV 處理 - 包絡數據 (前5行):")
        print(envelope_data_csv.head())
    if bp_data_csv is not None:
        logger.info("CSV 處理 - 僅帶通數據 (前5行):")
        print(bp_data_csv.head())

    # 清理假檔案
    import os
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)

    # C3D 檔案測試需要一個實際的 .c3d 檔案
    # 假設您有一個名為 'sample.c3d' 的檔案，其中包含名為 'EMG_Channel1' 和 'OtherAnalog_Sensor2' 的通道
    # logger.info("\n--- 測試 C3D 檔案處理 (EMG_Process_Combined) ---")
    # c3d_file_path_example = 'your_actual_sample.c3d' # 替換為您的 C3D 檔案路徑
    # if os.path.exists(c3d_file_path_example) and ezc3d is not None:
    #     envelope_data_c3d, bp_data_c3d = EMG_Process_Combined(
    #         raw_data_path=c3d_file_path_example,
    #         target_down_freq=1000,
    #         bandpass_cutoff=[20, 450],
    #         envelope_lowpass_freq=6,
    #         notch_cutoff_list=[[49, 51]], # 假設 50Hz 市電
    #         c3d_select_keywords=["EMG", "Sensor2"], # 選擇包含 "EMG" 或 "Sensor2" 的通道
    #         channel_rename_map={'EMG_Channel1': 'Muscle_X', 'OtherAnalog_Sensor2': 'Aux_Y'}
    #     )
    #     if envelope_data_c3d is not None:
    #         logger.info("C3D 處理 - 包絡數據 (前5行):")
    #         print(envelope_data_c3d.head())
    #     if bp_data_c3d is not None:
    #         logger.info("C3D 處理 - 僅帶通數據 (前5行):")
    #         print(bp_data_c3d.head())
    # elif ezc3d is None:
    #    logger.warning("未安裝 ezc3d，跳過 C3D 檔案測試。")
    # else:
    #    logger.warning(f"C3D 測試檔案 '{c3d_file_path_example}' 未找到，跳過 C3D 測試。")




