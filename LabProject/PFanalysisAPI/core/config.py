# src/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict # Pydantic V2
# from pydantic import BaseSettings, Field # Pydantic V1 (Field is optional for defaults)
from typing import List, Dict, Optional, Union, Tuple

# --- 滑鼠/遊戲靈敏度設定 ---
class MouseSensitivitySettings(BaseSettings):
    """滑鼠或遊戲內靈敏度相關設定"""
    dpi: int = 800
    sensitivity: float = 1.0
    # yaw: float = 0.022  # CS2 預設值 (有重複定義，取後者或選擇一個)
    yaw_cs2: float = 0.022 # 為 CS2 單獨命名
    yaw_valorant: float = 0.07 # Valorant 靈敏度

# --- 動態捕捉系統設定 ---
class MotionCaptureFilterSettings(BaseSettings):
    """動態捕捉系統中各類數據的濾波截止頻率設定"""
    marker_freq_cutoff: Optional[float] = 20.0
    fp_freq_cutoff: Optional[float] = 30.0
    analog_freq_cutoff: Optional[float] = None # 預設不濾波一般類比訊號

class MotionCaptureSettings(BaseSettings):
    """動態捕捉系統相關設定"""
    filters: MotionCaptureFilterSettings = MotionCaptureFilterSettings()
    rename_markers: Dict[str, str] = {
        'MOS1': 'M1', 'MOS2': 'M2', 'MOS3': 'M3', 'MOS4': 'M4',
        'RHO': 'R.Shoulder', 'RSHO': 'R.Shoulder',
        'RUEL': 'R.Elbow.Lat', 'RUEM': 'R.Elbow.Med',
        'RUS': 'R.Wrist.Uln', 'RRS': 'R.Wrist.Rad',
        'RTB1': 'R.Thumb1', 'RTB2': 'R.Thumb2', 'RTB3': 'R.Thumb3',
        'RID1': 'R.I.Finger1', 'RID2': 'R.I.Finger2', 'RID3': 'R.I.Finger3',
        'RMD1': 'R.M.Finger1', 'RMD2': 'R.M.Finger2', 'RMD3': 'R.M.Finger3',
        'RRG1': 'R.R.Finger1', 'RRG2': 'R.R.Finger2',
        'RLT1': 'R.P.Finger1', 'RLT2': 'R.P.Finger2',
    }
    remove_prefixes: List[str] = ["S03:", "MarkerSet:"] # 注意: "S03" 和 "S03:" 不同
    butterworth_order: int = 4 # Standard 4th order Butterworth

# --- EMG 訊號處理設定 ---
class EMGFilterSettings(BaseSettings):
    """EMG 訊號的濾波參數"""
    # c_factor: float = 0.802 # 如果 c 是常數且固定，可以直接用在計算中或定義在 EMGSettings
    bandpass_cutoff_raw: List[float] = [20, 450] # 原始的截止頻率
    lowpass_freq_raw: float = 10 # 原始的截止頻率
    
    # 可以定義計算後的屬性，但 Pydantic V1/V2 處理方式不同
    # 為了簡單起見，服務層可以直接使用 raw 值和 c_factor 計算
    # 或者，如果 c_factor 也來自設定，則可以更動態
    # bandpass_cutoff: List[float] = [bandpass_cutoff_raw[0] / c_factor, bandpass_cutoff_raw[1] / c_factor]
    # lowpass_freq: float = lowpass_freq_raw / c_factor

    # Notch filter cutoffs - 這些是頻率對 [low, high] 的列表
    # 原始 notch_cutoff 和 c3d_notch_cutoff 命名有些模糊，這裡統一為針對 CSV 和 C3D 的
    csv_notch_cutoff_list: List[List[float]] = [
        [59, 61], [295.5, 296.5], [369.5, 370.5],
        [179, 181], [299, 301], [419, 421],
    ]
    c3d_notch_cutoff_list: List[List[float]] = [
        [49, 51], [99.5, 100.5], [149.5, 150.5], [199.5, 200.5],
        [249.5, 250.5], [299.5, 300.5], [349.5, 350.5],
        [295, 297], [369, 371], [73, 75], [399, 401]
    ]

class EMGProcessingParameters(BaseSettings):
    """EMG 處理的移動窗格等參數"""
    time_of_window_seconds: float = 0.1  # 窗格長度 (秒)
    overlap_percentage: float = 0.5    # 重疊百分比 (0.0 to 1.0)

class EMGChannelSettings(BaseSettings):
    """EMG 頻道的識別與重命名"""
    # 用於在重命名後的欄位中最終篩選EMG頻道
    emg_channel_identifier_keyword: str = "EMG_" # 例如 "EMG_"
    
    # 原始 CSV 欄位名中的識別符 -> 希望重命名成的名稱
    csv_recolumns_name_map: Dict[str, str] = {
        'Mini sensor 1: EMG 1': 'EMG_Extensor_Carpi_Radialis',
        'Mini sensor 2: EMG 2': 'EMG_Flexor_Carpi_Radialis',
        'Mini sensor 3: EMG 3': 'EMG_Triceps_Brachii',
        'Quattro sensor 4: EMG.A 4': 'EMG_Extensor_Carpi_Ulnaris',
        'Quattro sensor 4: EMG.B 4': 'EMG_1st_Dorsal_Interosseous',
        'Quattro sensor 4: EMG.C 4': 'EMG_Abductor_Digiti_Quinti',
        'Quattro sensor 4: EMG.D 4': 'EMG_Extensor_Indicis',
        'Avanti sensor 5: EMG 5': 'EMG_Biceps_Brachii'
    }
    # 原始 C3D Analog Label 中的識別符 -> 希望重命名成的名稱
    c3d_recolumns_name_map: Dict[str, str] = {
        'ExtRad': 'EMG_Extensor_Carpi_Radialis',
        'FleRad': 'EMG_Flexor_Carpi_Radialis',
        'Triceps': 'EMG_Triceps_Brachii', # 注意：原始有重複的 Triceps，字典中 key 必須唯一
        # 'Triceps': 'EMG_Triceps_Brachii_Lateral', # 如果是不同肌肉，需要不同 key
        'ExtUlnar': 'EMG_Extensor_Carpi_Ulnaris',
        'DorInter': 'EMG_1st_Dorsal_Interosseous',
        'AbdDigMin': 'EMG_Abductor_Digiti_Quinti',
        'ExtInd': 'EMG_Extensor_Indicis',
        'Biceps': 'EMG_Biceps_Brachii',
    }
    # (可選) 如果 c3d_analog_cha 和 muscle_name 仍然需要作為獨立的配置項
    # c3d_analog_channel_identifiers: List[str] = ["ExtUlnar", "DorInter", "AbdDigMin", "ExtInd"]
    # target_muscle_names_ordered: List[str] = [
    #     'Extensor Carpi Radialis', 'Flexor Carpi Radialis', 'Triceps Brachii',
    #     'Extensor Carpi Ulnaris', '1st Dorsal Interosseous',
    #     'Abductor Digiti Quinti', 'Extensor Indicis', 'Biceps Brachii'
    # ]

class EMGSettings(BaseSettings):
    """EMG 相關的所有設定"""
    downsample_freq: int = 1000
    # 關於 c=0.802 的因子，最好在服務層處理，或明確其含義
    # 如果 c_factor 是一個可配置項，可以加在這裡
    c_factor_for_filters: float = 0.802 
    filters: EMGFilterSettings = EMGFilterSettings()
    processing_params: EMGProcessingParameters = EMGProcessingParameters()
    channels: EMGChannelSettings = EMGChannelSettings()
    
    # FFT 和 MDF 相關設定 (從 cursor_prompt_emg_fft_api_tw 整合)
    mdf_window_duration_seconds: float = 1.0
    truncate_fft_to_power_of_2: bool = True
    truncate_mdf_segment_fft: bool = True # MDF 窗格的 FFT 是否截斷

# --- 主設定類別 ---
class Settings(BaseSettings):
    """應用程式的總體設定"""
    PROJECT_NAME: str = "EMG_FFT_Analysis_Service"
    PROJECT_VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True # 開發時設為 True

    # CORS 設定
    # 確保 BACKEND_CORS_ORIGINS 是字串列表
    BACKEND_CORS_ORIGINS: Union[str, List[str]] = "http://localhost:8000,http://127.0.0.1:8000"
    
    # Uvicorn 伺服器設定
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000

    # 各功能模組的設定實例
    mouse_sensitivity: MouseSensitivitySettings = MouseSensitivitySettings()
    motion_capture: MotionCaptureSettings = MotionCaptureSettings()
    emg: EMGSettings = EMGSettings()

    # 來自 cursor_prompt_emg_fft_api_tw 的其他設定
    # 這些已經部分整合到 EMGSettings 中，這裡列出以確保覆蓋
    DEFAULT_FS_FALLBACK: int = 1000 # 當無法計算Fs時的備用值
    CSV_TIME_COLUMN_NAME: Optional[str] = "Time" # CSV 中時間欄位的名稱 (重命名後)

    # Pydantic V2 的方式來指定 .env 檔案
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra="ignore")

    # Pydantic V1 的方式
    # class Config:
    #     env_file = ".env"
    #     env_file_encoding = 'utf-8'
    #     extra = "ignore" # 忽略 .env 中多餘的變數

# 創建一個全域可用的設定實例
settings = Settings()

# --- 使用範例 (可以在其他模組中 import settings) ---
if __name__ == "__main__":
    print(f"專案名稱: {settings.PROJECT_NAME}")
    print(f"除錯模式: {settings.DEBUG}")
    print(f"CORS 來源: {settings.BACKEND_CORS_ORIGINS}")
    print(f"滑鼠 DPI: {settings.mouse_sensitivity.dpi}")
    print(f"動捕 Marker 濾波截止頻率: {settings.motion_capture.filters.marker_freq_cutoff}")
    print(f"EMG 降採樣頻率: {settings.emg.downsample_freq}")
    print(f"EMG C Factor: {settings.emg.c_factor_for_filters}")
    # 動態計算 bandpass_cutoff
    bp_low = settings.emg.filters.bandpass_cutoff_raw[0] / settings.emg.c_factor_for_filters
    bp_high = settings.emg.filters.bandpass_cutoff_raw[1] / settings.emg.c_factor_for_filters
    print(f"EMG 計算後帶通濾波: [{bp_low:.2f}, {bp_high:.2f}] Hz")
    print(f"EMG CSV Notch 列表: {settings.emg.filters.csv_notch_cutoff_list}")
    print(f"EMG CSV 重命名映射 (第一個): {list(settings.emg.channels.csv_recolumns_name_map.items())[0]}")
    print(f"EMG 頻道識別關鍵字: {settings.emg.channels.emg_channel_identifier_keyword}")
    print(f"MDF 窗格時長: {settings.emg.mdf_window_duration_seconds} 秒")
    print(f"CSV 時間欄位名: {settings.CSV_TIME_COLUMN_NAME}")
