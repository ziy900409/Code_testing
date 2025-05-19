
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



# %%



# %%


# %%
plot_fft_data_output(results_c3d)
# %%
import matplotlib.pyplot as plt
import numpy as np
import math

# %%
plot_mdf_over_time(results_c3d)
fft_results_data = results_c3d
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


