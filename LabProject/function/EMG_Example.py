# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 09:33:09 2025

@author: Hsin.YH.Yang
"""


import sys
import os
# 路徑改成你放自己code的資料夾
sys.path.append(os.path.abspath(r"D:\BenQ_Project\gitgit\Code_testing\LabProject\function"))
import EMG_function as emg
import pandas as pd
import numpy as np
# %%
# parameter setting
smoothing = 'lowpass'
end_name = "_ed"
c = 0.802
lowpass_cutoff = 10/c
duration = 1
start_time = 0
end_time = 20

# %%
emg_path = r"D:/BenQ_Project/S01_HorizontalShot_Rep_2.1.csv"
emg_save_path = r"D:\BenQ_Project\\"
save_name = "test"
MVC_value = pd.read_excel(r"D:\BenQ_Project\S01_all_MVC.xlsx")
MVC_value = MVC_value.iloc[-1, 2:]
 # 前處理EMG data
processing_data, bandpass_filtered_data = emg.EMG_processing(emg_path,
                                                             smoothing=smoothing)
 # 擷取 EMG 資料.
emg_fs = 1 / (processing_data.iloc[1, 0] - processing_data.iloc[0, 0])


# 畫 bandpass 後之資料圖
emg.plot_plot(bandpass_filtered_data, str(emg_save_path),
              save_name, "_Bandpass")
# 畫smoothing 後之資料圖
emg.plot_plot(processing_data, str(emg_save_path),
              save_name, str(smoothing + "_"))
# 畫 FFT analysis 的圖
emg.Fourier_plot(emg_path,
                 (str(emg_save_path)),
                 save_name)
emg.Fourier_plot(emg_path,
                 (str(emg_save_path)),
                 (save_name),
                 notch=True)
# writting data in worksheet
file_name = emg_save_path + save_name + end_name + ".xlsx"
pd.DataFrame(processing_data).to_excel(emg_save_path + save_name + "_lowpass.xlsx",
                                       sheet_name='Sheet1', index=False, header=True)
# 計算 iMVC
emg_iMVC = pd.DataFrame(np.zeros(np.shape(processing_data)),
                        columns=processing_data.columns)
emg_iMVC.iloc[:, 0] = processing_data.iloc[:, 0].values
emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[:, 1:].values),
                                 MVC_value.values)*100
pd.DataFrame(emg_iMVC).to_excel(emg_save_path + save_name + "_iMVC.xlsx",
                                sheet_name='Sheet1', index=False, header=True)
# 進行中頻率分析
med_freq_data, slope_data = emg.median_frquency(emg_path,
                                                duration, emg_save_path, save_name)
# 儲存斜率的資料，並合併成一個資料表
emg_mean = pd.DataFrame(np.zeros([int(np.shape(emg_iMVC)[0]/int(duration * emg_fs)), np.shape(emg_iMVC)[1]]),
                        columns=processing_data.columns)
for col in range(np.shape(emg_iMVC)[1]):
    for row in range(int(np.shape(emg_iMVC)[0]/int(duration * emg_fs))):
        index = int(duration * emg_fs)
        emg_mean.iloc[row, col] = np.mean(
            emg_iMVC.iloc[row*index:(row+1)*index, col])

pd.DataFrame(emg_iMVC).to_excel(emg_save_path + save_name + "_iMVC.xlsx",
                                sheet_name='Sheet1', index=False, header=True)
pd.DataFrame(emg_mean).to_excel(emg_save_path + save_name + "_mvcMean.xlsx",
                                sheet_name='Sheet1', index=False, header=True)

emg_slope = emg.iMVC_plot(emg_mean, save_name, emg_save_path)


