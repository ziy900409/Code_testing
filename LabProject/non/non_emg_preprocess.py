# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:12:31 2024

@author: Hsin.YH.Yang
"""
import pandas as pd
import numpy as np
import time
import os
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"E:\Hsin\git\git\Code_testing\LabProject\function")
# sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\function")
import gen_function as func
import Kinematic_function as kincal
import plotFig_function as FigPlot
import EMG_function as emg

import gc
# %%
folder_path = r"D:\BenQ_Project\01_UR_lab\2024_11 上海Major\Major_Asymmetric\\"
# folder_path = r"D:\BenQ_Project\01_UR_lab\2024_07 non-symmetry\\"
motion_folder = "1.motion\\"
emg_folder = "2.EMG\\"
subfolder = "2.LargeFlick\\"
motion_type = ["Cortex\\", "Vicon\\"]

# cortex_folder = ["S11", "S12", "S13", "S14", "S15",
#                  "S16", "S17", "S18", "S19", 
#                 "S20", "S21"]

vicon_folder = ["S01",
                #"S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09",
                #"S10", "S11", "S12",
                ]


RawData_folder = ""
processingData_folder = "4.process_data\\"
save_place = "4.GridShot\\"

motion_folder_path = folder_path + motion_folder
emg_folder_path = folder_path + emg_folder

# stage_file_path = r"D:\BenQ_Project\01_UR_lab\2024_07 non-symmetry\ZowieNonSymmetry_StagingFile_20240929.xlsx"
stage_file_path = r"E:/Hsin/BenQ/ZOWIE non-sym/ZowieNonSymmetry_StagingFile_20240929.xlsx"
all_mouse_name = ['_EC2_', '_ECN1_', '_ECN2_', '_ECO_', '_HS_']
muscle_name = ['Extensor carpi radialis', 'Flexor Carpi Radialis', 'Triceps',
               'Extensor carpi ulnaris', '1st. dorsal interosseous', 
               'Abductor digiti quinti', 'Extensor Indicis', 'Biceps']

smoothing = "lowpass"
end_name = "_ed"

# %%
# 取得所有 motion data folder list
# 去除有“.“開頭的檔案 and 只獲得資料夾路徑，排除其他可能的檔案格式
motion_folder_list = []
for sub in motion_type:
    motion_folder_list = motion_folder_list + \
        [sub + f for f in os.listdir(motion_folder_path + sub) if not f.startswith('.') \
         and os.path.isdir(os.path.join((motion_folder_path + sub), f))]
# 取得所有 processing data folder list
processing_folder_path = folder_path + "\\" + processingData_folder + "\\"
processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.') \
                          and os.path.isdir(os.path.join(processing_folder_path, f))]

# %% 資料前處理 : bandpass filter, absolute value, smoothing, trunkcut data
# 處理MVC data
tic = time.process_time()
for folder_name in cortex_folder:
    tic = time.process_time()
    MVC_folder_path = folder_path + emg_folder + folder_name
    csv_list = func.Read_File(MVC_folder_path, ".csv")
    MVC_file_list = [file for file in csv_list if 'MVC' in file]
    
    # c3d_file_path = gen.Read_File(MVC_folder_path, ".c3d")
    # MVC_list = csv_MVC_list + c3d_file_path

    fig_save_path = folder_path + processingData_folder + folder_name + "\\2.emg\\2.MVC\\"
    print("Now processing MVC data in " + folder_name)
    for MVC_path in MVC_file_list:
        print(MVC_path)
        # data = pd.read_csv(MVC_path, encoding='UTF-8')
        data_save_path = folder_path + processingData_folder + folder_name + "\\2.emg\\2.MVC\\" 
        # 將檔名拆開 deal with filename and add extension with _ed
        filepath, tempfilename = os.path.split(MVC_path)
        filename, extension = os.path.splitext(tempfilename)
        # 畫圖
        # 前處理EMG data
        processing_data, bandpass_filtered_data = emg.EMG_processing(MVC_path, smoothing="lowpass")

        # 畫 bandpass 後之資料圖
        emg.plot_plot(bandpass_filtered_data, str(fig_save_path),
                     filename, "_Bandpass")
        # 畫smoothing 後之資料圖
        emg.plot_plot(processing_data, str(fig_save_path),
                     filename, str("_" + smoothing))
        # 畫 FFT analysis 的圖
        emg.Fourier_plot(MVC_path,
                        (fig_save_path),
                        filename)
        emg.Fourier_plot(MVC_path,
                        (fig_save_path),
                        filename,
                        notch=True)

        # writting data in worksheet
        file_name = fig_save_path + filename + end_name + ".xlsx"
        pd.DataFrame(processing_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    toc = time.process_time()
    print("Total Time:",toc-tic)  
# 找最大值
for folder_name in cortex_folder:
    print("To find the maximum value of all of MVC data in: " + folder_name)
    tic = time.process_time()
    emg.Find_MVC_max(folder_path + processingData_folder + folder_name + "\\2.emg\\2.MVC\\",
                     folder_path + processingData_folder + folder_name + "\\2.emg\\")
    toc = time.process_time()
    print("Total Time:",toc-tic)
toc = time.process_time()
print("MVC Data Total Time Spent: ",toc-tic)
gc.collect()

# %% for Vicon format

tic = time.process_time()
for folder_name in vicon_folder:
    tic = time.process_time()
    MVC_folder_path = folder_path + motion_folder + motion_type[1] + folder_name
    csv_list = func.Read_File(MVC_folder_path, ".c3d")
    MVC_file_list = [file for file in csv_list if 'MVC' in file]
    
    # c3d_file_path = gen.Read_File(MVC_folder_path, ".c3d")
    # MVC_list = csv_MVC_list + c3d_file_path

    fig_save_path = folder_path + processingData_folder + folder_name + "\\2.emg\\2.MVC\\"
    print("Now processing MVC data in " + folder_name)
    for MVC_path in MVC_file_list:
        print(MVC_path)
        # data = pd.read_csv(MVC_path, encoding='UTF-8')
        data_save_path = folder_path + processingData_folder + folder_name + "\\2.emg\\2.MVC\\" 
        # 將檔名拆開 deal with filename and add extension with _ed
        filepath, tempfilename = os.path.split(MVC_path)
        filename, extension = os.path.splitext(tempfilename)
        # 畫圖
        # 前處理EMG data
        processing_data, bandpass_filtered_data = emg.EMG_processing(MVC_path, smoothing="lowpass")

        # 畫 bandpass 後之資料圖
        emg.plot_plot(bandpass_filtered_data, str(fig_save_path),
                     filename, "_Bandpass")
        # 畫smoothing 後之資料圖
        emg.plot_plot(processing_data, str(fig_save_path),
                     filename, str("_" + smoothing))
        # 畫 FFT analysis 的圖
        emg.Fourier_plot(MVC_path,
                        (fig_save_path),
                        filename)
        emg.Fourier_plot(MVC_path,
                        (fig_save_path),
                        filename,
                        notch=True)

        # writting data in worksheet
        file_name = fig_save_path + filename + end_name + ".xlsx"
        pd.DataFrame(processing_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    toc = time.process_time()
    print("Total Time:",toc-tic)  
# 找最大值
for folder_name in vicon_folder:
    print("To find the maximum value of all of MVC data in: " + folder_name)
    tic = time.process_time()
    emg.Find_MVC_max(folder_path + processingData_folder + folder_name + "\\2.emg\\2.MVC\\",
                     folder_path + processingData_folder + folder_name + "\\2.emg\\")
    toc = time.process_time()
    print("Total Time:",toc-tic)
toc = time.process_time()
print("MVC Data Total Time Spent: ",toc-tic)
gc.collect()






















