# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:45:15 2024

@author: Hsin.YH.Yang
"""
import ezc3d
import pandas as pd
import numpy as np
import time
import os
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\4. code")
import AllSeries_emg_func_20240327 as emg
import AllSeries_general_func_20240327 as gen
import gc

# %% parameter setting
# path setting
data_path = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\2. EMG\\"
RawData_folder = "raw_data\\"
processingData_folder = "processing_data\\"
static_folder = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\5. Statics\\"
MVC_folder = "MVC\\"
sub_folder = ""
fig_save = "figure\\"
end_name = "_ed"
smoothing = 'lowpass'
# parameter setting
duration = 1
all_mouse_name = ['_Gpro_', '_U_', '_ZA_', '_S_', '_FK_', '_EC_']
muscle_name = ['Extensor carpi radialis', 'Flexor Carpi Radialis', 'Triceps',
               'Extensor carpi ulnaris', '1st. dorsal interosseous', 
               'Abductor digiti quinti', 'Extensor Indicis', 'Biceps']
# %% 欲儲存之資料



# %% 讀取資料夾路徑

rowdata_folder_path = data_path + RawData_folder + "\\" + sub_folder
# 去除有“.“開頭的檔案
rowdata_folder_list  = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.')]
processing_folder_path = data_path + "\\" + processingData_folder + "\\" + sub_folder
processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.')]
# read staging file
staging_file_path = r"D:\BenQ_Project\01_UR_lab\09_ZowieAllSeries\ZowieAllSeries_StagingFile_20240326.xlsx"
# %% 分析疲勞
tic = time.process_time()
# 建立 slope data 要儲存之位置
all_slope_data = pd.DataFrame({}, columns = ['data_name', 'mouse']+ muscle_name)
for i in range(len(rowdata_folder_list)):
    print(0)
    csv_file_path = gen.Read_File(data_path + RawData_folder +  rowdata_folder_list[i],
                               ".csv")
    c3d_file_path = gen.Read_File(data_path + RawData_folder +  rowdata_folder_list[i],
                               ".c3d")
    all_file_path = csv_file_path + c3d_file_path
    grid_shot_file_list = [file for file in all_file_path if 'GridShot' in file]
    # 讀取all MVC data
    MVC_value = pd.read_excel(processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + rowdata_folder_list[i] + '_all_MVC.xlsx')
    MVC_value = MVC_value.iloc[-1, 2:]
    for ii in range(len(grid_shot_file_list)):
        
        save_name, extension = os.path.splitext(grid_shot_file_list[ii].split('\\', -1)[-1])
        fig_svae_path = data_path + processingData_folder + rowdata_folder_list[i] + "\\" + \
            "FatigueAnalysis\\figure\\"
        data_svae_path = data_path + processingData_folder + rowdata_folder_list[i] + "\\" + \
            "FatigueAnalysis\\data\\"
        print(data_svae_path + save_name + "_MedFreq.xlsx")
        # 將檔名拆開
        filepath, tempfilename = os.path.split(grid_shot_file_list[ii])
        filename, extension = os.path.splitext(tempfilename)
        # 處理 .csv 檔案
        if ".csv" in grid_shot_file_list[ii]:
            print(data_svae_path + save_name + "_MedFreq.xlsx")
            # 處理檔名及定義路徑
            
            stage_file = pd.read_excel(staging_file_path, sheet_name=rowdata_folder_list[i])
            # 找分期檔中的檔名
            for iii in range(np.shape(stage_file)[0]):
                if save_name in str(stage_file.loc[iii, 'EMG_File']):
                    # print(stage_file.loc[iii, 'EMG_File'])
                    mouse_name = stage_file.loc[iii, 'Mouse']
                    break
            # 將檔名加上滑鼠名稱
            save_name = save_name + "_" + mouse_name
        elif ".c3d" in grid_shot_file_list[ii]:
            save_name = save_name
        # 前處理EMG data
        processing_data, bandpass_filtered_data = emg.EMG_processing(grid_shot_file_list[ii], smoothing=smoothing)

        # 畫 bandpass 後之資料圖
        emg.plot_plot(bandpass_filtered_data, str(fig_svae_path),
                      filename, "_Bandpass")
        # 畫smoothing 後之資料圖
        emg.plot_plot(processing_data, str(fig_svae_path),
                      filename, str(smoothing + "_"))
        # writting data in worksheet
        file_name = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + MVC_folder + "\\data\\" + filename + end_name + ".xlsx"
        pd.DataFrame(processing_data).to_excel(data_svae_path + save_name + "_lowpass.xlsx",
                                               sheet_name='Sheet1', index=False, header=True)
        # 計算 iMVC
        emg_iMVC = pd.DataFrame(np.zeros(np.shape(processing_data)),
                                columns=processing_data.columns)
        emg_iMVC.iloc[:, 0] = processing_data.iloc[:, 0].values
        emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[:, 1:].values),
                                         MVC_value.values)*100
        
        pd.DataFrame(emg_iMVC).to_excel(data_svae_path + save_name + "_iMVC.xlsx",
                                        sheet_name='Sheet1', index=False, header=True)
        # 進行中頻率分析
        med_freq_data, slope_data = emg.median_frquency(grid_shot_file_list[ii],
                                                        duration, fig_svae_path, save_name)
        # 儲存斜率的資料，並合併成一個資料表
        slope_data['mouse'] = [mouse_name]
        slope_data['data_name'] = [save_name]
        all_slope_data = pd.concat([all_slope_data, slope_data])
        # 繪製傅立葉轉換圖，包含要不要做 notch filter
        # emg.Fourier_plot(grid_shot_file_list[ii], fig_svae_path, save_name)
        # emg.Fourier_plot(grid_shot_file_list[ii], fig_svae_path, save_name, notch=True)
        # 將檔案儲存成.xlsx
        # med_freq_data.to_excel(data_svae_path + save_name + "_MedFreq.xlsx")
        print(data_svae_path + save_name + "_MedFreq.xlsx")
        '''
        已不需要
        # elif ".c3d" in grid_shot_file_list[ii]:
        #     save_name = save_name
        #     print(data_svae_path + save_name + "_MedFreq.xlsx")
        #     # 前處理EMG data
        #     processing_data, bandpass_filtered_data = emg.EMG_processing(grid_shot_file_list[ii], smoothing="lowpass")

        #     # 畫 bandpass 後之資料圖
        #     emg.plot_plot(bandpass_filtered_data, str(fig_svae_path),
        #                  filename, "Bandpass_")
        #     # 畫smoothing 後之資料圖
        #     emg.plot_plot(processing_data, str(fig_svae_path),
        #                  filename, str(smoothing + "_"))
        #     # 中頻率分析
        #     med_freq_data, slope_data = emg.median_frquency(grid_shot_file_list[ii],
        #                                                     duration, fig_svae_path, save_name)
        #     # 儲存斜率的資料，並合併成一個資料表
        #     slope_data['mouse'] = [mouse_name]
        #     slope_data['data_name'] = [save_name]
        #     all_slope_data = pd.concat([all_slope_data, slope_data])
        #     # 繪製傅立葉轉換圖，包含要不要做 notch filter
        #     # emg.Fourier_plot(grid_shot_file_list[ii], fig_svae_path, save_name)
        #     # emg.Fourier_plot(grid_shot_file_list[ii], fig_svae_path, save_name, notch=True)
        #     # 將檔案儲存成.xlsx
        #     # med_freq_data.to_excel(data_svae_path + save_name + "_MedFreq.xlsx")
        '''
        
all_slope_data.to_excel(static_folder + "MedFreq_Static.xlsx")
toc = time.process_time()
print("Fatigue Analysis Total Time Cost: ",toc-tic)
gc.collect()
# %% 繪出各受試者、各滑鼠對應肌肉中頻率
'''
開發中，尚未完成
'''
# 找出各滑鼠在不同受試者下的檔案
all_file = gen.Read_File(data_path + processingData_folder,
                         '.xlsx', subfolder=True)

path_dict = {'_ZA_': [], '_Gpro_': [], '_S_': [], '_U_': [], '_EC_': [], '_FK_': []}

for ii in range(len(all_mouse_name)):
    for i in range(len(all_file)):    
        if "FatigueAnalysis" in all_file[i] and all_mouse_name[ii] in all_file[i]:
            print(all_mouse_name[ii], all_file[i])
            # 將特定的滑鼠名稱放置在各自的位置
            if all_mouse_name[ii] in path_dict:
                path_dict[all_mouse_name[ii]].append(all_file[i])

# 將各自的滑鼠資料取出，找出最短的數值，裁切各自資料，平均後畫圖
data_array = np.empty((len(all_mouse_name), 18, 188, 8)) # 創建一個空的 3D 陣列
for mouse in range(len(all_mouse_name)):
    for subject in range(len(path_dict[all_mouse_name[mouse]])):
        print(all_mouse_name[mouse], subject)
        print(subject)
        # 讀取資料
        read_data = pd.read_excel(path_dict[all_mouse_name[mouse]][subject])
        if read_data.shape[1] <= 188:
            # 如果 data 的形狀小於等於 188，則將 data 填充到 data_array 中
            data_array[mouse, subject, :read_data.shape[0], :] = read_data.iloc[:, 1:]
        else:
            # 如果 data 的形狀大於 188，則擴展 data_array 的大小以容納 data
            data_array[mouse, subject, :read_data.shape[0], :] = read_data.iloc[:189, 1:]
            
'''
note
還需要檢查資料中有沒有 NAN or 0
'''
data_array[0, 0, :read_data.shape[0], :]



   
    # 設定圖片儲存位置，確認是否有做 notch
# %%
import matplotlib.pyplot as plt
import math
n = int(math.ceil(len(all_mouse_name)/2))

# 開始畫圖
for muscle in range(len(muscle_name)):
    plt.figure(figsize=(2*n+1,10))
    fig, axs = plt.subplots(n, 2, figsize = (10,12))
    for col in range(len(all_mouse_name)):
        x, y = col - n*math.floor(abs(col)/n), math.floor(abs(col)/n)
        for subject in range(len(data_array[0, :, 0, 0])): 
            axs[x, y].plot(data_array[col, subject, :, muscle])
            # axs[x, y].plot(xf, 2.0/N * abs(yf[0:int(N/2)]))
            axs[x, y].set_title(all_mouse_name[col], fontsize = 16)
            print(muscle_name[muscle], all_mouse_name[col], subject)
            print(np.count_nonzero(data_array[col, subject, :, muscle]))
    
    
    # 設定整張圖片之參數
    plt.suptitle(str(muscle_name[muscle]), fontsize = 16)
    plt.tight_layout()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("time (second)", fontsize = 14)
    plt.ylabel("frequency (Hz)", fontsize = 14)
    # plt.savefig(save, dpi=200, bbox_inches = "tight")
    plt.show()





# %% 資料前處理 : bandpass filter, absolute value, smoothing, trunkcut data
# 處理MVC data
tic = time.process_time()
for i in range(len(rowdata_folder_list)):
    tic = time.process_time()
    MVC_folder_path = rowdata_folder_path + "\\" + rowdata_folder_list[i] + "\\" + MVC_folder
    csv_MVC_list = gen.Read_File(MVC_folder_path, ".csv")
    c3d_file_path = gen.Read_File(MVC_folder_path, ".c3d")
    MVC_list = csv_MVC_list + c3d_file_path
    fig_save_path = processing_folder_path + "\\" + rowdata_folder_list[i] + "\\"
    print("Now processing MVC data in " + rowdata_folder_list[i])
    for MVC_path in MVC_list:
        print(MVC_path)
        # data = pd.read_csv(MVC_path, encoding='UTF-8')
        data_save_path = processing_folder_path + '\\' + rowdata_folder_list[i] + MVC_folder + "\\data\\"  
        # deal with filename and add extension with _ed
        filepath, tempfilename = os.path.split(MVC_path)
        filename, extension = os.path.splitext(tempfilename)
        # 畫圖
        # 前處理EMG data
        processing_data, bandpass_filtered_data = emg.EMG_processing(MVC_path, smoothing="lowpass")
        # 將檔名拆開
        filepath, tempfilename = os.path.split(MVC_path)
        filename, extension = os.path.splitext(tempfilename)
        # 畫 bandpass 後之資料圖
        emg.plot_plot(bandpass_filtered_data, str(fig_save_path + MVC_folder + "\\smoothing\\"),
                     filename, "_Bandpass")
        # 畫smoothing 後之資料圖
        emg.plot_plot(processing_data, str(fig_save_path + MVC_folder + "\\smoothing\\"),
                     filename, str("_" + smoothing ))
        # 畫 FFT analysis 的圖
        emg.Fourier_plot(MVC_path,
                        (fig_save_path + MVC_folder + "FFT"),
                        filename)
        emg.Fourier_plot(MVC_path,
                        (fig_save_path + MVC_folder + "FFT"),
                        filename,
                        notch=True)

        # writting data in worksheet
        file_name = processing_folder_path + '\\' + rowdata_folder_list[i] + '\\' + MVC_folder + "\\data\\" + filename + end_name + ".xlsx"
        pd.DataFrame(processing_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    toc = time.process_time()
    print("Total Time:",toc-tic)  
# 找最大值
for i in range(len(rowdata_folder_list)):
    print("To find the maximum value of all of MVC data in: " + rowdata_folder_list[i])
    tic = time.process_time()
    emg.Find_MVC_max(processing_folder_path + '\\' + rowdata_folder_list[i] + "\\" +  MVC_folder + "\\data\\",
                 processing_folder_path + '\\' + rowdata_folder_list[i])
    toc = time.process_time()
    print("Total Time:",toc-tic)
toc = time.process_time()
print("MVC Data Total Time Spent: ",toc-tic)
gc.collect()
# %% 計算 iMVC













