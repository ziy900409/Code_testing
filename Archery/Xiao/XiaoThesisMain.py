# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:18:26 2024

目前問題 安卡期判定, 繪圖

程式結構
1. 處理c3d
    1.1. 讀取.c3d
    1.2. 找出 trigger on 時間
    1.3. 計算動作資料
    • E1: 當L.Wrist.Rad Z軸高度超過T10 Z軸高度 擷取此段資料。
    • E5: 擷取直到弓身低於T10 Z軸高度，停止擷取。
    • 資料分期點:
    • E2 舉弓頂點時間:根據全段資料，以L.Wrist.Rad Z軸判定，回傳位置峰值
    數值與對應時間點，即時運算角度後取角度峰直數值與對應時間點
    • E3 當L.Wrist.Rad Z軸高度等於L. Acromion 進行標記
    • E4 放箭時間:根據資料末端2000點判定，即時運算移動平均, R. Elbow Lat
    X軸超出前1秒數據3個標準差，判定為放箭

2. 同步EMG時間
    
@author: Hsin.Yang 05.May.2024
"""
import gc
import os
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"E:\Hsin\git\git\Code_testing\Archery\Xiao")
# sys.path.append(r"D:\BenQ_Project\git\Code_testing\Archery\Xiao")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import XiaoThesisMotionFunction as mot
import XiaoThesisGeneralFunction as gen
import XiaoThesisEMGFunction as emg
from detecta import detect_onset
from scipy import signal

from datetime import datetime

# 获取当前日期和时间
now = datetime.now()

# 将日期转换为指定格式
formatted_date = datetime.now().strftime('%Y-%m-%d-%H:%M')
print("当前日期：", formatted_date)
# %% parameter setting 
staging_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\Archery_stage_v1_input.xlsx"
# c3d_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\R01\SH1_1OK.c3d"
data_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\202405\\"
# staging_path = r"D:\BenQ_Project\python\Archery\Archery_stage_v1_input.xlsx"
# c3d_path = r"D:\BenQ_Project\python\Archery\R01\SH1_1OK.c3d"
# data_path = r"C:/Users/angel/Documents/NTSU/data/112_Plan2_YFMSArchery/"
# data_path = r"D:\BenQ_Project\python\Archery\\"


# ------------------------------------------------------------------------
# 設定資料夾
folder_paramter = {
                  "first_layer": {
                                  "motion":["\\motion\\"],
                                  "EMG": ["\\EMG\\"],
                                  },
                  "second_layer":{
                                  "motion":["\\Raw_Data\\", "\\Processing_Data\\"],
                                  "EMG": ["\\Raw_Data\\", "\\Processing_Data\\"],
                                  },
                  "third_layer":{
                                  "motion":["Method_1"],
                                  "EMG": ["Method_1", "Method_2"],
                                  },
                  "fourth_layer":{
                                  "motion":["\\motion\\"],
                                  "EMG": ["motion", "MVC", "SAVE", "X"],
                                  }
                  }


# 第三層 ----------------------------------------------------------------------

fig_save = "\\figure"
# downsampling frequency
down_freq = 1000
# 抓放箭時候前後秒數
# example : [秒數*採樣頻率, 秒數*採樣頻率]
release = [5*down_freq, 1*down_freq]
# 設定移動平均數與移動均方根之參數
# 更改window length, 更改overlap length
time_of_window = 0.1 # 窗格長度 (單位 second)
overlap_len = 0.5 # 百分比 (%)
# 預處理資料可修改檔名，並新增標籤，如：S2_MVC_Rep_1.16 -> S2_MVC_Rep_1.16_low
end_name = "_ed"
# 平滑處理方式 ex: lowpass, rms, moving
smoothing_method = 'rms'
# cutoff frequency
c = 0.802
lowpass_cutoff = 10/c
# median frequency duration
duration = 1 # unit : second
# processing_folder_path = data_path + processingData_folder

# ---------------------找放箭時間用----------------------------
# 設定最接近放箭位置之acc sensor的欄位編號，建議看完三軸資料再選最大的
# 可設定數字或是欄位名稱：ex: R EXTENSOR GROUP: ACC.Y 1 or 5
release_acc = 7
# 設定放箭的振幅大小值
release_peak = 1.0

# %% 路徑設置

all_rawdata_folder_path = {"motion": [], "EMG": []}
all_processing_folder_path = {"motion": [], "EMG": []}
# 定義 motion
for method in folder_paramter["third_layer"]["motion"]:
    # 定義 rawdata folder path
    rawdata_folder_path = data_path + folder_paramter["first_layer"]["motion"][0] + \
        folder_paramter["second_layer"]["motion"][0] + method
    # 讀取 rawdata folder path
    rawdata_folder_list = [os.path.join(rawdata_folder_path, f) \
                           for f in os.listdir(rawdata_folder_path) \
                           if not f.startswith('.') and os.path.isdir(os.path.join(rawdata_folder_path, f))]
    # 將路徑加到 all_rawdata_folder_path
    all_rawdata_folder_path["motion"].extend(rawdata_folder_list)
    # 定義 processing folder path, 改變 second layer
    processing_folder_path = data_path + folder_paramter["first_layer"]["motion"][0] + \
        folder_paramter["second_layer"]["motion"][1] + method
    processing_folder_list = [os.path.join(processing_folder_path, f) \
                              for f in os.listdir(processing_folder_path) \
                                  if not f.startswith('.') and \
                                      os.path.isdir(os.path.join(processing_folder_path, f))]
    all_processing_folder_path["motion"].extend(processing_folder_list)
    
# 定義 EMG folder path
for method in folder_paramter["third_layer"]["EMG"]:
    # 定義 rawdata folder path
    rawdata_folder_path = data_path + folder_paramter["first_layer"]["EMG"][0] + \
        folder_paramter["second_layer"]["EMG"][0] + method
    # 讀取 rawdata folder path
    rawdata_folder_list = [os.path.join(rawdata_folder_path, f) \
                           for f in os.listdir(rawdata_folder_path) \
                           if not f.startswith('.') and os.path.isdir(os.path.join(rawdata_folder_path, f))]
    # 將路徑加到 all_rawdata_folder_path
    all_rawdata_folder_path["EMG"].extend(rawdata_folder_list)
    # 定義 processing folder path, 改變 second layer
    processing_folder_path = data_path + folder_paramter["first_layer"]["EMG"][0] + \
        folder_paramter["second_layer"]["EMG"][1] + method
    processing_folder_list = [os.path.join(processing_folder_path, f) \
                              for f in os.listdir(processing_folder_path) \
                                  if not f.startswith('.') and \
                                      os.path.isdir(os.path.join(processing_folder_path, f))]
    all_processing_folder_path["EMG"].extend(processing_folder_list)
        

gc.collect(generation=2)

# %%
"""
1. 將分期檔與檔案對應
2. 找分期時間
3. 繪圖
"""
subject_list = ["R01", "R02"]

for subject in subject_list:
    for motion_folder in all_rawdata_folder_path["motion"]:
        for emg_folder in all_rawdata_folder_path["EMG"]:
            if subject in motion_folder and subject in emg_folder:
                print(motion_folder)
                print(emg_folder)
                # read staging file
                staging_file = pd.read_excel(staging_path,
                                             sheet_name=subject)
                motion_list = gen.Read_File(motion_folder,
                                            ".c3d",
                                            subfolder=False)
                emg_list = gen.Read_File(emg_folder,
                                         ".csv",
                                         subfolder=True)
                # 從分期檔來找檔案
                for idx in range(len(staging_file["Motion_filename"])):
                    for motion_file in motion_list:
                        for emg_file in emg_list:
                            if staging_file["Motion_filename"][idx] in motion_file \
                                and staging_file["EMG_filename"][idx] in emg_file:
                                    # print(motion_file)
                                    # print(emg_file)
                                    filepath, tempfilename = os.path.split(motion_file)
                                    # filename, extension = os.path.splitext(tempfilename)
                                    # read .c3d
                                    motion_info, motion_data, analog_info, analog_data, np_motion_data = mot.read_c3d(motion_file)
                                    # read .csv
                                    processing_data, bandpass_filtered = emg.EMG_processing(emg_file, smoothing=smoothing_method)
                                    # rename columns name
                                    rename_columns = motion_data.columns.str.replace("2023 Archery_Rev:", "")
                                    motion_data.columns = rename_columns
                                    # filting motion data
                                    lowpass_sos = signal.butter(2, lowpass_cutoff,  btype='low', fs=motion_info['frame_rate'], output='sos')
                                    filted_motion = pd.DataFrame(np.empty(np.shape(motion_data)),
                                                                 columns = motion_data.columns)
                                    filted_motion.iloc[:, 0] = motion_data.iloc[:, 0]
                                    for i in range(np.shape(motion_data)[1]-1):
                                        filted_motion.iloc[:, i+1] = signal.sosfiltfilt(lowpass_sos,
                                                                                        motion_data.iloc[:, i+1].values)
                                    # temp parameter
                                    # 定義開始時間
                                    start_index = staging_file["Start_index_frame"][idx]
                                    # 定義結束時間
                                    if staging_file["End_index_frame"][idx] == '-':
                                        end_index = len(motion_data)
                                    else: 
                                        end_index = staging_file["End_index_frame"][idx]
                                    # 定義基本參數
                                    motion_sampling_rate = motion_info["frame_rate"]
                                    emg_smaple_rate = int(1 / (bandpass_filtered.iloc[1, 0] - bandpass_filtered.iloc[0, 0]))
                                    # 定義所需要的 markerset, 時間都從 Start_index_frame 開始
                                    L_Wrist_Rad_z = motion_data.loc[start_index:, ["Frame", "L.Wrist.Rad_z"]].reset_index(drop=True)
                                    T10_z = motion_data.loc[start_index:, ["Frame", "T10_z"]].reset_index(drop=True)
                                    L_Acromion_z = motion_data.loc[start_index:, ["Frame", "L.Acromion_z"]].reset_index(drop=True)
                                    R_Elbow_Lat_x = motion_data.loc[start_index:, ["Frame", "R.Epi.Lat_x"]].reset_index(drop=True)
                                    bow_middle = motion_data.loc[start_index:, ["Frame", "Middle_z"]].reset_index(drop=True)
                                    C7 = motion_data.loc[start_index:, ["Frame", "C7_x", "C7_y", "C7_z"]].reset_index(drop=True)
                                    R_Finger = motion_data.loc[start_index:, ["Frame", "R.Finger_x", "R.Finger_y", "R.Finger_z"]].reset_index(drop=True)
                                    # 抓分期時間
                                    # 找L.Wrist.Rad Z, T10 Z, L.Acromion Z, R.Elbow Lat X
                                    
                                    # 0. 抓 trigger onset
                                    # analog channel: C63
                                    triggrt_on = detect_onset(analog_data["C63"]*-1,
                                                              np.mean(analog_data["C63"][:100]*-1)*0.8,
                                                              n_above=0, n_below=0, show=True)
                                    # 0.1. 換算 EMG 時間
                                    emg_start_index = int((start_index - (triggrt_on[0, 0]/analog_info["frame_rate"])) \
                                        / motion_sampling_rate * emg_smaple_rate)
                                    emg_end_index = int((end_index - (triggrt_on[0, 0]/analog_info["frame_rate"])) \
                                        / motion_sampling_rate * emg_smaple_rate)
                                    
                                    # 1. E1: 當L.Wrist.Rad Z軸高度超過T10 Z軸高度 擷取此段資料。
                                    # 布林判斷L_Wrist_Rad_z["L.Wrist.Rad_z"] > T10_z["T10_z"] 的第一個 TRUE 位置
                                    E1_idx = (L_Wrist_Rad_z["L.Wrist.Rad_z"] > T10_z["T10_z"]).idxmax() + start_index
                                    
                                    # 2. E2: 舉弓頂點時間:根據全段資料，以L.Wrist.Rad Z軸判定，回傳位置峰值
                                    #    數值與對應時間點，即時運算角度後取角度峰直數值與對應時間點
                                    E2_idx = L_Wrist_Rad_z["L.Wrist.Rad_z"].idxmax() + start_index
                                    
                                    # 3. E3: 當L.Wrist.Rad Z軸高度等於L. Acromion Z進行標記
                                    # 找兩者相減的最小值
                                    E3_idx_v1 = abs(L_Wrist_Rad_z["L.Wrist.Rad_z"] - L_Acromion_z["L.Acromion_z"]).idxmin() + start_index
                                    # 找安卡期 R.Finger 最貼近 C7 or Front.Head 的時間點 
                                    E3_cal = []
                                    for i in range(len(C7)):
                                        E3_cal.append(gen.euclidean_distance(C7.loc[i, ["C7_x", "C7_y", "C7_z"]],
                                                                             R_Finger.loc[i, ["R.Finger_x", "R.Finger_y", "R.Finger_z"]]))
                                    E3_idx_v2 = np.array(E3_cal).argmin()
                                    """
                                    2024-06-03-23:46 改到這裡
                                    需要改掉
                                    1. 放箭時間:根據資料末端2000點判定，即時運算移動平均, R. Elbow Lat X軸
                                    改成用放箭時間判斷
                                    """
                                    # 4. E4: 放箭時間:根據資料末端2000點判定，即時運算移動平均, R. Elbow Lat X軸
                                    #       超出前1秒數據3個標準差，判定為放箭
                                    E4_idx = detect_onset(-filted_motion.loc[start_index:end_index, "R.Epi.Lat_x"].values,
                                                          np.mean(-filted_motion.loc[end_index-750:end_index, "R.Epi.Lat_x"].values) + \
                                                              np.std(filted_motion.loc[end_index-750:end_index, "R.Epi.Lat_x"].values),
                                                          n_above=0, n_below=0, show=True)
                                    # 5. E5: 擷取直到弓身低於T10 Z軸高度，停止擷取。
                                    # find bow_middle < T10_z and time begin from E2
                                    E5_idx = (bow_middle.loc[E2_idx:, "Middle_z"] < T10_z.loc[E2_idx:, "T10_z"]).idxmax()
                                    # 6. 舉弓角度計算
                                    # T10, L_Wrist_Rad, L_Wrist_Rad 在 T10 平面的投影點
                                    T10 = motion_data.loc[start_index:, ["Frame","T10_x", "T10_y", "T10_z"]]
                                    L_Wrist_Rad = motion_data.loc[start_index:, ["Frame", "L.Wrist.Rad_z"]]
                                    
                                    # 7. 繪圖
                                    fig, axes = plt.subplots(4, 1, figsize=(8, 10))  
                                    # 绘制第一个子图
                                    axes[0].plot(filted_motion.loc[:, 'Frame'].values,
                                                 filted_motion.loc[:, "R.Epi.Lat_x"].values,
                                                 color='blue')  # 假设 data1 是一个 Series 或 DataFrame
                                    axes[0].axvline(E1_idx/motion_info['frame_rate'], color='r', linestyle='--') # trigger onset
                                    axes[0].axvline((E2_idx)/motion_info['frame_rate'], # R.I.Finger down
                                                    color='c', linestyle='--') 
                                    axes[0].axvline((E3_idx_v1)/motion_info['frame_rate'], # R.I.Finger up
                                                    color='c', linestyle='--')
                                    axes[0].axvline((E3_idx_v2)/motion_info['frame_rate'], # R.I.Finger up
                                                    color='c', linestyle='--')
                                    axes[0].axvline((E3_idx_v2)/motion_info['frame_rate'], # R.I.Finger up
                                                    color='c', linestyle='--')
                                    axes[0].set_xlim(0, analog_data['Frame'].iloc[-1])
                                    axes[0].set_title('R.I.Finger3_x')  # 设置子图标题
                                    # 绘制第二个子图
                                    axes[1].plot(filted_motion.loc[:, 'Frame'].values,
                                                 filted_motion.loc[:, 'R.I.Finger3_y'].values,
                                                 color='blue')  # 假设 data2 是一个 Series 或 DataFrame
                                    axes[1].axvline((recoil_begin)/motion_info['frame_rate'], color='r', linestyle='--') # trigger onset
                                    axes[1].axvline((recoil_begin + 575)/motion_info['frame_rate'], # R.I.Finger down
                                                    color='c', linestyle='--') 
                                    axes[1].set_xlim(0, analog_data['Frame'].iloc[-1])
                                    axes[1].set_title('R.I.Finger3_y')  # 设置子图标题
                                    # 绘制第三个子图
                                    axes[2].plot(filted_motion.loc[:, 'Frame'].values,
                                                 filted_motion.loc[:, 'R.I.Finger3_z'].values,
                                                 color='blue')  # 假设 data2 是一个 Series 或 DataFrame
                                    axes[2].axvline((recoil_begin)/motion_info['frame_rate'], color='r', linestyle='--') # trigger onset
                                    axes[2].axvline((recoil_begin + 575)/motion_info['frame_rate'], # R.I.Finger down
                                                    color='c', linestyle='--') 
                                    axes[2].set_xlim(0, analog_data['Frame'].iloc[-1])
                                    axes[2].set_title('R.I.Finger3_z')  # 设置子图标题
                                    # 绘制第四个子图
                                    axes[3].plot(analog_data['Frame'].values,
                                                 analog_data['trigger1'],
                                                 color='blue')  # 假设 data2 是一个 Series 或 DataFrame)  # 假设 data2 是一个 Series 或 DataFrame
                                    axes[3].axvline(pa_start_onset[0, 0]/analog_info['frame_rate'], # trigger onset
                                                    color='r', linestyle='--') 
                                    axes[3].set_xlim(0, analog_data['Frame'].iloc[-1])
                                    axes[3].set_title('Analog')  # 设置子图标题
                                    # 添加整体标题
                                    fig.suptitle(filename)  # 设置整体标题
                                    # 调整子图之间的间距
                                    plt.tight_layout()
                                    plt.savefig(fig_svae_path,
                                                dpi=100)
                                    # 显示图形
                                    plt.show()



# %% 資料前處理 : bandpass filter, absolute value, smoothing
"""
3. 資料前處理: 
    3.1. 需至 function code 修改設定參數
        3.1.1. down_freq = 1800
        # downsampling frequency
        3.1.2. bandpass_cutoff = [8/0.802, 450/0.802]
        # 帶通濾波頻率
        3.1.3. lowpass_freq = 20/0.802
        # 低通濾波頻率
        3.1.4. time_of_window = 0.1 # 窗格長度 (單位 second)
        # 設定移動平均數與移動均方根之參數
        3.1.5. overlap_len = 0.5 # 百分比 (%)
        # 更改window length, 更改overlap length
        
    3.2. 資料處理順序
        3.2.1. bandpsss filter, smoothing data.
        3.2.2. 將處理後之 MVC data 存成 .xlsx 檔案.
        3.2.3. motion data 僅繪圖，資料貯存在 motion data 裁切的部分
"""
# 處理MVC data

for i in range(len(all_rawdata_folder_path)):
    tic = time.process_time()
    for subfolder in folder_paramter["subject_subfolder"]:
        MVC_folder_path = all_rawdata_folder_path[i] + "\\" + subfolder + "\\" + MVC_folder
        MVC_list = af.Read_File(MVC_folder_path, ".csv")
        fig_save_path = all_rawdata_folder_path[i].replace("Raw_Data", "Processing_Data") \
            + "\\" + subfolder + "\\" + fig_save
        print("Now processing MVC data in " + all_rawdata_folder_path[i] + "\\" +  subfolder)
        for MVC_path in MVC_list:
            print(MVC_path)
            # 讀取資料
            data = pd.read_csv(MVC_path, encoding='UTF-8')
            # EMG data 前處理
            processing_data, bandpass_filtered_data = af.EMG_processing(data, smoothing=smoothing_method)
            # 將檔名拆開
            filepath, tempfilename = os.path.split(MVC_path)
            filename, extension = os.path.splitext(tempfilename)
            # 畫 FFT analysis 的圖
            af.Fourier_plot(data,
                         (fig_save_path + "\\FFT\\MVC"),
                         filename)
            # 畫 bandpass 後之資料圖
            af.plot_plot(bandpass_filtered_data, str(fig_save_path + "\\processing\\smoothing\\" + MVC_folder),
                         filename, "Bandpass_")
            # 畫smoothing 後之資料圖
            af.plot_plot(processing_data, str(fig_save_path + "\\processing\\smoothing\\" + MVC_folder),
                         filename, str(smoothing_method + "_"))
            # writting data in worksheet
            file_name =  all_rawdata_folder_path[i].replace("Raw_Data", "Processing_Data")\
                + "\\" + subfolder + "\\data\\" + MVC_folder + '\\' + filename + end_name + ".xlsx"
            pd.DataFrame(processing_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    
    
        # 預處理shooting data
        # for mac version replace "\\" by '/'
        Shooting_path = all_rawdata_folder_path[i] + "\\" + subfolder + "\\" + motion_folder
        Shooting_list = af.Read_File(Shooting_path, '.csv')
        for ii in range(len(Shooting_list)):
            # 印出說明
            x = PrettyTable()
            x.field_names = ["平滑方法", "folder", "shooting_file"]
            x.add_row([smoothing_method, all_rawdata_folder_path[i].split("\\")[-1],
                       Shooting_list[ii].split('\\')[-1]])
            print(x)
            # 讀取資料
            data = pd.read_csv(Shooting_list[ii], encoding='UTF-8')
            # EMG data 前處理
            processing_data, bandpass_filtered_data = af.EMG_processing(data, smoothing="lowpass")
            # 設定 EMG data 資料儲存路徑
            # 將檔名拆開
            filepath, tempfilename = os.path.split(Shooting_list[ii])
            filename, extension = os.path.splitext(tempfilename)
            # 畫 FFT analysis 的圖
            af.Fourier_plot(data,
                            (fig_save_path + "\\FFT\\motion"),
                            filename)
            # 畫 bandpass 後之資料圖
            af.plot_plot(bandpass_filtered_data, str(fig_save_path + "\\processing\\bandpass\\" + motion_folder),
                         filename, "Bandpass_")
            # 畫前處理後之資料圖
            af.plot_plot(processing_data, str(fig_save_path + "\\processing\\smoothing\\" + motion_folder),
                         filename, str("_" + smoothing_method))
    toc = time.process_time()
    print("Total Time:",toc-tic)  
gc.collect(generation=2)



























