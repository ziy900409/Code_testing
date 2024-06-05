# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:40:02 2024

@author: Hsin.YH.Yang
"""

import pandas as pd
import numpy as np
import time
import os
\
import sys
from scipy import signal, interpolate 
# 路徑改成你放自己code的資料夾
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\allseries")
# import AllSeries_emg_func_20240327 as emg
import AllSeries_general_func_20240327 as gen
import AllSeries_emg_func_20240327 as emg
# import gc
from detecta import detect_onset
from datetime import datetime
import matplotlib.pyplot as plt
import gc
# 获取当前日期和时间
now = datetime.now()

# 将日期转换为指定格式
formatted_date = now.strftime("%Y-%m-%d")

# 输出格式化后的日期
print("当前日期：", formatted_date)
# matplotlib 設定中文顯示，以及圖片字型
# mpl.rcParams['font.family'] = 'Microsoft Sans Serif'
# mpl.rcParams["font.sans-serif"] = ["'Microsoft Sans Serif"]
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False   # 步驟二（解決坐標軸負數的負號顯示問題）
font = {'family': 'serif',
        'color':  'k',
        'weight': 'bold',
        'size': 20,
        }
# %%
data_path = r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\\"
RawData_folder = "raw_data\\"
processingData_folder = "processing_data\\"
static_folder = r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\\"
motion_folder = "1. Motion\\"
emg_forder = "2. EMG\\"
MVC_folder = "MVC\\"
sub_folder = ""
fig_save = "figure\\"
end_name = "_ed"
# parameter setting
smoothing = 'lowpass'
c = 0.802
lowpass_cutoff = 10/c
duration = 1
all_mouse_name = ['Gpro', 'ZOWIE_U', 'ZOWIE_ZA', 'ZOWIE_S', 'ZOWIE_FK', 'ZOWIE_EC']
emg_muscle_name = ['Mini sensor 1: EMG 1', 'Mini sensor 2: EMG 2', 'Mini sensor 3: EMG 3',
                   'Quattro sensor 4: EMG.A 4', 'Quattro sensor 4: EMG.B 4',
                   'Quattro sensor 4: EMG.C 4', 'Quattro sensor 4: EMG.D 4',
                   'Avanti sensor 5: EMG 5']
muscle_name = ['Extensor Carpi Radialis', 'Flexor Carpi Radialis', 'Triceps Brachii',
               'Extensor Carpi Ulnaris', '1st Dorsal Interosseous', 
               'Abductor Digiti Quinti', 'Extensor Indicis', 'Biceps Brachii']
MVC_order = {"0": ['1st dorsal interosseous'],
             "1": ['Abductor digiti quinti'],
             "2": ['Extensor Indicis'],
             "3": ['Extensor carpi ulnaris'],
             "4": ['Flexor Carpi Radialis'],
             "5": ['Extensor carpi radialis'],
             "6": ['Biceps'],
             "7": ['Triceps']}
# 替換相異欄位名稱
c3d_recolumns_cortex = {
                        'EC2 Wight:R.I.Finger3_x': 'R.I.Finger3_x',# 食指遠端關節
                        'EC2 Wight:R.I.Finger3_y': 'R.I.Finger3_y',
                        'EC2 Wight:R.I.Finger3_z': 'R.I.Finger3_z',
                        'EC2 Wight:R.M.Finger1_x': 'R.M.Finger1_x',# 中指掌指關節
                        'EC2 Wight:R.M.Finger1_y': 'R.M.Finger1_y',
                        'EC2 Wight:R.M.Finger1_z': 'R.M.Finger1_z'
                      }
# 定義滑鼠marker為 M2, M3, M4 的平均值
mouse_marker = [
                'M2_x', 'M2_y','M2_z', #  mouse2
                'M3_x', 'M3_y', 'M3_z', # mouse3
                'M4_x', 'M4_y', 'M4_z' # mouse4
                ]
test_mouse = ["_Gpro_", "_FK_", "_ZA_", "_S_", "_U_"]
# %%

def generate_mouse_dict(new_name="Mouse"):
    mouse_dict = {}
    for i in range(1, 5):
        mouse_dict[f'{new_name}:M{i}_x'] = f'M{i}_x'
        mouse_dict[f'{new_name}:M{i}_y'] = f'M{i}_y'
        mouse_dict[f'{new_name}:M{i}_z'] = f'M{i}_z'
    return mouse_dict


# %%
rowdata_folder_path = data_path + motion_folder + RawData_folder + "\\" + sub_folder
# 去除有“.“開頭的檔案
rowdata_folder_list  = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.')]
processing_folder_path = data_path + motion_folder + "\\" + processingData_folder + "\\" + sub_folder
processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.')]
# EMG folder
emg_rowdata_folder_path = data_path + emg_forder + RawData_folder + "\\" + sub_folder
# 去除有“.“開頭的檔案
emg_rowdata_folder_list  = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.')]
emg_processing_folder_path = data_path + emg_forder + "\\" + processingData_folder + "\\" + sub_folder
emg_processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.')]
# read staging file
staging_file_path = r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\ZowieAllSeriesRedo_StagingFile_20240531.xlsx"

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

tic = time.process_time()
for folder in range(len(emg_rowdata_folder_list)):
    # 讀資料夾下的 c3d data
    csv_list = gen.Read_File(data_path + emg_forder + RawData_folder + emg_rowdata_folder_list[folder],
                             ".csv", subfolder=False)
   
    
    for csv_file in csv_list:
        if "MVC" in csv_file:
        
            fig_save_path = data_path + emg_forder + processingData_folder + emg_rowdata_folder_list[folder] \
                + "\\" + MVC_folder + "\\" 
            data_save_path = data_path + emg_forder + processingData_folder + emg_rowdata_folder_list[folder] \
                + "\\" + MVC_folder + "\\data\\"  
            print(csv_file)
            # 畫圖
            # 前處理EMG data
            processing_data, bandpass_filtered_data = emg.EMG_processing(csv_file, smoothing=smoothing)
            # 將檔名拆開
            filepath, tempfilename = os.path.split(csv_file)
            filename, extension = os.path.splitext(tempfilename)
            # 畫 bandpass 後之資料圖
            emg.plot_plot(bandpass_filtered_data, str(fig_save_path + "\\smoothing\\"),
                         filename, "_Bandpass")
            # 畫smoothing 後之資料圖
            emg.plot_plot(processing_data, str(fig_save_path + "\\smoothing\\"),
                         filename, str("_" + smoothing))
            # 畫 FFT analysis 的圖
            emg.Fourier_plot(csv_file,
                            (fig_save_path + "\\FFT"),
                            filename)
            emg.Fourier_plot(csv_file,
                            (fig_save_path + "\\FFT"),
                            filename,
                            notch=True)
            # writting data in worksheet
            file_name = data_save_path + filename + end_name + ".xlsx"
            pd.DataFrame(processing_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
        toc = time.process_time()

    toc = time.process_time()
    print("Total Time:",toc-tic)  
gc.collect(generation=2)

# 找最大值
            
for i in range(len(emg_rowdata_folder_list)):
    print("To find the maximum value of all of MVC data in: " + emg_rowdata_folder_list[i])
    staging_file = pd.read_excel(staging_file_path,
                                 sheet_name=emg_rowdata_folder_list[i])
    tic = time.process_time()
    emg.Find_MVC_max_order(staging_file,
                           emg_processing_folder_path + '\\' + emg_rowdata_folder_list[i] + "\\" +  MVC_folder + "\\data\\",
                           emg_processing_folder_path + '\\' + emg_rowdata_folder_list[i] + "\\" +  MVC_folder)
    toc = time.process_time()
    print("Total Time:",toc-tic)
toc = time.process_time()
print("MVC Data Total Time Spent: ",toc-tic)
gc.collect()

# %% 找 MVC 最大值

# %% 分析壓槍
"""
1.1. 檔名定義, 任務基本資訊定義
1.2. 讀取 c3d file and EMG file 以及更新資料欄位名稱
1.3. 定義基本參數, motion and EMG data's samlping rate, find trigger onset time
1.4. motion data filting 

"""
tic = time.process_time()
# 建立 slope data 要儲存之位置
all_dis_data = pd.DataFrame({}, columns = ['mouse'])
# 0. 依序讀取所有的 rowdata folder 下的資料 ------------------------------------
for folder in range(len(rowdata_folder_list)):
    # 讀資料夾下的 c3d data
    c3d_list = gen.Read_File(data_path + motion_folder + RawData_folder + rowdata_folder_list[folder],
                             ".c3d", subfolder=False)
    csv_list = gen.Read_File(data_path + emg_forder + RawData_folder + rowdata_folder_list[folder],
                             ".csv", subfolder=False)
    # 讀分期檔
    staging_file = pd.read_excel(staging_file_path,
                                 sheet_name = rowdata_folder_list[folder])
    # 讀 MVC max file
    MVC_value = pd.read_excel(data_path + emg_forder + processingData_folder + \
                              rowdata_folder_list[folder] + "\\" + MVC_folder + \
                                rowdata_folder_list[folder] + "_all_MVC.xlsx")
    # 只取 all MVC data 數字部分
    MVC_value = MVC_value.iloc[-1, 2:]
    
    # 找出存在 staging file 中的 Recoil
    motion_list = {"motion": [],
                   "emg": []}
    for index in range(len(staging_file)):
        for i in range(len(c3d_list)):
            if pd.isna(staging_file['Motion_File_C3D'][index]) != 1 \
                and staging_file['Motion_File_C3D'][index] in c3d_list[i] \
                    and "Recoil" in c3d_list[i]:
                motion_list["motion"].append(c3d_list[i])
        for i in range(len(csv_list)):
            if pd.isna(staging_file['EMG_File'][index]) != 1 \
                and staging_file['EMG_File'][index] in csv_list[i] \
                    and "Recoil" in csv_list[i]:
                motion_list["emg"].append(csv_list[i])
    # 1. 依序處理所讀取的 motion 以及 emg file
    for ii in range(len(motion_list["motion"])):
        filepath, tempfilename = os.path.split(motion_list["motion"][ii])
        filename, extension = os.path.splitext(tempfilename)
        for iii in range(len(staging_file['Motion_File_C3D'])):
            if tempfilename == staging_file['Motion_File_C3D'][iii]:
                print(motion_list["motion"][ii])
                # 1. 基本資料處理------------------------------------------------------
                # 1.1. 檔名定義, 任務基本資訊定義 ---------------------------------------
                # save_name, extension = os.path.splitext(motion_list["motion"][ii].split('\\', -1)[-1])
                fig_svae_path = data_path + motion_folder + processingData_folder + \
                                rowdata_folder_list[folder] + "\\" + filename + ".jpg"
                
                # 定義受試者編號以及滑鼠名稱
                trial_info = filename.split("_")
                # 定義開始時間
                recoil_begin = int(staging_file['Recoil_begin'][iii])
                recoil_end = recoil_begin + 575
                # 1.2. 讀取 c3d 以及 emg file -----------------------------------------
                motion_info, motion_data, analog_info, analog_data, np_motion_data = gen.read_c3d(motion_list["motion"][ii])
                replace_columns = [item.replace('EC2 Wight:', '') for item in motion_data.columns]
                # 處理欄位名稱問題
                for mouse in all_mouse_name:
                    if trial_info[2] in mouse:
                        mouse_dict = generate_mouse_dict(new_name=mouse)
                # 合併人體 marker set 以及滑鼠 marker set
                
                # 更新 motion data's columns name
                motion_data.columns = replace_columns
                motion_data.rename(columns=mouse_dict, inplace=True)
                # 1.3. 定義基本參數, sampling rate -------------------------------------
                motion_sampling_rate = 1/motion_info['frame_rate']
                
                # find the time of trigger on, 找出大於前 500 frame 平均的三個標準差
                pa_start_onset = detect_onset(analog_data['trigger1'], # 將資料轉成一維型態
                                              np.std(analog_data['trigger1'][:500])*3,
                                              n_above=10, n_below=2, show=True)
                motion_onset = int(pa_start_onset[0, 0]/10 + 240)
                # 1.4. 基本資料處理, filting motion data
                lowpass_sos = signal.butter(2, lowpass_cutoff,  btype='low', fs=motion_info['frame_rate'], output='sos')
                filted_motion = pd.DataFrame(np.empty([575, np.shape(motion_data)[1]]),
                                             columns = motion_data.columns)
                filted_motion.iloc[:, 0] = motion_data.iloc[recoil_begin:recoil_end, 0]
                for i in range(np.shape(motion_data)[1]-1):
                    filted_motion.iloc[:, i+1] = signal.sosfiltfilt(lowpass_sos,
                                                                    motion_data.iloc[recoil_begin:recoil_end, i+1].values)
                # 2. 資料處理 ----------------------------------------------------------
                # 2.1. 找出食指按壓的時間點
                """
                R.I.Finger3
                1. 從 trigger on 後開始三秒，計算 50 frame 的平均，
                2. 找開始 "小於" 及 "大於" 該平均的時間、閾值設定 10%
                3. 計算食指 x 軸負向的速度
                4. 讀分期檔，開始時間再加 575 frame
                """
                
                # %% motion  繪圖 -----------------------------------------------
                # 创建一个包含四个子图的图形，并指定子图的布局
                fig, axes = plt.subplots(4, 1, figsize=(8, 10))  
                # 绘制第一个子图
                axes[0].plot(filted_motion.loc[:, 'Frame'].values,
                             filted_motion.loc[:, 'R.I.Finger3_x'].values,
                             color='blue')  # 假设 data1 是一个 Series 或 DataFrame
                axes[0].axvline(motion_onset/motion_info['frame_rate'], color='r', linestyle='--') # trigger onset
                axes[0].axvline((recoil_begin)/motion_info['frame_rate'], # R.I.Finger down
                                color='c', linestyle='--') 
                axes[0].axvline((recoil_begin + 575)/motion_info['frame_rate'], # R.I.Finger up
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
                # %% 繪製壓槍軌跡
                """
                mouse 2, 3, 4
                是否可以處理 marker 旋轉問題 ?
                """
                fig, axes = plt.subplots(2, 1, figsize=(8, 10))  
                # 绘制第一个子图
                axes[0].plot(filted_motion.loc[:, 'M1_x'].values,
                             filted_motion.loc[:, 'M1_y'].values,
                             color='blue')  # 假设 data1 是一个 Series 或 DataFrame
                axes[0].axvline(motion_onset/motion_info['frame_rate'], color='r', linestyle='--') # trigger onset
                axes[0].axvline((recoil_begin)/motion_info['frame_rate'], # R.I.Finger down
                                color='c', linestyle='--') 
                axes[0].axvline((recoil_begin + 575)/motion_info['frame_rate'], # R.I.Finger up
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
                
                # %
                # 3. 處理 EMG --------------------------------------------------
                processing_data, bandpass_filtered_data = emg.EMG_processing(motion_list["emg"][ii], smoothing=smoothing)
                emg_smaple_rate = int(1 / (bandpass_filtered_data.iloc[1, 0] - bandpass_filtered_data.iloc[0, 0]))
                emg_data_svae = data_path + emg_forder+ processingData_folder + \
                    rowdata_folder_list[folder] + "\\" + "Recoil\\data\\"
                emg_fig_save = data_path + emg_forder + processingData_folder + \
                    rowdata_folder_list[folder] + "\\" + "Recoil\\figure\\"
                
                # 計算 iMVC，分別為 processing data and bandpass data
                bandpass_iMVC = pd.DataFrame(np.empty(np.shape(bandpass_filtered_data)),
                                             columns=bandpass_filtered_data.columns)
                # 取得時間
                bandpass_iMVC.iloc[:, 0] = bandpass_filtered_data.iloc[:, 0].values
                # 除以 MVC 最大值
                bandpass_iMVC.iloc[:, 1:] = np.divide(abs(bandpass_filtered_data.iloc[:, 1:].values),
                                                      MVC_value.values)*100
                # processing data
                processing_iMVC = pd.DataFrame(np.empty(np.shape(processing_data)),
                                               columns=processing_data.columns)
                # 取得時間
                processing_iMVC.iloc[:, 0] = processing_data.iloc[:, 0].values
                # 除以 MVC 最大值
                processing_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[:, 1:].values),
                                                        MVC_value.values)*100
                # 畫 bandpass 後之資料圖
                emg.plot_plot(bandpass_filtered_data, str(emg_fig_save),
                              filename, "_Bandpass")
                # 畫smoothing 後之資料圖
                emg.plot_plot(processing_data, str(emg_fig_save),
                              filename, str(smoothing + "_"))
                # 畫 FFT analysis 的圖
                emg.Fourier_plot(motion_list["emg"][ii],
                                (emg_fig_save),
                                filename)
                emg.Fourier_plot(motion_list["emg"][ii],
                                (emg_fig_save),
                                filename,
                                notch=True)
                
                # EMG 與 motion 時間換算
                emg_recoil_begin = int((recoil_begin - motion_onset)/ motion_info["frame_rate"] * emg_smaple_rate)
                emg_recoil_end = emg_recoil_begin + int((575) / motion_info["frame_rate"] * emg_smaple_rate)
                # 將資料輸出成 EXCEL
                processing_iMVC.iloc[emg_recoil_begin:emg_recoil_end, :].to_excel(emg_data_svae + \
                                                                                  filename + "_iMVC.xlsx")
                # 3.2. EMG 繪圖 ----------------------------------------------- 
                # 创建一个包含四个子图的图形，并指定子图的布局
                fig, axes = plt.subplots(3, 1, figsize=(8, 10))  
                # 绘制第一个子图
                axes[0].plot(processing_iMVC.loc[emg_recoil_begin:emg_recoil_end, 'time'].values,
                             processing_iMVC.loc[emg_recoil_begin:emg_recoil_end, '1st Dorsal Interosseous'].values,
                             color='blue')  # 假设 data1 是一个 Series 或 DataFrame

                axes[0].set_xlim(processing_iMVC['time'].iloc[emg_recoil_begin],
                                 processing_iMVC['time'].iloc[emg_recoil_end])
                axes[0].set_xlabel('time', fontsize=14)
                axes[0].set_ylabel('iMVC (%)', fontsize=14)
                axes[0].set_title('1st. Dorsal Interosseous', fontsize=16)  # 设置子图标题
                # 绘制第二个子图
                axes[1].plot(processing_iMVC.loc[emg_recoil_begin:emg_recoil_end, 'time'].values,
                             processing_iMVC.loc[emg_recoil_begin:emg_recoil_end, 'Abductor Digiti Quinti'].values,
                             color='blue')  # 假设 data1 是一个 Series 或 DataFrame

                axes[1].set_xlim(processing_iMVC['time'].iloc[emg_recoil_begin],
                                 processing_iMVC['time'].iloc[emg_recoil_end])
                axes[1].set_xlabel('time', fontsize=14)
                axes[1].set_ylabel('iMVC (%)', fontsize=14)
                axes[1].set_title('Abductor Digiti Quinti', fontsize=16)  # 设置子图标题
                # 绘制第三个子图
                axes[2].plot(processing_iMVC.loc[emg_recoil_begin:emg_recoil_end, 'time'].values,
                             np.divide(processing_iMVC.loc[emg_recoil_begin:emg_recoil_end, '1st Dorsal Interosseous'].values,
                                       processing_iMVC.loc[emg_recoil_begin:emg_recoil_end, 'Abductor Digiti Quinti'].values),
                             color='blue')  # 假设 data2 是一个 Series 或 DataFrame

                axes[2].set_xlim(processing_iMVC['time'].iloc[emg_recoil_begin],
                                 processing_iMVC['time'].iloc[emg_recoil_end])
                axes[2].set_xlabel('time', fontsize=14)
                axes[2].set_ylabel('Index', fontsize=14)
                axes[2].set_title('肌肉共同收縮比值', fontsize=16)  # 设置子图标题
                # 添加整体标题
                fig.suptitle(filename)  # 设置整体标题
                # 调整子图之间的间距
                plt.tight_layout()
                plt.savefig(emg_fig_save + "\\" + filename + ".jpg",
                            dpi=100)
                # 显示图形
                plt.show()
                
# %% 分析 Recoil cocontraction
output_data = pd.DataFrame({},
                           columns=["mouse"])
Gpro_output_data = np.empty([8, 6388, 8]) # muscle * data len * subject
FK_output_data = np.empty([8, 6388, 8]) # muscle * data len * subject
U_output_data = np.empty([8, 6388, 8]) # muscle * data len * subject
S_output_data = np.empty([8, 6388, 8]) # muscle * data len * subject
ZA_output_data = np.empty([8, 6388, 8]) # muscle * data len * subject
for subject in range(len(emg_processing_folder_list)):
    file_list = gen.Read_File(data_path + emg_forder + processingData_folder + \
                             emg_processing_folder_list[subject] + "\\Recoil\\data",
                             ".xlsx", subfolder=False)
    # 將不同滑鼠的數值取出
    for mouse in range(len(test_mouse)):
        output_data = np.empty([8, 6388, 3])
        i = 0
        for file in file_list:
            if test_mouse[mouse] in file:
                print(test_mouse[mouse])
                filepath, tempfilename = os.path.split(file)
                filename, extension = os.path.splitext(tempfilename)
                subject_mouse = filename.split("_")[-3]
                raw_data = pd.read_excel(file)
                for muscle in range(len(muscle_name)):
                    
                    output_data[muscle, :, i] = raw_data.loc[:, muscle_name[muscle]]
                print(i)
                i = i + 1    
        for muscle in range(len(muscle_name)):
            if test_mouse[mouse] == "_Gpro_":
                Gpro_output_data[muscle, :, subject] = np.mean(output_data[muscle, :, :], axis=1)
            elif test_mouse[mouse] == "_FK_":
                FK_output_data[muscle, :, subject] = np.mean(output_data[muscle, :, :], axis=1)
            elif test_mouse[mouse] == "_U_":
                U_output_data[muscle, :, subject] = np.mean(output_data[muscle, :, :], axis=1)
            elif test_mouse[mouse] == "_S_":
                S_output_data[muscle, :, subject] = np.mean(output_data[muscle, :, :], axis=1)
            elif test_mouse[mouse] == "_ZA_":
                ZA_output_data[muscle, :, subject] = np.mean(output_data[muscle, :, :], axis=1)
            
# 创建一个 Pandas Excel writer 使用 openpyxl 作为引擎
Gpro_output_path = r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\\" + \
    'Recoil_Gpro_' + formatted_date + '.xlsx'
FK_output_path = r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\\" + \
    'Recoil_FK_' + formatted_date + '.xlsx'
U_output_path = r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\\" + \
    'Recoil_U_' + formatted_date + '.xlsx'
S_output_path = r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\\" + \
    'Recoil_S_' + formatted_date + '.xlsx'
ZA_output_path = r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\\" + \
    'Recoil_ZA_' + formatted_date + '.xlsx'
    
with pd.ExcelWriter(Gpro_output_path, engine='openpyxl') as writer:
    for muscle in range(len(muscle_name)):
        # 将每个数据分组转换为 DataFrame
        df = pd.DataFrame(Gpro_output_data[muscle, :, :], columns=[f'Sub{j+1}' for j in range(8)])
        # 将 DataFrame 写入到不同的工作表
        df.to_excel(writer, sheet_name=muscle_name[muscle], index=False)
with pd.ExcelWriter(FK_output_path, engine='openpyxl') as writer:
    for muscle in range(len(muscle_name)):
        # 将每个数据分组转换为 DataFrame
        df = pd.DataFrame(FK_output_data[muscle, :, :], columns=[f'Sub{j+1}' for j in range(8)])
        # 将 DataFrame 写入到不同的工作表
        df.to_excel(writer, sheet_name=muscle_name[muscle], index=False)
with pd.ExcelWriter(U_output_path, engine='openpyxl') as writer:
    for muscle in range(len(muscle_name)):
        # 将每个数据分组转换为 DataFrame
        df = pd.DataFrame(U_output_data[muscle, :, :], columns=[f'Sub{j+1}' for j in range(8)])
        # 将 DataFrame 写入到不同的工作表
        df.to_excel(writer, sheet_name=muscle_name[muscle], index=False)
with pd.ExcelWriter(S_output_path, engine='openpyxl') as writer:
    for muscle in range(len(muscle_name)):
        # 将每个数据分组转换为 DataFrame
        df = pd.DataFrame(S_output_data[muscle, :, :], columns=[f'Sub{j+1}' for j in range(8)])
        # 将 DataFrame 写入到不同的工作表
        df.to_excel(writer, sheet_name=muscle_name[muscle], index=False)
with pd.ExcelWriter(ZA_output_path, engine='openpyxl') as writer:
    for muscle in range(len(muscle_name)):
        # 将每个数据分组转换为 DataFrame
        df = pd.DataFrame(ZA_output_data[muscle, :, :], columns=[f'Sub{j+1}' for j in range(8)])
        # 将 DataFrame 写入到不同的工作表
        df.to_excel(writer, sheet_name=muscle_name[muscle], index=False)

# %%
Gpro_1st = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\Recoil_Gpro_2024-06-04.xlsx",
                          sheet_name="1st Dorsal Interosseous")

Gpro_Abd = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\Recoil_Gpro_2024-06-04.xlsx",
                          sheet_name="Abductor Digiti Quinti")

FK_1st = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\Recoil_FK_2024-06-04.xlsx",
                          sheet_name="1st Dorsal Interosseous")

FK_Abd = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\Recoil_FK_2024-06-04.xlsx",
                          sheet_name="Abductor Digiti Quinti")

U_1st = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\Recoil_U_2024-06-04.xlsx",
                          sheet_name="1st Dorsal Interosseous")

U_Abd = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\Recoil_U_2024-06-04.xlsx",
                          sheet_name="Abductor Digiti Quinti")

S_1st = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\Recoil_S_2024-06-04.xlsx",
                          sheet_name="1st Dorsal Interosseous")

S_Abd = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\Recoil_S_2024-06-04.xlsx",
                          sheet_name="Abductor Digiti Quinti")

ZA_1st = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\Recoil_ZA_2024-06-04.xlsx",
                          sheet_name="1st Dorsal Interosseous")

ZA_Abd = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\2024_05 ZOWIE AllSeries Appendix\3. Statistics\Recoil_ZA_2024-06-04.xlsx",
                          sheet_name="Abductor Digiti Quinti")

# %%
# 3.2. EMG 繪圖 ----------------------------------------------- 
# 创建一个包含四个子图的图形，并指定子图的布局
labels = ['Gpro', 'FK', 'U', 'S', 'ZA']
fig, axes = plt.subplots(3, 1, figsize=(8, 10))  
# 绘制第一个子图
axes[0].plot(np.mean(Gpro_1st, axis=1),
             color='r', label="Gpro")  # 假设 data1 是一个 Series 或 DataFrame
axes[0].plot(np.mean(FK_1st, axis=1),
             color='g', label="FK") 
axes[0].plot(np.mean(U_1st, axis=1),
             color='b', label="U") 
axes[0].plot(np.mean(S_1st, axis=1),
             color='c', label="S") 
axes[0].plot(np.mean(ZA_1st, axis=1),
             color='m', label="ZA") 
axes[0].set_xlabel('time', fontsize=14)
axes[0].set_ylabel('iMVC (%)', fontsize=14)
axes[0].set_title('1st. Dorsal Interosseous', fontsize=16)  # 设置子图标题
# 绘制第二个子图
axes[1].plot(np.mean(Gpro_Abd, axis=1),
             color='r', label="Gpro")  # 假设 data1 是一个 Series 或 DataFrame
axes[1].plot(np.mean(FK_Abd, axis=1),
             color='g', label="FK") 
axes[1].plot(np.mean(U_Abd, axis=1),
             color='b', label="U") 
axes[1].plot(np.mean(S_Abd, axis=1),
             color='c', label="S") 
axes[1].plot(np.mean(ZA_Abd, axis=1),
             color='m', label="ZA") 
axes[1].set_xlabel('time', fontsize=14)
axes[1].set_ylabel('iMVC (%)', fontsize=14)
axes[1].set_title('Abductor Digiti Quinti', fontsize=16)  # 设置子图标题
# 绘制第三个子图
axes[2].plot(np.divide(np.mean(Gpro_1st.values, axis=1),
                       np.mean(Gpro_Abd.values, axis=1)),
             color='r', label="Gpro")  # 假设 data2 是一个 Series 或 DataFrame

axes[2].plot(np.divide(np.mean(FK_1st.values, axis=1),
                       np.mean(FK_Abd.values, axis=1)),
             color='g', label="FK")  # 假设 data2 是一个 Series 或 DataFrame
axes[2].plot(np.divide(np.mean(U_1st.values, axis=1),
                       np.mean(U_Abd.values, axis=1)),
             color='b', label="U")  # 假设 data2 是一个 Series 或 DataFrame
axes[2].plot(np.divide(np.mean(S_1st.values, axis=1),
                       np.mean(S_Abd.values, axis=1)),
             color='c', label="S")  # 假设 data2 是一个 Series 或 DataFrame
axes[2].plot(np.divide(np.mean(ZA_1st.values, axis=1),
                       np.mean(ZA_Abd.values, axis=1)),
             color='m', label="ZA")  # 假设 data2 是一个 Series 或 DataFrame
axes[2].set_xlabel('time', fontsize=14)
axes[2].set_ylabel('index', fontsize=14)
axes[2].set_title('肌肉共同收縮比值', fontsize=16)  # 设置子图标题
# 添加整体标题
fig.suptitle("XXXX")  # 设置整体标题
# 调整子图之间的间距
# 在主图外部添加图例
fig.legend(labels=labels, loc='lower center', ncol=5, fontsize=12)
# plt.tight_layout()
# 调整布局以防止重叠，并为图例腾出空间
plt.tight_layout(rect=[0, 0.04, 1, 1])
# plt.savefig(emg_fig_svae + "\\" + filename + ".jpg",
#             dpi=100)
# 显示图形
plt.show()        





















