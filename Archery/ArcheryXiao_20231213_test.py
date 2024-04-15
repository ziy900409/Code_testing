# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:16:28 2023

流程：
1. 設定資料夾路徑與程式存放路徑
2. 先抓 release time
3. 預處理MVC
    3.1 檢查MVC統計資料
4. 處理motion

Dec 12 2023.
更新項目: 

@author: Hsin.YH.Yang
"""
# %% import library
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"E:\Hsin\git\git\Code_testing\Archery")
# 將read_c3d function 加進現有的工作環境中

import ArcheryFunction_20231214 as af
import os
import time
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import warnings
from scipy import signal
import gc

# %% 設定欲處理的資料路徑
# 資料路徑
data_path = r"E:\Hsin\NTSU_lab\Archery\NSTC\\"

folder_paramter = {"data_path": r"E:\Hsin\NTSU_lab\Archery\NSTC",
                   "method_subfolder": [""],
                   "subject_subfolder": [""],
                   "staging_file":[]}

# 設定分期檔路徑
# staging_data = pd.read_excel(r"//",  "//"       )

# %% 基本資料夾及參數設定
# 設定資料夾
RawData_folder = "\\Raw_Data"
processingData_folder = "\\Processing_Data"
fig_save = "\\figure"
# 子資料夾名稱
sub_folder = "\\\\"
# 動作資料夾名稱
motion_folder = "motion"
# MVC資料夾名稱
MVC_folder = "MVC"
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
smoothing_method = 'lowpass'
# median frequency duration
duration = 1 # unit : second
processing_folder_path = data_path + processingData_folder

# ---------------------找放箭時間用----------------------------
# 設定最接近放箭位置之acc sensor的欄位編號，建議看完三軸資料再選最大的
# 可設定數字或是欄位名稱：ex: R EXTENSOR GROUP: ACC.Y 1 or 5
release_acc = 17
# 設定放箭的振幅大小值
release_peak = 1.5
# --------------------定義繪圖順序----------------------------
order_mapping = {'R EXT': 1, 'R FLX': 2, 'R UT': 3,
                 'R LT': 4, 'R LAT': 5, 'R PD': 6,
                 'L LT': 7, 'L MD': 8}

# %% 路徑設置
all_rawdata_folder_path = []
all_processing_folder_path = []


for method in folder_paramter["method_subfolder"]:
    rowdata_folder_path = data_path + RawData_folder + "\\" + method
    # 去除有“.“開頭的檔案 and 只獲得資料夾路徑，排除其他可能的檔案格式
    rowdata_folder_list = [f for f in os.listdir(rowdata_folder_path) if not f.startswith('.') \
                           and os.path.isdir(os.path.join(rowdata_folder_path, f))]
        
    for i in range(len(rowdata_folder_list)):
        all_rawdata_folder_path.append((data_path + RawData_folder + "\\" \
                                        + method + "\\" + rowdata_folder_list[i]))
    
    
    processing_folder_path = data_path + "\\" + processingData_folder + "\\" + method
    # 去除有“.“開頭的檔案 and 只獲得資料夾路徑，排除其他可能的檔案格式
    processing_folder_list = [f for f in os.listdir(processing_folder_path) if not f.startswith('.') \
                              and os.path.isdir(os.path.join(processing_folder_path, f))]
        
    for ii in range(len(processing_folder_list)):
        all_processing_folder_path.append((data_path + processingData_folder + "\\" \
                                           + method + "\\" + processing_folder_list[ii]))
        
    del rowdata_folder_list, processing_folder_list
gc.collect(generation=2)

# %% 1. 清除所有 processing data 下.xlsx 與 .jpg 檔案
"""
1. 刪除檔案 (本動作刪除之檔案皆無法復原)
    1.1. 刪除processing data 資料夾下所有 .xlsx 與 .jpg 檔案
    1.2. 僅需執行一次

"""
def print_warning_banner():
    print("**************************************************")
    print("*                                                *")
    print("*     警告：這是一個警告標語！                     *")
    print("*     執行將會刪除資料夾下所有 .xlsx 與 .jpg 文件  *")
    print("*     此步驟無法回復所刪除之檔案                   *")
    print("*                                                *")
    print("**************************************************")
tic = time.process_time()    
print_warning_banner()
user_input = input("是否繼續執行刪除文件？(Y/N): ").strip().upper()
if user_input == "Y":
    # 在这里写下后续的代码
    print("繼續執行後續代碼...")

    # 先清除所有 processing data 下 MVC 所有的資料    
    for i in range(len(all_processing_folder_path)):
        # 創建儲存資料夾路徑
        folder_list = []
        # 列出所有 processing data 下之所有資料夾
        for dirPath, dirNames, fileNames in os.walk(all_processing_folder_path[i]):
            if os.path.isdir(dirPath):
                folder_list.append(dirPath)
        for ii in folder_list:
        # 清除所有 .xlsx 檔案
            print(ii)
            af.remove_file(ii, ".xlsx")
            af.remove_file(ii, ".jpg")

elif user_input == "N":
    print("取消執行後續。")
else:
    print("無效輸入，請输入 Y 或 N")
toc = time.process_time()
print("刪除檔案總共花費時間: ",toc-tic)


# %% 找放箭時間
"""
2. 找放箭時間
 2.1. 需至 function code 找參數設定，修改範例如下 :
     2.1.1. release_acc = 5
     可設定數字或是欄位名稱：ex: R EXTENSOR GROUP: ACC.Y 1 or 5
     2.1.2. release_peak = 2
     設定放箭的振幅大小值
            
"""
tic = time.process_time()
for i in range(len(all_rawdata_folder_path)):
    row_folder = all_rawdata_folder_path[i].split("\\")[-1]
    for ii in range(len(all_processing_folder_path)): 
        process_folder = all_processing_folder_path[ii].split("\\")[-1]
        # 
        if row_folder == process_folder:
            for subfolder in folder_paramter["subject_subfolder"]:
                motion_folder_path = all_rawdata_folder_path[i] + "\\" + subfolder + "\\" + motion_folder
                af.find_release_time(all_rawdata_folder_path[i] + "\\" + subfolder + "\\" + motion_folder,
                                     all_processing_folder_path[i] + "\\" + subfolder)
                print("圖片存檔路徑: ", all_processing_folder_path[i])
toc = time.process_time()
print("Release Time Total Time Spent: ", toc-tic)
gc.collect(generation=2)

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
# 處理MVC data 與 shooting data

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
        
# %% 找 MVC 最大值
"""
4. 
"""
for i in range(len(all_processing_folder_path)):
    for subfolder in folder_paramter["subject_subfolder"]:
        print("To find the maximum value of all of MVC data in: " + all_processing_folder_path[i].split("\\")[-1])
        tic = time.process_time()
        af.Find_MVC_max(all_processing_folder_path[i] + "\\" + subfolder + "\\data\\" + MVC_folder,
                        all_processing_folder_path[i] + "\\" + subfolder)
        toc = time.process_time()
        print("Total Time:",toc-tic)
toc = time.process_time()
print("MVC Data Total Time Spent: ",toc-tic)      
gc.collect(generation=2)

# %% 計算 iMVC : trunkcut data and caculate iMVC value
''' 
處理shooting data
# ----------------------------------------------------------------------------
# 1. 取出所有Raw資料夾
# 2. 獲得Raw folder路徑下的“射箭”資料夾，並讀取所有.cvs file
# 3. 讀取processing folder路徑下的ReleaseTiming，並讀取檔案
# 4. 依序前處理“射箭”資料夾下的檔案
# 4.1 bandpass filting
# 4.2 trunkcut data by release time
# 4.3 依切割檔案計算moving average
# 4.4 輸出moving average to excel file
------------------------------------------------------------------------------
'''
tic = time.process_time()

# a = []
# 開始處理 motion 資料
for i in range(len(all_rawdata_folder_path)):
    for subfolder in folder_paramter["subject_subfolder"]:
        # 讀取路徑下所有的 shooting motion file
        Shooting_path = all_rawdata_folder_path[i] + "\\" + subfolder + "\\" + motion_folder
        Shooting_list = af.Read_File(Shooting_path, '.csv')
        # 設定儲存照片路徑
        fig_save_path = all_rawdata_folder_path[i].replace("Raw_Data", "Processing_Data") \
            + "\\" + subfolder + "\\" + fig_save        
        # 讀取 all MVC data
        MVC_value = pd.read_excel(all_rawdata_folder_path[i].replace("Raw_Data", "Processing_Data") \
                                  + "\\" + subfolder + '\\' + all_rawdata_folder_path[i].split("\\")[-1] \
                                      + "_" + subfolder + '_all_MVC.xlsx')
        # 只取 all MVC data 數字部分
        MVC_value = MVC_value.iloc[-1, 2:]
        # 確認資料夾路徑下是否有 staging file
        StagingFile_Exist = os.path.exists(all_rawdata_folder_path[i].replace("Raw_Data", "Processing_Data") \
                                           + "\\" + subfolder + "\\" + all_rawdata_folder_path[i].split('\\')[-1] \
                                               + "_" + subfolder + '_ReleaseTiming.xlsx')
        if StagingFile_Exist:
        # 讀取 staging file    
            staging_data = pd.read_excel(all_rawdata_folder_path[i].replace("Raw_Data", "Processing_Data") \
                                         + "\\" + subfolder + "\\" + all_rawdata_folder_path[i].split('\\')[-1] \
                                             + "_" + subfolder + '_ReleaseTiming.xlsx')
    # ---------------------------------------------------------------------------------------
        for ii in range(len(Shooting_list)):
            """
            1. 新增條件判斷是否有 staging file
            """
            # 讀取資料
            data = pd.read_csv(Shooting_list[ii], encoding='UTF-8')
            # 判斷是否有 staging file 是否為空。如果沒有就再找放箭時間點
            # 如果有 staging file 則直接讀取檔案 
            if StagingFile_Exist:
                for iii in range(len(staging_data['FileName'])):
                    # for mac version replace "\\" by '/'
                    if Shooting_list[ii].split('\\')[-1] == staging_data['FileName'][iii].split('\\')[-1]:
                        print(staging_data['FileName'][iii])
                        print("XXXXXX")
                        # 印出漂亮的說明
                        x = PrettyTable()
                        x.field_names = ["平滑方法", "folder", "shooting_file", "Staging_file"]
                        x.add_row([smoothing_method, all_rawdata_folder_path[i].split("\\")[-1], \
                                   Shooting_list[ii].split('\\')[-1], staging_data['FileName'][iii].split('\\')[-1]])
                        print(x)
                        # 擷取檔名
                        filepath, tempfilename = os.path.split(Shooting_list[ii])
                        filename, extension = os.path.splitext(tempfilename)
                        # rewrite file name
                        """
                        確認一下要不要加 subfolder name
                        """
                        if staging_data['Time Frame(降1000Hz)'][iii] != "Nan":
                            release_idx = int(staging_data['Time Frame(降1000Hz)'][iii])
                        else:
                            release_idx = "Nan"
                        
                        file_name = all_rawdata_folder_path[i].replace("Raw_Data", "Processing_Data") + "\\" \
                            + subfolder + "\\data\\" + motion_folder + '\\' + subfolder + "_" + filename \
                                + "_iMVC" + end_name + ".xlsx"
                        
            # 如果沒有 staging file 則使用加速規自行判斷
            else:
                file_name = all_rawdata_folder_path[i].replace("Raw_Data", "Processing_Data") + "\\" \
                    + subfolder + "\\data\\" + motion_folder + '\\' + subfolder + "_" + filename \
                        + "_iMVC" + end_name + ".xlsx"
                # to find R EXTENSOR CARPI RADIALIS: ACC X data [43]
                
                Extensor_ACC = data.iloc[:, release_acc]
                # acc sampling rate
                acc_freq = int(1/np.mean(np.array(data.iloc[2:11, (release_acc - 1)])
                                         - np.array(data.iloc[1:10, (release_acc - 1)])))
                # there should change with different subject
                peaks, _ = signal.find_peaks(Extensor_ACC*-1, height = release_peak)
                
                if peaks.any():
                    # Because ACC sampling frequency is 148Hz and EMG has been down sampling to 1000Hz
                    release_time = data.iloc[peaks[0], release_acc-1]
                    release_idx = int((peaks[0]/acc_freq)*1000)
                else:
                    release_idx = "Nan"  
                    
                  
            # ---------------------------------------------------------------------------         
            if release_idx != "Nan":
                # pre-processing data
                processing_data, bandpass_filtered_data = af.EMG_processing(data, smoothing=smoothing_method)
                # get release time
               
                # release_samp_freq = int(1/(processing_data.iloc[1, 0] - processing_data.iloc[0, 0]))
                # 去做條件判斷要輸出何種資料
                if smoothing_method == 'lowpass':
                    ## 擷取 EMG data
                    # 計算MVC值
                    emg_iMVC = pd.DataFrame(np.empty([release[0]+release[1], np.shape(processing_data)[1]]),
                                            columns=processing_data.columns)
                    emg_iMVC.iloc[:, 0] = processing_data.iloc[release_idx-release[0]:release_idx+release[1], 0].values
                    emg_iMVC.iloc[:, 1:] = np.divide(abs(processing_data.iloc[release_idx-release[0]:release_idx+release[1], 1:].values),
                                                     MVC_value.values)*100
                elif smoothing_method == 'rms' or smoothing_method == 'moving':
                    # 找出最接近秒數的索引值
                    start_idx = np.abs(processing_data.iloc[:, 0] - (release_idx - release[0])/down_freq).argmin()
                    # # 由於 python 取數字需多 +1
                    end_idx = np.abs(processing_data.iloc[:, 0] - (release_idx + release[1])/down_freq).argmin()
                    print(processing_data.loc[start_idx, "time"], processing_data.loc[end_idx, "time"])
                    if (release_idx + release[1])/down_freq > processing_data.loc[processing_data.index[-1], "time"]:
                        warnings.warn("時間開始位置不一", RuntimeWarning)
                        print("原始數據短於設定擊發後時間，請減少擊發後時間")
                                    
                    # Sep 13 2023.  修正 end_inx 會有不一致的情形, 但是總訊號會少 1 frame
                    while int(end_idx - start_idx) > \
                        int((sum(release) - down_freq * time_of_window) / (down_freq*time_of_window*(1-overlap_len))) + 1:
                        end_idx = end_idx - 1
                    while int(end_idx - start_idx) < \
                        int((sum(release) - down_freq * time_of_window) / (down_freq*time_of_window*(1-overlap_len))) + 1:
                        end_idx = end_idx + 1
        
                    rms_data = processing_data.iloc[start_idx:end_idx, :].reset_index(drop=True)
        
                    emg_iMVC = pd.DataFrame(np.zeros(np.shape(rms_data)),
                                            columns=processing_data.columns)
                    emg_iMVC.iloc[:, 0] = rms_data.iloc[:, 0].values
                    # 加絕對值，以避免數值趨近 0 時，會出現負數問題
                    emg_iMVC.iloc[:, 1:] = np.divide(abs(rms_data.iloc[:, 1:].values),
                                                     MVC_value.values)*100
                print(file_name)
                    # writting data in worksheet
                pd.DataFrame(emg_iMVC).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
            else: # 通常為 fatigue file
                # pre-processing data
                processing_data, bandpass_filtered_data = af.EMG_processing(data)
                # 計算MVC值
                shooting_iMVC = pd.DataFrame(np.zeros([np.shape(processing_data)[0], np.shape(processing_data)[1]]),
                                             columns=processing_data.columns)
                shooting_iMVC.iloc[:, 0] = processing_data.iloc[:, 0]
                shooting_iMVC.iloc[:, 1:] = np.divide(processing_data.iloc[:, 1:], MVC_value.values)*100
                pd.DataFrame(processing_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
                af.median_frquency(data, duration, fig_save_path, filename)
            # 刪除分期檔以避免造成誤用
            try:
                if staging_data in globals() or staging_data in locals():
                    del staging_data
            except TypeError:
                print('The variable does not exist')

toc = time.process_time()
print("Motion Data Total Time Spent: ",toc-tic)
gc.collect(generation=2)
# %% 畫 Mean std cloud 圖                    
'''
1. 列出檔案夾路徑，並設定讀取motion資料夾
2. 給 mean_std_cloud function 繪圖
'''
tic = time.process_time()
for i in range(len(all_processing_folder_path)):
    motion_folder_path = all_processing_folder_path[i] + "\\data\\motion"
    save_path =  all_processing_folder_path[i]
    print("圖片存檔路徑: ", save_path)
    af.compare_mean_std_cloud(all_processing_folder_path[i] + "\\" + folder_paramter["subject_subfolder"][0] +"\\data\\motion",
                              all_processing_folder_path[i] + "\\" + folder_paramter["subject_subfolder"][1] +"\\data\\motion",
                              save_path, all_processing_folder_path[i].split("\\")[-1],
                              smoothing_method, [release[0]/down_freq, release[1]/down_freq],
                              self_oreder=False)
toc = time.process_time()
print("Total Time Spent: ",toc-tic)
gc.collect(generation=2)
        
# %% wavelet analysis

# shot_list = []
# for i in range(len(rowdata_folder_list)):
#     # print(rowdata_folder_list[i])
#     # 預處理shooting data
#     # for mac version replace "\\" by '/'
#     Shooting_path = rowdata_folder_path + '\\' + rowdata_folder_list[i] + "\\" + motion_folder
#     Shooting_list = af.Read_File(Shooting_path, '.csv')
#     shot_list = shot_list + Shooting_list









