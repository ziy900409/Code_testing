# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:21:31 2025

@author: Hsin.YH.Yang
"""
import numpy as np
import pandas as pd
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"D:\git\Code_testing\Gymnastics\Liao")
# sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\function")
import Liao_function as func
# import Kinematic_function as kincal
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import os

# %%

ana_threshold = 4
anc_fs = 1000
montion_fs = 250
# %% 處理 MVC

# %%
"""
5個分期點：啟動瞬間S、下蹲結束瞬間D、起跳瞬間T、展體瞬間O、著地瞬間L
分期方法1_力板
    分期點（一）：啟動瞬間S，使用Average小於5*SD做為啟動瞬間
分期方法2_ Motion
    分期點（二）：下蹲結束瞬間D，使用Hip Angle最屈曲瞬間
    分期點（三）：起跳瞬間T， 使用Hip Angle最伸展瞬間
    分期點（四）：展體瞬間O，使用Knee Angle第二次(在空中)伸展瞬間
    分期點（五）：著地瞬間L，地面反作用力大於”某數值"的第一瞬間

"""

"""
1. 同步訊號

力版採樣訊號只有1000
motion 250 hz

必須先抓騰空時間，再往前回推一秒內的次高峰

1. 讀取所有資料夾路徑，利用分期檔找到所需資料夾名稱
    A. 找到力版 (.anc), motion (.csv), EMG (.csv) 檔案路徑並讀檔
    
2. 找到分期時間
    A. 擺盪階段SP: P0 -> 分期檔["P0"]
    B. 預備階段-C PCP: P1 -> 分期檔["P1"]
    C. P2 : -> 分期檔["P2"]
    D. 啟動瞬間 S -> 分期檔["S啟動瞬間"]
    E. 下蹲加速減速轉換瞬間 C -> 分期檔["C下蹲加速減速轉換瞬間"]
    F. 下蹲結束瞬間 D -> 使用Hip Angle最屈曲瞬間
    G. 正衝量結束瞬間 T0 -> 分期檔["T0正衝量結束瞬間"]
    H. 起跳瞬間 T -> 分期檔["T起跳瞬間"]
    I. 展體瞬間 O -> 使用Knee第二次(在空中)伸展瞬間
    J. 著地瞬間 L -> 分期檔["L著地瞬間"]
    
3. 繪圖

"""


# read staging file
stage_data = pd.read_excel(r"D:\Hsin\NTSU_lab\Gymnastics\StagingFile_Liao_20250422.xlsx",
                           sheet_name="ALL")


folder_path = r"D:\Hsin\NTSU_lab\Gymnastics\BTS_experiment\Raw_Data\Method_1"

# 取得所有 motion data folder list
# 去除有“.“開頭的檔案 and 只獲得資料夾路徑，排除其他可能的檔案格式
folder_list = []
for root, dirs, files in os.walk(folder_path):
    # 去掉以「.」開頭的隱藏資料夾
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    # 把當前目錄下的每個子資料夾完整路徑加入 list
    for d in dirs:
        folder_list.append(os.path.join(root, d))



subject_list = stage_data['Subject'].dropna().unique()

for subject in subject_list:
    motion_list = [file for file in stage_data['motion file'] if file.startswith(subject)]
    sf1_paths = [p for p in folder_list if os.path.basename(p) == subject]
    # 尋找所有在資料夾下的 .csv 路徑
    csv_list = func.Read_File(sf1_paths[0],
                               ".csv",
                               subfolder=True)
    # 尋找所有在資料夾下的 .anc 路徑
    anc_list = func.Read_File(sf1_paths[0],
                               ".anc",
                               subfolder=True)
    for num in range(len(stage_data['motion file'])):
        for motion_file in motion_list:
            if motion_file in stage_data['motion file'][num]:
                # 找出三種資料的檔案路徑: motion. anc, EMG
                motion_path = [file for file in csv_list if motion_file in file]
                anc_path = [file for file in anc_list if stage_data['Force Plate file'][num] in file]
                EMG_path = [file for file in csv_list if stage_data['EMG file'][num] in file]   
                # 讀檔 motion, anc, EMG
                motion_data = pd.read_csv(motion_path[0],
                                          skiprows=2,
                                          header=[0, 1])
                motion_data.columns = motion_data.columns.droplevel([1])
                anc_data = pd.read_csv(anc_path[0],
                                       skiprows=8,
                                       sep="\s+",
                                       header=[0, 1, 2])
                anc_data.columns = anc_data.columns.droplevel([1, 2])
                first_header = anc_data.columns.get_level_values(0).tolist()
                # emg_data = 
                # 找到
                # 2. find peak with threshold (please parameter setting)
                peaks, _ = find_peaks(anc_data.loc[:, "C63"], height=ana_threshold)
                # 繪出 analog data 的起始時間
                plt.plot(anc_data.loc[:, "Name"], anc_data.loc[:, "C63"], label='Signal')
                plt.plot(anc_data.loc[peaks, "Name"], anc_data.loc[peaks, "C63"], 'ro', label='Peaks')
                plt.legend()
                plt.show()
                # 3. 找出 analog, motion 兩個時間最接近的 frame, 並定義 start index
                # 因為 motion 的 presetation graph 只有 frame number 沒有秒數
                # 所以不同檔案間的秒數轉換只能使用 anc_fs 轉換成 motion_fs
                # 轉換成 EMG 的秒數
                
                # 定義下肢運動學
                # R Hip Flex / Ext Joint Angle (deg)
                r_hip_ext = motion_data.loc[1:, 'R Hip Flexion / Extension Joint Angle (deg)'].reset_index(drop=True)
                # R Knee Flex / Ext Joint Angle (deg)
                r_knee_ext = motion_data.loc[1:, 'R Knee Flexion / Extension Joint Angle (deg)'].reset_index(drop=True)

                # 定義力版訊號
                # p0 = 
                fp1_z = fp_data.loc[:, 'FZ1']
                fp2_z = fp_data.loc[:, 'FZ2']

                # 分期點（一）：啟動瞬間S，使用Average小於5*SD做為啟動瞬間
                # 找兩力板的訊號
                fp1_z = fp_data.loc[:, 'FZ1']

                # 2. 分期點（二）：下蹲結束瞬間D，使用Hip Angle最屈曲瞬間 -------------------------
                # 從 starting frame 開始找，並且設定兩次動作的時間寬，以及至少需要平均值的三倍高
                peaks_event2, properties_event2 = find_peaks(r_hip_ext[starting_motion:],
                                                             prominence=1,
                                                             width=20,
                                                             height=np.mean(r_hip_ext[starting_motion:]))
                # 將找到的數值加回 starting frame
                peaks_event2 = peaks_event2 + starting_motion
                properties_event2['right_ips'] = properties_event2['right_ips'] + starting_motion
                properties_event2['left_ips'] = properties_event2['left_ips'] + starting_motion
                # 找出兩次最大髖伸展時間區間中的髖伸展最小值
                event_2_motion = peaks_event2[0]
                # 力版資料處理方式：必須先抓騰空時間，再往前回推一秒內的次高峰
                event_2_fp1 = int(np.argmax(fp1_z))
                # 3. 分期點（三）：起跳瞬間T， 使用Hip Angle最伸展瞬間-----------------------------
                event_3_motion = np.argmin(r_hip_ext[peaks_event2[0]:peaks_event2[1]]) + peaks_event2[0]

                # 力版數值小於特定值
                # 找出所有小於 -10 的索引
                indices = [index for index, value in enumerate(fp1_z[starting_analog:]) if value < 10]

                # 將找到的數值加回 starting frame
                event_3_fp1 = indices[0] + starting_analog


                # 4. 分期點（四）：展體瞬間O，使用Knee Angle第二次(在空中)伸展瞬間
                # 找到 R Knee Flex 最小值的下一個 frame
                # R Knee Flex / Ext Joint Angle (deg)
                event_4_motion = np.argmax(r_knee_ext[starting_motion:]) + starting_motion


                # 5. 分期點（五）：著地瞬間L，地面反作用力大於”某數值"的第一瞬間
                # 避免受試者可能落在FP1上，所以評估兩塊力版的受力時間，並以較早的時間為基準
                indices_fp2 = [index for index, value in enumerate(fp2_z[event_3_fp1:]) if value > 10]
                indices_fp1 = [index for index, value in enumerate(fp1_z[event_3_fp1:]) if value > 10]
                if indices_fp2 < indices_fp1:
                    event_5_fp2 = indices_fp2[0] + event_3_fp1
                else:
                    event_5_fp2 = indices_fp1[0] + event_3_fp1




# %%

# starting frame(motion)
starting_motion = int(16664/4)
starting_analog = int(16664)
# read motion, force plate, anc file
# 找到互相對應的檔名
motion_data = pd.read_csv(r"C:\Users\Hsin.YH.Yang\Downloads\論文資料CSV檔\論文資料CSV檔\MOTION\NSF11__1_ok_20250115.data.csv",
                          skiprows=2)


# 確保你的資料是數值型態
motion_num = pd.DataFrame(np.zeros([np.shape(motion_data)[0] -1,
                                    np.shape(motion_data)[1]]
                                   ),
                          columns=motion_data.columns)
for i in range(np.shape(motion_data)[1]):
    motion_num.iloc[:, i] = pd.to_numeric(motion_data.iloc[1:, i], errors='coerce')


fp_data = pd.read_csv(r"C:\Users\Hsin.YH.Yang\Downloads\論文資料CSV檔\論文資料CSV檔\FORCE PLATE\force_NSF11_BTS_2_ok.csv",
                      skiprows=4)
anc_data = pd.read_csv(r"C:\Users\Hsin.YH.Yang\Downloads\論文資料CSV檔\論文資料CSV檔\FORCE PLATE\anc_NSF11_BTS_2.csv",
                       skiprows=8)

# motion_info, motion_data, analog_info, analog_data, np_motion_data = func.read_c3d(motion_path,
#                                                                                    method='vicon')
# read EMG file

# 找 trigger 訊號
trigger_signal = anc_data.loc[2:, 'C63'].reset_index(drop=True)
peaks, _ = find_peaks(trigger_signal, height=np.mean(trigger_signal)*3)
plt.plot(trigger_signal)
plt.plot(peaks, trigger_signal[peaks], "x")
plt.plot(np.zeros_like(trigger_signal), "--", color="gray")
plt.show()
trigger = peaks[0]

# 定義下肢運動學
# R Hip Flex / Ext Joint Angle (deg)
r_hip_ext = motion_num.loc[1:, 'R Hip Flexion / Extension Joint Angle (deg)'].reset_index(drop=True)
# R Knee Flex / Ext Joint Angle (deg)
r_knee_ext = motion_num.loc[1:, 'R Knee Flexion / Extension Joint Angle (deg)'].reset_index(drop=True)

# 定義力版訊號
fp1_z = fp_data.loc[:, 'FZ1']
fp2_z = fp_data.loc[:, 'FZ2']

# 分期點（一）：啟動瞬間S，使用Average小於5*SD做為啟動瞬間
# 找兩力板的訊號
fp1_z = fp_data.loc[:, 'FZ1']

# 2. 分期點（二）：下蹲結束瞬間D，使用Hip Angle最屈曲瞬間 -------------------------
# 從 starting frame 開始找，並且設定兩次動作的時間寬，以及至少需要平均值的三倍高
peaks_event2, properties_event2 = find_peaks(r_hip_ext[starting_motion:],
                                             prominence=1,
                                             width=20,
                                             height=np.mean(r_hip_ext[starting_motion:]))
# 將找到的數值加回 starting frame
peaks_event2 = peaks_event2 + starting_motion
properties_event2['right_ips'] = properties_event2['right_ips'] + starting_motion
properties_event2['left_ips'] = properties_event2['left_ips'] + starting_motion
# 找出兩次最大髖伸展時間區間中的髖伸展最小值
event_2_motion = peaks_event2[0]
# 力版資料處理方式：必須先抓騰空時間，再往前回推一秒內的次高峰
event_2_fp1 = int(np.argmax(fp1_z))
# 3. 分期點（三）：起跳瞬間T， 使用Hip Angle最伸展瞬間-----------------------------
event_3_motion = np.argmin(r_hip_ext[peaks_event2[0]:peaks_event2[1]]) + peaks_event2[0]

# 力版數值小於特定值
# 找出所有小於 -10 的索引
indices = [index for index, value in enumerate(fp1_z[starting_analog:]) if value < 10]

# 將找到的數值加回 starting frame
event_3_fp1 = indices[0] + starting_analog


# 4. 分期點（四）：展體瞬間O，使用Knee Angle第二次(在空中)伸展瞬間
# 找到 R Knee Flex 最小值的下一個 frame
# R Knee Flex / Ext Joint Angle (deg)
event_4_motion = np.argmax(r_knee_ext[starting_motion:]) + starting_motion


# 5. 分期點（五）：著地瞬間L，地面反作用力大於”某數值"的第一瞬間
# 避免受試者可能落在FP1上，所以評估兩塊力版的受力時間，並以較早的時間為基準
indices_fp2 = [index for index, value in enumerate(fp2_z[event_3_fp1:]) if value > 10]
indices_fp1 = [index for index, value in enumerate(fp1_z[event_3_fp1:]) if value > 10]
if indices_fp2 < indices_fp1:
    event_5_fp2 = indices_fp2[0] + event_3_fp1
else:
    event_5_fp2 = indices_fp1[0] + event_3_fp1

# %%
# 繪製 R Hip Flexion / Extension Joint Angle (deg)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 建立 2 行 1 列的子圖佈局
# 第一張子圖 繪製髖關節--------------------------------------------------------------
axes[0, 0].plot(r_hip_ext)
# 繪製 Event 2
# axes[0, 0].plot(peaks_event2, r_hip_ext[peaks_event2], "x")
axes[0, 0].plot(event_2_motion, r_hip_ext[event_2_motion], "o",
             markerfacecolor="none",  # 中空
             markeredgecolor="g",  # 邊框顏色
             markersize=8)
# axes[0, 0].plot(event_2_fp1/4, r_hip_ext[int(event_2_fp1/4)], "o",
#              markerfacecolor="none",  # 中空
#              markeredgecolor="b",  # 邊框顏色
#              markersize=12)
# axes[0, 0].vlines(x=peaks_event2, ymin=r_hip_ext[peaks_event2] - properties_event2["prominences"],
#                ymax=r_hip_ext[peaks_event2], color="C1")
# axes[0, 0].hlines(y=properties_event2["width_heights"], xmin=properties_event2["left_ips"],
#                xmax=properties_event2["right_ips"], color="C1")
# 繪製 Event 3

# 圖資訊
axes[0, 0].set_title('R Hip Flexion / Extension Joint Angle (deg)')
axes[0, 0].set_xlabel('Frame')
axes[0, 0].set_ylabel('Angle (deg)')

# 第二張子圖 膝關節 ------------------------------------------------------------
axes[1, 0].plot(r_knee_ext)
axes[1, 0].plot(event_4_motion, r_knee_ext[event_4_motion], "o",
             markerfacecolor="none",  # 中空
             markeredgecolor="g",  # 邊框顏色
             markersize=8)

# 圖資訊
axes[1, 0].set_title('R Knee Flexion / Extension Joint Angle (deg)')
axes[1, 0].set_xlabel('Frame')
axes[1, 0].set_ylabel('Angle (deg)')

# 第三張子圖 FP1 --------------------------------------------------------------
axes[0, 1].plot(fp1_z)
# Event 2
axes[0, 1].plot(event_2_fp1, fp1_z[event_2_fp1], "o",
             markerfacecolor="none",  # 中空
             markeredgecolor="b",  # 邊框顏色
             markersize=12)
# Event 3
axes[0, 1].plot(event_3_fp1, fp1_z[event_3_fp1], "x",
             color='r')
axes[0, 1].plot(indices[-1] + starting_analog, fp1_z[indices[-1] + starting_analog], "x",
             color='r')


axes[0, 1].set_title('FP1 Z-axis Signal')
axes[0, 1].set_xlabel('Frame')
axes[0, 1].set_ylabel('Amplitude')

# 第四張子圖 FP2 --------------------------------------------------------------
axes[1, 1].plot(fp2_z)
axes[1, 1].plot(event_5_fp2, fp2_z[event_5_fp2], "x",
             color='r')
event_5_fp2

axes[1, 1].set_title('FP1 Z-axis Signal')
axes[1, 1].set_xlabel('Frame')
axes[1, 1].set_ylabel('Amplitude')

# 調整子圖間的間距並顯示圖形
plt.tight_layout()
plt.show()








# 繪圖














