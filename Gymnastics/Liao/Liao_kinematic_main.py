# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:21:31 2025

@author: Hsin.YH.Yang
"""
import numpy as np
import pandas as pd
import sys
# 路徑改成你放自己code的資料夾
# sys.path.append(r"E:\Hsin\git\git\Code_testing\LabProject\function")
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\function")
import gen_function as func
import Kinematic_function as kincal
from scipy.signal import find_peaks

import matplotlib.pyplot as plt




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
"""


# read staging file

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
fig, axes = plt.subplots(2, 2, figsize=(16, 8))  # 建立 2 行 1 列的子圖佈局
# 第一張子圖 繪製髖關節--------------------------------------------------------------
axes[0, 0].plot(r_hip_ext)
# 繪製 Event 2
axes[0, 0].plot(peaks_event2, r_hip_ext[peaks_event2], "x")
axes[0, 0].plot(event_2_motion, r_hip_ext[event_2_motion], "o",
             markerfacecolor="none",  # 中空
             markeredgecolor="g",  # 邊框顏色
             markersize=8)
axes[0, 0].plot(event_2_fp1/4, r_hip_ext[int(event_2_fp1/4)], "o",
             markerfacecolor="none",  # 中空
             markeredgecolor="b",  # 邊框顏色
             markersize=12)
axes[0, 0].vlines(x=peaks_event2, ymin=r_hip_ext[peaks_event2] - properties_event2["prominences"],
               ymax=r_hip_ext[peaks_event2], color="C1")
axes[0, 0].hlines(y=properties_event2["width_heights"], xmin=properties_event2["left_ips"],
               xmax=properties_event2["right_ips"], color="C1")
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














