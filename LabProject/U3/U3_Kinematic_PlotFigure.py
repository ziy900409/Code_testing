# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 20:08:48 2023

@author: Hsin.YH.Yang
"""

# %%
from scipy.signal import find_peaks
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc
# %%


# %% for Spider task
def emg_table(emg_data, index, direction, axis, file_name):
    # axis 0 : muscle, axis 1 : data length (50 ms), axis 2 : len(index)
    # index=peaks
    # emg_data = trun_emg_iMVC
    # 創建矩陣暫時的貯存空間
    tep_emg_data_table = pd.DataFrame({}, columns = ['檔名', 'direction', 'axis', 'type',
                                                 'Dor1st-mean', 'Dor1st-max', 'Dor3rd-mean', 'Dor3rd-max',
                                                 'abduciton-mean', 'abduciton-max', 'indict-mean', 'indict-max',
                                                 'ExtR-mean', 'ExtR-max', 'ExtU-mean', 'ExtU-max',
                                                 'FlexR-mean', 'FlexR-max', 'Biceps-mean', 'Biceps-max'])
    newnew_data = np.empty(shape=(np.shape(emg_data)[1]-1, 100, len(index)))
    for i in range(np.shape(emg_data)[1]-1):
        for ii in range(len(index)):
            if index[ii]*10+50 < len(emg_data.iloc[:, i+1]) and index[ii]*10 > 50: 
                # print(index[ii], ii)
                # 抓取峰值前後50ms的區間
                newnew_data[i, :, ii] = emg_data.iloc[index[ii]*10-50:index[ii]*10+50, i+1]
            # elif index[ii]*10+50 > len(emg_data.iloc[:, i+1]):
            #     print(index[ii], 0)
            #     newnew_data[i, :, ii] = emg_data.iloc[index[ii]*10-50:, i+1]
    # axis 0 : max or mean, axis 1 : muscle, axis 2 : len(index)
    new3_data = np.empty(shape=(2, np.shape(emg_data)[1]-1, len(index)))
    for iii in range(np.shape(emg_data)[1]-1):
        new3_data[0, iii, :] = np.mean(newnew_data[iii, :, :], axis=0)
        new3_data[1, iii, :] = np.max(newnew_data[iii, :, :], axis=0)
    tep_emg_data_table = pd.concat([tep_emg_data_table, 
                                    pd.DataFrame({'檔名': file_name,
                                                  'direction' :direction,
                                                  'axis': axis,
                                                  'type': 'all',
                                                  'Dor1st-mean':np.mean(new3_data[0, 0, :]),
                                                  'Dor1st-max':np.max(new3_data[1, 0, :]),
                                                  'Dor3rd-mean':np.mean(new3_data[0, 1, :]),
                                                  'Dor3rd-max':np.max(new3_data[1, 1, :]),
                                                  'abduciton-mean':np.mean(new3_data[0, 2, :]),
                                                  'abduciton-max':np.max(new3_data[1, 2, :]),
                                                  'indict-mean':np.mean(new3_data[0, 3, :]),
                                                  'indict-max':np.max(new3_data[1, 3, :]),
                                                  'ExtR-mean':np.mean(new3_data[0, 4, :]),
                                                  'ExtR-max':np.max(new3_data[1, 4, :]),
                                                  'ExtU-mean':np.mean(new3_data[0, 5, :]),
                                                  'ExtU-max':np.max(new3_data[1, 5, :]),
                                                  'FlexR-mean':np.mean(new3_data[0, 6, :]),
                                                  'FlexR-max':np.max(new3_data[1, 6, :]),
                                                  'Biceps-mean':np.mean(new3_data[0, 7, :]),
                                                  'Biceps-max':np.max(new3_data[1, 7, :])},
                                                 index=[0])
                                       ], ignore_index=True)
    for iiii in range(np.shape(new3_data)[2]):
        tep_emg_data_table = pd.concat([tep_emg_data_table, 
                                        pd.DataFrame({'檔名': file_name,
                                                      'direction' :direction,
                                                      'axis': axis,
                                                      'type': 'indi',
                                                      'Dor1st-mean':new3_data[0, 0, iiii],
                                                      'Dor1st-max':new3_data[1, 0, iiii],
                                                      'Dor3rd-mean':new3_data[0, 1, iiii],
                                                      'Dor3rd-max':new3_data[1, 1, iiii],
                                                      'abduciton-mean':new3_data[0, 2, iiii],
                                                      'abduciton-max':new3_data[1, 2, iiii],
                                                      'indict-mean':new3_data[0, 3, iiii],
                                                      'indict-max':new3_data[1, 3, iiii],
                                                      'ExtR-mean':new3_data[0, 4, iiii],
                                                      'ExtR-max':new3_data[1, 4, iiii],
                                                      'ExtU-mean':new3_data[0, 5, iiii],
                                                      'ExtU-max':new3_data[1, 5, iiii],
                                                      'FlexR-mean':new3_data[0, 6, iiii],
                                                      'FlexR-max':new3_data[1, 6, iiii],
                                                      'Biceps-mean':new3_data[0, 7, iiii],
                                                      'Biceps-max':new3_data[1, 7, iiii]},
                                                     index=[0])
                                        ], ignore_index=True)
    return tep_emg_data_table
# %%
# -----------------------plot figure-----------------------------------
# ---------------------------------------------------------------------
def find_and_filter_peaks(peak_value, ori_properties, threshold, num, max_iterations):
    # peak_value = -peak_data
    # ori_properties = Vel_valleys_properties[f'{5}']
    # max_iterations = 100
    del_num = []
    temp_peaks_properties = ori_properties
    max_iterations = 100
    iteration = 0
    # print(num, len(Vel_peaks_properties[f'{num}']['peak_heights']))
    while True:
        iteration += 1
        temp_peaks, temp_peaks_properties = find_peaks(peak_value,
                                                       height=threshold*max(temp_peaks_properties['peak_heights']))
        # print("find", temp_peaks_properties)
        # print(num, len(temp_peaks))
        if len(temp_peaks_properties['peak_heights']) >= 15  and not del_num:
            print(1, len(temp_peaks_properties['peak_heights']))
            output_peaks = temp_peaks
            break
        elif len(temp_peaks_properties['peak_heights']) >= 15 and del_num:
            print(2, len(temp_peaks_properties['peak_heights']))
            # print(temp_peaks_properties['peak_heights'])
            output_peaks = temp_peaks[~np.isin(temp_peaks_properties['peak_heights'], del_num)]
            # print(output_peaks)
            break
        # 如果只有一組數字，那就降低joint_threshold，再找一次
        elif len(temp_peaks_properties['peak_heights']) == 1 or len(del_num) >= len(temp_peaks_properties['peak_heights']):
            print(3, len(temp_peaks_properties['peak_heights']))
            threshold = threshold - 0.1
            del_num.append(max(temp_peaks_properties['peak_heights']))
        # 刪除數字最高的數值，再重找一次peaks
        else:
            if iteration == 1:
                print(4, len(temp_peaks_properties['peak_heights']))
                del_num.append(max(temp_peaks_properties['peak_heights']))
                temp_peaks_properties['peak_heights'] = temp_peaks_properties['peak_heights']\
                    [~np.isin(temp_peaks_properties['peak_heights'], max(temp_peaks_properties['peak_heights']))]
            elif iteration != 1:
                print(5, len(temp_peaks_properties['peak_heights']))
                temp_peaks_properties['peak_heights'] = temp_peaks_properties['peak_heights']\
                    [~np.isin(temp_peaks_properties['peak_heights'], del_num)]
                del_num.append(max(temp_peaks_properties['peak_heights']))
            # print(temp_valleys_properties['peak_heights'])
            # print("valleys max", max(temp_valleys_properties['peak_heights']))
            # print("max", max(temp_peaks_properties['peak_heights']))
        # 保護機制, 避免無法跳出迴圈
        if iteration >= max_iterations:
            print("Reached maximum iterations. Exiting the loop.")
            break
    del temp_peaks_properties, del_num
    gc.collect()
    return output_peaks
    
# %%

def plot_arm_angular(file_name, elbow, wrist, emg_data, joint_threshold, cal_method='vel'):
    # 存檔名稱
    save_path = r"E:\BenQ_Project\U3\09_Results\\"
    folder_name, tempfilename = file_name.split('\\', -1)[-2], file_name.split('\\', -1)[-1]
    save_name, extension = os.path.splitext(tempfilename)
    # 創建暫時貯存的矩陣
    tep_motion_data_table = pd.DataFrame({}, columns = ['檔名', '位置', 'method',
                                                        'Add-平均', 'Add-最大值',
                                                        'Abd-平均', 'Abd-最大值',
                                                        'Pro-平均', 'Pro-最大值',
                                                        'Sup-平均', 'Sup-最大值',
                                                        'Flex-平均', 'Flex-最大值',
                                                        'Ext-平均', 'Ext-最大值'])
    tep_emg_data_table = pd.DataFrame({}, columns = ['檔名', 'direction', 'axis', 'type',
                                                 'Dor1st-mean', 'Dor1st-max', 'Dor3rd-mean', 'Dor3rd-max',
                                                 'abduciton-mean', 'abduciton-max', 'indict-mean', 'indict-max',
                                                 'ExtR-mean', 'ExtR-max', 'ExtU-mean', 'ExtU-max',
                                                 'FlexR-mean', 'FlexR-max', 'Biceps-mean', 'Biceps-max'])

    # 測試用
    # elbow = Elbow_AngAcc
    # wrist =  Wrist_AngAcc
    # elbow = Elbow_AngVel
    # wrist = Wrist_AngVel
    if cal_method.lower() == 'vel':
        axis_name = dict({"y_axis": ["deg/s \n -Abduction / +Adduction", "deg/s \n -Supination / Pronation",
                                     "deg/s \n -Extension / +Flexion",
                                     "deg/s \n -Radial devi / +Ulnar devi", "deg/s \n -Supination / Pronation",
                                     "deg/s \n -Extension / +Flexion"],
                          "title": ["Elbow AngVel (X)", "Elbow AngVel (Y)", "Elbow AngVel (Z)",
                                    "Wrist AngVel (X)", "Wrist AngVel (Y)", "Wrist AngVel (Z)"]
                          })
        fig_title = 'Angular velocity of Elbow and Wrist'
    elif cal_method.lower() == 'acc':
        axis_name = dict({"y_axis": ["deg/${s^2}$ \n -Abduction / +Adduction", "deg/${s^2}$ \n -Supination / Pronation",
                                     "deg/${s^2}$ \n -Extension / +Flexion",
                                     "deg/${s^2}$ \n -Radial devi / +Ulnar devi", "deg/${s^2}$ \n -Supination / Pronation",
                                     "deg/${s^2}$ \n -Extension / +Flexion"],
                          "title": ["Elbow AngAcc (X)", "Elbow AngAcc (Y)", "Elbow AngAcc (Z)",
                                    "Wrist AngAcc (X)", "Wrist AngAcc (Y)", "Wrist AngAcc (Z)"]
                          })
        fig_title = 'Angular acceleration of Elbow and Wrist'
    time = np.arange(0, np.shape(elbow)[0])
    Vel_peaks = dict()
    Vel_valleys = dict()
    Vel_peaks_properties = dict()
    Vel_valleys_properties = dict()
    # 找關節"角速度"的峰值
    # 找出最大峰值速度。並使用joint angular threshold 設定要找峰值的低標位置
    for num in range(6):
        if num < 3:
            peak_data = elbow[:, num]
        else:
            peak_data = wrist[:, num-3]
        # 檢測正峰值
        _, Vel_peaks_properties[f'{num}'] = find_peaks(peak_data, height=0)  
        # 檢測負峰值
        _, Vel_valleys_properties[f'{num}'] = find_peaks(-peak_data, height=0)  # 检测负峰值
    # 找出大於max peak values * joint_threshold位置的索引值
    # 定義迴圈條件:
    # 1. 如果 find peaks 的數量 < 5，就使用次大的peak values，直到find peak的數量超過5
    # 2. 如果 find peaks 的數量 == 1, 就降低joint_threshold, 每次降0.1, 直到 find peaks 數量 > 1
    # 3. 最後依使用次大peak value的次數，刪除最大值的數量, 避免最大值影響分析結果
    for num in range(6):
        if num < 3:
            peak_data = elbow[:, num]
        else:
            peak_data = wrist[:, num-3]

        # 檢測正峰值 
        print("test", num)
        Vel_peaks[f'{num}'] = find_and_filter_peaks(peak_data, Vel_peaks_properties[f'{num}'], joint_threshold, num, 100)
        # 檢測負峰值
        Vel_valleys[f'{num}'] = find_and_filter_peaks(-peak_data, Vel_valleys_properties[f'{num}'], joint_threshold, num, 100)
        

    # 找關節"角加速度"的峰值
    # ---------------繪圖區域----------------
    # ---------------繪製關節角速度圖-------------------------
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    for fig_num in range(6):
        x, y = fig_num - 3*math.floor(abs(fig_num)/3), math.floor(abs(fig_num)/3)
        # print(fig_num, ax)
        if fig_num < 3:
            axs[x, y].plot(time, elbow[:, fig_num])
            axs[x, y].scatter(time[Vel_peaks[f'{fig_num}']],
                              elbow[Vel_peaks[f'{fig_num}'], fig_num], color='red', label='Peaks')
            axs[x, y].scatter(time[Vel_valleys[f'{fig_num}']],
                              elbow[Vel_valleys[f'{fig_num}'], fig_num], color='red', label='Peaks')
            axs[x, y].set_ylabel(axis_name["y_axis"][fig_num], fontsize = 14)
            axs[x, y].set_title(axis_name["title"][fig_num], fontsize = 14)
            axs[x, y].set_xlim(0, len(elbow[:, fig_num])+300)
        else:
            axs[x, y].plot(time, wrist[:, fig_num-3])
            # 繪出峰值速度位置
            axs[x, y].scatter(time[Vel_peaks[f'{fig_num}']],
                              wrist[Vel_peaks[f'{fig_num}'], fig_num-3], color='red', label='Peaks')
            axs[x, y].scatter(time[Vel_valleys[f'{fig_num}']],
                              wrist[Vel_valleys[f'{fig_num}'], fig_num-3], color='red', label='Peaks')
            axs[x, y].set_ylabel(axis_name["y_axis"][fig_num], fontsize = 14)
            axs[x, y].set_title(axis_name["title"][fig_num], fontsize = 14)
            axs[x, y].set_xlim(0, len(wrist[:, fig_num-3])+300)
    plt.tight_layout()
    plt.suptitle(str(fig_title + "(motion): " + save_name), fontsize = 18)
    plt.subplots_adjust(top=0.90)
    plt.grid(False)
    plt.savefig(str(save_path +"\\Motion_fig\\" + folder_name + "\\Spider\\" + save_name + "_Motion.jpg"),
                dpi=200, bbox_inches = "tight")
    plt.show()
    # -------------繪製EMG對應的關節最大峰值速度
    # emg_data = trun_emg_iMVC
    # new_data = processing_data
    # emg_data = processing_data
    # 從 EMG 紀錄開始的5秒~60秒,總共時長55秒
    n = int(math.ceil((np.shape(emg_data)[1] - 1) /2))
    plt.figure(figsize=(2*n+1,10))
    fig, axs = plt.subplots(n, 2, figsize = (10,12), sharex='col')
    for i in range(np.shape(emg_data)[1]-1):
        x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
        # 設定子圖之參數
        axs[x, y].plot(emg_data.iloc[:, 0], emg_data.iloc[:, i+1])
        for fig_num in range(len(Vel_peaks)):
            
            for ii in range(len(Vel_peaks[f'{fig_num}'])):
                # 避免繪圖+50ms超出資料範圍,
                if Vel_peaks[f'{fig_num}'][ii]*10+50 < len(emg_data.iloc[:, 0]):    
                # 抓取峰值前後50ms的區間
                    axs[x, y].axvspan(emg_data.iloc[Vel_peaks[f'{fig_num}'][ii]*10-50, 0],
                                      emg_data.iloc[Vel_peaks[f'{fig_num}'][ii]*10+50, 0],
                                      color='red', alpha=0.3)    
                else:
                    axs[x, y].axvspan(emg_data.iloc[Vel_peaks[f'{fig_num}'][ii]*10-50, 0],
                                      emg_data.iloc[-1, 0],
                                      color='red', alpha=0.3)
            for iii in range(len(Vel_valleys[f'{fig_num}'])):
                if Vel_valleys[f'{fig_num}'][iii]*10+50 < len(emg_data.iloc[:, 0]):
                    axs[x, y].axvspan(emg_data.iloc[Vel_valleys[f'{fig_num}'][iii]*10-50, 0],
                                      emg_data.iloc[Vel_valleys[f'{fig_num}'][iii]*10+50, 0],
                                      color='red', alpha=0.3)
                else:
                    axs[x, y].axvspan(emg_data.iloc[Vel_valleys[f'{fig_num}'][iii]*10-50, 0],
                                      emg_data.iloc[-1, 0],
                                      color='red', alpha=0.3)
        axs[x, y].set_title(emg_data.columns[i+1], fontsize=16)
        # 設定科學符號 : 小數點後幾位數
        axs[x, y].ticklabel_format(axis='y', style = 'scientific', scilimits = (-2, 2))
    # 設定整張圖片之參數
    plt.suptitle(str(fig_title + "(EMG): " + save_name), fontsize = 18)
    plt.tight_layout()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("time (second)", fontsize = 14)
    plt.ylabel("Voltage (V)", fontsize = 14)
    plt.savefig(str(save_path +"\\EMG_fig\\" + folder_name + "\\Spider\\" + save_name + "_EMG.jpg"),
                dpi=200, bbox_inches = "tight")
    plt.show()
    
    # 計算最大值,平均值
    # 將motion統計資料輸入進 data table
    for i in range(2):
        print(i, i)
        if i == 0:
            print(i)
            position = 'elbow'
            data = elbow

        elif i ==1:
            print(i)
            position = 'wrist'
            data = wrist
        tep_motion_data_table = pd.concat([tep_motion_data_table,
                                           pd.DataFrame({'檔名': file_name,
                                                         '位置': position,
                                                         'method': cal_method,
                                                         # peak
                                                         'Add-平均': np.mean(data[Vel_peaks[f'{0+i*3}'], 0]),
                                                         'Add-最大值': np.max(data[Vel_peaks[f'{0+i*3}'], 0]),
                                                         # valleys
                                                         'Abd-平均': np.mean(data[Vel_valleys[f'{0+i*3}'], 0]),
                                                         'Abd-最大值': np.max(data[Vel_valleys[f'{0+i*3}'], 0]),
                                                         # peak
                                                         'Pro-平均': np.mean(data[Vel_peaks[f'{1+i*3}'], 1]),
                                                         'Pro-最大值': np.max(data[Vel_peaks[f'{1+i*3}'], 1]),
                                                         # valleys
                                                         'Sup-平均': np.mean(data[Vel_valleys[f'{1+i*3}'], 1]),
                                                         'Sup-最大值': np.max(data[Vel_valleys[f'{1+i*3}'], 1]),
                                                         # peak
                                                         'Flex-平均': np.mean(data[Vel_peaks[f'{2+i*3}'], 2]),
                                                         'Flex-最大值': np.max(data[Vel_peaks[f'{2+i*3}'], 2]),
                                                         # valleys
                                                         'Ext-平均': np.mean(data[Vel_valleys[f'{2+i*3}'], 2]),
                                                         'Ext-最大值': np.max(data[Vel_valleys[f'{2+i*3}'], 2])},
                                                        index=[0])], ignore_index=True)
        
        # 計算 EMG data 在不同方向的數值大小
    direction = ['+', '-']
    axis = ['elbow_x', 'elbow_y', 'elbow_z', 'wrist_x', 'wrist_y', 'wrist_z']
    for i in range(len(direction)):
        if i < 1:
            print('peak')
            for ii in range(len(axis)):
                # print("peak", i, ii)
                emg_data_table = emg_table(emg_data, Vel_peaks[f'{ii}'],
                                      direction[i], axis[ii], file_name)
                tep_emg_data_table = pd.concat([tep_emg_data_table, emg_data_table])
        else:
            print('valleys')
            for ii in range(6):
                # print("valleys", i, ii)
                emg_data_table = emg_table(emg_data, Vel_valleys[f'{ii}'],
                                           direction[i], axis[ii], file_name)
                tep_emg_data_table = pd.concat([tep_emg_data_table, emg_data_table])


    return tep_motion_data_table, tep_emg_data_table

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------    
# %% for Blink task


def find_Z_peak(file_name, motion_data, motion_info, joint_threshold, emg_data):
    # 測試用
    # FigPlot.find_Z_peak(c3d_list[num],
    #                                                      new_motion_data, new_motion_info, joint_threshold,
    #                                                      trun_emg_iMVC)
    # motion_data = new_motion_data
    # motion_info = new_motion_info
    # emg_data = trun_emg_iMVC
    # 暫存資料位置
    tep_z_motion_table = pd.DataFrame({}, columns = ['檔名', 'vel_mean', 'vel_max', 'acc_mean', 'acc_max'])
    # 處理 file name,並處理存取位置
    save_path = r"E:\BenQ_Project\U3\09_Results\\"
    folder_name, tempfilename = file_name.split('\\', -1)[-2], file_name.split('\\', -1)[-1]
    save_name, extension = os.path.splitext(tempfilename)
    # 找到"R.M.Finger1"的索引位置
    target_strings = ["R.M.Finger1"]
    indices = []
    for target_str in target_strings:
        try:
            index = motion_info["LABELS"].index(target_str)
            indices.append(index)
        except ValueError:
            indices.append(None)
    m_finger1 = np.array(motion_data[indices, :, 2])
    
    # 找向上的"速度"
    diff_m_finger1 = np.diff(m_finger1[0, :], n=1) * motion_info["frame_rate"]
    _, Vel_peaks_properties = find_peaks(diff_m_finger1, height=0) 
    Vel_peaks, _  = find_peaks(diff_m_finger1,
                               height=joint_threshold*max(Vel_peaks_properties['peak_heights']))
    # 找向上的"加速度"
    diff2_m_finger1 = np.diff(m_finger1[0, :], n=2) * (motion_info["frame_rate"]**2)
    _, acc_peaks_properties = find_peaks(diff2_m_finger1, height=0) 
    acc_peaks, _  = find_peaks(diff2_m_finger1,
                               height=joint_threshold*max(acc_peaks_properties['peak_heights']))
    # 繪圖
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    for fig_num in range(2):
        x, y = fig_num - 3*math.floor(abs(fig_num)/3), math.floor(abs(fig_num)/3)
        # print(x, y)
        if fig_num == 0:
            # 繪製 z-axis 速度圖
            axs[x].plot(diff_m_finger1)
            axs[x].plot(Vel_peaks, diff_m_finger1[Vel_peaks], "x")
            axs[x].set_ylabel("mm/s", fontsize = 14)
            axs[x].set_title("the velocity of the marker", fontsize = 14)
        elif fig_num == 1:
            # 繪製 z-axis 加速度圖
            axs[x].plot(diff2_m_finger1)
            axs[x].plot(acc_peaks, diff2_m_finger1[acc_peaks], "x")
            axs[x].set_ylabel("mm/${s^2}$", fontsize = 14)
            axs[x].set_title("the acceleration of the marker", fontsize = 14)
    plt.tight_layout()
    plt.suptitle(str("z-axis vel/acc of R.M.Finger1: " + save_name), fontsize = 18)
    plt.subplots_adjust(top=0.90)
    plt.grid(False)
    plt.savefig(str(save_path +"\\Motion_fig\\" + folder_name + "\\Blink\\" + save_name + "_Zaxis-Motion.jpg"),
                dpi=200, bbox_inches = "tight")
    plt.show()
    # # 繪圖
    # plt.plot(diff2_m_finger1)
    # plt.plot(acc_peaks, diff2_m_finger1[acc_peaks], "x")
    # plt.show()
    
    # 處理EMG 找相應的EMG活化大小
    vel_emg_table  = emg_table(emg_data, Vel_peaks, "z", "vel", file_name)
    acc_emg_table  = emg_table(emg_data, acc_peaks, "z", "acc", file_name)
    tem_emg_table = pd.concat([vel_emg_table, acc_emg_table])
    # 找出向上的平均與最快速度
    # 找出motion data的平均值與最大值
    tep_z_motion_table = pd.concat([tep_z_motion_table,
                                    pd.DataFrame({'檔名':file_name,
                                                  'vel_mean': np.mean(diff_m_finger1[Vel_peaks]),
                                                  'vel_max': np.max(diff_m_finger1[Vel_peaks]),
                                                  'acc_mean': np.mean(diff2_m_finger1[acc_peaks]),
                                                  'acc_max':np.max(diff2_m_finger1[acc_peaks])},
                                                 index=[0])],
                                   ignore_index=True)
    
    return tep_z_motion_table, tem_emg_table
    
# %% 計算手指的關節角度
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    