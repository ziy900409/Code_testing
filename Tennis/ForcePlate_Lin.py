# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 19:49:02 2022
@author: user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from detecta import detect_onset
from detecta import detect_peaks
import scipy.integrate as si
import scipy.optimize as op
import os
# 顯示輸入中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

data_folder = r'D:\NTSU\TenLab\LinData\CMJ_SJ Data\RawData\CMJ'
figure_save_path = r'D:\NTSU\TenLab\LinData\CMJ_SJ Data\Results\CMJ'

def Read_File(x, y, subfolder='None'):
    
    # if subfolder = True, the function will run with subfolder
    folder_path = x
    data_type = y
    csv_file_list = []
    
    if subfolder:
        file_list_1 = []
        for dirPath, dirNames, fileNames in os.walk(x):
            # file_list = os.walk(folder_name)
            file_list_1.append(dirPath)
        # need to change here [1:]
        for ii in file_list_1[1:]:
            file_list = os.listdir(ii)
            for iii in file_list:
                if os.path.splitext(iii)[1] == data_type:
                    file_list_name = ii + '\\' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(x)                
        for i in folder_list:
            if os.path.splitext(i)[1] == data_type:
                file_list_name = folder_path + "\\" + i
                csv_file_list.append(file_list_name)                
        
    return csv_file_list

# create data frame
ForcePlate_data = pd.DataFrame({
               
                                    })



data_list = Read_File(data_folder, '.txt', subfolder=False)

for ii in data_list:
    # data path
    file_path = ii
    file_path_split = file_path.split('\\', -1)[-1]
    root, extension = os.path.splitext(ii)
    if extension == '.txt':
        # 使用np.loadtxt讀 .txt
        GRFv = np.loadtxt(file_path, delimiter=',')
        force_z = GRFv[:, 2]
        print(ii)
    elif extension == '.xlsx':
        GRFv = pd.read_excel(ii)
        force_z = GRFv.iloc[:, 2]
        print(ii)
    else:
        print('格式不符')
    

    
    # 平滑處理資料
    mean_force_z = pd.DataFrame()
    for i in range(int(len(force_z)/5)):
        mean_value = np.mean(force_z[5*i:5*i+5])
        mean_value = pd.Series(mean_value)
        mean_force_z = pd.concat([mean_force_z, mean_value], ignore_index=True) 
        # 去除零值
    
    res_data = mean_force_z.iloc[::-1] # 去除從後倒數回來的0值
    a = (res_data.values > 200).argmax(axis=0) # 先反轉矩陣，再用argmax，返回第一個boolean最大值
    cut_force_z = pd.DataFrame(mean_force_z.iloc[:int(2000-a-1), 0])

    
    # 參數設定
    sampling_rate = 1000 #採樣頻率 1000Hz
    data_time = len(cut_force_z)/sampling_rate
    time_span = np.linspace(0, data_time, len(cut_force_z))
    # 重新取樣參數定義
    re_sampling_rate = 200 #採樣頻率 1000Hz
    re_data_time = len(cut_force_z)/re_sampling_rate
    re_time_span = np.linspace(0, re_data_time, len(cut_force_z))
    
    # 計算騰空時間
    # 找出所有數值小於5N的位置
    jump_time = ((cut_force_z.values < 5).sum())/re_sampling_rate
    # 計算跳躍高度 (m)
    jump_high = 1/2 * 9.8 * (jump_time/2)**2 # 1/2*g*t^2
    
    # 計算最大衝擊力: (最大峰值力量 - 觸地前)/所經過時間
    # 觸地前時間 計算: (資料長度 - 最後一個零值)/重取樣時間
    touch_time = len(cut_force_z) - int((cut_force_z.iloc[::-1].values < 5).argmax(axis=0)) - 1 
    # 最大衝擊力 peak_force
    peak_force = np.argmax(cut_force_z.values)
    peak_impact = cut_force_z.iloc[peak_force]
    # 負荷率 load_factor
    load_factor = (cut_force_z.iloc[peak_force] - cut_force_z.iloc[touch_time]) \
        /((peak_force-touch_time)/re_sampling_rate)
    
    # 找下蹲期頂點
    # 利用找onset找下蹲期
    # https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/DetectOnset.ipynb
    down_period = detect_onset(-cut_force_z,
                               np.mean(-cut_force_z.iloc[:50, 0])*0.9,
                               n_above=10, n_below=0, show=False)
    # 利用下蹲期設定下蹲期的力量 (N)
    force_down_period = cut_force_z.iloc[down_period[0][0]:down_period[0][1], 0]
    # 找回下蹲期頂點的index
    peak_down_period = down_period[0][0] + int(np.argmax([-force_down_period]))
    
    # 設定體重 body_mass
    body_mass = np.mean(cut_force_z.iloc[0:10, 0])/9.8
    
    # 設定推蹬期
    acc_period = detect_onset(cut_force_z,
                              np.mean(cut_force_z.iloc[:50, 0])*1.03,
                              n_above=10, n_below=0, show=False)
    
    acc_curve_x = re_time_span[acc_period[0][0]:acc_period[0][1]]
    acc_curve_y = cut_force_z.iloc[acc_period[0][0]:acc_period[0][1], 0]
    
    # 使用polyfit來獲得擬合曲線函數
    parameter = np.polyfit(acc_curve_x, acc_curve_y, 26)
    # 使用poly1d計算曲線值
    acc_curve_fit_ploy = np.poly1d(parameter)
    # 計算微分 求倒數
    diff_acc_curve_poly = np.polyder(acc_curve_fit_ploy)
    diff_acc_curve_poly_1 = np.poly1d(diff_acc_curve_poly)
    # 代入數值，獲得曲線
    diff_acc_curve = diff_acc_curve_poly(acc_curve_x)
    # 解方程式
    slove = op.fsolve(diff_acc_curve_poly,
                      [re_time_span[acc_period[0][0]], re_time_span[acc_period[0][1]]])
    
    # 找第一個推蹬期的波峰
    #np.mean(acc_curve_fit(acc_curve_x), axis= 0)
    threshold = np.percentile(acc_curve_fit_ploy(acc_curve_x), 75) 
    acc_first_peak = detect_onset(acc_curve_fit_ploy(acc_curve_x),
                                  threshold, n_above=10, n_below=0, show=False)
    
    # 積分起跳衝量 jump_lmpulse
    # 上半部面積 - 下半部矩形
    jump_lmpulse = si.quad(np.poly1d(parameter),  acc_curve_x[0], acc_curve_x[-1])[0] \
        - acc_curve_fit_ploy(acc_curve_x)[0]*(acc_curve_x[-1] - acc_curve_x[0])
    # 計算起跳爆發力 explosive
    # 使用detect_peaks 尋找最大峰值
    ind = detect_peaks(acc_curve_y, mph=0, mpd=20, threshold=0, show=True)
    # (上升曲線第一個點 - 下蹲頂點)/經過時間
    # 除以重新取樣的時間 re_sampling_rate
    explosive = (cut_force_z.iloc[int((acc_curve_x[0] + ind[0]/re_sampling_rate)*re_sampling_rate), 0] \
                 - cut_force_z.iloc[peak_down_period, 0]) / \
        (int((acc_curve_x[0] + ind[0]/re_sampling_rate)*re_sampling_rate) \
         - peak_down_period) * re_sampling_rate
    # 測試曲線擬合程度: 最小二乘擬合
    x = []
    y = range(2, 200, 2)
    for i in range(2, 200, 2):
        parameter = np.polyfit(acc_curve_x, acc_curve_y, i)
        acc_curve_fit = np.poly1d(parameter)
        # 公式: sum(abs(origin - fit_curve)**2)
        diff = np.sum(abs(acc_curve_fit(acc_curve_x) - acc_curve_y)**2)
        x.append(diff)
        
    # 繪圖確認標記點
    # 重設圖片名字
    figure_name = file_path_split.split('.')[0]
    figure_save_name = figure_save_path + '\\' + figure_name + '.png'
    
    plt.figure(1)
    plt.plot(re_time_span, cut_force_z.iloc[:, 0])
    plt.plot(peak_down_period/re_sampling_rate, cut_force_z.iloc[peak_down_period, 0], 'o')
    plt.plot(peak_force/re_sampling_rate, cut_force_z.iloc[peak_force, 0], 'o')
    plt.plot(touch_time/re_sampling_rate, cut_force_z.iloc[touch_time, 0], 'o')
    plt.plot(acc_period[0]/re_sampling_rate, cut_force_z.iloc[acc_period[0], 0], 'o')
    plt.plot((acc_curve_x[0] + ind[0]/re_sampling_rate),
             cut_force_z.iloc[int((acc_curve_x[0] + ind[0]/re_sampling_rate)*re_sampling_rate), 0], 'o')
    plt.title(file_path_split)
    plt.savefig(figure_save_name, dpi=300)
    plt.show()
    
    
    
    # 將資料寫到excel
    add_ForcePlate_data = pd.DataFrame({
                                    'file_name': [figure_name],                            
                                    '體重body_mass': [body_mass],
                                    '跳躍高度jump_high': [jump_high],
                                    '最大衝擊力max_impact': [float(peak_impact)],
                                    '負荷率load_factor': [float(load_factor)],
                                    '起跳衝量jump_lmpulse': [float(jump_lmpulse)],
                                    '跳躍爆發力explosive': [explosive],
                                    'peak_down_period_1': [peak_down_period],
                                    'acc_curve_x_2': [acc_period[0][0]],
                                    'peak_push_time': [int((acc_curve_x[0] + ind[0]/re_sampling_rate)*re_sampling_rate)],
                                    'acc_curve_x_3': [acc_period[0][1]],
                                    'touch_time': [touch_time],
                                    'peak_force': [peak_force]
                                    })
    ForcePlate_data = pd.concat([ForcePlate_data, add_ForcePlate_data], ignore_index=True)

# # 繪圖確認推蹬期與擬合曲線的契合程度
plt.figure(2)
plt.plot(acc_curve_x, acc_curve_y, 'b', label='原始資料')
plt.plot(acc_curve_x, acc_curve_fit_ploy(acc_curve_x), 'r*', label = '擬合曲線')
# plt.plot(acc_curve_x, diff_acc_curve_poly_1(acc_curve_x))
# plt.plot(acc_curve_x, diff_acc_curve_poly_1(acc_curve_x), 'r*')
# plt.axhline(y=0, c='r', ls='--', lw=1)
plt.axvline(x=slove[0], c='r', ls='--', lw=1)
plt.axvline(x=slove[1], c='r', ls='--', lw=1)
plt.legend()

plt.show()

# # 畫曲線擬合程度
# plt.figure(3)
# plt.plot(x)
# plt.ylim(0, 1200)
# plt.xlabel('擬合多項式次方數')
# plt.ylabel('最小二乘法總合')
# plt.show()
