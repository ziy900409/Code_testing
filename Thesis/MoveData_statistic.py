#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:05:34 2022

@author: hui
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot
# import spm1d

# -----------------------Reading all of data path----------------------------
# using a recursive loop to traverse each folder
# and find the file extension has .csv
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
                    # replace "\\" to '/', due to MAC version
                    file_list_name = ii + '/' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(x)                
        for i in folder_list:
            if os.path.splitext(i)[1] == data_type:
                # replace "\\" to '/', due to MAC version
                file_list_name = folder_path + '/' + i
                csv_file_list.append(file_list_name)                
        
    return csv_file_list
# -----------------Draw std cloud---------------------
# 設定繪圖格式與字體
plt.style.use('seaborn-white')
# 顯示輸入中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
# 設定顏色格式
# 參考 Qualitative: https://matplotlib.org/stable/tutorials/colors/colormaps.html
palette = plt.get_cmap('Set1')
# 設定文字格式
front1 = {'family': 'Times News Roman',
          'weight': 'normal',
          'size': 18
    }
# 設定圖片文字格式
parameters = {'axes.labelsize': 24, # x 和 y 標籤的字型大小
          'axes.titlesize': 28, # 設定軸標題的字型大小
          'xtick.labelsize': 18, # 設定x軸刻度標籤的字型大小
          'ytick.labelsize': 18} # 設定y軸刻度標籤的字型大小
plt.rcParams.update(parameters)


def std_colud(muscle, columns_name):
    
    # 讀資料
    # data = pd.read_excel(r"D:\NTSU\TenLab\LinData\angle-100_.xlsx",
    #                       skiprows=1) # 跳過列數
    
    # data1 = data.iloc[:, 1:7] #設定資料欄位
    # data2 = data.iloc[:, 11:17] #設定資料欄位
    # data1 = data1.dropna(axis=0) #丟棄NAN的值
    # data2 = data2.dropna(axis=0) #丟棄NAN的值
    # 設定資料
    data1 = muscle[0]
    data2 = muscle[1]
    # 設定圖片大小
    
    iters = list(range(len(data1))) #設定資料的x軸
    
    fig, ax = plt.subplots(figsize=(20, 10))
    # 畫第一條線
    color = palette(0) # 設定顏色
    avg1 = np.mean(data1, axis=1) # 計算平均
    std1 = np.std(data1, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
    plt.plot(iters, avg1, '-o', color=color, label='algo1', linewidth=3)
    # plt.fill_between(iters, r1, r2, color=color, alpha=0.2)
    plt.errorbar(iters, avg1, yerr=std1, fmt="o")
    
    # 畫第二條線
    color = palette(1) # 設定顏色
    avg2 = np.mean(data2, axis=1) # 計畫平均
    std2 = np.std(data2, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg2, std2))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg2, std2)))
    plt.plot(iters, avg2, '-o', color=color, label='algo1', linewidth=3) # 畫平均線
    plt.errorbar(iters, avg2, yerr=std2, fmt="o")
    # plt.fill_between(iters, r1, r2, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊
    
    # 圖片的格式設定
    # x_major_locator = MultipleLocator(10) # 設置x軸刻度為10的倍數
    ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    ax.spines['right'].set_visible(False) #去掉右邊框
    ax.spines['top'].set_visible(False) # 去掉上邊框
    plt.title('圖片名稱', fontsize=28) # 設定圖片名稱與大小
    plt.legend(loc='upper right', bbox_to_anchor = (0.9, 1), prop=front1) # 圖例位置
    ax.set_xlabel('x軸單位') # x軸名稱
    ax.set_ylabel('y軸名稱') # y軸名稱



# ---------Read each pattern-------------------

pattern_1_folder = '/Users/hui/Documents/NTSU/ChenData_test/RMS_move_35/pattern1/'
pattern_2_folder = '/Users/hui/Documents/NTSU/ChenData_test/RMS_move_35/pattern2/'

pattern_1_folder_list = os.listdir(pattern_1_folder)
# 去除有“.“開頭的檔案
pattern_1_folder_list  = [f for f in os.listdir(pattern_1_folder) if not f.startswith('.')]
pattern_2_folder_list = os.listdir(pattern_2_folder)
pattern_2_folder_list  = [f for f in os.listdir(pattern_2_folder) if not f.startswith('.')]

# ---------------------------pattern 1---------------------------------
# 矩陣預定義
one_mean_R_Supraspinatus = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle1
one_mean_L_Supraspinatus = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle2
one_mean_R_Biceps = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle3
one_mean_R_Triceps = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle4
one_mean_R_Deltoid = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle5
one_mean_L_Deltoid = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle6
one_mean_R_TrapeziusUpper = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle7
one_mean_L_TrapeziusUpper = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle8
one_mean_R_TrapeziusLower = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle9
one_mean_L_TrapeziusLower = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle10
one_mean_R_ErectorSpinae = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle11
one_mean_L_ErectorSpinae = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle12
one_mean_R_LatissimusDorsi = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle13
one_mean_L_LatissimusDorsi = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle14
one_mean_R_Extensor = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle15
one_mean_R_Flexor = np.zeros([35, np.shape(pattern_1_folder_list)[0]]) #muscle16

for i in range(len(pattern_1_folder_list)):
    print(pattern_1_folder_list[i])
    # replace "\\" to '/', due to MAC version
    file_name = pattern_1_folder + '/' + pattern_1_folder_list[i] \
        + '/iMVC/韻律'
    file_list = Read_File(file_name, '.xlsx', subfolder = False)
    # 預定義矩陣放置所有檔案的資料
    R_Supraspinatus = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle1
    L_Supraspinatus = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle2
    R_Biceps = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle3
    R_Triceps = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle4
    R_Deltoid = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle5
    L_Deltoid = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle6
    R_TrapeziusUpper = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle7
    L_TrapeziusUpper = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle8
    R_TrapeziusLower = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle9
    L_TrapeziusLower = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle10
    R_ErectorSpinae = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle11
    L_ErectorSpinae = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle12
    R_LatissimusDorsi = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle13
    L_LatissimusDorsi = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle14
    R_Extensor = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle15
    R_Flexor = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle16
    
    for ii in range(len(file_list)):    
        EMG_data = pd.read_excel(file_list[ii])
        print(file_list[ii])
        # 輸入資料矩陣
        R_Supraspinatus.iloc[:, ii] = EMG_data.iloc[:,1]
        L_Supraspinatus.iloc[:,ii] = EMG_data.iloc[:,2]
        R_Biceps.iloc[:,ii] = EMG_data.iloc[:,3]
        R_Triceps.iloc[:,ii] = EMG_data.iloc[:,4]
        R_Deltoid.iloc[:,ii] = EMG_data.iloc[:,5]
        L_Deltoid.iloc[:,ii] = EMG_data.iloc[:,6]
        R_TrapeziusUpper.iloc[:,ii] = EMG_data.iloc[:,7]
        L_TrapeziusUpper.iloc[:,ii] = EMG_data.iloc[:,8]
        R_TrapeziusLower.iloc[:,ii] = EMG_data.iloc[:,9]
        L_TrapeziusLower.iloc[:,ii]= EMG_data.iloc[:,10]
        R_ErectorSpinae.iloc[:,ii]= EMG_data.iloc[:,11]
        L_ErectorSpinae.iloc[:,ii] = EMG_data.iloc[:,12]
        R_LatissimusDorsi.iloc[:,ii] = EMG_data.iloc[:,13]
        L_LatissimusDorsi.iloc[:,ii] = EMG_data.iloc[:,14]
        R_Extensor.iloc[:,ii] = EMG_data.iloc[:,15]
        R_Flexor.iloc[:,ii] = EMG_data.iloc[:,16]
        
    R_Supraspinatus = R_Supraspinatus.dropna(axis=1, how='all')
    L_Supraspinatus = L_Supraspinatus.dropna(axis=1, how='all')
    R_Biceps = R_Biceps.dropna(axis=1, how='all')
    R_Triceps = R_Triceps.dropna(axis=1, how='all')
    R_Deltoid = R_Deltoid.dropna(axis=1, how='all')
    L_Deltoid = L_Deltoid.dropna(axis=1, how='all')
    R_TrapeziusUpper = R_TrapeziusUpper.dropna(axis=1, how='all')
    L_TrapeziusUpper = L_TrapeziusUpper.dropna(axis=1, how='all')
    R_TrapeziusLower = R_TrapeziusLower.dropna(axis=1, how='all')
    L_TrapeziusLower = L_TrapeziusLower.dropna(axis=1, how='all')
    R_ErectorSpinae = R_ErectorSpinae.dropna(axis=1, how='all')
    L_ErectorSpinae = L_ErectorSpinae.dropna(axis=1, how='all')
    R_LatissimusDorsi = R_LatissimusDorsi.dropna(axis=1, how='all')
    L_LatissimusDorsi = L_LatissimusDorsi.dropna(axis=1, how='all')
    R_Extensor = R_Extensor.dropna(axis=1, how='all')
    R_Flexor = R_Flexor.dropna(axis=1, how='all')
    # 計算每位受使者的平均值
    one_mean_R_Supraspinatus[:, i] = np.mean(R_Supraspinatus, axis=1) #muscle1
    one_mean_L_Supraspinatus[:, i] = np.mean(L_Supraspinatus, axis=1) #muscle2
    one_mean_R_Biceps[:, i] = np.mean(R_Biceps, axis=1) #muscle3
    one_mean_R_Triceps[:, i] = np.mean(R_Triceps, axis=1) #muscle4
    one_mean_R_Deltoid[:, i] = np.mean(R_Deltoid, axis=1) #muscle5
    one_mean_L_Deltoid[:, i] = np.mean(L_Deltoid, axis=1) #muscle6
    one_mean_R_TrapeziusUpper[:, i] = np.mean(R_TrapeziusUpper, axis=1) #muscle7
    one_mean_L_TrapeziusUpper[:, i] = np.mean(L_TrapeziusUpper, axis=1) #muscle8
    one_mean_R_TrapeziusLower[:, i] = np.mean(R_TrapeziusLower, axis=1) #muscle9
    one_mean_L_TrapeziusLower[:, i] = np.mean(L_TrapeziusLower, axis=1) #muscle10
    one_mean_R_ErectorSpinae[:, i] = np.mean(R_ErectorSpinae, axis=1) #muscle11
    one_mean_L_ErectorSpinae[:, i] = np.mean(L_ErectorSpinae, axis=1) #muscle12
    one_mean_R_LatissimusDorsi[:, i] = np.mean(R_LatissimusDorsi, axis=1) #muscle13
    one_mean_L_LatissimusDorsi[:, i] = np.mean(L_LatissimusDorsi, axis=1) #muscle14
    one_mean_R_Extensor[:, i] = np.mean(R_Extensor, axis=1) #muscle15
    one_mean_R_Flexor[:, i] = np.mean(R_Flexor, axis=1) #muscle16
    
# ---------------------------pattern 2---------------------------------
# 矩陣預定義
two_mean_R_Supraspinatus = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle1
two_mean_L_Supraspinatus = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle2
two_mean_R_Biceps = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle3
two_mean_R_Triceps = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle4
two_mean_R_Deltoid = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle5
two_mean_L_Deltoid = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle6
two_mean_R_TrapeziusUpper = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle7
two_mean_L_TrapeziusUpper = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle8
two_mean_R_TrapeziusLower = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle9
two_mean_L_TrapeziusLower = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle10
two_mean_R_ErectorSpinae = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle11
two_mean_L_ErectorSpinae = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle12
two_mean_R_LatissimusDorsi = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle13
two_mean_L_LatissimusDorsi = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle14
two_mean_R_Extensor = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle15
two_mean_R_Flexor = np.zeros([35, np.shape(pattern_2_folder_list)[0]]) #muscle16

for i in range(len(pattern_2_folder_list)):
    print(pattern_2_folder_list[i])
    # replace "\\" to '/', due to MAC version
    file_name = pattern_2_folder + '/' + pattern_2_folder_list[i] \
        + '/iMVC/韻律'
    file_list = Read_File(file_name, '.xlsx', subfolder = False)
    # 預定義矩陣放置所有檔案的資料
    R_Supraspinatus = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle1
    L_Supraspinatus = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle2
    R_Biceps = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle3
    R_Triceps = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle4
    R_Deltoid = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle5
    L_Deltoid = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle6
    R_TrapeziusUpper = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle7
    L_TrapeziusUpper = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle8
    R_TrapeziusLower = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle9
    L_TrapeziusLower = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle10
    R_ErectorSpinae = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle11
    L_ErectorSpinae = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle12
    R_LatissimusDorsi = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle13
    L_LatissimusDorsi = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle14
    R_Extensor = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle15
    R_Flexor = pd.DataFrame(np.zeros([35, np.shape(file_list)[0]])) #muscle16
    
    for ii in range(len(file_list)):    
        EMG_data = pd.read_excel(file_list[ii])
        print(file_list[ii])
        # 輸入資料矩陣
        R_Supraspinatus.iloc[:, ii] = EMG_data.iloc[:,1]
        L_Supraspinatus.iloc[:,ii] = EMG_data.iloc[:,2]
        R_Biceps.iloc[:,ii] = EMG_data.iloc[:,3]
        R_Triceps.iloc[:,ii] = EMG_data.iloc[:,4]
        R_Deltoid.iloc[:,ii] = EMG_data.iloc[:,5]
        L_Deltoid.iloc[:,ii] = EMG_data.iloc[:,6]
        R_TrapeziusUpper.iloc[:,ii] = EMG_data.iloc[:,7]
        L_TrapeziusUpper.iloc[:,ii] = EMG_data.iloc[:,8]
        R_TrapeziusLower.iloc[:,ii] = EMG_data.iloc[:,9]
        L_TrapeziusLower.iloc[:,ii]= EMG_data.iloc[:,10]
        R_ErectorSpinae.iloc[:,ii]= EMG_data.iloc[:,11]
        L_ErectorSpinae.iloc[:,ii] = EMG_data.iloc[:,12]
        R_LatissimusDorsi.iloc[:,ii] = EMG_data.iloc[:,13]
        L_LatissimusDorsi.iloc[:,ii] = EMG_data.iloc[:,14]
        R_Extensor.iloc[:,ii] = EMG_data.iloc[:,15]
        R_Flexor.iloc[:,ii] = EMG_data.iloc[:,16]
        
        
    R_Supraspinatus = R_Supraspinatus.dropna(axis=1, how='all')
    L_Supraspinatus = L_Supraspinatus.dropna(axis=1, how='all')
    R_Biceps = R_Biceps.dropna(axis=1, how='all')
    R_Triceps = R_Triceps.dropna(axis=1, how='all')
    R_Deltoid = R_Deltoid.dropna(axis=1, how='all')
    L_Deltoid = L_Deltoid.dropna(axis=1, how='all')
    R_TrapeziusUpper = R_TrapeziusUpper.dropna(axis=1, how='all')
    L_TrapeziusUpper = L_TrapeziusUpper.dropna(axis=1, how='all')
    R_TrapeziusLower = R_TrapeziusLower.dropna(axis=1, how='all')
    L_TrapeziusLower = L_TrapeziusLower.dropna(axis=1, how='all')
    R_ErectorSpinae = R_ErectorSpinae.dropna(axis=1, how='all')
    L_ErectorSpinae = L_ErectorSpinae.dropna(axis=1, how='all')
    R_LatissimusDorsi = R_LatissimusDorsi.dropna(axis=1, how='all')
    L_LatissimusDorsi = L_LatissimusDorsi.dropna(axis=1, how='all')
    R_Extensor = R_Extensor.dropna(axis=1, how='all')
    R_Flexor = R_Flexor.dropna(axis=1, how='all')
    # 計算每位受使者的平均值
    two_mean_R_Supraspinatus[:, i] = np.mean(R_Supraspinatus, axis=1) #muscle1
    two_mean_L_Supraspinatus[:, i] = np.mean(L_Supraspinatus, axis=1) #muscle2
    two_mean_R_Biceps[:, i] = np.mean(R_Biceps, axis=1) #muscle3
    two_mean_R_Triceps[:, i] = np.mean(R_Triceps, axis=1) #muscle4
    two_mean_R_Deltoid[:, i] = np.mean(R_Deltoid, axis=1) #muscle5
    two_mean_L_Deltoid[:, i] = np.mean(L_Deltoid, axis=1) #muscle6
    two_mean_R_TrapeziusUpper[:, i] = np.mean(R_TrapeziusUpper, axis=1) #muscle7
    two_mean_L_TrapeziusUpper[:, i] = np.mean(L_TrapeziusUpper, axis=1) #muscle8
    two_mean_R_TrapeziusLower[:, i] = np.mean(R_TrapeziusLower, axis=1) #muscle9
    two_mean_L_TrapeziusLower[:, i] = np.mean(L_TrapeziusLower, axis=1) #muscle10
    two_mean_R_ErectorSpinae[:, i] = np.mean(R_ErectorSpinae, axis=1) #muscle11
    two_mean_L_ErectorSpinae[:, i] = np.mean(L_ErectorSpinae, axis=1) #muscle12
    two_mean_R_LatissimusDorsi[:, i] = np.mean(R_LatissimusDorsi, axis=1) #muscle13
    two_mean_L_LatissimusDorsi[:, i] = np.mean(L_LatissimusDorsi, axis=1) #muscle14
    two_mean_R_Extensor[:, i] = np.mean(R_Extensor, axis=1) #muscle15
    two_mean_R_Flexor[:, i] = np.mean(R_Flexor, axis=1) #muscle16
    
muscle_name = pd.DataFrame({
                'R_Supraspinatus':[one_mean_R_Supraspinatus, two_mean_R_Supraspinatus],
                'L_Supraspinatus':[one_mean_L_Supraspinatus, two_mean_L_Supraspinatus],
                'R_Biceps':[one_mean_R_Biceps, two_mean_R_Biceps],
                'R_Triceps':[one_mean_R_Triceps, two_mean_R_Triceps],
                'R_Deltoid':[one_mean_R_Deltoid, two_mean_R_Deltoid],
                'L_Deltoid':[one_mean_L_Deltoid, two_mean_L_Deltoid],
                'R_TrapeziusUpper':[one_mean_R_TrapeziusUpper, two_mean_R_TrapeziusUpper],
                'L_TrapeziusUpper':[one_mean_L_TrapeziusUpper, two_mean_L_TrapeziusUpper],
                'R_TrapeziusLower':[one_mean_R_TrapeziusLower, two_mean_R_TrapeziusLower],
                'L_TrapeziusLower':[one_mean_L_TrapeziusLower, two_mean_L_TrapeziusLower],
                'R_ErectorSpinae':[one_mean_R_ErectorSpinae, two_mean_R_ErectorSpinae],
                'L_ErectorSpinae':[one_mean_L_ErectorSpinae, two_mean_L_ErectorSpinae],
                'R_LatissimusDorsi':[one_mean_R_LatissimusDorsi, two_mean_R_LatissimusDorsi],
                'L_LatissimusDorsi':[one_mean_L_LatissimusDorsi, two_mean_L_LatissimusDorsi],
                'R_Extensor':[one_mean_R_Extensor, two_mean_R_Extensor],
                'R_Flexor':[one_mean_R_Flexor, two_mean_R_Flexor]
                
                
    })
