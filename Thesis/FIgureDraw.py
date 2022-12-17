# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:20:48 2022

@author: user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math

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

data_sheet_name = ['L_Deltoid', 'L_ErectorSpine', 'L_LatissimusDorsi', 'L_Supraspinatus',
                   'L_TrapeziusLower', 'L_TrapeziusUpper', 
                   'R_Biceps', 
                   'R_Deltoid', 'R_ErectorSpinae', 'R_LatissimusDorsi', 'R_Supraspinatus',
                   'R_TrapeziusLower', 'R_TrapeziusUpper',
                   'R_Triceps']

raw_xspan = [[0, 26.5, 34], [1], [2], [3], [4], [5], [6],
         [7], [8], [9], [10], [11, 26.5, 27.5], [12], [13]]
for x in range(len(raw_xspan)):
    for xx in range(len(raw_xspan[x])):
        raw_xspan[x][xx] = raw_xspan[x][xx]*100 - 2500

change_xspan = [[0, 29.5, 30.5], [1, 7.5, 8.5, 30.5, 31.5], [2, 7.5, 8.5, 10.5, 12.5],[3], [4, 22.5, 23.5, 30.5, 31.5], [5],[6, 9.5, 10.5],
                [7], [8, 22.5, 23.5], [9, 14.5, 15.5, 24.5, 25.5, 28.5, 29.5, 31.5, 32.5], [10], [11, 20.5, 21.5, 22.5, 23.5],[12], [13, 3.5, 4.5, 25.5, 26.5]]
for x in range(len(change_xspan)):
    for xx in range(len(change_xspan[x])):
        change_xspan[x][xx] = (change_xspan[x][xx]+1)*100 - 2500

fig, axs = plt.subplots(7,2, dpi=100, figsize=(24, 32))

for name in range(len(data_sheet_name)):
    print(name - 7*math.ceil((name-6)/7))
    print( math.floor(name/7))
    raw_data = pd.read_excel(r'F:/HsinYang/Thesis/Archery_move40_statistic_20221014.xlsx', sheet_name = data_sheet_name[name])
    # 讀資料
    # 原始值
    data1 = raw_data.iloc[:, 1:12] #設定資料欄位
    data2 = raw_data.iloc[:, 15:23] #設定資料欄位
    data1 = data1.dropna(axis=0) #丟棄NAN的值
    data2 = data2.dropna(axis=0) #丟棄NAN的值
    iters = list(range(-2500, 1000, 100)) #設定資料的x軸
    # 設定圖片大小
    # fig, ax = plt.subplots(8, 2, name, figsize=(20, 10))
    # 畫第一條線
    color = palette(0) # 設定顏色
    avg1 = np.mean(data1, axis=1) # 計算平均
    std1 = np.std(data1, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].plot(iters, avg1, color=color, label='pattern 1', linewidth=3)
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].fill_between(iters, r1, r2, color=color, alpha=0.2)
    
    # 畫第二條線
    color = palette(1) # 設定顏色
    avg2 = np.mean(data2, axis=1) # 計畫平均
    std2 = np.std(data2, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg2, std2))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg2, std2)))
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].plot(iters, avg2, color=color, label='pattern 2', linewidth=3) # 畫平均線
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].fill_between(iters, r1, r2, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊
    
    # 圖片的格式設定
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].set_title(data_sheet_name[name], fontsize=20)
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].set_xlabel('Time (msec)', fontsize=20)
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].set_ylabel('Muscle Activation (%)', fontsize=20)
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].legend(loc='upper right')
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].axvline(x=0, ymin=0.0, ymax=1.0, color='b', linestyle='--', alpha=0.3)
    if len(raw_xspan[name]) > 1:
        for ii in range(int((len(raw_xspan[name])-1)/2)):
            axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].axvspan(raw_xspan[name][1+ii*2], raw_xspan[name][2+ii*2], facecolor='0.5', alpha=0.5)
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].legend(prop={"size":16})          
# fig.suptitle('肌肉活化程度時序圖', fontsize=36)
plt.subplots_adjust(
                    left=0.125,
                    bottom=-0.51,
                    right=1.3,
                    wspace=0.2,
                    hspace=0.2)
plt.show()
##############################################################################
##############################################################################
##############################################################################
# 畫變化量
# 計算變化量
fig, axs = plt.subplots(7,2, dpi=100, figsize=(24, 32))
for name in range(len(data_sheet_name)):
    print(name - 7*math.ceil((name-6)/7))
    print( math.floor(name/7))
    raw_data = pd.read_excel(r'F:/HsinYang/Thesis/Archery_move40_statistic_20221014.xlsx', sheet_name = data_sheet_name[name])

    # 讀資料
    # 原始值
    data1 = raw_data.iloc[:, 1:12] #設定資料欄位
    data2 = raw_data.iloc[:, 15:23] #設定資料欄位
    
    
    change_data1 = np.zeros((np.shape(data1)[0] - 1,np.shape(data1)[1] ))
    for i in range(np.shape(data1)[0]-1):
        change_data1[i, :] = (data1.iloc[i+1, :] - data1.iloc[i, :])/0.1 
    
    change_data2 = np.zeros((np.shape(data2)[0] - 1,np.shape(data2)[1] ))
    for i in range(np.shape(data1)[0]-1):
        change_data2[i, :] = (data2.iloc[i+1, :] - data2.iloc[i, :])/0.1 
    iters = list(range(-2400, 1000, 100)) #設定資料的x軸
    color = palette(0) # 設定顏色
    avg1 = np.mean(change_data1, axis=1) # 計算平均
    std1 = np.std(change_data1, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].plot(iters, avg1, color=color, label='pattern 1', linewidth=3)
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].fill_between(iters, r1, r2, color=color, alpha=0.2)
    
    # 畫第二條線
    color = palette(1) # 設定顏色
    avg2 = np.mean(change_data2, axis=1) # 計畫平均
    std2 = np.std(change_data2, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg2, std2))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg2, std2)))
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].plot(iters, avg2, color=color, label='pattern 2', linewidth=3) # 畫平均線
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].fill_between(iters, r1, r2, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊
    
    # 圖片的格式設定
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].set_title(data_sheet_name[name], fontsize=20)
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].set_xlabel('Time (msec)', fontsize=20)
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].set_ylabel('Muscle Activation Changing rate', fontsize=20)
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].legend(loc='upper right')
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].axvline(x=0, ymin=0.0, ymax=1.0, color='b', linestyle='--', alpha=0.3)
    if len(change_xspan[name]) > 1:
        for ii in range(int((len(change_xspan[name])-1)/2)):
            axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].axvspan(change_xspan[name][1+ii*2], change_xspan[name][2+ii*2], facecolor='0.5', alpha=0.5)
    axs[name - 7*math.ceil((name-6)/7), math.floor(name/7)].legend(prop={"size":16})        
# fig.suptitle('肌肉活化程度變化量時序圖', fontsize=36)
plt.subplots_adjust(
                    left=0.125,
                    bottom=-0.51,
                    right=1.3,
                    wspace=0.2,
                    hspace=0.2)
plt.show()

##############################################################################
##############################################################################
##############################################################################
#前臂變化量
forearm_sheet_name = ['R_Extensor', 'R_Flexor']
forearm_xspan = [[0, 25.5, 26.5], [1, 25.5, 27.5], [2], [3]]
for x in range(len(forearm_xspan)):
    for xx in range(len(forearm_xspan[x])):
        forearm_xspan[x][xx] = forearm_xspan[x][xx]*100 - 2500

fig, axs = plt.subplots(2,2, dpi=100, figsize=(24, 12))

for name in range(len(forearm_sheet_name)):

    raw_data = pd.read_excel(r'F:/HsinYang/Thesis/Archery_move40_statistic_20221014.xlsx', sheet_name = forearm_sheet_name[name])

    # 讀資料
    # 原始值
    data1 = raw_data.iloc[:, 1:12] #設定資料欄位
    data2 = raw_data.iloc[:, 15:23] #設定資料欄位
    
    
    change_data1 = np.zeros((np.shape(data1)[0] - 1,np.shape(data1)[1] ))
    for i in range(np.shape(data1)[0]-1):
        change_data1[i, :] = (data1.iloc[i+1, :] - data1.iloc[i, :])/0.1 
    
    change_data2 = np.zeros((np.shape(data2)[0] - 1,np.shape(data2)[1] ))
    for i in range(np.shape(data1)[0]-1):
        change_data2[i, :] = (data2.iloc[i+1, :] - data2.iloc[i, :])/0.1 

    color = palette(0) # 設定顏色
    iters = list(range(-2500, 1000, 100))
    iters_1 = list(range(-2400, 1000, 100))#設定資料的x軸
    # 畫第一條線
    avg1 = np.mean(data1, axis=1) # 計算平均
    std1 = np.std(data1, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
    axs[0, name].plot(iters, avg1, color=color, label='pattern 1', linewidth=3)
    axs[0, name].fill_between(iters, r1, r2, color=color, alpha=0.2)
    
    # 畫第二條線
    color = palette(1) # 設定顏色
    avg2 = np.mean(data2, axis=1) # 計畫平均
    std2 = np.std(data2, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg2, std2))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg2, std2)))
    axs[0, name].plot(iters, avg2, color=color, label='pattern 2', linewidth=3) # 畫平均線
    axs[0, name].fill_between(iters, r1, r2, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊
    
    # 圖片的格式設定
    axs[0, name].set_title(data_sheet_name[name], fontsize=20)
    axs[0, name].set_xlabel('Time (msec)', fontsize=20)
    axs[0, name].set_ylabel('Muscle Activation (%)', fontsize=20)
    axs[0, name].legend(loc='upper right')
    axs[0, name].axvline(x=0, ymin=0.0, ymax=1.0, color='b', linestyle='--', alpha=0.3)
    if len(raw_xspan[name]) > 1:
        for ii in range(int((len(forearm_xspan[name])-1)/2)):
            axs[0, name].axvspan(forearm_xspan[name][1+ii*2], forearm_xspan[name][2+ii*2], facecolor='0.5', alpha=0.5)
    axs[0, name].legend(prop={"size":16})
    #################################################################
    # 變化率
    #################################################################
    color = palette(0)
    avg11 = np.mean(change_data1, axis=1) # 計算平均
    std11 = np.std(change_data1, axis=1) # 計算標準差
    r11 = list(map(lambda x: x[0]-x[1], zip(avg11, std11))) # 畫一個標準差以內的線
    r21 = list(map(lambda x: x[0]+x[1], zip(avg11, std11)))
    axs[1, name].plot(iters_1, avg11, color=color, label='pattern 1', linewidth=3)
    axs[1, name].fill_between(iters_1, r11, r21, color=color, alpha=0.2)
    
    # 畫第二條線
    color = palette(1) # 設定顏色 
    avg22 = np.mean(change_data2, axis=1) # 計畫平均
    std22 = np.std(change_data2, axis=1) # 計算標準差
    r12 = list(map(lambda x: x[0]-x[1], zip(avg22, std22))) # 畫一個標準差以內的線
    r22 = list(map(lambda x: x[0]+x[1], zip(avg22, std22)))
    axs[1, name].plot(iters_1, avg22, color=color, label='pattern 2', linewidth=3, markersize=12 ) # 畫平均線
    axs[1, name].fill_between(iters_1, r12, r22, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊
    
    # 圖片的格式設定
    axs[1, name].set_title(data_sheet_name[name], fontsize=20)
    axs[1, name].set_xlabel('Time (msec)', fontsize=20)
    axs[1, name].set_ylabel('Muscle Activation Changing Rate', fontsize=20)
    axs[1, name].legend(loc='upper right')
    axs[1, name].axvline(x=0, ymin=0.0, ymax=1.0, color='b', linestyle='--', alpha=0.3)

    if len(forearm_xspan[name+1]) > 1:
        for ii in range(int((len(forearm_xspan[name])-1)/2)):
            print(name)
            axs[1, name].axvspan(forearm_xspan[name+1][1+ii*2], forearm_xspan[name+1][2+ii*2], facecolor='0.5', alpha=0.5)
    axs[1, name].legend(prop={"size":16})
# fig.suptitle('下手臂_肌肉活化程度時序圖', fontsize=36)
plt.subplots_adjust(
                    left=0.125,
                    bottom=-0.51,
                    right=1.3,
                    wspace=0.2,
                    hspace=0.2)
plt.show()
















