# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:04:47 2024

@author: Hsin.YH.Yang
"""
# %%
import pandas as pd
import os
import sys
sys.path.append(r"E:\Hsin\git\git\Code_testing\baseball")
# 將read_c3d function 加進現有的工作環境中
import BaseballFunction_20230516 as af
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# %% 0. parameter estting
folder_path = r"E:\Hsin\NTSU_lab\Baseball\Processing_Data"
data_sheet = ["Stage2", "Stage3"]

# %% 1. read staging file
staging_file = pd.read_excel(r"E:\Hsin\NTSU_lab\Baseball\motion分期肌電用_20240420.xlsx",
                             sheet_name='T2_memo3') # 改變要找stage2 or stage3
staging_file = staging_file.dropna(axis=0, thresh=14)
folder_file_list = os.listdir(folder_path)
all_file_list = []

for folder in folder_file_list: 
    file_list = af.Read_File(folder_path + "\\" + folder + "\data\motion",
                             ".xlsx",
                             subfolder=False)
    for file in file_list:
        if "_ed" in file:
            # print(file)
            all_file_list.append(file)

ture_file_path = []
for file_name in staging_file["EMG_File"]:
    # print(file_name)
    for file_path in all_file_list:
        # print(file_path)
        if file_name in file_path:
            print(file_path)
            ture_file_path.append(file_path)
            
# %% truncate data
'''
1. 讀取資料，分別為 stage2, stage3
2. 內插成101個點
'''

muscle_data = np.zeros([len(ture_file_path), 150, 6])

for file in range(len(ture_file_path)):
    for sheet in data_sheet:
        emg_data = pd.read_excel(ture_file_path[file], sheet_name=sheet)
        print(ture_file_path[file])
        for i in range(np.shape(emg_data)[1] - 1): # 減去時間欄位
            # 內插函數
            x = emg_data.iloc[:, 0] # time
            y = emg_data.iloc[:, i+1]
            f = interp1d(x, y, kind='cubic')
            
            if sheet == "Stage2":
                x_new = np.linspace(emg_data.iloc[0, 0], emg_data.iloc[-1, 0], 100)
                y_new = f(x_new)
                muscle_data[file, :100, i] = y_new
            elif sheet == "Stage3":
                x_new = np.linspace(emg_data.iloc[0, 0], emg_data.iloc[-1, 0], 50)
                y_new = f(x_new)
                muscle_data[file, 100:, i] = y_new

arr_muscle_data = np.zeros([6, 150, len(ture_file_path)])
for i in range(np.shape(arr_muscle_data)[0]):
    for ii in range(np.shape(arr_muscle_data)[2]):
        arr_muscle_data[i, :, ii] = muscle_data[ii, :, i]

# 修改儲存檔名
# save_file_name = r"E:\Hsin\NTSU_lab\Baseball\EMGdata_processing_T2_20230805.xlsx"
save_file_name = r"E:\Hsin\NTSU_lab\Baseball\EMGdata_processing_T2_20240516.xlsx"

with pd.ExcelWriter(save_file_name) as Writer:
    pd.DataFrame(arr_muscle_data[0, :, :]).to_excel(Writer, sheet_name="muscle1", index=False)
    pd.DataFrame(arr_muscle_data[1, :, :]).to_excel(Writer, sheet_name="muscle2", index=False)
    pd.DataFrame(arr_muscle_data[2, :, :]).to_excel(Writer, sheet_name="muscle3", index=False)
    pd.DataFrame(arr_muscle_data[3, :, :]).to_excel(Writer, sheet_name="muscle4", index=False)
    pd.DataFrame(arr_muscle_data[4, :, :]).to_excel(Writer, sheet_name="muscle5", index=False)
    pd.DataFrame(arr_muscle_data[5, :, :]).to_excel(Writer, sheet_name="muscle6", index=False)

# %% T1 快轉 VS 慢轉
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.pyplot import MultipleLocator
import math

# 設定繪圖格式與字體
plt.style.use('seaborn-white')
# 顯示輸入中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
# 設定顏色格式
# 參考 Qualitative: https://matplotlib.org/stable/tutorials/colors/colormaps.html
palette = pyplot.get_cmap('Set1')
# 設定文字格式
front1 = {
          'weight': 'normal',
          'size': 18
    }
# 設定圖片文字格式
parameters = {'axes.labelsize': 24, # x 和 y 標籤的字型大小
          'axes.titlesize': 28, # 設定軸標題的字型大小
          'xtick.labelsize': 18, # 設定x軸刻度標籤的字型大小
          'ytick.labelsize': 18} # 設定y軸刻度標籤的字型大小
plt.rcParams.update(parameters)

# 讀資料
# data = arr_muscle_data
excel_file = pd.ExcelFile(r"E:\Hsin\NTSU_lab\Baseball\EMGdata_processing_T1_20240516.xlsx")

# 获取所有分页（sheet）的名称
sheet_names = excel_file.sheet_names
t1_data = np.zeros([6, 150, len(ture_file_path)])


for name in range(len(sheet_names)):
  t1_data[name, :, :] = pd.read_excel(r"E:\Hsin\NTSU_lab\Baseball\EMGdata_processing_T1_20240516.xlsx",
                                      sheet_name = sheet_names[name])
data = t1_data
columns_name = ['Briceps Brachii','Triceps Brachii', 'Extensor Carpi Radialis',
                'Extensor Carpi Ulnaris', 'Flexor Carpi Radialis', 'Flexor Carpi Ulnaris']
n = int(math.ceil((np.shape(data)[0]) /2))
fig, axs = plt.subplots(n, 2, figsize = (10,12), sharex='col')
for i in range(np.shape(data)[0]):
    # 確定繪圖順序與位置
    x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
    color = palette(0) # 設定顏色
    
    data1 = pd.DataFrame(data[i, :, [5, 8, 9, 10, 12, 13, 14, 15, 17, 18]].T) #設定資料欄位 快轉
    data2 = pd.DataFrame(data[i, :, [0, 1, 2, 3, 4, 6, 7, 11, 16]].T) #設定資料欄位 慢轉
    data1 = data1.dropna(axis=0) #丟棄NAN的值
    data2 = data2.dropna(axis=0) #丟棄NAN的值
    iters = list(range(len(data1)))
    # 設定計算資料
    color = palette(0) # 設定顏色
    avg1 = np.mean(data1, axis=1) # 計算平均
    std1 = np.std(data1, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
    axs[x, y].plot(iters, avg1, color=color, label='快轉組', linewidth=3)
    axs[x, y].fill_between(iters, r1, r2, color=color, alpha=0.2)
    
    # 畫第二條線
    color = palette(1) # 設定顏色
    avg2 = np.mean(data2, axis=1) # 計畫平均
    std2 = np.std(data2, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg2, std2))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg2, std2)))
    axs[x, y].plot(iters, avg2, color=color, label='慢轉組', linewidth=3) # 畫平均線
    axs[x, y].fill_between(iters, r1, r2, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊
    # 圖片的格式設定
    axs[x, y].set_title(columns_name[i], fontsize=12)
    axs[x, y].legend(loc="upper left") # 圖例位置
    axs[x, y].grid(True, linestyle='-.')
    # 畫放箭時間
    axs[x, y].set_xlim(0, 150)
    axs[x, y].axvline(x=100, color = 'darkslategray', linewidth=1, linestyle = '--')
plt.suptitle(str("mean std cloud: "), fontsize=16)
plt.tight_layout()
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("time", fontsize = 14)
plt.ylabel("muscle activation (%)", fontsize = 14)
# plt.savefig(save, dpi=200, bbox_inches = "tight")
plt.show()
# %% spm T1 快轉 VS 慢轉
import spm1d
dataset    = spm1d.data.uv1d.t1.Random()
Y,mu       = dataset.get_data()
t  = spm1d.stats.ttest(Y, mu)  #mu is 0 by default
ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
ti.plot()

for i in range(np.shape(data)[0]):
    data1 = pd.DataFrame(data[i, :, [5, 8, 9, 10, 12, 13, 14, 15, 17, 18]]) #設定資料欄位 快轉
    data2 = pd.DataFrame(data[i, :, [0, 1, 2, 3, 4, 6, 7, 11, 16]]) #設定資料欄位 慢轉

    # t  = spm1d.stats.ttest2(data1, data2, equal_var=False)
    # ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
    # ti.plot()
    # plt.show()
    plt.figure(figsize=(5,6))
    plt.subplot(2,1,1)
    spm1d.plot.plot_mean_sd(data1,
                            linecolor='r',
                            facecolor=(1,0.7,0.7),
                            edgecolor='r',
                            label='快轉組')
    spm1d.plot.plot_mean_sd(data2,
                            linecolor='b',
                            facecolor=(0.7,0.7,1),
                            edgecolor='b',
                            label='慢轉組')
    plt.legend(loc = "upper left")

    plt.subplot(2,1,2)
    t  = spm1d.stats.ttest2(data1,
                            data2,
                            equal_var=False)
    ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
    ti.plot()
    plt.suptitle(str(columns_name[i]), fontsize=16)
    plt.tight_layout()


# %% T1 vs T2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.pyplot import MultipleLocator
import math

# read data
# read sheet name
# 读取 Excel 文件
excel_file = pd.ExcelFile(r"E:\Hsin\NTSU_lab\Baseball\EMGdata_processing_T1_20230805.xlsx")

# 获取所有分页（sheet）的名称
sheet_names = excel_file.sheet_names

t1_data = np.zeros([6, 202, len(ture_file_path)])
t2_data = np.zeros([6, 202, len(ture_file_path)])

for name in range(len(sheet_names)):
  t1_data[name, :, :] = pd.read_excel(r"E:\Hsin\NTSU_lab\Baseball\EMGdata_processing_T1_20230805.xlsx",
                                      sheet_name = sheet_names[name])
  t2_data[name, :, :] = pd.read_excel(r"E:\Hsin\NTSU_lab\Baseball\EMGdata_processing_T2_20230805.xlsx",
                                      sheet_name = sheet_names[name])
# 設定繪圖格式與字體
plt.style.use('seaborn-white')
# 顯示輸入中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
# 設定顏色格式
# 參考 Qualitative: https://matplotlib.org/stable/tutorials/colors/colormaps.html
palette = pyplot.get_cmap('Set1')
# 設定文字格式
front1 = {
          'weight': 'normal',
          'size': 18
    }
# 設定圖片文字格式
parameters = {'axes.labelsize': 24, # x 和 y 標籤的字型大小
          'axes.titlesize': 28, # 設定軸標題的字型大小
          'xtick.labelsize': 18, # 設定x軸刻度標籤的字型大小
          'ytick.labelsize': 18} # 設定y軸刻度標籤的字型大小
plt.rcParams.update(parameters)

# 讀資料
data = arr_muscle_data

columns_name = ['Briceps Brachii','Triceps Brachii', 'Extensor Carpi Radialis',
                'Extensor Carpi Ulnaris', 'Flexor Carpi Radialis', 'Flexor Carpi Ulnaris']
n = int(math.ceil((np.shape(arr_muscle_data)[0]) /2))
fig, axs = plt.subplots(n, 2, figsize = (10,12), sharex='col')
for i in range(np.shape(arr_muscle_data)[0]):
    # 確定繪圖順序與位置
    x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
    color = palette(0) # 設定顏色
    
    data1 = pd.DataFrame(t1_data[i, :, :]) #設定資料欄位
    data2 = pd.DataFrame(t2_data[i, :, :]) #設定資料欄位
    data1 = data1.dropna(axis=0) #丟棄NAN的值
    data2 = data2.dropna(axis=0) #丟棄NAN的值
    iters = list(range(len(data1)))
    # 設定計算資料
    color = palette(0) # 設定顏色
    avg1 = np.mean(data1, axis=1) # 計算平均
    std1 = np.std(data1, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
    axs[x, y].plot(iters, avg1, color=color, label='T1', linewidth=3)
    axs[x, y].fill_between(iters, r1, r2, color=color, alpha=0.2)
    
    # 畫第二條線
    color = palette(1) # 設定顏色
    avg2 = np.mean(data2, axis=1) # 計畫平均
    std2 = np.std(data2, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg2, std2))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg2, std2)))
    axs[x, y].plot(iters, avg2, color=color, label='T2', linewidth=3) # 畫平均線
    axs[x, y].fill_between(iters, r1, r2, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊
    # 圖片的格式設定
    axs[x, y].set_title(columns_name[i], fontsize=12)
    axs[x, y].legend(loc="upper left") # 圖例位置
    axs[x, y].grid(True, linestyle='-.')
    # 畫放箭時間
    # axs[x, y].set_xlim(-(release[0]), release[1])
    axs[x, y].axvline(x=100, color = 'darkslategray', linewidth=1, linestyle = '--')
plt.suptitle(str("mean std cloud: "), fontsize=16)
plt.tight_layout()
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("time (second)", fontsize = 14)
plt.ylabel("muscle activation (%)", fontsize = 14)
# plt.savefig(save, dpi=200, bbox_inches = "tight")
plt.show()

# %% spm T1 vs T2

t1_data = np.zeros([6, 202, len(ture_file_path)])
t2_data = np.zeros([6, 202, len(ture_file_path)])

for name in range(len(sheet_names)):
  t1_data[name, :, :] = pd.read_excel(r"E:\Hsin\NTSU_lab\Baseball\EMGdata_processing_T1_20230805.xlsx",
                                      sheet_name = sheet_names[name])
  t2_data[name, :, :] = pd.read_excel(r"E:\Hsin\NTSU_lab\Baseball\EMGdata_processing_T2_20230805.xlsx",
                                      sheet_name = sheet_names[name])
  
for i in range(np.shape(data)[0]):
    data1 = pd.DataFrame(t1_data[i, :, :].T) #設定資料欄位 T1
    data2 = pd.DataFrame(t2_data[i, :, :].T) #設定資料欄位 T2
    data1 = data1.dropna(axis=0) #丟棄NAN的值
    data2 = data2.dropna(axis=0) #丟棄NAN的值

    # t  = spm1d.stats.ttest2(data1, data2, equal_var=False)
    # ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
    # ti.plot()
    # plt.show()
    plt.figure(figsize=(5,6))
    plt.subplot(2,1,1)
    spm1d.plot.plot_mean_sd(data1,
                            linecolor='r',
                            facecolor=(1,0.7,0.7),
                            edgecolor='r',
                            label='T1')
    spm1d.plot.plot_mean_sd(data2,
                            linecolor='b',
                            facecolor=(0.7,0.7,1),
                            edgecolor='b',
                            label='T2')
    plt.legend(loc = "upper left")

    plt.subplot(2,1,2)
    t  = spm1d.stats.ttest2(data1,
                            data2,
                            equal_var=False)
    ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
    ti.plot()
    plt.suptitle(str(columns_name[i]), fontsize=16)
    plt.tight_layout()





