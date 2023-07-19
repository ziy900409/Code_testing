# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:04:11 2023

@author: Hsin.YH.Yang
"""

import pandas as pd
import spm1d
import numpy as np
import matplotlib.pyplot as plt
'''
1. 改成自己的檔案路徑
2. sheet_name改不同的excel分頁
參考網站
https://spm1d.org/doc/Stats1D/onetwosample.html#paired-t-test
'''
data = pd.read_excel(r"C:/Users/hsin.yh.yang/Downloads/dart velocity kinematic analysis.xlsx",
                     sheet_name="Dart  Y Velocity",
                     skiprows=1)

adv_Dart_X_Velocity = data.iloc[2:, 1:9].astype(float)
int_Dart_X_Velocity = data.iloc[2:, 10:18].astype(float)
plt.figure(figsize=(5,6))
plt.subplot(2,1,1)
spm1d.plot.plot_mean_sd(np.transpose(adv_Dart_X_Velocity.values),
                        linecolor='b',
                        facecolor=(0.7,0.7,1),
                        edgecolor='b',
                        label='advance')
spm1d.plot.plot_mean_sd(np.transpose(int_Dart_X_Velocity.values),
                        linecolor='r',
                        facecolor=(1,0.7,0.7),
                        edgecolor='r',
                        label='intermedia')
plt.legend()

plt.subplot(2,1,2)
t  = spm1d.stats.ttest2(np.transpose(adv_Dart_X_Velocity.values),
                        np.transpose(int_Dart_X_Velocity.values),
                        equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=False)
ti.plot()
plt.tight_layout()
# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.pyplot import MultipleLocator
# 設定繪圖格式與字體
plt.style.use('seaborn-white')
# 顯示輸入中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
# 設定顏色格式
# 參考 Qualitative: https://matplotlib.org/stable/tutorials/colors/colormaps.html
palette = pyplot.get_cmap('Set1')
# 設定文字格式

# # 讀資料
# data = pd.read_excel(r"D:\NTSU\TenLab\LinData\angle-100_.xlsx",
#                      skiprows=1) # 跳過列數
# data1 = data.iloc[:, 1:7] #設定資料欄位
# data2 = data.iloc[:, 11:17] #設定資料欄位
data1 = data.iloc[2:, 1:9].astype(float)
data2 = data.iloc[2:, 10:18].astype(float)
data1 = data1.dropna(axis=0) #丟棄NAN的值
data2 = data2.dropna(axis=0) #丟棄NAN的值
iters = list(range(len(data1))) #設定資料的x軸

# 設定圖片大小
fig, ax = plt.subplots(figsize=(20, 10))
# 畫第一條線
color = palette(0) # 設定顏色
avg1 = np.mean(data1, axis=1) # 計算平均
std1 = np.std(data1, axis=1) # 計算標準差
r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
plt.plot(iters, avg1, color=color, label='algo1', linewidth=3)
plt.fill_between(iters, r1, r2, color=color, alpha=0.2)

# 畫第二條線
color = palette(1) # 設定顏色
avg2 = np.mean(data2, axis=1) # 計畫平均
std2 = np.std(data2, axis=1) # 計算標準差
r1 = list(map(lambda x: x[0]-x[1], zip(avg2, std2))) # 畫一個標準差以內的線
r2 = list(map(lambda x: x[0]+x[1], zip(avg2, std2)))
plt.plot(iters, avg2, color=color, label='algo1', linewidth=3) # 畫平均線
plt.fill_between(iters, r1, r2, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊

# 圖片的格式設定
x_major_locator = MultipleLocator(10) # 設置x軸刻度為10的倍數
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.spines['right'].set_visible(False) #去掉右邊框
ax.spines['top'].set_visible(False) # 去掉上邊框
plt.title('圖片名稱', fontsize=28) # 設定圖片名稱與大小
plt.legend(loc='upper right', bbox_to_anchor = (0.9, 1)) # 圖例位置
ax.set_xlabel('x軸單位') # x軸名稱
ax.set_ylabel('y軸名稱') # y軸名稱
plt.xlim(0,100) # x軸刻度


