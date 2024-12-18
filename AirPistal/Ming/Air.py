# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 08:58:41 2022

@author: user
"""

import pandas as pd
import spm1d
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress, pearsonr
# 顯示中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
# %%
# origin_path = r'F:\HsinYang\NTSU\TenLab\Shooting\MingData\FingerForce\01-2.trigger force 原始值(綜合).xlsx'
origin_path = r"E:\Hsin\NTSU_lab\AirPistol\04-2.trigger force 每發減初始值 除以 個人峰值減初始值(綜合).xlsx"
raw_data = pd.read_excel(origin_path, sheet_name='減起點除以峰值', header=0, skiprows =2)

plt.figure(1, figsize=(6,4), dpi=300)
spm1d.plot.plot_mean_sd(np.transpose(raw_data.iloc[:, 1:11]), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='男選手')
plt.axvline(x=104, c='r', ls='--', lw=1)
# spm1d.plot.plot_mean_sd(Y1, label='Normal')
spm1d.plot.plot_mean_sd(np.transpose(raw_data.iloc[:, 14:25]), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='女選手')
plt.xlabel('時間 (s)')
plt.ylabel('扣壓扳機力量(gw)')
plt.title('原始力量 平均值-標準差圖')
plt.legend()
plt.show(dpi=300)

normal_path = r'F:\HsinYang\NTSU\TenLab\Shooting\MingData\FingerForce\04-2.trigger force 每發減初始值 除以 個人峰值減初始值(綜合).xlsx'
normal_data = pd.read_excel(normal_path, sheet_name='減起點除以峰值', header=0, skiprows =2)

plt.figure(2)
spm1d.plot.plot_mean_sd(np.transpose(normal_data.iloc[:, 1:13]), linecolor='b', facecolor=(0,1,1), edgecolor='c', label='優秀')
plt.axvline(x=104, c='r', ls='--', lw=1)
# spm1d.plot.plot_mean_sd(Y1, label='Normal')
spm1d.plot.plot_mean_sd(np.transpose(normal_data.iloc[:, 16:30]), linecolor='r', facecolor=(0.9290,0.6940,0.1250), edgecolor='y', label='一般')
plt.xlabel('時間 (s)')
plt.ylabel('標準化力量(比值)')
plt.title('標準化力量 平均值-標準差圖')
plt.legend()
plt.show()


# %%
palette = plt.get_cmap('Set1')
color = palette(1)
data1 = raw_data.iloc[:, 1:11]
data2 = raw_data.iloc[:, 14:25]

plt.figure(2, figsize=(6,4), dpi=300)

iters = list(np.linspace(-1, 0.5, len(data1)))

# 設定計算資料
avg1 = np.mean(data1, axis=1) # 計算平均
std1 = np.std(data1, axis=1) # 計算標準差
r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
plt.plot(iters, avg1, color=color, label='男選手', linewidth=3)
plt.fill_between(iters, r1, r2, color=color, alpha=0.2)
# 找所有數值的最大值，方便畫括弧用
yy = max(r2)
# 畫第二條線
color = palette(0) # 設定顏色
avg2 = np.mean(data2, axis=1) # 計畫平均
std2 = np.std(data2, axis=1) # 計算標準差
r1 = list(map(lambda x: x[0]-x[1], zip(avg2, std2))) # 畫一個標準差以內的線
r2 = list(map(lambda x: x[0]+x[1], zip(avg2, std2)))
# 找所有數值的最大值，方便畫括弧用
yy = max([yy, max(r2)])
plt.plot(iters, avg2, color=color, label='女選手', linewidth=3) # 畫平均線
plt.fill_between(iters, r1, r2, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊
# 圖片的格式設定
plt.title("男女選手_標準化扣壓扳機力量", fontsize=12)
plt.legend(loc="upper left") # 圖例位置
plt.xlim(-1, 0.5)  # X軸從0開始
x_ticks = np.arange(-1, 0.6, 0.1)  # np.arange(起點, 終點+0.1, 間隔)
plt.xticks(x_ticks)
ax = plt.gca()  # 取得當前軸
ax.spines['top'].set_visible(False)  # 移除上邊界線
ax.spines['right'].set_visible(False)  # 移除左邊界線
  # Y軸從0開始
# 畫放箭時間
plt.axvline(x=0, color = 'darkslategray', linewidth=1, linestyle = '--')
plt.show()


# %%【第1個分頁】
# 1. 擊發前變化量-握力
raw_data = pd.read_excel(r"E:\Hsin\NTSU_lab\AirPistol\圖-標準力量-相關性.xlsx",
                         sheet_name="1.標準化力量")

data_x = raw_data["擊發前變化量"]
data_y = raw_data["右手最大握力"]


x_male = raw_data["擊發前變化量"][raw_data["性別"]=="男"]
y_male = raw_data["右手最大握力"][raw_data["性別"]=="男"]

x_female = raw_data["擊發前變化量"][raw_data["性別"]=="女"]
y_female = raw_data["右手最大握力"][raw_data["性別"]=="女"]

# %%【第1個分頁】 
# 2. 擊發前變化量-BMI握力
data_x = raw_data["擊發前變化量"]
data_y = raw_data["BMI握力"]


x_male = raw_data["擊發前變化量"][raw_data["性別"]=="男"]
y_male = raw_data["BMI握力"][raw_data["性別"]=="男"]

x_female = raw_data["擊發前變化量"][raw_data["性別"]=="女"]
y_female = raw_data["BMI握力"][raw_data["性別"]=="女"]
# %% 【第2個分頁】
# 3. 擊發前變化量-握力
raw_data = pd.read_excel(r"E:\Hsin\NTSU_lab\AirPistol\圖-標準力量-相關性.xlsx",
                         sheet_name="2.標準化力量除以握力")

data_x = raw_data["擊發前變化量"]
data_y = raw_data["右手最大握力"]


x_male = raw_data["擊發前變化量"][raw_data["性別"]=="男"]
y_male = raw_data["右手最大握力"][raw_data["性別"]=="男"]

x_female = raw_data["擊發前變化量"][raw_data["性別"]=="女"]
y_female = raw_data["右手最大握力"][raw_data["性別"]=="女"]
# %% 【第3個分頁】
# 4. 擊發前變化量-BMI握力

raw_data = pd.read_excel(r"E:\Hsin\NTSU_lab\AirPistol\圖-標準力量-相關性.xlsx",
                         sheet_name="3.標準化力量除以BMI握力")

data_x = raw_data["擊發前變化量"]
data_y = raw_data["BMI握力"]


x_male = raw_data["擊發前變化量"][raw_data["性別"]=="男"]
y_male = raw_data["BMI握力"][raw_data["性別"]=="男"]

x_female = raw_data["擊發前變化量"][raw_data["性別"]=="女"]
y_female = raw_data["BMI握力"][raw_data["性別"]=="女"]
# %%
palette = plt.get_cmap('Set1')
# 計算線性回歸（趨勢線）
all_slope, all_intercept, r_value, p_value, std_err = linregress(data_x, data_y)
male_slope, male_intercept, r_value, p_value, std_err = linregress(x_male, y_male)
female_slope, female_intercept, r_value, p_value, std_err = linregress(x_female, y_female)

# 計算相關係數
# correlation, _ = pearsonr(data_x, data_y)
# 統一 X 軸範圍
x_min = min(min(data_x), min(x_male), min(x_female))
x_max = max(max(data_x), max(x_male), max(x_female))
x_range = np.linspace(x_min, x_max, 100) 

# 繪製散佈圖
plt.figure(figsize=(8, 6), dpi=600)
# 繪製所有人
plt.scatter(data_x, data_y, color='blue')  # 散佈圖
plt.plot(data_x, all_slope * data_x + all_intercept, color='black', label='all')  # 趨勢線
# 繪製男生
plt.scatter(x_male, y_male, color=palette(1))  # 散佈圖
plt.plot(data_x, male_slope * data_x + male_intercept, color=palette(1), label='male')  # 趨勢線
# 繪製女生
plt.scatter(x_female, y_female, color=palette(0))  # 散佈圖
plt.plot(data_x, female_slope * data_x + female_intercept, color=palette(0), label='female')  # 趨勢線

# 標籤與標題
# plt.xlabel('X 軸')
# plt.ylabel('Y 軸')
plt.title('4. 擊發前變化量-BMI握力')
plt.legend()

# 顯示圖形
plt.show()





















