# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 10:49:44 2022

@author: user
"""


import pandas as pd
import spm1d
import numpy as np
import matplotlib.pyplot as plt


# 顯示中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

palette = plt.get_cmap('Set1')
    

# %%
def compare_fig(data1, data2):
    fig, axs = plt.subplots(figsize = (12, 8))
    i = 0 # 更改顏色用
    
    # 確定繪圖順序與位置
    # x, y = muscle - n*math.floor(abs(muscle)/n), math.floor(abs(muscle)/n) 
    iters = list(np.linspace(-1, 0.5, 151))
    # 設定計算資料
    avg1 = np.mean(data1, axis=1) # 計算平均
    std1 = np.std(data1, axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
    axs.plot(iters, avg1, color="b", linewidth=3)
    axs.fill_between(iters, r1, r2, color="b", alpha=0.2)
    
    
    avg1_v2 = np.mean(data2, axis=1) # 計算平均
    std1_v2 = np.std(data2, axis=1) # 計算標準差
    r1_v2 = list(map(lambda x: x[0]-x[1], zip(avg1_v2, std1_v2))) # 畫一個標準差以內的線
    r2_v2 = list(map(lambda x: x[0]+x[1], zip(avg1_v2, std1_v2)))
    axs.plot(iters, avg1_v2, color="r", linewidth=3)
    axs.fill_between(iters, r1_v2, r2_v2, color="r", alpha=0.2)
    return fig
            








# %%
time_value = np.linspace(-1, 0.5, 151)

origin_path = r"D:\BenQ_Project\python\MingData\扳機力量\力量\01-2.trigger force 原始值(綜合)(未達顯著).xlsx"
raw_data = pd.read_excel(origin_path, sheet_name='減起點', header=0, skiprows =2)
labels = 14

time_tick = np.linspace(-1, 0.5, 151)
time_tick = np.around(time_tick, decimals=1)

# %%
fig = compare_fig(raw_data.iloc[:, 1:13], raw_data.iloc[:, 16:30])
plt.tick_params(axis='x', labelsize=labels)
plt.tick_params(axis='y', labelsize=labels)
plt.xlim(-1, 0.5)
plt.show()

fig.savefig(r'D:\BenQ_Project\python\MingData\FingerForce\01-1原始力量平均值標準差時間圖_1.png', dpi =600)

fig1, ax = plt.subplots(1)
fig1.set_size_inches(12, 8)
parameters = {'axes.labelsize': 20,
              'axes.titlesize': 30}
plt.rcParams.update(parameters)
spm1d.plot.plot_mean_sd(np.transpose(raw_data.iloc[:, 1:13]), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='優秀選手')
plt.axvline(x=100, c='r', ls='--', lw=1)
# spm1d.plot.plot_mean_sd(Y1, label='Normal')
spm1d.plot.plot_mean_sd(np.transpose(raw_data.iloc[:, 16:30]), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='一般選手')
# plt.xlabel('時間')
# plt.ylabel('扣壓扳機力量(gw)')
# plt.title('原始力量 平均值-標準差圖')
# plt.xticks(np.transpose(raw_data.iloc[1, 1:13]),time_value)
# plt.legend()
ax.set_xticklabels(time_tick)
plt.tick_params(axis='x', labelsize=labels)
plt.tick_params(axis='y', labelsize=labels)
plt.show()
fig1.savefig(r'D:\BenQ_Project\python\MingData\FingerForce\01-1原始力量平均值標準差時間圖_1.png', dpi =600)


# %%
normal_path = r"D:\BenQ_Project\python\MingData\扳機力量\力量\04-2.trigger force 每發減初始值 除以 個人峰值減初始值(綜合)(0.14-0.5顯著).xlsx"
normal_data = pd.read_excel(normal_path, sheet_name='減起點除以峰值', header=0, skiprows =2)

fig2 = plt.figure(2)
fig2.set_size_inches(12, 8)
parameters = {'axes.labelsize': 20,
          'axes.titlesize': 30}
plt.rcParams.update(parameters)
spm1d.plot.plot_mean_sd(np.transpose(normal_data.iloc[:, 1:13]), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='優秀選手')
plt.axvline(x=100, c='r', ls='--', lw=1)
# spm1d.plot.plot_mean_sd(Y1, label='Normal')
spm1d.plot.plot_mean_sd(np.transpose(normal_data.iloc[:, 16:30]), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='一般選手')
# plt.xlabel('時間')
# plt.ylabel('標準化力量(比值)')
# plt.title('標準化力量 平均值-標準差圖')
p = plt.axvspan(114, 150, facecolor='0.5', alpha=0.5)
plt.tick_params(axis='x', labelsize=labels)
plt.tick_params(axis='y', labelsize=labels)

# plt.legend()
plt.show()
fig2.savefig(r'D:\BenQ_Project\python\MingData\FingerForce\01-2標準化力量-平均值標準差時間圖_1.png', dpi =600)

d = np.transpose(normal_data.iloc[:, 1:13].values) - np.transpose(normal_data.iloc[:, 16:28].values)    # pairwise differences
sd = d.std(axis=0, ddof=1)   # standard deviation of the pairwise differences
print( sd )

Y1 = np.transpose(normal_data.iloc[1:, 1:13].values)
Y2 = np.transpose(normal_data.iloc[1:, 16:30].values)

# %%
normal_path_2 = r"D:\BenQ_Project\python\MingData\扳機力量\變異係數\05-2.trigger force 原始值 變異係數(綜合)(未達顯著).xlsx"
normal_data_2 = pd.read_excel(normal_path_2, sheet_name='減起點', header=0, skiprows =2)

plt.figure(3)
fig3 = plt.figure(3)
fig3.set_size_inches(12, 8)
parameters = {'axes.labelsize': 20,
          'axes.titlesize': 30}
plt.rcParams.update(parameters)
spm1d.plot.plot_mean_sd(np.transpose(normal_data_2.iloc[:, 1:13]), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='優秀選手')
plt.axvline(x=100, c='r', ls='--', lw=1)
# spm1d.plot.plot_mean_sd(Y1, label='Normal')
spm1d.plot.plot_mean_sd(np.transpose(normal_data_2.iloc[:, 16:30]), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='一般選手')
# plt.xlabel('時間')
# plt.ylabel('扣壓扳機力量(gw)')
# plt.title('原始力量變異係數 平均值-標準差圖')
# plt.legend()
plt.tick_params(axis='x', labelsize=labels)
plt.tick_params(axis='y', labelsize=labels)
plt.show()
fig3.savefig(r'D:\BenQ_Project\python\MingData\FingerForce\02-1原始力量變異係數-平均值標準差時間圖_1.png', dpi =600)

# %%
normal_path_1 = r"D:\BenQ_Project\python\MingData\扳機力量\變異係數\06-2.trigger force 個人的每發減掉最小值 變異係數(綜合)(-0.03~-0.02顯著).xlsx"
normal_data_1 = pd.read_excel(normal_path_1, sheet_name='減起點', header=0, skiprows =2)

plt.figure(4)
fig4 = plt.figure(4)
fig4.set_size_inches(12, 8)
parameters = {'axes.labelsize': 20,
          'axes.titlesize': 30}
plt.rcParams.update(parameters)
spm1d.plot.plot_mean_sd(np.transpose(normal_data_1.iloc[:, 1:13]), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='優秀選手')
plt.axvline(x=100, c='r', ls='--', lw=1)
# spm1d.plot.plot_mean_sd(Y1, label='Normal')
spm1d.plot.plot_mean_sd(np.transpose(normal_data_1.iloc[:, 16:30]), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='一般選手')
# plt.xlabel('時間')
# plt.ylabel('標準化力量(比值)')
# plt.title('標準化力量變異係數 平均值-標準差圖')
p = plt.axvspan(96, 98, facecolor='0.5', alpha=0.5)
# plt.legend()
plt.tick_params(axis='x', labelsize=labels)
plt.tick_params(axis='y', labelsize=labels)
plt.show()
fig4.savefig(r'D:\BenQ_Project\python\MingData\FingerForce\02-2標準化力量變異係數-平均值標準差時間圖_1.png', dpi =600)

# %%
# ----------------------new code--------------------------
diff_path = r"D:\BenQ_Project\python\MingData\扳機力量\每秒變化量\06-1.trigger force 原始值每秒變化量(綜合).xlsx"
diff_data = pd.read_excel(diff_path, sheet_name='每秒變化量', header=0, skiprows =2)

fig5 = plt.figure(5)
fig5.set_size_inches(12, 8)
parameters = {'axes.labelsize': 20,
          'axes.titlesize': 30}
plt.rcParams.update(parameters)
spm1d.plot.plot_mean_sd(np.transpose(diff_data.iloc[:, 1:13]), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='優秀選手')
plt.axvline(x=100, c='r', ls='--', lw=1)
# spm1d.plot.plot_mean_sd(Y1, label='Normal')
spm1d.plot.plot_mean_sd(np.transpose(diff_data.iloc[:, 16:30]), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='一般選手')
# plt.xlabel('時間')
# plt.ylabel('每秒力量變化量(gw/s)')
# plt.title('原始力量 每秒變化量')
# plt.xticks(np.transpose(raw_data.iloc[:, 1]),time_value)
p = plt.axvspan(8, 9, facecolor='0.5', alpha=0.5)
p1 = plt.axvspan(14, 15, facecolor='0.5', alpha=0.5)
p1 = plt.axvspan(92, 93, facecolor='0.5', alpha=0.5)
p2 = plt.axvspan(103, 107, facecolor='0.5', alpha=0.5)
p3 = plt.axvspan(110, 113, facecolor='0.5', alpha=0.5)
p4 = plt.axvspan(114, 115, facecolor='0.5', alpha=0.5)
p5 = plt.axvspan(132, 136, facecolor='0.5', alpha=0.5)
# plt.legend()
plt.tick_params(axis='x', labelsize=labels)
plt.tick_params(axis='y', labelsize=labels)
plt.show()
fig5.savefig(r'D:\BenQ_Project\python\MingData\FingerForce\03-1原始力量每秒變化量-平均值標準差時間圖_2.png', dpi =600)
# %%

diff_path_1 = r"D:\BenQ_Project\python\MingData\扳機力量\每秒變化量\06-2.trigger force 每發減初始值 除以 個人峰值減初始值 每秒變化量(綜合).xlsx"
diff_data_1 = pd.read_excel(diff_path_1, sheet_name='每秒變化量', header=0, skiprows =2)

fig6 = plt.figure(6)
fig6.set_size_inches(12, 8)
parameters = {'axes.labelsize': 20,
          'axes.titlesize': 30}
plt.rcParams.update(parameters)
spm1d.plot.plot_mean_sd(np.transpose(diff_data_1.iloc[:, 1:13]), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='優秀選手')
plt.axvline(x=100, c='r', ls='--', lw=1)
# spm1d.plot.plot_mean_sd(Y1, label='Normal')
spm1d.plot.plot_mean_sd(np.transpose(diff_data_1.iloc[:, 16:30]), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='一般選手')
# plt.xlabel('時間')
# plt.ylabel('每秒力量變化量(比值/s)')
# plt.title('標準化力量 每秒變化量')
# plt.xticks(np.transpose(raw_data.iloc[1, 1:13]),time_value)
p1 = plt.axvspan(8, 9, facecolor='0.5', alpha=0.5)
p2 = plt.axvspan(71, 72, facecolor='0.5', alpha=0.5)
p3 = plt.axvspan(104, 115, facecolor='0.5', alpha=0.5)
p3 = plt.axvspan(132, 136, facecolor='0.5', alpha=0.5)
# plt.legend()
plt.tick_params(axis='x', labelsize=labels)
plt.tick_params(axis='y', labelsize=labels)
plt.show()
fig6.savefig(r'D:\BenQ_Project\python\MingData\FingerForce\03-2標準化力量每秒變化量-平均值標準差時間圖_3.png', dpi =600)