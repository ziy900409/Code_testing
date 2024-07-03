# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 08:58:41 2022

@author: user
"""

import pandas as pd
import spm1d
import numpy as np
import matplotlib.pyplot as plt
# 顯示中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

origin_path = r'F:\HsinYang\NTSU\TenLab\Shooting\MingData\FingerForce\01-2.trigger force 原始值(綜合).xlsx'
raw_data = pd.read_excel(origin_path, sheet_name='減起點', header=0, skiprows =2)

plt.figure(1)
spm1d.plot.plot_mean_sd(np.transpose(raw_data.iloc[:, 1:13]), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='Slow')
plt.axvline(x=104, c='r', ls='--', lw=1)
# spm1d.plot.plot_mean_sd(Y1, label='Normal')
spm1d.plot.plot_mean_sd(np.transpose(raw_data.iloc[:, 16:30]), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Fast')
plt.xlabel('時間 (s)')
plt.ylabel('扣壓扳機力量(gw)')
plt.title('原始力量 平均值-標準差圖')
plt.legend()
plt.show()

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









