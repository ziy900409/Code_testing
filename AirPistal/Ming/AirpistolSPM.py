# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:55:58 2025

@author: Hsin.YH.Yang
"""

import spm1d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
# 顯示中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

palette = plt.get_cmap('Set1')

# %%
file_path = [r"D:\BenQ_Project\python\MingData\SPM Force time-series\01-2.trigger force 原始值(綜合).xlsx",
             r"D:\BenQ_Project\python\MingData\SPM Force time-series\04-2.trigger force 每發減初始值 除以 個人峰值減初始值(綜合).xlsx",
             r"D:\BenQ_Project\python\MingData\SPM Force time-series\05-2.trigger force 原始值 變異係數(綜合).xlsx",
             r"D:\BenQ_Project\python\MingData\SPM Force time-series\06-1.trigger force 原始值每秒變化量(綜合).xlsx",
             r"D:\BenQ_Project\python\MingData\SPM Force time-series\06-2.trigger force 個人的每發減掉最小值 變異係數(綜合).xlsx",
             r"D:\BenQ_Project\python\MingData\SPM Force time-series\06-2.trigger force 每發減初始值 除以 個人峰值減初始值 每秒變化量(綜合).xlsx",
             r"D:\BenQ_Project\python\MingData\SPM Force time-series\(3秒5秒一組)04-2.trigger force 每發減初始值 除以 個人峰值減初始值(綜合).xlsx",
             r"D:\BenQ_Project\python\MingData\SPM Force time-series\(3秒5秒一組)04-2.trigger force 每發減初始值 除以 個人峰值減初始值(綜合).xlsx",
             r"D:\BenQ_Project\python\MingData\SPM Force time-series\(3秒5秒一組)06-2.trigger force 個人的每發減掉最小值 變異係數(綜合).xlsx",
             r"D:\BenQ_Project\python\MingData\SPM Force time-series\(3秒5秒一組)06-2.trigger force 個人的每發減掉最小值 變異係數(綜合).xlsx",
             r"D:\BenQ_Project\python\MingData\SPM Force time-series\(3秒5秒一組)06-2.trigger force 每發減初始值 除以 個人峰值減初始值 每秒變化量(綜合).xlsx",
             r"D:\BenQ_Project\python\MingData\SPM Force time-series\(3秒5秒一組)06-2.trigger force 每發減初始值 除以 個人峰值減初始值 每秒變化量(綜合).xlsx"
             
             ]
file_sheet = ["減起點",
              "減起點除以峰值",
              "減起點",
              "每秒變化量",
              "減起點",
              "每秒變化量",
              "3秒一組T",
              "5秒一組T",
              "3秒一組T",
              "5秒一組T",
              "3秒一組T",
              "5秒一組T"
              ]
file_skiprow = [2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3]

# %%

for i in range(len(file_path)):
    raw_data = pd.read_excel(file_path[i],
                             sheet_name=file_sheet[i],
                             header=0,
                             skiprows =file_skiprow[i])
    fig_title = os.path.split(file_path[i])
    
    t  = spm1d.stats.ttest2(raw_data.iloc[:, 1:13].T,
                            raw_data.iloc[:, 16:30].T,
                            equal_var=False)
    ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
    
    plt.close('all')
    ### plot mean and SD:
    fig,AX = plt.subplots( 1, 2, figsize=(8, 3.5) )
    ax     = AX[0]
    plt.sca(ax)
    spm1d.plot.plot_mean_sd(raw_data.iloc[:, 1:13].T,
                            linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b',  label='優秀選手')
    spm1d.plot.plot_mean_sd(raw_data.iloc[:, 16:30].T,
                            linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='一般選手')
    ax.axhline(y=0, color='k', linestyle=':')
    ax.set_xlabel('Time (%)')
    ax.set_ylabel('Plantar arch angle  (deg)')
    ax.set_title(file_sheet[i])
    ### plot SPM results:
    ax     = AX[1]
    plt.sca(ax)
    ti.plot()
    ti.plot_threshold_label(fontsize=8)
    ti.plot_p_values(size=10, offset_all_clusters=(0,0.9))
    ax.set_xlabel('Time (%)')
    plt.title(fig_title[-1])
    plt.tight_layout()
    plt.show()