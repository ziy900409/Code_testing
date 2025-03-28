# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 08:45:10 2024

@author: Hsin.YH.Yang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import sys
import os

# 獲取目前腳本的絕對路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
# 計算 function 目錄的相對位置
function_dir = os.path.join(current_dir, "..", "function")
# 加入 sys.path
sys.path.append(function_dir)

# 讀取 CSV 文件，並指定欄位名稱
file_path = "IndexData.csv"  # 請修改成你的文件路徑
df = pd.read_csv(file_path, header=None, names=['X', 'Y', 'Z'])

# 取得 Z 軸數據
z_values = df["Z"].values

# 找到 Z 軸的局部最小值索引
order = 5  # 設定區間大小，可根據數據調整
local_minima_idx = argrelextrema(z_values, np.less, order=order)[0]

# 計算 Z 軸的平均值
z_mean = np.mean(z_values)

# 設定閾值：小於 (平均值 - 0.05) 的點才視為局部最小值
threshold = z_mean - 0.05

# 篩選符合閾值條件的局部最小值
filtered_minima_idx = [idx for idx in local_minima_idx if z_values[idx] < threshold]

# 繪製 Z 軸數據與篩選後的局部最小值
plt.figure(figsize=(12, 5))
plt.plot(df.index, z_values, label='Z-Axis', color='b', alpha=0.7)
plt.scatter(filtered_minima_idx, z_values[filtered_minima_idx], color='r', label='Filtered Local Minima', zorder=3)
plt.axhline(threshold, color='g', linestyle='--', label=f'Threshold ({threshold:.2f})')
plt.xlabel("Frame")
plt.ylabel("Z Value")
plt.title("Filtered Local Minima of Z-Axis")
plt.legend()
plt.show()

# 輸出篩選後的局部最小值數據
filtered_minima_data = pd.DataFrame({
    "Frame": filtered_minima_idx,
    "Z Value": z_values[filtered_minima_idx]
})

# 存成 CSV
filtered_minima_data.to_csv("Filtered_Local_Minima.csv", index=False)

# 顯示篩選後的數據
print(filtered_minima_data.head())


# 讀取 CSV 文件，並指定欄位名稱
file_path = "IndexData.csv"  # 請修改成你的文件路徑
df = pd.read_csv(file_path, header=None, names=['X', 'Y', 'Z'])

# 取得 Z 軸數據
z_values = df["Z"].values

# 找到 Z 軸的局部最小值索引
order = 5  # 設定區間大小，可根據數據調整
local_minima_idx = argrelextrema(z_values, np.less, order=order)[0]

# 計算 Z 軸的平均值
z_mean = np.mean(z_values)

# 設定閾值：小於 (平均值 - 0.05) 的點才視為局部最小值
threshold = z_mean - 0.05

# 篩選符合閾值條件的局部最小值
filtered_minima_idx = [idx for idx in local_minima_idx if z_values[idx] < threshold]

# 取得局部最小值對應的 X, Y 座標
filtered_x = df.loc[filtered_minima_idx, "X"]
filtered_y = df.loc[filtered_minima_idx, "Y"]

# 逆時針旋轉 90° (交換 X, Y 軸 並取負號)
plt.figure(figsize=(8, 8))
plt.scatter(df["Y"], -df["X"], c=df.index, cmap="viridis", alpha=0.7, s=5, label="Trajectory")
plt.scatter(filtered_y, -filtered_x, color="red", s=20, label="Local Minima", zorder=3)  # 標記局部最小值
plt.colorbar(label="Frame Index")
plt.xlabel("Y Axis (Rotated)")
plt.ylabel("X Axis (Rotated)")
plt.title("Rotated 2D Trajectory of Index Finger with Local Minima (90° Counterclockwise)")
plt.legend()
plt.show()
