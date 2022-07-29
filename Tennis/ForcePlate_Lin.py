# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 19:49:02 2022

@author: Hsin Yang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from detecta import detect_onset
import scipy.integrate as si
import scipy.optimize as op

file_path = r'F:\HsinYang\NTSU\TenLab\Tennis\ForcePlate_Lin\Trial00023.txt'
# 使用np.loadtxt讀 .txt
GRFv = np.loadtxt(file_path, delimiter=',')
force_z = GRFv[:, 2]
# force_z = pd.DataFrame()
# with open(file_path) as f:
#     for line in f.readlines():
#         s = line.split(',')
#         s = pd.DataFrame(s)
#         force_z = pd.concat([force_z, s.iloc[2]])
        
print(force_z)

# # 重設矩陣index，先將資料轉float
# force_z = pd.DataFrame(force_z, dtype='float') # 使用DataFrame將矩陣轉為float
# force_z = force_z.reset_index(drop=True)
# 平滑處理資料
mean_force_z = pd.DataFrame()
for i in range(int(len(force_z)/5)):
    mean_value = np.mean(force_z[5*i:5*i+5])
    mean_value = pd.Series(mean_value)
    mean_force_z = pd.concat([mean_force_z, mean_value], ignore_index=True) 


# 去除零值
res_data = mean_force_z.iloc[::-1] # 去除從後倒數回來的0值
a = (res_data.values!=0).argmax(axis=0) # 先反轉矩陣，再用argmax，返回第一個boolean最大值
cut_force_z = pd.DataFrame(mean_force_z.iloc[:int(2000-a-1) , 0])

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
peak_force = np.argmax(cut_force_z.values)
max_impact = (cut_force_z.iloc[peak_force] - cut_force_z.iloc[touch_time]) \
    /((peak_force-touch_time)/re_sampling_rate)

# 找下蹲期頂點
# 利用找onset找下蹲期
# https://nbviewer.org/github/BMClab/BMC/blob/master/notebooks/DetectOnset.ipynb
down_period = detect_onset(-cut_force_z,
                           np.mean(-cut_force_z.iloc[:50, 0])*0.9,
                           n_above=10, n_below=0, show=True)
# 利用下蹲期設定下蹲期的力量 (N)
force_down_period = cut_force_z.iloc[down_period[0][0]:down_period[0][1], 0]
# 找回下蹲期頂點的index
peak_down_period = down_period[0][0] + int(np.argmax([-force_down_period]))

# 設定體重
body_mass = np.mean(cut_force_z.iloc[down_period[0][0]-110:down_period[0][0]-100, 0])/9.8

# 設定推蹬期
acc_period = detect_onset(cut_force_z,
                          np.mean(cut_force_z.iloc[:50, 0])*1.05,
                          n_above=10, n_below=0, show=True)

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
threshold = np.percentile(acc_curve_fit_ploy(acc_curve_x), 75) #np.mean(acc_curve_fit(acc_curve_x), axis= 0)
acc_first_peak = detect_onset(acc_curve_fit_ploy(acc_curve_x),
                              threshold, n_above=10, n_below=0, show=True)


# 積分起跳衝量
area = si.quad(np.poly1d(parameter),  acc_curve_x[0], acc_curve_x[-1]) \
    - acc_curve_fit_ploy(acc_curve_x)[0]*(acc_curve_x[-1] - acc_curve_x[0])

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
plt.figure(1)
plt.plot(re_time_span, cut_force_z.iloc[:, 0])
plt.plot(peak_down_period/re_sampling_rate, cut_force_z.iloc[peak_down_period, 0], 'o')
plt.plot(peak_force/re_sampling_rate, cut_force_z.iloc[peak_force, 0], 'o')
plt.plot(touch_time/re_sampling_rate, cut_force_z.iloc[touch_time, 0], 'o')
plt.plot(acc_period[0]/re_sampling_rate, cut_force_z.iloc[acc_period[0], 0], 'o')
plt.axvline(x=slove[0], c='r', ls='--', lw=1)
plt.axvline(x=slove[1], c='r', ls='--', lw=1)
plt.show()

# 繪圖確認推蹬期與擬合曲線的契合程度
plt.figure(2)
plt.plot(acc_curve_x, acc_curve_y, 'b')
plt.plot(acc_curve_x, acc_curve_fit_ploy(acc_curve_x), 'r*')
# plt.plot(acc_curve_x, diff_acc_curve_poly_1(acc_curve_x))
# plt.plot(acc_curve_x, diff_acc_curve_poly_1(acc_curve_x), 'r*')
# plt.axhline(y=0, c='r', ls='--', lw=1)
plt.axvline(x=slove[0], c='r', ls='--', lw=1)
plt.axvline(x=slove[1], c='r', ls='--', lw=1)

plt.show()

# # 畫曲線擬合程度
# plt.figure(3)
# plt.plot(x)
# plt.ylim(0, 1200)
# plt.show()


