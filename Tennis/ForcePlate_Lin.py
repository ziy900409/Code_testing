"""
Created on Mon Jul 25 19:49:02 2022

@author: Hsin Yang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'F:\HsinYang\NTSU\TenLab\Tennis\ForcePlate_Lin\Trial00022.txt'


force_z = pd.DataFrame()
with open(file_path) as f:
    for line in f.readlines():
        s = line.split(',')
        s = pd.DataFrame(s)
        force_z = pd.concat([force_z, s.iloc[2]])
        
print(force_z)


force_z = pd.DataFrame(force_z, dtype='float') # 使用DataFrame將矩陣轉為float
force_z[force_z < 1]= np.nan # 將data<1的值都轉為NAN
force_z = force_z.dropna(how='all') # 去除所有NAN的列

force_z = force_z.reset_index(drop=True)
# 參數設定
sampling_rate = 1000 #採樣頻率 1000Hz
data_time = len(force_z)/sampling_rate
time_span = np.linspace(0, data_time, len(force_z))


plt.figure(1)
plt.plot(time_span, force_z.iloc[:, 0])
plt.show()
