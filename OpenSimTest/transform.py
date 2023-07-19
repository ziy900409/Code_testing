# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:08:48 2023

@author: Hsin.YH.Yang
"""
import pandas as pd
import numpy as np

# %% 座標軸從 Z 朝上，改成 Y 朝上

data_path = r"C:\Users\Public\BenQ\OpenSim\Models\MoBL-ARMS Upper Extremity Model\BenQtest\S01_Blink1_1-EC2 Wight.trc"
example_data = pd.read_csv(data_path, header = None, delimiter='	', skiprows=6, encoding='UTF-8', on_bad_lines='skip')

data = pd.DataFrame(np.zeros(np.shape(example_data.iloc[:200, :80])))
data.iloc[6:200, :2] = example_data.iloc[6:200, :2]

for i in range(int((np.shape(example_data.iloc[6:200, :80])[1]-2)/3)):
    data.iloc[6:200, 2 + 3*i] = example_data.iloc[6:200, 2 + 3*i] # x
    data.iloc[6:200, 3 + 3*i] = example_data.iloc[6:200, 4 + 3*i] # x
    data.iloc[6:200, 4 + 3*i] = -example_data.iloc[6:200, 3 + 3*i] # x
    
# %% 計算 Elbow med 之距離

# read tpose
tpose_path = r"E:\Motion Analysis\U3 Research\S03\S03_Tpose_4-EC2 Wight_Elbow.trc"
tpose_data = pd.read_csv(tpose_path, header = None, delimiter='	', skiprows=6, encoding='UTF-8', on_bad_lines='skip')


# 原點設定為 UA1
# 建立座標軸， 原點為 UA1
## 1. 長軸 UA1 -> Elbow.Lat
## 2. Axis 2 : plane point UA3， 垂直軸 : UA1 -> Elbow.Lat Cross UA1 -> UA3
## 3. Axis 3 : 
    
# %% 
raw_data = pd.read_excel(r"C:\Users\hsin.yh.yang\Downloads\01_01.xlsx", header=None)

data = raw_data.iloc[212:529, :]

columns_name = []
for i in range(len(raw_data.columns)):
    columns_name.append(str(raw_data.iloc[0, i] + '_' + raw_data.iloc[1, i]))
    

# 計算手腕與手臂夾角 : 利用 C1, F1 (中點), F2
# 食指 MCP2 B2, B1, F1
# PIP2 : B3, B2, B1
# DIP2 : B4, B3, B2 
