# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:23:04 2022

@author: user
"""

import pandas as pd
import glob
import spm1d
import numpy as np
import matplotlib.pyplot as plt

# 顯示中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
Muscle = "L_Deltoid"

# 設定檔案路徑
# 第一組
File_path11 = "D:/NTSU/TenLab/Archery/Archery_20211118/ProcessingData/iMVC/S1"
File_name11 = glob.glob(File_path11 + "/*" + Muscle + "*timing*.xlsx")
File_path12 = "D:/NTSU/TenLab/Archery/Archery_20211118/ProcessingData/iMVC/S2"
File_name12 = glob.glob(File_path12 + "/*" + Muscle + "*timing*.xlsx")
File_path13 = "D:/NTSU/TenLab/Archery/Archery_20211118/ProcessingData/iMVC/S5"
File_name13 = glob.glob(File_path13 + "/*" + Muscle + "*timing*.xlsx")
File_path14 = "D:/NTSU/TenLab/Archery/Archery_20211118/ProcessingData/iMVC/S6"
File_name14 = glob.glob(File_path14 + "/*" + Muscle + "*timing*.xlsx")
File_path15 = "D:/NTSU/TenLab/Archery/Archery_20211118/ProcessingData/iMVC/S8"
File_name15 = glob.glob(File_path15 + "/*" + Muscle + "*timing*.xlsx")
File_path16 = "D:/NTSU/TenLab/Archery/Archery_20211118/ProcessingData/iMVC/S9"
File_name16 = glob.glob(File_path16 + "/*" + Muscle + "*timing*.xlsx")
File_path17 = "D:/NTSU/TenLab/Archery/Archery_20211118/ProcessingData/iMVC/S11"
File_name17 = glob.glob(File_path17 + "/*" + Muscle + "*timing*.xlsx")

File_path18 = "D:/NTSU/TenLab/Archery/Archery_20211124/ProcessingData/iMVC/S2"
File_name18 = glob.glob(File_path18 + "/*" + Muscle + "*timing*.xlsx")
File_path19 = "D:/NTSU/TenLab/Archery/Archery_20211124/ProcessingData/iMVC/S4"
File_name19 = glob.glob(File_path19 + "/*" + Muscle + "*timing*.xlsx")
File_path110 = "D:/NTSU/TenLab/Archery/Archery_20211124/ProcessingData/iMVC/S5"
File_name110 = glob.glob(File_path110 + "/*" + Muscle + "*timing*.xlsx")
File_path111 = "D:/NTSU/TenLab/Archery/Archery_20211124/ProcessingData/iMVC/S8"
File_name111 = glob.glob(File_path111 + "/*" + Muscle + "*timing*.xlsx")
File_path112 = "D:/NTSU/TenLab/Archery/Archery_20211124/ProcessingData/iMVC/S9"
File_name112 = glob.glob(File_path112 + "/*" + Muscle + "*timing*.xlsx")
File_path113 = "D:/NTSU/TenLab/Archery/Archery_20211124/ProcessingData/iMVC/S10"
File_name113 = glob.glob(File_path113 + "/*" + Muscle + "*timing*.xlsx")
# 第二組
File_path21 = "D:/NTSU/TenLab/Archery/Archery_20211118/ProcessingData/iMVC/S3"
File_name21 = glob.glob(File_path21 + "/*" + Muscle + "*timing*.xlsx")
File_path22 = "D:/NTSU/TenLab/Archery/Archery_20211118/ProcessingData/iMVC/S4"
File_name22 = glob.glob(File_path22 + "/*" + Muscle + "*timing*.xlsx")
File_path23 = "D:/NTSU/TenLab/Archery/Archery_20211118/ProcessingData/iMVC/S7"
File_name23 = glob.glob(File_path23 + "/*" + Muscle + "*timing*.xlsx")
File_path24 = "D:/NTSU/TenLab/Archery/Archery_20211118/ProcessingData/iMVC/S10"
File_name24 = glob.glob(File_path24 + "/*" + Muscle + "*timing*.xlsx")

File_path25 = "D:/NTSU/TenLab/Archery/Archery_20211124/ProcessingData/iMVC/S1"
File_name25 = glob.glob(File_path25 + "/*" + Muscle + "*timing*.xlsx")
File_path26 = "D:/NTSU/TenLab/Archery/Archery_20211124/ProcessingData/iMVC/S3"
File_name26 = glob.glob(File_path26 + "/*" + Muscle + "*timing*.xlsx")
File_path27 = "D:/NTSU/TenLab/Archery/Archery_20211124/ProcessingData/iMVC/S6"
File_name27 = glob.glob(File_path27 + "/*" + Muscle + "*timing*.xlsx")
File_path28 = "D:/NTSU/TenLab/Archery/Archery_20211124/ProcessingData/iMVC/S7"
File_name28 = glob.glob(File_path28 + "/*" + Muscle + "*timing*.xlsx")

# Read data with specific name
# group 1
Muscle_data11 = pd.read_excel(File_name11[0])
Muscle_data12 = pd.read_excel(File_name12[0])
Muscle_data13 = pd.read_excel(File_name13[0])
Muscle_data14 = pd.read_excel(File_name14[0])
Muscle_data15 = pd.read_excel(File_name15[0])
Muscle_data16 = pd.read_excel(File_name16[0])
Muscle_data17 = pd.read_excel(File_name17[0])
Muscle_data18 = pd.read_excel(File_name18[0])
Muscle_data19 = pd.read_excel(File_name19[0])
Muscle_data110 = pd.read_excel(File_name110[0])
Muscle_data111 = pd.read_excel(File_name111[0])
Muscle_data112 = pd.read_excel(File_name112[0])
Muscle_data113 = pd.read_excel(File_name113[0])

Group_1_data = np.hstack((Muscle_data11, Muscle_data12, Muscle_data13,
                          Muscle_data14, Muscle_data15, Muscle_data16,
                          Muscle_data17, Muscle_data18, Muscle_data19,
                          Muscle_data110, Muscle_data111, Muscle_data112,
                          Muscle_data113))
# group 2
Muscle_data21 = pd.read_excel(File_name21[0])
Muscle_data22 = pd.read_excel(File_name22[0])
Muscle_data23 = pd.read_excel(File_name23[0])
Muscle_data24 = pd.read_excel(File_name24[0])
Muscle_data25 = pd.read_excel(File_name25[0])
Muscle_data26 = pd.read_excel(File_name26[0])
Muscle_data27 = pd.read_excel(File_name27[0])
Muscle_data28 = pd.read_excel(File_name28[0])

Group_2_data = np.hstack((Muscle_data21, Muscle_data22, Muscle_data23,
                          Muscle_data24, Muscle_data25, Muscle_data26,
                          Muscle_data27, Muscle_data28))

# extract column name with "Unnamed"
Not_0_Group_1_data = Group_1_data[:,Group_1_data.sum(axis=0)!=0]
Not_0_Group_2_data = Group_2_data[:,Group_2_data.sum(axis=0)!=0]

# Transpose matrix
Transpose_Group_1_data = np.transpose(Not_0_Group_1_data)
Transpose_Group_2_data = np.transpose(Not_0_Group_2_data)

fig1, ax1 = plt.subplots()
spm1d.plot.plot_mean_sd(Transpose_Group_1_data, linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='Group 1')
spm1d.plot.plot_mean_sd(Transpose_Group_2_data, linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Group 2')
plt.title('左手 後三角肌 (Left Deltoid)')
plt.axvline(x=5000, c='r', ls='--', lw=1)

fig2, ax2 = plt.subplots()
t  = spm1d.stats.ttest2(Transpose_Group_1_data, Transpose_Group_2_data, equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=False, interp=True)
plt.title('左手 後三角肌 (Left Deltoid)')
ti.plot()