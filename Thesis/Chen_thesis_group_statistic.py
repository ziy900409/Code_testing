# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:50:46 2022
@author: user
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import spm1d

# -----------------------Reading all of data path----------------------------
# using a recursive loop to traverse each folder
# and find the file extension has .csv
def Read_File(x, y, subfolder='None'):
    
    # if subfolder = True, the function will run with subfolder
    folder_path = x
    data_type = y
    csv_file_list = []
    
    if subfolder:
        file_list_1 = []
        for dirPath, dirNames, fileNames in os.walk(x):
            # file_list = os.walk(folder_name)
            file_list_1.append(dirPath)
        # need to change here [1:]
        for ii in file_list_1[1:]:
            file_list = os.listdir(ii)
            for iii in file_list:
                if os.path.splitext(iii)[1] == data_type:
                    file_list_name = ii + '\\' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(x)                
        for i in folder_list:
            if os.path.splitext(i)[1] == data_type:
                file_list_name = folder_path + "\\" + i
                csv_file_list.append(file_list_name)                
        
    return csv_file_list

# ---------Read each pattern-------------------
pattern_1_folder = r'D:\NTSU\ChenDissertationDataProcessing\EMG_Data\ProcessingData\RMS\pattern1_rms'
pattern_2_folder = r'D:\NTSU\ChenDissertationDataProcessing\EMG_Data\ProcessingData\RMS\pattern2_rms'

pattern_1_folder_list = os.listdir(pattern_1_folder)
pattern_2_folder_list = os.listdir(pattern_2_folder)

# ---------------------------pattern 1---------------------------------
# 矩陣預定義
one_mean_R_Supraspinatus = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle1
one_mean_L_Supraspinatus = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle2
one_mean_R_Biceps = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle3
one_mean_R_Triceps = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle4
one_mean_R_Deltoid = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle5
one_mean_L_Deltoid = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle6
one_mean_R_TrapeziusUpper = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle7
one_mean_L_TrapeziusUpper = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle8
one_mean_R_TrapeziusLower = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle9
one_mean_L_TrapeziusLower = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle10
one_mean_R_ErectorSpinae = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle11
one_mean_L_ErectorSpinae = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle12
one_mean_R_LatissimusDorsi = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle13
one_mean_L_LatissimusDorsi = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle14
one_mean_R_Extensor = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle15
one_mean_R_Flexor = np.zeros([6000, np.shape(pattern_1_folder_list)[0]]) #muscle16

for i in range(len(pattern_1_folder_list)):
    print(pattern_1_folder_list[i])
    file_name = pattern_1_folder + '\\' + pattern_1_folder_list[i] \
        + '\\iMVC\\韻律'
    file_list = Read_File(file_name, '.xlsx', subfolder = False)
    # 預定義矩陣放置所有檔案的資料
    R_Supraspinatus = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle1
    L_Supraspinatus = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle2
    R_Biceps = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle3
    R_Triceps = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle4
    R_Deltoid = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle5
    L_Deltoid = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle6
    R_TrapeziusUpper = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle7
    L_TrapeziusUpper = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle8
    R_TrapeziusLower = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle9
    L_TrapeziusLower = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle10
    R_ErectorSpinae = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle11
    L_ErectorSpinae = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle12
    R_LatissimusDorsi = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle13
    L_LatissimusDorsi = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle14
    R_Extensor = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle15
    R_Flexor = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle16
    
    for ii in range(len(file_list)):    
        EMG_data = pd.read_excel(file_list[ii])
        print(file_list[ii])
        # 輸入資料矩陣
        R_Supraspinatus.iloc[:, ii] = EMG_data.iloc[:,1]
        L_Supraspinatus.iloc[:,ii] = EMG_data.iloc[:,2]
        R_Biceps.iloc[:,ii] = EMG_data.iloc[:,3]
        R_Triceps.iloc[:,ii] = EMG_data.iloc[:,4]
        R_Deltoid.iloc[:,ii] = EMG_data.iloc[:,5]
        L_Deltoid.iloc[:,ii] = EMG_data.iloc[:,6]
        R_TrapeziusUpper.iloc[:,ii] = EMG_data.iloc[:,7]
        L_TrapeziusUpper.iloc[:,ii] = EMG_data.iloc[:,8]
        R_TrapeziusLower.iloc[:,ii] = EMG_data.iloc[:,9]
        L_TrapeziusLower.iloc[:,ii]= EMG_data.iloc[:,10]
        R_ErectorSpinae.iloc[:,ii]= EMG_data.iloc[:,11]
        L_ErectorSpinae.iloc[:,ii] = EMG_data.iloc[:,12]
        R_LatissimusDorsi.iloc[:,ii] = EMG_data.iloc[:,13]
        L_LatissimusDorsi.iloc[:,ii] = EMG_data.iloc[:,14]
        R_Extensor.iloc[:,ii] = EMG_data.iloc[:,15]
        R_Flexor.iloc[:,ii] = EMG_data.iloc[:,16]
        
    R_Supraspinatus = R_Supraspinatus.dropna(axis=1, how='all')
    L_Supraspinatus = L_Supraspinatus.dropna(axis=1, how='all')
    R_Biceps = R_Biceps.dropna(axis=1, how='all')
    R_Triceps = R_Triceps.dropna(axis=1, how='all')
    R_Deltoid = R_Deltoid.dropna(axis=1, how='all')
    L_Deltoid = L_Deltoid.dropna(axis=1, how='all')
    R_TrapeziusUpper = R_TrapeziusUpper.dropna(axis=1, how='all')
    L_TrapeziusUpper = L_TrapeziusUpper.dropna(axis=1, how='all')
    R_TrapeziusLower = R_TrapeziusLower.dropna(axis=1, how='all')
    L_TrapeziusLower = L_TrapeziusLower.dropna(axis=1, how='all')
    R_ErectorSpinae = R_ErectorSpinae.dropna(axis=1, how='all')
    L_ErectorSpinae = L_ErectorSpinae.dropna(axis=1, how='all')
    R_LatissimusDorsi = R_LatissimusDorsi.dropna(axis=1, how='all')
    L_LatissimusDorsi = L_LatissimusDorsi.dropna(axis=1, how='all')
    R_Extensor = R_Extensor.dropna(axis=1, how='all')
    R_Flexor = R_Flexor.dropna(axis=1, how='all')
    # 計算每位受使者的平均值
    one_mean_R_Supraspinatus[:, i] = np.mean(R_Supraspinatus, axis=1) #muscle1
    one_mean_L_Supraspinatus[:, i] = np.mean(L_Supraspinatus, axis=1) #muscle2
    one_mean_R_Biceps[:, i] = np.mean(R_Biceps, axis=1) #muscle3
    one_mean_R_Triceps[:, i] = np.mean(R_Triceps, axis=1) #muscle4
    one_mean_R_Deltoid[:, i] = np.mean(R_Deltoid, axis=1) #muscle5
    one_mean_L_Deltoid[:, i] = np.mean(L_Deltoid, axis=1) #muscle6
    one_mean_R_TrapeziusUpper[:, i] = np.mean(R_TrapeziusUpper, axis=1) #muscle7
    one_mean_L_TrapeziusUpper[:, i] = np.mean(L_TrapeziusUpper, axis=1) #muscle8
    one_mean_R_TrapeziusLower[:, i] = np.mean(R_TrapeziusLower, axis=1) #muscle9
    one_mean_L_TrapeziusLower[:, i] = np.mean(L_TrapeziusLower, axis=1) #muscle10
    one_mean_R_ErectorSpinae[:, i] = np.mean(R_ErectorSpinae, axis=1) #muscle11
    one_mean_L_ErectorSpinae[:, i] = np.mean(L_ErectorSpinae, axis=1) #muscle12
    one_mean_R_LatissimusDorsi[:, i] = np.mean(R_LatissimusDorsi, axis=1) #muscle13
    one_mean_L_LatissimusDorsi[:, i] = np.mean(L_LatissimusDorsi, axis=1) #muscle14
    one_mean_R_Extensor[:, i] = np.mean(R_Extensor, axis=1) #muscle15
    one_mean_R_Flexor[:, i] = np.mean(R_Flexor, axis=1) #muscle16
    
# ---------------------------pattern 2---------------------------------
# 矩陣預定義
two_mean_R_Supraspinatus = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle1
two_mean_L_Supraspinatus = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle2
two_mean_R_Biceps = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle3
two_mean_R_Triceps = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle4
two_mean_R_Deltoid = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle5
two_mean_L_Deltoid = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle6
two_mean_R_TrapeziusUpper = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle7
two_mean_L_TrapeziusUpper = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle8
two_mean_R_TrapeziusLower = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle9
two_mean_L_TrapeziusLower = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle10
two_mean_R_ErectorSpinae = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle11
two_mean_L_ErectorSpinae = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle12
two_mean_R_LatissimusDorsi = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle13
two_mean_L_LatissimusDorsi = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle14
two_mean_R_Extensor = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle15
two_mean_R_Flexor = np.zeros([6000, np.shape(pattern_2_folder_list)[0]]) #muscle16

for i in range(len(pattern_2_folder_list)):
    print(pattern_2_folder_list[i])
    file_name = pattern_2_folder + '\\' + pattern_2_folder_list[i] \
        + '\\iMVC\\韻律'
    file_list = Read_File(file_name, '.xlsx', subfolder = False)
    # 預定義矩陣放置所有檔案的資料
    R_Supraspinatus = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle1
    L_Supraspinatus = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle2
    R_Biceps = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle3
    R_Triceps = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle4
    R_Deltoid = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle5
    L_Deltoid = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle6
    R_TrapeziusUpper = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle7
    L_TrapeziusUpper = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle8
    R_TrapeziusLower = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle9
    L_TrapeziusLower = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle10
    R_ErectorSpinae = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle11
    L_ErectorSpinae = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle12
    R_LatissimusDorsi = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle13
    L_LatissimusDorsi = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle14
    R_Extensor = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle15
    R_Flexor = pd.DataFrame(np.zeros([6000, np.shape(file_list)[0]])) #muscle16
    
    for ii in range(len(file_list)):    
        EMG_data = pd.read_excel(file_list[ii])
        print(file_list[ii])
        # 輸入資料矩陣
        R_Supraspinatus.iloc[:, ii] = EMG_data.iloc[:,1]
        L_Supraspinatus.iloc[:,ii] = EMG_data.iloc[:,2]
        R_Biceps.iloc[:,ii] = EMG_data.iloc[:,3]
        R_Triceps.iloc[:,ii] = EMG_data.iloc[:,4]
        R_Deltoid.iloc[:,ii] = EMG_data.iloc[:,5]
        L_Deltoid.iloc[:,ii] = EMG_data.iloc[:,6]
        R_TrapeziusUpper.iloc[:,ii] = EMG_data.iloc[:,7]
        L_TrapeziusUpper.iloc[:,ii] = EMG_data.iloc[:,8]
        R_TrapeziusLower.iloc[:,ii] = EMG_data.iloc[:,9]
        L_TrapeziusLower.iloc[:,ii]= EMG_data.iloc[:,10]
        R_ErectorSpinae.iloc[:,ii]= EMG_data.iloc[:,11]
        L_ErectorSpinae.iloc[:,ii] = EMG_data.iloc[:,12]
        R_LatissimusDorsi.iloc[:,ii] = EMG_data.iloc[:,13]
        L_LatissimusDorsi.iloc[:,ii] = EMG_data.iloc[:,14]
        R_Extensor.iloc[:,ii] = EMG_data.iloc[:,15]
        R_Flexor.iloc[:,ii] = EMG_data.iloc[:,16]
        
        
    R_Supraspinatus = R_Supraspinatus.dropna(axis=1, how='all')
    L_Supraspinatus = L_Supraspinatus.dropna(axis=1, how='all')
    R_Biceps = R_Biceps.dropna(axis=1, how='all')
    R_Triceps = R_Triceps.dropna(axis=1, how='all')
    R_Deltoid = R_Deltoid.dropna(axis=1, how='all')
    L_Deltoid = L_Deltoid.dropna(axis=1, how='all')
    R_TrapeziusUpper = R_TrapeziusUpper.dropna(axis=1, how='all')
    L_TrapeziusUpper = L_TrapeziusUpper.dropna(axis=1, how='all')
    R_TrapeziusLower = R_TrapeziusLower.dropna(axis=1, how='all')
    L_TrapeziusLower = L_TrapeziusLower.dropna(axis=1, how='all')
    R_ErectorSpinae = R_ErectorSpinae.dropna(axis=1, how='all')
    L_ErectorSpinae = L_ErectorSpinae.dropna(axis=1, how='all')
    R_LatissimusDorsi = R_LatissimusDorsi.dropna(axis=1, how='all')
    L_LatissimusDorsi = L_LatissimusDorsi.dropna(axis=1, how='all')
    R_Extensor = R_Extensor.dropna(axis=1, how='all')
    R_Flexor = R_Flexor.dropna(axis=1, how='all')
    # 計算每位受使者的平均值
    two_mean_R_Supraspinatus[:, i] = np.mean(R_Supraspinatus, axis=1) #muscle1
    two_mean_L_Supraspinatus[:, i] = np.mean(L_Supraspinatus, axis=1) #muscle2
    two_mean_R_Biceps[:, i] = np.mean(R_Biceps, axis=1) #muscle3
    two_mean_R_Triceps[:, i] = np.mean(R_Triceps, axis=1) #muscle4
    two_mean_R_Deltoid[:, i] = np.mean(R_Deltoid, axis=1) #muscle5
    two_mean_L_Deltoid[:, i] = np.mean(L_Deltoid, axis=1) #muscle6
    two_mean_R_TrapeziusUpper[:, i] = np.mean(R_TrapeziusUpper, axis=1) #muscle7
    two_mean_L_TrapeziusUpper[:, i] = np.mean(L_TrapeziusUpper, axis=1) #muscle8
    two_mean_R_TrapeziusLower[:, i] = np.mean(R_TrapeziusLower, axis=1) #muscle9
    two_mean_L_TrapeziusLower[:, i] = np.mean(L_TrapeziusLower, axis=1) #muscle10
    two_mean_R_ErectorSpinae[:, i] = np.mean(R_ErectorSpinae, axis=1) #muscle11
    two_mean_L_ErectorSpinae[:, i] = np.mean(L_ErectorSpinae, axis=1) #muscle12
    two_mean_R_LatissimusDorsi[:, i] = np.mean(R_LatissimusDorsi, axis=1) #muscle13
    two_mean_L_LatissimusDorsi[:, i] = np.mean(L_LatissimusDorsi, axis=1) #muscle14
    two_mean_R_Extensor[:, i] = np.mean(R_Extensor, axis=1) #muscle15
    two_mean_R_Flexor[:, i] = np.mean(R_Flexor, axis=1) #muscle16
    
    
# spm1d繪圖
# spm1d繪圖
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
save_path = r'D:\NTSU\ChenDissertationDataProcessing\EMG_Data\RMS_results\\'
# 1 右棘上肌
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_R_Supraspinatus), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_R_Supraspinatus), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('右棘上肌_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)

ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_R_Supraspinatus), np.transpose(two_mean_R_Supraspinatus), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('右棘上肌_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()
plt.savefig((save_path + 'R_Supraspinatus_ttest2.jpg'), dpi =300)

# 2 左棘上肌
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_L_Supraspinatus), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_L_Supraspinatus), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('左棘上肌_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)

ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_L_Supraspinatus), np.transpose(two_mean_L_Supraspinatus), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('左棘上肌_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()
plt.savefig((save_path + 'L_Supraspinatus_ttest2.jpg'), dpi =300)

# 3 右二頭肌
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_R_Biceps), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_R_Biceps), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('右二頭肌_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)

ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_R_Biceps), np.transpose(two_mean_R_Biceps), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('右二頭肌_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()
plt.savefig((save_path + 'R_Biceps_ttest2.jpg'), dpi =300)
        
# 4 右三頭肌
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_R_Triceps), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_R_Triceps), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('右三頭肌_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)

ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_R_Triceps), np.transpose(two_mean_R_Triceps), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('右三頭肌_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()
plt.savefig((save_path + 'R_Triceps_ttest2.jpg'), dpi =300)

# 5 右三角肌
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_R_Deltoid), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_R_Deltoid), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('右三角肌_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)

ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_R_Deltoid), np.transpose(two_mean_R_Deltoid), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('右三角肌_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()
plt.savefig((save_path + 'R_Deltoid_ttest2.jpg'), dpi =300)

# 6 左三頭肌
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_L_Deltoid), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_L_Deltoid), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('左三頭肌_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)

ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_L_Deltoid), np.transpose(two_mean_L_Deltoid), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('左三頭肌_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()
plt.savefig((save_path + 'L_Deltoid_ttest2.jpg'), dpi =300)

# 7 右上斜方
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_R_TrapeziusUpper), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_R_TrapeziusUpper), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('右上斜方_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)

ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_R_TrapeziusUpper), np.transpose(two_mean_R_TrapeziusUpper), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('右上斜方_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()
plt.savefig((save_path + 'R_TrapeziusUpper_ttest2.jpg'), dpi =300)

# 8 左上斜方
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_L_TrapeziusUpper), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_L_TrapeziusUpper), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('左上斜方_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)


ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_L_TrapeziusUpper), np.transpose(two_mean_L_TrapeziusUpper), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('左上斜方_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()
plt.savefig((save_path + 'L_TrapeziusUpper_ttest2.jpg'), dpi =300)

# 9 右下斜方
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_R_TrapeziusLower), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_R_TrapeziusLower), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('右下斜方_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)


ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_R_TrapeziusLower), np.transpose(two_mean_R_TrapeziusLower), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('右下斜方_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()
plt.savefig((save_path + 'R_TrapeziusLower_ttest2.jpg'), dpi =300)

# 10 左下斜方
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_L_TrapeziusLower), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_L_TrapeziusLower), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('左下斜方_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)


ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_L_TrapeziusLower), np.transpose(two_mean_L_TrapeziusLower), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('左下斜方_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()
plt.savefig((save_path + 'L_TrapeziusLower_ttest2.jpg'), dpi =300)

# 11 右束脊肌
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_R_ErectorSpinae), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_R_ErectorSpinae), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('右束脊肌_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)



ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_R_ErectorSpinae), np.transpose(two_mean_R_ErectorSpinae), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('右束脊肌_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()
plt.savefig((save_path + 'R_ErectorSpinae_ttest2.jpg'), dpi =300)

# 12 左束脊肌
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_L_ErectorSpinae), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_L_ErectorSpinae), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('左束脊肌_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)


ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_L_ErectorSpinae), np.transpose(two_mean_L_ErectorSpinae), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('左束脊肌_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()
plt.savefig((save_path + 'L_ErectorSpinae_ttest2.jpg'), dpi =300)

# 13 右擴背肌
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_R_LatissimusDorsi), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_R_LatissimusDorsi), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('右擴背肌_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)


ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_R_LatissimusDorsi), np.transpose(two_mean_R_LatissimusDorsi), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('右擴背肌_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()  
plt.savefig((save_path + 'R_LatissimusDorsi_ttest2.jpg'), dpi =300)

# 14 左擴背肌
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_L_LatissimusDorsi), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_L_LatissimusDorsi), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('左擴背肌_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)


ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_L_LatissimusDorsi), np.transpose(two_mean_L_LatissimusDorsi), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('左擴背肌_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()  
plt.savefig((save_path + 'L_LatissimusDorsi_ttest2.jpg'), dpi =300)

# 15 右伸腕肌
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_R_Extensor), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_R_Extensor), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('右伸腕肌_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)


ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_R_Extensor), np.transpose(two_mean_R_Extensor), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('右伸腕肌_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()  
plt.savefig((save_path + 'R_Extensor_ttest2.jpg'), dpi =300)

# 16 右屈腕肌
fig= plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
spm1d.plot.plot_mean_sd(np.transpose(one_mean_R_Flexor), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='pattern1')
spm1d.plot.plot_mean_sd(np.transpose(two_mean_R_Flexor), linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='pattern2')
plt.legend()
plt.title('右屈腕肌_StdCloud', fontsize=18)
ax1.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax1.set_ylabel('肌肉活化程度 (%)', fontsize=14) # y軸名稱
plt.axvline(x=5000, c='r', ls='--', lw=1)


ax2 = plt.subplot(1,2,2)
t  = spm1d.stats.ttest2(np.transpose(one_mean_R_Flexor), np.transpose(two_mean_R_Flexor), equal_var=False)
ti = t.inference(alpha=0.05, two_tailed=True, interp=False)
plt.axvline(x=5000, c='r', ls='--', lw=1)
plt.title('右屈腕肌_ttest2', fontsize=18)
ax2.set_xlabel('秒數 (毫秒)', fontsize=14) # x軸名稱
ax2.set_ylabel('t值', fontsize=14) # y軸名稱
ti.plot()          
plt.savefig((save_path + 'R_Flexor_ttest2.jpg'), dpi =300) 

# 17 右半身繪圖
#  pattern1
fig= plt.figure(figsize=(14,8))  

data_time = np.linspace(-2.5, 0.5, 6000)
plt.plot(data_time, np.mean(one_mean_R_Supraspinatus, axis=1), color='lightgray', label='右棘上')
plt.plot(data_time, np.mean(one_mean_R_Biceps, axis=1), color='darkgray', label='右二頭')
plt.plot(data_time, np.mean(one_mean_R_Triceps, axis=1), color='red', label='右三頭')
plt.plot(data_time, np.mean(one_mean_R_Deltoid, axis=1), color='chocolate', label='右三角')
plt.plot(data_time, np.mean(one_mean_R_TrapeziusUpper, axis=1), color='dimgray', label='右上斜方')
plt.plot(data_time, np.mean(one_mean_R_TrapeziusLower, axis=1), color='orange', label='右下斜方')
plt.plot(data_time, np.mean(one_mean_R_ErectorSpinae, axis=1), color='green', label='右束脊')
plt.plot(data_time, np.mean(one_mean_R_LatissimusDorsi, axis=1), color='cyan', label='右擴背')
plt.plot(data_time, np.mean(one_mean_R_Extensor, axis=1), color='blue', label='右伸腕')
plt.plot(data_time, np.mean(one_mean_R_Flexor, axis=1), color='purple', label='右屈腕')
plt.axvline(x=0, c='r', ls='--', lw=1)
plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
# plt.xticks(np.arange(-2.5,0.5,0.1))
plt.grid(axis='x', linestyle='-.')
plt.title('pattern1 右半身 各肌群活化時序圖', fontsize=18)
plt.tight_layout()
plt.savefig((save_path + 'pattern1_右半身_各肌群活化時序圖_1.jpg'), dpi =300) 
plt.show()

#  pattern2
fig= plt.figure(figsize=(14,8))  

data_time = np.linspace(-2.5, 0.5, 6000)
plt.plot(data_time, np.mean(two_mean_R_Supraspinatus, axis=1), color='lightgray', label='右棘上')
plt.plot(data_time, np.mean(two_mean_R_Biceps, axis=1), color='darkgray', label='右二頭')
plt.plot(data_time, np.mean(two_mean_R_Triceps, axis=1), color='red', label='右三頭')
plt.plot(data_time, np.mean(two_mean_R_Deltoid, axis=1), color='chocolate', label='右三角')
plt.plot(data_time, np.mean(two_mean_R_TrapeziusUpper, axis=1), color='dimgray', label='右上斜方')
plt.plot(data_time, np.mean(two_mean_R_TrapeziusLower, axis=1), color='orange', label='右下斜方')
plt.plot(data_time, np.mean(two_mean_R_ErectorSpinae, axis=1), color='green', label='右束脊')
plt.plot(data_time, np.mean(two_mean_R_LatissimusDorsi, axis=1), color='cyan', label='右擴背')
plt.plot(data_time, np.mean(two_mean_R_Extensor, axis=1), color='blue', label='右伸腕')
plt.plot(data_time, np.mean(two_mean_R_Flexor, axis=1), color='purple', label='右屈腕')
plt.axvline(x=0, c='r', ls='--', lw=1)
plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
# plt.xticks(np.arange(-2.5,0.5,0.1))
plt.grid(axis='x', linestyle='-.')
plt.title('pattern2 右半身 各肌群活化時序圖', fontsize=18)
plt.tight_layout()
plt.savefig((save_path + 'pattern2_右半身_各肌群活化時序圖_1.jpg'), dpi =300)
plt.show()

# 18 左半身繪圖
#  pattern1
fig= plt.figure(figsize=(14,8))  

data_time = np.linspace(-2.5, 0.5, 6000)
plt.plot(data_time, np.mean(one_mean_L_Supraspinatus, axis=1), color='lightgray', label='左棘上')
plt.plot(data_time, np.mean(one_mean_L_Deltoid, axis=1), color='chocolate', label='左三角')
plt.plot(data_time, np.mean(one_mean_L_TrapeziusUpper, axis=1), color='dimgray', label='左上斜方')
plt.plot(data_time, np.mean(one_mean_L_TrapeziusLower, axis=1), color='orange', label='左下斜方')
plt.plot(data_time, np.mean(one_mean_L_ErectorSpinae, axis=1), color='green', label='左束脊')
plt.plot(data_time, np.mean(one_mean_L_LatissimusDorsi, axis=1), color='cyan', label='右擴背')
plt.axvline(x=0, c='r', ls='--', lw=1)
plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
plt.xticks(np.arange(-2.5,0.5,0.1))
plt.grid(axis='x', linestyle='-.')
plt.title('pattern1 左半身 各肌群活化時序圖', fontsize=18)
plt.tight_layout()
plt.savefig((save_path + 'pattern1_左半身_各肌群活化時序圖.jpg'), dpi =300) 
plt.show()

#  pattern2
fig= plt.figure(figsize=(14,8))  

data_time = np.linspace(-2.5, 0.5, 6000)
plt.plot(data_time, np.mean(two_mean_L_Supraspinatus, axis=1), color='lightgray', label='左棘上')
plt.plot(data_time, np.mean(two_mean_L_Deltoid, axis=1), color='chocolate', label='左三角')
plt.plot(data_time, np.mean(two_mean_L_TrapeziusUpper, axis=1), color='dimgray', label='左上斜方')
plt.plot(data_time, np.mean(two_mean_L_TrapeziusLower, axis=1), color='orange', label='左下斜方')
plt.plot(data_time, np.mean(two_mean_L_ErectorSpinae, axis=1), color='green', label='左束脊')
plt.plot(data_time, np.mean(two_mean_L_LatissimusDorsi, axis=1), color='cyan', label='右擴背')
plt.axvline(x=0, c='r', ls='--', lw=1)
plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
plt.xticks(np.arange(-2.5,0.5,0.1))
plt.grid(axis='x', linestyle='-.')
plt.title('pattern2 左半身 各肌群活化時序圖', fontsize=18)
plt.tight_layout()
plt.savefig((save_path + 'pattern2_左半身_各肌群活化時序圖.jpg'), dpi =300) 
plt.show()