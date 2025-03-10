

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:58:20 2022
@author: Hsin Yang
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spm1d

# 自動讀檔用，可判斷路徑下所有檔名，不管有沒有子資料夾
# 可針對不同副檔名的資料作判讀
def Read_File(x, y, subfolder='None'):

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

# -----------------code start--------------------------
# -----------------code start--------------------------

file_path = r'F:\HsinYang\NTSU\TenLab\Shooting\MingData\AfterFilting\TRC-AfterFilting\General'


file_list = Read_File(file_path, '.xlsx', subfolder=False)
columns_name = ['mean_std_x', 'mean_std_y', 'mean_std_z',
                'sum_dis_x', 'sum_dis_y', 'sum_dis_z']
data = np.zeros((len(file_list), 6))
data_file_name = list(np.zeros((len(file_list), 1)))

# Raw data
xx_data = []
xx_data = pd.DataFrame(xx_data)
yy_data = []
yy_data = pd.DataFrame(yy_data)
zz_data = []
zz_data = pd.DataFrame(zz_data)

# mean data
mean_xx_data = []
mean_xx_data = pd.DataFrame(mean_xx_data)
mean_yy_data = []
mean_yy_data = pd.DataFrame(mean_yy_data)
mean_zz_data = []
mean_zz_data = pd.DataFrame(mean_zz_data)

for ii in range(len(file_list)):
    x_data = pd.read_excel(file_list[ii], sheet_name='x')
    y_data = pd.read_excel(file_list[ii], sheet_name='y')
    z_data = pd.read_excel(file_list[ii], sheet_name='z')
    mean_x = np.mean(x_data, axis=1)
    
    xx_data = pd.concat([xx_data, mean_x], axis=1)
    yy_data = pd.concat([yy_data, y_data], axis=1)
    zz_data = pd.concat([zz_data, z_data], axis=1)
    # mean matrix caculate
    # mean_x = np.mean(x_data, axis=1)
    # standard deviation caculate
    std_x = np.std(x_data)
    std_y = np.std(y_data)
    std_z = np.std(z_data)
    mean_std_x = np.mean(std_x)
    mean_std_y = np.mean(std_y)
    mean_std_z = np.mean(std_z)
    max_x = max(x_data)
    #caculate sum of distance
    dis_x = np.zeros(np.shape(x_data))
    dis_y = np.zeros(np.shape(x_data))
    dis_z = np.zeros(np.shape(x_data))
    for i in range(np.shape(x_data)[0]-1):
        dis_x[i, :] = x_data.iloc[i+1, :] - x_data.iloc[i, :]
        dis_y[i, :] = y_data.iloc[i+1, :] - y_data.iloc[i, :]
        dis_z[i, :] = z_data.iloc[i+1, :] - z_data.iloc[i, :]
    
    sum_dis_x = np.sum(np.abs(dis_x))/np.shape(x_data)[1]
    sum_dis_y = np.sum(np.abs(dis_y))/np.shape(x_data)[1]
    sum_dis_z = np.sum(np.abs(dis_z))/np.shape(x_data)[1]
    # writing data in the data matrix
    data[ii, :] = [mean_std_x, mean_std_y, mean_std_z,
                   sum_dis_x, sum_dis_y, sum_dis_z]
    data_file_name[ii] = [file_list[ii]]

data = pd.DataFrame(data, columns=columns_name)
data_file_name = pd.DataFrame(data_file_name, columns=['file_name'])
# write data to excel
# slove_path = r"F:\HsinYang\NTSU\TenLab\Shooting\MingData\AfterFilting\General_output_2.xlsx"
# writer = pd.ExcelWriter(slove_path, engine='xlsxwriter')

# Write each dataframe to a different worksheet.
# data_file_name.to_excel(writer, startcol=0, index=False, header=True)
# data.to_excel(writer, startcol=1, index=False, header=True)
# writer.save()


# spm code



# # 顯示輸入中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# plot TrapeziusUpper_muscle
plt.figure(1)
plt.plot(xx_data, label='label_first')
spm1d.plot.plot_mean_sd(np.transpose(xx_data), linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='Slow')
# spm1d.plot.plot_mean_sd(Y1, label='Normal')
# spm1d.plot.plot_mean_sd(Y2, linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Fast')
plt.xlabel('Time (seconds)')
plt.ylabel('Muscle activation(%)')
plt.title('上斜方肌' )
plt.grid()
plt.legend()
plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'TrapeziusUpper.jpg'), dpi =300)
