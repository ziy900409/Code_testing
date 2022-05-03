# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:31:50 2022
version 2.3
@author: Hsin Yang, 20220429
"""

import os
import pandas as pd
import numpy as np
from pandas import DataFrame
import math
import time

start = time.time()
# 自動讀檔用，可判斷路徑下所有檔名，不管有沒有子資料夾
# 可針對不同副檔名的資料作判讀
def Read_File(x, subfolder='None'):
    # if subfolder = True, the function will run with subfolder
    folder_path = x
    csv_file_list = []
    
    if subfolder:
        file_list_1 = []
        for dirPath, dirNames, fileNames in os.walk(x):
            # file_list = os.walk(folder_name)
            file_list_1.append(dirPath)
                
        for ii in file_list_1:
            file_list = os.listdir(ii)
            for iii in file_list:
                # 抓副檔名.trc的檔案，可修正為抓多種不同副檔名
                if os.path.splitext(iii)[1] == ".xlsx":
                    file_list_name = ii + '\\' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(x)                
        for i in folder_list:
            if os.path.splitext(i)[1] == ".xlsx":
                # 抓副檔名.trc的檔案
                file_list_name = folder_path + "\\" + i
                csv_file_list.append(i)                
        
    return csv_file_list
# to calaulate velocity
# 計算速度用，n為一次跨多少區間
def velocity_cal(x1, x2, n):
    Vx = []
    for i in range(len(x1)-n):
        b = (x1.iloc[i+n] - x2.iloc[i])
        Vx.append(b)
        
    return Vx
# 計算夾角
def included_angle(x0, x1, y0, y1):
    # 掌骨角度 仰角計算 chen
    # calculate the vector
    x0_x1 = x0 - x1
    y0_y1 = y0 - y1
    # calculate included angle
    dot_value_1 = [x0_x1.iloc[iii, 🙂.values.dot(y0_y1.iloc[iii, 🙂.values)
                 for iii in range(len(x1))]
    l_x_1 = np.sqrt([x0_x1.iloc[iii, 🙂.values.dot(x0_x1.iloc[iii, 🙂.values)
                    for iii in range(len(x1))])
    l_y_1 = np.sqrt([y0_y1.iloc[iii, 🙂.values.dot(y0_y1.iloc[iii, 🙂.values)
                    for iii in range(len(x1))])
    # to calculate dot values
    elevation_angle = dot_value_1/(l_x_1*l_y_1)
    # to calculate acos values
    acos_elevation_angle = [math.degrees(math.acos(elevation_angle[iii])) 
                              for iii in range(len(elevation_angle))]
    return acos_elevation_angle
# -------------------------code staring---------------------------------

# read staging file
# 設定分期檔的路徑
staging_file_path = r"D:\NTSU\TenLab\computer mouse\S3_Frame_TeskRecord.xlsx"
staging_file_data = pd.read_excel(staging_file_path, sheet_name="large range")

# read file list
# setting file's folder
# 設定目標資料夾
data_folder_path = r'D:\NTSU\TenLab\computer mouse\BenQ'
data_file_list = Read_File(data_folder_path, subfolder=True)

mean_matrix = np.zeros([len(data_file_list), 28])
mean_columns_name = ['FileName', 'X1', 'Y1', 'Z1', '合1',
             'X2', 'Y2', 'Z2', '合2',
             'X3', 'Y3', 'Z3', '合3',
             'X4', 'Y4', 'Z4', '合4',
              '最大變異', '掌骨仰角', '掌骨橈尺偏角', '掌骨側傾角',
             '大拇指近節', '食指近節', '食指中節',
             '中指近節', '無名指近節', '小指近節', 'Var差異']
mean_matrix = pd.DataFrame(mean_matrix, columns = mean_columns_name)

for i in range(len(data_file_list)):
    data = pd.read_excel(data_file_list[i], skiprows=1)
    # 建立存放資料的矩陣
    combine_matrix = np.zeros([np.shape(data)[0], 26])
    columns_name = ['X1', 'Y1', 'Z1', '合1',
                    'X2', 'Y2', 'Z2', '合2',
                    'X3', 'Y3', 'Z3', '合3',
                    'X4', 'Y4', 'Z4', '合4',
                    '最大變異', '掌骨仰角', '掌骨橈尺偏角', '掌骨側傾角',
                    '大拇指近節', '食指近節', '食指中節',
                    '中指近節', '無名指近節', '小指近節']
    combine_matrix = pd.DataFrame(combine_matrix, columns = columns_name)
    # 計算手掌速度
    wrist_x = np.add(data.iloc[:, 14].values, data.iloc[:, 17].values)
    wrist_x = pd.DataFrame(np.divide(wrist_x, 2))
    hand_x = np.add(wrist_x, pd.DataFrame(data.iloc[:, 35].values))
    hand_x = pd.DataFrame(np.divide(hand_x, 2)) 
    x1 = pd.DataFrame(np.divide(velocity_cal(hand_x, hand_x, 4), 4/180))
    wrist_y = np.add(data.iloc[:, 15].values, data.iloc[:, 18].values)
    wrist_y = pd.DataFrame(np.divide(wrist_y, 2))
    hand_y = np.add(wrist_y, pd.DataFrame(data.iloc[:, 36].values))
    hand_y = pd.DataFrame(np.divide(hand_y, 2))
    y1 = pd.DataFrame(np.divide(velocity_cal(hand_y, hand_y, 4), 4/180))
    wrist_z = np.add(data.iloc[:, 16].values, data.iloc[:, 19].values)
    wrist_z = pd.DataFrame(np.divide(wrist_z, 2))
    hand_z = np.add(wrist_x, pd.DataFrame(data.iloc[:, 37].values))
    hand_z = pd.DataFrame(np.divide(hand_z, 2))
    z1 = pd.DataFrame(np.divide(velocity_cal(hand_z, hand_z, 4), 4/180))
    com1 = pd.DataFrame((np.multiply(x1, x1) + np.multiply(y1, y1) + np.multiply(z1, z1))**0.5)
    # 計算食指速度
    x2 = pd.DataFrame(np.divide(velocity_cal(data.iloc[:, 32], data.iloc[:, 32], 4), 4/180))
    y2 = pd.DataFrame(np.divide(velocity_cal(data.iloc[:, 33], data.iloc[:, 33], 4), 4/180))
    z2 = pd.DataFrame(np.divide(velocity_cal(data.iloc[:, 34], data.iloc[:, 34], 4), 4/180))
    com2 = pd.DataFrame((np.multiply(x2, x2) + np.multiply(y2, y2) + np.multiply(z2, z2))**0.5)
    # 計算中指速度
    x3 = pd.DataFrame(np.divide(velocity_cal(data.iloc[:, 38], data.iloc[:, 38], 4), 4/180))
    y3 = pd.DataFrame(np.divide(velocity_cal(data.iloc[:, 39], data.iloc[:, 39], 4), 4/180))
    z3 = pd.DataFrame(np.divide(velocity_cal(data.iloc[:, 40], data.iloc[:, 40], 4), 4/180))
    com3 = pd.DataFrame((np.multiply(x3, x3) + np.multiply(y3, y3) + np.multiply(z3, z3))**0.5)
    # 計算滑鼠速度
    x4 = pd.DataFrame(np.divide(velocity_cal(data.iloc[:, 77], data.iloc[:, 77], 4), 4/180))
    y4 = pd.DataFrame(np.divide(velocity_cal(data.iloc[:, 78], data.iloc[:, 78], 4), 4/180))
    z4 = pd.DataFrame(np.divide(velocity_cal(data.iloc[:, 79], data.iloc[:, 79], 4), 4/180))
    com4 = pd.DataFrame((np.multiply(x4, x4) + np.multiply(y4, y4) + np.multiply(z4, z4))**0.5)
    # 最大變異
    # 找最大跟最小
    max_var = [math.dist(data.iloc[ii, [77, 78, 79]], data.iloc[ii, [71, 72, 73]])
               for ii in range(len(data.iloc[:, 79]))]
    # 掌骨仰角
    wrist_xz = np.add(data.iloc[:, [14, 16]].values, data.iloc[:, [17, 19]].values)
    wrist_xz = pd.DataFrame(np.divide(wrist_xz, 2))
    hand_xz = np.add(wrist_xz, data.iloc[:, [35, 37]].values)
    hand_xz = pd.DataFrame(np.divide(hand_xz, 2))
    elbow_xz = pd.DataFrame(data.iloc[:, [65, 67]].values)
    hand_bone_angle = included_angle(wrist_xz, hand_xz,
                                     elbow_xz, wrist_xz)
    # 橈尺偏角
    rad_xy = pd.DataFrame(data.iloc[:, [14, 15]].values)
    uln_xy = pd.DataFrame(data.iloc[:, [17, 18]].values)
    wrist_xy = np.add(data.iloc[:, [14, 15]].values, data.iloc[:, [17, 18]].values)
    wrist_xy = pd.DataFrame(np.divide(wrist_xy, 2))
    hand_xy = np.add(wrist_xy, data.iloc[:, [35, 36]].values)
    hand_xy = pd.DataFrame(np.divide(hand_xy, 2))
    Radioulnar_angle = included_angle(wrist_xy, hand_xy,
                                      rad_xy, uln_xy)
    # 側傾角
    rad_yz = pd.DataFrame(data.iloc[:, [15, 16]].values)
    uln_yz = pd.DataFrame(data.iloc[:, [18, 19]].values)
    Table1_yz = pd.DataFrame(data.iloc[:, [54, 55]].values)
    Table4_yz = pd.DataFrame(data.iloc[:, [63, 64]].values)
    side_angle = included_angle(rad_yz, uln_yz,
                                Table4_yz, Table1_yz)
    # 大拇指 近節指骨
    wrist_xyz = np.add(data.iloc[:, [14, 15, 16]].values, data.iloc[:, [17, 18, 19]].values)
    #wrist_xyz = pd.DataFrame(np.divide(wrist_xz, 2))
    wrist_xyz = pd.DataFrame(np.divide(wrist_xyz, 2))
    thumb_1_xyz = pd.DataFrame(data.iloc[:, [20, 21, 22]].values)
    thumb_2_xyz = pd.DataFrame(data.iloc[:, [23, 24, 25]].values)
    thumb_proximal = included_angle(thumb_1_xyz, wrist_xyz,
                                    thumb_1_xyz, thumb_2_xyz)
    # 食指 近節指骨
    finger_1_xyz = pd.DataFrame(data.iloc[:, [26, 27, 28]].values)
    finger_2_xyz = pd.DataFrame(data.iloc[:, [29, 30, 31]].values)
    finger_proximal = included_angle(finger_1_xyz, wrist_xyz,
                                     finger_1_xyz, finger_2_xyz)
    # 食指 中節指骨
    finger_3_xyz = pd.DataFrame(data.iloc[:, [32, 33, 34]].values)
    finger_middle = included_angle(finger_2_xyz, finger_1_xyz,
                                   finger_2_xyz, finger_3_xyz)
    # 中指 近節指骨
    m_finger_1_xyz = pd.DataFrame(data.iloc[:, [35, 36, 37]].values)
    m_finger_2_xyz = pd.DataFrame(data.iloc[:, [38, 39, 40]].values)
    m_finger_promixal = included_angle(m_finger_1_xyz, wrist_xyz,
                                       m_finger_1_xyz, m_finger_2_xyz)
    # 無名指 近指關節
    r_finger_1_xyz = pd.DataFrame(data.iloc[:, [41, 42, 43]].values)
    r_finger_2_xyz = pd.DataFrame(data.iloc[:, [44, 45, 46]].values)
    r_finger_promixal = included_angle(r_finger_1_xyz, wrist_xyz,
                                       r_finger_1_xyz, r_finger_2_xyz)
    # 小指 近指關節
    p_finger_1_xyz = pd.DataFrame(data.iloc[:, [47, 48, 49]].values)
    p_finger_2_xyz = pd.DataFrame(data.iloc[:, [50, 51, 52]].values)
    p_finger_promixal = included_angle(p_finger_1_xyz, wrist_xyz,
                                       p_finger_1_xyz, p_finger_2_xyz)
    # combine multiple list
    velocity_list = np.hstack((x1, y1, z1, com1,
                               x2, y2, z2, com2,
                               x3, y3, z3, com3,
                               x4, y4, z4, com4))
    angle_list = np.transpose(np.vstack((max_var, hand_bone_angle, Radioulnar_angle, side_angle,
                                         thumb_proximal, finger_proximal, finger_middle,
                                         m_finger_promixal, r_finger_promixal, p_finger_promixal)))
    combine_matrix.iloc[4:, 0:16] = velocity_list
    combine_matrix.iloc[:, 16:] = angle_list
    # 將資料寫進excel, if needed
    # 資料存在不同檔案夾
    file_name, extension = os.path.split(data_file_list[i])
    file_name1, extension1 = extension.split('.')
    file_name2 = r"D:\NTSU\TenLab\computer mouse\taa"
    file_name = file_name2 + '\\' + file_name1 + '_cal.xlsx'
    DataFrame(combine_matrix).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
    # ------------------------------to capture specific data frame-----------------------------
    # ------------------------------to capture specific data frame-----------------------------
    for iii in range(len(staging_file_data['FileName'])):
        if data_file_list[i] == staging_file_data['FileName'][iii]:
            print(i)
            print(iii)
            print(staging_file_data['FileName'][iii])
            print(data_file_list[i])
            # read data
            mouse_data = combine_matrix
            # using staging file to extract data
            # 利用分期檔抓時間點
            start_frame = int(staging_file_data['Start Frame'][iii])
            end_frame = int(staging_file_data['End Frame'][iii])
            extract_data = mouse_data.iloc[start_frame:end_frame+1, 🙂
            # calculate mean value
            # 計算平均值
            mean_extract_data = np.average(extract_data.values, axis=0)
            # assign data to claculate matrix
            mean_matrix.iloc[i, 0] = data_file_list[i]
            mean_matrix.iloc[i, 1:-1] = mean_extract_data
            mean_matrix.iloc[i, -1] = max(extract_data.iloc[:, 16].values) - min(extract_data.iloc[:, 16].values)
                
output_name = r'D:\NTSU\TenLab\computer mouse\S3_result.xlsx'
DataFrame(mean_matrix).to_excel(output_name, sheet_name='Sheet1', index=False, header=True)
print("The time used to execute this is given below")
end = time.time()
print(end - start)
