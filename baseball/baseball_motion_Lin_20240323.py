# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:14:21 2024

@author: h7058
"""
# %% 0. import library
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(r"E:\Hsin\git\git\Code_testing\baseball")
# 將read_c3d function 加進現有的工作環境中
import BaseballFunction_20230516 as af
import spm1d

# data path setting
motion_folder_path = r"E:\Hsin\NTSU_lab\Baseball\衛宣博論運動學\\"


# %% 1. read staging file
staging_file = pd.read_excel(r"E:\Hsin\NTSU_lab\Baseball\motion分期肌電用_20240317.xlsx",
                             sheet_name='memo4')
staging_file = staging_file.dropna(axis=0, thresh=14)


file_list = af.Read_File(motion_folder_path,
                         ".xlsx",
                         subfolder=True)

# %% 解決 finger 問題
# 設定擷取資料
save_finger_data = pd.DataFrame({ })

# file_name = 
finger_file_list = []
for file_path in file_list:
    # print(file_path)
    if "finger" in file_path:
        finger_file_list.append(file_path)



for i in range(np.shape(staging_file)[0]):
    motion_name = str(staging_file["Subject"][i]) + "_" +  str(staging_file["Task"][i]) + "_" + \
        str(staging_file["ball"][i]) + "_" + str(staging_file["trial"][i])
    # print(motion_name)    
    for file_path in finger_file_list:
        # print(file_path)
        if motion_name in file_path:
            filepath, tempfilename = os.path.split(file_path)
            print(motion_name, tempfilename)
            # 讀取資料，以及去除不要的欄位
            motion_data = pd.read_excel(file_path, skiprows=3)
            motion_data = motion_data.iloc[10:, :].reset_index(drop=True)
            # motion_data_1 = motion_data.dropna(axis=0, thresh=5)

            delta_time = 1/240
            vel_motion_data = pd.DataFrame((motion_data.iloc[1:, :].values - motion_data.iloc[0:-1, :].values) / delta_time,
                                           columns=motion_data.columns).astype(float) # 將資料轉乘 float
            vel_motion_data.loc[:, 'Frame#'] = motion_data.loc[1:, 'Frame#'].values
            
    
            # 設定分期時間
            kneeTop = motion_data.loc[motion_data['Frame#'] == staging_file["Kneetop"][i]].index.to_numpy()
            footContact = motion_data.loc[motion_data['Frame#'] == staging_file["foot contact"][i]].index.to_numpy()
            ser = motion_data.loc[motion_data['Frame#'] == staging_file["shoulder external rotation"][i]].index.to_numpy()
            release = motion_data.loc[motion_data['Frame#'] == staging_file["release"][i]].index.to_numpy()
            

            filepath, tempfilename = os.path.split(file_path)
            # Knee Top -> foot contact
            add_KT_data = pd.DataFrame({
                'file_name':[tempfilename],
                'phase':['KT-FC'],
                'phase_time':[str(kneeTop[0]+1) + "-" + str(footContact[0]+1)],
                # Angle1:R.IF1-R.IF2-R.IF3-R.IF2
                'max_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].max()],
                'time_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].idxmax(), 'Time']],
                'vel_max_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].max()],
                'vel_min_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].min()],
                'vel_max_time_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].idxmax(), 'Frame#'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].idxmin(), 'Frame#'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                    
                # Angle2:R.IF2-R.IF3-R.IF4-R.IF3
                'max_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].max()],
                'time_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].idxmax(), 'Time']],
                'vel_max_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].max()],
                'vel_min_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].min()],
                'vel_max_time_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].idxmax(), 'Frame#'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].idxmin(), 'Frame#'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                # Angle3:R.IF3-R.IF4-R.IF5-R.IF4
                'max_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].max()],
                'time_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].idxmax(), 'Time']],
                'vel_max_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].max()],
                'vel_min_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].min()],
                'vel_max_time_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].idxmax(), 'Frame#'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].idxmin(), 'Frame#'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                # Angle4:R.MF1-R.MF2-R.MF3-R.MF2
                'max_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].max()],
                'time_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].idxmax(), 'Time']],
                'vel_max_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].max()],
                'vel_min_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].min()],
                'vel_max_time_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].idxmax(), 'Frame#'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].idxmin(), 'Frame#'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                # Angle5:R.MF2-R.MF3-R.MF4-R.MF3
                'max_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].max()],
                'time_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].idxmax(), 'Time']],
                'vel_max_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].max()],
                'vel_min_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].min()],
                'vel_max_time_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].idxmax(), 'Frame#'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].idxmin(), 'Frame#'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                # Angle6:R.MF3-R.MF4-R.MF5-R.MF4
                'max_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].max()],
                'time_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].idxmax(), 'Time']],
                'vel_max_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].max()],
                'vel_min_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].min()],
                'vel_max_time_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].idxmax(), 'Frame#'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].idxmin(), 'Frame#'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                })
            save_finger_data = pd.concat([save_finger_data, add_KT_data], ignore_index=True)  
            # foot contact -> shoulder external rotation
            add_FC_data = pd.DataFrame({
                'file_name':[tempfilename],
                'phase':['FC-SER'],
                'phase_time':[str(footContact[0]+1) + "-" + str(ser[0]+1)],
                # Angle1:R.IF1-R.IF2-R.IF3-R.IF2
                'max_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [motion_data.loc[footContact[0]-1:ser[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].max()],
                'time_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].idxmax(), 'Time']],
                'vel_max_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].max()],
                'vel_min_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].min()],
                'vel_max_time_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].idxmax(), 'Frame#'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].idxmin(), 'Frame#'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                # Angle2:R.IF2-R.IF3-R.IF4-R.IF3
                'max_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [motion_data.loc[footContact[0]-1:ser[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].max()],
                'time_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].idxmax(), 'Time']],
                'vel_max_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].max()],
                'vel_min_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].min()],
                'vel_max_time_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].idxmax(), 'Frame#'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].idxmin(), 'Frame#'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                # Angle3:R.IF3-R.IF4-R.IF5-R.IF4
                'max_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [motion_data.loc[footContact[0]-1:ser[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].max()],
                'time_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].idxmax(), 'Time']],
                'vel_max_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].max()],
                'vel_min_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].min()],
                'vel_max_time_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].idxmax(), 'Frame#'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].idxmin(), 'Frame#'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                # Angle4:R.MF1-R.MF2-R.MF3-R.MF2
                'max_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [motion_data.loc[footContact[0]-1:ser[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].max()],
                'time_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].idxmax(), 'Time']],
                'vel_max_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].max()],
                'vel_min_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].min()],
                'vel_max_time_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].idxmax(), 'Frame#'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].idxmin(), 'Frame#'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                # Angle5:R.MF2-R.MF3-R.MF4-R.MF3
                'max_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [motion_data.loc[footContact[0]-1:ser[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].max()],
                'time_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].idxmax(), 'Time']],
                'vel_max_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].max()],
                'vel_min_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].min()],
                'vel_max_time_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].idxmax(), 'Frame#'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].idxmin(), 'Frame#'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                # Angle6:R.MF3-R.MF4-R.MF5-R.MF4
                'max_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [motion_data.loc[footContact[0]-1:ser[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].max()],
                'time_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].idxmax(), 'Time']],
                'vel_max_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].max()],
                'vel_min_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].min()],
                'vel_max_time_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].idxmax(), 'Frame#'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].idxmin(), 'Frame#'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                })
            save_finger_data = pd.concat([save_finger_data, add_FC_data], ignore_index=True) 
            
            # shoulder external rotation -> release
            add_SER_data = pd.DataFrame({
                'file_name':[tempfilename],
                'phase':['SER-RE'],
                'phase_time':[str(ser[0]+1) + "-" + str(release[0]+1)],
                # Angle1:R.IF1-R.IF2-R.IF3-R.IF2
                'max_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [motion_data.loc[ser[0]-1:release[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].max()],
                'time_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].idxmax(), 'Time']],
                'vel_max_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [vel_motion_data.loc[ser[0]:release[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].max()],
                'vel_min_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [vel_motion_data.loc[ser[0]:release[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].min()],
                'vel_max_time_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].idxmax(), 'Frame#'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_Angle1:R.IF1-R.IF2-R.IF3-R.IF2':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'Angle1:R.IF1-R.IF2-R.IF3-R.IF2'].idxmin(), 'Frame#'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                # Angle2:R.IF2-R.IF3-R.IF4-R.IF3
                'max_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [motion_data.loc[ser[0]-1:release[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].max()],
                'time_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].idxmax(), 'Time']],
                'vel_max_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [vel_motion_data.loc[ser[0]:release[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].max()],
                'vel_min_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [vel_motion_data.loc[ser[0]:release[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].min()],
                'vel_max_time_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].idxmax(), 'Frame#'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_Angle2:R.IF2-R.IF3-R.IF4-R.IF3':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'Angle2:R.IF2-R.IF3-R.IF4-R.IF3'].idxmin(), 'Frame#'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                # Angle3:R.IF3-R.IF4-R.IF5-R.IF4
                'max_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [motion_data.loc[ser[0]-1:release[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].max()],
                'time_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].idxmax(), 'Time']],
                'vel_max_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [vel_motion_data.loc[ser[0]:release[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].max()],
                'vel_min_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [vel_motion_data.loc[ser[0]:release[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].min()],
                'vel_max_time_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].idxmax(), 'Frame#'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_Angle3:R.IF3-R.IF4-R.IF5-R.IF4':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'Angle3:R.IF3-R.IF4-R.IF5-R.IF4'].idxmin(), 'Frame#'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                # Angle4:R.MF1-R.MF2-R.MF3-R.MF2
                'max_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [motion_data.loc[ser[0]-1:release[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].max()],
                'time_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].idxmax(), 'Time']],
                'vel_max_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [vel_motion_data.loc[ser[0]:release[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].max()],
                'vel_min_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [vel_motion_data.loc[ser[0]:release[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].min()],
                'vel_max_time_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].idxmax(), 'Frame#'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_Angle4:R.MF1-R.MF2-R.MF3-R.MF2':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'Angle4:R.MF1-R.MF2-R.MF3-R.MF2'].idxmin(), 'Frame#'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                # Angle5:R.MF2-R.MF3-R.MF4-R.MF3
                'max_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [motion_data.loc[ser[0]-1:release[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].max()],
                'time_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].idxmax(), 'Time']],
                'vel_max_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [vel_motion_data.loc[ser[0]:release[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].max()],
                'vel_min_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [vel_motion_data.loc[ser[0]:release[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].min()],
                'vel_max_time_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].idxmax(), 'Frame#'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_Angle5:R.MF2-R.MF3-R.MF4-R.MF3':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'Angle5:R.MF2-R.MF3-R.MF4-R.MF3'].idxmin(), 'Frame#'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                # Angle6:R.MF3-R.MF4-R.MF5-R.MF4
                'max_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [motion_data.loc[ser[0]-1:release[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].max()],
                'time_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].idxmax(), 'Time']],
                'vel_max_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [vel_motion_data.loc[ser[0]:release[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].max()],
                'vel_min_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [vel_motion_data.loc[ser[0]:release[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].min()],
                'vel_max_time_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].idxmax(), 'Frame#'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_Angle6:R.MF3-R.MF4-R.MF5-R.MF4':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'Angle6:R.MF3-R.MF4-R.MF5-R.MF4'].idxmin(), 'Frame#'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                })
            save_finger_data = pd.concat([save_finger_data, add_SER_data], ignore_index=True) 
                

# %% 解決 elbow 問題
# 設定擷取資料
save_elbow_data = pd.DataFrame({
               
                                    })

# file_name = 
elbow_file_list = []
for file_path in file_list:
    # print(file_path)
    if "Elbow" in file_path:
        elbow_file_list.append(file_path)
        print(file_path)



for i in range(np.shape(staging_file)[0]):
    motion_name = str(staging_file["Subject"][i]) + "_" +  str(staging_file["Task"][i]) + "_" + \
        str(staging_file["ball"][i]) + "_" + str(staging_file["trial"][i])
    # print(motion_name)    
    for file_path in elbow_file_list:
        # print(file_path)
        if motion_name in file_path:
            # print(file_path)
            print(1)
            # 讀取資料，以及去除不要的欄位
            motion_data = pd.read_excel(file_path, skiprows=6)
            motion_data = motion_data.loc[1:, ['Unnamed: 0', 'R Elbow Flex-Ext', 'R Elbow Var-Val',
                                               'R Pronation-Supination', 'R Wrist Flex-Ext', 'R Wrist Rad-ula']
                                          ].reset_index(drop=True).astype(float) # 設定數值類型與重新定義index
            
            # file_path = r"E:\Hsin\NTSU_lab\Baseball\衛宣博論運動學\T1\T1前臂原資料\S01_T1_FB_1_Elbow.xlsx"
            
            delta_time = 1/240
            vel_motion_data = pd.DataFrame((motion_data.iloc[1:, :].values - motion_data.iloc[0:-1, :].values) / delta_time,
                                           columns=motion_data.columns)
            vel_motion_data.loc[:, 'Unnamed: 0'] = motion_data.loc[1:, 'Unnamed: 0'].values

            
            # 設定分期時間
            kneeTop = motion_data.loc[motion_data['Unnamed: 0'] == staging_file["Kneetop"][i]].index.to_numpy()
            footContact = motion_data.loc[motion_data['Unnamed: 0'] == staging_file["foot contact"][i]].index.to_numpy()
            ser = motion_data.loc[motion_data['Unnamed: 0'] == staging_file["shoulder external rotation"][i]].index.to_numpy()
            release = motion_data.loc[motion_data['Unnamed: 0'] == staging_file["release"][i]].index.to_numpy()
            
            filepath, tempfilename = os.path.split(file_path)
            # Knee Top -> foot contact
            add_KT_data = pd.DataFrame({
                'file_name':[tempfilename],
                'phase':['KT-FC'],
                'phase_time':[str(kneeTop[0]+1) + "-" + str(footContact[0]+1)],
                # R Elbow Flex-Ext
                'max_R Elbow Flex-Ext':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'R Elbow Flex-Ext'].max()],
                'time_R Elbow Flex-Ext':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'R Elbow Flex-Ext'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Elbow Flex-Ext':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Elbow Flex-Ext'].max()],
                'vel_min_R Elbow Flex-Ext':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Elbow Flex-Ext'].min()],
                'vel_max_time_R Elbow Flex-Ext':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Elbow Flex-Ext'].idxmax(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_R Elbow Flex-Ext':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Elbow Flex-Ext'].idxmin(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                # R Elbow Var-Val
                'max_R Elbow Var-Val':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'R Elbow Var-Val'].max()],
                'time_R Elbow Var-Val':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'R Elbow Var-Val'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Elbow Var-Val':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Elbow Var-Val'].max()],
                'vel_min_R Elbow Var-Val':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Elbow Var-Val'].min()],
                'vel_max_time_R Elbow Var-Val':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Elbow Var-Val'].idxmax(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_R Elbow Var-Val':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Elbow Var-Val'].idxmin(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                # R Pronation-Supination
                'max_R Pronation-Supination':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'R Pronation-Supination'].max()],
                'time_R Pronation-Supination':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'R Pronation-Supination'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Pronation-Supination':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Pronation-Supination'].max()],
                'vel_min_R Pronation-Supination':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Pronation-Supination'].min()],
                'vel_max_time_R Pronation-Supination':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Pronation-Supination'].idxmax(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_R Pronation-Supination':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Pronation-Supination'].idxmin(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                # R Wrist Flex-Ext
                'max_R Wrist Flex-Ext':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'R Wrist Flex-Ext'].max()],
                'time_R Wrist Flex-Ext':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'R Wrist Flex-Ext'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Wrist Flex-Ext':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Wrist Flex-Ext'].max()],
                'vel_min_R Wrist Flex-Ext':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Wrist Flex-Ext'].min()],
                'vel_max_time_R Wrist Flex-Ext':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Wrist Flex-Ext'].idxmax(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_R Wrist Flex-Ext':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Wrist Flex-Ext'].idxmin(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                # R Wrist Rad-ula
                'max_R Wrist Rad-ula':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'R Wrist Rad-ula'].max()],
                'time_R Wrist Rad-ula':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'R Wrist Rad-ula'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Wrist Rad-ula':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Wrist Rad-ula'].max()],
                'vel_min_R Wrist Rad-ula':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Wrist Rad-ula'].min()],
                'vel_max_time_R Wrist Rad-ula':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Wrist Rad-ula'].idxmax(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_R Wrist Rad-ula':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Wrist Rad-ula'].idxmin(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                })
            save_elbow_data = pd.concat([save_elbow_data, add_KT_data], ignore_index=True)    
            # foot contact -> shoulder external rotation
            add_FC_data = pd.DataFrame({
                'file_name':[tempfilename],
                'phase':['FC-SER'],
                'phase_time':[str(footContact[0]+1) + "-" + str(ser[0]+1)],
                # R Elbow Flex-Ext
                'max_R Elbow Flex-Ext':
                    [motion_data.loc[footContact[0]-1:ser[0], 'R Elbow Flex-Ext'].max()],
                'time_R Elbow Flex-Ext':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'R Elbow Flex-Ext'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Elbow Flex-Ext':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Elbow Flex-Ext'].max()],
                'vel_min_R Elbow Flex-Ext':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Elbow Flex-Ext'].min()],
                'vel_max_time_R Elbow Flex-Ext':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Elbow Flex-Ext'].idxmax(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_R Elbow Flex-Ext':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Elbow Flex-Ext'].idxmin(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                # R Elbow Var-Val
                'max_R Elbow Var-Val':
                    [motion_data.loc[footContact[0]-1:ser[0], 'R Elbow Var-Val'].max()],
                'time_R Elbow Var-Val':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'R Elbow Var-Val'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Elbow Var-Val':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Elbow Var-Val'].max()],
                'vel_min_R Elbow Var-Val':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Elbow Var-Val'].min()],
                'vel_max_time_R Elbow Var-Val':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Elbow Var-Val'].idxmax(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_R Elbow Var-Val':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Elbow Var-Val'].idxmin(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                # R Pronation-Supination
                'max_R Pronation-Supination':
                    [motion_data.loc[footContact[0]-1:ser[0], 'R Pronation-Supination'].max()],
                'time_R Pronation-Supination':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'R Pronation-Supination'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Pronation-Supination':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Pronation-Supination'].max()],
                'vel_min_R Pronation-Supination':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Pronation-Supination'].min()],
                'vel_max_time_R Pronation-Supination':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Pronation-Supination'].idxmax(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_R Pronation-Supination':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Pronation-Supination'].idxmin(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                # R Wrist Flex-Ext
                'max_R Wrist Flex-Ext':
                    [motion_data.loc[footContact[0]-1:ser[0], 'R Wrist Flex-Ext'].max()],
                'time_R Wrist Flex-Ext':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'R Wrist Flex-Ext'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Wrist Flex-Ext':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Wrist Flex-Ext'].max()],
                'vel_min_R Wrist Flex-Ext':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Wrist Flex-Ext'].min()],
                'vel_max_time_R Wrist Flex-Ext':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Wrist Flex-Ext'].idxmax(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_R Wrist Flex-Ext':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Wrist Flex-Ext'].idxmin(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                # R Wrist Rad-ula
                'max_R Wrist Rad-ula':
                    [motion_data.loc[footContact[0]-1:ser[0], 'R Wrist Rad-ula'].max()],
                'time_R Wrist Rad-ula':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'R Wrist Rad-ula'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Wrist Rad-ula':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Wrist Rad-ula'].max()],
                'vel_min_R Wrist Rad-ula':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Wrist Rad-ula'].min()],
                'vel_max_time_R Wrist Rad-ula':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Wrist Rad-ula'].idxmax(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_R Wrist Rad-ula':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Wrist Rad-ula'].idxmin(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                })
            save_elbow_data = pd.concat([save_elbow_data, add_FC_data], ignore_index=True)   
            # shoulder external rotation -> release
            add_SER_data = pd.DataFrame({
                'file_name':[tempfilename],
                'phase':['SER-RE'],
                'phase_time':[str(ser[0]+1) + "-" + str(release[0]+1)],
                # R Elbow Flex-Ext
                'max_R Elbow Flex-Ext':
                    [motion_data.loc[ser[0]-1:release[0], 'R Elbow Flex-Ext'].max()],
                'time_R Elbow Flex-Ext':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'R Elbow Flex-Ext'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Elbow Flex-Ext':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Elbow Flex-Ext'].max()],
                'vel_min_R Elbow Flex-Ext':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Elbow Flex-Ext'].min()],
                'vel_max_time_R Elbow Flex-Ext':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Elbow Flex-Ext'].idxmax(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_R Elbow Flex-Ext':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Elbow Flex-Ext'].idxmin(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                # R Elbow Flex-Ext
                'max_R Elbow Var-Val':
                    [motion_data.loc[ser[0]-1:release[0], 'R Elbow Var-Val'].max()],
                'time_R Elbow Var-Val':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'R Elbow Var-Val'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Elbow Var-Val':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Elbow Var-Val'].max()],
                'vel_min_R Elbow Var-Val':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Elbow Var-Val'].min()],
                'vel_max_time_R Elbow Var-Val':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Elbow Var-Val'].idxmax(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_R Elbow Var-Val':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Elbow Var-Val'].idxmin(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                # R Pronation-Supination
                'max_R Pronation-Supination':
                    [motion_data.loc[ser[0]-1:release[0], 'R Pronation-Supination'].max()],
                'time_R Pronation-Supination':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'R Pronation-Supination'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Pronation-Supination':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Pronation-Supination'].max()],
                'vel_min_R Pronation-Supination':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Pronation-Supination'].min()],
                'vel_max_time_R Pronation-Supination':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Pronation-Supination'].idxmax(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_R Pronation-Supination':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Pronation-Supination'].idxmin(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                # R Wrist Flex-Ext
                'max_R Wrist Flex-Ext':
                    [motion_data.loc[ser[0]-1:release[0], 'R Wrist Flex-Ext'].max()],
                'time_R Wrist Flex-Ext':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'R Wrist Flex-Ext'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Wrist Flex-Ext':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Wrist Flex-Ext'].max()],
                'vel_min_R Wrist Flex-Ext':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Wrist Flex-Ext'].min()],
                'vel_max_time_R Wrist Flex-Ext':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Wrist Flex-Ext'].idxmax(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_R Wrist Flex-Ext':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Wrist Flex-Ext'].idxmin(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                # R Wrist Rad-ula
                'max_R Wrist Rad-ula':
                    [motion_data.loc[ser[0]-1:release[0], 'R Wrist Rad-ula'].max()],
                'time_R Wrist Rad-ula':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'R Wrist Rad-ula'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Wrist Rad-ula':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Wrist Flex-Ext'].max()],
                'vel_min_R Wrist Rad-ula':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Wrist Flex-Ext'].min()],
                'vel_max_time_R Wrist Rad-ula':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Wrist Flex-Ext'].idxmax(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_R Wrist Rad-ula':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Wrist Flex-Ext'].idxmin(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                })
            save_elbow_data = pd.concat([save_elbow_data, add_SER_data], ignore_index=True)

# %% 解決 Shoulder 問題
save_shoulder_data = pd.DataFrame({ })

# file_name = 
shoulder_file_list = []
for file_path in file_list:
    # print(file_path)
    if "Shoulder" in file_path:
        shoulder_file_list.append(file_path)
        print(file_path)



for i in range(np.shape(staging_file)[0]):
    motion_name = str(staging_file["Subject"][i]) + "_" +  str(staging_file["Task"][i]) + "_" + \
        str(staging_file["ball"][i]) + "_" + str(staging_file["trial"][i])
    # print(motion_name)    
    for file_path in shoulder_file_list:
        # print(file_path)
        if motion_name in file_path:
            # print(file_path)
            filepath, tempfilename = os.path.split(file_path)
            print(motion_name, tempfilename)
            # 讀取資料，以及去除不要的欄位
            motion_data = pd.read_excel(file_path, skiprows=6)
            motion_data = motion_data.loc[1:, ['Unnamed: 0', 'R Shoulder Elevation', 'R Shoulder Hor Add',
                                               'R Shoulder Rotation']
                                          ].reset_index(drop=True).astype(float) # 設定數值類型與重新定義index
            
            
            
            delta_time = 1/240
            vel_motion_data = pd.DataFrame((motion_data.iloc[1:, :].values - motion_data.iloc[0:-1, :].values) / delta_time,
                                           columns=motion_data.columns).astype(float) # 將資料轉乘 float
            vel_motion_data.loc[:, 'Unnamed: 0'] = motion_data.loc[1:, 'Unnamed: 0'].values
            # 設定分期時間
            kneeTop = motion_data.loc[motion_data['Unnamed: 0'] == staging_file["Kneetop"][i]].index.to_numpy()
            footContact = motion_data.loc[motion_data['Unnamed: 0'] == staging_file["foot contact"][i]].index.to_numpy()
            ser = motion_data.loc[motion_data['Unnamed: 0'] == staging_file["shoulder external rotation"][i]].index.to_numpy()
            release = motion_data.loc[motion_data['Unnamed: 0'] == staging_file["release"][i]].index.to_numpy()

            
            # Knee Top -> foot contact
            add_KT_data = pd.DataFrame({
                'file_name':[tempfilename],
                'phase':['KT-FC'],
                'phase_time':[str(kneeTop[0]+1) + "-" + str(footContact[0]+1)],
                'max_R Shoulder Elevation':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'R Shoulder Elevation'].max()],
                'time_R Shoulder Elevation':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'R Shoulder Elevation'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Shoulder Elevation':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Shoulder Elevation'].max()],
                'vel_min_R Shoulder Elevation':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Shoulder Elevation'].min()],
                'vel_max_time_R Shoulder Elevation':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Shoulder Elevation'].idxmax(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_R Shoulder Elevation':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Shoulder Elevation'].idxmin(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                # R Shoulder Hor Add
                'max_R Shoulder Hor Add':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'R Shoulder Hor Add'].max()],
                'time_R Shoulder Hor Add':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'R Shoulder Hor Add'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Shoulder Hor Add':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Shoulder Hor Add'].max()],
                'vel_min_R Shoulder Hor Add':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Shoulder Hor Add'].min()],
                'vel_max_time_R Shoulder Hor Add':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Shoulder Hor Add'].idxmax(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_R Shoulder Hor Add':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Shoulder Hor Add'].idxmin(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                # R Shoulder Rotation
                'max_R Shoulder Rotation':
                    [motion_data.loc[kneeTop[0]-1:footContact[0], 'R Shoulder Rotation'].max()],
                'time_R Shoulder Rotation':
                    [motion_data.loc[motion_data.loc[kneeTop[0]-1:footContact[0], 'R Shoulder Rotation'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Shoulder Rotation':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Shoulder Rotation'].max()],
                'vel_min_R Shoulder Rotation':
                    [vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Shoulder Rotation'].min()],
                'vel_max_time_R Shoulder Rotation':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Shoulder Rotation'].idxmax(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                'vel_min_time_R Shoulder Rotation':
                    [(vel_motion_data.loc[vel_motion_data.loc[kneeTop[0]:footContact[0], 'R Shoulder Rotation'].idxmin(), 'Unnamed: 0'] -\
                      kneeTop[0]-2)/(footContact[0] - kneeTop[0])],
                })
            save_shoulder_data = pd.concat([save_shoulder_data, add_KT_data], ignore_index=True)    
            # foot contact -> shoulder external rotation
            add_FC_data = pd.DataFrame({
                'file_name':[tempfilename],
                'phase':['FC-SER'],
                'phase_time':[str(footContact[0]+1) + "-" + str(ser[0]+1)],
                # R Shoulder Elevation
                'max_R Shoulder Elevation':
                    [motion_data.loc[footContact[0]-1:ser[0], 'R Shoulder Elevation'].max()],
                'time_R Shoulder Elevation':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'R Shoulder Elevation'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Shoulder Elevation':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Shoulder Elevation'].max()],
                'vel_min_R Shoulder Elevation':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Shoulder Elevation'].min()],
                'vel_max_time_R Shoulder Elevation':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]-1:ser[0], 'R Shoulder Elevation'].idxmax(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_R Shoulder Elevation':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]-1:ser[0], 'R Shoulder Elevation'].idxmin(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                # R Shoulder Hor Add
                'max_R Shoulder Hor Add':
                    [motion_data.loc[footContact[0]-1:ser[0], 'R Shoulder Hor Add'].max()],
                'time_R Shoulder Hor Add':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'R Shoulder Hor Add'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Shoulder Hor Add':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Shoulder Hor Add'].max()],
                'vel_min_R Shoulder Hor Add':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Shoulder Hor Add'].min()],
                'vel_max_time_R Shoulder Hor Add':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Shoulder Hor Add'].idxmax(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_R Shoulder Hor Add':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Shoulder Hor Add'].idxmin(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                # R Shoulder Rotation
                'max_R Shoulder Rotation':
                    [motion_data.loc[footContact[0]-1:ser[0], 'R Shoulder Rotation'].max()],
                'time_R Shoulder Rotation':
                    [motion_data.loc[motion_data.loc[footContact[0]-1:ser[0], 'R Shoulder Rotation'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Shoulder Rotation':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Shoulder Rotation'].max()],
                'vel_min_R Shoulder Rotation':
                    [vel_motion_data.loc[footContact[0]:ser[0], 'R Shoulder Rotation'].min()],
                'vel_max_time_R Shoulder Rotation':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Shoulder Rotation'].idxmax(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])],
                'vel_min_time_R Shoulder Rotation':
                    [(vel_motion_data.loc[vel_motion_data.loc[footContact[0]:ser[0], 'R Shoulder Rotation'].idxmin(), 'Unnamed: 0'] -\
                      footContact[0]-2)/(ser[0] - footContact[0])]
                    })
            save_shoulder_data = pd.concat([save_shoulder_data, add_FC_data], ignore_index=True)   
            # shoulder external rotation -> release
            add_SER_data = pd.DataFrame({
                'file_name':[tempfilename],
                'phase':['SER-RE'],
                'phase_time':[str(ser[0]+1) + "-" + str(release[0]+1)],
                # R Shoulder Elevation
                'max_R Shoulder Elevation':
                    [motion_data.loc[ser[0]-1:release[0], 'R Shoulder Elevation'].max()],
                'time_R Shoulder Elevation':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'R Shoulder Elevation'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Shoulder Elevation':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Shoulder Elevation'].max()],
                'vel_min_R Shoulder Elevation':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Shoulder Elevation'].min()],
                'vel_max_time_R Shoulder Elevation':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Shoulder Elevation'].idxmax(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_R Shoulder Elevation':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Shoulder Elevation'].idxmin(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                # R Shoulder Hor Add
                'max_R Shoulder Hor Add':
                    [motion_data.loc[ser[0]-1:release[0], 'R Shoulder Hor Add'].max()],
                'time_R Shoulder Hor Add':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'R Shoulder Hor Add'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Shoulder Hor Add':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Shoulder Hor Add'].max()],
                'vel_min_R Shoulder Hor Add':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Shoulder Hor Add'].min()],
                'vel_max_time_R Shoulder Hor Add':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Shoulder Hor Add'].idxmax(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_R Shoulder Hor Add':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Shoulder Hor Add'].idxmin(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                # R Shoulder Rotation
                'max_R Shoulder Rotation':
                    [motion_data.loc[ser[0]-1:release[0], 'R Shoulder Rotation'].max()],
                'time_R Shoulder Rotation':
                    [motion_data.loc[motion_data.loc[ser[0]-1:release[0], 'R Shoulder Rotation'].idxmax(), 'Unnamed: 0']],
                'vel_max_R Shoulder Rotation':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Shoulder Rotation'].max()],
                'vel_min_R Shoulder Rotation':
                    [vel_motion_data.loc[ser[0]:release[0], 'R Shoulder Rotation'].min()],
                'vel_max_time_R Shoulder Rotation':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Shoulder Rotation'].idxmax(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                'vel_min_time_R Shoulder Rotation':
                    [(vel_motion_data.loc[vel_motion_data.loc[ser[0]:release[0], 'R Shoulder Rotation'].idxmin(), 'Unnamed: 0'] -\
                      ser[0]-2)/(release[0] - ser[0])],
                    })
            save_shoulder_data = pd.concat([save_shoulder_data, add_SER_data], ignore_index=True)


# %% 輸出成EXCEL
# save_file_name = r"E:\Hsin\NTSU_lab\Baseball\motion_statistic_data_20240325.xlsx"


# with pd.ExcelWriter(save_file_name) as Writer:
#     save_finger_data.to_excel(Writer, sheet_name="finger", index=False)
#     save_elbow_data.to_excel(Writer, sheet_name="elbow", index=False)
#     save_shoulder_data.to_excel(Writer, sheet_name="shoulder", index=False)





# %% 繪圖用
staging_file = pd.read_excel(r"E:\Hsin\NTSU_lab\Baseball\motion分期肌電用_20240420.xlsx",
                             sheet_name='T1_memo2')
staging_file = staging_file.dropna(axis=0, thresh=14)


pos_cloname = ["食指遠端指關節角度", "食指近端指關節角度", "食指掌指關節角度",
                "中指遠端指關節角度", "中指近端指關節角度", "中指掌指關節角度"]
vel_colname = ["食指遠端指關節角速度", "食指近端指關節角速度", "食指掌指關節角速度",
               "中指遠端指關節角速度", "中指近端指關節角速度", "中指掌指關節角速度"]
subject_name = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10",
                "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20"]
fast_group = ["S07", "S10", "S11", "S12", "S14", "S15", "S16", "S17", "S19", "S20"]
slow_group = ["S01", "S02", "S03", "S04", "S05", "S06", "S08", "S09", "S13", "S18"]
# %%
# create data
finger_distal_angle = pd.DataFrame(np.zeros([202, 20]), 
                                   columns = subject_name)
finger_proximal_angle = pd.DataFrame(np.zeros([202, 20]), 
                                     columns = subject_name)
finger_plam_angle = pd.DataFrame(np.zeros([202, 20]), 
                                 columns = subject_name)
med_distal_angle = pd.DataFrame(np.zeros([202, 20]), 
                                columns = subject_name)
med_proximal_angle = pd.DataFrame(np.zeros([202, 20]), 
                                  columns = subject_name)
med_plam_angle = pd.DataFrame(np.zeros([202, 20]), 
                              columns = subject_name)
finger_distal_vel = pd.DataFrame(np.zeros([202, 20]), 
                                 columns = subject_name)
finger_proximal_vel = pd.DataFrame(np.zeros([202, 20]), 
                                   columns = subject_name)
finger_plam_vel = pd.DataFrame(np.zeros([202, 20]), 
                               columns = subject_name)
med_distal_vel = pd.DataFrame(np.zeros([202, 20]), 
                              columns = subject_name)
med_proximal_vel = pd.DataFrame(np.zeros([202, 20]), 
                                columns = subject_name)
med_plam_vel = pd.DataFrame(np.zeros([202, 20]), 
                            columns = subject_name)



 
for subject in range(len(staging_file["Subject"])):
    print(staging_file["Subject"][subject])
    finger_data = pd.read_excel(r"E:\Hsin\NTSU_lab\Baseball\finger_motion_figure.xlsx",
                                sheet_name = staging_file["Subject"][subject])
    # define time
    # footContact = staging_file.loc[subject, "foot contact"]
    SER = staging_file.loc[subject, "shoulder external rotation"]
    # 找到符合條件的索引位置
    SER_idx = finger_data.index[finger_data["Frame#"] == SER][0]
    # 1. stage2
    angle_data = finger_data.loc[:SER_idx, pos_cloname]
    vel_data = finger_data.loc[1:SER_idx, vel_colname]
    # 內插成 100 data point
    Y = spm1d.util.interp(angle_data.values.T, Q=101)
    Z = spm1d.util.interp(vel_data.values.T, Q=101)
    # 儲存食指角度
    finger_distal_angle.loc[:100, staging_file["Subject"][subject]] = Y[0, :]
    finger_proximal_angle.loc[:100, staging_file["Subject"][subject]] = Y[1, :]
    finger_plam_angle.loc[:100, staging_file["Subject"][subject]] = Y[2, :]
    med_distal_angle.loc[:100, staging_file["Subject"][subject]] = Y[3, :]
    med_proximal_angle.loc[:100, staging_file["Subject"][subject]] = Y[4, :]
    med_plam_angle.loc[:100, staging_file["Subject"][subject]] = Y[5, :]
    # 儲存中指角度
    finger_distal_vel.loc[:100, staging_file["Subject"][subject]] = Z[0, :]
    finger_proximal_vel.loc[:100, staging_file["Subject"][subject]] = Z[1, :]
    finger_plam_vel.loc[:100, staging_file["Subject"][subject]] = Z[2, :]
    med_distal_vel.loc[:100, staging_file["Subject"][subject]] = Z[3, :]
    med_proximal_vel.loc[:100, staging_file["Subject"][subject]] = Z[4, :]
    med_plam_vel.loc[:100, staging_file["Subject"][subject]] = Z[5, :]
    # 2. stage3
    angle_data = finger_data.loc[SER_idx+1:, pos_cloname]
    vel_data = finger_data.loc[SER_idx+1:, vel_colname].dropna(axis=0)
    # 內插手指角速度
    Y = spm1d.util.interp(angle_data.values.T, Q=101)
    Z = spm1d.util.interp(vel_data.values.T, Q=101)
    # 儲存食指角度
    finger_distal_angle.loc[100+1:, staging_file["Subject"][subject]] = Y[0, :]
    finger_proximal_angle.loc[100+1:, staging_file["Subject"][subject]] = Y[1, :]
    finger_plam_angle.loc[100+1:, staging_file["Subject"][subject]] = Y[2, :]
    med_distal_angle.loc[100+1:, staging_file["Subject"][subject]] = Y[3, :]
    med_proximal_angle.loc[100+1:, staging_file["Subject"][subject]] = Y[4, :]
    med_plam_angle.loc[100+1:, staging_file["Subject"][subject]] = Y[5, :]
    # 儲存中指角度
    finger_distal_vel.loc[100+1:, staging_file["Subject"][subject]] = Z[0, :]
    finger_proximal_vel.loc[100+1:, staging_file["Subject"][subject]] = Z[1, :]
    finger_plam_vel.loc[100+1:, staging_file["Subject"][subject]] = Z[2, :]
    med_distal_vel.loc[100+1:, staging_file["Subject"][subject]] = Z[3, :]
    med_proximal_vel.loc[100+1:, staging_file["Subject"][subject]] = Z[4, :]
    med_plam_vel.loc[100+1:, staging_file["Subject"][subject]] = Z[5, :]
    
    
# %%
# coluns name
finger_columns = ["食指遠端指關節角度", "食指近端指關節角度", "食指掌指關節角度",
                  "食指遠端指關節角速度", "食指近端指關節角速度", "食指掌指關節角速度"]
med_columns = ["中指遠端指關節角度", "中指近端指關節角度", "中指掌指關節角度",
               "中指遠端指關節角速度", "中指近端指關節角速度", "中指掌指關節角速度"]

# create multi-dimension matrix
fast_dict = np.zeros((6, # muscle name without time
                      202, # time length
                      10)) # subject number


fast_dict[0, :, :] = finger_distal_angle.loc[:, fast_group]
fast_dict[0, :, :] = finger_proximal_angle.loc[:, fast_group]
fast_dict[0, :, :] = finger_plam_angle.loc[:, fast_group]
fast_dict[0, :, :] = finger_distal_vel.loc[:, fast_group]
fast_dict[0, :, :] = finger_proximal_vel.loc[:, fast_group]
fast_dict[0, :, :] = finger_plam_vel.loc[:, fast_group]


slow_dict = np.zeros((6, # muscle name without time
                      202, # time length
                      10)) # subject number
slow_dict[0, :, :] = finger_distal_angle.loc[:, slow_group]
slow_dict[0, :, :] = finger_proximal_angle.loc[:, slow_group]
slow_dict[0, :, :] = finger_plam_angle.loc[:, slow_group]
slow_dict[0, :, :] = finger_distal_vel.loc[:, slow_group]
slow_dict[0, :, :] = finger_proximal_vel.loc[:, slow_group]
slow_dict[0, :, :] = finger_plam_vel.loc[:, slow_group]



# 設定圖片大小
# 畫第一條線
# save = savepath + "\\mean_std_" + filename + ".jpg"
# n = int(math.ceil((np.shape(type2_dict)[0]) /2))

# 設置圖片大小
# plt.figure(figsize=(2*n+1,10))
# 設定繪圖格式與字體
# plt.style.use('seaborn-white')
# 顯示輸入中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
palette = plt.get_cmap('Set1')
fig, axs = plt.subplots(2, 3, figsize = (10,12), sharex='col')
for i in range(np.shape(fast_dict)[0]):
    # 確定繪圖順序與位置
    x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
    color = palette(0) # 設定顏色
    iters = list(np.linspace(-release[0], release[1], 
                             len(type1_dict[0, :, 0])))
    # 設定計算資料
    avg1 = np.mean(type1_dict[i, :, :], axis=1) # 計算平均
    std1 = np.std(type1_dict[i, :, :], axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
    axs[x, y].plot(iters, avg1, color=color, label='before', linewidth=3)
    axs[x, y].fill_between(iters, r1, r2, color=color, alpha=0.2)
    # 找所有數值的最大值，方便畫括弧用
    yy = max(r2)
    # 畫第二條線
    color = palette(1) # 設定顏色
    avg2 = np.mean(type2_dict[i, :, :], axis=1) # 計畫平均
    std2 = np.std(type2_dict[i, :, :], axis=1) # 計算標準差
    r1 = list(map(lambda x: x[0]-x[1], zip(avg2, std2))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg2, std2)))
    # 找所有數值的最大值，方便畫括弧用
    yy = max([yy, max(r2)])
    axs[x, y].plot(iters, avg2, color=color, label='after', linewidth=3) # 畫平均線
    axs[x, y].fill_between(iters, r1, r2, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊
    # 圖片的格式設定
    axs[x, y].set_title(example_data.columns[i+1], fontsize=12)
    axs[x, y].legend(loc="lower left") # 圖例位置
    axs[x, y].grid(True, linestyle='-.')
    # 畫放箭時間
    axs[x, y].set_xlim(-(release[0]), release[1])
    axs[x, y].axvline(x=0, color = 'darkslategray', linewidth=1, linestyle = '--')

    
plt.suptitle(str("mean std cloud: " + filename), fontsize=16)
plt.tight_layout()
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("time (second)", fontsize = 14)
plt.ylabel("muscle activation (%)", fontsize = 14)
plt.savefig(save, dpi=200, bbox_inches = "tight")
plt.show()    





