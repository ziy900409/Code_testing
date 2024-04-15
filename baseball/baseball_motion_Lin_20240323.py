# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:14:21 2024

@author: h7058
"""
# %% 0. import library
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(r"E:\Hsin\git\git\Code_testing\baseball")
# 將read_c3d function 加進現有的工作環境中
import BaseballFunction_20230516 as af

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
save_file_name = r"E:\Hsin\NTSU_lab\Baseball\motion_statistic_data_20240325.xlsx"


with pd.ExcelWriter(save_file_name) as Writer:
    save_finger_data.to_excel(Writer, sheet_name="finger", index=False)
    save_elbow_data.to_excel(Writer, sheet_name="elbow", index=False)
    save_shoulder_data.to_excel(Writer, sheet_name="shoulder", index=False)













