# -*- coding: utf-8 -*-
"""
Created on Sun May 11 19:06:48 2025

@author: User
"""

import pandas as pd
import numpy as np
import warnings
import os

# %% Reading all of data path
# using a recursive loop to traverse each folder
# and find the file extension has .csv
def Read_File(file_path, file_type, subfolder=None):
    '''
    Parameters
    ----------
    x : str
        給予欲讀取資料之路徑.
    y : str
        給定欲讀取資料之副檔名.
    subfolder : boolean, optional
        是否子資料夾一起讀取. The default is 'None'.

    Returns
    -------
    csv_file_list : list
        回給所有路徑下的資料絕對路徑.

    '''
    # if subfolder = True, the function will run with subfolder

    csv_file_list = []
    
    if subfolder:
        file_list_1 = []
        for dirPath, dirNames, fileNames in os.walk(file_path):
            # file_list = os.walk(folder_name)
            file_list_1.append(dirPath)
        # need to change here [1:]
        for ii in file_list_1[1:]:
            file_list = os.listdir(ii)
            for iii in file_list:
                if os.path.splitext(iii)[1] == file_type:
                    # replace "\\" to '/', due to MAC version
                    file_list_name = ii + '\\' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(file_path)                
        for i in folder_list:
            if os.path.splitext(i)[1] == file_type:
                # replace "\\" to '/', due to MAC version
                file_list_name = file_path + "\\" + i
                csv_file_list.append(file_list_name)                
        
    return csv_file_list

# %%

file_list = Read_File(r"D:\Hsin\NTSU_lab\WALK",
                      file_type=".xlsx")

columns_name = [
    "FP time", "SI time(s)", "plate 1 COPX analog", "plate 1 COPY analog",
    "plate 2 COPX analog", "plate 2 COPY analog", "left_cop_x(mm)",
    "t_cop_y(mm)", "right_cop_x(mm)", "right_cop_y(mm)"
    ]

# S1_ST_WALK_1  畫圖tang
columns_name = [
    "motion frame number", "time", "time2", "pecentage", "plate 1右 COPX analog",
    "plate 2 左COPX analog", "plate 2 COPY analog", "time2", "left_copx",
    "left_cop_y(mm)", "right_cop_x(mm)", "right_cop_y(mm)"
                ]

"""
S1_ST_WALK_1  畫圖tang

Si長軸right copy = -(M2-M$2)+E$2
    M:"right_cop_y(mm)"
    E: "plate 1右 COPX analog"
    cor 1/2 =PEARSON(E2:E110,R2:R110)
    後1/2=PEARSON(E112:E200,R112:R200)
    full=PEARSON(E2:E199,R2:R199)

motion 在SI = -(E2-E$2)+M$2
    M:"right_cop_y(mm)"
    E: "plate 1右 COPX analog"
    =PEARSON(U2:U110,M2:M110)

x =-(K2-K$2)+G$2
    K: "left_cop_y(mm)"
    G: "plate 2 左COPX analog"
    
    cor 1/2 =PEARSON(G166:G276,W166:W276)
    後1/2=PEARSON=PEARSON(G276:G359,W276:W359)
    full=PEARSON(G168:G359,W168:W359)

SIright copx =-(L2-L$2)+F$2
    L: "right_cop_x(mm)"
    F: "plate 1 COPY analog"
    cor 1/2 =PEARSON(F2:F110,Z2:Z110)
    後1/2=PEARSON =PEARSON(F112:F200,Z112:Z200)
    full =PEARSON(F2:F199,Z2:Z199)

SIleft copx =-(J2-J$2)+H$2
    J: "left_copx"
    H: "plate 2 COPY analog"
    
    cor 1/2 =PEARSON(H170:H276,AB170:AB276)
    後1/2=PEARSON =PEARSON(H276:H359,AB276:AB359)
    full =PEARSON(H170:H359,AB170:AB359)

"""


raw_data = pd.read_excel(r"D:\Hsin\NTSU_lab\WALK\S1_ST_WALK_1  畫圖tang.xlsx")

























