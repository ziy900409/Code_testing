# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:18:26 2024


• E1: 當L.Wrist.Rad Z軸高度超過T10 Z軸高度 擷取此段資料。
• E5: 擷取直到弓身低於T10 Z軸高度，停止擷取。
• 資料分期點:
• E2 舉弓頂點時間:根據全段資料，以L.Wrist.Rad Z軸判定，回傳位置峰值
數值與對應時間點，即時運算角度後取角度峰直數值與對應時間點
• E3 當L.Wrist.Rad Z軸高度等於L. Acromion 進行標記
• E4 放箭時間:根據資料末端2000點判定，即時運算移動平均, R. Elbow Lat
X軸超出前1秒數據3個標準差，判定為放箭
    
@author: Hsin.Yang 05.May.2024
"""
# %%
import os
import numpy as np

# %%

def Read_File(file_path, file_type, subfolder=False):
    '''
    Parameters
    ----------
    x : str
        給予欲讀取資料之路徑.
    y : str
        給定欲讀取資料之副檔名.
    subfolder : boolean, optional
        是否子資料夾一起讀取. The default is 'False'.

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
    # 排除可能會擷取到暫存檔的問題，例如：~$test1_C06_SH1_Rep_2.2_iMVC_ed.xlsx                
    csv_file_list = [file for file in csv_file_list if not "~$" in file]
    return csv_file_list

# %%

def euclidean_distance(point1, point2):
    """
    計算兩個三維點之間的歐幾里得距離
    
    參數：
    point1, point2: 列表或元組，包含三個元素表示三維座標，例如 (x, y, z)
    
    返回值：
    兩點之間的歐幾里得距離
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point1 - point2)
    return distance

# %% 查找满足条件的索引值
def find_index(data, threshold, window_size):
    for i in range(len(data) - window_size + 1):
        if np.all(data[i:i + window_size] < threshold):
            return i
    return None


    