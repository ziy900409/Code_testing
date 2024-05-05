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

# %%

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