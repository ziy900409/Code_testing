# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:52:35 2024

@author: Hsin.YH.Yang
"""
# %% import package
import numpy as np
import pandas as pd
import os



# %%define funciton
"""
1. 判斷中心圓座標與周圍圓座標的相對位置，切分為四象限，以定義直角三角形
2. 在中心圓周上找出離周圍圓圓心最遠的點，兩者相減必須小於周圍圓的半徑
"""       
def edge_calculate(circle_x, circle_y, edge_cir_x, edge_cir_y, circle_radius):
    """
    计算中心圆与边界圆之间的边界点坐标.

    Parameters
    ----------
    circle_x : float
        中心圆的 x 坐标.
    circle_y : float
        中心圆的 y 坐标.
    edge_cir_x : float
        边界圆的 x 坐标.
    edge_cir_y : float
        边界圆的 y 坐标.
    circle_radius : float
        中心圆的半径.

    Returns
    -------
    edge_point : list
        边界点的坐标 [x, y].

    """
    # 计算距离
    distance = np.sqrt((circle_x - edge_cir_x) ** 2 + (circle_y - edge_cir_y) ** 2)

    sin_angle = np.where(circle_x == edge_cir_x,
                          0,
                          abs(circle_y - edge_cir_y) / np.sqrt((circle_x - edge_cir_x)**2 \
                                                              + (circle_y - edge_cir_y)**2))
    cos_angle = np.where(circle_x == edge_cir_x,
                         0,
                         abs(circle_x - edge_cir_x) / np.sqrt((circle_x - edge_cir_x)**2 \
                                                              + (circle_y - edge_cir_y)**2))
    default_edge_point = [circle_x, circle_y]
    # 确保距离不为零，避免除以零错误
    if distance == 0:
        # 在这里添加处理方式，例如返回一个默认角度值或者设置一个很小的距离值
        return default_edge_point
    # 根据中心圆与边界圆的位置关系计算边界点的位置
    # 第一象限
    if circle_x - edge_cir_x > 0 and circle_y - edge_cir_y > 0:
        edge_point = [(circle_x + circle_radius*cos_angle),
                    (circle_y + circle_radius*sin_angle)]
    # 第二象限
    elif circle_x - edge_cir_x < 0 and circle_y - edge_cir_y > 0:
        edge_point = [(circle_x - circle_radius*cos_angle),
                    (circle_y + circle_radius*sin_angle)]
    # 第三象限
    elif circle_x - edge_cir_x < 0 and circle_y - edge_cir_y < 0:
        edge_point = [(circle_x - circle_radius*cos_angle),
                    (circle_y - circle_radius*sin_angle)]
    # 第四象限
    elif circle_x - edge_cir_x > 0 and circle_y - edge_cir_y < 0:
        edge_point = [(circle_x + circle_radius*cos_angle),
                    (circle_y - circle_radius*sin_angle)]
    # 如果躺在X軸上
    elif circle_y - edge_cir_y == 0:
    # 在周圍圓的右側
        if circle_x - edge_cir_x > 0:
            edge_point = [(circle_x + circle_radius),
                         (circle_y)]
        # 在周圍圓的左側
        elif circle_x - edge_cir_x < 0:
                 edge_point = [(circle_x - circle_radius),
                              (circle_y)]
    # 如果躺在Y軸上
    elif circle_x - edge_cir_x == 0:
        # 在周圍圓的上方
        if circle_y - edge_cir_y > 0:
            edge_point = [(circle_x),
                        (circle_y + circle_radius)]
        # 在周圍圓的下方
        elif circle_y - edge_cir_y < 0:
             edge_point = [(circle_x),
                             (circle_y - circle_radius)]
    return edge_point
# %% 找到所有 True 的索引，並且判斷索引值是否為連續整數
def find_true_indices_and_check_continuity(bool_list):
    # 找到所有True的索引
    true_indices = [i for i, val in enumerate(bool_list) if val]
    
    # 检查这些索引是否是连续的
    if len(true_indices) < 2:
        # 只有一个或者没有True值，视为连续
        return True
    
    # 判断索引值是否为连续的整數
    is_continuous = all(true_indices[i] + 1 == true_indices[i + 1] for i in range(len(true_indices) - 1))
    
    return not is_continuous

# %% 判斷數列中是否同時包含正數與負數
def count_sign_changes(num_list):
    sign_changes = 0
    current_sign = None
    
    for num in num_list:
        if num > 0:
            new_sign = 'positive'
        elif num < 0:
            new_sign = 'negative'
        else:
            continue
        
        if current_sign is not None and current_sign != new_sign:
            sign_changes += 1
        
        current_sign = new_sign
    
    return sign_changes

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

# %% 將Pandas Series轉換為可序列化的列表
def convert_series_to_list(sd3_data_format):
    for key in sd3_data_format['{t_x_y}']:
        for i, series in enumerate(sd3_data_format['{t_x_y}'][key]):
            if isinstance(series, pd.Series):
                sd3_data_format['{t_x_y}'][key][i] = series.tolist()
    return sd3_data_format
                
# %% 將可序列化的列表轉換回Pandas Series
def convert_list_to_series(sd3_data_format):
    for key in sd3_data_format['{t_x_y}']:
        for i, lst in enumerate(sd3_data_format['{t_x_y}'][key]):
            if isinstance(lst, list):
                sd3_data_format['{t_x_y}'][key][i] = pd.Series(lst)
    return sd3_data_format




























