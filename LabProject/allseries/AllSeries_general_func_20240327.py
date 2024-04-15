# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:54:01 2024

@author: Hsin.YH.Yang
"""
# %%
import ezc3d
import pandas as pd
import numpy as np
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

# %% read c3d
def read_c3d(path):
    """
    Parameters
    ----------
    path : str
        key in c3d data path.
    Returns
    -------
    motion_info : dict
        Contains: frame rate, first frame, last frame, size(number of infrared markers).
    motion_data : DataFrame
        data strcuture like .trc file.
    analog_info : dict
        Contains: frame rate, first frame, last frame, size(number of analog channel).
    analog_data : DataFrame
        data structure like .anc file.
    np_motion_data : np.array
        numpy.array with N marker x M frmae x 3.
    -------
    
    example:
        motion_info, motion_data, analog_info, analog_data, np_motion_data = read_c3d(Your_Path)
    
    Author: Hsin Yang. 2023.01.20
    """
    # 1. read c3d file
    # path = r"E:\Motion Analysis\U3 Research\S01\S01_1VS1_1.c3d"
    c = ezc3d.c3d(path)
    
    # 數據的基本資訊，使用dict儲存
    # 1.1 information of motion data
    motion_info = c['header']['points']
    # add Unit in motion information
    motion_info.update({"UNITS": c['parameters']["POINT"]["UNITS"]["value"],
                               "LABELS": c['parameters']["POINT"]["LABELS"]["value"]})
    # 1.2 information of analog data
    analog_info = c['header']['analogs']
    # 2. convert c3d motion data to DataFrame format
    ## 2.1 create column's name of motion data
    motion_axis = ['x', 'y', 'z']
    motion_markers = []
    for marker_name in c['parameters']['POINT']['LABELS']['value']:
        for axis in motion_axis:
            name = marker_name + '_' + axis
            motion_markers.append(name)
    # 2.2 create x, y, z matrix to store motion data
    motion_data = pd.DataFrame(np.zeros([c['header']['points']['last_frame']+1, # last frame + 1
                                          len(c['parameters']['POINT']['LABELS']['value'])*3]), # marker * 3
                                columns=motion_markers) 
    # 使用numpy.array來貯存資料
    np_motion_data = np.empty(shape=(len(c['parameters']['POINT']['LABELS']['value']),
                                     np.shape(c['data']['points'])[-1], 3),
                              dtype=float)

    for i in range(len(c['parameters']['POINT']['LABELS']['value'])):
        np_motion_data[i, :, :] = np.transpose(c['data']['points'][:3, i, :])
    # 2.3 key in data into matrix
    for i in range(len(c['parameters']['POINT']['LABELS']['value'])):
        # print(1*i*3, 1*i*3+3)
        # transpose matrix to key in data
        motion_data.iloc[:, 1*i*3:1*i*3+3] = np.transpose(c['data']['points'][:3, i, :])
    # 2.4 insert time frame
    ## 2.4.1 create time frame
    motion_time = np.linspace(
                                0, # start
                              ((c['header']['points']['last_frame'])/c['header']['points']['frame_rate']), # stop = last_frame/frame_rate
                              num = (np.shape(c['data']['points'])[-1]) # num = last_frame
                              )
    # ### 2.4.2 insert time frame to motion data
    motion_data.insert(0, 'Frame', motion_time)
    # 3. convert c3d analog data to DataFrame format
    #    force plate data (analog = force plate)
    ## 3.1 create force plate channel name
    analog_channel = c['parameters']['ANALOG']['LABELS']['value']
    ## 3.2 create a matrix to store force plate data
    analog_data = pd.DataFrame(np.zeros([np.shape(c['data']['analogs'])[-1], # last frame + 1
                                         len(analog_channel)]), 
                               columns=analog_channel)
    analog_data.iloc[:, :] = np.transpose(c['data']['analogs'][0, :, :])
    ## 3.3 insert time frame
    ### 3.3.1 create time frame
    analog_time = np.linspace(
                                0, # start
                              ((c['header']['analogs']['last_frame'])/c['header']['analogs']['frame_rate']), # stop = last_frame/frame_rate
                              num = (np.shape(c['data']['analogs'])[-1]) # num = last_frame
                              )
    analog_data.insert(0, 'Frame', analog_time)
    # synchronize data (optional)
    return motion_info, motion_data, analog_info, analog_data, np_motion_data