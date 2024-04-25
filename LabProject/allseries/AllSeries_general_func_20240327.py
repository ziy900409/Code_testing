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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
# %%

def removeoutliers_array(datain):
# REMOVEOUTLIERS   Remove outliers from data using the Thompson Tau method.
#    For vectors, REMOVEOUTLIERS(datain) removes the elements in datain that
#    are considered outliers as defined by the Thompson Tau method. This
#    applies to any data vector greater than three elements in length, with
#    no upper limit (other than that of the machine running the script).
#    Additionally, the output vector is sorted in ascending order.
# 
#    Example: If datain = [1 34 35 35 33 34 37 38 35 35 36 150]
# 
#    then removeoutliers(datain) will return the vector:
#        dataout = 33 34 34 35 35 35 35 36 37 38
# 
#    See also MEDIAN, STD, MIN, MAX, VAR, COV, MODE.
#    This function was written by Vince Petaccio on July 30, 2009.
    tau = [1.150, 1.393, 1.572, 1.656, 1.711, 1.749, 1.777, 1.798, 1.815, \
           1.829, 1.840, 1.849, 1.858, 1.865, 1.871, 1.876, 1.881, 1.885, \
        1.889, 1.893, 1.896, 1.899, 1.902, 1.904, 1.906, 1.908, 1.910, \
        1.911, 1.913, 1.914, 1.916, 1.917, 1.919, 1.920, 1.921, 1.922, \
        1.923, 1.924]
    n = len(datain); #Determine the number of samples in datain
    if n < 3:
        print('ERROR: There must be at least 3 samples in the' \
            ' data set in order to use the removeoutliers function.')
    else:
        S = np.std(datain); #Calculate S, the sample standard deviation
        xbar = np.mean(datain) #Calculate the sample mean
        #tau is a vector containing values for Thompson's Tau

        #Determine the value of S times Tau
        if n > len(tau):
            TS = 1.960*S #For n > 40
        else:
            TS = tau[n]*S #For samples of size 3 < n < 40
        
        #Sort the input data vector so that removing the extreme values
        #becomes an arbitrary task
        dataout = np.sort(datain)
        #Compare the values of extreme high data points to TS
        while abs((max(dataout)-xbar)) > TS:
            dataout = dataout[1:(len(dataout)-1)]
            #Determine the NEW value of S times Tau
            S = np.std(dataout)
            xbar = np.mean(dataout)
            if len(dataout) > len(tau):
                TS = 1.960*S; #For n > 40
            else:
                TS = tau(len(dataout))*S #For samples of size 3 < n < 40
            
        
        # Compare the values of extreme low data points to TS.
        # Begin by determining the NEW value of S times Tau
            S = np.std(dataout)
            xbar = np.mean(dataout)
            if len(dataout) > len(tau):
                TS=1.960*S; # For n > 40
            else:
                TS=tau(len(dataout))*S; #For samples of size 3 < n < 40
            
        while abs((min(dataout)-xbar)) > TS:
            dataout = dataout[2:(len(dataout))]
            #Determine the NEW value of S times Tau
            S = np.std(dataout)
            xbar = np.mean(dataout)
            if len(dataout) > len(tau):
                TS = 1.960*S # For n > 40
            else:
                TS = tau(len(dataout))*S #For samples of size 3 < n < 40
    return dataout
# %%
def removeoutliers(datain):
    '''
    REMOVEOUTLIERS   
        Remove outliers from data using the Thompson Tau method.
        For vectors, REMOVEOUTLIERS(datain) removes the elements in datain that
        are considered outliers as defined by the Thompson Tau method. This
        applies to any data vector greater than three elements in length, with
        no upper limit (other than that of the machine running the script).
        Additionally, the output vector is sorted in ascending order.

        Example: If datain = [1 34 35 35 33 34 37 38 35 35 36 150]

        then removeoutliers(datain) will return the vector:
            dataout = 33 34 34 35 35 35 35 36 37 38

        See also MEDIAN, STD, MIN, MAX, VAR, COV, MODE.
        This function was written by Vince Petaccio on July 30, 2009.
        
        remove data by column in pd.DataFrame type
        modify by Hsin.Yang April 16, 2024

    Parameters
    ----------
    datain : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    tau = [1.150, 1.393, 1.572, 1.656, 1.711, 1.749, 1.777, 1.798, 1.815, \
           1.829, 1.840, 1.849, 1.858, 1.865, 1.871, 1.876, 1.881, 1.885, \
        1.889, 1.893, 1.896, 1.899, 1.902, 1.904, 1.906, 1.908, 1.910, \
        1.911, 1.913, 1.914, 1.916, 1.917, 1.919, 1.920, 1.921, 1.922, \
        1.923, 1.924]
    n = len(datain) #Determine the number of samples in datain
    dataout = datain
    
    if n < 3:
        print('ERROR: There must be at least 3 samples in the' \
            ' data set in order to use the removeoutliers function.')
    else:

        
        for column in range(np.shape(datain)[1]):
            S = np.std(datain.iloc[:, column]) # Calculate S, the sample standard deviation
            xbar = np.mean(datain.iloc[:, column]) # Calculate the sample mean
            #tau is a vector containing values for Thompson's Tau

            #Determine the value of S times Tau
            if n > len(tau):
                TS = 1.960*S #For n > 40
            else:
                TS = tau[n]*S #For samples of size 3 < n < 40
            # Sort the input data vector so that removing the extreme values
            # becomes an arbitrary task
            # dataout = np.sort(datain)
            #Compare the values of extreme high data points to TS
            while abs((dataout.iloc[:, column].max() - xbar)) > TS:
                dataout.iloc[dataout.iloc[:, column].argmax(), column] = np.nan
                # dataout = dataout[1:(len(dataout)-1)]
                #Determine the NEW value of S times Tau
                S = np.std(dataout.iloc[:, column])
                xbar = np.mean(dataout.iloc[:, column])
                if len(dataout.iloc[:, column].dropna()) > len(tau):
                    TS = 1.960*S; #For n > 40
                else:
                    TS = tau[len(dataout.iloc[:, column].dropna())]*S #For samples of size 3 < n < 40
                
            
            # Compare the values of extreme low data points to TS.
            # Begin by determining the NEW value of S times Tau
                S = np.std(dataout.iloc[:, column])
                xbar = np.mean(dataout.iloc[:, column])
                if len(dataout.iloc[:, column].dropna()) > len(tau):
                    TS=1.960*S; # For n > 40
                else:
                    TS=tau[len(dataout.iloc[:, column].dropna())]*S; #For samples of size 3 < n < 40
                
            while abs((dataout.iloc[:, column].min() - xbar)) > TS:
                # dataout = dataout[2:(len(dataout))]
                dataout.iloc[dataout.iloc[:, column].argmin(), column] = np.nan
                #Determine the NEW value of S times Tau
                S = np.std(dataout.iloc[:, column])
                xbar = np.mean(dataout.iloc[:, column])
                if len(dataout) > len(tau):
                    TS = 1.960*S # For n > 40
                else:
                    TS = tau[len(dataout.iloc[:, column].dropna())]*S #For samples of size 3 < n < 40
    return dataout
            
            
 
# %% iqr_removeoutlier
def iqr_removeoutlier(datain, show=False):
    """
    This function uses the interquartile range (IQR) method to identify and remove outliers from
    the input dataframe. Outliers are identified for each column in the dataframe and replaced
    with NaN values.
    
    
    Parameters
    ----------
    datain : pandas.DataFrame
        Input dataframe containing the data with potential outliers.
        
    show : bool
        A flag to determine whether to draw a Box Plot figure. Default is False.

    Returns
    -------
    dataout : pandas.DataFrame
        The data from datain with removed outliers based on the interquartile range (IQR) method.
        Outliers are replaced with NaN values.
        
        
    This function was written by Hsin Yang on April 18, 2024.
    """
    # Using the interquartile range to find outliers
    # datain = subject_data
    dataout = datain
    for column in range(np.shape(datain)[1]):
        # caculate q1
        q1 = np.percentile(datain.iloc[:, column], 25)
        # caculate q3
        q3 = np.percentile(datain.iloc[:, column], 75)
        # cacualte IQR
        iqr = q3 - q1
        # To find the data position which samll than qi - 1.5*iqr
        q1_positions = list(np.where(datain.iloc[:, column] < (q1 - 1.5*iqr))[0])
        q3_positions = list(np.where(datain.iloc[:, column] > (q3 + 1.5*iqr))[0])
        # Outliers are replaced with NaN values
        dataout.iloc[[q1_positions + q3_positions], column] = np.nan
    # draw figure
    if show:
        # Box plot
        plt.figure(figsize=(10, 6))
        datain.boxplot(patch_artist=True, meanline=False, showmeans=False,
                       boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='r', markersize=6))
        plt.title('Box Plot of Multiple Datasets with IQR Highlighted', fontsize=14)
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.xticks(rotation=45)
        
    return dataout

# %% Statistical outlier detection

def zscore_removeoutlier(datain):
    dataout = datain
    for column in range(np.shape(datain)[1]):
        # 计算数据的平均值和标准差
        data_mean = np.mean(datain.iloc[:, column])
        data_std = np.std(datain.iloc[:, column])
        
        # caculate Z-score
        z_scores = (datain.iloc[:, column] - data_mean) / data_std
        # To find the data position which samll than qi - 1.5*iqr
        small_positions = list(np.where(z_scores.iloc[:, column] < 3)[0])
        big_positions = list(np.where(z_scores.iloc[:, column] > 3)[0])

        # Outliers are replaced with NaN values
        dataout.iloc[[small_positions + big_positions], column] = np.nan
    return dataout


# %%
import numpy as np  # 引入NumPy庫用於數學運算

def included_angle(x0, x1, x2):
    # 將參數轉換為NumPy陣列以進行向量運算
    x0 = np.array(x0)
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    # 計算向量A和向量B
    vector_A = x0 - x1  # 向量A是從x1到x0的差
    vector_B = x2 - x1  # 向量B是從x1到x2的差
    
    # 計算向量A和向量B的點積
    dot_product = np.sum(vector_A * vector_B, axis=1)
    
    # 計算向量A和向量B的大小或模長度
    magnitude_A = np.linalg.norm(vector_A, axis=1)
    magnitude_B = np.linalg.norm(vector_B, axis=1)
    
    # 計算夾角的cosine值
    cosines = dot_product / (magnitude_A * magnitude_B)
    
    # 確保cosine值在合法範圍內（-1到1之間）
    cosines = np.clip(cosines, -1, 1)
    
    # 計算夾角的弧度和角度
    angle_radians = np.arccos(cosines)
    angle_degrees = np.degrees(angle_radians)
    
    # 將角度範圍調整為0到360度
    angle_degrees_360 = (angle_degrees + 360) % 360
    
    # 返回調整後的夾角值
    return angle_degrees_360













