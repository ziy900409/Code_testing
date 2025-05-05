# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:34:21 2025

@author: Hsin.YH.Yang
"""
import ezc3d
import math
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import butter, filtfilt
from typing import Dict, Any, Optional, Tuple, List



# %%
def mean_std_cloud(data1, data2, filename):
    '''

    Parameters
    ----------
    data_path : str
        給定motion data的資料夾路徑.
    savepath : str
        存檔路徑.
    filename : str
        受試者資料夾名稱，ex: S1.
        
    Returns
    -------
    None.

    '''
    # data_path = motion_folder_path
    # filename = processing_folder_list[i]
    # release = [release[0]/down_freq, release[1]/down_freq]
    # savepath = save_path
    
    # file_list = Read_File(data_path, ".xlsx", subfolder=False)
    
    type1, type2 = [], []
    # 2. 找尋特定檔案名稱
    # for ii in range(len(file_list)):
    #     if before_fatigue in file_list[ii]:
    #         type1.insert(0, file_list[ii])
    #     if after_fatigue in file_list[ii]:
    #         type2.insert(0, file_list[ii])
    # 說明兩組資料各幾筆
    print("before fatigue: ", len(type1))
    print("after_fatigue: ", len(type2))
    # read example data
    example_data = pd.read_excel(type1[0])

    # create multi-dimension matrix
    type1_dict = np.zeros(((np.shape(example_data)[1] - 1), # muscle name without time
                           (np.shape(example_data)[0]), # time length
                           len(type1)))                 # subject number

    for ii in range(len(type1)):
        # read data
        type1_data = pd.read_excel(type1[ii])
        for iii in range(np.shape(example_data)[1] - 1): # exclude time
            type1_dict[iii, :, ii] = type1_data[example_data.columns[iii + 1]]
    
    type2_dict = np.zeros(((np.shape(example_data)[1] - 1), # muscle name without time
                           (np.shape(example_data)[0]), # time length
                           len(type2)))                 # subject number

    for ii in range(len(type2)):
        type2_data = pd.read_excel(type2[ii])
        for iii in range(np.shape(example_data)[1] - 1): # exclude time
            type2_dict[iii, :, ii] = type2_data[example_data.columns[iii + 1]]
    
    # 設定圖片大小
    # 畫第一條線
    # save = savepath + "\\mean_std_" + filename + ".jpg"
    n = int(math.ceil((np.shape(type2_dict)[0]) /2))
    # 設置圖片大小
    plt.figure(figsize=(2*n+1,10))
    # 設定繪圖格式與字體
    # plt.style.use('seaborn-white')
    # 顯示輸入中文
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    palette = plt.get_cmap('Set1')
    fig, axs = plt.subplots(n, 2, figsize = (10,12), sharex='col')
    for i in range(np.shape(type2_dict)[0]):
        # 確定繪圖順序與位置
        x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
        color = palette(0) # 設定顏色
        # 都改成100個點
        iters = list(np.linspace(0, 100))
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
        # axs[x, y].set_xlim(-(release[0]), release[1])
        axs[x, y].axvline(x=0, color = 'darkslategray', linewidth=1, linestyle = '--')
        
    plt.suptitle(str("mean std cloud: " + filename), fontsize=16)
    plt.tight_layout()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("time (second)", fontsize = 14)
    plt.ylabel("muscle activation (%)", fontsize = 14)
    # plt.savefig(save, dpi=200, bbox_inches = "tight")
    plt.show()
    
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
def removeoutliers(datain, show=False):
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
    tau = [1.150, 1.393, 1.572, 1.656, 1.711, 1.749, 1.777, 1.798, 1.815,
          1.829, 1.840, 1.849, 1.858, 1.865, 1.871, 1.876, 1.881, 1.885,
          1.889, 1.893, 1.896, 1.899, 1.902, 1.904, 1.906, 1.908, 1.910,
          1.911, 1.913, 1.914, 1.916, 1.917, 1.919, 1.920, 1.921, 1.922,
          1.923, 1.924]
    n = len(datain) #Determine the number of samples in datain
    dataout = datain.copy() # 確保不修改原始 datain

    if n < 3:
        print('ERROR: There must be at least 3 samples in the'
             ' data set in order to use the removeoutliers function.')
    else:
        if show: 
            num_cols = datain.shape[1]
            fig, axes = plt.subplots(1, num_cols * 2, figsize=(15 * num_cols, 5))
            if num_cols == 1:
                axes = [axes[0], axes[1]] # 處理單一欄位的情況

        for i, column in enumerate(datain.columns):
            col_data_in = datain[column].dropna()
            n_col = len(col_data_in)
            tau_val = tau[n_col - 1] if n_col <= len(tau) else 1.960
            S = np.std(col_data_in)
            xbar = np.mean(col_data_in)
            TS = tau_val * S
            if show:
                sns.histplot(col_data_in, ax=axes[i*2], kde=True)
                axes[i*2].set_title(f'Original Distribution - {column}')

            temp_dataout = dataout[column].copy()
            original_indices = temp_dataout.index

            while True:
                valid_data = temp_dataout.dropna()
                if len(valid_data) < 3:
                    break
                S_new = np.std(valid_data)
                xbar_new = np.mean(valid_data)
                n_new = len(valid_data)
                tau_new = tau[n_new - 1] if n_new <= len(tau) else 1.960
                TS_new = tau_new * S_new

                abs_dev = abs(valid_data - xbar_new)
                max_abs_dev_index = abs_dev.idxmax()
                max_abs_dev = abs_dev.max()

                if max_abs_dev > TS_new:
                    temp_dataout.loc[max_abs_dev_index] = np.nan
                else:
                    break
            dataout[column] = temp_dataout

            if show:
                sns.histplot(dataout[column].dropna(), ax=axes[i*2 + 1], kde=True)
                axes[i*2 + 1].set_title(f'Outliers Removed (Thompson Tau) - {column}')

        if show:
            plt.tight_layout()
            plt.show()

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

def zscore_removeoutlier(datain, threshold=3, show=False):
    dataout = datain.copy() # 確保不修改原始 datain

    if show:
        num_cols = datain.shape[1]
        fig, axes = plt.subplots(1, num_cols * 2, figsize=(15 * num_cols, 5))
        if num_cols == 1:
            axes = [axes[0], axes[1]] # 處理單一欄位的情況

    for i, column in enumerate(datain.columns):
        col_data = datain[column].dropna()
        data_mean = np.mean(col_data)
        data_std = np.std(col_data)
        z_scores = abs(stats.zscore(col_data))

        if show:
            sns.histplot(col_data, ax=axes[i*2], kde=True)
            axes[i*2].set_title(f'Original Distribution - {column}')

        outlier_indices = col_data.index[z_scores > threshold]
        dataout.loc[outlier_indices, column] = np.nan

        if show:
            sns.histplot(dataout[column].dropna(), ax=axes[i*2 + 1], kde=True)
            axes[i*2 + 1].set_title(f'Outliers Removed (Z-score > {threshold}) - {column}')

    if show:
        plt.tight_layout()
        plt.show()

    return dataout
