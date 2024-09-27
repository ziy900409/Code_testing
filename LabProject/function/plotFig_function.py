import os
import pandas as pd
import numpy as np
from scipy import signal
import ezc3d
import math
import logging #print 警告用
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"sys.path.append(rD:\BenQ_Project\git\Code_testing\LabProject\function")
import gen_function as gen
from scipy.interpolate import interp1d
# 設置繪圖參數 --------------------------------------------------------------
compare_name = {"group1" :["SH1"],
                "group2": ["SHM"]}
muscle_name = ["R EXT: EMG 1", "R TRI : EMG 2", "R FLX: EMG 3",
               "R BI: EMG 4", "R UT: EMG 5", "R LT: EMG 6"]

# 創造資料儲存位置
time_ratio = {"E1-E2": 1,
              "E2-E3-1": 1,
              "E3-1-E3-2": 0.5,
              "E3-2-E4": 3,
              "E4-E5": 0.2}

total_time = 0
for ratio in time_ratio.keys():
    total_time += time_ratio[ratio]

time_length = int(total_time * 10 * 2)

# %%
def compare_mean_std_cloud(data_path, savepath, filename, smoothing,
                           compare_name = compare_name,
                           muscle_name = muscle_name,
                           compare_group = False,
                           subfolder=False):
    
    '''

    比较多个数据集的均值和标准差，并生成相应的图表。

    Parameters
    ----------
    data_path : str
        要比较数据的文件夹路径.
    savepath : str
        保存图表的文件夹路径.
    filename : str
        保存图表的文件名.
    smoothing : bool
        是否对数据进行平滑处理.
    compare_name : list of str, optional
        需要比较的数据集名称列表，默认为 ["SH1", "SHM"].
    muscle_name : list of str, optional
        肌肉名称列表，默认为 ["R EXT: EMG 1", "R TRI : EMG 2", "R FLX: EMG 3",
        "R BI: EMG 4", "R UT: EMG 5", "R LT: EMG 6"].

    Returns

    @author: Hsin.Yang 05.May.2024
    '''

    
    # data_path = r'E:\Hsin\NTSU_lab\Archery\Xiao\202406\202405\EMG\Processing_Data\Method_1\R01\data\motion'
    data_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\202406\202405\EMG\Processing_Data"
    
    # 找出所有資料夾下的 .xlsx 檔案
    if subfolder:
        data_list = gen.Read_File(data_path, ".xlsx", subfolder=True)
    else:
        data_list = gen.Read_File(data_path, ".xlsx", subfolder=False)
    
    compare_data = {key: [] for key in compare_name}
    if compare_group:
        for key in compare_name:
            for i in compare_name[key]:
                for ii in range(len(data_list)):
                    if i in data_list[ii]:
                        compare_data[key].append(data_list[ii])
    else:
        for i in range(len(compare_name)):
            for ii in range(len(data_list)):
                if compare_name[i] in data_list[ii]:
                    compare_data[compare_name[i]].append(data_list[ii])
                
    # 假设 muscle_name 和 total_time 已经定义
    muscle_length = len(muscle_name)

    # 初始化空字典来存储数据数组
    data_arrays = {}
    # 根据 compare_data 的长度，创建相应数量的数据数组
    # muscle name * time length * subject number
    for key in compare_data:
        subject_count = len(compare_data[key])
        data_arrays[key] = np.empty([muscle_length, time_length, subject_count])

    for key in compare_data:
        for idx in range(len(compare_data[key])):
            time_idx = 0
            for period in time_ratio.keys():
                raw_data = pd.read_excel(compare_data[key][idx],
                                         sheet_name=period)
                for muscle in range(len(muscle_name)):
                    time_period = int(time_ratio[period]*10*2)
                    # print(muscle_name[muscle])
                    # print(time_period)
                    # 使用 cubic 將資料內插
                    x = raw_data.iloc[:, 0] # time
                    y = raw_data.loc[:, muscle_name[muscle]]
                    if len(x) < 4:
                        f = interp1d(x, y, kind='linear')
                    else:
                        f = interp1d(x, y, kind='cubic')
                    x_new = np.linspace(raw_data.iloc[0, 0], raw_data.iloc[-1, 0],
                                        time_period)
                    y_new = f(x_new)
                    data_arrays[key][muscle, time_idx:time_idx + time_period, idx] = y_new
                time_idx = time_idx + time_period
    # 設定圖片大小
    # 畫第一條線
    save = savepath + "\\mean_std_" + filename + ".jpg"
    n = int(math.ceil((len(muscle_name)) /2))
    # 設置圖片大小
    # plt.figure(figsize=(n+1,10))
    # 設定繪圖格式與字體
    # plt.style.use('seaborn-white')
    # 顯示輸入中文
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    palette = plt.get_cmap('Set1')
    
    fig, axs = plt.subplots(n, 2, figsize = (10, 2*n+1), sharex='col')
    i = 0 # 更改顏色用
    for key in compare_data:
        for muscle in range(len(muscle_name)):
            # 確定繪圖順序與位置
            x, y = muscle - n*math.floor(abs(muscle)/n), math.floor(abs(muscle)/n) 
            iters = list(np.linspace(0, 114, time_length))
            # 設定計算資料
            avg1 = np.mean(data_arrays[key][muscle, :, :], axis=1) # 計算平均
            std1 = np.std(data_arrays[key][muscle, :, :], axis=1) # 計算標準差
            r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
            r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
            axs[x, y].plot(iters, avg1, color=palette(i), label=key, linewidth=3)
            axs[x, y].fill_between(iters, r1, r2, color=palette(i), alpha=0.2)
            
            # 圖片的格式設定
            axs[x, y].set_title(muscle_name[muscle], fontsize=12)
            axs[x, y].legend(loc="upper left") # 圖例位置
            # axs[x, y].grid(True, linestyle='-.')
            # 畫放箭時間
            axs[x, y].set_xlim(0, time_length)
            axs[x, y].axvline(x=110, color = 'darkslategray', linewidth=1, linestyle = '--')
        i += 1
    # plt.suptitle(str("mean std cloud: " + filename), fontsize=16)
    plt.tight_layout()
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("time (second)", fontsize = 14)
    plt.ylabel("muscle activation (%)", fontsize = 14)
    # plt.savefig(save, dpi=200, bbox_inches = "tight")
    plt.show()
    

        # # 畫花括號
        # curlyBrace(fig, axs[x, y], [shooting_time["stage1"][0], yy], [shooting_time["stage1"][1], yy],
        #            0.05, bool_auto=True, str_text="", color=shooting_time["stage1"][2],
        #            lw=2, int_line_num=1, fontdict=font)
        # curlyBrace(fig, axs[x, y], [shooting_time["stage2"][0], yy], [shooting_time["stage2"][1], yy],
        #            0.05, bool_auto=True, str_text="", color=shooting_time["stage2"][2],
        #            lw=2, int_line_num=1, fontdict=font)
        # curlyBrace(fig, axs[x, y], [shooting_time["stage3"][0], yy], [shooting_time["stage3"][1], yy],
        #            0.05, bool_auto=True, str_text="", color=shooting_time["stage3"][2],
        #            lw=2, int_line_num=1, fontdict=font)
        # curlyBrace(fig, axs[x, y], [shooting_time["stage4"][0], yy], [shooting_time["stage4"][1], yy],
        #            0.05, bool_auto=True, str_text="", color=shooting_time["stage4"][2],
        #            lw=2, int_line_num=1, fontdict=font)
# %%

def compare_mean_std_cloud_onecol(data_path, savepath, filename, smoothing,
                                  compare_name = compare_name,
                                  muscle_name = muscle_name,
                                  compare_group = False,
                                  subfolder=False):
    
    '''

    比较多个数据集的均值和标准差，并生成相应的图表。

    Parameters
    ----------
    data_path : str
        要比较数据的文件夹路径.
    savepath : str
        保存图表的文件夹路径.
    filename : str
        保存图表的文件名.
    smoothing : bool
        是否对数据进行平滑处理.
    compare_name : list of str, optional
        需要比较的数据集名称列表，默认为 ["SH1", "SHM"].
    muscle_name : list of str, optional
        肌肉名称列表，默认为 ["R EXT: EMG 1", "R TRI : EMG 2", "R FLX: EMG 3",
        "R BI: EMG 4", "R UT: EMG 5", "R LT: EMG 6"].

    Returns

    @author: Hsin.Yang 05.May.2024
    '''

    time_accum = {"E1-E2": 10,
                  "E2-E3-1": 20,
                  "E3-1-E3-2": 25,
                  "E3-2-E4": 55,
                  "E4-E5": 57}
    # data_path = r'E:\Hsin\NTSU_lab\Archery\Xiao\202406\202405\EMG\Processing_Data\Method_1\R01\data\motion'
    # data_path = r"E:\Hsin\NTSU_lab\Archery\Xiao\202406\202405\EMG\Processing_Data"
    
    # 找出所有資料夾下的 .xlsx 檔案
    if subfolder:
        data_list = gen.Read_File(data_path, ".xlsx", subfolder=True)
    else:
        data_list = gen.Read_File(data_path, ".xlsx", subfolder=False)
    
    compare_data = {key: [] for key in compare_name}
    if compare_group:
        for key in compare_name:
            for i in compare_name[key]:
                for ii in range(len(data_list)):
                    if i in data_list[ii]:
                        compare_data[key].append(data_list[ii])
    else:
        for i in range(len(compare_name)):
            for ii in range(len(data_list)):
                if compare_name[i] in data_list[ii]:
                    compare_data[compare_name[i]].append(data_list[ii])
                
    # 假设 muscle_name 和 total_time 已经定义
    muscle_length = len(muscle_name)

    # 初始化空字典来存储数据数组
    data_arrays = {}
    # 根据 compare_data 的长度，创建相应数量的数据数组
    # muscle name * time length * subject number
    for key in compare_data:
        subject_count = len(compare_data[key])
        data_arrays[key] = np.empty([muscle_length, time_length, subject_count])

    for key in compare_data:
        for idx in range(len(compare_data[key])):
            time_idx = 0
            for period in time_ratio.keys():
                raw_data = pd.read_excel(compare_data[key][idx],
                                         sheet_name=period)
                for muscle in range(len(muscle_name)):
                    time_period = int(time_ratio[period]*10*2)
                    # print(muscle_name[muscle])
                    # print(time_period)
                    # 使用 cubic 將資料內插
                    x = raw_data.iloc[:, 0] # time
                    y = raw_data.loc[:, muscle_name[muscle]]
                    if len(x) < 4:
                        f = interp1d(x, y, kind='linear')
                    else:
                        f = interp1d(x, y, kind='cubic')
                    x_new = np.linspace(raw_data.iloc[0, 0], raw_data.iloc[-1, 0],
                                        time_period)
                    y_new = f(x_new)
                    data_arrays[key][muscle, time_idx:time_idx + time_period, idx] = y_new
                time_idx = time_idx + time_period
    # 設定圖片大小
    # 畫第一條線
    save = savepath + "\\mean_std_" + filename + ".jpg"
    n = len(muscle_name)
    # 設置圖片大小
    # plt.figure(figsize=(n+1,10))
    # 設定繪圖格式與字體
    # plt.style.use('seaborn-white')
    # 顯示輸入中文
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    palette = plt.get_cmap('Set1')
    
    fig, axs = plt.subplots(n, 1, figsize = (8, 2*n+1), sharex='col')
    i = 0 # 更改顏色用
    for key in compare_data:
        for muscle in range(len(muscle_name)):
            # 確定繪圖順序與位置
            # x, y = muscle - n*math.floor(abs(muscle)/n), math.floor(abs(muscle)/n) 
            iters = list(np.linspace(0, 114, time_length))
            # 設定計算資料
            avg1 = np.mean(data_arrays[key][muscle, :, :], axis=1) # 計算平均
            std1 = np.std(data_arrays[key][muscle, :, :], axis=1) # 計算標準差
            r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
            r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
            axs[muscle].plot(iters, avg1, color=palette(i), label=key, linewidth=3)
            axs[muscle].fill_between(iters, r1, r2, color=palette(i), alpha=0.2)
            
            # 圖片的格式設定
            axs[muscle].set_title(muscle_name[muscle], fontsize=12)
            axs[muscle].legend(loc="upper left") # 圖例位置
            # axs[x, y].grid(True, linestyle='-.')
            # 畫放箭時間
            axs[muscle].set_xlim(0, time_length)
            # axs[muscle].axvline(x=110, color = 'darkslategray', linewidth=1, linestyle = '--')
            
            

            for time_key in time_accum.keys():
                axs[muscle].axvline(time_accum[time_key]*2,
                                    color='darkslategray', linestyle='--', linewidth=1) # trigger onset+
        i += 1 # 更變顏色用
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
