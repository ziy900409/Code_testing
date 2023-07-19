# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 21:21:10 2023

@author: Hsin.YH.Yang
"""

# %% import package
import pandas as pd
import numpy as np
import time
import os
import math
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["'Microsoft Sans Serif"]
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False   # 步驟二（解決坐標軸負數的負號顯示問題）

# %% 所使用Function
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
# %% 小波轉換
import numpy as np
import matplotlib.pyplot as plt
import pywt
import mne
 
# 需要分析的四个频段
iter_freqs = [
    {'name': 'Delta', 'fmin': 0, 'fmax': 4},
    {'name': 'Theta', 'fmin': 4, 'fmax': 8},
    {'name': 'Alpha', 'fmin': 8, 'fmax': 13},
    {'name': 'Beta', 'fmin': 13, 'fmax': 35},
]
 
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
mne.set_log_level(False)
########################################小波包变换-重构造分析不同频段的特征(注意maxlevel，如果太小可能会导致)#########################
def TimeFrequencyWP(data, fs, wavelet, maxlevel = 8):
    # 小波包变换这里的采样频率为250，如果maxlevel太小部分波段分析不到
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # 计算maxlevel最小频段的带宽
    freqBand = fs/(2**maxlevel)
    #######################根据实际情况计算频谱对应关系，这里要注意系数的顺序
    # 绘图显示
    fig, axes = plt.subplots(len(iter_freqs)+1, 1, figsize=(10, 7), sharex=True, sharey=False)
    # 绘制原始数据
    axes[0].plot(data)
    axes[0].set_title('原始数据')
    for iter in range(len(iter_freqs)):
        # 构造空的小波包
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if (iter_freqs[iter]['fmin']<=bandMin and iter_freqs[iter]['fmax']>= bandMax):
                # 给新构造的小波包参数赋值
                new_wp[freqTree[i]] = wp[freqTree[i]].data
        # 绘制对应频率的数据
        axes[iter+1].plot(new_wp.reconstruct(update=True))
        # 设置图名
        axes[iter+1].set_title(iter_freqs[iter]['name'])
    plt.show()
 
 
if __name__ == '__main__':
    # 读取筛选好的epoch数据
    epochsCom = mne.read_epochs(r'F:\BaiduNetdiskDownload\BCICompetition\BCICIV_2a_gdf\Train\Fif\A02T_epo.fif')
    dataCom = epochsCom[10].get_data()[0][0][0:1024]
    TimeFrequencyWP(dataCom,250,wavelet='db4', maxlevel=8)
# %% read data
forder_path = r'E:\python\EMG_Data\COP\raw_data\\'
results_path = r'E:\python\EMG_Data\COP\results\\'
# 去除有“.“開頭的檔案
rowdata_folder_list  = [f for f in os.listdir(forder_path) if not f.startswith('.')]

# %% Asymmetry 計算
'''
Asymmetry 計算說明

使用第一個點做中線基準，並計算落在中線以左或是以佑的落點數.

1. mean : 平均距離，所有點與中線距離的平均值.
2. SD (標準差) : 標準差，所有點與中線距離的標準差 (左/右).
3. rms (方均跟) : 標準差，所有點與中線距離的方均根 (左/右). 
4. |Max| : 在所有落點中找出最偏左或右的最大值.
5. 落點數 : 有幾個COP點落在中線以左/右.
6. 所佔時間比 : COP落在中線以左/右所占的時間除以總評估時間.
7. 偏左/右距離總合 : 所有落在中線以左的COP點，偏離中線距離的總合

'''
data_table = pd.DataFrame({'資料夾' : [],
                           '檔名' : [],
                           '偏左-平均距離': [],
                           '偏左-標準差': [],
                           '偏左-方均根': [],
                           '偏左-最大值': [],
                           '偏左-落點數': [],
                           '偏左-距離總和':[],
                           '偏右-平均距離': [],
                           '偏右-標準差': [],
                           '偏右-方均根': [],
                           '偏右-最大值': [],
                           '偏右-落點數': [],
                           '偏右-距離總和': []
                                       })
# 計算基礎數值用
for num in range(len(rowdata_folder_list)):
    file_list = Read_File(str(forder_path + rowdata_folder_list[num]), '.txt', subfolder=False)
    print(rowdata_folder_list[num])
    for ii in range(len(file_list)):
        force_data = pd.read_csv(file_list[ii])
        force_data = force_data.iloc[:-1, :]
        if "Cop X(mm)" in force_data.columns:
            x, y = 'Cop X(mm)', 'Cop Y(mm)'
        elif "Cop X" in force_data.columns:
            x, y = "Cop X", "Cop Y"
        # 找出基準點
        start_point = np.mean(force_data.loc[:, [x, y]], axis=0)
        left_ind = []
        right_ind = []
        for i in range(np.shape(force_data)[0]):
            if force_data.loc[i, x] >= start_point[0]:
                right_ind.append(i)
            else:
                left_ind.append(i)
        
        # 1. 計算平均距離與距離總和
        ## 1.1. 左側
        left_sum = 0
        for i in range(len(left_ind)):
            left_sum = left_sum + force_data.loc[left_ind[i], x] - start_point[0]
        left_mean = left_sum / (len(left_ind) - 1)
        ## 1.2. 右側
        right_sum = 0
        for i in range(len(right_ind)):
            right_sum = right_sum + force_data.loc[right_ind[i], x] - start_point[0]
        right_mean = right_sum / (len(right_ind) - 1)
        
        # 2. 計算標準差
        ## 2.1. 左側
        left_std = np.std(force_data.loc[left_ind, x], ddof=0)
        ## 2.2. 右側
        right_std = np.std(force_data.loc[right_ind, x], ddof=0)
        
        # 3. 計算方均根
        ## 3.1. 左側
        left_rms = 0
        for i in range(len(left_ind)):
            left_rms = left_rms + (force_data.loc[left_ind[i], x] - start_point[0])**2
        left_rms = np.sqrt(left_rms/(len(left_ind)-1))
        ## 3.2. 右側
        right_rms = 0
        for i in range(len(right_ind)):
            right_rms = right_rms + (force_data.loc[right_ind[i], x] - start_point[0])**2
        right_rms = np.sqrt(right_rms/(len(right_ind)-1))
        
        # 4. 找最大值
        ## 4.1. 左側
        left_max = max(force_data.loc[left_ind, x]) - start_point[0]
        ## 4.2. 右側
        right_max = max(force_data.loc[right_ind, x]) - start_point[0]
        
        # 5. 落點數
        ## 5.1. 左側
        left_len = len(left_ind)
        ## 5.2. 右側
        right_len = len(right_ind)
        
        # 6. 設定檔名與輸出資料
        filepath, tempfilename = os.path.split(file_list[ii])
        filename, extension = os.path.splitext(tempfilename)
        data_table = pd.concat([data_table, pd.DataFrame({'資料夾' : rowdata_folder_list[num],
                                                          '檔名' : filename,
                                                          '偏左-平均距離': left_mean,
                                                          '偏左-標準差': left_std,
                                                          '偏左-方均根': left_rms,
                                                          '偏左-最大值': left_max,
                                                          '偏左-落點數': left_len,
                                                          '偏左-距離總和':left_sum,
                                                          '偏右-平均距離': right_mean,
                                                          '偏右-標準差': right_std,
                                                          '偏右-方均根': right_rms,
                                                          '偏右-最大值': right_max,
                                                          '偏右-落點數': right_len,
                                                          '偏右-距離總和': right_sum
                                                          }, index=[0])],
                               ignore_index=True)
        # data_table.append({'資料夾' : rowdata_folder_list[num],
        #                                 '檔名' : filename,
        #                                 '偏左-平均距離': left_mean,
        #                                 '偏左-標準差': left_std,
        #                                 '偏左-方均根': left_rms,
        #                                 '偏左-最大值': left_max,
        #                                 '偏左-落點數': left_len,
        #                                 '偏左-距離總和':left_sum,
        #                                 '偏右-平均距離': right_mean,
        #                                 '偏右-標準差': right_std,
        #                                 '偏右-方均根': right_rms,
        #                                 '偏右-最大值': right_max,
        #                                 '偏右-落點數': right_len,
        #                                 '偏右-距離總和': right_sum
        #                                 }, ignore_index=True)
# 輸出資料
pd.DataFrame(data_table).to_excel(str(results_path + 'all_table.xlsx'),
                                  sheet_name='Sheet1', index=False, header=True)
        
# %% 計算極座標
'''
計算說明 :
    方法 : 前一個點減去後一個點
                 ↗ P1
               ∕ 
          P0 ∕------→ (1, 0)
    1. 計算偏移位移量的方向 : 計算P0 -> P1 與 P -> (0, 1) 之夾角
    2. 計算偏移位移量的向量 : 計算P0 -> P1 之向量大小
'''

tic = time.process_time()
# 計算各子資料夾下的資料
for num in range(len(rowdata_folder_list)):
    file_list = Read_File(str(forder_path + rowdata_folder_list[num]), '.txt', subfolder=False)
    print(rowdata_folder_list[num])
    # 每一新的資料夾就重新歸零
    x = [math.radians(i) for i in range(0,360,45)]  # 轉換成弧度後變成串列，每 45 度為一單位
    y = [0, 0, 0, 0, 0, 0, 0, 0]
    # 創建一個pd.DataFrame以儲存向量

    vector = {'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[]}
    for ii in range(len(file_list)):
        force_data = pd.read_csv(file_list[ii])
        force_data = force_data.loc[:, ['Cop X(mm)', 'Cop Y(mm)']].dropna(axis=0)
        # 將數值平滑處理，姻緣使數值看起來已有做過lowpass filter，因此只做粗粒化處理
        # 粗粒化 : 五筆做一次平均
        # smooting_data = pd.DataFrame(np.zeros([int(np.ceil(np.shape(force_data)[0]/5)), np.shape(force_data)[1]]),
        #                              columns = force_data.columns)
        # for iii in range(np.shape(smooting_data)[0]):
        #     if (iii+1)*10 < np.shape(force_data)[0]:
        #         smooting_data.iloc[iii, :] = np.mean(force_data.iloc[iii*5:5+iii*5], axis=0)
        #     else:
        #         smooting_data.iloc[iii, :] = np.mean(force_data.iloc[iii*5:], axis=0)
        
        for i in range(np.shape(force_data)[0] - 1):
            # 計算向量夾角
            vector_1 = force_data.loc[i+1, ['Cop X(mm)', 'Cop Y(mm)']].values - force_data.loc[i, ['Cop X(mm)', 'Cop Y(mm)']].values
            vector_2 = force_data.loc[i, ['Cop X(mm)', 'Cop Y(mm)']].values + np.array([1, 0])
            cos_b = np.dot(vector_1, vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
            sin_b = np.cross(vector_1, vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
            # 判斷 COS(theta) 的所屬象限
            if sin_b < 0:
                B = math.degrees(-(math.acos(cos_b))) % 360
            else:
                B = math.degrees(math.acos(cos_b))
            ## 1. 繪製極座標用，並分為八等分
            # 1.1. 計算該方向之位移總和
            # 1.2. 計算該方向之向量大小的次數並計算平均與標準差
            np.linalg.norm(vector_1)
            # print(i, B, np.linalg.norm(vector_1))
            if math.degrees(np.pi/8) > B >= 0 or math.degrees(2*np.pi) > B >= math.degrees(2*np.pi - np.pi/8):
                y[0] += 1*np.linalg.norm(vector_1)
                vector['0'].append(np.linalg.norm(vector_1))
            elif math.degrees(3*np.pi/8) > B >= math.degrees(np.pi/8):
                y[1] += 1*np.linalg.norm(vector_1)
                vector['1'].append(np.linalg.norm(vector_1))
            elif math.degrees(5*np.pi/8) > B >= math.degrees(3*np.pi/8):
                y[2] += 1*np.linalg.norm(vector_1)
                vector['2'].append(np.linalg.norm(vector_1))
            elif math.degrees(7*np.pi/8) > B >= math.degrees(5*np.pi/8):
                y[3] += 1*np.linalg.norm(vector_1)
                vector['3'].append(np.linalg.norm(vector_1))
            elif math.degrees(9*np.pi/8) > B >= math.degrees(7*np.pi/8):
                y[4] += 1*np.linalg.norm(vector_1)
                vector['4'].append(np.linalg.norm(vector_1))
            elif math.degrees(11*np.pi/8) > B >= math.degrees(9*np.pi/8):
                y[5] += 1*np.linalg.norm(vector_1)
                vector['5'].append(np.linalg.norm(vector_1))
            elif math.degrees(13*np.pi/8) > B >= math.degrees(11*np.pi/8):
                y[6] += 1*np.linalg.norm(vector_1)
                vector['6'].append(np.linalg.norm(vector_1))
            elif math.degrees(15*np.pi/8) > B >= math.degrees(13*np.pi/8):
                y[7] += 1*np.linalg.norm(vector_1)
                vector['7'].append(np.linalg.norm(vector_1))
            else:
                print(vector_1)

    # 將第一個值放到最後，以封閉雷達圖
    x.append(x[0])
    y.append(y[0])
    # 設定繪圖格式與字體
    plt.style.use('seaborn-white')
    palette = plt.get_cmap('Set1')
    # 設定圖形顯示中文
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    ## 1. -------繪製方位角的雷達圖------------------------
    fig, ax1 = plt.subplots(figsize=(10,5), 
                            subplot_kw=dict(projection="polar"))
    ax1.plot(x, y)
    trans, _ , _ = ax1.get_xaxis_text1_transform(-10)
    ax1.text(np.deg2rad(0), -0.3, "左右方向", transform=trans, 
             rotation=0, ha="center", va="center")
    ax1.text(np.deg2rad(270), -0.22, "前後方向", transform=trans, 
             rotation=0, ha="center", va="center")
    plt.fill(x,y,alpha=0.2)
    plt.title(str(rowdata_folder_list[num] + ' : 擊發前身體晃動位移總和 (mm)'))
    plt.tight_layout()
    plt.savefig(str(results_path + rowdata_folder_list[num]+"_擊發前身體晃動位移總和 (mm).jpg"),
                dpi=200, bbox_inches = "tight")
    plt.show()
    ## 2. -------繪製向量的雷達圖-------------------------
    # 設定圖片大小
    # 畫第一條線
    # save = savepath + "\\mean_std_" + filename + ".jpg"
    # 設置圖片大小
    plt.figure(figsize=(10,5))
    color = palette(0) # 設定顏色
    # 設定計算資料
    avg1 = [np.mean(j) for i, j in vector.items()]
    std1 = [np.std(j) for i, j in vector.items()]
    r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
    # 將第一個值放到最後，以封閉雷達圖
    avg1.append(avg1[0])
    std1.append(std1[0])
    r1.append(r1[0])
    r2.append(r2[0])
    plt.polar(x, avg1, color=color)
    plt.fill_between(x, r1, r2, color=color, alpha=0.2)
    plt.title(str(rowdata_folder_list[num] + ' : 擊發前身體晃動位移平均與標準差 (mm)'))
    plt.tight_layout()
    plt.xlabel("前後方向", fontsize = 14)
    plt.ylabel("左右方向", fontsize = 14, labelpad=25)
    plt.savefig(str(results_path + rowdata_folder_list[num]+"_擊發前身體晃動位移平均與標準差 (mm).jpg"),
                dpi=200, bbox_inches = "tight")
    plt.show()
toc = time.process_time()
print("Release Time Total Time Spent: ",toc-tic)
# %% 
# %% 計算極座標
'''
計算說明 :
    方法 : 前一個點減去後一個點
                 ↗ P1
               ∕ 
          P0 ∕------→ (1, 0)
    1. 計算偏移位移量的方向 : 計算P0 -> P1 與 P -> (0, 1) 之夾角
    2. 計算偏移位移量的向量 : 計算P0 -> P1 之向量大小
'''

tic = time.process_time()
# 計算各子資料夾下的資料
for num in range(len(rowdata_folder_list)):
    file_list = Read_File(str(forder_path + rowdata_folder_list[num]), '.txt', subfolder=False)
    print(rowdata_folder_list[num])
    # 每一新的資料夾就重新歸零
    x = [math.radians(i) for i in range(0,360,90)]  # 轉換成弧度後變成串列，每 45 度為一單位
    y = [0, 0, 0, 0]
    # 創建一個pd.DataFrame以儲存向量

    vector = {'0':[], '1':[], '2':[], '3':[]}
    for ii in range(len(file_list)):
        force_data = pd.read_csv(file_list[ii])
        force_data = force_data.loc[:, ['Cop X(mm)', 'Cop Y(mm)']].dropna(axis=0)
        # 將數值平滑處理，姻緣使數值看起來已有做過lowpass filter，因此只做粗粒化處理
        # 粗粒化 : 五筆做一次平均
        # smooting_data = pd.DataFrame(np.zeros([int(np.ceil(np.shape(force_data)[0]/5)), np.shape(force_data)[1]]),
        #                              columns = force_data.columns)
        # for iii in range(np.shape(smooting_data)[0]):
        #     if (iii+1)*10 < np.shape(force_data)[0]:
        #         smooting_data.iloc[iii, :] = np.mean(force_data.iloc[iii*5:5+iii*5], axis=0)
        #     else:
        #         smooting_data.iloc[iii, :] = np.mean(force_data.iloc[iii*5:], axis=0)
        
        for i in range(np.shape(smooting_data)[0] - 1):
            # 計算向量夾角
            vector_1 = force_data.loc[i+1, ['Cop X(mm)', 'Cop Y(mm)']].values - force_data.loc[i, ['Cop X(mm)', 'Cop Y(mm)']].values
            vector_2 = force_data.loc[i, ['Cop X(mm)', 'Cop Y(mm)']].values + np.array([1, 0])
            cos_b = np.dot(vector_1, vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
            sin_b = np.cross(vector_1, vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
            # 判斷 COS(theta) 的所屬象限
            if sin_b < 0:
                B = math.degrees(-(math.acos(cos_b))) % 360
            else:
                B = math.degrees(math.acos(cos_b))
            ## 1. 繪製極座標用，並分為八等分
            # 1.1. 計算該方向之位移總和
            # 1.2. 計算該方向之向量大小的次數並計算平均與標準差
            np.linalg.norm(vector_1)
            # print(i, B, np.linalg.norm(vector_1))
            if math.degrees(np.pi/4) > B >= 0 or math.degrees(2*np.pi) > B >= math.degrees(2*np.pi - np.pi/4):
                y[0] += 1*np.linalg.norm(vector_1)
                vector['0'].append(np.linalg.norm(vector_1))
            elif math.degrees(3*np.pi/4) > B >= math.degrees(np.pi/4):
                y[1] += 1*np.linalg.norm(vector_1)
                vector['1'].append(np.linalg.norm(vector_1))
            elif math.degrees(5*np.pi/4) > B >= math.degrees(3*np.pi/4):
                y[2] += 1*np.linalg.norm(vector_1)
                vector['2'].append(np.linalg.norm(vector_1))
            elif math.degrees(7*np.pi/4) > B >= math.degrees(5*np.pi/4):
                y[3] += 1*np.linalg.norm(vector_1)
                vector['3'].append(np.linalg.norm(vector_1))


    # 將第一個值放到最後，以封閉雷達圖
    x.append(x[0])
    y.append(y[0])
    # 設定繪圖格式與字體
    plt.style.use('seaborn-white')
    palette = plt.get_cmap('Set1')
    # 設定圖形顯示中文
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    ## 1. -------繪製方位角的雷達圖------------------------
    fig, ax1 = plt.subplots(figsize=(10,5), 
                            subplot_kw=dict(projection="polar"))
    ax1.plot(x, y)
    trans, _ , _ = ax1.get_xaxis_text1_transform(-10)
    ax1.text(np.deg2rad(0), -0.3, "左右方向", transform=trans, 
             rotation=0, ha="center", va="center")
    ax1.text(np.deg2rad(270), -0.22, "前後方向", transform=trans, 
             rotation=0, ha="center", va="center")
    plt.fill(x,y,alpha=0.2)
    plt.title(str(rowdata_folder_list[num] + ' : 擊發前身體晃動位移總和 (mm)'))
    plt.tight_layout()
    plt.show()
    ## 2. -------繪製向量的雷達圖-------------------------
    # 設定圖片大小
    # 畫第一條線
    # save = savepath + "\\mean_std_" + filename + ".jpg"
    # 設置圖片大小
    plt.figure(figsize=(10,5))
    color = palette(0) # 設定顏色
    # 設定計算資料
    avg1 = [np.mean(j) for i, j in vector.items()]
    std1 = [np.std(j) for i, j in vector.items()]
    r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
    r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
    # 將第一個值放到最後，以封閉雷達圖
    avg1.append(avg1[0])
    std1.append(std1[0])
    r1.append(r1[0])
    r2.append(r2[0])
    plt.polar(x, avg1, color=color)
    # ax1.plot(iters, avg1, color=color, label='before', linewidth=3)
    plt.fill_between(x, r1, r2, color=color, alpha=0.2)
    plt.title(str(rowdata_folder_list[num] + ' : 擊發前身體晃動位移平均與標準差 (mm)'))
    plt.tight_layout()
    plt.xlabel("前後方向", fontsize = 14)
    plt.ylabel("左右方向", fontsize = 14, labelpad=25)
    plt.show()
toc = time.process_time()
print("Release Time Total Time Spent: ",toc-tic)







