# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:19:08 2023

@author: Hsin.YH.Yang
"""
# %% import package
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 顯示輸入中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
from collections import Counter
import csv
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
a = Read_File(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2 EVT DP495按鍵原始數據20230524\轉檔後資料\20230602_083853-L"
              ,'.csv')

# %% 設定檔案路徑
import chardet
final_assemble_R = r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2 EVT DP495按鍵原始數據20230524\轉檔後資料\20230602_083428-R"
final_assemble_L = r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2 EVT DP495按鍵原始數據20230524\轉檔後資料\20230602_083853-L"

final_R_list = Read_File(final_assemble_R, ".csv")
final_L_list = Read_File(final_assemble_L, ".csv")

final_R_table = pd.read_csv(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2 EVT DP495按鍵原始數據20230524\轉檔後資料\result(v2.1)-R.csv")
final_L_table = pd.read_csv(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2 EVT DP495按鍵原始數據20230524\轉檔後資料\result(v2.1)-L.csv",
                            encoding='Big5')

text = open(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2 EVT DP495按鍵原始數據20230524\轉檔後資料\result(v2.1)-L.csv",
            'rb').read()
print(chardet.detect(text))
# %% 目標找出兩側最不穩定的區段
'''
1. 讀總表 : 使用總表的判定時間去繪圖，將圖分為六個區段，並去觀看哪一區段的較為不穩定
'''
# 取出所有前行程資料
fig, axs = plt.subplots(3, 2, figsize = (10,12))
for i in range(20):
    filepath, tempfilename = os.path.split(final_R_list[i])
    filename, extension = os.path.splitext(tempfilename)
    for ii in range(len(final_R_table['DUT'])):
        if filename == final_R_table['DUT'][ii].replace("#", ""):
            data = pd.read_csv(final_R_list[i], header=None).transpose()
            # 轉置資料，並將 NAN fill with 0 point
            data = pd.DataFrame(data.iloc[1:, :].values, columns=data.iloc[0, :])
            data = data.fillna(0)
            ## FWD 繪圖
            # 畫 FWD 第一段
            # 找最接近值的index
            start_index = min(range(len(data.iloc[:, 0])),
                              key=lambda x: abs(data.iloc[x, 0] - final_R_table['D(P1)_按壓開始位置'][ii]))
            end_index = min(range(len(data.iloc[:, 0])),
                            key=lambda x: abs(data.iloc[x, 0] - final_R_table['D(P12)_按壓轉折位置'][ii]))
            # 畫圖
            axs[0, 0].plot(data.iloc[start_index:end_index, 0], data.iloc[start_index:end_index, 1], color='b')
            axs[0, 0].set_title("FWD 按壓開始 -> 轉折", fontsize = 12)
            # 畫 FWD 第二段
            start_index = min(range(len(data.iloc[:, 0])),
                              key=lambda x: abs(data.iloc[x, 0] - final_R_table['D(P12)_按壓轉折位置'][ii]))
            end_index = min(range(len(data.iloc[:, 0])),
                            key=lambda x: abs(data.iloc[x, 0] - final_R_table['D(P2)_按壓觸發位置'][ii]))
            # 畫圖
            axs[1, 0].plot(data.iloc[start_index:end_index, 0], data.iloc[start_index:end_index, 1], color='b')
            axs[1, 0].set_title("FWD 按壓轉折 -> 觸發", fontsize = 12)
            # 畫 FWD 第三段      
            start_index = min(range(len(data.iloc[:, 0])),
                              key=lambda x: abs(data.iloc[x, 0] - final_R_table['D(P3)_按壓後位置'][ii]))
            end_index = min(range(len(data.iloc[:, 0])),
                            key=lambda x: abs(data.iloc[x, 0] - final_R_table['D(P4)_按壓結束位置'][ii]))
            axs[2, 0].plot(data.iloc[start_index:end_index, 0], data.iloc[start_index:end_index, 1], color='b')
            axs[2, 0].set_title("FWD 按壓後位置 -> 結束", fontsize = 12)
            ## BWD 繪圖
            # 畫 BWD 第一段
            start_index = min(range(len(data.iloc[:, 2])),
                              key=lambda x: abs(data.iloc[x, 2] - final_R_table['D(R1)_回彈開始位置'][ii]))
            end_index = min(range(len(data.iloc[:, 2])),
                            key=lambda x: abs(data.iloc[x, 2] - final_R_table['D(R2)_回彈前位置'][ii]))
            axs[0, 1].plot(data.iloc[start_index:end_index, 2], data.iloc[start_index:end_index, 3], color='b')
            axs[0, 1].set_title("BWD 回彈開始 -> 回彈前", fontsize = 12)
            # 畫 BWD 第二段
            start_index = min(range(len(data.iloc[:, 2])),
                              key=lambda x: abs(data.iloc[x, 2] - final_R_table['D(R3)_回彈位置'][ii]))
            end_index = min(range(len(data.iloc[:, 2])),
                            key=lambda x: abs(data.iloc[x, 2] - final_R_table['D(R34)_回彈轉折位置'][ii]))
            axs[1, 1].plot(data.iloc[start_index:end_index, 2], data.iloc[start_index:end_index, 3], color='b')
            axs[1, 1].set_title("BWD 回彈位置 -> 轉折", fontsize = 12)
            # 畫 BWD 第三段
            start_index = min(range(len(data.iloc[:, 2])),
                              key=lambda x: abs(data.iloc[x, 2] - final_R_table['D(R34)_回彈轉折位置'][ii]))
            end_index = min(range(len(data.iloc[:, 2])),
                            key=lambda x: abs(data.iloc[x, 2] - final_R_table['D(R4)_回彈結束位置'][ii]))
            axs[2, 1].plot(data.iloc[start_index:end_index, 2], data.iloc[start_index:end_index, 3], color='b')
            axs[2, 1].set_title("BWD 回彈轉折 -> 結束", fontsize = 12)

            


# %% 讀總表的資料

inputFile = pd.read_excel(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2-checklist-20230526-finish.xlsx",
                          sheet_name='U2-EVT-data')

columns_name_all = inputFile.columns
# %% 3. 擷取 Rating 和 Data 資料
right_key = ["右_", "-右"]
left_key = ["左_", "-左"]
# 設定評分資料的欄位名稱
right_rating_columns = []
left_rating_columns = []
for columns_name in inputFile.iloc[:, 11:47].columns:
    if right_key[0] in columns_name:
        right_rating_columns.append(columns_name)
    elif left_key[0] in columns_name:
        left_rating_columns.append(columns_name)
# 設定輸入資料的欄位名稱
right_input_columns = []
left_input_columns = []
for columns_name in inputFile.iloc[:, 211:].columns:
    if right_key[1] in columns_name:
        right_input_columns.append(columns_name)
    elif left_key[1] in columns_name:
        left_input_columns.append(columns_name)
# 設定資料
## 評分資料
## 重設 index 與丟棄全為 NAN 的列 
right_rating = inputFile.loc[15:136, right_rating_columns].dropna(axis=0, how='all')
left_rating = inputFile.loc[15:136, left_rating_columns].dropna(axis=0, how='all')
## 輸入資料
right_input = inputFile.loc[right_rating.index, right_input_columns].dropna(axis=0, how='all')
left_input = inputFile.loc[15:136, left_input_columns].dropna(axis=0, how='all')
## reset index
right_rating = right_rating.reset_index(drop=True)
left_rating = left_rating.reset_index(drop=True)
right_input = right_input.reset_index(drop=True)
left_input = left_input.reset_index(drop=True)
# %% 4. Rating 調整為兩類，並新增 1 總評等欄位 (有任一 1 即為 1)
#  先將 A B 放置為一組，C 單獨一組
right_rating.insert(0, 'final_score', 0)
# for i in range(len(np.shape(right_rating)[1])):
    

right_rating = right_rating.replace(['A', 'B', 'C', 'D', 'E'], [0, 0, 0, 1, 1])
# 設定如果有任一欄位有C以上的值，就將 final score 設定為 1
right_rating.iloc[:, 0]  = (right_rating.iloc[:, 1:].apply(np.sum, axis=1) != 0).values
right_rating.iloc[:, 0] = right_rating.iloc[:, 0].replace([True, False], [1, 0])
# twoCategory.replace(['C'], 1, inplace=True)  # DE 為 1
# 先捨棄全為0或是Nan的欄位
# right_rating = right_rating.dropna(axis=1)

# 統計各欄位 0 1 數量
# categoryCounts = twoCategory.apply(pd.Series.value_counts).rename_axis('Category')

# %% 繪製資料分布圖
# 繪製資料分布圖
import seaborn as sns 
for ii in right_rating:
    print(ii)
    fig = plt.figure(figsize = (20, 25)) 
    j = 0 
    for i in right_input.columns: 
        plt.subplot(7, 4, j+1) 
        sns.distplot(right_input[i][right_rating[ii]==0], color='g', label = 'A') 
        sns.distplot(right_input[i][right_rating[ii]==1], color='r', label = 'B')
        plt.title(right_input.columns[j])
        plt.legend(loc='best')
        j += 1 
    fig.suptitle(ii, fontsize = 12) 
    fig.tight_layout() 
    fig.subplots_adjust(top=0.95) 
    plt.show()

# %% 計算lab量測資料與factory量測資料的差異
'''
1. 分別讀取工廠量測與實驗室量測之總表
2. 分開按鍵位置與量測機台號碼即可解決
2. 注意單位換算即可
'''
self_test_data = pd.read_excel(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2滑鼠按鍵測試.xlsx")
L_factor_data = pd.read_excel(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2量測資料整理.xlsx",
                              sheet_name='L')
R_factor_data = pd.read_excel(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2量測資料整理.xlsx",
                              sheet_name='R')
import glob
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

mean_self_data = pd.DataFrame(np.zeros([int(np.shape(self_test_data)[0]/3), int(np.shape(self_test_data)[1])])
                              , columns = self_test_data.columns)
# mean_self_data = pd.DataFrame(np.zeros([int(3), int(np.shape(self_test_data)[1])])
#                               , columns = self_test_data.columns)

for i in range(int(np.shape(self_test_data)[0]/3)):
    mean_self_data.iloc[i, 2:] = np.mean(self_test_data.iloc[0+3*i:3+3*i, 2:])
    mean_self_data.iloc[i, 1] = self_test_data.iloc[0+3*i, 1]

different_values = pd.DataFrame(np.zeros([int(np.shape(self_test_data)[0]/3), int(np.shape(self_test_data)[1])])
                              , columns = self_test_data.columns)
for ii in range(int(np.shape(mean_self_data)[0])):
    for iii in range(int(np.shape(L_factor_data)[0])):
        if L_factor_data['DUT'][iii].replace('#', '') == str(int(mean_self_data['subject_number'][ii])) \
            and L_factor_data['button'][iii] == mean_self_data['buttom'][ii]:
                print(L_factor_data['DUT'][iii], L_factor_data['button'][iii])
                print(int(mean_self_data['subject_number'][ii]), mean_self_data['buttom'][ii])
                different_values['ID'][ii] = L_factor_data['DUT'][iii]
                different_values['buttom'][ii] = L_factor_data['button'][iii]
                different_values['EntireTravel'][ii] = mean_self_data['EntireTravel'][ii] - L_factor_data['整機_總行程-左'][iii]
                different_values['PreTravel_Com'][ii] = mean_self_data['PreTravel_Com'][ii] - L_factor_data['整機_按壓前行程-左'][iii]
                different_values['PostTravel_Com'][ii] = mean_self_data['PostTravel_Com'][ii] - L_factor_data['整機_按壓後行程-左'][iii]
                different_values['PostTravel_Re'][ii] = mean_self_data['PostTravel_Re'][ii] - L_factor_data['整機_回彈後行程-左'][iii]
                different_values['PreTravel_Re'][ii] = mean_self_data['PreTravel_Re'][ii] - L_factor_data['整機_回彈前行程-左'][iii]
                different_values['SwitchPeakForce_Com'][ii] = mean_self_data['SwitchPeakForce_Com'][ii]/0.0098 - L_factor_data['整機_F(P2)_按壓力-左'][iii]
                different_values['SwitchPeakForce_Re'][ii] = mean_self_data['SwitchPeakForce_Re'][ii]/0.0098 - L_factor_data['整機_F(R3)_回彈力-左'][iii]
                different_values['Preload_Com'][ii] = mean_self_data['Preload_Com'][ii] - L_factor_data['整機_D(P12)_按壓轉折位置-左'][iii]
                different_values['Preload_Re'][ii] = mean_self_data['Preload_Re'][ii] - L_factor_data['整機_D(R34)_回彈轉折位置-左'][iii]
                

pd.DataFrame(different_values).to_excel(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2_different.xlsx",
                                        sheet_name='Sheet1', index=False, header=True)

# %% 繪圖用
'''
1. 設定存檔路徑
2. load總表：
    2.1 取按鍵規格定義資料
3. 實驗室資料：
    3.1 先讀總表，以抓取備註定義，不然只能知道量測序號
    3.2 將實驗室量測的資料一筆一筆分開
    3.3 設定繪圖開始的位置，threshold設定0.015 (林博設定0.03)
4. 設定條件判斷式
    4.1 使用工廠量測資料之資料夾名稱與檔案名稱，來對實驗室的量測資料
5. .csv讀檔的問題
    5.1 由於工廠量測之資料分為FWD與BWD，因此當資料出現 FWD len < BWD len，會出現讀檔錯誤
    解決方法 : 
        5.1.1 使用 csv package 分行讀
        5.1.2 總共讀檔兩次，第一次先知道所有資料行的長度，並抓取最大長度
        5.1.3 使用最大長度創建矩陣，再重讀資料行數
'''
# 設定存檔路徑
save_path = r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2_量測差異比較\\"
# read data
all_self_data = pd.read_csv(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2-CW 75 pcs 左右鍵規格定義.csv",
                            low_memory=False)
all_self_table = pd.read_csv(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2-CW 75 pcs 左右鍵規格定義資料表格.csv")
# all_self_table = all_self_table.iloc[2:, :].reset_index(drop=True)
all_self_table = all_self_table.dropna(how='all', axis=0)
all_self_table = all_self_table.dropna(how='all', axis=1)
# 讀取工廠量測資料
L_factor_data = pd.read_excel(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\U2\U2量測資料整理.xlsx",
                              sheet_name='L')

# 將實驗室量測的 data 一筆一筆分開
self_data = np.zeros((int(np.floor(np.shape(all_self_data)[1]/7)), np.shape(all_self_data)[0]-2, 2))
for i in range(np.shape(self_data)[0]):
    self_data[i, :, :] = all_self_data.iloc[2:, 0 + i*7:2+ i*7].values



# 計算DUT的數量
# 依左右鍵及編號分開，三支三支一組
result = Counter(all_self_table['DUT']).most_common()


for i in range(len(result)):
    ind_lab_data = all_self_table['項目1'][all_self_table['DUT'] == result[i][0]]
    # print(ind_lab_data)
    # 設定threshold
    ind1 = np.where(self_data[ind_lab_data.index[0], :, 0] > 0.015)[0][[0, -1]]
    ind2 = np.where(self_data[ind_lab_data.index[1], :, 0] > 0.015)[0][[0, -1]]
    ind3 = np.where(self_data[ind_lab_data.index[2], :, 0] > 0.015)[0][[0, -1]]
    # print(ind1, ind2, ind3)
    data1 = pd.DataFrame(self_data[ind_lab_data.index[0], ind1[0]:ind1[1], :]).dropna(axis=1, how='all')
    data2 = pd.DataFrame(self_data[ind_lab_data.index[1], ind2[0]:ind2[1], :]).dropna(axis=1, how='all')
    data3 = pd.DataFrame(self_data[ind_lab_data.index[2], ind3[0]:ind3[1], :]).dropna(axis=1, how='all')
    # 找到最大最小值，重畫X軸
    ## data1
    ind_begin_1 = np.linspace(0, (max(data1.iloc[:, 1]) - data1.iloc[0, 1]), np.argmax(data1.iloc[:, 1]))
    ind_end_1 = np.linspace((max(data1.iloc[:, 1]) - data1.iloc[-1, 1]), 0, (len(data1.iloc[:, 1]) - np.argmax(data1.iloc[:, 1])))
    ind_x_1 = np.concatenate((ind_begin_1, ind_end_1))
    data1.iloc[:, 1] = ind_x_1
    ## data2
    ind_begin_2 = np.linspace(0, (max(data2.iloc[:, 1]) - data2.iloc[0, 1]), np.argmax(data2.iloc[:, 1]))
    ind_end_2 = np.linspace((max(data2.iloc[:, 1]) - data2.iloc[-1, 1]), 0, (len(data2.iloc[:, 1]) - np.argmax(data2.iloc[:, 1])))
    ind_x_2 = np.concatenate((ind_begin_2, ind_end_2))
    data2.iloc[:, 1] = ind_x_2
    ## data3
    ind_begin_3 = np.linspace(0, (max(data3.iloc[:, 1]) - data3.iloc[0, 1]), np.argmax(data3.iloc[:, 1]))
    ind_end_3 = np.linspace((max(data3.iloc[:, 1]) - data3.iloc[-1, 1]), 0, (len(data3.iloc[:, 1]) - np.argmax(data3.iloc[:, 1])))
    ind_x_3 = np.concatenate((ind_begin_3, ind_end_3))
    data3.iloc[:, 1] = ind_x_3
    for ii in range(np.shape(L_factor_data)[0]):
        if all_self_table['試驗註記'][int(ind_lab_data.index[0])].split('-')[0] == L_factor_data['DUT'][ii].replace('#', '') \
            and all_self_table['試驗註記'][int(ind_lab_data.index[0])].split('-')[1] == L_factor_data['button'][ii]:
                print(L_factor_data['path'][ii])
                print(all_self_table['試驗註記'][int(ind_lab_data.index[0])])
                # 讀取寶德量測資料
                data4 = pd.read_csv(L_factor_data['path'][ii], header=None,
                                    encoding='UTF-8', on_bad_lines='warn', engine='python').transpose()
                # 讀取 CSV 檔案，為解決 FWD 與 BWD 長短不一致的問題
                if np.shape(data4)[1] < 4:
                    with open(L_factor_data['path'][ii], newline='') as csvfile:
                        # 讀取 CSV 檔案內容
                        rows = csv.reader(csvfile)
                        row_len = []
                        # 以迴圈輸出每一列
                        for row in rows:
                            # print(row)
                            name = row[0]
                            age = float(row[1])
                            row_len.append(len(row))
                            # print(row)
                    with open(L_factor_data['path'][ii], newline='') as csvfile:
                        rows = csv.reader(csvfile)   
                        data4 = pd.DataFrame(np.zeros((len(row_len), max(row_len))))
                        i = 0
                        for row in rows:
                            data4.iloc[i, :row_len[i]] = row
                            i = i + 1
                    data4 = data4.transpose()
                    data4 = pd.DataFrame(data4.iloc[1:, :].values, columns=data4.iloc[0, :]).astype(float)
                    data4 = data4.fillna(0)
                    fdw_data = data4.iloc[:, [0,1]]
                    fdw_data = fdw_data.loc[~(fdw_data==0).all(axis=1)]
                    new_data_4 = pd.DataFrame(np.concatenate((fdw_data.iloc[:, [0,1]].values, data4.iloc[:, [2,3]].values), axis=0))

                else:
                    data4 = pd.DataFrame(data4.iloc[1:, :].values, columns=data4.iloc[0, :])
                    
                    data4 = data4.fillna(0)
                    # 轉置資料，並將 NAN fill with 0 point
                    new_data_4 = pd.DataFrame(np.concatenate((data4.iloc[:, [0,1]].values, data4.iloc[:, [2,3]].values), axis=0))

                # new_data_4 = new_data_4.loc[~(new_data_4==0).all(axis=1)]
                plt.figure()
                # 寶德量測資料
                plt.plot(new_data_4.iloc[:, 0], new_data_4.iloc[:, 1], label = 'factory',linewidth = 0.5, color='b')
                plt.hlines(L_factor_data['整機_F(P2)_按壓力-左'][ii], 0, 1, linewidth = 0.5, color='b')
                plt.hlines(L_factor_data['整機_F(R3)_回彈力-左'][ii], 0, 1, linewidth = 0.5, color='b')
                # 實驗室量測資料
                plt.plot(data1.iloc[:, 1], data1.iloc[:, 0]/0.0098, label = 'test1',linewidth = 0.5)
                plt.plot(data2.iloc[:, 1], data2.iloc[:, 0]/0.0098, label = 'test2',linewidth = 0.5)
                plt.plot(data3.iloc[:, 1], data3.iloc[:, 0]/0.0098, label = 'test3',linewidth = 0.5)
                plt.hlines(all_self_table['SwitchPeakForce_Com'][int(ind_lab_data.index[0])]/0.0098, 0, 1, linewidth = 0.5, color = 'r')
                plt.hlines(all_self_table['SwitchPeakForce_Re'][int(ind_lab_data.index[0])]/0.0098, 0, 1, linewidth = 0.5, color = 'r')
                force_diff = (abs(all_self_table['SwitchPeakForce_Com'][int(ind_lab_data.index[0])]/0.0098 - L_factor_data['整機_F(P2)_按壓力-左'][ii]) + \
                              abs(all_self_table['SwitchPeakForce_Re'][int(ind_lab_data.index[0])]/0.0098 - L_factor_data['整機_F(R3)_回彈力-左'][ii]))
                if 10 > force_diff >= 5:
                    plt.annotate('按壓力差值超過5g', (0.7, 1), color = 'c')
                elif force_diff > 10:
                    plt.annotate('按壓力差值超過10g', (0.7, 1), color = 'r')
                plt.xlim(0, 1)
                plt.ylim(0, 100)
                plt.title(all_self_table['試驗註記'][int(ind_lab_data.index[0])])
                plt.legend()
                plt.xlabel("按壓距離 (mm)")
                plt.ylabel("按壓力量 (g)")
                plt.savefig(str(save_path + all_self_table['試驗註記'][int(ind_lab_data.index[0])] + '.jpg'),
                            dpi=200, bbox_inches = "tight")
                plt.show()


# %% EC2CW pick 3

def read_csv(data, data_path):
    if np.shape(data)[1] < 4:
        with open(data_path, newline='') as csvfile:
            # 讀取 CSV 檔案內容
            rows = csv.reader(csvfile)
            row_len = []
            # 以迴圈輸出每一列
            for row in rows:
                # print(row)
                row_len.append(len(row))
                # print(row)
                with open(data_path, newline='') as csvfile:
                    rows = csv.reader(csvfile)   
                    data = pd.DataFrame(np.zeros((len(row_len), max(row_len))))
                    i = 0
                    for row in rows:
                        data.iloc[i, :row_len[i]] = row
                        i = i + 1
                    data = data.transpose()
                    data = pd.DataFrame(data.iloc[1:, :].values, columns=data.iloc[0, :]).astype(float)
                    data = data.fillna(0)
                    fdw_data = data.iloc[:, [0,1]]
                    fdw_data = fdw_data.loc[~(fdw_data==0).all(axis=1)]
                    new_data_4 = pd.DataFrame(np.concatenate((fdw_data.iloc[:, [0,1]].values, data.iloc[:, [2,3]].values), axis=0))
    else:
        data = pd.DataFrame(data.iloc[1:, :].values, columns=data.iloc[0, :])
        
        data = data.fillna(0)
        # 轉置資料，並將 NAN fill with 0 point
        new_data_4 = pd.DataFrame(np.concatenate((data.iloc[:, [0,1]].values, data.iloc[:, [2,3]].values), axis=0))
    
        # new_data_4 = new_data_4.loc[~(new_data_4==0).all(axis=1)]
    return new_data_4
# %%
# Table data
table_data = pd.read_excel(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\20230627\compare\compare_list.xlsx")

for i in range(np.shape(table_data)[0]):
    # 讀取寶德量測資料
    bad_data = pd.read_csv(table_data.loc[i, 'bad_right'], header=None,
                    encoding='UTF-8', on_bad_lines='warn', engine='python').transpose()
    good_data = pd.read_csv(table_data.loc[i, 'good_right'], header=None,
                    encoding='UTF-8', on_bad_lines='warn', engine='python').transpose()
    # 讀取 CSV 檔案，為解決 FWD 與 BWD 長短不一致的問題
    bad_data = read_csv(bad_data, table_data.loc[i, 'bad_right'])
    good_data = read_csv(good_data, table_data.loc[i, 'bad_right'])
    
    plt.figure()
    # 寶德量測資料
    plt.plot(bad_data.iloc[:, 0], bad_data.iloc[:, 1], label = 'bad',linewidth = 0.5, color='r')
    # plt.hlines(L_factor_data['整機_F(P2)_按壓力-左'][ii], 0, 1, linewidth = 0.5, color='b')
    # plt.hlines(L_factor_data['整機_F(R3)_回彈力-左'][ii], 0, 1, linewidth = 0.5, color='b')
    plt.plot(good_data.iloc[:, 0], good_data.iloc[:, 1], label = 'good',linewidth = 0.5, color='b')
    plt.legend()
    plt.show()
    


















































