# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:10:09 2023

@author: Hsin.YH.Yang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import math
import scipy.stats as stats
from prettytable import PrettyTable
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# %% 3. Create violin plot.

def plot_violin(data, y_axis, y_axis_name, fig_title, save_name):
    n = int(np.ceil(len(y_axis) / 2 ))
    
    fig, axs = plt.subplots(n, 2, figsize=(n*3, 10))
    for fig_num, axs in enumerate(axs.flatten()):
        # xx, yy = fig_num - 3*math.floor(abs(fig_num)/3), math.floor(abs(fig_num)/3)
        # print(fig_num, ax)
        sns.violinplot(x="mouse", y=data.columns[y_axis[fig_num]], data=input_data, palette="Set3", ax = axs)
        axs.set_xlabel("mouse", fontsize=14)
        axs.set_ylabel(y_axis_name, fontsize=14)
        axs.set_title(data.columns[max_ind[fig_num]], fontsize = 16)
    plt.tight_layout()    
    plt.suptitle(str(fig_title + "(motion): " + save_name), fontsize = 20)
    plt.subplots_adjust(top=0.90)
    plt.grid(False)
    plt.savefig(str(r"E:\BenQ_Project\U3\09_Results\statistic_fig\\" + fig_title + "_Motion.jpg"))
    plt.show()
# %% read data
# Blink
data = pd.read_excel(r"E:\BenQ_Project\U3\09_Results\data\All_Blink_motion_table.xlsx")
# Spider
# data = pd.read_excel(r"E:\BenQ_Project\U3\09_Results\data\All_Blink_motion_table.xlsx")
task_name = "Blink_Large"
# %% 設定資料分群 motion

# nolumn_name = data.columns[6:]
max_ind = []
mean_ind = []
for i in range(len(data.columns)):
    if "mean" in data.columns[i]:
        mean_ind.append(i)
    elif "max" in data.columns[i]:
        max_ind.append(i)
    
target_subjects = ["S02", "S06", "S08"] # 小手
# target_subjects = ["S01", "S03", "S04", "S05"] # 大手
indexes = [i for i, subject in enumerate(data["subject"]) if any(target in subject for target in target_subjects)]

data = data.loc[indexes, :]


# 設定圖片標頭
data_head = dict({"position": ["elbow", "wrist"],
                  "cal": [max_ind, mean_ind],
                  "method":["vel", "acc"]})
x = PrettyTable()
for position in data_head["position"]:
    for method in range(len(data_head["method"])):
        for cal in range(len(data_head["cal"])):
            if cal == 0:
                cal_method = "Max"    
            elif cal == 1:
                cal_method = "Mean"
            if method == 0:
                y_axis_name = "deg/s"
            elif method == 1:
                y_axis_name = "deg/${s^2}$"
            fig_title = task_name + " " + position + " " + data_head["method"][method] + " " + cal_method
            input_index = (data["method"] == data_head["method"][method]) & (data["position"] == position)
            input_data = data.loc[input_index, :]
            y_axis = data_head["cal"][cal]
            plot_violin(input_data, y_axis, y_axis_name = y_axis_name,
                        fig_title = fig_title, save_name = fig_title)

                
print(x)
#　統計檢定
pairwise_tukeyhsd_results = []
x = PrettyTable()
for position in data_head["position"]:
    for method in data_head["method"]:
        for cal in range(len(data_head["cal"])):
            # fig_title = task_name + " " + position + " " + data_head["method"][method] + " " + cal_method
            group1_index = (data["method"] == method) & (data["position"] == position) & (data["mouse"] == "U2")
            group2_index = (data["method"] == method) & (data["position"] == position) & (data["mouse"] == "C")
            group3_index = (data["method"] == method) & (data["position"] == position) & (data["mouse"] == "C1")
            group4_index = (data["method"] == method) & (data["position"] == position) & (data["mouse"] == "D1")
            # 設定資料

            # input_data = data.loc[input_index, :]
            y_axis = data_head["cal"][cal]
            # plot_violin(input_data, y_axis, y_axis_name = y_axis_name,
            #             fig_title = fig_title, save_name = fig_title)
            for column_num in data_head["cal"][cal]:
                # stat_data = data.loc[input_index, input_data.columns[column_num]]
                # 進行 Shapiro-Wilk 檢定
                # statistic, p_value = stats.shapiro(stat_data)
                # 設定資料
                group1_data = data.loc[group1_index, data.columns[column_num]]
                group2_data = data.loc[group2_index, data.columns[column_num]]
                group3_data = data.loc[group3_index, data.columns[column_num]]
                group4_data = data.loc[group4_index, data.columns[column_num]]
                # 統計分析
                statistic, p_value = stats.f_oneway(data.loc[group1_index, data.columns[column_num]],
                                                    data.loc[group2_index, data.columns[column_num]],
                                                    data.loc[group3_index, data.columns[column_num]],
                                                    data.loc[group4_index, data.columns[column_num]])
                # 判斷結果
                alpha = 0.05
                if p_value > alpha:
                    result = "無顯著差異"
                else:
                    result = "顯著差異"
                x.field_names = ["正態檢定方法", "數據欄位", "結論"]
                x.add_row(["anova", data.columns[column_num], result])
                # 進行事後檢定
                print(data.columns[column_num])
                data_1 = group1_data + group2_data + group3_data + group4_data
                labels = ['U2'] * len(group1_data) + ['C'] * len(group2_data) + \
                    ['C1'] * len(group3_data) + ['D1'] * len(group4_data)
                tukey_result = pairwise_tukeyhsd(data_1, labels, alpha=0.05)
                pairwise_tukeyhsd_results.append(tukey_result)
                print(tukey_result)

print(x)
output_file_path = r'E:\BenQ_Project\U3\09_Results\Blink_all_tukey_results.txt'
with open(output_file_path, 'w') as f:
    for result in pairwise_tukeyhsd_results:
        f.write(str(result))
        f.write('\n\n')  # 加入空行分隔不同結果
  


# %% 畫 EMG MedFrequency結果            
emg_data = pd.read_excel(r"E:\BenQ_Project\U3\09_Results\data\All_Spyder_MedFreq_table.xlsx")
task_name = "Spider_MedFreq_Large"
mucsle_name = ['Extensor Rad', 'Extensor Uln', 'Flexor Rad', 'first dorsal interossei',
               'third dorsal interossei', 'abductor digiti minimi muscle', 'index finger indict',
               'Biceps']
# target_subjects = ["S02", "S06", "S08"] # 小手
target_subjects = ["S01", "S03", "S04", "S05"] # 大手
# target_subjects = ["S02", "S06", "S08"] + ["S01", "S03", "S04", "S05"] # all
indexes = [i for i, subject in enumerate(emg_data["subject"]) if any(target in subject for target in target_subjects)]

emg_data = emg_data.loc[indexes, :]
group1_emg = emg_data[emg_data["mouse"] == "U2"]
group2_emg = emg_data[emg_data["mouse"] == "C"]
group3_emg = emg_data[emg_data["mouse"] == "C1"]
group4_emg = emg_data[emg_data["mouse"] == "D1"]
# 設定圖片標頭
data_table = pd.DataFrame({}, columns = ["muscle", "U2", "C", "C1", "D1"])
x = PrettyTable()
for muscle in mucsle_name:
    x_span = np.arange(0, 46, 1)
    data1_y= np.array([])
    data1_x= np.array([])
    data2_y= np.array([])
    data2_x= np.array([])
    data3_y= np.array([])
    data3_x= np.array([])
    data4_y= np.array([])
    data4_x= np.array([])
    
    # slope, intercept = np.polyfit(x_span, data1, 1)
    # # 使用 polyval 函數計算趨勢線的預測值
    # trend_line = np.polyval([slope, intercept], x)
    data1 = group1_emg[group1_emg["columns_num"] == muscle].iloc[:, 5:-1]
    data2 = group2_emg[group2_emg["columns_num"] == muscle].iloc[:, 5:-1]
    data3 = group3_emg[group3_emg["columns_num"] == muscle].iloc[:, 5:-1]
    data4 = group4_emg[group4_emg["columns_num"] == muscle].iloc[:, 5:-1]

    for num in range(np.shape(data1)[0]):
        data1_y = np.concatenate([data1_y, ((data1.iloc[num, :].values-np.mean(data1.iloc[num, :].values)))])
        data1_x = np.concatenate([data1_x, x_span])
    for num in range(np.shape(data2)[0]):
        data2_y = np.concatenate([data2_y, (data2.iloc[num, :].values - np.mean(data2.iloc[num, :].values))])
        data2_x = np.concatenate([data2_x, x_span])
    for num in range(np.shape(data3)[0]):
        data3_y = np.concatenate([data3_y, (data3.iloc[num, :].values - np.mean(data3.iloc[num, :].values))])
        data3_x = np.concatenate([data3_x, x_span])
    for num in range(np.shape(data4)[0]):
        data4_y = np.concatenate([data4_y, (data4.iloc[num, :].values - np.mean(data4.iloc[num, :].values))])
        data4_x = np.concatenate([data4_x, x_span])
    # 使用 polyfit 函數進行線性回歸，獲取斜率和截距
    slope1, intercept = np.polyfit(data1_x, data1_y, 1)
    # # 使用 polyval 函數計算趨勢線的預測值
    trend_line_1 = np.polyval([slope1, intercept], data1_x)
    plt.scatter(data1_x, data1_y, c ="blue", s=2)
    plt.plot(data1_x, trend_line_1, color='blue', label=f'U2: Slope = {slope1:.2f}')


    # data 2
    slope2, intercept = np.polyfit(data2_x, data2_y, 1)
    trend_line_2 = np.polyval([slope2, intercept], data2_x)
    plt.scatter(data2_x, data2_y, c ="red", s=2)
    plt.plot(data2_x, trend_line_2, color='red', label=f'C: Slope = {slope2:.2f}')


    # data 3
    slope3, intercept = np.polyfit(data3_x, data3_y, 1)
    trend_line_3 = np.polyval([slope3, intercept], data3_x)
    plt.scatter(data3_x, data3_y, c ="green", s=2)
    plt.plot(data3_x, trend_line_3, color='green', label=f'C1: Slope = {slope3:.2f}')


    # data 4
    slope4, intercept = np.polyfit(data4_x, data4_y, 1)
    trend_line_4 = np.polyval([slope4, intercept], data4_x)
    plt.scatter(data4_x, data4_y, c ="orange", s=2)
    plt.plot(data4_x, trend_line_4, color='orange', label=f'D1: Slope = {slope4:.2f}')


    plt.title(str(task_name + "_" + muscle))
    plt.legend(loc='upper right')
    plt.savefig(str(r"E:\BenQ_Project\U3\09_Results\statistic_fig\\" + task_name + "_" + muscle + ".jpg"),
                dpi=200)
    plt.show()
    # 插入表格數值
    data_table = pd.concat([data_table,
                            pd.DataFrame({"muscle":muscle, "U2":slope1, "C":slope2, "C1":slope3, "D1":slope4},
                                         index=[0])],
                           ignore_index=True)
    # 統計分析
    statistic, p_value = stats.f_oneway(data1_y,
                                        data2_y,
                                        data3_y,
                                        data4_y)
    # 判斷結果
    alpha = 0.05
    if p_value > alpha:
        result = "無顯著差異"
    else:
        result = "顯著差異"
    x.field_names = ["正態檢定方法", "數據欄位", "結論"]
    x.add_row(["anova", muscle, result])
    # 進行事後檢定
    print(muscle)
print(x)
    # data_11 = data1_y + data2_y + data3_y + data4_y
    # labels = ['U2'] * len(data1_y) + ['C'] * len(data2_y) + \
    #         ['C1'] * len(data3_y) + ['D1'] * len(data4_y)
    # tukey_result = pairwise_tukeyhsd(data_11, labels, alpha=0.05)
    # pairwise_tukeyhsd_results.append(tukey_result)
    # print(tukey_result)

# %%
allemg_data = pd.read_excel(r"E:\BenQ_Project\U3\09_Results\data\All_Blink_emg_table.xlsx")

emg_table = allemg_data[allemg_data["type"]=="all"].reset_index()
del allemg_data
task_name = "Blink_Large"

#  找出欄位名稱中含有max, mean的欄位索引
max_ind = []
mean_ind = []
for i in range(len(emg_table.columns)):
    if "mean" in emg_table.columns[i]:
        mean_ind.append(i)
    elif "max" in emg_table.columns[i]:
        max_ind.append(i)
    
# target_subjects = ["S02", "S06", "S08"] # 小手
# target_subjects = ["S01", "S03", "S04", "S05"] # 大手
# target_subjects = ["S02", "S06", "S08"] + ["S01", "S03", "S04", "S05"] # all
indexes = [i for i, subject in enumerate(emg_table["subject"]) if any(target in subject for target in target_subjects)]
emg_table = emg_table.loc[indexes, :]

# 找出含有以下名稱的行
table_head = dict({"axis":["elbow_x", "elbow_y", "elbow_z", "wrist_x", "wrist_y", "wrist_z"],
                   "direction":["+", "-"],
                   "cal": [max_ind, mean_ind],
                   "subject":target_subjects,
                   "mouse":["U2", "C", "C1", "D1"]})

columns_name = ["direction", "axis", "subject", "mouse"] + list(emg_table.iloc[:, mean_ind].columns)
mean_data_table = pd.DataFrame({}, columns = columns_name)
for direction in table_head["direction"]:
    for axis in table_head["axis"]:
        for subject in table_head["subject"]:
            for mouse in table_head["mouse"]:
                input_index = (emg_table["direction"] == direction) & (emg_table["axis"] == axis) &\
                    (emg_table["subject"] == subject) & (emg_table["mouse"] == mouse)
                input_data = emg_table.loc[input_index, :]
                input_data = input_data.iloc[:, mean_ind]
                mean_input_data = np.mean(input_data, axis=0)
                mean_data_table = pd.concat([mean_data_table,
                                              pd.DataFrame([([direction, axis, subject, mouse] + list(mean_input_data))],
                                                          columns=columns_name,
                                                          index=[0])],
                                            ignore_index=True)

            print(direction, axis)

            fig_title = task_name + " " + axis + " " + direction + " " + cal_method
            input_index = (emg_table["direction"] == direction) & (emg_table["axis"] == axis)
            input_data = emg_table.loc[input_index, :]


            
            

for direction in table_head["direction"]:
    for axis in table_head["axis"]:
        for cal in range(len(table_head["cal"])):
            if cal == 0:
                cal_method = "Max"    
            elif cal == 1:
                cal_method = "Mean"
            print(direction, axis)

            fig_title = task_name + " " + axis + " " + direction + " " + cal_method
            input_index = (emg_table["direction"] == direction) & (emg_table["axis"] == axis)
            input_data = emg_table.loc[input_index, :]
            y_axis = table_head["cal"][cal]
            plot_violin(input_data, y_axis, y_axis_name = "percentage (%)",
                        fig_title = fig_title, save_name = fig_title)

#　統計檢定
pairwise_tukeyhsd_results = []
x = PrettyTable()
for direction in table_head["direction"]:
    for axis in table_head["axis"]:
        for cal in range(len(table_head["cal"])):
            # fig_title = task_name + " " + position + " " + table_head["method"][method] + " " + cal_method
            group1_index = (emg_table["direction"] == direction) & (emg_table["axis"] == axis) & (emg_table["mouse"] == "U2")
            group2_index = (emg_table["direction"] == direction) & (emg_table["axis"] == axis) & (emg_table["mouse"] == "C")
            group3_index = (emg_table["direction"] == direction) & (emg_table["axis"] == axis) & (emg_table["mouse"] == "C1")
            group4_index = (emg_table["direction"] == direction) & (emg_table["axis"] == axis) & (emg_table["mouse"] == "D1")
            # 設定資料

            # input_data = data.loc[input_index, :]
            y_axis = table_head["cal"][cal]
            # plot_violin(input_data, y_axis, y_axis_name = y_axis_name,
            #             fig_title = fig_title, save_name = fig_title)
            for column_num in table_head["cal"][cal]:
                # stat_data = data.loc[input_index, input_data.columns[column_num]]
                # 進行 Shapiro-Wilk 檢定
                # statistic, p_value = stats.shapiro(stat_data)
                # 設定資料
                group1_data = emg_table.loc[group1_index, emg_table.columns[column_num]]
                group2_data = emg_table.loc[group2_index, emg_table.columns[column_num]]
                group3_data = emg_table.loc[group3_index, emg_table.columns[column_num]]
                group4_data = emg_table.loc[group4_index, emg_table.columns[column_num]]
                # 統計分析
                statistic, p_value = stats.f_oneway(emg_table.loc[group1_index, emg_table.columns[column_num]],
                                                    emg_table.loc[group2_index, emg_table.columns[column_num]],
                                                    emg_table.loc[group3_index, emg_table.columns[column_num]],
                                                    emg_table.loc[group4_index, emg_table.columns[column_num]])
                # 判斷結果
                alpha = 0.05
                if p_value > alpha:
                    result = "無顯著差異"
                else:
                    result = "顯著差異"
                x.field_names = ["正態檢定方法", "數據欄位", "結論"]
                x.add_row(["anova", emg_table.columns[column_num], result])
                # 進行事後檢定
                print(emg_table.columns[column_num])
                data_1 = group1_data + group2_data + group3_data + group4_data
                labels = ['U2'] * len(group1_data) + ['C'] * len(group2_data) + \
                         ['C1'] * len(group3_data) + ['D1'] * len(group4_data)
                tukey_result = pairwise_tukeyhsd(data_1, labels, alpha=0.05)
                pairwise_tukeyhsd_results.append(tukey_result)
                print(tukey_result)


output_file_path = r"E:\BenQ_Project\U3\09_Results\\" + task_name + "_tukey_results.txt"
with open(output_file_path, 'w') as f:
    for result in pairwise_tukeyhsd_results:
        f.write(str(result))
        f.write('\n\n')  # 加入空行分隔不同結果


# %% 

















