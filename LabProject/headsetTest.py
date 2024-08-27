# -*- coding: utf-8 -*-
"""
analysis the answer sheet of heatset test

填答邏輯
1. 方向對：
    只要8個方向有數字即可算對
2. 人數對：
    8個方向有數字，且數字與答案相符
3. 樓層對：
    8個方向對，人數不一定需要相符，但是要確認樓層是否正確


@author: Hsin.YH.Yang, written by May 08 2024
"""
# %% import library
import pandas as pd
import numpy as np
import os
from datetime import datetime


# %% parameter
# 方向設定
subject_info = ["受試者", "耳機", "編號"
                ]
direct_columns = ["前","右前","右","右後",
                  "後","左後","左","左前",
                    "L", "M", "R"
                  ]
floor_columns = ["前_2F","前_1F","前_B","右前_2F","右前_1F","右前_B","右_2F","右_1F","右_B",
                  "右後_2F","右後_1F","右後_B","後_2F","後_1F","後_B","左後_2F","左後_1F","左後_B",
                  "左_2F","左_1F","左_B","左前_2F","左前_1F","左前_B",
                    "L", "M", "R"
                  ]
headset_name = ["耳罩式耳機A", "耳罩式耳機B", "耳罩式耳機C", #"耳罩式耳機D", # "耳罩式耳機A_I1",
                "入耳式耳機1", "入耳式耳機2", #"入耳式耳機3",
                "音效盒1", "音效盒2", "音效盒3", "音效盒4"
                ]
# data_path = r"D:\BenQ_Project\01_UR_lab\2024_05 耳機\S01_答案.xlsx"
subject_answer_folder = r"D:\BenQ_Project\01_UR_lab\2024_05 耳機\BQC test\answer\\"
# subject_answer_folder = r"D:\BenQ_Project\01_UR_lab\2024_05 耳機\subject_answer\\"

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
# %% read data
answer_sheet = pd.read_excel(r"D:\BenQ_Project\01_UR_lab\2024_05 耳機\BQC test\答案_0722.xlsx")


# answer_sheet = pd.read_excel(r"E:\Hsin\BenQ\BQC電競耳機測試答案S1-S5\答案_0722.xlsx")

# row_num = 30

# read data path
all_data_path = Read_File(subject_answer_folder,
                          ".xlsx")
all_data_path = [f for f in all_data_path if not os.path.split(f)[1].startswith(('.', '~$'))]
# data_sheet = pd.read_excel(data_path,
#                            sheet_name="耳罩式耳機A")
# 0. 紀錄表格
# 方向對
direction_answer = pd.DataFrame(np.zeros((len(all_data_path)*len(headset_name),
                                          len(subject_info + direct_columns))),
                                columns=[subject_info + direct_columns])
# 方向對 答案
direction_question = pd.DataFrame(np.zeros((len(all_data_path)*len(headset_name),
                                          len(subject_info + direct_columns))),
                                  columns=[subject_info + direct_columns])

# 方向對(人數權重) + 0.5
direction_ans_weight = pd.DataFrame(np.zeros((len(all_data_path)*len(headset_name),
                                          len(subject_info + direct_columns))),
                                    columns=[subject_info + direct_columns])

# 方向對(人數權重)
direction_ans_weight_00 = pd.DataFrame(np.zeros((len(all_data_path)*len(headset_name),
                                          len(subject_info + direct_columns))),
                                    columns=[subject_info + direct_columns])

# 人數對
number_answer = pd.DataFrame(np.zeros((len(all_data_path)*len(headset_name),
                                       len(subject_info + direct_columns))),
                             columns=[subject_info + direct_columns])
# 樓層對
# columns change to floor columns name
floor_answer = pd.DataFrame(np.zeros((len(all_data_path)*len(headset_name),
                                      len(subject_info + floor_columns))),
                             columns=[subject_info + floor_columns])
# 方向對，簡易版
sam_direction_answer = pd.DataFrame({},
                                    columns=[subject_info + ["測試順序"] + direct_columns])

# 方向對，答案全寫
sam_direction_answer_11 = pd.DataFrame({},
                                    columns=[subject_info + ["測試順序"] + direct_columns])
# 方向對，統計比例
sam_direction_answer_noO = pd.DataFrame({},
                                        columns=[subject_info + ["測試順序"] + direct_columns])

# 方向對，去掉 NAN vs NAN
# 計算正確的欄位數，並除以出現方位數
direction_answer_noO = pd.DataFrame(np.zeros((len(all_data_path)*len(headset_name),
                                              len(subject_info + direct_columns))),
                                    columns=[subject_info + direct_columns])
# 方向對的題數，有出現目標，並且有填答
direction_answer_noOO = pd.DataFrame(np.zeros((len(all_data_path)*len(headset_name),
                                              len(subject_info + direct_columns))),
                                    columns=[subject_info + direct_columns])
# 沒有出現目標，但是有填寫
direction_answer_noX = pd.DataFrame(np.zeros((len(all_data_path)*len(headset_name),
                                              len(subject_info + direct_columns))),
                                    columns=[subject_info + direct_columns])
# 有出現目標，但是沒有填寫
direction_answer_noXX = pd.DataFrame(np.zeros((len(all_data_path)*len(headset_name),
                                                 len(subject_info + direct_columns))),
                                     columns=[subject_info + direct_columns])
# 各方向出題數量
answer_direction_noOO = pd.DataFrame(np.zeros((len(all_data_path)*len(headset_name),
                                              len(subject_info + direct_columns))),
                                    columns=[subject_info + direct_columns])
# 0. 方向對 簡化版
# 依照欄位編號、方向對答案
# 再加上受試者編號以及耳機

for subject in range(len(all_data_path)):
    print(all_data_path[subject])
    excel_data = pd.ExcelFile(all_data_path[subject])

    # Get the number of sheets
    number_of_sheets = len(excel_data.sheet_names)
    for headset in range(number_of_sheets):
        for name in range(len(headset_name)):
            if excel_data.sheet_names[headset] == headset_name[name]:
                data_sheet = pd.read_excel(all_data_path[subject],
                                           sheet_name = excel_data.sheet_names[headset])
                temp_direction_answer = pd.DataFrame(np.zeros((len(data_sheet),
                                                              len(subject_info + direct_columns)+1)),
                                                    columns=[subject_info + ["測試順序"] + direct_columns])
                temp_direction_answerr_noO = pd.DataFrame(np.zeros((len(data_sheet),
                                                                    len(subject_info + direct_columns)+1)),
                                                          columns=[subject_info + ["測試順序"] + direct_columns])
                columns_number = subject*6 + headset
                print(columns_number)
        
                for row in range(len(data_sheet)):
                    temp_direction_answer.loc[row, "受試者"] = data_sheet["受試者"][0]
                    temp_direction_answer.loc[row, "耳機"] = excel_data.sheet_names[headset]
                    temp_direction_answer.loc[row, "測試順序"] = data_sheet.loc[row, "測試順序"]
                    temp_direction_answerr_noO.loc[row, "受試者"] = data_sheet["受試者"][0]
                    temp_direction_answerr_noO.loc[row, "耳機"] = excel_data.sheet_names[headset]
                    temp_direction_answerr_noO.loc[row, "測試順序"] = data_sheet.loc[row,"測試順序"]
                    for i in range(len(answer_sheet)):
                        # 對欄位
                        if data_sheet.loc[row, "編號"] == answer_sheet.loc[i, "編號"]:
                            print(data_sheet.loc[row, "編號"])
                            temp_direction_answer.loc[row, "編號"] = data_sheet.loc[row, "編號"]
              
                            temp_direction_answerr_noO.loc[row, "編號"] = data_sheet.loc[row, "編號"]
        
                            # 對方位
                            for direct in direct_columns:
                                if pd.isna(data_sheet.loc[row, direct]) == False and \
                                    pd.isna(answer_sheet.loc[i, direct]) == False:
                                    temp_direction_answer.loc[row, direct] = data_sheet.loc[row, direct] \
                                         / answer_sheet.loc[i, direct_columns].fillna(0).sum()
                                    if data_sheet.loc[row, direct] > answer_sheet.loc[i, direct]:
                                        temp_direction_answerr_noO.loc[row, direct] = answer_sheet.loc[i, direct] 
                                    else:
                                        temp_direction_answerr_noO.loc[row, direct] = data_sheet.loc[row, direct]
                                        
                sam_direction_answer = pd.concat([sam_direction_answer, temp_direction_answer],
                                                  ignore_index=True)
                sam_direction_answer_noO = pd.concat([sam_direction_answer_noO, temp_direction_answerr_noO],
                                                     ignore_index=True)



# 1. 方向對
# 依照欄位編號、方向對答案
# 再加上受試者編號以及耳機
# columns_number = 0
for subject in range(len(all_data_path)):
    print(all_data_path[subject])
    excel_data = pd.ExcelFile(all_data_path[subject])

    # Get the number of sheets
    number_of_sheets = len(excel_data.sheet_names)
    for headset in range(number_of_sheets):
        for name in range(len(headset_name)):
            if excel_data.sheet_names[headset] == headset_name[name]:
                data_sheet = pd.read_excel(all_data_path[subject],
                                           sheet_name = excel_data.sheet_names[headset])
                print(excel_data.sheet_names[headset])
                columns_number = subject*len(headset_name) + name
                print(columns_number)
                direction_answer.loc[columns_number, "受試者"] = data_sheet["受試者"][0]
                direction_answer.loc[columns_number, "耳機"] = excel_data.sheet_names[headset]
                direction_question.loc[columns_number, "受試者"] = data_sheet["受試者"][0]
                direction_question.loc[columns_number, "耳機"] = excel_data.sheet_names[headset]
                
                direction_ans_weight.loc[columns_number, "受試者"] = data_sheet["受試者"][0]
                direction_ans_weight.loc[columns_number, "耳機"] = excel_data.sheet_names[headset]
                
                direction_ans_weight_00.loc[columns_number, "受試者"] = data_sheet["受試者"][0]
                direction_ans_weight_00.loc[columns_number, "耳機"] = excel_data.sheet_names[headset]
                
                direction_answer_noO.loc[columns_number, "受試者"] = data_sheet["受試者"][0]
                direction_answer_noO.loc[columns_number, "耳機"] = excel_data.sheet_names[headset]
                direction_answer_noX.loc[columns_number, "受試者"] = data_sheet["受試者"][0]
                direction_answer_noX.loc[columns_number, "耳機"] = excel_data.sheet_names[headset]
                direction_answer_noXX.loc[columns_number, "受試者"] = data_sheet["受試者"][0]
                direction_answer_noXX.loc[columns_number, "耳機"] = excel_data.sheet_names[headset]
                direction_answer_noOO.loc[columns_number, "受試者"] = data_sheet["受試者"][0]
                direction_answer_noOO.loc[columns_number, "耳機"] = excel_data.sheet_names[headset]
                answer_direction_noOO.loc[columns_number, "受試者"] = data_sheet["受試者"][0]
                answer_direction_noOO.loc[columns_number, "耳機"] = excel_data.sheet_names[headset]
                
                for row in range(np.shape(data_sheet)[0]):
                    for i in range(len(answer_sheet)):
                        # 對欄位
                        if data_sheet.loc[row, "編號"] == answer_sheet.loc[i, "編號"]:
                            # print(data_sheet.loc[row, "編號"])
                            # 對方位
                            direction_answer.loc[columns_number, "編號"] = data_sheet.loc[row, "編號"]
                            direction_question.loc[columns_number, "編號"] = data_sheet.loc[row, "編號"]
                            direction_ans_weight.loc[columns_number, "編號"] = data_sheet.loc[row, "編號"]
                            direction_ans_weight_00.loc[columns_number, "編號"] = data_sheet.loc[row, "編號"]
                            direction_answer_noO.loc[columns_number, "編號"] = data_sheet.loc[row, "編號"]
                            direction_answer_noX.loc[columns_number, "編號"] = data_sheet.loc[row, "編號"]
                            direction_answer_noXX.loc[columns_number, "編號"] = data_sheet.loc[row, "編號"]
                            direction_answer_noOO.loc[columns_number, "編號"] = data_sheet.loc[row, "編號"]
                            answer_direction_noOO.loc[columns_number, "編號"] = data_sheet.loc[row, "編號"]
          
                            for direct in direct_columns:
                                # 方向對
                                if pd.isna(answer_sheet.loc[i, direct]) == False and\
                                    pd.isna(data_sheet.loc[row, direct]) == False:
                                    direction_answer.loc[columns_number, direct] = direction_answer.loc[columns_number, direct].values + \
                                        (1/np.sum(answer_sheet.loc[i, direct_columns]))
                                # 方向對 答案
                                if pd.isna(answer_sheet.loc[i, direct]) == False:
                                    direction_question.loc[columns_number, direct] = direction_question.loc[columns_number, direct].values + 1
                                # 方向對(人數權重) + 0.5
                                if pd.isna(answer_sheet.loc[i, direct]) == False and\
                                    pd.isna(data_sheet.loc[row, direct]) == False:
                                    if answer_sheet.loc[i, direct] == data_sheet.loc[row, direct]:
                                        direction_ans_weight.loc[columns_number, direct] = direction_ans_weight.loc[columns_number, direct].values + \
                                            (1/np.sum(answer_sheet.loc[i, direct_columns]))
                                    else:
                                        direction_ans_weight.loc[columns_number, direct] = direction_ans_weight.loc[columns_number, direct].values + \
                                            (0.5/np.sum(answer_sheet.loc[i, direct_columns]))
                                # 方向對(人數權重)
                                if pd.isna(answer_sheet.loc[i, direct]) == False and\
                                    pd.isna(data_sheet.loc[row, direct]) == False:    
                                    direction_ans_weight_00.loc[columns_number, direct] = direction_ans_weight_00.loc[columns_number, direct].values + \
                                        (data_sheet.loc[row, direct]//np.sum(answer_sheet.loc[i, direct_columns]))
                                # 計算正確的欄位數，並除以出現方位數
                                if pd.isna(answer_sheet.loc[i, direct]) == False and\
                                    pd.isna(data_sheet.loc[row, direct]) != pd.isna(answer_sheet.loc[i, direct]):
                                    direction_answer_noO.loc[columns_number, direct] = direction_answer_noO.loc[columns_number, direct].values + \
                                        1/ (len(answer_sheet.loc[i, direct_columns]) - pd.isna(answer_sheet.loc[i, direct_columns]).sum())
                                # 沒有出現目標，但是有填寫
                                if pd.isna(answer_sheet.loc[i, direct]) == True and\
                                    pd.isna(data_sheet.loc[row, direct]) == False:
                                    direction_answer_noX.loc[columns_number, direct] = direction_answer_noX.loc[columns_number, direct].values + 1
                                # 有出現目標，但是沒有填寫
                                if pd.isna(answer_sheet.loc[i, direct]) == False and\
                                    pd.isna(data_sheet.loc[row, direct]) == True:
                                    direction_answer_noXX.loc[columns_number, direct] = direction_answer_noXX.loc[columns_number, direct].values + 1
                                #  計算各方向出題數量
                                if pd.isna(answer_sheet.loc[i, direct]) == False:
                                    answer_direction_noOO.loc[columns_number, direct] = answer_direction_noOO.loc[columns_number, direct].values + \
                                        answer_sheet.loc[i, direct]
                                    if pd.isna(data_sheet.loc[row, direct]) == False:
                                        if data_sheet.loc[row, direct] > answer_sheet.loc[i, direct]:
                                            direction_answer_noOO.loc[columns_number, direct] = direction_answer_noOO.loc[columns_number, direct].values + \
                                                answer_sheet.loc[i, direct]
                                        else:
                                            direction_answer_noOO.loc[columns_number, direct] = direction_answer_noOO.loc[columns_number, direct].values + \
                                                    data_sheet.loc[row, direct]
                            
                        
                        
# 2. 人數對：
for subject in range(len(all_data_path)):
    print(all_data_path[subject])
    excel_data = pd.ExcelFile(all_data_path[subject])

    # Get the number of sheets
    number_of_sheets = len(excel_data.sheet_names)
    for headset in range(number_of_sheets):
        for name in range(len(headset_name)):
            if excel_data.sheet_names[headset] == headset_name[name]:
                data_sheet = pd.read_excel(all_data_path[subject],
                                           sheet_name = excel_data.sheet_names[headset])
                print(excel_data.sheet_names[headset])
                columns_number = subject*len(headset_name) + name
                print(columns_number)
                number_answer.loc[columns_number, "受試者"] = data_sheet["受試者"][0]
                number_answer.loc[columns_number, "耳機"] = excel_data.sheet_names[headset]
                for row in range(np.shape(data_sheet)[0]):
                    for i in range(len(answer_sheet)):
                        # 對欄位
                        if data_sheet.loc[row, "編號"] == answer_sheet.loc[i, "編號"]:
                            number_answer.loc[columns_number, "編號"] = data_sheet.loc[row, "編號"]
                            print(data_sheet.loc[row, "編號"])
                            # 對方位
                            for direct in direct_columns:
                                if data_sheet.loc[row, direct] != answer_sheet.loc[i, direct]:
                                    # 排除兩個答案都是NAN的問題，如果兩個都是NAN，也算對
                                    if pd.notna(data_sheet.loc[row, direct]) and pd.notna(answer_sheet.loc[i, direct]):
                                        number_answer.loc[columns_number, direct] = number_answer.loc[columns_number, direct].values + 1
                                        print(direct)                       
# 3. 樓層對
for subject in range(len(all_data_path)):
    print(all_data_path[subject])
    excel_data = pd.ExcelFile(all_data_path[subject])

    # Get the number of sheets
    number_of_sheets = len(excel_data.sheet_names)
    for headset in range(number_of_sheets):
        for name in range(len(headset_name)):
            if excel_data.sheet_names[headset] == headset_name[name]:
                data_sheet = pd.read_excel(all_data_path[subject],
                                           sheet_name = excel_data.sheet_names[headset])
                print(excel_data.sheet_names[headset])
                columns_number = subject*len(headset_name) + name
                print(columns_number)
                floor_answer.loc[columns_number, "受試者"] = data_sheet["受試者"][0]
                floor_answer.loc[columns_number, "耳機"] = excel_data.sheet_names[headset]
                for row in range(np.shape(data_sheet)[0]):
                    for i in range(len(answer_sheet)):
                        # 對欄位
                        if data_sheet.loc[row, "編號"] == answer_sheet.loc[i, "編號"]:
                            print(data_sheet.loc[row, "編號"])
                            floor_answer.loc[columns_number, "編號"] = data_sheet.loc[row, "編號"]
                            # 對方位
                            for floor in floor_columns:
                                print(floor)
                                if data_sheet.loc[row, floor] != answer_sheet.loc[i, floor]:
                                    # 排除兩個答案都是NAN的問題，如果兩個都是NAN，也算對
                                    if pd.notna(data_sheet.loc[row, floor]) and pd.notna(answer_sheet.loc[i, floor]):
                                        floor_answer.loc[columns_number, floor] = floor_answer.loc[columns_number, floor].values + 1
                                        print(direct)    


# 0. 方向對 簡化版
# 依照欄位編號、方向對答案
# 再加上受試者編號以及耳機
# for subject in range(len(all_data_path)):
    # for headset in range(len(headset_name)):
    #     data_sheet = pd.read_excel(all_data_path[subject],
    #                                 sheet_name=headset_name[headset])
    #     temp_direction_answer = pd.DataFrame(np.zeros((len(data_sheet),
    #                                                   len(subject_info + direct_columns)+1)),
    #                                         columns=[subject_info + ["測試順序"] + direct_columns])
    #     temp_direction_answerr_noO = pd.DataFrame(np.zeros((len(data_sheet),
    #                                                         len(subject_info + direct_columns)+1)),
    #                                               columns=[subject_info + ["測試順序"] + direct_columns])
    #     columns_number = subject*6 + headset
    #     print(columns_number)

    #     for row in range(len(data_sheet)):
    #         temp_direction_answer.loc[row, "受試者"] = data_sheet["受試者"][0]
    #         temp_direction_answer.loc[row, "耳機"] = headset_name[headset]
    #         temp_direction_answer.loc[row, "測試順序"] = data_sheet.loc[row, "測試順序"]
    #         temp_direction_answerr_noO.loc[row, "受試者"] = data_sheet["受試者"][0]
    #         temp_direction_answerr_noO.loc[row, "耳機"] = headset_name[headset]
    #         temp_direction_answerr_noO.loc[row, "測試順序"] = data_sheet.loc[row, "測試順序"]
    #         for i in range(len(answer_sheet)):
    #             # 對欄位
    #             if data_sheet.loc[row, "編號"] == answer_sheet.loc[i, "編號"]:
    #                 print(data_sheet.loc[row, "編號"])
    #                 # 對方位
    #                 for direct in direct_columns:
    #                     if pd.isna(data_sheet.loc[row, direct]) == False and \
    #                         pd.isna(answer_sheet.loc[i, direct]) == False:
    #                         temp_direction_answer.loc[row, direct] = data_sheet.loc[row, direct] \
    #                              / answer_sheet.loc[i, direct_columns].fillna(0).sum()
    #                     elif pd.isna(data_sheet.loc[row, direct]) == True and \
    #                         pd.isna(answer_sheet.loc[i, direct]) == False:
    #                         temp_direction_answer.loc[row, direct] = -1
    #                         # if data_sheet.loc[row, direct] > answer_sheet.loc[i, direct]:
    #                         #     temp_direction_answerr_noO.loc[row, direct] = answer_sheet.loc[i, direct] 
    #                         # else:
    #                         #     temp_direction_answerr_noO.loc[row, direct] = data_sheet.loc[row, direct]
                                
    #     sam_direction_answer_11 = pd.concat([sam_direction_answer_11, temp_direction_answer],
    #                                       ignore_index=True)
        # sam_direction_answer_noO = pd.concat([sam_direction_answer_noO, temp_direction_answerr_noO],
        #                                      ignore_index=True)

# %% 將檔案輸出成 EXCEL
save_file = r"D:\BenQ_Project\01_UR_lab\2024_05 耳機\答案統計_" + datetime.now().strftime('%m%d%H%M') + ".xlsx"

# save_file = r"E:\Hsin\BenQ\BQC電競耳機測試答案S1-S5\答案統計_" + datetime.now().strftime('%m%d%H%M') + ".xlsx"

with pd.ExcelWriter(save_file) as Writer:
    sam_direction_answer.to_excel(Writer, sheet_name="方向對-簡易", index=True)
    direction_answer.to_excel(Writer, sheet_name="方向對", index=True)
    direction_question.to_excel(Writer, sheet_name="方向對_出題數量", index=True)
    
    direction_ans_weight.to_excel(Writer, sheet_name="方向對(人數權重05)", index=True)
    direction_ans_weight_00.to_excel(Writer, sheet_name="方向對(人數權重)", index=True)
    
    number_answer.to_excel(Writer, sheet_name="人數錯", index=True)
    floor_answer.to_excel(Writer, sheet_name="樓層錯", index=True)
    answer_direction_noOO.to_excel(Writer, sheet_name="各方向出題數量(人數)", index=True)
    direction_answer_noOO.to_excel(Writer, sheet_name="各方向答對題數", index=True)
    direction_answer_noX.to_excel(Writer, sheet_name="方向_沒出現但有填", index=True)
    direction_answer_noXX.to_excel(Writer, sheet_name="方向_有出現但沒填", index=True)
    
   

# %% 統計分析        
# 創建統計表格
# number_statistic = pd.DataFrame(np.zeros((len(headset_name),
#                                           len(subject_info + direct_columns))),
#                                 columns=[subject_info + direct_columns])

# number_statistic = pd.DataFrame(np.zeros((len(headset_name),
#                                           len(subject_info + direct_columns))),
#                                 columns=[subject_info + direct_columns])
# # 依耳機做分組        
# headset_name = ["耳罩式耳機A", "耳罩式耳機B", "耳罩式耳機C", "耳罩式耳機D",
#                 "入耳式耳機1", "入耳式耳機2"]
# for headset in range(len(headset_name)):
#     temp_idx = []
#     for i, row in direction_answer.iterrows():
#         if row["耳機"] == headset_name[headset]:
#             temp_idx.append(i)
#     temp_data = direction_answer.loc[temp_idx, :]
#     for subject in direction_answer["受試者"].value_counts().index:
#         temp_data[temp_data["受試者"] == subject[0]].index[0]

#         print(subject[0])
        
#     number_mean = np.mean(direction_answer.loc[temp_idx, direct_columns], axis=0)
#     number_mean.loc[:, direct_columns]
#     number_statistic.loc[headset, direct_columns] = number_mean
#     number_statistic.loc[headset, "受試者"] = "direction_mean"
#     number_statistic.loc[headset, "耳機"] = headset_name[headset]
    
#     # filtered_rows = direction_answer.loc[direction_answer["耳機"] == headset]
#     # filtered_rows = direction_answer.loc[direction_answer.loc[:, "耳機"] == headset, :]

#     # cal_data = direction_answer.iloc[filtered_rows, :]
     
# for headset in headset_name:
#     filtered_rows = direction_answer.loc[direction_answer.loc[:, "耳機"] == headset]
#     print(f"耳機 {headset} 的資料筆數: {len(filtered_rows)}")



# print(direction_answer[direction_answer["耳機"] == "耳罩式耳機A"].index)























