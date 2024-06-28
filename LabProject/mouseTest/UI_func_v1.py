# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:49:48 2024

@author: Hsin.YH.Yang
"""
# %% import library

import os
import sys
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\mouseTest")
import Analysis_function_v1 as func
import pandas as pd

# %%

def open_txt(store_data, path):
    with open(path, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ', 1)
            # 處理數值範圍
            if key in ['width_range', 'distance_range']:
                value = value.strip('[]').split(', ')
                value = [float(v) for v in value]
            # 處理其他項目
            else:
                try:
                    value = int(value)
                except ValueError:
                    pass
            store_data[key] = value
    return store_data
# %%
current_path = os.path.dirname(os.path.abspath(__file__))

def find_temp(task = "DragDropTask"):
    
    temp_params = {}
    if str(task + "_temp.txt") in os.listdir(current_path):
        temp_txt_path = current_path + "\\" + str(task + "_temp.txt")
        
        # 1. 如果當前路徑有 temp 檔案, 讀取檔案

        store_data = open_txt(temp_params, temp_txt_path)
        for key in temp_params:
            if key =="folder_path":
                temp_file_exist = os.path.exists(str(store_data["folder_path"] + \
                                                     "\\" + str(task + "_temp.txt")))
            else:
                temp_file_exist = False
        if len(store_data) > 0 and temp_file_exist: # 這個要再確認
            temp_path = str(temp_params["folder_path"] + "\\" + str(task + "_temp.txt"))
            temp_params = {}
            store_data = open_txt(temp_params, temp_path)
            # 2. 利用上次的 temp 路徑找上次測驗是第幾個 task
            judge_B01 = func.Read_File(store_data["folder_path"],
                                       ".csv",
                                       subfolder=False)
            # 只記錄包含 DragDropTask 的檔案路徑
            DargDropFile_list = []
            for i in range(len(judge_B01)):
                if "DragDropTask" in judge_B01[i]:
                    filepath, tempfilename = os.path.split(judge_B01[i])
                    filename, extension = os.path.splitext(tempfilename)
                    DargDropFile_list.append(filename)
            # 將 list 用 "-" 分開
            split_list = pd.DataFrame([item.split('-') for item in DargDropFile_list],
                                      columns=['Task', 'Subject', 'Condition', 'Block', 'time'])
            # 找到現在在測試的受試者及條件
            condition_set = (
                (split_list['Subject'] == store_data["user_id"]) &
                (split_list['Condition'] == store_data["condition"])
                )
            condition_indices = split_list.index[condition_set].tolist()
            # 目標 table
            target_list = split_list.iloc[condition_indices, :]
            # 設定預設值
            org_user_id = store_data["user_id"]
            org_condition = store_data["condition"]
            org_folder_path = store_data["folder_path"]
            if len(target_list) > 0:
                # 將 Block 列轉換為數字（去掉 'B' 並轉換為整數）
                target_list['Block_numeric'] = target_list['Block'].str.extract('(\d+)').astype(int)
                # 找出 Block 的最大值
                if max(target_list["Block_numeric"]) < 9:    
                    max_block_numeric = "B0" + str(max(target_list["Block_numeric"]) + 1)
                else:
                    max_block_numeric = "B" + str(max(target_list["Block_numeric"]) + 1)
                # 設定 UI 預設文字
                org_block = max_block_numeric
            else:
                org_block = "B01"
        else:
            org_user_id = "S01"
            org_condition = "C01"
            org_block = "B01"
            org_folder_path = current_path
    else:
        org_user_id = "S01"
        org_condition = "C01"
        org_block = "B01"
        org_folder_path = current_path
    # 輸出資料
    org_info = {"user_id": org_user_id,
                "condition": org_condition,
                "block": org_block,
                "folder_path": org_folder_path}
    return org_info
    
# %%



























