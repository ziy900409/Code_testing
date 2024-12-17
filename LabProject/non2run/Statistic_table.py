# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:57:00 2024

@author: Hsin.YH.Yang
"""

import os
import numpy as np
import pandas as pd
from itertools import product
from openpyxl import load_workbook, Workbook
from datetime import datetime

# %%
data_path = [r"D:/BenQ_Project/01_UR_lab/2024_11 Shanghai CS Major/4. Statistics/3. SpiderShot/All_Spider_Data.xlsx"]

file_sheet = ["arm_motion", "FingerIncludeAngle", "FingerAngle",
              "MedFreq", "iMVC_cal", "iMVCslope"]

subject = ["S01", "S02", "S03", "S04",
           "S05", "S06", "S07", "S08",
           "S09", "S10", "S11", "S12"]

mouse = ["A", "C", "EC2", "HS"]


# 使用 itertools.product 生成所有組合
combinations = list(product(subject, mouse))
# 获取当前日期和时间
now = datetime.now()
# 将日期转换为指定格式
formatted_date = now.strftime("%m-%d-%H%M")


# %%



for file in data_path:
    
    save_name = "All_Spider_Statistic_" + formatted_date
    save_name = data_path[0].replace("All_Spider_Data", save_name)
    
    if os.path.exists(save_name):
        workbook = load_workbook(save_name)
        print(f"檔案 {save_name} 已存在，成功打開。")
    else:
        workbook = Workbook()
        sheet = workbook.active
        workbook.save(save_name)
        
        workbook = load_workbook(save_name)
    
    for ind_sheet in file_sheet:
        data = pd.read_excel(data_path[0],
                             sheet_name = ind_sheet)
        # 生成一個資料儲存的地方
        pos_data = pd.DataFrame(np.zeros([len(combinations), len(data.columns)]),
                                 columns = data.columns)
        
        for idx in range(len(combinations)):
            filtered_df = data[(data['subject'] == combinations[idx][0]) & \
                               (data['mouse'] == combinations[idx][1])]
            if filtered_df.index.any():
                for column in filtered_df.columns:
                    # print(column)
                    if type(filtered_df.loc[filtered_df.index[0], column]) != str and \
                        type(filtered_df.loc[filtered_df.index[0], column]) != bool:
                        pos_data.loc[idx, column] = np.mean(filtered_df.loc[filtered_df.index, column])
                    else:
                        pos_data.loc[idx, column] = filtered_df.loc[filtered_df.index[0], column]
        # 將 pos_data 寫入新分頁
        with pd.ExcelWriter(save_name, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            pos_data.to_excel(writer, sheet_name=ind_sheet, index=False)
 
        
        
       
    

                    



















