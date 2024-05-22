"""
分析滑鼠移動軌跡

程式架構:
1. 先製作 trace file
https://www.yorku.ca/mack/FittsLawSoftware/doc/FittsTrace.html
2. sd2 Output File: sequence-by-sequence
https://www.yorku.ca/mack/FittsLawSoftware/doc/FittsTask.html

計算參數:
tracer file:
     participant code
     condition code
     block code
     sequence - sequence number
     A - target amplitude (diameter of the layout circle in pixels)
     W - target width (diameter of target circle in pixels)
     trial - trial number within the sequence
     from_x - x coordinate of beginning of trial
     from_y - y coordinate of beginning of trial
     to_x - centre x coordinate of target
     to_y - centre y coordinate of target
     identifier ("t=", "x=", or "y=")

@author: Hsin.YH.Yang, written by May 02 2024
"""

import numpy as np
import pandas as pd
import json

# %%
raw_data = pd.read_csv(r"D:\BenQ_Project\FittsDragDropTest\mouse_data.csv")

# 設定儲存格式

                

# %%
"""
purpose: 找出所有 trail by trail 的順序及路徑

1. 先利用 event 區分資料
2. 
"""
raw_data['Event'].duplicated().sum()

duplicate_info = {}

# 找出每個欄位的重複值及其數量
for column in raw_data.columns:
    duplicated_values = raw_data[column][raw_data[column].duplicated()]
    if not duplicated_values.empty:
        duplicate_counts = duplicated_values.value_counts().to_dict()
        duplicate_info[column] = duplicate_counts

print("每個欄位的重複值及其數量:")
print(duplicate_info)
duplicate_info['Event']

duplicate_info['Participant'].keys()

duplicate_info['Condition']

duplicate_info['Block']
# 不同難度
duplicate_info['Sequence']

duplicate_info['Trial']
duplicate_info["Amplitudes"]
duplicate_info["Width"]


# 創建 sd1 表格包含 Amplitudes * Width * Trial, 格式為 pandas.DataFrame
sd1_table = pd.DataFrame({},
                         columns = ["Participant", "Condition", "Session", "Group", "Block",
                                    "Trial", "A", "W", "Ae", "dx", "PT(ms)", "ST(ms)", "MT(ms)",
                                    "Errors", "TRE", "TAC", "MDC", "ODC", "MV", "ME", "MO"]
                         )
# 創建 sd2 表格包含 Amplitudes * Width
sd2_table = pd.DataFrame({},
                         columns = ["Participant", "Condition", "Session", "Group", "Block",
                                    "Trial", "ID", "A", "W", "Ae", "We", "IDe(bits)", "PT(ms)",
                                    "ST(ms)", "MT(ms)", "ER(%)", "TP(bps)", "TRE", "TAC", "MDC",
                                    "ODC", "MV", "ME", "MO"]
                         )
# 創建 sd3 表格包含 Amplitudes * Width * Trial, 格式為 Json
sd3_data_format = {
                  "info": {
                      "participants":[],
                      "condition": [],
                      "blocks": []
                      },
                  "task":{
                      "Sequence":[],
                      "A": [],
                      "W": [],
                      "Trial": [],
                      "from_x": [],
                      "from_y": [],
                      "to_x": [],
                      "to_y": []
                      },
                  "{t_x_y}":{
                      "t":[],
                      "x": [],
                      "y": []
                      }
                  }

# 填入任務參數
# 將info.keys轉成list的數值，並依照數值大小作排列
trial_info = sorted([int(key) for key in duplicate_info['Trial'].keys()])
trial_info.remove(14)
amplitude_info = sorted([int(key) for key in duplicate_info["Amplitudes"].keys()])
width_info = sorted([int(key) for key in duplicate_info["Width"].keys()])


trial_num = len(trial_info) * len(amplitude_info) * len(width_info)

for i in range(len(duplicate_info["Amplitudes"])):
    for ii in range(len(duplicate_info["Width"])):
        for iii in range(len(duplicate_info['Trial'])-1):
            idx = i*len(trial_info)*len(width_info) + ii*len(trial_info) + iii 
            print(i, ii, idx)
            # 將任務條件輸入至表格
            sd1_table.loc[idx, "Participant"] = list(duplicate_info['Participant'].keys())[0] # 受試者
            sd1_table.loc[idx, "Condition"] = list(duplicate_info['Condition'].keys())[0] # 不同條件
            sd1_table.loc[idx, "Block"] = list(duplicate_info['Block'].keys())[0] # 第幾次測試
            sd1_table.loc[idx, "A"] = amplitude_info[i]
            sd1_table.loc[idx, "W"] = width_info[ii]
            sd1_table.loc[idx, "Trial"] = trial_info[iii]

# 找出每筆 trial, 數量為 Amplitudes * Width * Trial, 並依照 time 作排列
for i in range(len(duplicate_info["Amplitudes"])):
    for ii in range(len(duplicate_info["Width"])):
        for iii in range(len(duplicate_info['Trial'])-1):
            print(amplitude_info[i], width_info[ii], trial_info[iii])
            condition = (
                        (raw_data['Event'] == 'MOUSEPOS') &
                        (raw_data['Amplitudes'] == amplitude_info[i]) &
                        (raw_data['Width'] == width_info[ii]) &
                        (raw_data['Trial'] == trial_info[iii])
                        )
            matched_indices = raw_data.index[condition].tolist()
            # 找出目標物的圓心
            target_cond = (
                            (raw_data['Event'] == 'EdgeCirclePos') &
                            (raw_data['Amplitudes'] == amplitude_info[i]) &
                            (raw_data['Width'] == width_info[ii]) &
                            (raw_data['Trial'] == trial_info[iii])
                            )
            target_indices = raw_data.index[target_cond].tolist()
            # 更新info部分
            sd3_data_format["info"]["participants"].append(raw_data["Participant"][0])
            sd3_data_format["info"]["condition"].append(raw_data["Condition"][0])
            sd3_data_format["info"]["blocks"].append(raw_data["Block"][0])

            # 更新task部分
            sd3_data_format["task"]["A"].append(raw_data["Amplitudes"][matched_indices[0]])
            sd3_data_format["task"]["W"].append(raw_data["Width"][matched_indices[0]])
            sd3_data_format["task"]["Trial"].append(raw_data["Trial"][matched_indices[0]])
            sd3_data_format["task"]["Sequence"].append(raw_data["Sequence"][matched_indices[0]])
            sd3_data_format["task"]["from_x"].append(raw_data["Pos_x"][matched_indices[-1]])
            sd3_data_format["task"]["from_y"].append(raw_data["Pos_y"][matched_indices[-1]])
            sd3_data_format["task"]["to_x"].append(raw_data["Pos_x"][target_indices[0]])
            sd3_data_format["task"]["to_y"].append(raw_data["Pos_y"][target_indices[0]])
            
            # 更新 {t_x_y} 部分, 每一次拖曳的軌跡
            sd3_data_format["{t_x_y}"]["t"].append(raw_data["time"][matched_indices])
            sd3_data_format["{t_x_y}"]["x"].append(raw_data["Pos_x"][matched_indices])
            sd3_data_format["{t_x_y}"]["y"].append(raw_data["Pos_y"][matched_indices])

# %% 計算資料
"""
pygame.time.get_ticks() 單位為毫秒
"""
# Target Re-entry (TRE). 
# If the pointer enters the target region, leaves, 
# then re-enters the target region, then target re-entry (TRE ) occurs. 
# If this behaviour is recorded twice in a sequence of ten trials, TRE is reported as 0.2 per trial

# 計算 sd2 table 所需參數
# 表格欄位名稱 xFrom, yFrom, xTo, yTo, xSelect, ySelect, MT
sd2_calcu = pd.DataFrame({},
                         columns=["xFrom", "yFrom", "xTo", "yTo",
                                  "xSelect", "ySelect", "MT"])

# 1. Throughput 
# mt: The movement times (ms) for each trial
mt = sd3_data_format["{t_x_y}"]["t"][0].values[-1] - sd3_data_format["{t_x_y}"]["t"][0].values[0]
amplitude = sd3_data_format["task"]["A"]
width = sd3_data_format["task"]["W"]
# mouse_from: The specified starting coordinates for each trial (center of the "from" target)
mouse_from = [sd3_data_format["task"]["from_x"][0], sd3_data_format["task"]["from_y"][0]]
# mouse_to: The specified ending coordinates for each trial (center of the "to" target)
moouse_to = [sd3_data_format["{t_x_y}"]["x"][0].values[-1], sd3_data_format["{t_x_y}"]["y"][0].values[-1]]
# mouse_select: The coordinates of selection where each trial was terminated
mouse_select = [sd3_data_format["task"]["to_x"][0], sd3_data_format["task"]["to_y"][0]]

     



















