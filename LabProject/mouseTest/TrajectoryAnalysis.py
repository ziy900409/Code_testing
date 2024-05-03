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
# 受試者
duplicate_info['Participant']
# 不同條件
duplicate_info['Condition']
# 第幾次測試
duplicate_info['Block']
# 不同難度
duplicate_info['Sequence']
# 每一次拖曳的軌跡
duplicate_info['Trial']
# 不同難度與每次拖曳軌跡的排列組合
all_combinations = []
# 外部循環迭代surrounding_circle_radius
for seq_info in duplicate_info['Sequence']:
    # 內部循環迭代target_amplitudes
    for trial_info in duplicate_info['Trial']:
        # 將此組合添加到所有組合列表中
        all_combinations.append({"Sequence": seq_info, "Trial": trial_info})
     
# to_x - centre x coordinate of target
# to_y - centre y coordinate of target
# 先找出 'Event' = 'EdgeCirclePos'，共13筆資料
edge_circle_pos = raw_data[raw_data['Event'] == 'EdgeCirclePos']
# from_x - x coordinate of beginning of trial
# from_y - y coordinate of beginning of trial
# 找出 'Event' = MOUSEBUTTONDOWN，共13筆資料
begin_mousedown = raw_data[raw_data['Event'] == 'MOUSEBUTTONDOWN']
# 判斷是否為成功的 task
# MOUSEBUTTONDOWN = MOUSEBUTTONUP_SUCC + MOUSEBUTTONUP_FAIL

















