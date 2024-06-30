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

# import numpy as np
import pandas as pd
import os
# import math

import sys
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\mouseTest")
import Analysis_function_v1 as func
import jsonpickle # 將資料轉換成json格式
import tkinter as tk
from tkinter import messagebox, filedialog
from datetime import datetime


# %% 設定 UI 介面來設定欲處理的檔案路徑
# 全局變量來存儲參數
params = {}

def submit():
    global params
    folder_path = entry_folder_path.get()
    width_range = entry_width_range.get()

    # 簡單的輸入檢查
    if not folder_path or not width_range:
        messagebox.showerror("輸入錯誤", "所有欄位都是必填的")
        return



    # 保存輸入的值到全局變量
    params = {
        "folder_path": folder_path,
        "width_range": width_range,
    }

    # 在這裡，你可以將輸入的值傳遞給你的主要程式邏輯
    print(f"Folder Path: {params['folder_path']}")

    # 清空輸入欄位
    entry_width_range.delete(0, tk.END)
    entry_folder_path.delete(0, tk.END)
    
    # 關閉主窗口
    root.destroy()

def select_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry_folder_path.delete(0, tk.END)
        entry_folder_path.insert(0, folder_selected)

# 創建主窗口
root = tk.Tk()
root.title("資料夾路徑設置")

# 設置窗口大小
window_width = 480
window_height = 180
root.geometry(f'{window_width}x{window_height}')

# 使窗口居中
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height/2 - window_height/2)
position_right = int(screen_width/2 - window_width/2)
root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

# 設置文字大小
font_label = ('Helvetica', 14)
font_entry = ('Helvetica', 14)

tk.Label(root, text="使用者編號:", font=font_label).grid(row=0, column=0, pady=5)

# 增加文件夾選擇
tk.Label(root, text="資料夾路徑:", font=font_label).grid(row=0, column=0, pady=5)
entry_folder_path = tk.Entry(root, font=font_entry)
entry_folder_path.grid(row=0, column=1, pady=5)
entry_folder_path.insert(0, os.path.dirname(os.path.abspath(__file__))) # 設置預設為當前路徑
select_folder_button = tk.Button(root, text="選擇資料夾", command=select_folder, font=font_label)
select_folder_button.grid(row=0, column=2, pady=5)

tk.Label(root, text="目標寬度比例:", font=font_label).grid(row=1, column=0, pady=5)
entry_width_range = tk.Entry(root, font=font_entry)
entry_width_range.grid(row=1, column=1, pady=5)
entry_width_range.insert(0, "0.6")  # 設置預設文字

# 創建並排列提交按鈕
submit_button = tk.Button(root, text="提交", command=submit, font=font_label)
submit_button.grid(row=6, columnspan=3, pady=20)

# 開始主事件循環
root.mainloop()

# %% 獲得所有資料夾下的檔案路徑
file_list = func.Read_File(params["folder_path"],
                           ".csv")

# raw_data = pd.read_csv(r"C:\Users\Hsin.YH.Yang\Desktop\test\DragDropTask-S01-C01-B01-06251017.csv")
select_cir_radius_ratio = float(params["width_range"])

# 找到路徑下方所有檔案
# 只記錄包含 DragDropTask 的檔案路徑
DargDropFile_list = {"path": [],
                     "filename": [],
                     'Task': [],
                     'Subject': [],
                     'Condition': [],
                     'Block': [],
                     'time': []}
split_list = pd.DataFrame(columns = ['Task', 'Subject', 'Condition', 'Block', 'time'])
for i in range(len(file_list)):
    if "DragDropTask" in file_list[i]:
        print(file_list[i])
        filepath, tempfilename = os.path.split(file_list[i])
        filename, extension = os.path.splitext(tempfilename)
        split_list = pd.DataFrame([filename.split('-') ],
                                  columns=['Task', 'Subject', 'Condition', 'Block', 'time'])
        # 將資料儲存到表格
        DargDropFile_list["path"].append(file_list[i])
        DargDropFile_list["filename"].append(filename)
        DargDropFile_list["Task"].append(split_list['Task'][0])
        DargDropFile_list["Subject"].append(split_list['Subject'][0])
        DargDropFile_list["Condition"].append(split_list['Condition'][0])
        DargDropFile_list["Block"].append(split_list['Block'][0])
        DargDropFile_list["time"].append(split_list['time'][0])
        
# %% 批次處理資料 s1d、s3d
all_sd2_table = pd.DataFrame({},
                         columns = ["Participant", "Condition", "Block",
                                    "Trial", "A", "W", "Ae", "We", "IDe(bits)", "PT(ms)",
                                    "ST(ms)", "MT(ms)", "ER(%)", "TP(bps)", "TRE", "TAC", "MDC",
                                    "ODC", "MV", "ME", "MO"]
                         )

for idx in range(len(DargDropFile_list["path"])):
    print(DargDropFile_list["path"][idx])
    raw_data = pd.read_csv(DargDropFile_list["path"][idx])
    duplicate_info = {}

    # 找出每個欄位的重複值及其數量
    for column in raw_data.columns:
        duplicated_values = raw_data[column][raw_data[column].duplicated()]
        if not duplicated_values.empty:
            duplicate_counts = duplicated_values.value_counts().to_dict()
            duplicate_info[column] = duplicate_counts
    # 填入任務參數
    # 將info.keys轉成list的數值，並依照數值大小作排列
    trial_info = sorted([int(key) for key in duplicate_info['Trial'].keys()])
    trial_info.remove(16)
    amplitude_info = sorted([int(key) for key in duplicate_info["Amplitudes"].keys()])
    width_info = sorted([int(key) for key in duplicate_info["Width"].keys()])
    # 初始化資料格式
    sd1_table, sd2_table, sd3_data_format = func.initial_format()
    # 將資料處理成 sd3 format
    sd3_data = func.sd3_formating(duplicate_info, raw_data, sd3_data_format,
                                  amplitude_info, width_info, trial_info)
    # 繪製軌跡
    func.draw_tracjectory(sd3_data, amplitude_info, width_info,
                          params["folder_path"])
    # 計算 sd1 table 所需參數
    sd1_data = func.sd1_formating(sd1_table, sd3_data, select_cir_radius_ratio)
    # 計算 sd2 table 所需參數, 使用 sd1 table 做計算
    sd2_data = func.sd2_formating(sd1_table, sd2_table, duplicate_info)
    # # 將Pandas Series轉換為可序列化的列表
    func.convert_series_to_list(sd3_data_format)
    # 將資料結構轉換為 JSON 字串
    json_str = jsonpickle.encode(sd3_data_format)
    # 設定 JSON 資料儲存路徑
    json_path = DargDropFile_list["path"][idx].replace('.csv', '.json')
    # 將 JSON 字串寫入文件
    with open(json_path, 'w') as jsonfile:
        jsonfile.write(json_str)
    # 將資料寫入 EXCEL
    sd1_path = DargDropFile_list["path"][idx].replace('.csv', '_sd1.xlsx')
    sd1_data.to_excel(sd1_path, index=False)
    # 合併 sd2 table
    all_sd2_table = pd.concat([all_sd2_table, sd2_data],
                              ignore_index=True)
    
all_sd2_table.to_excel(str(params["folder_path"] + "\\table_sd2_" + datetime.now().strftime('%m%d%H%M') + ".xlsx"),
                       index=False)








# %%
# with open(r'C:\Users\Hsin.YH.Yang\Desktop\test\sd3_data_format.json', 'r') as jsonfile:
#     json_str = jsonfile.read()
    

# # 將JSON字串轉換為資料結構
# sd3_data_format_loaded = jsonpickle.decode(json_str)

# # 將列表轉換回Pandas Series
# func.convert_list_to_series(sd3_data_format_loaded)




























