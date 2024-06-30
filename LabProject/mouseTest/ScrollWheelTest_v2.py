"""
待解決問題

1. 需要標準化不同滾輪的速度

程式架構

目標圓圈大小不會變化

@author: Hsin.YH.Yang, written by June 12 2024
"""

# %%
import pygame
import time
import random
import csv
import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import os
import sys
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\mouseTest")
import Analysis_function_v1 as func
import UI_func_v1 as ui
from datetime import datetime
# %% 判斷是否有暫存檔, 依照檔案中B01的數字來決定下一個數字
# 獲取當前程式所在的路徑
current_path = os.path.dirname(os.path.abspath(__file__))

org_info = ui.find_temp(task = "ScrollWheelTask")
# temp_params = {}
# if "ScrollWheelTask_temp.txt" in os.listdir(current_path):
#     print(0)
#     temp_txt_path = current_path + "\\" + "ScrollWheelTask_temp.txt"
    
#     # 1. 如果當前路徑有 temp 檔案, 讀取檔案
#     with open(temp_txt_path, 'r') as file:
#         for line in file:
#             key, value = line.strip().split(': ', 1)
#             # 處理數值範圍
#             if key in ['circle_radius', 'target_y']:
#                 value = value.strip('[]').split(', ')
#                 value = [float(v) for v in value]
#             # 處理其他項目
#             else:
#                 try:
#                     value = int(value)
#                 except ValueError:
#                     pass
#             temp_params[key] = value
#     temp_file_exist = os.path.exists(temp_params["folder_path"])
#     if len(temp_params) > 0 and temp_file_exist: # 這個要再確認
#         print(1)
#         # 2. 利用上次的 temp 路徑找上次測驗是第幾個 task
#         judge_B01 = func.Read_File(temp_params["folder_path"],
#                                    ".csv",
#                                    subfolder=False)
#         # 只記錄包含 DragDropTask 的檔案路徑
#         DargDropFile_list = []
#         for i in range(len(judge_B01)):
#             if "DragDropTask" in judge_B01[i]:
#                 filepath, tempfilename = os.path.split(judge_B01[i])
#                 filename, extension = os.path.splitext(tempfilename)
#                 DargDropFile_list.append(filename)
#         # 將 list 用 "-" 分開
#         split_list = pd.DataFrame([item.split('-') for item in DargDropFile_list],
#                                   columns=['Task', 'Subject', 'Condition', 'Block', 'time'])
#         # 找到現在在測試的受試者及條件
#         condition_set = (
#             (split_list['Subject'] == temp_params["user_id"]) &
#             (split_list['Condition'] == temp_params["condition"])
#             )
#         condition_indices = split_list.index[condition_set].tolist()
#         # 目標 table
#         target_list = split_list.iloc[condition_indices, :]
#         # 設定預設值
#         org_user_id = temp_params["user_id"]
#         org_condition = temp_params["condition"]
#         org_folder_path = temp_params["folder_path"]
#         if len(target_list) > 0:
#             print(2)
#             # 將 Block 列轉換為數字（去掉 'B' 並轉換為整數）
#             target_list['Block_numeric'] = target_list['Block'].str.extract('(\d+)').astype(int)
#             # 找出 Block 的最大值
#             if max(target_list["Block_numeric"]) < 9:    
#                 print(3)
#                 max_block_numeric = "B0" + str(max(target_list["Block_numeric"]) + 1)
#             else:
#                 max_block_numeric = "B" + str(max(target_list["Block_numeric"]) + 1)
#             # 設定 UI 預設文字
#             org_block = max_block_numeric
#         else:
#             org_block = "B01"
#     else:
#         org_user_id = "S01"
#         org_condition = "C01"
#         org_block = "B01"
#         org_folder_path = current_path
# else:
#     org_user_id = "S01"
#     org_condition = "C01"
#     org_block = "B01"
#     org_folder_path = current_path
# print(org_block)

# %% 設定 UI 介面來輸入受試者訊息

# 全局變量來存儲參數
params = {}

def submit():
    global params
    # user_id = entry_user_id.get()
    user_id = selected_user_id.get()
    condition = selected_condition.get()
    test_number = entry_test_number.get()
    # width_range = entry_width_range.get()
    distance_range = entry_distance_range.get()
    folder_path = entry_folder_path.get()

    # 簡單的輸入檢查
    if not user_id or not condition or not test_number or not distance_range or not folder_path:
        messagebox.showerror("輸入錯誤", "所有欄位都是必填的")
        return

    try:
        # width_range = eval(width_range)
        distance_range = eval(distance_range)
    except:
        messagebox.showerror("輸入錯誤", "難度(寬)和難度(距離)必須是有效的列表")
        return

    if not  (isinstance(distance_range, list) and len(distance_range) >= 2):
        messagebox.showerror("輸入錯誤", "難度(距離)必須是包含或以上的兩個數字列表")
        return

    # 保存輸入的值到全局變量
    params = {
        "user_id": user_id,
        "condition": condition,
        "test_number": test_number,
        # "circle_radius": width_range,
        "target_y": distance_range,
        "folder_path": folder_path
    }

    # 在這裡，你可以將輸入的值傳遞給你的主要程式邏輯
    print(f"User ID: {params['user_id']}")
    print(f"Condition: {params['condition']}")
    print(f"Test Number: {params['test_number']}")
    # print(f"Width Range: {params['circle_radius']}")
    print(f"Distance Range: {params['target_y']}")
    print(f"Folder Path: {params['folder_path']}")

    # 清空輸入欄位
    # entry_user_id.delete(0, tk.END)
    entry_test_number.delete(0, tk.END)
    # entry_width_range.delete(0, tk.END)
    entry_distance_range.delete(0, tk.END)
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
root.title("參數輸入")

# 設置窗口大小
window_width = 480
window_height = 360
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

# 使用條件下拉選單選項
user_ids = ["S01", "S02", "S03", "S04", "S05", "S06",
            "S07", "S08", "S09", "S10", "S11", "S12"]
selected_user_id = tk.StringVar(root)
selected_user_id.set(org_info["user_id"])  # 設置預設值
user_id_menu = tk.OptionMenu(root, selected_user_id, *user_ids)
user_id_menu.config(font=font_label)
user_id_menu.grid(row=0, column=1, pady=5)
user_id_menu_widget = user_id_menu.nametowidget(user_id_menu.menuname)
user_id_menu_widget.config(font=font_label)

# 使用條件下拉選單選項
tk.Label(root, text="使用條件:", font=font_label).grid(row=1, column=0, pady=5)
conditions =  ["C01", "C02", "C03", "C04", "C05", "C06",
               "C07", "C08", "C09", "C10", "C11", "C12"]
selected_condition = tk.StringVar(root)
selected_condition.set(org_info["condition"])  # 設置預設值
condition_menu = tk.OptionMenu(root, selected_condition, *conditions)
condition_menu.config(font=font_label)
condition_menu.grid(row=1, column=1, pady=5)
condition_menu_widget = condition_menu.nametowidget(condition_menu.menuname)
condition_menu_widget.config(font=font_label)

tk.Label(root, text="第幾次測試:", font=font_label).grid(row=2, column=0, pady=5)
entry_test_number = tk.Entry(root, font=font_entry)
entry_test_number.grid(row=2, column=1, pady=5)
entry_test_number.insert(0, org_info["block"])  # 設置預設文字

# tk.Label(root, text="目標寬度:", font=font_label).grid(row=3, column=0, pady=5)
# entry_width_range = tk.Entry(root, font=font_entry)
# entry_width_range.grid(row=3, column=1, pady=5)
# entry_width_range.insert(0, "[50, 100]")  # 設置預設文字

tk.Label(root, text="目標距離:", font=font_label).grid(row=3, column=0, pady=5)
entry_distance_range = tk.Entry(root, font=font_entry)
entry_distance_range.grid(row=3, column=1, pady=5)
entry_distance_range.insert(0, "[500, 1000, 3000]")  # 設置預設文字

# 增加文件夾選擇
tk.Label(root, text="資料夾路徑:", font=font_label).grid(row=4, column=0, pady=5)
entry_folder_path = tk.Entry(root, font=font_entry)
entry_folder_path.grid(row=4, column=1, pady=5)
entry_folder_path.insert(0, org_info["folder_path"]) # 設置預設為當前路徑
select_folder_button = tk.Button(root, text="選擇資料夾", command=select_folder, font=font_label)
select_folder_button.grid(row=4, column=2, pady=5)

# 創建並排列提交按鈕
submit_button = tk.Button(root, text="提交", command=submit, font=font_label)
submit_button.grid(row=5, columnspan=3, pady=20)

# 開始主事件循環
root.mainloop()


# %% 儲存一個 .txt 的暫存檔
if "ScrollWheelTask_temp.txt" in os.listdir(current_path) and \
    len(org_info) > 0:
    # print(0)
    file_path = params["folder_path"]
else:
    file_path = current_path
    
txt_file_name =  os.path.join(file_path, "ScrollWheelTask_temp.txt")
with open(txt_file_name, 'w') as file:
    for key, value in params.items():
        file.write(f'{key}: {value}\n')
        
# %%
# 初始化 Pygame
pygame.init()
# 定义不同难度级别的圆圈半径和目标物初始位置
# 记录滚轮输入的数据
wheel_data = []
# 记录目标物的位置
target_positions = []
# 紀錄滑鼠按鍵時間
mouse_click_events = []
# ------------基本受測資料------------------------
Participant = params["user_id"]
Condition = params["condition"]
Block = params["test_number"]
file_name = params["user_id"] + "-" + params["condition"] + "-" + params["test_number"] \
    + "-" + datetime.now().strftime('%m%d%H%M') + ".csv"
# 設定輸出檔案儲存路徑
data_save_path = os.path.join(file_path, ("ScrollWheelTask-" + file_name))
# 定义一些颜色
WHITE = (255, 255, 255)
RED = (213, 89, 111)
GREEN = (0, 255, 0)
BLUE = (129, 175, 196)
YELLOW = (243, 221, 153)

# 屏幕尺寸
infoObject = pygame.display.Info()

screen_width = infoObject.current_w
screen_height = infoObject.current_h

# 定义目标物的属性
target_radius = 25
target_x = screen_width // 2
target_y = screen_height 

# 定义圆圈的位置和大小（初始难度）
circle_x = screen_width // 2
circle_y = screen_height // 2


all_combinations = []
# for i in params["circle_radius"]:
for ii in params["target_y"]:
    # print(screen_height, ii, screen_height - ii)
    all_combinations.append({'circle_radius': target_radius*1.6,
                             'target_y': screen_height // 2 + ii})
    all_combinations.append({'circle_radius': target_radius*1.6,
                             'target_y': screen_height // 2 - ii})

random.shuffle(all_combinations)
current_level = 0
circle_radius = all_combinations[current_level]["circle_radius"]
target_y = all_combinations[current_level]["target_y"]

#%%

# 创建屏幕对象
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Mouse Wheel Task")

# 游戏主循环标志
running = True

# 上次滚动事件的时间
last_time = time.time()
# 
# 游戏主循环
while running:
    # 绘制背景
    screen.fill(WHITE)
    ButtonDown = False
    # 定義字體大小
    font = pygame.font.Font(None, 36)

    # 事件处理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEWHEEL:
            # 计算当前时间
            current_time = time.time()
            # 计算时间差
            time_diff = current_time - last_time
            last_time = current_time
            # 更新目标物的位置
            target_y -= event.y * 10  # 调整目标物移动速度，需再確認！！
            # 记录滚轮的输入数据（像素/秒）
            wheel_data.append((Participant, Condition, Block, # 受試者, 條件, 第幾次測試
                               (current_level + 1), # sequemce
                               circle_y, all_combinations[current_level]["target_y"], # ampltude, width
                               "WHEELDATA", target_y, pygame.time.get_ticks()))
            # 紀錄如果圓圈剛好到目標圓圈的正中心
            if (target_x, target_y) == (circle_x, circle_y):
                wheel_data.append((Participant, Condition, Block, # 受試者, 條件, 第幾次測試
                                   (current_level + 1), # sequemce
                                   circle_y, all_combinations[current_level]["target_y"], # ampltude, width
                                   "WHEELDATA_TOUCH", target_y, pygame.time.get_ticks()))
            # # 记录目标物的位置
            # target_positions.append((target_x, target_y))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 or event.button == 3:
                ButtonDown = True
            # 检查目标物是否在圆圈内
            edge_point = func.edge_calculate(target_x, target_y, circle_x, circle_y,  
                                             target_radius)
            distance = ((edge_point[0] - circle_x) ** 2 + (edge_point[1] - circle_y) ** 2) ** 0.5
            # distance = ((target_x - circle_x) ** 2 + (target_y - circle_y) ** 2) ** 0.5
            if distance <= circle_radius and ButtonDown:
                mouse_click_events.append((Participant, Condition, Block, # 受試者, 條件, 第幾次測試
                                           (current_level + 1), # sequemce
                                           circle_y, all_combinations[current_level]["target_y"], # ampltude, width
                                           "MOUSEBUTTONDOWN", target_y, pygame.time.get_ticks()))
                
                current_level += 1
                if current_level < len(all_combinations):
                    # event, mouse position, time
                    circle_radius = all_combinations[current_level]["circle_radius"]
                    target_y = all_combinations[current_level]["target_y"]
                    
                else:
                    with open(data_save_path, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Participant', 'Condition', 'Block',
                                         'Sequence',
                                         'circle_x', 'Distance',
                                         'Event', 'Pos_y', 'time'])
                        for part, cond, blo, sec, amp, wid, event, t, pos in wheel_data:
                            writer.writerow([part, cond, blo, sec, amp, wid, event, t, pos])
                        for part, cond, blo, sec, amp, wid, event, t , pos in mouse_click_events:
                            writer.writerow([part, cond, blo, sec, amp, wid, event, t, pos])
                    pygame.quit()
                    sys.exit()

    # 绘制背景
    screen.fill(WHITE)
    
    # 检查目标物是否在圆圈内并设置圆圈颜色
    edge_point = func.edge_calculate(target_x, target_y, circle_x, circle_y,  
                                     target_radius)
    
    distance = ((edge_point[0] - circle_x) ** 2 + (edge_point[1] - circle_y) ** 2) ** 0.5
    if distance <= circle_radius:
        circle_color = RED
        inside_circle = True
    else:
        circle_color = BLUE
        inside_circle = False
    
    # 绘制圆圈
    pygame.draw.circle(screen, circle_color, (circle_x, circle_y), circle_radius, 4)
    
    # 绘制目标物
    pygame.draw.circle(screen, RED, (target_x, target_y), target_radius)
    
    # 绘制目标物与圆圈中心的连线
    pygame.draw.line(screen, YELLOW, (target_x, target_y), (circle_x, circle_y), 4)

    # 顯示 task 資訊
    # 顯示第幾個任務
    tests_info = font.render(f"Sequence {current_level + 1} of {len(all_combinations)}",
                             True, (0, 0, 0))
    # tests_para = font.render(f"(Target Distance = {all_combinations[current_level]['target_y'] - screen_height // 2}, Target Y pos = {target_y})",
    #                          True, (0, 0, 0))
    tests_para = font.render(f"Target Distance = {all_combinations[current_level]['target_y'] - screen_height // 2}",
                             True, (0, 0, 0))
    screen.blit(tests_info, (20, 20))
    screen.blit(tests_para, (20, 60))
    
    # 更新屏幕
    pygame.display.flip()

        


