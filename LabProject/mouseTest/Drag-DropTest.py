
"""
待處理功能：

1. 修改第幾次測試內容 從提交之後往後改
2. 做一個暫存檔，儲存資料夾路徑跟受試者資訊
3. 改成下拉式選單，或是不要英文字


------------

程式架構
1. 產生圓圈： 
    1.1. 建立一個圓圈類，每個圓圈物件代表一個拖曳任務目標。
    1.2. 在每個拖曳任務開始前，產生一個或多個這樣的圓圈物件並顯示在螢幕上。
2. 處理滑鼠拖曳事件： 
    2.1. 使用 pygame.MOUSEBUTTONDOWN 和 pygame.MOUSEMOTION 事件來處理滑鼠的拖曳操作。
    2.2. 當滑鼠按下並移動時，更新拖曳物件的位置。
    2.3. 當滑鼠按鍵釋放，當成一次事件結束，並重新更新圓圈位置
3. 偵測拖曳碰撞： 
    3.1. 在每次滑鼠移動時，偵測滑鼠是否與任何拖曳物件相交，
    3.2. 如果相交，則將目前拖曳物件標記為活動物件 (改成相交且滑鼠左鍵or中鍵按壓)。
    3.3. 當滑鼠釋放時，偵測是否有活動對象，並根據需要執行相應的操作（如移動、放置等）。
4. 計算操作時間與距離： 
    4.1. 在每次拖曳操作完成後，記錄拖曳操作的起始位置、結束位置以及所花費的時間。
    4.2. 根據 Fitts's Law 的公式，計算操作的 ID（Index of Difficulty）和各項指標（如 MT、TP）。
5. 顯示結果： 
    5.1. 將每次拖曳操作的結果（如操作時間、ID、TP 等）顯示在螢幕上，以便使用者查看。

----------

已解決
1. 拖曳順序問題 
    1.1. 完成各個trial後，出現下一個數字，數字應該出現在對角線
    1.2. trial 完成的判定為放開左鍵
2. 輸出檔案問題
    2.1. 游標位置紀錄應從第一次正確點擊開始，並且至最後一個trial完成後結束
3. 判定成功失敗方式：
    3.1. 原本：判定方式為只要中心圓與周圍圓的圓心距離小於周圍圓的半徑
    3.2. 改良後：中心圓的"圓周最遠距離"與周圍圓的圓心距離小於周圍圓的半徑
    distance = math.sqrt((surrounding_circles[completed_positions+1][0] - circle_x) ** 2 + \
                        (surrounding_circles[completed_positions+1][1] - circle_y) ** 2)
    # 改成只有進到有標記數字的圓圈才算成功
    if distance <= surrounding_circle_radius:
        completed_positions += 1
        mouse_click_events.append((Participant, Condition, Block,
                                (current_test_index + 1), (completed_positions+1), # sequemce, trial
                                current_test['target_amplitudes'], surrounding_circle_radius,
                                "MOUSEBUTTONUP_SUCC", event.pos, pygame.time.get_ticks()))
        find = 1
        break
4. 滑鼠位置採樣方式：
    4.1. 原本是滑鼠移動才採樣，現在改為"固定時間採樣"
        只有滑鼠移動才紀錄，但是可能會有採樣誤差
    elif event.type == pygame.MOUSEMOTION:
        # 保存鼠标移动轨迹
                mouse_trajectory.append((Participant, Condition, Block, 
                                     (current_test_index + 1), (completed_positions+1), # sequemce, trial
                                     current_test['target_amplitudes'], surrounding_circle_radius,
                                     event.pos, current_time))
        # 更新下一次採樣時間
        print((current_test['target_amplitudes'], surrounding_circle_radius,
               event.pos, current_time))
5. 獲得所有周圍圓的中心位置
    5.1. 利用 record edge 讓中心圓位置只被紀錄一次

6. 應有判定方式確認任務是成功還是失敗
   判斷是否為成功的 trail
   例如： MOUSEBUTTONDOWN = MOUSEBUTTONUP_SUCC + MOUSEBUTTONUP_FAIL

7. 設定不同難度問題
    7.1. 不同難度應該隨機出現

8. 自動產生檔案
    8.1. 是否可以自行輸入? UI?
    8.2 輸出檔案自動產生
    
9. 所有難度的測試都應該要從滑鼠按下去"中心圓"才開始

@author: Hsin.YH.Yang, written by May 02 2024
"""
# %% import library
import pygame
import sys
import math
import csv
import os
# import numpy as np
import random
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\mouseTest")
import Analysis_function_v1 as func
import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd

# from tkinter import messagebox
from datetime import datetime
# %% 判斷是否有暫存檔, 依照檔案中B01的數字來決定下一個數字
# 獲取當前程式所在的路徑
current_path = os.path.dirname(os.path.abspath(__file__))

temp_params = {}
if "DragDropTest_temp.txt" in os.listdir(current_path):
    print(0)
    temp_txt_path = current_path + "\\" + "DragDropTest_temp.txt"
    
    # 1. 如果當前路徑有 temp 檔案, 讀取檔案
    with open(temp_txt_path, 'r') as file:
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
            temp_params[key] = value
    temp_file_exist = os.path.exists(temp_params["folder_path"])
    if len(temp_params) > 0 and temp_file_exist: # 這個要再確認
        print(1)
        # 2. 利用上次的 temp 路徑找上次測驗是第幾個 task
        judge_B01 = func.Read_File(temp_params["folder_path"],
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
            (split_list['Subject'] == temp_params["user_id"]) &
            (split_list['Condition'] == temp_params["condition"])
            )
        condition_indices = split_list.index[condition_set].tolist()
        # 目標 table
        target_list = split_list.iloc[condition_indices, :]
        # 設定預設值
        org_user_id = temp_params["user_id"]
        org_condition = temp_params["condition"]
        org_folder_path = temp_params["folder_path"]
        if len(target_list) > 0:
            print(2)
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
    

# %% 設定 UI 介面來輸入受試者訊息

# 全局變量來存儲參數
params = {}

def submit():
    global params
    # user_id = entry_user_id.get()
    user_id = selected_user_id.get()
    condition = selected_condition.get()
    test_number = entry_test_number.get()
    width_range = entry_width_range.get()
    distance_range = entry_distance_range.get()
    folder_path = entry_folder_path.get()

    # 簡單的輸入檢查
    if not user_id or not condition or not test_number or not width_range or not distance_range or not folder_path:
        messagebox.showerror("輸入錯誤", "所有欄位都是必填的")
        return

    try:
        width_range = eval(width_range)
        distance_range = eval(distance_range)
    except:
        messagebox.showerror("輸入錯誤", "難度(寬)和難度(距離)必須是有效的列表")
        return

    if not (isinstance(width_range, list) and isinstance(distance_range, list) and
            len(width_range) == 2 and len(distance_range) == 2):
        messagebox.showerror("輸入錯誤", "難度(寬)和難度(距離)必須是包含兩個數字的列表")
        return

    # 保存輸入的值到全局變量
    params = {
        "user_id": user_id,
        "condition": condition,
        "test_number": test_number,
        "width_range": width_range,
        "distance_range": distance_range,
        "folder_path": folder_path
    }

    # 在這裡，你可以將輸入的值傳遞給你的主要程式邏輯
    print(f"User ID: {params['user_id']}")
    print(f"Condition: {params['condition']}")
    print(f"Test Number: {params['test_number']}")
    print(f"Width Range: {params['width_range']}")
    print(f"Distance Range: {params['distance_range']}")
    print(f"Folder Path: {params['folder_path']}")

    # 清空輸入欄位
    # entry_user_id.delete(0, tk.END)
    entry_test_number.delete(0, tk.END)
    entry_width_range.delete(0, tk.END)
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
selected_user_id.set(org_user_id)  # 設置預設值
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
selected_condition.set(org_condition)  # 設置預設值
condition_menu = tk.OptionMenu(root, selected_condition, *conditions)
condition_menu.config(font=font_label)
condition_menu.grid(row=1, column=1, pady=5)
condition_menu_widget = condition_menu.nametowidget(condition_menu.menuname)
condition_menu_widget.config(font=font_label)

tk.Label(root, text="第幾次測試:", font=font_label).grid(row=2, column=0, pady=5)
entry_test_number = tk.Entry(root, font=font_entry)
entry_test_number.grid(row=2, column=1, pady=5)
entry_test_number.insert(0, org_block)  # 設置預設文字

tk.Label(root, text="目標寬度:", font=font_label).grid(row=3, column=0, pady=5)
entry_width_range = tk.Entry(root, font=font_entry)
entry_width_range.grid(row=3, column=1, pady=5)
entry_width_range.insert(0, "[30, 40]")  # 設置預設文字

tk.Label(root, text="目標距離:", font=font_label).grid(row=4, column=0, pady=5)
entry_distance_range = tk.Entry(root, font=font_entry)
entry_distance_range.grid(row=4, column=1, pady=5)
entry_distance_range.insert(0, "[200, 400]")  # 設置預設文字

# 增加文件夾選擇
tk.Label(root, text="資料夾路徑:", font=font_label).grid(row=5, column=0, pady=5)
entry_folder_path = tk.Entry(root, font=font_entry)
entry_folder_path.grid(row=5, column=1, pady=5)
entry_folder_path.insert(0, org_folder_path) # 設置預設為當前路徑
select_folder_button = tk.Button(root, text="選擇資料夾", command=select_folder, font=font_label)
select_folder_button.grid(row=5, column=2, pady=5)

# 創建並排列提交按鈕
submit_button = tk.Button(root, text="提交", command=submit, font=font_label)
submit_button.grid(row=6, columnspan=3, pady=20)

# 開始主事件循環
root.mainloop()


# %% 儲存一個 .txt 的暫存檔
if "DragDropTest_temp.txt" in os.listdir(current_path) and \
    len(temp_params) > 0:
    print(0)
    file_path = params["folder_path"]
else:
    file_path = current_path
    
txt_file_name =  os.path.join(current_path, "DragDropTest_temp.txt")
with open(txt_file_name, 'w') as file:
    for key, value in params.items():
        file.write(f'{key}: {value}\n')
        
        
# %% 基礎參數設定及初始化
# ------------基本受測資料------------------------
Participant = params["user_id"]
Condition = params["condition"]
Block = params["test_number"]
file_name = params["user_id"] + "-" + params["condition"] + "-" + params["test_number"] \
    + "-" + datetime.now().strftime('%m%d%H%M') + ".csv"
# 設定輸出檔案儲存路徑
data_save_path = os.path.join(file_path, ("DragDropTask-" + file_name))
# -------------定义颜色--------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
CENTER_COLOR = (220, 190, 255)
PURPLE = (220, 190, 255)  # 淡紫色
# 定义圆的初始位置和大小
# circle_radius = 20
# 记录已经拖拽到的位置数量
completed_positions = 0
record_edge = 1
# 标记是否有活动圆圈
active_circle = False
circle_touched = False
# 保存鼠标軌跡和按鍵事件的列表
mouse_click_events = []
mouse_positions = []
edge_circle = []

# 設定採樣間隔（毫秒）
sampling_interval = 10  # 每10毫秒採樣一次
# 計時器初始化
next_sample_time = pygame.time.get_ticks() + sampling_interval
# 設定不同難度
# 定义周围圆的数量
num_surrounding_circles = 16
# 設定多個不同難度的測試, 保存所有可能的組合
all_combinations = []
for i in params["width_range"]:
    for ii in params["distance_range"]:
        all_combinations.append({"surrounding_circle_radius": i,
                                 'target_amplitudes': ii})
# 隨機所有難度測試
random.shuffle(all_combinations)
# 定義中心圓的半徑，為最小周圍圓的0.6倍
circle_radius = min(params["width_range"])*0.8
# 當前測試索引
current_test_index = 0
# 選擇第一個測試
current_test = all_combinations[current_test_index]
# %%
# 初始化 Pygame
pygame.init()

# 设置窗口大小和标题
infoObject = pygame.display.Info()
WINDOW_WIDTH = infoObject.current_w
WINDOW_HEIGHT = infoObject.current_h
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Drag and Drop to Finish Test')

# 定義中心點位置
center_x = WINDOW_WIDTH // 2
center_y = WINDOW_HEIGHT // 2

# 計算起始圓心位置

for i in range(len(all_combinations)):
    all_combinations[i]['circle_x'] = center_x + \
          all_combinations[i]['target_amplitudes'] * math.cos(0)
    all_combinations[i]['circle_y'] = center_y + \
        all_combinations[i]['target_amplitudes'] * math.sin(0)


circle_x = all_combinations[0]['circle_x'] 
circle_y = all_combinations[0]['circle_y'] 


# 游戏循环
while True:
    window.fill(WHITE)
    # 根據當前測試參數繪製周圍的圓
    surrounding_circle_radius = current_test["surrounding_circle_radius"]
    # 定義字體大小
    font = pygame.font.Font(None, 36)
    # 顯示 task 資訊
    # 顯示第幾個任務
    tests_info = font.render(f"Sequence {current_test_index + 1} of {len(all_combinations)}",
                              True, (0, 0, 0))
    tests_para = font.render(f"(A = {current_test['target_amplitudes']}, W = {surrounding_circle_radius})",
                               True, (0, 0, 0))
    window.blit(tests_info, (20, 20))
    window.blit(tests_para, (20, 60))
    # 獲取滑鼠位置
    mouse_pos = pygame.mouse.get_pos()
    # 在視窗上顯示文字信息
    # mouse_info = font.render(f"Mouse Position: ({mouse_pos[0]}, {mouse_pos[1]})", True, (0, 0, 0))
    # window.blit(mouse_info, (20, 100))
    # 計算出所有周圍圓圈的所在位置
    angle_step = math.radians(360 / num_surrounding_circles)
    surrounding_circles = []
    for i in range(num_surrounding_circles):
        if i % 2 == 0:
            angle_rad = int(i / 2) * angle_step
        else:
            angle_rad = int(i / 2) * angle_step + 3.14
        x = center_x + int(current_test['target_amplitudes'] * math.cos(angle_rad))
        y = center_y + int(current_test['target_amplitudes'] * math.sin(angle_rad))
        surrounding_circles.append((x, y))
    # 繪製周圍圓的外框，根据是否被碰到选择绘制的颜色
    for (x, y) in surrounding_circles:
        if circle_touched:
            pygame.draw.circle(window, BLACK, (x, y),
                               surrounding_circle_radius, width=2)
            pygame.draw.circle(window, PURPLE, (surrounding_circles[completed_positions+1]),
                               surrounding_circle_radius)
        else:
            pygame.draw.circle(window, BLACK, (x, y), surrounding_circle_radius, width=2)
    # 绘制圆心标记数字
    number_text = font.render(str(completed_positions+1), True, RED)
    if completed_positions >= 0:
        text_rect = number_text.get_rect(center=surrounding_circles[completed_positions+1])
        window.blit(number_text, text_rect)
    # 利用 record edge 讓中心圓位置只被紀錄一次
    if record_edge == completed_positions+1:
        record_edge += 1
        # 紀錄周圍圓的位置
        edge_circle.append((Participant, Condition, Block, # 受試者, 條件, 第幾次測試
                            (current_test_index + 1), (completed_positions+1), # sequemce, trial
                            current_test['target_amplitudes'], surrounding_circle_radius, # ampltude, width
                            "EdgeCirclePos", surrounding_circles[completed_positions+1], # event, edge circle position 
                            pygame.time.get_ticks())) # time
    
    # 繪製中心圓
    pygame.draw.circle(window, BLACK, (circle_x, circle_y), circle_radius)

    # 處理事件
    for event in pygame.event.get():
        current_time = pygame.time.get_ticks()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 檢測鼠標是否與中心圓相交
            distance = math.sqrt((circle_x  - event.pos[0]) ** 2 + (circle_y - event.pos[1]) ** 2)
            if distance <= circle_radius:
                active_circle = True
                mouse_click_events.append((Participant, Condition, Block, # 受試者, 條件, 第幾次測試
                                           (current_test_index + 1), (completed_positions+1), # sequemce, trial
                                           current_test['target_amplitudes'], surrounding_circle_radius, # ampltude, width
                                           "MOUSEBUTTONDOWN", event.pos, pygame.time.get_ticks())) # event, mouse position, time
        elif event.type == pygame.MOUSEBUTTONUP:
            if active_circle:
                active_circle = False
            distance = math.sqrt((circle_x - event.pos[0]) ** 2 + (circle_y - event.pos[1]) ** 2)
            # 如果按鍵釋放時，游標沒有碰到中心原，就不算事件的紀錄
            if distance > circle_radius:
                active_circle = False
                break
            # 檢測是否拖拽到了周圍的圓圈位置
            # 如果拖拽到，就 completed_positions+1
            find = 0
            # 定義指定周圍圓的圓心
            edge_cir_x = surrounding_circles[completed_positions+1][0]
            edge_cir_y = surrounding_circles[completed_positions+1][1]
            edge_point = func.edge_calculate(circle_x, circle_y, edge_cir_x, edge_cir_y, 
                                             circle_radius)
            # 計算周圍圓與中心圓的最遠位置的距離
            distance = math.sqrt((edge_point[0] - edge_cir_x) ** 2 + (edge_point[1] - edge_cir_y) ** 2)
            # 距離必須小於周圍圓的半徑
            if distance <= surrounding_circle_radius:
                completed_positions += 1
                mouse_click_events.append((Participant, Condition, Block,
                                            (current_test_index + 1), (completed_positions+1), # sequemce, trial
                                            current_test['target_amplitudes'], surrounding_circle_radius,
                                            "MOUSEBUTTONUP_SUCC", event.pos, pygame.time.get_ticks()))
                find = 1
                break
            # 如果找不到有在任何圓圈內，但是又偵測到MOUSEBUTTONUP，算失敗
            if find == 0:
                completed_positions += 1
                mouse_click_events.append((Participant, Condition, Block, 
                                           (current_test_index + 1), (completed_positions+1), # sequemce, trial
                                           current_test['target_amplitudes'], surrounding_circle_radius,
                                           "MOUSEBUTTONUP_FAIL", event.pos, pygame.time.get_ticks()))
                break
        elif event.type == pygame.MOUSEMOTION:
            if active_circle:
                # 检测是否拖拽到了周围的圆圈位置
                # 如果有拖曳到，就标记相应的外圆圈被碰到
                for i, (x, y) in enumerate(surrounding_circles):
                    # 定義指定周圍圓的圓心
                    edge_cir_x = surrounding_circles[completed_positions+1][0]
                    edge_cir_y = surrounding_circles[completed_positions+1][1]
                    edge_point = func.edge_calculate(circle_x, circle_y, edge_cir_x, edge_cir_y,
                                                     circle_radius)
                    # 計算周圍圓與中心圓的最遠位置的距離
                    distance = math.sqrt((edge_point[0] - edge_cir_x) ** 2 + (edge_point[1] - edge_cir_y) ** 2)
                    if distance <= surrounding_circle_radius:
                       circle_touched = True
                    else:
                        circle_touched = False
    
    # 固定時間採樣一次滑鼠位置，檢查是否到達採樣時間
    if pygame.time.get_ticks() >= next_sample_time:
        # 保存滑鼠位置
        mouse_pos = pygame.mouse.get_pos()
        # 更新下一次採樣時間
        next_sample_time = pygame.time.get_ticks() + sampling_interval
        # 保存鼠标移动轨迹
        mouse_positions.append((Participant, Condition, Block, 
                                (current_test_index + 1), (completed_positions+1), # sequemce, trial
                                  current_test['target_amplitudes'], surrounding_circle_radius,
                                  mouse_pos, current_time))

    # 如果完成了所有位置，切換到下一個測試
    if completed_positions+1 == num_surrounding_circles:
        # 重置完成的位置
        completed_positions = 0
        record_edge = 1
        circle_touched = False
        # 切換到下一個測試
        current_test_index += 1
        if current_test_index < len(all_combinations):
            current_test = all_combinations[current_test_index]
            circle_x = all_combinations[current_test_index]['circle_x']
            circle_y = all_combinations[current_test_index]['circle_y']
        else:
            # 將滑鼠軌跡和事件資料寫入 CSV 檔案
            with open(data_save_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Participant', 'Condition', 'Block',
                                  'Sequence', 'Trial',
                                  'Amplitudes', 'Width',
                                'Event', 'time', 'Pos_x', 'Pos_y'])
                # for part, cond, blo, sec, trial, amp, wid, pos, time in mouse_trajectory:
                #     writer.writerow([part, cond, blo, sec, trial, amp, wid, 'MOUSEMOTION', time, pos[0], pos[1]])
                for part, cond, blo, sec, trial, amp, wid, pos, time in mouse_positions:
                    writer.writerow([part, cond, blo, sec, trial, amp, wid, 'MOUSEPOS', time, pos[0], pos[1]])
                for part, cond, blo, sec, trial, amp, wid, event, pos, time in mouse_click_events:
                    writer.writerow([part, cond, blo, sec, trial, amp, wid, event, time, pos[0], pos[1]])
                for part, cond, blo, sec, trial, amp, wid, event, pos, time in edge_circle:
                    writer.writerow([part, cond, blo, sec, trial, amp, wid, event, time, pos[0], pos[1]])
            # 如果已經達到最後一個測試，則退出遊戲
            pygame.quit()
            sys.exit()

    # 如果有活动圆圈，根据鼠标位置更新圆圈位置
    if active_circle:
        circle_x, circle_y = pygame.mouse.get_pos()

    pygame.display.update()

