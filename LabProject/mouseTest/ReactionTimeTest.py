import tkinter as tk
from tkinter import messagebox, filedialog
# import pandas as pd
import csv
import pygame
import os
import sys
import random
import time
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\mouseTest")
sys.path.append("E:\Hsin\git\git\Code_testing\LabProject\mouseTest")
import Analysis_function_v1 as func
import UI_func_v1 as ui
from datetime import datetime
# %%

# 獲取當前程式所在的路徑
current_path = os.path.dirname(os.path.abspath(__file__))


# %%

# 初始化參數
params = {}
reaction_times = []
test_count = 0
max_tests = 3
Condition = 0
current_color = None
end_pressed = False  # 添加全局變數以追蹤是否按下結束按鈕
running = True

def submit():
    global params
    global reaction_times
    global test_count
    global max_tests
    global current_color
    global selected_user_id
    global selected_condition
    global entry_test_number
    # global entry_width_range
    # global entry_distance_range
    global entry_folder_path

    # 獲取輸入的值
    user_id = selected_user_id.get()
    condition = selected_condition.get()
    test_number = entry_test_number.get()
    folder_path = entry_folder_path.get()

    # 簡單的輸入檢查
    if not user_id or not condition or not test_number or not folder_path:
        messagebox.showerror("輸入錯誤", "所有欄位都是必填的")
        return

    # 保存輸入的值到全局變量
    params = {
        "user_id": user_id,
        "condition": condition,
        "block": test_number,
        "folder_path": folder_path
    }

    # 清空輸入欄位
    entry_test_number.delete(0, tk.END)
    entry_folder_path.delete(0, tk.END)
    
    # 關閉主窗口
    root.destroy()

def select_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry_folder_path.delete(0, tk.END)
        entry_folder_path.insert(0, folder_selected)

def end_program():
    global end_pressed  # 声明全局变量
    
    end_pressed = True  # 更新变量状态
    root.destroy()
    pygame.quit()

def show_input_dialog(org_info):
    global root
    global selected_user_id
    global selected_condition
    global entry_test_number
    global entry_folder_path
    
    root = tk.Tk()
    root.title("Reaction Time Task")
    
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
    user_ids = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12"]
    selected_user_id = tk.StringVar(root)
    selected_user_id.set(org_info["user_id"]) # 設置預設值
    user_id_menu = tk.OptionMenu(root, selected_user_id, *user_ids)
    user_id_menu.config(font=font_label)
    user_id_menu.grid(row=0, column=1, pady=5)
    user_id_menu_widget = user_id_menu.nametowidget(user_id_menu.menuname)
    user_id_menu_widget.config(font=font_label)

    tk.Label(root, text="使用條件:", font=font_label).grid(row=1, column=0, pady=5)
    conditions = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12"]
    selected_condition = tk.StringVar(root)
    selected_condition.set(org_info["condition"]) # 設置預設值
    condition_menu = tk.OptionMenu(root, selected_condition, *conditions)
    condition_menu.config(font=font_label)
    condition_menu.grid(row=1, column=1, pady=5)
    condition_menu_widget = condition_menu.nametowidget(condition_menu.menuname)
    condition_menu_widget.config(font=font_label)

    tk.Label(root, text="第幾次測試:", font=font_label).grid(row=2, column=0, pady=5)
    entry_test_number = tk.Entry(root, font=font_entry)
    entry_test_number.grid(row=2, column=1, pady=5)
    entry_test_number.insert(0, org_info["block"])  # 設置預設文字

    # 增加文件夾選擇
    tk.Label(root, text="資料夾路徑:", font=font_label).grid(row=3, column=0, pady=5)
    entry_folder_path = tk.Entry(root, font=font_entry)
    entry_folder_path.grid(row=3, column=1, pady=5)
    entry_folder_path.insert(0, org_info["folder_path"]) # 設置預設為當前路徑
    select_folder_button = tk.Button(root, text="選擇資料夾", command=select_folder, font=font_label)
    select_folder_button.grid(row=3, column=2, pady=5)

    # 創建並排列提交按鈕
    submit_button = tk.Button(root, text="提交", command=submit, font=font_label)
    submit_button.grid(row=4, columnspan=3, pady=20)

    # 創建並排列結束按鈕
    end_button = tk.Button(root, text="結束", command=end_program, font=font_label)
    end_button.grid(row=5, columnspan=3, pady=20)

    # 開始主事件循環
    root.mainloop()
    return end_pressed

# def save_reaction_times_to_excel(save_path):
#     if reaction_times:  # 檢查是否有反應時間記錄
#         df = pd.DataFrame({'Test Number': range(1, len(reaction_times) + 1), 'Reaction Time (seconds)': reaction_times})
#         df.to_excel(save_path, index=False)

def run_reaction_test(org_info):
    global reaction_times
    global test_count
    global max_tests
    global current_color
    global running
    
    # 初始化 Pygame
    pygame.init()
    infoObject = pygame.display.Info()

    screen_width = infoObject.current_w
    screen_height = infoObject.current_h

    # 設定畫面尺寸
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("反應時間測試")

    # 設定顏色
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    GRAY = (128, 128, 128)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    # 設定字體
    font = pygame.font.Font(None, 74)

    # 初始化變數
    reaction_times = []
    test_count = 0
    current_color = GRAY
    start_time = None
    change_time = None
    countdown_start = None

    # 主循環
    running = True
    while running:
        screen.fill(WHITE)
        
        tests_info = font.render(f"Subject ID: {org_info['user_id']}",
                                 True, (0, 0, 0))
        tests_para = font.render(f"Condition: {org_info['condition']}",
                                 True, (0, 0, 0))
        test_block = font.render(f"Block: {org_info['block']}",
                                 True, (0, 0, 0))
        screen.blit(tests_info, (20, 20))
        screen.blit(tests_para, (20, 60))
        screen.blit(test_block, (20, 100))
        
        # 畫出圓形
        pygame.draw.circle(screen, current_color,
                           (screen_width // 2, screen_height //2), 100)
        
        # 如果在倒數計時階段，顯示倒數計時
        if current_color == GRAY:
            if countdown_start is None:
                countdown_start = time.time()
            countdown = 3 - int(time.time() - countdown_start)
            if countdown <= 0:
                current_color = RED
                countdown_start = None
            else:
                text = font.render(str(countdown), True, BLACK)
                screen.blit(text, (screen_width // 2 - 20, screen_height //2 - 23))
        
        pygame.display.flip()
        
        # 檢查事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and (event.button == 6 or event.button == 7):
                if current_color == GREEN:
                    reaction_time = time.time() - change_time
                    if event.button == 7:
                        reaction_times.append((org_info["user_id"], org_info["condition"],
                                               org_info["block"], "forward",
                                               reaction_time))
                    elif event.button == 6:
                        reaction_times.append((org_info["user_id"], org_info["condition"],
                                               org_info["block"], "backward",
                                               reaction_time))
                    test_count += 1
                    current_color = GRAY
                    change_time = None
                    countdown_start = None
                    
                    # 顯示反應時間
                    screen.fill(WHITE)
                    text = font.render(f"Reaction time: {reaction_time:.3f} seconds", True, BLACK)
                    screen.blit(text, (screen_width // 2 - 350, screen_height //2 - 150))
                    pygame.display.flip()
                    pygame.time.wait(1500)
                    
                    if test_count >= max_tests:
                        running = False
                        break
        
        # 如果是紅燈，設定隨機的變色時間
        if current_color == RED and change_time is None:
            change_time = time.time() + random.uniform(1, 5)
        
        # 如果時間到了，變成綠燈
        if current_color == RED and time.time() >= change_time:
            current_color = GREEN
            change_time = time.time()

    # 將反應時間輸出到Excel
    # save_reaction_times_to_excel(save_path)

    # 結束 Pygame
    pygame.quit()
    return reaction_times

def temp_save(org_info, task = "ReactionTimeTask"):
    if str(task + "_temp.txt") in os.listdir(current_path) and \
        len(org_info) > 0:
        print(0)
        file_path = org_info["folder_path"]
    else:
        file_path = current_path
        
    txt_file_name =  os.path.join(file_path, str(task + "_temp.txt"))
    with open(txt_file_name, 'w') as file:
        for key, value in params.items():
            file.write(f'{key}: {value}\n')
    return file_path


# %%
"""
執行順序應該是
1. 獲得 temp file
2. 顯示對話視窗
3. 更新 temp file，並且更新受試者資訊
4. 執行反應測試
4.1. 紀錄反應時間
5. 重複 1~4，直到按下結束按鈕

"""

# 運行反應時間測試
# 顯示輸入對話框，獲取初始參數
# show_input_dialog(org_info)

# 運行反應時間測試
org_info = ui.find_temp_v2(task = "ReactionTimeTask")
all_reaction_time = []
n = 0
while not end_pressed:
    if n == 0:
        n = n + 1
    elif n != 0:
        # 找 temp 暫存檔
        org_info = ui.find_temp_v2(task = "ReactionTimeTask")
    show_input_dialog(org_info)
    file_path = temp_save(org_info, task = "ReactionTimeTask")
    # org_info = ui.find_temp_v2(task = "ReactionTimeTask")
    file_name = org_info["user_id"] + "-" + datetime.now().strftime('%m%d%H%M') + ".csv"
    data_save_path = os.path.join(org_info["folder_path"], ("ReactionTimeTask-" + file_name))
    if end_pressed:
        break
    indi_reaction_time = run_reaction_test(org_info)
    all_reaction_time.append(indi_reaction_time)
        
if len(all_reaction_time[0]) > 0:
    # 將反應時間和受試者資料寫入 CSV 檔案
    with open(data_save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Participant', 'Condition', 'Block', 'event', 'time'])
        for part, cond, blo, event, sec in all_reaction_time[0]:
            # print([part, cond, blo, sec])
            writer.writerow([part, cond, blo, event, sec])















