import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import pygame
import os
import random
import time

# 初始化參數
params = {}
reaction_times = []
test_count = 0
max_tests = 3
Condition = 0
current_color = None
running = True
end_pressed = False  # 添加全局變數以追蹤是否按下結束按鈕

def submit():
    global params
    global reaction_times
    global test_count
    global max_tests
    global current_color
    global selected_user_id
    global selected_condition
    global entry_test_number
    global entry_width_range
    global entry_distance_range
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
        "test_number": test_number,
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
    save_reaction_times_to_excel()
    end_pressed = True  # 更新变量状态
    root.destroy()
    pygame.quit()

def show_input_dialog():
    global root
    global selected_user_id
    global selected_condition
    global entry_test_number
    global entry_folder_path
    
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
    user_ids = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12"]
    selected_user_id = tk.StringVar(root)
    selected_user_id.set(user_ids[0])
    user_id_menu = tk.OptionMenu(root, selected_user_id, *user_ids)
    user_id_menu.config(font=font_label)
    user_id_menu.grid(row=0, column=1, pady=5)
    user_id_menu_widget = user_id_menu.nametowidget(user_id_menu.menuname)
    user_id_menu_widget.config(font=font_label)

    tk.Label(root, text="使用條件:", font=font_label).grid(row=1, column=0, pady=5)
    conditions = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12"]
    selected_condition = tk.StringVar(root)
    selected_condition.set(conditions[0])
    condition_menu = tk.OptionMenu(root, selected_condition, *conditions)
    condition_menu.config(font=font_label)
    condition_menu.grid(row=1, column=1, pady=5)
    condition_menu_widget = condition_menu.nametowidget(condition_menu.menuname)
    condition_menu_widget.config(font=font_label)

    tk.Label(root, text="第幾次測試:", font=font_label).grid(row=2, column=0, pady=5)
    entry_test_number = tk.Entry(root, font=font_entry)
    entry_test_number.grid(row=2, column=1, pady=5)

    # 增加文件夾選擇
    tk.Label(root, text="資料夾路徑:", font=font_label).grid(row=3, column=0, pady=5)
    entry_folder_path = tk.Entry(root, font=font_entry)
    entry_folder_path.grid(row=3, column=1, pady=5)
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

def end_program():
    global end_pressed  # 声明全局变量
    save_reaction_times_to_excel()
    end_pressed = True  # 更新变量状态
    root.destroy()
    pygame.quit()

def show_input_dialog():
    global root
    global selected_user_id
    global selected_condition
    global entry_test_number
    global entry_folder_path
    
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
    user_ids = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12"]
    selected_user_id = tk.StringVar(root)
    selected_user_id.set(user_ids[0])
    user_id_menu = tk.OptionMenu(root, selected_user_id, *user_ids)
    user_id_menu.config(font=font_label)
    user_id_menu.grid(row=0, column=1, pady=5)
    user_id_menu_widget = user_id_menu.nametowidget(user_id_menu.menuname)
    user_id_menu_widget.config(font=font_label)

    tk.Label(root, text="使用條件:", font=font_label).grid(row=1, column=0, pady=5)
    conditions = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12"]
    selected_condition = tk.StringVar(root)
    selected_condition.set(conditions[0])
    condition_menu = tk.OptionMenu(root, selected_condition, *conditions)
    condition_menu.config(font=font_label)
    condition_menu.grid(row=1, column=1, pady=5)
    condition_menu_widget = condition_menu.nametowidget(condition_menu.menuname)
    condition_menu_widget.config(font=font_label)

    tk.Label(root, text="第幾次測試:", font=font_label).grid(row=2, column=0, pady=5)
    entry_test_number = tk.Entry(root, font=font_entry)
    entry_test_number.grid(row=2, column=1, pady=5)

    # 增加文件夾選擇
    tk.Label(root, text="資料夾路徑:", font=font_label).grid(row=3, column=0, pady=5)
    entry_folder_path = tk.Entry(root, font=font_entry)
    entry_folder_path.grid(row=3, column=1, pady=5)
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


def save_reaction_times_to_excel():
    if reaction_times:  # 檢查是否有反應時間記錄
        df = pd.DataFrame({'Test Number': range(1, len(reaction_times) + 1), 'Reaction Time (seconds)': reaction_times})
        df.to_excel(os.path.join(params['folder_path'], 'reaction_times.xlsx'), index=False)

def run_reaction_test():
    global reaction_times
    global test_count
    global max_tests
    global current_color
    global running
    
    # 初始化 Pygame
    pygame.init()

    # 設定畫面尺寸
    screen = pygame.display.set_mode((800, 600))
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
    while running:
        screen.fill(WHITE)
        
        # 畫出圓形
        pygame.draw.circle(screen, current_color, (400, 300), 100)
        
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
                screen.blit(text, (380, 250))
        
        pygame.display.flip()
        
        # 檢查事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if current_color == GREEN:
                    reaction_time = time.time() - change_time
                    reaction_times.append(reaction_time)
                    test_count += 1
                    current_color = GRAY
                    change_time = None
                    countdown_start = None
                    
                    # 顯示反應時間
                    screen.fill(WHITE)
                    text = font.render(f"Reaction time: {reaction_time:.3f} seconds", True, BLACK)
                    screen.blit(text, (100, 250))
                    pygame.display.flip()
                    pygame.time.wait(2000)
                    
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
    save_reaction_times_to_excel()

    # 結束 Pygame
    pygame.quit()
    
# 顯示輸入對話框，獲取初始參數
show_input_dialog()

# 運行反應時間測試
while running:
    run_reaction_test()
    if not end_pressed:  # 確認測試未被終止
        show_input_dialog()
