# # 創建並排列各個 Label 和 Entry 小部件
# tk.Label(root, text="使用者編號:").grid(row=0, column=0)
# entry_user_id = tk.Entry(root)
# entry_user_id.grid(row=0, column=1)
# entry_user_id.insert(0, "S01")  # 設置預設文字

# tk.Label(root, text="使用條件:").grid(row=1, column=0)
# entry_condition = tk.Entry(root)
# entry_condition.grid(row=1, column=1)
# entry_condition.insert(0, "C01")  # 設置預設文字

# tk.Label(root, text="第幾次測試:").grid(row=2, column=0)
# entry_test_number = tk.Entry(root)
# entry_test_number.grid(row=2, column=1)
# entry_test_number.insert(0, "1")  # 設置預設文字

# tk.Label(root, text="難度(寬):").grid(row=3, column=0)
# entry_width_range = tk.Entry(root)
# entry_width_range.grid(row=3, column=1)
# entry_width_range.insert(0, "[20, 30]")  # 設置預設文字

# tk.Label(root, text="難度(距離):").grid(row=4, column=0)
# entry_distance_range = tk.Entry(root)
# entry_distance_range.grid(row=4, column=1)
# entry_distance_range.insert(0, "[200, 400]")  # 設置預設文字


# %%
import tkinter as tk
from tkinter import messagebox

# 全局變量來存儲參數
params = {}

def submit():
    global params
    user_id = entry_user_id.get()
    condition = entry_condition.get()
    test_number = entry_test_number.get()
    width_range = entry_width_range.get()
    distance_range = entry_distance_range.get()

    # 簡單的輸入檢查
    if not user_id or not condition or not test_number or not width_range or not distance_range:
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
        "distance_range": distance_range
    }

    # 在這裡，你可以將輸入的值傳遞給你的主要程式邏輯
    print(f"User ID: {params['user_id']}")
    print(f"Condition: {params['condition']}")
    print(f"Test Number: {params['test_number']}")
    print(f"Width Range: {params['width_range']}")
    print(f"Distance Range: {params['distance_range']}")

    # 清空輸入欄位
    entry_user_id.delete(0, tk.END)
    entry_condition.delete(0, tk.END)
    entry_test_number.delete(0, tk.END)
    entry_width_range.delete(0, tk.END)
    entry_distance_range.delete(0, tk.END)
    
    # 關閉主窗口
    root.destroy()

# 創建主窗口
root = tk.Tk()
root.title("參數輸入")

# 設置窗口大小
window_width = 400
window_height = 300
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

# 創建並排列各個 Label 和 Entry 小部件
tk.Label(root, text="使用者編號:", font=font_label).grid(row=0, column=0, pady=5)
entry_user_id = tk.Entry(root, font=font_entry)
entry_user_id.grid(row=0, column=1, pady=5)
entry_user_id.insert(0, "S01")  # 設置預設文字

tk.Label(root, text="使用條件:", font=font_label).grid(row=1, column=0, pady=5)
entry_condition = tk.Entry(root, font=font_entry)
entry_condition.grid(row=1, column=1, pady=5)
entry_condition.insert(0, "C01")  # 設置預設文字

tk.Label(root, text="第幾次測試:", font=font_label).grid(row=2, column=0, pady=5)
entry_test_number = tk.Entry(root, font=font_entry)
entry_test_number.grid(row=2, column=1, pady=5)
entry_test_number.insert(0, "1")  # 設置預設文字

tk.Label(root, text="難度(寬):", font=font_label).grid(row=3, column=0, pady=5)
entry_width_range = tk.Entry(root, font=font_entry)
entry_width_range.grid(row=3, column=1, pady=5)
entry_width_range.insert(0, "[20, 30]")  # 設置預設文字

tk.Label(root, text="難度(距離):", font=font_label).grid(row=4, column=0, pady=5)
entry_distance_range = tk.Entry(root, font=font_entry)
entry_distance_range.grid(row=4, column=1, pady=5)
entry_distance_range.insert(0, "[200, 400]")  # 設置預設文字

# 創建並排列提交按鈕
submit_button = tk.Button(root, text="提交", command=submit, font=font_label)
submit_button.grid(row=5, columnspan=2, pady=20)

# 開始主事件循環
root.mainloop()

# 假設你在這裡需要使用提交的參數，可以這樣調用
print("已提交的參數：")
print(params)


