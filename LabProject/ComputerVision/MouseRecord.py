import tkinter as tk
from tkinter import filedialog, messagebox
from pynput import mouse, keyboard
import threading
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime

# 滑鼠座標和按鍵紀錄
mouse_positions = []
mouse_clicks = []
# 滑鼠記錄狀態
recording = False
# 儲存的資料夾路徑
save_folder = ""

def select_folder():
    """選擇儲存資料的資料夾"""
    global save_folder
    folder = filedialog.askdirectory()
    if folder:
        save_folder = folder
        folder_label.config(text=f"Save Folder: {save_folder}")
    else:
        messagebox.showwarning("Warning", "No folder selected!")

def generate_filename():
    """根據時間戳自動生成檔案名稱"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"mouse_data_{timestamp}"

def save_data():
    """將滑鼠資料儲存到指定的資料夾"""
    if not save_folder:
        messagebox.showerror("Error", "Please select a save folder first!")
        return
    filename = generate_filename()
    file_path = os.path.join(save_folder, f"{filename}.txt")
    with open(file_path, "w") as f:
        f.write("Mouse Movements (Time, X, Y):\n")
        for pos in mouse_positions:
            f.write(f"{pos[0]}, {pos[1]}, {pos[2]}\n")
        f.write("\nMouse Clicks (Time, Button, Action):\n")
        for click in mouse_clicks:
            f.write(f"{click[0]}, {click[1]}, {click[2]}\n")
    messagebox.showinfo("Success", f"Mouse data saved to {file_path}")
    generate_visualization(file_path)

def generate_visualization(file_path):
    """生成滑鼠移動的可視化圖表"""
    if not mouse_positions:
        messagebox.showwarning("Warning", "No data to visualize!")
        return

    x_coords = [pos[1] for pos in mouse_positions]
    y_coords = [pos[2] for pos in mouse_positions]

    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', markersize=2, label="Mouse Movement")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Mouse Movement Visualization")
    plt.legend()
    plt.gca().invert_yaxis()  # 符合螢幕座標系
    plt.grid(True)

    # 儲存圖表到同一資料夾
    img_path = file_path.replace(".txt", ".png")
    plt.savefig(img_path)
    plt.close()
    messagebox.showinfo("Visualization", f"Mouse movement visualization saved to {img_path}")

def start_stop_recording():
    """切換記錄狀態"""
    global recording
    recording = not recording
    if recording:
        status_label.config(text="Recording: ON", fg="green")
    else:
        status_label.config(text="Recording: OFF", fg="red")
        save_data()

def update_mouse_position(x, y):
    """更新滑鼠位置到 UI"""
    position_label.config(text=f"Mouse Position: ({x}, {y})")

def mouse_listener():
    """滑鼠監聽器"""
    def on_move(x, y):
        if recording:
            timestamp = time.time()
            mouse_positions.append((timestamp, x, y))
            update_mouse_position(x, y)

    def on_click(x, y, button, pressed):
        if recording:
            timestamp = time.time()
            action = "Pressed" if pressed else "Released"
            mouse_clicks.append((timestamp, button, action))
            print(f"Mouse {action} at ({x}, {y}) with {button}")

    def on_scroll(x, y, dx, dy):
        if recording:
            print(f"Mouse scrolled at ({x}, {y}) with delta ({dx}, {dy})")

    with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
        listener.join()

def keyboard_listener():
    """鍵盤監聽器，監控空白鍵切換記錄狀態"""
    def on_press(key):
        if key == keyboard.Key.space:
            start_stop_recording()

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def start_listeners():
    """啟動滑鼠與鍵盤監聽器的執行緒"""
    threading.Thread(target=mouse_listener, daemon=True).start()
    threading.Thread(target=keyboard_listener, daemon=True).start()

# 創建 Tkinter UI
app = tk.Tk()
app.title("Mouse Tracker")
app.geometry("500x400")

# 狀態顯示
status_label = tk.Label(app, text="Recording: OFF", fg="red", font=("Arial", 16))
status_label.pack(pady=10)

# 滑鼠位置顯示
position_label = tk.Label(app, text="Mouse Position: (0, 0)", font=("Arial", 14))
position_label.pack(pady=10)

# 資料夾選擇顯示
folder_label = tk.Label(app, text="Save Folder: Not selected", font=("Arial", 12))
folder_label.pack(pady=10)

# 選擇資料夾按鈕
select_folder_button = tk.Button(app, text="Select Save Folder", command=select_folder, font=("Arial", 12))
select_folder_button.pack(pady=5)

# 開始/停止按鈕
toggle_button = tk.Button(app, text="Start/Stop (Space)", command=start_stop_recording, font=("Arial", 14))
toggle_button.pack(pady=20)

# 啟動監聽器
start_listeners()

# 啟動 Tkinter 主迴圈
app.mainloop()
