import pyautogui
import subprocess
import time
from pynput import mouse
import psutil

# 开启 Microsoft Word 并打开指定文件
def open_word(file_path):
    subprocess.Popen(['start', 'winword', file_path], shell=True)
    time.sleep(5)  # 等待 Word 启动

# 关闭 Microsoft Word
def close_word():
    # 使用 pyautogui 模拟键盘快捷键 Alt+F4 关闭 Word
    pyautogui.hotkey('alt', 'f4')

# 检查 Word 进程是否仍在运行
def is_word_running():
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] == 'WINWORD.EXE':
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

# 记录滑鼠事件
mouse_events = []

def on_move(x, y):
    mouse_events.append(('move', x, y, time.time()))

def on_click(x, y, button, pressed):
    mouse_events.append(('click', button.name, 'pressed' if pressed else 'released', x, y, time.time()))

def on_scroll(x, y, dx, dy):
    mouse_events.append(('scroll', dx, dy, x, y, time.time()))

# 开始监听滑鼠事件
listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
listener.start()

try:
    # 指定 Word 文件的路径
    word_file_path = r"D:\BenQ_Project\FittsDragDropTest\ScrollWheelTest2\test.docx"
    
    # 打开指定的 Word 文件
    open_word(word_file_path)

    print("Word 已经打开，正在记录滑鼠事件...")

    # 持续检查 Word 是否仍在运行
    while True:
        if not is_word_running():
            print("Word 已经关闭。")
            break
        time.sleep(1)  # 每秒检查一次

finally:
    # 停止监听滑鼠事件
    listener.stop()

    # 将滑鼠事件保存到文件中
    with open(r"D:\BenQ_Project\FittsDragDropTest\ScrollWheelTest2\mouse_events.txt", 'w') as f:
        for event in mouse_events:
            f.write(f"{event}\n")

    print("滑鼠事件已保存到 mouse_events.txt。")
