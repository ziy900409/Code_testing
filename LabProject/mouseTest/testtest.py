"""
2024 0613 16:33 版本
"""

import tkinter as tk
from PIL import Image, ImageTk
import json
import time
import random

def load_text_from_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

data_path = r"D:\BenQ_Project\FittsDragDropTest\ScrollWheelTest2\\"

web_contents = {
    "first_page": {
        "title": "ZOWIE 滑鼠設計中的運動科學",
        "subtitle_1": "人機創新實驗室",
        "text_1": load_text_from_file(data_path + "first_page_1.txt"),
        "image_1": data_path + "first_page_fig1.jpg",
        "subtitle_2": "玩家注重什麼？",
        "text_2": load_text_from_file(data_path + "first_page_2.txt"),
        "image_2": data_path + "first_page_fig2.jpg",
        "subtitle_3": "從運動科學看EC-CW的優點",
        "text_3": load_text_from_file(data_path + "first_page_3.txt"),
        "image_3": data_path + "first_page_fig3.jpg"
    },
    "second_page": {
        "title": "《VALORANT特戰英豪》的最佳螢幕設定",
        "subtitle_1": "",
        "text_1": load_text_from_file(data_path + "second_page_1.txt"),
        "image_1": data_path + "second_page_fig1.jpg",
        "subtitle_2": "電競選手專訪",
        "text_2": load_text_from_file(data_path + "first_page_2.txt"),
        "image_2": data_path + "second_page_fig2.jpg",
        "subtitle_3": "沒有最好的設定，只有最適合你的設定",
        "text_3": load_text_from_file(data_path + "second_page_3.txt"),
        "image_3": data_path + "second_page_fig3.jpg"
    },
    "third_page": {
        "title": "《VALORANT特戰英豪》的最佳螢幕設定",
        "subtitle_1": "",
        "text_1": load_text_from_file(data_path + "third_page_1.txt"),
        "image_1": data_path + "third_page_fig1.jpg",
        "subtitle_2": "電競選手專訪",
        "text_2": load_text_from_file(data_path + "third_page_2.txt"),
        "image_2": data_path + "third_page_fig2.jpg",
        "subtitle_3": "沒有最好的設定，只有最適合你的設定",
        "text_3": load_text_from_file(data_path + "third_page_3.txt"),
        "image_3": data_path + "third_page_fig3.jpg"
    }
}

class ContentNavigator:
    def __init__(self, root, content):
        self.root = root
        self.content = content
        self.page_keys = list(content.keys())
        self.page_index = -1
        self.events = []

        self.root.title("Content Navigator with Scrolling")

        self.canvas = tk.Canvas(root)
        self.scroll_y = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        self.root.bind("<MouseWheel>", self.on_scroll)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y.pack(side="right", fill="y")

        self.title_label = tk.Label(self.frame, text="", font=("Arial", 20, "bold"))
        self.title_label.pack(pady=10)

        self.subtitle_label_1 = tk.Label(self.frame, text="", font=("Arial", 16))
        self.subtitle_label_1.pack(pady=10)

        self.text_label_1 = tk.Label(self.frame, text="", wraplength=400, justify="left")
        self.text_label_1.pack(pady=10)

        self.image_label_1 = tk.Label(self.frame)
        self.image_label_1.pack(pady=10)
        
        self.subtitle_label_2 = tk.Label(self.frame, text="", font=("Arial", 16))
        self.subtitle_label_2.pack(pady=10)

        self.text_label_2 = tk.Label(self.frame, text="", wraplength=400, justify="left")
        self.text_label_2.pack(pady=10)

        self.image_label_2 = tk.Label(self.frame)
        self.image_label_2.pack(pady=10)

        self.subtitle_label_3 = tk.Label(self.frame, text="", font=("Arial", 16))
        self.subtitle_label_3.pack(pady=10)

        self.text_label_3 = tk.Label(self.frame, text="", wraplength=400, justify="left")
        self.text_label_3.pack(pady=10)

        self.image_label_3 = tk.Label(self.frame)
        self.image_label_3.pack(pady=10)

        self.next_button = tk.Button(self.frame, text="下一頁", command=self.show_random_page, borderwidth=0, highlightthickness=0)
        self.next_button.pack(pady=10)

        self.show_random_page()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def show_random_page(self):
        if len(self.page_keys) == 0:
            self.save_events()
            self.root.destroy()
            return

        self.page_index = random.randint(0, len(self.page_keys) - 1)
        self.show_page(self.content[self.page_keys[self.page_index]])
        self.page_keys.pop(self.page_index)

        self.canvas.yview_moveto(0)

    def show_page(self, page_content):
        self.title_label.config(text=page_content["title"])
        self.subtitle_label_1.config(text=page_content["subtitle_1"])
        self.text_label_1.config(text=page_content["text_1"])
        self.load_image(self.image_label_1, page_content["image_1"])

        self.subtitle_label_2.config(text=page_content["subtitle_2"])
        self.text_label_2.config(text=page_content["text_2"])
        self.load_image(self.image_label_2, page_content["image_2"])

        self.subtitle_label_3.config(text=page_content["subtitle_3"])
        self.text_label_3.config(text=page_content["text_3"])
        self.load_image(self.image_label_3, page_content["image_3"])

        self.frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def load_image(self, label, image_path):
        image = Image.open(image_path)
        resized_image = image.resize((400, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(resized_image)
        label.config(image=photo)
        label.image = photo

    def on_scroll(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")
        self.log_event("scroll", {"x": event.x, "y": event.y, "delta": event.delta})

    def on_mouse_move(self, event):
        self.log_event("move", {"x": event.x, "y": event.y})

    def log_event(self, event_type, event_info):
        event_time = time.time()
        self.events.append({
            "type": event_type,
            "info": event_info,
            "time": event_time,
            "page": self.page_index
        })

    def save_events(self):
        with open(r"D:\BenQ_Project\git\Code_testing\LabProject\mouseTest\templates\mouse_events.json", "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=4, ensure_ascii=False)

    def on_closing(self):
        self.save_events()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ContentNavigator(root, web_contents)
    root.mainloop()
