import tkinter as tk
from PIL import Image, ImageTk

class MouseTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Mouse Tracker with Scrolling")

        # 创建一个可滚动的画布
        self.canvas = tk.Canvas(root)
        self.scroll_y = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        # 创建一个框架，并将其放置在画布上
        self.frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        # 绑定鼠标滚轮事件
        self.root.bind("<MouseWheel>", self.on_scroll)

        # 将滚动条和画布放置在窗口上
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y.pack(side="right", fill="y")

        # 向框架中添加内容（文字和图片）
        self.add_content()

        # 更新画布的滚动区域
        self.frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def add_content(self):
        # 添加一些文字
        text1 = tk.Label(self.frame, text="这是一些文字内容。这是一些文字内容。这是一些文字内容。这是一些文字内容。这是一些文字内容。",
                         wraplength=200, justify="left")
        text1.pack(pady=10)
        
        # 添加图片
        image = Image.open(r"D:\BenQ_Project\01_UR_lab\00_UR\figure\CS2CT.png")  # 替换为你的图片路径
        photo = ImageTk.PhotoImage(image)
        img_label = tk.Label(self.frame, image=photo)
        img_label.image = photo  # 保持引用防止图片被垃圾回收
        img_label.pack(pady=10)
        
        # 添加更多文字
        text2 = tk.Label(self.frame, text="这是更多的文字内容。")
        text2.pack(pady=10)

    def on_scroll(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")

if __name__ == "__main__":
    root = tk.Tk()
    app = MouseTracker(root)
    root.mainloop()
