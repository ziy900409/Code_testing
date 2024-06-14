import tkinter as tk
import random
import time


class ReactionTimeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("滑鼠側鍵點擊測試")

        self.canvas = tk.Canvas(root, width=200, height=200)
        self.canvas.pack()

        self.circle = self.canvas.create_oval(50, 50, 150, 150, fill="gray", outline="")
        self.canvas.tag_bind(self.circle, "<Button-1>", self.on_click)
        self.num_tests = 3
        self.test_count = 0
        self.total_reaction_time = 0
        self.reaction_times = []
        self.run_tests()

    def run_tests(self):
        if self.test_count < self.num_tests:
            self.change_color()
            print(self.test_count)
            
   
    def change_color(self):
        self.canvas.itemconfig(self.circle, fill="green")
        self.start_time = time.time()

        def change_to_gray():
            self.canvas.itemconfig(self.circle, fill="gray")
            self.end_time = time.time()
            reaction_time = self.end_time - self.start_time
            self.total_reaction_time += reaction_time
            self.reaction_times.append(reaction_time)
            self.test_count += 1
            if self.test_count < self.num_tests:
                self.root.after(1000 * random.randint(0, 3), self.run_tests)

        self.root.after(1000, change_to_gray)

    def on_click(self, event):
        if self.canvas.itemcget(self.circle, "fill") == "green":
            self.end_time = time.time()
            reaction_time = self.end_time - self.start_time
            self.total_reaction_time += reaction_time
            self.reaction_times.append(reaction_time)
            print("點擊反應時間:", round(reaction_time, 3), "秒")
            
            if self.test_count == self.num_tests-1:
                self.calculate_average_reaction_time()
                self.root.quit()  # 結束遊戲
         
            
    def calculate_average_reaction_time(self):
        average_reaction_time = sum(self.reaction_times) / len(self.reaction_times)
        print("平均點擊反應時間:", round( average_reaction_time, 3), "秒")

root = tk.Tk()
app = ReactionTimeApp(root)
root.mainloop()
