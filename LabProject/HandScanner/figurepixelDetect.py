import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os

# 全域變數
click_positions = []
canvas = None
img_tk = None
img_original = None
image_on_canvas = None
scale = 1.0
text_ids = []

def open_image():
    global img_tk, canvas, img_original, image_on_canvas, scale, click_positions, text_ids

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.bmp *.jpeg")])
    if not file_path:
        return

    img_original = Image.open(file_path)
    scale = 1.0
    render_image()

    click_positions.clear()
    text_ids.clear()
    update_table()

def render_image():
    """依照 scale 重新繪製圖片並刷新 canvas"""
    global img_tk, image_on_canvas, text_ids

    if img_original is None:
        return

    new_size = (int(img_original.width * scale), int(img_original.height * scale))
    img_resized = img_original.resize(new_size, Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_resized)

    canvas.delete("all")
    text_ids.clear()
    image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    # 重新畫點與編號
    for i, (x, y) in enumerate(click_positions):
        draw_point(x, y, i + 1)

    canvas.config(scrollregion=canvas.bbox(tk.ALL))

def on_click(event):
    global click_positions

    if img_original is None:
        return

    x = int(canvas.canvasx(event.x) / scale)
    y = int(canvas.canvasy(event.y) / scale)

    click_positions.append((x, y))
    draw_point(x, y, len(click_positions))
    update_table()

def draw_point(x, y, index):
    """畫紅點與編號於縮放後的座標"""
    global text_ids

    x_scaled = x * scale
    y_scaled = y * scale
    canvas.create_oval(x_scaled - 3, y_scaled - 3, x_scaled + 3, y_scaled + 3, fill="red")
    text_id = canvas.create_text(x_scaled + 10, y_scaled, text=str(index), fill="blue", font=("Arial", 12, "bold"))
    text_ids.append(text_id)

def update_table():
    copy_text.delete("1.0", tk.END)
    for row in table.get_children():
        table.delete(row)

    for i, (x, y) in enumerate(click_positions, 1):
        table.insert("", "end", values=(i, x, y))
        copy_text.insert(tk.END, f"{i}\t{x}\t{y}\n")

def clear_points():
    global click_positions, text_ids
    click_positions.clear()
    text_ids.clear()
    render_image()
    update_table()

def zoom(event):
    global scale
    if img_original is None:
        return

    if event.delta > 0 and scale < 5.0:
        scale *= 1.1
    elif event.delta < 0 and scale > 0.2:
        scale /= 1.1
    render_image()

def on_resize(event):
    canvas.config(scrollregion=canvas.bbox(tk.ALL))

def export_to_csv():
    if not click_positions:
        messagebox.showinfo("提示", "沒有標記資料可以匯出！")
        return
    path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("點次,X,Y\n")
        for i, (x, y) in enumerate(click_positions, 1):
            f.write(f"{i},{x},{y}\n")
    messagebox.showinfo("成功", f"已匯出至：{os.path.basename(path)}")

# === 主視窗建立 ===
root = tk.Tk()
root.title("圖片標點工具")
root.geometry("1000x700")
root.resizable(True, True)

# === 左側 Canvas 畫布區 ===
canvas_frame = ttk.Frame(root)
canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=10, pady=10)

canvas = tk.Canvas(canvas_frame, bg='white')
canvas.pack(fill="both", expand=True)
canvas.bind("<Button-1>", on_click)
canvas.bind("<MouseWheel>", zoom)
canvas.bind("<Configure>", on_resize)

# === 操作按鈕 ===
btn_frame = ttk.Frame(canvas_frame)
btn_frame.pack(pady=10)
ttk.Button(btn_frame, text="開啟圖片", command=open_image).grid(row=0, column=0, padx=5)
ttk.Button(btn_frame, text="清除標記", command=clear_points).grid(row=0, column=1, padx=5)

# === 右側資訊欄位 ===
table_frame = ttk.Frame(root)
table_frame.pack(side=tk.RIGHT, fill="y", padx=10, pady=10)

ttk.Label(table_frame, text="標記座標 (像素)").pack()

# Treeview 表格
columns = ("點次", "X", "Y")
table = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
for col in columns:
    table.heading(col, text=col)
    table.column(col, anchor="center", width=80)
table.pack()

# 複製區塊
ttk.Label(table_frame, text="（可複製）").pack(pady=(10, 0))
copy_text = tk.Text(table_frame, height=10, width=30)
copy_text.pack()

# 匯出 CSV 按鈕
ttk.Button(table_frame, text="匯出 CSV", command=export_to_csv).pack(pady=5)

# === 執行主程式 ===
root.mainloop()
