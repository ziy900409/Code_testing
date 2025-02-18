import logging
import os
import tkinter as tk
from datetime import datetime
from io import BytesIO
from tkinter import ttk, messagebox

import PIL.Image
import PIL.ImageTk
import twain
from twain.lowlevel import constants
import math


# 設置日誌
logging.basicConfig(level=logging.INFO)

# 變數來存儲選擇的掃描儀名稱
selected_scanner = None

# 初始化 Tkinter
root = tk.Tk()
root.title("掃描儀控制")

# 創建 TWAIN SourceManager
sm = twain.SourceManager(root)

# 圖片顯示區域
image_label = None


def select():
    """選擇掃描儀"""
    global selected_scanner

    with sm as source_manager:
        scanners = source_manager.GetSourceList()
        if not scanners:
            messagebox.showerror("錯誤", "未偵測到任何掃描儀，請確認已連接並安裝驅動程式！")
            return
        
        # 創建一個選擇掃描儀的對話框
        scanner_window = tk.Toplevel(root)
        scanner_window.title("選擇掃描儀")

        tk.Label(scanner_window, text="選擇掃描儀:").pack(pady=10)

        scanner_var = tk.StringVar()
        scanner_var.set(scanners[0])  # 預設選擇第一台掃描儀

        # 建立掃描儀選擇的下拉選單
        scanner_dropdown = ttk.Combobox(scanner_window, textvariable=scanner_var, values=scanners, state="readonly")
        scanner_dropdown.pack(pady=10)

        # 按鈕來確定選擇
        def confirm_selection():
            global selected_scanner
            selected_scanner = scanner_var.get()
            messagebox.showinfo("成功", f"已選擇掃描儀: {selected_scanner}")
            scanner_window.destroy()

        tk.Button(scanner_window, text="確定", command=confirm_selection).pack(pady=10)


def scan():
    """執行掃描並固定掃描範圍為 A4"""
    global selected_scanner

    if not selected_scanner:
        messagebox.showerror("錯誤", "請先選擇掃描儀！")
        return

    show_ui = False
    dpi = 300  # 設定解析度
    scan_num = 1

    with twain.SourceManager(None) as source_manager:
        try:
            scanners = source_manager.GetSourceList()
        except twain.exceptions.GeneralFailure:
            messagebox.showerror("錯誤", "無法獲取掃描儀清單，請確保掃描儀已正確安裝並連接！")
            return

        if selected_scanner not in scanners:
            messagebox.showerror("錯誤", f"選擇的掃描儀 '{selected_scanner}' 不可用，請重新選擇。")
            return

        print(f"使用掃描儀: {selected_scanner}")

        sd = source_manager.OpenSource(selected_scanner)
        if not sd:
            messagebox.showerror("錯誤", "無法開啟掃描儀！")
            return

        # **設定單位為公分**
        sd.SetCapability(constants.ICAP_UNITS, constants.TWTY_UINT16, constants.TWUN_CENTIMETERS) # 設定單位為公分

         # **設定 A4 掃描範圍 (左上角 0,0 到 右下角 21.0 x 29.7)**
        scan_area = [0.0, 0.0, 21.0, 29.7]  # 左上角 (0,0) 到 右下角 (21.0, 29.7)
        sd.SetCapability(constants.ICAP_FRAMES, constants.TWTY_FRAME, scan_area)

        
        # **設定解析度**
        sd.SetCapability(constants.ICAP_XRESOLUTION, constants.TWTY_FIX32, dpi)
        sd.SetCapability(constants.ICAP_YRESOLUTION, constants.TWTY_FIX32, dpi)

        # **開始掃描**
        sd.RequestAcquire(show_ui=show_ui, modal_ui=False)
        sd.ModalLoop()

        more = 1
        while more:
            (handle, has_more) = sd.XferImageNatively()
            more = has_more
            print(f"has_more: {has_more}")

            if handle is None:
                messagebox.showerror("錯誤", "未成功獲取掃描圖像")
                return

            # **轉換為 BMP**
            bmp_bytes = twain.dib_to_bm_file(handle)

            # 轉換為 PIL 影像
            img = PIL.Image.open(BytesIO(bmp_bytes))

            # **儲存掃描圖像**
            if not os.path.exists("imgs"):
                os.makedirs("imgs")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"imgs/scan_{timestamp}_{scan_num:03d}.jpg"
            img.save(file_name, format='jpeg')
            print(f"已儲存: {file_name}")

            # 顯示圖片
            show_image(file_name)

            scan_num += 1

        messagebox.showinfo("完成", "掃描完成並保存圖片！")



# 儲存點擊的座標與標記 ID
click_positions = []
measurement_records = []  # 存儲測量結果
dpi = 300  # 掃描影像的 DPI
record_window = None  # 記錄視窗
line_ids = []  # 存放畫布上的線條 ID
text_ids = []  # 存放畫布上的文字 ID
point_ids = []  # 存放畫布上的點擊標記 ID

def show_image(file_name):
    """顯示最近掃描的圖片，提供點擊功能來測量距離，並記錄測量結果"""
    global image_label, canvas, img_tk, click_positions, record_window, line_ids, text_ids, point_ids

    # 讀取圖片
    num_size = 2
    resize_width = 210*num_size
    resize_height = 297*num_size
    img = PIL.Image.open(file_name)
    img_width, img_height = img.size  # 取得影像大小
    img = img.resize((resize_width, resize_height), PIL.Image.Resampling.LANCZOS)  # 縮小顯示，但保留原解析度

    img_tk = PIL.ImageTk.PhotoImage(img)

    # 創建新視窗來顯示影像
    img_window = tk.Toplevel()
    img_window.title("掃描結果 - 點擊兩點測量距離")

    # 創建 Canvas 來顯示影像
    canvas = tk.Canvas(img_window, width=resize_width, height=resize_height)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    # 創建紀錄視窗（如果尚未開啟）
    if record_window is None or not record_window.winfo_exists():
        create_record_window()

    # 點擊座標事件
    def on_click(event):
        global click_positions

        # 轉換點擊座標到原始影像大小
        scale_x = img_width / resize_width
        scale_y = img_height / resize_height
        x = int(event.x * scale_x)
        y = int(event.y * scale_y)

        # 存儲點擊位置
        click_positions.append((x, y))

        # **標記點擊的座標**
        point_id = canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")
        point_ids.append(point_id)  # 存入點擊標記 ID

        if len(click_positions) == 2:
            draw_line_and_measure()

    def draw_line_and_measure():
        global click_positions, line_ids, text_ids

        # 取得兩點座標
        (x1, y1), (x2, y2) = click_positions

        # 計算像素距離
        pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # 轉換為實際長度
        inch_distance = round(pixel_distance / dpi, 3)  # 轉換為英吋（四捨五入到三位數）
        cm_distance = round(inch_distance * 2.54, 3)  # 轉換為公分

        # 轉換座標到畫布上的縮放比例
        scale_x = resize_width / img_width
        scale_y = resize_height / img_height
        x1_scaled, y1_scaled = x1 * scale_x, y1 * scale_y
        x2_scaled, y2_scaled = x2 * scale_x, y2 * scale_y

        # **畫出連接兩點的紅線**
        line_id = canvas.create_line(x1_scaled, y1_scaled, x2_scaled, y2_scaled, fill="red", width=2)
        line_ids.append(line_id)

        # **標記編號**
        measurement_index = len(measurement_records) + 1  # 設定測量編號
        mid_x = (x1_scaled + x2_scaled) / 2
        mid_y = (y1_scaled + y2_scaled) / 2
        text_id = canvas.create_text(mid_x, mid_y, text=str(measurement_index), font=("Arial", 12, "bold"), fill="blue")
        text_ids.append(text_id)

        # **記錄測量結果**
        measurement_records.append((measurement_index, round(pixel_distance, 3), inch_distance, cm_distance))
        update_record_window()

        # 顯示結果
        messagebox.showinfo("測量結果", f"編號: {measurement_index}\n"
                                       f"像素距離: {round(pixel_distance, 3)} px\n"
                                       f"實際距離: {inch_distance} 英吋\n"
                                       f"實際距離: {cm_distance} 公分")

        click_positions = []  # 重置點擊點

    # 綁定點擊事件
    canvas.bind("<Button-1>", on_click)

    # 新增清除按鈕
    clear_button = tk.Button(img_window, text="清除畫面", command=clear_canvas)
    clear_button.pack(pady=5)



    def clear_canvas():
        """清除畫布上的所有標記、連線，並同步清除測量紀錄"""
        global line_ids, text_ids, point_ids, measurement_records

        # 清除畫布上的所有標記和連線
        for line in line_ids:
            canvas.delete(line)
        for text in text_ids:
            canvas.delete(text)
        for point in point_ids:
            canvas.delete(point)

        # 清空記錄
        line_ids.clear()
        text_ids.clear()
        point_ids.clear()
        measurement_records.clear()  # 清除測量紀錄數據
        update_record_window()  # 更新測量記錄視窗

        messagebox.showinfo("清除完成", "已清除所有測量標記、連線與記錄")



def create_record_window():
    """創建一個新視窗來顯示測量記錄"""
    global record_window, record_tree

    # 創建新視窗
    record_window = tk.Toplevel()
    record_window.title("測量記錄")
    record_window.geometry("400x300")

    # 創建表格
    columns = ("編號", "像素距離", "英吋距離", "公分距離")
    record_tree = ttk.Treeview(record_window, columns=columns, show="headings")

    # 設定標題
    for col in columns:
        record_tree.heading(col, text=col)
        record_tree.column(col, anchor="center")

    record_tree.pack(expand=True, fill="both")

    # 清除記錄按鈕
    clear_button = tk.Button(record_window, text="清除記錄", command=clear_records)
    clear_button.pack(pady=5)


def update_record_window():
    """更新測量記錄視窗中的數據"""
    global record_tree

    # 清空舊數據
    for row in record_tree.get_children():
        record_tree.delete(row)

    # 插入新數據（四捨五入到小數點後三位）
    for record in measurement_records:
        formatted_record = (
            record[0],  # 編號
            round(record[1], 3),  # 像素距離
            round(record[2], 3),  # 英吋距離
            round(record[3], 3)   # 公分距離
        )
        record_tree.insert("", "end", values=formatted_record)



def clear_records():
    """清空測量記錄"""
    global measurement_records
    measurement_records = []
    update_record_window()



# 建立 GUI 介面
frm = ttk.Frame(root, padding=10)
frm.grid()

ttk.Button(frm, text="選擇掃描儀", command=select).grid(column=0, row=0, pady=10)
ttk.Button(frm, text="開始掃描", command=scan).grid(column=0, row=1, pady=10)

root.mainloop() 