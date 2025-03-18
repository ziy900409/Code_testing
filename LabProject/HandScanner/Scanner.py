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
        sd.SetCapability(constants.ICAP_UNITS, constants.TWTY_UINT16, constants.TWUN_CENTIMETERS)

        # **設定 A4 掃描範圍 (左上角 0,0 到 右下角 21.0 x 29.7 cm)**
        scan_area = [0.0, 0.0, 21.0, 29.7]  # A4 紙張範圍（單位: cm）
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





def show_image(file_name):
    """顯示最近掃描的圖片"""
    global image_label

    # 讀取圖片
    img = PIL.Image.open(file_name)
    img = img.resize((400, 300), PIL.Image.Resampling.LANCZOS)  # 調整大小以適應 GUI

    img_tk = PIL.ImageTk.PhotoImage(img)

    # 檢查是否已經有圖片顯示，若有則更新，否則新增
    if image_label is None:
        image_label = tk.Label(root, image=img_tk)
        image_label.image = img_tk  # 保存引用，避免垃圾回收
        image_label.grid(column=0, row=2, pady=10)
    else:
        image_label.configure(image=img_tk)
        image_label.image = img_tk  # 更新圖片


# 建立 GUI 介面
frm = ttk.Frame(root, padding=10)
frm.grid()

ttk.Button(frm, text="選擇掃描儀", command=select).grid(column=0, row=0, pady=10)
ttk.Button(frm, text="開始掃描", command=scan).grid(column=0, row=1, pady=10)

root.mainloop()
