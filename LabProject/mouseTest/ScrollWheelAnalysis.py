# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:36:24 2024

@author: Hsin.YH.Yang
"""

import json

# 假設你有一個名為 data.json 的 JSON 文件
file_path = r"D:\BenQ_Project\git\Code_testing\LabProject\mouseTest\templates\mouse_events.json"

# 打開並讀取 JSON 文件
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 顯示讀取到的數據
print(data)


def load_text_from_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    return content

web_contents = {
    "title": "ZOWIE 滑鼠設計中的運動科學",
    "first_page":
    {   "subtitle": "人機創新實驗室",
        "text": load_text_from_file(r"D:\BenQ_Project\git\Code_testing\LabProject\mouseTest\templates\first_page.txt"),
        "image": r"D:\BenQ_Project\01_UR_lab\00_UR\figure\CS2CT.png"
        },
    "second_page": 
    {"text": "这是第二篇的文字内容。这是第二篇的文字内容。这是第二篇的文字内容。",
     "image": r"D:\BenQ_Project\01_UR_lab\00_UR\figure\CS2CT.png"},
    "third_page":
    {"text": "这是第三篇的文字内容。这是第三篇的文字内容。这是第三篇的文字内容。",
     "image": r"D:\BenQ_Project\01_UR_lab\00_UR\figure\CS2CT.png"}
             }