# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:31:32 2025

@author: Hsin.YH.Yang
"""

import pandas as pd
import matplotlib.pyplot as plt
import re

# 讀取 txt 檔案
file_path = r"C:\Users\Hsin.YH.Yang\Desktop\Katowice\test3.txt"



# 解析函數
def parse_line(line):
    # 正則表達式解析數據
    pattern = r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}\.\d{2} (?:上午|下午)) Macro Record: (\d+) \| (\d+) \| (\d+) \| (\d+) \| (.+)'
    match = re.match(pattern, line)
    if match:
        timestamp, frame, x, y, code, action = match.groups()
        return timestamp, int(x), int(y), int(code), action.strip()
    return None

# 讀取並解析所有行
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parsed = parse_line(line.strip())
        if parsed:
            data.append(parsed)

# 建立 DataFrame
df = pd.DataFrame(data, columns=['timestamp', 'x', 'y', 'code', 'action'])

# 修正日期格式函數
def convert_chinese_am_pm(timestamp_str):
    # 轉換「上午」為「AM」，「下午」為「PM」
    timestamp_str = timestamp_str.replace("上午", "AM").replace("下午", "PM")
    return timestamp_str

# 讀取數據並修正日期格式
df['timestamp'] = df['timestamp'].apply(convert_chinese_am_pm)

# 轉換為 datetime，修正格式為 %m/%d/%Y %I:%M:%S.%f %p
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %I:%M:%S.%f %p')

# 視覺化滑鼠移動軌跡
plt.figure(figsize=(10, 6))
plt.plot(df['x'], df['y'], marker='o', linestyle='-', color='blue', label='Mouse Movement')

# 標記點擊事件
click_events = df[df['action'].str.contains('Click')]
plt.scatter(click_events['x'], click_events['y'], color='red', s=100, label='Click Event')

# 翻轉Y軸，使其符合螢幕座標系統
plt.gca().invert_yaxis()
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Mouse Movement Trajectory')
plt.legend()
plt.grid(True)
plt.show()
