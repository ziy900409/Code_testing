# -*- coding: utf-8 -*-
"""
Created on Sun May 25 21:10:33 2025

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
from matplotlib.ticker import MaxNLocator
plt.rcParams['font.sans-serif'] =  ['Roboto']  
plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號
plt.rcParams['figure.dpi'] = 150


def plot_median_freq_slope_comparison(group1: dict, group2: dict,
                                      title: str = "Median Frequency Slope Comparison",
                                      ylabel: str = "Fatigue Index",
                                      turn: bool = True,
                                      selected_keys: list = None,
                                      label_list: list = None,
                                      show_values: bool = True,
                                      custom_xticklabels: Optional[Dict[str, str]] = None):
    """
    繪製兩組 Median Frequency Slope 的比較柱狀圖（可指定 key，支援絕對值顯示）

    Parameters:
    - group1, group2: dict，key 為肌肉名稱，value 為 slope 值
    - title: 圖表標題
    - selected_keys: list[str]，若提供，只繪製這些 key
    - show_values: 是否在柱子上顯示數值（會顯示絕對值）
    """
    if label_list is None:
        label_list = ["Pre", "Pos"]
    
    common_keys = set(group1.keys()) & set(group2.keys())
    
    if selected_keys:
        keys = [k for k in selected_keys if k in common_keys]
    else:
        keys = sorted(common_keys)

    if not keys:
        print("❌ 找不到有效的 key 可比較，請確認 key 是否存在於兩組資料中")
        return
    if turn:
        values1 = [-(group1[k]) for k in keys]
        values2 = [-(group2[k]) for k in keys]
    else:
        values1 = [(group1[k]) for k in keys]
        values2 = [(group2[k]) for k in keys]
        
    # --- START: 修改部分 ---
    # 根據 custom_xticklabels 的類型（字典或列表）生成顯示標籤，以避免 AttributeError
    display_labels = keys
    if custom_xticklabels:
        if isinstance(custom_xticklabels, dict):
            # 如果是字典，使用 .get() 方法安全地獲取標籤
            display_labels = [custom_xticklabels.get(k, k) for k in keys]
        elif isinstance(custom_xticklabels, list):
            # 如果是列表，檢查長度是否匹配
            if len(custom_xticklabels) == len(keys):
                display_labels = custom_xticklabels
            else:
                print(f"⚠️ 警告: 'custom_xticklabels' 列表的長度 ({len(custom_xticklabels)}) 與 key 的數量 ({len(keys)}) 不符。將使用原始 key 作為標籤。")
    # --- END: 修改部分 ---

    # 根據 custom_xticklabels 生成顯示用的標籤
    # display_labels = [custom_xticklabels.get(k, k) for k in keys] if custom_xticklabels else keys

    x = np.arange(len(keys))
    # width = 0.35
    
    # x = np.arange(len(keys)) * 1.5  # 放大 x 軸間距
    # width = 0.3  # 稍微窄一點避免重疊

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 2.7), 4.4))
    # bars1 = ax.bar(x - width/2, values1, width, label=label_list[0], color="#3B3B3B")
    # bars2 = ax.bar(x + width/2, values2, width, label=label_list[1], color="#CC0040")
    
    width = 0.25       # 原本 0.35 → 改小一點
    offset = 0.14      # 控制兩組柱子之間的間距
    
    # bars1 = ax.bar(x - offset, values1, width, label=label_list[0], color="#CC0040")
    # bars1 = ax.bar(x - offset, values1, width, label=label_list[0],
    #               edgecolor="#CC0040",edgecolor='none', facecolor='none', hatch='\\\\', linewidth=2)
    
    bars1 = ax.bar(x - offset, values1, width, label=label_list[0],
                   edgecolor="#3B3B3B",   # 用斜線顏色
                   facecolor='none',
                   hatch='\\\\\\',
                   linewidth=0.1          # ✅ 非常細的外框線
                   )
    bars2 = ax.bar(x + offset, values2, width, label=label_list[1], color="#CC0040")
    
    # ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.8)
    # ax.set_ylabel(ylabel)
    # ax.set_title(title, fontsize=20, pad=20, fontweight='bold') # pad 參數控制標題與圖表的間距
    # ax.text(0.5, -0.15, title, fontsize=16, ha='center', va='top', transform=ax.transAxes)

    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.spines['left'].set_visible(False)   # 左框線
    ax.spines['right'].set_visible(False)  # 右框線
    ax.spines['top'].set_visible(False)    # 上框線
    ax.spines['bottom'].set_visible(False) # 下框線（依需要）
    
    # 將 X 軸刻度與標籤移至頂部
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='both', length=0) # 隱藏刻度線本身，但保留標籤
    ax.axhline(y=0, color='#3B3B3B', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, ha='center', fontsize=16, color='#5A5A5A', fontweight='bold') # 現在這會作用在頂部
    
    if custom_xticklabels:
        ax.set_xticklabels(display_labels, ha='center') # 使用自訂標籤
    else:
        ax.set_xticklabels(keys, ha='center')
    # ax.legend()
    ax.grid(True, axis='y', linestyle=(0, (10, 5)), alpha=0.7)
    
    
    # ✅ 在柱子上加上絕對值數字
    if show_values:
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{abs(height):.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{abs(height):.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10)

    plt.tight_layout()
    plt.show()
    


# %%
def plot_performance_comparison(
    data: Dict[str, Dict[str, float]],
    title: str = "Performance Comparison (Pre vs. Post)",
    pre_post_labels: List[str] = ['實驗前', '實驗後'],
    # pre_post_colors: List[str] = ['#CC0040', '#CC0040'],
    custom_texts: Optional[Dict[str, Tuple[str, str]]] = None,
    color: List[str] = ['#CC0040', '#212121', '#F1A012']
):
    # 設定中文字型與負號正確顯示
    # plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # 取得所有的指標名稱，例如 'Accuracy', 'TTK (ms)'
    metrics = list(data.keys())
    num_metrics = len(metrics)
    if num_metrics == 0:
        print("❌ 資料為空，無法繪圖。")
        return

    # 根據指標數自動決定子圖的列數與行數
    cols = min(3, num_metrics)
    rows = math.ceil(num_metrics / cols)

    # 建立對應數量的子圖
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 5), squeeze=False)
    axs = axs.flatten()

    # 為每個指標畫一張子圖
    for i, metric_name in enumerate(metrics):
        ax = axs[i]
        metric_data = data.get(metric_name, {})

        # 取得 pre 和 pos 的數值
        value_pre = metric_data.get('pre', 0)
        value_pos = metric_data.get('pos', 0)

        # bar 的位置參數
        x = np.arange(1)
        width = 0.2
        offset = 0.12
        if color and len(metrics) == len(color):
            color_ind = color[i]
        else:
            color_ind = '#CC0040'
        # 畫 pre 的 bar（外框）
        bars1 = ax.bar(x - offset,
                       [value_pre],
                       width,
                       label=pre_post_labels[0],
                       edgecolor="#3B3B3B",
                       facecolor='none',
                       hatch='\\\\\\',
                       linewidth=0.1)

        # 畫 pos 的 bar（實心）
        bars2 = ax.bar(x + offset,
                       [value_pos],
                       width,
                       label=pre_post_labels[1],
                       color=color_ind)

        # 隱藏上、左、右邊框，只保留下邊框並設顏色與寬度
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#5A5A5A')
        ax.spines['bottom'].set_linewidth(2)

        # 移除 x 軸刻度
        ax.set_xticks([])
        ax.tick_params(axis='x', bottom=False)

        # 處理標籤（主標題與單位）
        if custom_texts and metric_name in custom_texts:
            custom_pair = custom_texts[metric_name]
            if isinstance(custom_pair, (list, tuple)) and len(custom_pair) == 2:
                main_text, unit_text = custom_pair
            else:
                main_text, unit_text = metric_name, ""
        else:
            # 若 metric 包含 (單位)
            if "(" in metric_name and metric_name.endswith(")"):
                parts = metric_name.rsplit("(", 1)
                main_text = parts[0].strip()
                unit_text = f"({parts[1]}"
            else:
                main_text, unit_text = metric_name, ""

        # 若有單位則加前導空格
        if unit_text:
            unit_text = f" {unit_text}"

        # 手動置中標籤使用的 Y 座標
        y_pos = -0.05
        advanced_centering_success = False

        try:
            # 渲染一次圖形以取得文字寬度
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            # 隱藏測量用的文字
            t_main = ax.text(0, 0, main_text, fontsize=16, fontweight='bold', alpha=0.0)
            t_unit = ax.text(0, 0, unit_text, fontsize=16, fontweight='normal', alpha=0.0)

            # 取得像素寬度
            w_main_px = t_main.get_window_extent(renderer).width
            w_unit_px = t_unit.get_window_extent(renderer).width
            ax_width_px = ax.get_window_extent().width

            # 移除測量用文字
            t_main.remove()
            t_unit.remove()

            # 若子圖與主文字寬度都有效
            if ax_width_px > 0 and w_main_px > 0:
                # 計算兩段文字的總寬比例
                total_text_width_ratio = (w_main_px + w_unit_px) / ax_width_px
                start_x_main = 0.5 - (total_text_width_ratio / 2)
                start_x_unit = start_x_main + (w_main_px / ax_width_px)

                # 主標題與單位文字置中顯示
                ax.text(start_x_main, y_pos, main_text, ha='left', va='top', transform=ax.transAxes,
                        fontsize=18, fontweight='bold', color='black')
                ax.text(start_x_unit, y_pos, unit_text, ha='left', va='top', transform=ax.transAxes,
                        fontsize=18, fontweight='normal', color='#5A5A5A')

                advanced_centering_success = True

        except Exception as e:
            print(f"進階置中渲染失敗: {e}")

        # 如果置中方法失敗，使用基本方法（主文字靠右，單位靠左）
        if not advanced_centering_success:
            ax.text(0.5, y_pos, main_text, ha='right', va='top', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color='black')
            ax.text(0.5, y_pos, unit_text, ha='left', va='top', transform=ax.transAxes,
                    fontsize=12, fontweight='normal', color='gray')

        # 將 Y 軸刻度顯示在右側
        ax.yaxis.tick_right()
        # ax.set_ylabel(y_axis_label, fontsize=16, labelpad=30, rotation=270, color="#868686")
        
        ax.tick_params(axis='y', right=False, labelright=True, labelsize=12, labelcolor="#5A5A5A")

        # X 軸左右預留空間
        ax.set_xlim(-0.4, 0.4)

        # 加入 Y 軸虛線格線
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # 計算最大最小值並設定 Y 軸範圍
        max_val = max(value_pre, value_pos) if metric_data else 1
        min_val = min(value_pre, value_pos) if metric_data else 0
        ax.set_ylim(bottom=min(0, min_val * 1.2), top=max_val * 1.25)

        # 加入 bar 上方數值（目前註解掉）
        # for bar in bars1 + bars2:
        #     height = bar.get_height()
        #     ax.annotate(f'{height:.2f}',
        #                 xy=(bar.get_x() + bar.get_width() / 2, height),
        #                 xytext=(0, 3 if height >= 0 else -15),
        #                 textcoords="offset points",
        #                 ha='center', va='bottom' if height >= 0 else 'top',
        #                 fontsize=10)

    # 多餘的子圖設為不可見
    for i in range(num_metrics, len(axs)):
        axs[i].set_visible(False)

    # 主標題與整體版面微調
    # fig.suptitle(title, fontsize=20, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ==========================================================
# 以下為一個可以用來測試上述函式的範例
# ==========================================================
if __name__ == '__main__':
    performance_data = {
        "Accuracy": {'pre': 0.62, 'pos': 0.73},
        "Efficiency Ratio": {'pre': 0.81, 'pos': 0.85},
        "TTK (ms)": {'pre': 60.11, 'pos': 51.88}
    }
    
    # 建立一個自訂標籤的對照表
    custom_label_texts = {
        "Accuracy": ("準確度", "(%)"),
        "Efficiency Ratio": ("效率指標", ""), # 如果沒有單位，可以留空
        "TTK (ms)": ("擊殺時間", "(ms)")
    }

    # 呼叫函式時，傳入這個對照表
    plot_performance_comparison(
        data=performance_data,
        title="整體效能比較 (實驗前 vs. 實驗後)",
        custom_texts=custom_label_texts # 在這裡傳入
    )
# %%

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional

def plot_grouped_bar_chart(
    data: List[Dict],
    y_axis_label: str = "Performance",
    output_path: str = "grouped_bar_chart.png",
    fixed_x_range: Tuple[float, float] = (0, 214), # Your fixed X-axis range
    fixed_bar_width: float = 9.0 # Your fixed width for a single bar
):
    """
    Creates a bar chart with a fixed X-axis range and fixed bar widths.
    The spacing between groups is calculated dynamically to fit.
    """
    # --- 1. Setup ---
        
    if not data:
        print("❌ Error: No data provided.")
        return
    
    group_names = [item['group_name'] for item in data]
    num_groups = len(group_names)

    fig, ax = plt.subplots(figsize=(18, 5)) # Keep a consistent output image size

    # --- 2. Bar Positioning (NEW DYNAMIC SPACING LOGIC) ---
    x_min, x_max = fixed_x_range
    ax.set_xlim(x_min, x_max) # Apply the fixed range

    # Calculate the total width occupied by all bars
    total_bars_width = num_groups * 2 * fixed_bar_width 
    
    # Check if the bars can physically fit in the given range
    if total_bars_width >= (x_max - x_min):
        print(f"❌ Error: The fixed bar widths ({total_bars_width}) exceed the fixed X-axis range ({x_max - x_min}).")
        print("  Please reduce the number of groups or the fixed_bar_width.")
        return

    # Calculate the remaining space to be used for gaps
    total_gap_space = (x_max - x_min) - total_bars_width
    
    # Divide the gap space evenly. There is always one more gap than the number of groups.
    # (e.g., 2 groups have 3 gaps: margin-group1-gap-group2-margin)
    gap_size = total_gap_space / (num_groups + 1)
    
    # Calculate the positions for each group's center
    offset = fixed_bar_width / 2
    x_positions = []
    for i in range(num_groups):
        # The center of group i is the starting margin + previous bars and gaps + half of the current group's width
        group_center = gap_size * (i + 1) + fixed_bar_width * (2 * i) + fixed_bar_width
        x_positions.append(group_center)
    # --- 3. Y-Axis Automatic Scaling (NEW LOGIC) ---
    
    # First, find the min and max values across ALL data to determine the scale
    all_values = []
    for item in data:
        all_values.append(item.get('pre', 0))
        all_values.append(item.get('post', 0))
    
    min_val = min(all_values)
    max_val = max(all_values)
    
    # Set a preliminary Y-axis range with some padding
    ax.set_ylim(bottom=min(0, min_val * 1.2), top=max_val * 1.2)

    # Use MaxNLocator to automatically find 5 nice intervals (which creates 6 tick marks/lines)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    
    # --- 4. Styling the Chart ---
    # ax.set_ylabel(y_axis_label, fontsize=12)
    # ax.set_ylim(bottom=0, top=100) 

    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_names, fontsize=44, color='#757575')
    ax.tick_params(axis='x', pad=15, length=0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('#5A5A5A')
    ax.spines['bottom'].set_linewidth(2)

    ax.yaxis.grid(True, linestyle='-', color='#5A5A5A', zorder=0, alpha=0.7)
    # ax.set_yticklabels(fontsize=20, color='#757575')
    # 將 Y 軸刻度顯示在右側
    ax.yaxis.tick_right()
    # Add a thicker, solid line specifically at the Y=0 position to emphasize it
    ax.axhline(y=0, color='#5A5A5A', linewidth=4, zorder=0, alpha=0.7)
    
    # --- 3. Draw Bars for Each Group ---
    for i, item in enumerate(data):
        ax.bar(
            x_positions[i] - offset,
            item['pre'],
            fixed_bar_width, # Use the fixed bar width
            edgecolor=item['color'],
            facecolor='none',
            hatch=item.get('hatch', '///'),
            linewidth=0.5,
            zorder=3 
        )
        ax.bar(
            x_positions[i] + offset,
            item['post'],
            fixed_bar_width, # Use the fixed bar width
            color=item['color'],
            zorder=3 
        )

    
    ax.tick_params(axis='y', right=False, labelright=True, labelsize=36, labelcolor="#5A5A5A")

    # --- 5. Save and Show ---
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.show()
    plt.close(fig)
    print(f"Grouped bar chart with fixed spacing saved to '{output_path}'")
if __name__ == '__main__':
    # Example with 4 groups
    # data_4_groups = [
    #     {'group_name': 'Mouse A', 'pre': 0.42, 'post': 0.36, 'color': '#CC0040', 'hatch': '\\\\'},
    #     {'group_name': 'EC2', 'pre': 0.44, 'post': 0.37, 'color': '#000000', 'hatch': '\\\\'},
    #     # {'group_name': 'Mouse C', 'pre': 75, 'post': 60, 'color': '#F1A012', 'hatch': '\\\\'},
    #     # {'group_name': 'Mouse D', 'pre': 80, 'post': 75, 'color': '#7A4EDF', 'hatch': '\\\\'}
    # ]
    # data_4_groups = [
    #     {'group_name': 'Mouse A', 'pre': 45, 'post': 49, 'color': '#CC0040', 'hatch': '\\\\'},
    #     {'group_name': 'EC2', 'pre': 46, 'post': 48, 'color': '#000000', 'hatch': '\\\\'},
    #     # {'group_name': 'Mouse C', 'pre': 75, 'post': 60, 'color': '#F1A012', 'hatch': '\\\\'},
    #     # {'group_name': 'Mouse D', 'pre': 80, 'post': 75, 'color': '#7A4EDF', 'hatch': '\\\\'}
    # ]
    mouse1 = 'EC2'
    mouse2 = 'EC1'
    # TTK
    data_4_groups = [
        {'group_name': mouse1, 'pre': 0.39, 'post': 0.475, 'color': '#CC0040', 'hatch': '\\\\'},
        {'group_name': mouse2, 'pre': 0.368, 'post': 0.393, 'color': '#000000', 'hatch': '\\\\'},
        # {'group_name': 'Mouse C', 'pre': 75, 'post': 60, 'color': '#F1A012', 'hatch': '\\\\'},
        # {'group_name': 'Mouse D', 'pre': 80, 'post': 75, 'color': '#7A4EDF', 'hatch': '\\\\'}
    ]
    
    # The function will automatically calculate the spacing to fit 4 groups
    # with bars of width 9 into the 0-214 range.
    plot_grouped_bar_chart(
        data=data_4_groups,
        fixed_x_range=(0, 214),
        fixed_bar_width=9.0,
        output_path="TTK.png"
    )
    
    ## Shotcount
    data_4_groups = [
        {'group_name': mouse1, 'pre': 38, 'post': 31, 'color': '#CC0040', 'hatch': '\\\\'},
        {'group_name': mouse2, 'pre': 37, 'post': 32, 'color': '#000000', 'hatch': '\\\\'},
        # {'group_name': 'Mouse C', 'pre': 75, 'post': 60, 'color': '#F1A012', 'hatch': '\\\\'},
        # {'group_name': 'Mouse D', 'pre': 80, 'post': 75, 'color': '#7A4EDF', 'hatch': '\\\\'}
    ]
    
    # The function will automatically calculate the spacing to fit 4 groups
    # with bars of width 9 into the 0-214 range.
    plot_grouped_bar_chart(
        data=data_4_groups,
        fixed_x_range=(0, 214),
        fixed_bar_width=9.0,
        output_path="ShotCounts.png"
    )
    
    ## accuracy
    data_4_groups = [
        {'group_name': mouse1, 'pre': 0.891, 'post': 0.667, 'color': '#CC0040', 'hatch': '\\\\'},
        {'group_name': mouse2, 'pre': 0.894, 'post': 0.875, 'color': '#000000', 'hatch': '\\\\'},
        # {'group_name': 'Mouse C', 'pre': 75, 'post': 60, 'color': '#F1A012', 'hatch': '\\\\'},
        # {'group_name': 'Mouse D', 'pre': 80, 'post': 75, 'color': '#7A4EDF', 'hatch': '\\\\'}
    ]
    
    # The function will automatically calculate the spacing to fit 4 groups
    # with bars of width 9 into the 0-214 range.
    plot_grouped_bar_chart(
        data=data_4_groups,
        fixed_x_range=(0, 214),
        fixed_bar_width=9.0,
        output_path="Accuracy.png"
    )
# %%

import matplotlib.pyplot as plt
import numpy as np

def create_radar_chart(labels, data_series, output_path):
    """
    Creates and saves a final, styled radar chart.
    Y-axis labels are now guaranteed to be on the top layer.
    """
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 110)
    ax.grid(False)

    # --- Set Y-axis Ticks and Labels ---
    y_tick_positions = [20, 40, 60, 80, 100]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels([str(y) for y in y_tick_positions], color="#959595", size=8)
    ax.set_rlabel_position(0)
    
    # --- THIS IS THE FIX (Part 1) ---
    # Set a high zorder on the Y-axis labels to bring them to the front.
    for label in ax.get_yticklabels():
        label.set_verticalalignment('top')
        label.set_zorder(10) # A high number ensures it's on top of data fills
    # --------------------------------

    # --- Manually draw grids and axes (with low zorder) ---
    for y_value in y_tick_positions:
        grid_points = [y_value] * len(angles)
        if y_value == 100:
            ax.plot(angles, grid_points, color='#E5E5E5', linestyle='-', linewidth=1.5, zorder=1)
        else:
            ax.plot(angles, grid_points, color='#E5E5E5', linestyle='-', linewidth=0.5, zorder=1)
        
        ax.fill(angles, grid_points, color='#F2F2F2', zorder=-10) # zorder places it in the background

    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 100], color='#E5E5E5', linestyle='-', linewidth=1, zorder=1)

    # --- Plot the actual data series ---
    for series in data_series:
        values = series['values'] + series['values'][:1]
        
        # --- THIS IS THE FIX (Part 2) ---
        # Set a higher zorder for the data line, and a lower zorder for the fill
        ax.plot(angles, values, color=series['color'], linewidth=1, label=series['label'], zorder=3)
        # ax.fill(angles, values, color=series['color'], alpha=0.25, zorder=2)
        # --------------------------------

    # --- Final appearance settings ---
    ax.set_thetagrids(np.degrees(angles[:-1]), [])
    ax.spines['polar'].set_visible(False)

    # --- Save the figure ---
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
    plt.close(fig)
# --- Main execution block to generate the chart ---
if __name__ == '__main__':
    
    pentagon_labels = ['Focus', 'Accuracy', 'Speed', 'Stamina', 'Control']

    pre_data = {
        'label': 'Pre',
        'values': [80, 75, 70, 85, 80],
        'color': '#000000' # Black
    }
    
    post_data = {
        'label': 'Post',
        'values': [83, 80, 60, 75, 90],
        'color': '#CC0040' # Red
    }

    create_radar_chart(
        labels=pentagon_labels, 
        data_series=[pre_data, post_data], 
        output_path='my_pentagonal_radar_chart.png'
    )
# %%

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import math # Make sure math is imported

def plot_pixel_perfect_bars(
    data: List[Dict],
    bar_width_px: int = 35, # The fixed width of a single bar in pixels
    y_axis_label: str = "Performance",
    output_path: str = "pixel_perfect_chart.png",
    dpi: int = 300
):
    """
    Creates a bar chart with fixed bar widths and precise pixel spacing between groups.
    The total width of the chart is dynamic based on the number of groups.
    """
    # --- 1. Validate Input ---
    num_groups = len(data)
    if num_groups > 4:
        print(f"❌ Error: This layout supports a maximum of 4 data groups. You provided {num_groups}.")
        return
    if num_groups == 0:
        print("❌ Error: No data provided.")
        return

    # --- 2. Define Layout Spacing in Pixels based on Rules ---
    inner_spacing_px = 0
    if num_groups == 2:
        inner_spacing_px = 44
    elif num_groups == 3:
        inner_spacing_px = 41
    elif num_groups == 4:
        inner_spacing_px = 32
    
    # Let's assume the outer spacing (left/right margins) is equal to the inner spacing for a balanced look.
    outer_spacing_px = inner_spacing_px if num_groups > 1 else 44 # Use a default margin for a single group

    # --- 3. Calculate All Pixel Dimensions ---
    # The space for one group (e.g., a "Pre" and "Post" bar). Let's make them touch.
    group_block_width_px = bar_width_px * 2 
    
    # Calculate the total width needed for the figure
    total_bar_widths = num_groups * group_block_width_px
    total_inner_spacing = max(0, num_groups - 1) * inner_spacing_px
    total_outer_spacing = 2 * outer_spacing_px
    total_figure_width_px = total_bar_widths + total_inner_spacing + total_outer_spacing

    # Convert total pixel width to inches for figsize
    figure_width_inches = total_figure_width_px / dpi
    figure_height_inches = 4 # A fixed height

    # --- 4. Create the Plot ---
    fig, ax = plt.subplots(figsize=(figure_width_inches, figure_height_inches), dpi=dpi)
    
    # Set the x-axis limits to match our pixel calculations
    ax.set_xlim(0, total_figure_width_px)
    
    # --- 5. Draw Bars at Precise Positions ---
    x_tick_positions = []
    current_x = outer_spacing_px # Start after the left margin
    
    for item in data:
        # Calculate the center of the current group block
        group_center = current_x + (group_block_width_px / 2)
        x_tick_positions.append(group_center)
        
        # Calculate positions for the two bars within the group
        pre_bar_x = group_center - (bar_width_px / 2)
        post_bar_x = group_center + (bar_width_px / 2)

        # Draw the "Pre" bar (hatched)
        ax.bar(pre_bar_x, item['pre'], bar_width_px, align='center', edgecolor=item['color'], facecolor='none', hatch=item.get('hatch', '///'), linewidth=1.5)
        # Draw the "Post" bar (solid)
        ax.bar(post_bar_x, item['post'], bar_width_px, align='center', color=item['color'])
        
        # Move the starting point for the next group
        current_x += group_block_width_px + inner_spacing_px
        
    # --- 6. Styling the Chart (similar to before) ---
    ax.set_ylabel(y_axis_label, fontsize=12)
    ax.set_ylim(bottom=0, top=100)
    
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([item['group_name'] for item in data], fontsize=14, color='grey')
    ax.tick_params(axis='x', length=0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('lightgray')

    ax.yaxis.grid(True, linestyle='--', color='lightgray', zorder=-10)
    ax.tick_params(axis='y', length=0)

    # --- 7. Save and Show ---
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, transparent=True)
    plt.show()
    plt.close(fig)
    print(f"Pixel-perfect bar chart saved to '{output_path}'")
if __name__ == '__main__':
    # 1. Define the data in the new list format
    performance_data = [
        {
            'group_name': 'Mouse 1',
            'pre': 85,
            'post': 40,
            'color': '#CC0040',
            'hatch': '///'
        },
        {
            'group_name': 'Mouse 2',
            'pre': 95,
            'post': 50,
            'color': '#000000',
            'hatch': '\\\\\\'
        },
        {
            'group_name': 'Mouse 3',
            'pre': 85,
            'post': 40,
            'color': '#F1A012',
            'hatch': '///'
        },
        {
            'group_name': 'Mouse 4',
            'pre': 95,
            'post': 50,
            'color': '#7A4EDF',
            'hatch': '\\\\\\'
        }
    ]

    # 2. Call the new function
    plot_pixel_perfect_bars(
        data=performance_data, 
        y_axis_label="TTK (ms)", # Example label
        output_path="performance_comparison.png"
    )       
    
    
