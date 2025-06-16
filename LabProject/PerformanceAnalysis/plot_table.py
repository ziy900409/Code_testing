# -*- coding: utf-8 -*-
"""
Created on Sun May 25 21:10:33 2025

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
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
                   edgecolor="#CC0040",   # 用斜線顏色
                   facecolor='none',
                   hatch='\\\\',
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
    pre_post_colors: List[str] = ['#CC0040', '#CC0040'],
    custom_texts: Optional[Dict[str, Tuple[str, str]]] = None
):
    # 設定中文字型與負號正確顯示
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False

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

        # 畫 pre 的 bar（外框）
        bars1 = ax.bar(x - offset,
                       [value_pre],
                       width,
                       label=pre_post_labels[0],
                       edgecolor=pre_post_colors[0],
                       facecolor='none',
                       hatch='\\\\',
                       linewidth=0.1)

        # 畫 pos 的 bar（實心）
        bars2 = ax.bar(x + offset,
                       [value_pos],
                       width,
                       label=pre_post_labels[1],
                       color=pre_post_colors[1])

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
        
        ax.tick_params(axis='y', right=False, labelright=True, labelsize=9, labelcolor="#5A5A5A")

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
# if __name__ == '__main__':
#     performance_data = {
#         "Accuracy": {'pre': 0.62, 'pos': 0.73},
#         "Efficiency Ratio": {'pre': 0.81, 'pos': 0.85},
#         "TTK (ms)": {'pre': 60.11, 'pos': 51.88}
#     }
    
#     # 建立一個自訂標籤的對照表
#     custom_label_texts = {
#         "Accuracy": ("準確度", "(%)"),
#         "Efficiency Ratio": ("效率指標", ""), # 如果沒有單位，可以留空
#         "TTK (ms)": ("擊殺時間", "(ms)")
#     }

#     # 呼叫函式時，傳入這個對照表
#     plot_performance_comparison(
#         data=performance_data,
#         title="整體效能比較 (實驗前 vs. 實驗後)",
#         custom_texts=custom_label_texts # 在這裡傳入
#     )

