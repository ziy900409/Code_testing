# -*- coding: utf-8 -*-
"""
Created on Sun May 25 21:10:33 2025

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np



def plot_median_freq_slope_comparison(group1: dict, group2: dict,
                                      title: str = "Median Frequency Slope Comparison",
                                      ylabel: str = "Fatigue Index",
                                      turn: bool = True,
                                      selected_keys: list = None,
                                      label_list: list = None,
                                      show_values: bool = True):
    """
    繪製兩組 Median Frequency Slope 的比較柱狀圖（可指定 key，支援絕對值顯示）

    Parameters:
    - group1, group2: dict，key 為肌肉名稱，value 為 slope 值
    - title: 圖表標題
    - selected_keys: list[str]，若提供，只繪製這些 key
    - show_values: 是否在柱子上顯示數值（會顯示絕對值）
    """
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
    x = np.arange(len(keys))
    # width = 0.35
    
    # x = np.arange(len(keys)) * 1.5  # 放大 x 軸間距
    # width = 0.3  # 稍微窄一點避免重疊

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 6))
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
    # ax.set_title(title, fontsize=16)
    ax.text(0.5, -0.15, title, fontsize=16, ha='center', va='top', transform=ax.transAxes)

    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.spines['left'].set_visible(False)   # 左框線
    ax.spines['right'].set_visible(False)  # 右框線
    ax.spines['top'].set_visible(False)    # 上框線
    ax.spines['bottom'].set_visible(False) # 下框線（依需要）
    
    ax.axhline(y=0, color='#3B3B3B', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, ha='center')
    ax.legend()
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