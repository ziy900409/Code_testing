# -*- coding: utf-8 -*-
"""
Created on Sun May 25 21:10:33 2025

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np



def plot_median_freq_slope_comparison(group1: dict, group2: dict,
                                      title="Median Frequency Slope Comparison",
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

    values1 = [abs(group1[k]) for k in keys]
    values2 = [abs(group2[k]) for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 6))
    bars1 = ax.bar(x - width/2, values1, width, label=label_list[0], color='dodgerblue')
    bars2 = ax.bar(x + width/2, values2, width, label=label_list[1], color='orange')

    ax.set_ylabel('Fatigue Index')
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha='right')
    ax.legend()
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

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