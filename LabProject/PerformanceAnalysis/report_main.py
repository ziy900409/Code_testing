# -*- coding: utf-8 -*-
"""
Created on Sat May 24 16:33:25 2025

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
import io

# --- 範例資料 ---
mouse_data = {
    "Mouse_A": {
        "pre_test": {"accuracy": 0.85, "reaction_time": 0.42, "movement_efficiency": 0.88, "trajectory_stability": 0.77},
        "post_test": {"accuracy": 0.82, "reaction_time": 0.45, "movement_efficiency": 0.85, "trajectory_stability": 0.75}
    },
    "Mouse_B": {
        "pre_test": {"accuracy": 0.78, "reaction_time": 0.50, "movement_efficiency": 0.80, "trajectory_stability": 0.70},
        "post_test": {"accuracy": 0.80, "reaction_time": 0.48, "movement_efficiency": 0.82, "trajectory_stability": 0.73}
    },
    "Mouse_C": {
        "pre_test": {"accuracy": 0.90, "reaction_time": 0.38, "movement_efficiency": 0.92, "trajectory_stability": 0.85},
        "post_test": {"accuracy": 0.88, "reaction_time": 0.40, "movement_efficiency": 0.90, "trajectory_stability": 0.83}
    },
    "Mouse_D": {
        "pre_test": {"accuracy": 0.82, "reaction_time": 0.47, "movement_efficiency": 0.75, "trajectory_stability": 0.68},
        "post_test": {"accuracy": 0.75, "reaction_time": 0.52, "movement_efficiency": 0.70, "trajectory_stability": 0.65}
    }
}

metrics_labels = ["accuracy", "reaction_time", "movement_efficiency", "trajectory_stability"]
metrics_labels_cn = ["準確度", "反應時間 (s)", "移動效率", "軌跡穩定性"] # 中文標籤供圖表使用

# --- Helper Functions for Calculations ---
def calculate_averages_and_retention(data):
    processed_data = {}
    for mouse, tests in data.items():
        averages = {}
        retention_ratios = []
        for metric in metrics_labels:
            pre_val = tests["pre_test"][metric]
            post_val = tests["post_test"][metric]
            averages[metric] = (pre_val + post_val) / 2
            if pre_val != 0: # Avoid division by zero
                retention_ratios.append(post_val / pre_val)
            else:
                retention_ratios.append(0) # Or some other indicator for no pre-test data

        averages["retention_rate"] = np.mean(retention_ratios) if retention_ratios else 0
        
        pre_total_score = np.mean([tests["pre_test"][m] for m in metrics_labels])
        post_total_score = np.mean([tests["post_test"][m] for m in metrics_labels])
        
        processed_data[mouse] = {
            "averages": averages,
            "pre_total_score": pre_total_score,
            "post_total_score": post_total_score
        }
    return processed_data

# --- Plotting Functions ---
def plot_radar_chart(processed_data, prs):
    labels = metrics_labels + ["retention_rate"]
    labels_cn_radar = metrics_labels_cn + ["留存率"]
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Complete the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    mouse_names = list(processed_data.keys())
    for i, mouse_name in enumerate(mouse_names):
        values = [processed_data[mouse_name]["averages"][label] for label in labels]
        values += values[:1] # Complete the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=mouse_name)
        ax.fill(angles, values, alpha=0.25)

    ax.set_yticklabels([]) # Hide default y-axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_cn_radar, fontsize=10)
    
    # Custom y-axis grid and labels
    y_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    for tick in y_ticks:
        ax.plot(angles, [tick] * len(angles), linestyle='--', color='grey', linewidth=0.75)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(tick) for tick in y_ticks], fontsize=8, color='grey')


    plt.title("滑鼠綜合表現雷達圖 (前後測平均值)", size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    img_stream.seek(0)
    return img_stream

def plot_overall_scores_bar_chart(processed_data, prs):
    mouse_names = list(processed_data.keys())
    pre_scores = [processed_data[mouse]["pre_total_score"] for mouse in mouse_names]
    post_scores = [processed_data[mouse]["post_total_score"] for mouse in mouse_names]

    x = np.arange(len(mouse_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pre_scores, width, label='前測總分 (平均四指標)')
    rects2 = ax.bar(x + width/2, post_scores, width, label='後測總分 (平均四指標)')

    ax.set_ylabel('平均分數')
    ax.set_title('各滑鼠前後測總分比較', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(mouse_names, rotation=45, ha="right")
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    
    ax.set_ylim(0, max(max(pre_scores, default=0), max(post_scores, default=0)) * 1.2) # Adjust y-limit for labels

    plt.tight_layout()
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', dpi=300)
    plt.close(fig)
    img_stream.seek(0)
    return img_stream

def plot_accuracy_change_line_chart(mouse_name, pre_accuracy, post_accuracy, prs):
    fig, ax = plt.subplots(figsize=(8, 5))
    time_points = ['前測 (Pre-test)', '後測 (Post-test)']
    accuracies = [pre_accuracy, post_accuracy]

    ax.plot(time_points, accuracies, marker='o', linestyle='-', color='b')
    
    # Add value labels
    for i, acc in enumerate(accuracies):
        ax.text(time_points[i], acc + 0.01, f'{acc:.2f}', ha='center', va='bottom')

    ax.set_ylabel('準確度 (Accuracy)')
    ax.set_title(f'{mouse_name} - 準確度前後測變化', fontsize=14)
    ax.set_ylim(min(accuracies) * 0.9, max(accuracies) * 1.1 if max(accuracies) > 0 else 0.1) # Dynamic Y limit
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', dpi=300)
    plt.close(fig)
    img_stream.seek(0)
    return img_stream

def plot_individual_metrics_bar_chart(mouse_name, pre_metrics, post_metrics, prs):
    x = np.arange(len(metrics_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, [pre_metrics[m] for m in metrics_labels], width, label='前測')
    rects2 = ax.bar(x + width/2, [post_metrics[m] for m in metrics_labels], width, label='後測')

    ax.set_ylabel('指標數值')
    ax.set_title(f'{mouse_name} - 各項指標前後測比較', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels_cn, rotation=45, ha="right")
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    
    all_values = [pre_metrics[m] for m in metrics_labels] + [post_metrics[m] for m in metrics_labels]
    ax.set_ylim(0, max(all_values, default=0) * 1.2)


    plt.tight_layout()
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png', dpi=300)
    plt.close(fig)
    img_stream.seek(0)
    return img_stream

# --- Main Script to Generate PowerPoint ---
def generate_report(data):
    prs = Presentation()
    processed_overall_data = calculate_averages_and_retention(data)

    # --- Slide 1: Summary Report ---
    slide_layout = prs.slide_layouts[5] # Blank layout
    slide = prs.slides.add_slide(slide_layout)
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.5))
    title.text_frame.text = "滑鼠比較總結報告"
    title.text_frame.paragraphs[0].font.size = Pt(24)
    title.text_frame.paragraphs[0].font.bold = True

    # Add Radar Chart
    radar_img_stream = plot_radar_chart(processed_overall_data, prs)
    slide.shapes.add_picture(radar_img_stream, Inches(0.5), Inches(1.0), width=Inches(4.5))
    
    # Add Overall Scores Bar Chart
    overall_bar_img_stream = plot_overall_scores_bar_chart(processed_overall_data, prs)
    slide.shapes.add_picture(overall_bar_img_stream, Inches(5.0), Inches(1.0), width=Inches(4.8))


    # --- Slides 2-5: Individual Mouse Analysis ---
    for i, (mouse_name, tests) in enumerate(data.items()):
        slide = prs.slides.add_slide(slide_layout)
        
        title = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.5))
        title.text_frame.text = f"單一滑鼠分析：{mouse_name}"
        title.text_frame.paragraphs[0].font.size = Pt(24)
        title.text_frame.paragraphs[0].font.bold = True

        # Add Accuracy Change Line Chart
        pre_accuracy = tests["pre_test"]["accuracy"]
        post_accuracy = tests["post_test"]["accuracy"]
        accuracy_line_img_stream = plot_accuracy_change_line_chart(mouse_name, pre_accuracy, post_accuracy, prs)
        pic = slide.shapes.add_picture(accuracy_line_img_stream, Inches(0.5), Inches(1.0), width=Inches(4.5))
        
        # Add Individual Metrics Bar Chart
        individual_bar_img_stream = plot_individual_metrics_bar_chart(mouse_name, tests["pre_test"], tests["post_test"], prs)
        pic2 = slide.shapes.add_picture(individual_bar_img_stream, Inches(5.0), Inches(1.0), width=Inches(4.8))
        
        # Add Text Description
        accuracy_change_percent = ((post_accuracy - pre_accuracy) / pre_accuracy) * 100 if pre_accuracy != 0 else 0
        movement_efficiency_change = tests["post_test"]["movement_efficiency"] - tests["pre_test"]["movement_efficiency"]
        
        description_text = f"分析摘要：\n"
        description_text += f"- {mouse_name} 在疲勞後（後測）準確度變化 {accuracy_change_percent:.1f}%。\n"
        
        if movement_efficiency_change > 0.01:
            description_text += f"- 移動效率提升 {movement_efficiency_change:.2f}。"
        elif movement_efficiency_change < -0.01:
            description_text += f"- 移動效率降低 {abs(movement_efficiency_change):.2f}。"
        else:
            description_text += f"- 移動效率表現相對穩定。"
            
        # Add reaction time observation
        reaction_time_change = tests["post_test"]["reaction_time"] - tests["pre_test"]["reaction_time"]
        if reaction_time_change > 0.01: # Slower
            description_text += f"\n- 反應時間減慢 {reaction_time_change:.2f} 秒。"
        elif reaction_time_change < -0.01: # Faster
            description_text += f"\n- 反應時間加快 {abs(reaction_time_change):.2f} 秒。"
        else:
            description_text += f"\n- 反應時間變化不大。"

        textbox = slide.shapes.add_textbox(Inches(0.5), Inches(4.5), Inches(9), Inches(2.5)) # Adjusted position and size
        tf = textbox.text_frame
        tf.word_wrap = True
        p = tf.add_paragraph()
        p.text = description_text
        p.font.size = Pt(12)

    # Save the presentation
    file_path = "Mouse_Comparison_Report.pptx"
    prs.save(file_path)
    print(f"簡報已儲存至：{file_path}")

# --- Run the report generation ---
if __name__ == "__main__":
    # 設定 Matplotlib 使用支援中文的字體 (例如：Microsoft JhengHei)
    # 這部分可能需要根據您的系統環境調整
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Heiti TC', 'sans-serif'] # 優先使用微軟正黑體
        plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
    except Exception as e:
        print(f"設定中文字體時發生錯誤: {e}. 圖表中的中文可能無法正確顯示。")
        print("請確保您的系統已安裝 'Microsoft JhengHei' 或其他支援中文的字體，並修改 plt.rcParams['font.sans-serif'] 設定。")

    generate_report(mouse_data)