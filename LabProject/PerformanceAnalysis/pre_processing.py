# %%
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 09:28:13 2025

1. 找出所有的Z軸局部最小值
    1.1.  scipy.signal.argrelextrema 找出資料中的局部極值點（最大值或最小值）
            order = 5
    1.2. 小於 (平均值 - 0.05) 的點才視為局部最小值
            small than 0.05
    1.3. 加入最小 frame 間隔條件 or 兩Z軸局部最小值差異超過閾值
            min_frame_gap = 8, min_z_diff = 0.2

2. 計算
    2.1. 找出每一次目標擊殺的開槍數
        2.1.1. 以滑鼠點擊次數計算，使用Z軸局部最小值，如果兩次Z軸局部最小值的視角差
                小於某個閾值，則視為仍在瞄準同一個目標
        2.1.2. 找出完成擊殺的 frame 以及上一個視角大於閾值的視角位置
                data format
            	GroupID   Frames          Shot Count   Frame Start   Frame End   Frame Span
                -------   --------------  -----------  ------------  ----------  -----------
                1       [53.0, 71.0]          1           53.0         71.0        18.0
    2.2. 計算
        2.2.1. 指標
            o. 擊殺數, 命中率？
            a. Throughput (Mouse Travel Efficiency): 
            b. Mouse Speed (°/s): 找出整段時間內的最大值 or 平均值，單位換算成視角
            c. Initial Move Angle: 初始 5 個 frame 的移動方向與最終擊殺目標位置的視角差
            d. Full Path Time: 
            e. Reaction Time: 從這次目標擊殺到某個 frame 移動速度超過一個閾值 
                扣掉直接回中的反應時間
            i. 一槍擊殺的次數, 二槍, 三槍...
            j. 超過目標的次數， 還沒到目標就開槍的次數
            k. Mouse Travel Efficiency: idea path/real path
        2.2.2. 不同方向的計算: 全部方向綜合, 分四個方向 (四象限)

3. 畫出圖形
    3.1. 將"手部移動距離"換算成"視角移動"
        3.1.1. 因為不知道視角的絕對位置，所以視角使用手部位置的兩個frame換算視角差，
                並且使用累計視角以可視化視角位移
    3.2. 視覺化所有移動的 V-T 圖: 分成不同象限, 不同開槍次數
        3.2.1. std cloud

@author: Hsin.YH.Yang
"""
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"D:\BenQ_Project\gitgit\Code_testing\LabProject\PerformanceAnalysis")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from numpy.linalg import norm
from scipy.interpolate import interp1d

import Spider_function as func
plt.rcParams['font.sans-serif'] = ['Noto Sans TC']  # 改為你實際有的
plt.rcParams['axes.unicode_minus'] = False    # 避免座標軸負號亂碼

# %%

vicon2cortex = {'MOS1': 'M1',
                'MOS2': 'M2',
                'MOS3': 'M3',
                'MOS4': 'M4',
                'RHO': 'R.Shoulder',
                'RSHO': 'R.Shoulder',
                'RUEL': 'R.Elbow.Lat',
                'RUEM': 'R.Elbow.Med',
                'RUS': 'R.Wrist.Uln',
                'RRS': 'R.Wrist.Rad',
                'RTB1': 'R.Thumb1',
                'RTB2': 'R.Thumb2',
                'RTB3': 'R.Thumb3',
                'RID1': 'R.I.Finger1',
                'RID2': 'R.I.Finger2',
                'RID3': 'R.I.Finger3',
                'RMD1': 'R.M.Finger1',
                'RMD2': 'R.M.Finger2',
                'RMD3': 'R.M.Finger3',
                'RRG1': 'R.R.Finger1',                
                'RRG2': 'R.R.Finger2',
                'RLT1': 'R.P.Finger1',
                'RLT2': 'R.P.Finger2',                
                }

# === 參數設定 ===
DPI = 800
sensitivity = 1.0
yaw = 0.022  # CS2 預設值

# %%

data_path = r"D:/BenQ_Project/01_UR_lab/2024_11 Shanghai CS Major/1. Motion/Major_weight/S06/20241206/S06_SpiderShot_S1_1.c3d"

combine_dict, descriptions = func.read_c3d(data_path,
                                      prefix="S06", rename=vicon2cortex)

# %% analysis spider shot
"""
1. 找出所有的Z軸局部最小值
    1.1.  scipy.signal.argrelextrema 找出資料中的局部極值點（最大值或最小值）
            order = 5
    1.2. 小於 (平均值 - 0.05) 的點才視為局部最小值
            small than 0.05
    1.3. 加入最小 frame 間隔條件 or 兩Z軸局部最小值差異超過閾值
            min_frame_gap = 8, min_z_diff = 0.2
"""
def find_Zaxis_min(df, order=5, min_frame_gap=8,
                   min_z_diff=0.2, threshold=0.05,
                   show=True, showVel=True):
    """
    根據 Z 軸資料找出局部最小值點，並根據時間間隔與 Z 值變化篩選有效點

    paremeters：
        data: dict，包含 marker 資料的結構，例如 data["markers"]["R.I.Finger3"]
        order: int，局部最小值搜尋的視窗大小（預設為 5）
        min_frame_gap: int，兩個最小值點之間的最小 Frame 間距（預設為 8）
        min_z_diff: float，當 frame 間距不夠，Z 值需大於此差異才保留（預設 0.2）
        threshold: float，用來計算是否夠低（平均值 - threshold），預設為 0.05
        show: bool，是否繪製視覺化結果
    
    return：
        final_minima_idx: list，篩選後有效的 Z 軸局部最小值 index
        filtered_minima_data: 包含 Z 軸局部最小值點對應視角資訊的 DataFrame
    """
    # z_values = combine_dict["markers"]["R.I.Finger3"][:, 2]
    # 從指定 marker 中擷取 Z 軸資料（第3維）
    z_values = df["Z"].values
    # 計算 Z 軸平均值並定義 threshold 門檻
    z_mean = np.mean(z_values)
    threshold = z_mean - threshold
    
    # 使用 scipy 的 argrelextrema 尋找局部最小值
    local_minima_idx = argrelextrema(z_values, np.less, order=order)[0]
    
    # 篩選出 Z 值必須低於門檻的極小值
    filtered_minima_idx = [idx for idx in local_minima_idx if z_values[idx] < threshold]
    
    # === # 接著加入條件：兩點間距不能太短，或差異要夠大 ===
    final_minima_idx = []
    
    for idx in filtered_minima_idx:
        # 第一次直接加入
        if not final_minima_idx:
            final_minima_idx.append(idx)
            continue
        # 計算與上一個最小值的 frame 差
        last_idx = final_minima_idx[-1]
        frame_diff = idx - last_idx
    
        if frame_diff >= min_frame_gap:
            # 相隔夠遠，直接加入
            final_minima_idx.append(idx)  
        else:
            z_diff = abs(z_values[idx] - z_values[last_idx])
            if z_diff < min_z_diff:
                # 差異小 → 只保留 Z 值較小者
                if z_values[idx] < z_values[last_idx]:
                    final_minima_idx[-1] = idx  # 替換
                # 否則不做任何處理（保留原來的）
            else:
                # 雖然近，但差異夠大 → 一起保留
                final_minima_idx.append(idx) 
    # === 匯出包含視角資料的最小值 ===
    filtered_minima_data = pd.DataFrame({
        "Frame": final_minima_idx,
        "Z Value": df["Z"][final_minima_idx],
        "Yaw Angle (°)": df["cum_yaw_deg"].iloc[final_minima_idx].values,
        "Pitch Angle (°)": df["cum_pitch_deg"].iloc[final_minima_idx].values
    })
    print(filtered_minima_data)
    # filtered_minima_data.to_csv("Filtered_Local_Minima_Final_ViewAngle.csv", index=False)
    
    # 輸出篩選後的局部最小值數據
    # filtered_minima_data = pd.DataFrame({
    #     "Frame": filtered_minima_idx,
    #     "Z Value": z_values[filtered_minima_idx]
    # })
    
    # # 存成 CSV
    # filtered_minima_data.to_csv("Filtered_Local_Minima.csv", index=False)
    
    # 顯示篩選後的數據
    # print(filtered_minima_data.head())
    
    # 7️⃣ 取得篩選過的 Z 軸局部最小值對應的視角位置
    filtered_yaw = df.loc[final_minima_idx, "cum_yaw_deg"]
    filtered_pitch = df.loc[final_minima_idx, "cum_pitch_deg"]
    if show:
    # 繪製 Z 軸數據與篩選後的局部最小值
        plt.figure(figsize=(12, 5))
        plt.plot(z_values, label='Z-Axis', color='b', alpha=0.7)
        plt.scatter(final_minima_idx, z_values[final_minima_idx], color='r', label='Filtered Local Minima', zorder=3)
        plt.axhline(threshold, color='g', linestyle='--', label=f'Threshold ({threshold:.2f})')
        plt.xlabel("Frame")
        plt.ylabel("Z Value")
        plt.title("Filtered Local Minima of Z-Axis")
        plt.legend()
        plt.show()

    
    # 8️⃣ 若 show=True，畫出基本視角軌跡圖（紅色標出最小值）
    if show:
    # === 視角軌跡圖（逆時針旋轉視角等價於畫 pitch vs yaw）===
        plt.figure(figsize=(8, 8))
        plt.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"], c=df.index, cmap="viridis", alpha=0.7, s=5, label="View Angle Trajectory")
        plt.scatter(filtered_pitch, filtered_yaw, color="red", s=20, label="Final Local Minima", zorder=3)
        plt.colorbar(label="Frame Index")
        plt.xlabel("Pitch Angle (Vertical) °")
        plt.ylabel("Yaw Angle (Horizontal, Rotated) °")
        plt.title("視角軌跡轉換後的 Z 軸局部最小值分析")
        plt.legend()
        plt.show()
        
    # 9️⃣ 若 showVel=True，畫出以滑鼠速度作為顏色的視角軌跡圖
    if showVel:
        # === 取得局部最小值對應的視角資料 ===
        filtered_yaw   = df.loc[final_minima_idx, "cum_yaw_deg"]
        filtered_pitch = df.loc[final_minima_idx, "cum_pitch_deg"]

        # === 繪圖：以視角軌跡繪圖，使用滑鼠速度作為顏色依據 ===
        plt.figure(figsize=(8, 8))
        sc = plt.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"],
                         c=df["speed"], cmap="plasma", alpha=0.7, s=5,
                         label="View Angle Trajectory")
        # 標記篩選後的局部最小值
        plt.scatter(filtered_pitch, filtered_yaw, color="red", s=20,
                    label="Final Local Minima", zorder=3)

        # 以滑鼠速度 (mm/s) 作為 colorbar 的標示
        plt.colorbar(sc, label="Mouse Speed (mm/s)")
        plt.xlabel("Pitch Angle (Vertical) °")
        plt.ylabel("Yaw Angle (Horizontal, Rotated) °")
        plt.title("View Angle Trajectory Colored by Mouse Speed")
        plt.legend()
        plt.show()
    return filtered_minima_data
# %%
def ConverUnit2Angle(combine_dict, descriptions,
                     DPI=800, sens=1, yaw=0.022):
    """
    將食指的 3D Marker 資料（單位 mm）轉換為滑鼠視角變化（°），
    並計算其速度、繪製視覺化軌跡（選擇性）。

    parameters：
        data: dict，包含 marker 資料的結構，例如 data["markers"]["R.I.Finger3"]
        DPI: int，滑鼠解析度（預設為 800）
        sens: float，遊戲內靈敏度（預設為 1.0）
        yaw: float，遊戲內 yaw 係數（視角靈敏度）（預設 0.022）
        show: bool，是否畫出基本視角軌跡圖（True 則顯示）
        showVel: bool，是否畫出速度上色的視角軌跡圖（True 則顯示）

    return：
        df: 包含視角與速度資訊的 DataFrame
        
    """
    
    # 1️⃣ 將 marker 中的資料轉為 DataFrame，欄位為 X, Y, Z (單位 mm)
    df = pd.DataFrame(combine_dict["markers"]["R.I.Finger3"],
                      columns=["X", "Y", "Z"])
    # 2️⃣ 將 Y 軸反轉，以符合滑鼠視角的方向（向上為正）
    df["Y"] = -df["Y"]
    # 3️⃣ 計算相鄰 frame 的滑鼠移動距離（單位 mm）
    delta_x_mm = df["X"].diff().fillna(0)
    delta_y_mm = df["Y"].diff().fillna(0)
    
    
    # 4️⃣ 將滑鼠移動量換算成視角變化（°）
    # 公式：度數 = mm / 25.4（英吋） * DPI * sensitivity * yaw
    df["yaw_deg"] = delta_x_mm / 25.4 * DPI * sens * yaw     # 水平視角變化
    df["pitch_deg"] = delta_y_mm / 25.4 * DPI * sens * yaw   # 垂直視角變化
    
    df["cum_yaw_deg"] = df["yaw_deg"].cumsum()     # 累積水平視角（轉向左/右）
    df["cum_pitch_deg"] = df["pitch_deg"].cumsum() # 累積垂直視角（往上/下）
    
    
    # 6️⃣ 計算滑鼠速度（需知道取樣率）
    sampling_rate = descriptions["motion info"]["frame_rate"]
    dt = 1 / sampling_rate
    
    df["speed"] = np.sqrt((delta_x_mm / dt)**2 + (delta_y_mm / dt)**2)  # mm/s 實體速度
    df["yaw_speed_dps"]   = df["cum_yaw_deg"].diff().fillna(0) / dt     # 水平角速度 (°/s)
    df["pitch_speed_dps"] = df["cum_pitch_deg"].diff().fillna(0) / dt   # 垂直角速度 (°/s)
    df["angle_speed_dps"] = np.sqrt(df["yaw_speed_dps"]**2 + \
                                       df["pitch_speed_dps"]**2)  # 合成角速度
    
    return df
# %%

def findZminGroup(df, final_minima_idx, angle_merge_threshold=5, show=True):
    """
    根據 Z 軸局部最小值列表，計算其與前後點的視角差，並將視角變化小的點群視為同一擊殺動作。
    最後以視角空間視覺化結果並回傳每個擊殺群的 frame 資訊與初始移動角度。

    parameters:
        df : pd.DataFrame
            包含 X/Y/Z 與視角欄位（cum_yaw_deg, cum_pitch_deg）的完整資料。
        final_minima_idx : list[int]
            經過篩選後的 Z 軸局部最小值的 frame 編號。
        angle_merge_threshold = 5 
            若與前一點視角差 < 5°，視為同一組
        show : bool
            是否顯示視覺化圖。

    return:
        grouped_df : pd.DataFrame
            分群後的擊殺點資料，每組包含 frame 範圍與初始角度。
    """
    # === [1] 計算每個局部最小值與「前一個/後一個」的視角差（歐氏距離）===
    angle_diffs = []
    for i in range(len(final_minima_idx)):
        idx_curr = final_minima_idx[i]
    
        yaw_curr = df["cum_yaw_deg"].iloc[idx_curr]
        pitch_curr = df["cum_pitch_deg"].iloc[idx_curr]
    
        # 計算與前一個的角度差
        if i == 0:
            diff_prev = np.nan  # 沒有前一筆
        else:
            idx_prev = final_minima_idx[i - 1]
            yaw_prev = df["cum_yaw_deg"].iloc[idx_prev]
            pitch_prev = df["cum_pitch_deg"].iloc[idx_prev]
            diff_prev = np.linalg.norm([yaw_curr - yaw_prev, pitch_curr - pitch_prev])
    
        # 計算與後一個的角度差
        if i == len(final_minima_idx) - 1:
            diff_next = np.nan  # 沒有下一筆
        else:
            idx_next = final_minima_idx[i + 1]
            yaw_next = df["cum_yaw_deg"].iloc[idx_next]
            pitch_next = df["cum_pitch_deg"].iloc[idx_next]
            diff_next = np.linalg.norm([yaw_curr - yaw_next, pitch_curr - pitch_next])
        # 儲存至 dict
        angle_diffs.append({
            "Frame": idx_curr,
            "Z Value": df["Z"].iloc[idx_curr],
            "Angle_Diff_To_Prev_Minima (°)": diff_prev,
            "Angle_Diff_To_Next_Minima (°)": diff_next
        })
    # === [2] 輸出 DataFrame 並儲存為 CSV ===
    angle_diffs_df = pd.DataFrame(angle_diffs)
    # angle_diffs_df.to_csv("Z_Minima_ViewAngle_Comparison.csv", index=False)
    print(angle_diffs_df.head())
    
    # === [3] 將結果轉成 numpy array 便於處理 ===
    angle_array = angle_diffs_df[["Frame", 
                                  "Angle_Diff_To_Prev_Minima (°)", 
                                  "Angle_Diff_To_Next_Minima (°)"]].to_numpy()
    
    # === [4] 根據與前一筆的角度差進行「點群合併」 ===
    # 找出第一次擊發，以及最後一次擊發的位置，利用視角差當成閾值
    grouped_frames = []
   
    i = 0
    last_frame = None  # 記錄上一組的最後一筆
    
    while i < len(angle_array) - 1:
        current_group = []
        
        # 新群組要接上上一組結尾（若有）
        if last_frame is not None:
            current_group.append(last_frame)  # 接續上一組的結尾
            
        # 加入目前起點 frame
        current_group.append(angle_array[i][0])  # 加入目前起點 frame
        j = i + 1
    
        # 接下來的點若與前一點差值 < 閾值，繼續合併    
        while j < len(angle_array):
            angle_diff = angle_array[j][1]  # Angle_Diff_To_Prev_Minima (°)
        
            if pd.isna(angle_diff) or angle_diff >= angle_merge_threshold:
                break
        
            current_group.append(angle_array[j][0])
            j += 1
            
        # 若點數大於 1，視為有效群組
        if len(current_group) > 1:
            grouped_frames.append(current_group)
            last_frame = current_group[-1]  # 更新本輪最後 frame
        else:
            last_frame = angle_array[i][0]  # 還是更新這筆（即使沒成群）
    
        i = j  # 繼續從下一個起點開始
    # === [5] 建立分群結果的 DataFrame ===
    grouped_df = pd.DataFrame({
        "Group ID": list(range(1, len(grouped_frames) + 1)),
        "Frames": grouped_frames,
        "Shot Count": [len(g) - 1 for g in grouped_frames],
        "Frame Start": [min(g) for g in grouped_frames],
        "Frame End": [max(g) for g in grouped_frames],
    })
    grouped_df["Frame Span"] = grouped_df["Frame End"] - grouped_df["Frame Start"]
    
    # === [6] 計算每組起始方向與目標方向的夾角 ===
    initial_angles = []
    
    for _, row in grouped_df.iterrows():
        start_idx = int(row["Frame Start"])
        end_idx = int(row["Frame End"])
    
        # === 目標向量：start → end ===
        goal_vec = np.array([
            df["X"].iloc[end_idx] - df["X"].iloc[start_idx],
            df["Y"].iloc[end_idx] - df["Y"].iloc[start_idx]
        ])
    
        max_len = end_idx - start_idx
        found_valid = False
        current_len = 5  # 起始向量初始長度
    
        while current_len <= max_len:
            move_range = df.iloc[start_idx : start_idx + current_len]
    
            if len(move_range) < 2:
                break  # 無法構成向量
    
            # === 計算初始向量（首尾） ===
            init_vec = np.array([
                move_range["X"].iloc[-1] - move_range["X"].iloc[0],
                move_range["Y"].iloc[-1] - move_range["Y"].iloc[0]
            ])
    
            # === 若 init_vec 或 goal_vec 長度為 0，略過 ===
            if norm(init_vec) == 0 or norm(goal_vec) == 0:
                break
    
            # === 計算夾角 ===
            cos_theta = np.dot(init_vec, goal_vec) / (norm(init_vec) * norm(goal_vec))
    
            # (沒有設置)若夾角小於 90 度，接受此向量 
            if cos_theta:
                angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
                initial_angles.append(angle_deg)
                found_valid = True
                break
    
            # 否則繼續加長
            current_len += 1
    
        # 如果找不到符合條件的向量（全部都 > 45°）
        if not found_valid:
            initial_angles.append(np.nan)
    
    # === 存回 grouped_df ===
    grouped_df["Initial Move Angle (°)"] = initial_angles
    
    # === 加入四象限方向分類 ===
    def classify_quadrant(dx, dy):
        if dx > 0 and dy > 0:
            return "Q1"
        elif dx < 0 and dy > 0:
            return "Q2"
        elif dx < 0 and dy < 0:
            return "Q3"
        elif dx > 0 and dy < 0:
            return "Q4"
        else:
            return "Center/Undefined"
    
    quadrants = []
    for _, row in grouped_df.iterrows():
        start_idx = int(row["Frame Start"])
        end_idx = int(row["Frame End"])
        dx = df["cum_yaw_deg"].iloc[end_idx] - df["cum_yaw_deg"].iloc[start_idx]
        dy = df["cum_pitch_deg"].iloc[end_idx] - df["cum_pitch_deg"].iloc[start_idx]
        quadrants.append(classify_quadrant(dx, dy))
    grouped_df["Direction Quadrant"] = quadrants
    
    # === [7] 視覺化整體擊殺視角分群 ===
    all_minima_deg_x = df["cum_pitch_deg"].iloc[final_minima_idx]
    all_minima_deg_y = df["cum_yaw_deg"].iloc[final_minima_idx]
    if show:
        # === 繪圖開始 ===
        plt.figure(figsize=(8, 8))
        
        # 背景點（全視角軌跡）
        plt.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"] , alpha=0.3, s=5, label="All Points")
        
        # 所有最小值點
        plt.scatter(all_minima_deg_x, all_minima_deg_y, color='blue', s=40, label="Z Minima")
        
        # ✅ 使用 grouped_frames 分群畫圓（轉為視角單位）
        for group in grouped_frames:
            group_x = df["cum_pitch_deg"].iloc[group]
            group_y = df["cum_yaw_deg"].iloc[group]
            plt.scatter(group_x, group_y, facecolors='none', edgecolors='red',
                        s=120, linewidths=2)
        
        
        # === 圖例與標籤 ===
        plt.xlabel("Yaw Angle (°)")
        plt.ylabel("Pitch Angle (°)")
        plt.title("Z 最小值分群視覺化（以視角為單位）")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.show()
    return grouped_df

# %%

def findZminGroup(df, final_minima_idx, angle_merge_threshold=5, show=True):
    """
    
    根據 Z 軸局部最小值列表，自動分群擊殺動作，
    並提供最佳擊殺起始點 (NEW Frame Start)、初始移動角度、視角象限分類，
    最後以滑鼠速度著色顯示視角軌跡。
    
    parameters:
        df (pd.DataFrame): 含 X, Y, Z 及折算後視角欄位 (cum_yaw_deg, cum_pitch_deg, speed)
        final_minima_idx (list[int]): 已過濾的 Z 軸局部最小值 frame 索引
        angle_merge_threshold (float): 合併視角群組閾值 (°)
        show (bool): 是否顯示最終速度著色軌跡圖
    
    return:
        grouped_df (pd.DataFrame): 包含下列欄位的擊殺群組資訊
            - Group ID
            - Frames
            - Shot Count
            - Frame Start / End / Span
            - Initial Move Angle (°)
            - NEW Frame Start
            - Direction Quadrant
    """

    # --- 工具函數: 使用滑動視窗計算滑鼠移動方向起始點 ---
    def find_directional_start(df, frame_start, frame_end,
                                window_size=3, angle_threshold=45, min_magnitude=0.5):
        # 計算目標向量 (start -> end)
        goal_vec = np.array([
            df["X"].iloc[frame_end] - df["X"].iloc[frame_start],
            df["Y"].iloc[frame_end] - df["Y"].iloc[frame_start]
        ])
        if norm(goal_vec) == 0:
            return frame_start  # 若無移動則回傳原始起點
        # 滑動視窗: 從 start+1 開始檢查
        for offset in range(1, frame_end - frame_start - window_size):
            p0 = df[["X", "Y"]].iloc[frame_start + offset].values
            # 取 moving window length 的平均
            p1 = np.mean(df[["X", "Y"]].iloc[frame_start + offset + window_size].values, axis=0)
            init_vec = p1 - p0
            # 忽略太小的抖動
            if norm(init_vec) < min_magnitude: 
                continue
            # 計算兩個向量夾角
            cos_theta = np.dot(init_vec, goal_vec) / (norm(init_vec) * norm(goal_vec))
            angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

            if angle_deg <= angle_threshold:
                return frame_start + offset  # 找到符合條件的第一個起始 frame

        return frame_start

    # === [1] 計算前後角度差 ===
    angle_diffs = []
    for i in range(len(final_minima_idx)):
        idx_curr = final_minima_idx[i]
        yaw_curr = df["cum_yaw_deg"].iloc[idx_curr]
        pitch_curr = df["cum_pitch_deg"].iloc[idx_curr]
        diff_prev = np.nan if i == 0 else np.linalg.norm([
            yaw_curr - df["cum_yaw_deg"].iloc[final_minima_idx[i - 1]],
            pitch_curr - df["cum_pitch_deg"].iloc[final_minima_idx[i - 1]]
        ])
        diff_next = np.nan if i == len(final_minima_idx) - 1 else np.linalg.norm([
            yaw_curr - df["cum_yaw_deg"].iloc[final_minima_idx[i + 1]],
            pitch_curr - df["cum_pitch_deg"].iloc[final_minima_idx[i + 1]]
        ])
        angle_diffs.append({
            "Frame": idx_curr,
            "Z Value": df["Z"].iloc[idx_curr],
            "Angle_Diff_To_Prev_Minima (°)": diff_prev,
            "Angle_Diff_To_Next_Minima (°)": diff_next
        })
    angle_array = pd.DataFrame(angle_diffs)[["Frame", "Angle_Diff_To_Prev_Minima (°)", "Angle_Diff_To_Next_Minima (°)"]].to_numpy()

    # === [2] 合併視角差小於閾值的點 ===
    grouped_frames = []
    i = 0
    last_frame = None
    while i < len(angle_array) - 1:
        current_group = []
        if last_frame is not None:
            current_group.append(last_frame)
        current_group.append(angle_array[i][0])
        j = i + 1
        while j < len(angle_array):
            if pd.isna(angle_array[j][1]) or angle_array[j][1] >= angle_merge_threshold:
                break
            current_group.append(angle_array[j][0])
            j += 1
        if len(current_group) > 1:
            grouped_frames.append(current_group)
            last_frame = current_group[-1]
        else:
            last_frame = angle_array[i][0]
        i = j

    # === [3] 建立 DataFrame ===
    grouped_df = pd.DataFrame({
        "Group ID": list(range(1, len(grouped_frames) + 1)),
        "Frames": grouped_frames,
        "Shot Count": [len(g) - 1 for g in grouped_frames],
        "Frame Start": [min(g) for g in grouped_frames],
        "Frame End": [max(g) for g in grouped_frames],
    })
    grouped_df["Frame Span"] = grouped_df["Frame End"] - grouped_df["Frame Start"]

    # === [4] 初始移動角度與最佳起點 ===
    new_starts, angles = [], []
    for _, row in grouped_df.iterrows():
        s, e = int(row["Frame Start"]), int(row["Frame End"])
        new_s = find_directional_start(df, s, e)
        new_starts.append(new_s)

        goal_vec = np.array([df["X"].iloc[e] - df["X"].iloc[s], df["Y"].iloc[e] - df["Y"].iloc[s]])
        max_len = e - s
        found = False
        for l in range(5, max_len + 1):
            move = df.iloc[s:s + l]
            init_vec = np.array([move["X"].iloc[-1] - move["X"].iloc[0], move["Y"].iloc[-1] - move["Y"].iloc[0]])
            if norm(init_vec) == 0 or norm(goal_vec) == 0:
                break
            cos_theta = np.dot(init_vec, goal_vec) / (norm(init_vec) * norm(goal_vec))
            if cos_theta:
                angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
                angles.append(angle_deg)
                found = True
                break
        if not found:
            angles.append(np.nan)
    grouped_df["Initial Move Angle (°)"] = angles
    grouped_df["NEW Frame Start"] = new_starts

    # === [5] 象限分類 ===
    def classify_quadrant(dx, dy):
        if dx > 0 and dy > 0: return "Q1"
        if dx < 0 and dy > 0: return "Q2"
        if dx < 0 and dy < 0: return "Q3"
        if dx > 0 and dy < 0: return "Q4"
        return "Center/Undefined"
    grouped_df["Direction Quadrant"] = grouped_df.apply(
        lambda r: classify_quadrant(
            df["cum_yaw_deg"].iloc[int(r["Frame End"])] - df["cum_yaw_deg"].iloc[int(r["Frame Start"])],
            df["cum_pitch_deg"].iloc[int(r["Frame End"])] - df["cum_pitch_deg"].iloc[int(r["Frame Start"])],
        ), axis=1)

    # === [6] 視覺化 ===
    if show:
       plt.figure(figsize=(8, 8))
       # 用 speed 決定顏色深淺
       sc = plt.scatter(
           df['cum_pitch_deg'], df['cum_yaw_deg'],
           c=df['speed'], cmap='plasma', alpha=0.7, s=5,
           label='View Angle Trajectory (colored by speed)'
       )
       plt.colorbar(sc, label='Mouse Speed (°/s)')  # 或 mm/s 依你的 speed 定義

       # 標記所有 Z 軸局部最小值
       minima_x = df['cum_pitch_deg'].iloc[final_minima_idx]
       minima_y = df['cum_yaw_deg'].iloc[final_minima_idx]
       plt.scatter(minima_x, minima_y,
                   color='red', s=20, label='Z Minima', zorder=3)

       plt.xlabel('Pitch Angle (°)')
       plt.ylabel('Yaw Angle (°)')
       plt.title('View Angle Trajectory Colored by Mouse Speed')
       plt.grid(True)
       plt.axis('equal')
       plt.legend()
       plt.show()

    return grouped_df

# %%

import pandas as pd
import numpy as np
from numpy.linalg import norm # 導入 norm 函數，方便計算向量長度
import matplotlib.pyplot as plt # 導入繪圖庫

def findZminGroup(df, final_minima_idx, angle_merge_threshold=5, show=True):
    """
    根據 Z 軸局部最小值列表 (通常代表射擊/點擊事件)，自動分群連續或接近的擊殺動作。
    並為每個群組計算關鍵屬性，例如：
    - 最佳化的擊殺起始幀 (NEW Frame Start)：排除初始微小抖動，找到真正開始移動的幀。
    - 初始移動角度 (Initial Move Angle)：該擊殺動作開始時的移動方向與總體方向的夾角。
    - 視角移動象限分類 (Direction Quadrant)：根據 Yaw 和 Pitch 的變化判斷主要移動方向。
    最後，可以選擇性地以滑鼠速度 (speed) 為顏色，視覺化顯示整個視角移動軌跡及 Z 最小值點。

    應用場景：常用於分析 FPS 遊戲玩家的瞄準和射擊模式。

    Parameters:
        df (pd.DataFrame): 包含以下欄位的 DataFrame：
                           - 'X', 'Y', 'Z': 可能代表原始感測器數據或處理後的座標。
                           - 'cum_yaw_deg', 'cum_pitch_deg': 累積計算的水平和垂直視角角度 (度)。
                           - 'speed': 計算出的每個時間點的滑鼠移動速度。
        final_minima_idx (list[int]): 經過濾篩選後的 Z 軸局部最小值點所在的幀 (frame) 索引列表。
                                      這些點通常被視為射擊或關鍵操作的發生點。
        angle_merge_threshold (float): 用於合併相鄰 Z 最小值點的角度閾值 (單位：度)。
                                       如果兩個相鄰最小值點之間的視角變化小於此閾值，
                                       它們可能被視為同一次連續擊殺動作的一部分。預設為 5 度。
        show (bool): 是否顯示最終的速度著色軌跡圖。預設為 True。

    Return:
        grouped_df (pd.DataFrame): 包含分析後擊殺群組資訊的 DataFrame，欄位如下：
            - Group ID: 群組的唯一識別碼 (從 1 開始)。
            - Frames: 屬於該群組的所有原始 Z 最小值幀索引列表。
            - Shot Count: 該群組包含的擊殺次數 (Z 最小值點數量 - 1，假設每個 Z min 是一個 shot)。
            - Frame Start: 該群組中，第一個 Z 最小值點的幀索引。
            - Frame End: 該群組中，最後一個 Z 最小值點的幀索引。
            - Frame Span: 該群組的持續時間 (Frame End - Frame Start)。
            - Initial Move Angle (°): 初始移動方向與群組總體移動方向的角度差。
            - NEW Frame Start: 經過 `find_directional_start` 計算後，更精確的移動起始幀。
            - Direction Quadrant: 根據 Yaw/Pitch 變化判斷的視角移動象限 (Q1-Q4)。
    """

    # --- 工具函數: 使用滑動視窗計算滑鼠移動方向起始點 ---
    # 目標：找到從 frame_start 到 frame_end 這段移動中，真正開始"有方向性"移動的那個 frame
    #       排除掉一開始可能存在的、方向不明顯的微小抖動 (micro-adjustment/jitter)
    def find_directional_start(df, frame_start, frame_end,
                               window_size=3, angle_threshold=45, min_magnitude=0.5):
        """
        使用滑動窗口，尋找一段軌跡中，初始移動方向與總體方向一致的起始點。

        Parameters:
            df (pd.DataFrame): 包含 'X', 'Y' 座標的 DataFrame。
            frame_start (int): 分析範圍的起始幀索引。
            frame_end (int): 分析範圍的結束幀索引。
            window_size (int): 滑動窗口的大小，用於計算初始移動向量。
            angle_threshold (float): 初始移動向量與總體目標向量之間的最大允許角度差 (度)。
            min_magnitude (float): 初始移動向量的最小長度閾值，用於忽略過小的抖動。

        Returns:
            int: 找到的具有方向性的移動起始幀索引。如果找不到或無移動，返回原始 frame_start。
        """
        # 計算目標向量 (從 frame_start 指向 frame_end 的向量)
        goal_vec = np.array([
            df["X"].iloc[frame_end] - df["X"].iloc[frame_start], # X 方向位移
            df["Y"].iloc[frame_end] - df["Y"].iloc[frame_start]  # Y 方向位移
        ])
        goal_norm = norm(goal_vec) # 計算目標向量的長度 (大小)

        # 如果起點和終點相同 (沒有移動)，直接返回原始起點
        if goal_norm == 0:
            return frame_start

        # 滑動視窗: 從 start+1 開始檢查，直到接近終點的位置
        # offset 代表當前檢查的 "潛在" 起始點相對於 frame_start 的偏移量
        # 檢查範圍是 [frame_start + 1, frame_end - window_size -1]
        for offset in range(1, frame_end - frame_start - window_size):
            # 當前檢查的幀索引
            current_frame_idx = frame_start + offset
            # 滑動窗口的起始點 (p0) 和結束點 (p1) 的座標
            # 注意：這裡直接取 iloc[index] 的 .values，效率稍低於先取 series 再取 values
            p0 = df[["X", "Y"]].iloc[current_frame_idx].values
            # p1 使用窗口內點的平均值，或許可以考慮只用窗口末端點 p_end = df[["X", "Y"]].iloc[current_frame_idx + window_size].values
            # 使用 mean 可以平滑掉一些噪點
            p1 = np.mean(df[["X", "Y"]].iloc[current_frame_idx + 1 : current_frame_idx + window_size + 1].values, axis=0)

            # 計算初始移動向量 (從 p0 指向 p1)
            init_vec = p1 - p0
            init_norm = norm(init_vec) # 計算初始移動向量的長度

            # 如果初始移動向量太短 (可能是噪點或微小抖動)，則忽略，繼續下一個 offset
            if init_norm < min_magnitude:
                continue

            # 計算初始移動向量 (init_vec) 與 總體目標向量 (goal_vec) 之間的夾角
            # 使用向量內積公式: a · b = |a| |b| cos(theta)
            # cos(theta) = (a · b) / (|a| |b|)
            cos_theta = np.dot(init_vec, goal_vec) / (init_norm * goal_norm)
            # 使用 np.clip 確保 cos_theta 在 [-1, 1] 範圍內，避免浮點數誤差導致 arccos 出錯
            angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

            # 如果夾角小於閾值，表示初始移動方向與總體方向大致一致
            # 我們就認為 current_frame_idx 是真正的移動起始點
            if angle_deg <= angle_threshold:
                return current_frame_idx # 找到第一個符合條件的起始 frame，立即返回

        # 如果遍歷完所有可能的 offset 都沒有找到符合條件的起始點，
        # 則返回原始的 frame_start
        return frame_start

    # === [1] 計算每個 Z 最小值點與其前後相鄰 Z 最小值點之間的視角差 ===
    # 目標：計算出用於後續分群的依據 - 相鄰射擊點之間的視角距離。
    angle_diffs = [] # 用於儲存每個 Z 最小值點及其角度差資訊
    num_minima = len(final_minima_idx)

    # 提前提取需要的數據列為 NumPy 陣列，以加速後續訪問
    all_indices = np.array(final_minima_idx)
    yaw_values = df["cum_yaw_deg"].iloc[all_indices].values
    pitch_values = df["cum_pitch_deg"].iloc[all_indices].values
    z_values = df["Z"].iloc[all_indices].values # 也許會用到 Z 值，先提出來

    for i in range(num_minima):
        # 當前的 Z 最小值點的幀索引和視角座標
        idx_curr = final_minima_idx[i] # 這裡仍用列表索引，但下方計算已用 NumPy 數組
        yaw_curr = yaw_values[i]
        pitch_curr = pitch_values[i]

        # 計算與前一個 Z 最小值點的視角差 (歐氏距離)
        # 第一個點沒有前一個點，設為 NaN
        if i == 0:
            diff_prev = np.nan
        else:
            # 計算 (yaw_curr - yaw_prev)^2 + (pitch_curr - pitch_prev)^2 的平方根
            diff_prev = np.linalg.norm([yaw_curr - yaw_values[i - 1], pitch_curr - pitch_values[i - 1]])

        # 計算與後一個 Z 最小值點的視角差 (歐氏距離)
        # 最後一個點沒有後一個點，設為 NaN
        if i == num_minima - 1:
            diff_next = np.nan
        else:
            # 計算 (yaw_curr - yaw_next)^2 + (pitch_curr - pitch_next)^2 的平方根
            diff_next = np.linalg.norm([yaw_curr - yaw_values[i + 1], pitch_curr - pitch_values[i + 1]])

        # 將當前點的資訊和計算出的角度差存入列表
        angle_diffs.append({
            "Frame": idx_curr, # 原始幀索引
            "Z Value": z_values[i], # 對應的 Z 值 (雖然這裡沒用到，但記錄下可能有用)
            "Angle_Diff_To_Prev_Minima (°)": diff_prev, # 與前一個點的角度差
            "Angle_Diff_To_Next_Minima (°)": diff_next  # 與後一個點的角度差
        })

    # 將包含角度差資訊的列表轉換為 Pandas DataFrame，再轉為 NumPy 陣列，方便後續索引
    # 只選取需要的欄位: Frame, Angle_Diff_To_Prev_Minima (°), Angle_Diff_To_Next_Minima (°)
    # angle_array 的結構: [[frame1, diff_prev1, diff_next1], [frame2, diff_prev2, diff_next2], ...]
    angle_array = pd.DataFrame(angle_diffs)[["Frame", "Angle_Diff_To_Prev_Minima (°)", "Angle_Diff_To_Next_Minima (°)"]].to_numpy()

    # === [2] 合併視角差小於閾值的點 ===
    # 目標：根據步驟 [1] 計算出的 "與前一個點的角度差"，將角度差小的連續 Z 最小值點合併成一個群組。
    #       這代表這些射擊點在視角上非常接近，可能屬於同一次瞄準/射擊動作。
    grouped_frames = [] # 儲存最終分好的群組，每個群組是一個包含幀索引的列表
    i = 0 # 當前處理的 angle_array 的索引
    last_frame = None # 追蹤上一個被成功分組的最後一個 frame (這個變數命名和用途可能需要釐清，看似用於處理邊界)
                      # 更新理解：last_frame 似乎沒有跨組傳遞信息，主要是在單次外層 while 循環中暫存。

    while i < len(angle_array): # 遍歷所有 Z 最小值點 (或其角度差資訊)
        current_group = [] # 初始化當前群組
        # 這個檢查似乎多餘，因為 current_group 在每次循環開始時都重新初始化了
        if last_frame is not None:
            current_group.append(last_frame) # 似乎想把上一組的結尾加入下一組開頭？這邏輯可能需要確認

        # 將當前點 (angle_array[i]) 的 frame 加入 current_group
        # angle_array[i][0] 是 frame 索引
        current_group.append(angle_array[i][0])

        # 開始向後查找，看有多少個後續的點可以合併到 current_group
        j = i + 1 # 從當前點的下一個點開始檢查
        while j < len(angle_array):
            # 檢查點 j 與其前一個點 (即點 j-1) 之間的角度差
            # angle_array[j][1] 就是 "Angle_Diff_To_Prev_Minima (°)"
            # 如果角度差是 NaN (例如第一個點) 或 大於等於合併閾值，則停止合併
            if pd.isna(angle_array[j][1]) or angle_array[j][1] >= angle_merge_threshold:
                break # 停止內層 while 循環，不再將點 j 加入 current_group

            # 如果角度差小於閾值，則將點 j 的 frame 加入 current_group
            current_group.append(angle_array[j][0])
            j += 1 # 繼續檢查下一個點 (j+1)

        # 內層 while 循環結束後，檢查 current_group 的大小
        # 如果大小大於 1，表示至少有兩個點被合併成一個群組
        if len(current_group) > 1:
            grouped_frames.append(current_group) # 將這個有效群組加入最終結果列表
            last_frame = current_group[-1] # 更新 last_frame 為此群組的最後一個 frame (這行似乎也沒實際作用於下次迭代)
        else: # 如果 current_group 只有一個點 (沒有成功合併)
            last_frame = angle_array[i][0] # 更新 last_frame 為這個單獨的點 (同樣，用途不明)

        # 更新外層循環的索引 i
        # 跳過所有已經被處理 (合併) 的點，直接從 j 開始下一次外層循環
        i = j

    # === [3] 建立包含群組資訊的 DataFrame ===
    # 目標：將分好的群組 (grouped_frames) 整理成結構化的 DataFrame，並計算基本屬性。
    if not grouped_frames: # 如果沒有找到任何群組 (例如 Z 最小值點太少或角度差都很大)
        print("Warning: No groups were formed based on the angle threshold.")
        # 返回一個空的或者包含特定結構的 DataFrame，避免後續代碼出錯
        return pd.DataFrame(columns=[
            "Group ID", "Frames", "Shot Count", "Frame Start", "Frame End",
            "Frame Span", "Initial Move Angle (°)", "NEW Frame Start", "Direction Quadrant"
        ])

    grouped_df = pd.DataFrame({
        "Group ID": list(range(1, len(grouped_frames) + 1)), # 群組 ID，從 1 開始
        "Frames": grouped_frames,                            # 每個群組包含的幀列表
        # Shot Count: 假設組內點數 n 代表 n-1 次射擊間隔，所以是 len(g)-1 次射擊
        "Shot Count": [len(g) - 1 for g in grouped_frames],
        "Frame Start": [min(g) for g in grouped_frames],      # 群組起始幀 (取組內最小幀)
        "Frame End": [max(g) for g in grouped_frames],        # 群組結束幀 (取組內最大幀)
    })
    # 計算每個群組的持續時間 (幀數)
    grouped_df["Frame Span"] = grouped_df["Frame End"] - grouped_df["Frame Start"]

    # === [4] 計算每個群組的初始移動角度與最佳化起始點 (修改版) ===
    new_starts = [] # 儲存每個群組計算出的 "NEW Frame Start"
    angles = []     # 儲存每個群組計算出的 "Initial Move Angle (°)"
    
    # 提取群組的原始起始和結束幀為 NumPy 陣列
    group_starts = grouped_df["Frame Start"].values.astype(int)
    group_ends = grouped_df["Frame End"].values.astype(int)
    
    # 提前提取原始 DataFrame 中可能需要重複訪問的列為 NumPy 陣列
    x_coords = df["X"].values
    y_coords = df["Y"].values
    
    # 遍歷每個群組
    for i in range(len(grouped_df)):
        s, e = group_starts[i], group_ends[i] # 當前群組的原始起始和結束幀
    
        # --- 計算 NEW Frame Start ---
        # 調用工具函數，找到這個群組 (s 到 e) 更精確的移動起始點
        # *** 假設 find_directional_start 已經過優化或按原樣使用 ***
        new_s = find_directional_start(df, s, e)
        new_starts.append(new_s)
    
        # --- 計算 Initial Move Angle (使用 new_s 作為起點) ---
        # 目標：找到從 new_s 開始的一小段初始移動 (長度 l)
        #       計算其方向向量 (init_vec)
        #       計算從 new_s 到 e 的總體移動方向向量 (goal_vec)
        #       計算 init_vec 和 goal_vec 的夾角
    
        # 檢查 new_s 和 e 是否有效，以及它們之間是否能形成向量
        if new_s >= e: # 如果找到的新起點等於或晚於結束點，無法計算角度
            angles.append(np.nan)
            continue
    
        # 計算總體目標向量 (從 new_s 指向 e)
        goal_vec = np.array([x_coords[e] - x_coords[new_s], y_coords[e] - y_coords[new_s]])
        goal_norm = norm(goal_vec)
    
        # 如果從 new_s 到 e 沒有移動，無法計算角度
        if goal_norm == 0:
            angles.append(np.nan)
            continue
    
        # 定義初始移動段的最小長度 (至少需要2個點才能形成向量)
        # 保持原來的 5 幀作為一個有意義的初始移動判斷標準
        min_initial_length = 5
        max_len_from_new_start = e - new_s # 從 new_s 開始的最大可能長度
    
        found = False # 標記是否找到了有效的初始移動角度
    
        # 如果從 new_s 開始的總長度不足以形成一個最小長度的初始移動段
        if max_len_from_new_start < min_initial_length:
             angles.append(np.nan) # 無法計算有意義的初始角度
             continue # 處理下一個群組
    
        # 嘗試不同長度的初始移動段 (從 min_initial_length 到 max_len_from_new_start)
        # 迭代的 l 代表從 new_s 開始的移動段包含的幀數 (點數)
        for l in range(min_initial_length, max_len_from_new_start + 1):
            # 提取初始移動段的數據 (從 new_s 到 new_s + l - 1)
            # 切片範圍是 [new_s, new_s + l)
            move_x = x_coords[new_s : new_s + l]
            move_y = y_coords[new_s : new_s + l]
    
            # 確保切片有效 (理論上 range 保證 l >= min_initial_length >= 2)
            if len(move_x) < 2: # 雙重檢查
                 continue
    
            # 計算初始移動向量 (從第一個點 new_s 到最後一個點 new_s + l - 1)
            init_vec = np.array([move_x[-1] - move_x[0], move_y[-1] - move_y[0]])
            init_norm = norm(init_vec)
    
            # 如果初始移動向量長度為 0，則無法計算角度，繼續嘗試更長的片段
            if init_norm == 0:
                continue
    
            # 計算初始移動向量與 (從 new_s 開始的) 總體目標向量的夾角
            # 分母 goal_norm 在前面已檢查不為 0, init_norm 在此處檢查不為 0
            cos_theta = np.dot(init_vec, goal_vec) / (init_norm * goal_norm)
            angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    
            angles.append(angle_deg) # 將計算出的第一個有效角度加入列表
            found = True             # 標記已找到
            break # 找到第一個有效的初始移動角度後，停止內層循環
    
        # 如果遍歷完所有可能的初始移動長度 l，都沒有找到有效的角度
        # (可能是因為所有初始片段的 init_norm 都為 0)
        if not found:
            angles.append(np.nan) # 添加 NaN
    
    # --- 後續代碼 ---
    # 將計算結果添加回 grouped_df
    grouped_df["Initial Move Angle (°)"] = angles
    grouped_df["NEW Frame Start"] = new_starts # new_starts 列表在此賦值

    # === [5] 象限分類 ===
    # 目標：根據每個群組起始點到結束點的累積 Yaw 和 Pitch 變化，判斷主要移動方向屬於哪個象限。
    def classify_quadrant(dx, dy):
        """根據 X (Yaw) 和 Y (Pitch) 的變化量判斷象限"""
        if dx > 0 and dy > 0: return "Q1" # 右上
        if dx < 0 and dy > 0: return "Q2" # 左上
        if dx < 0 and dy < 0: return "Q3" # 左下
        if dx > 0 and dy < 0: return "Q4" # 右下
        # 其他情況 (例如只在一個軸上移動，或沒有移動)
        if dx == 0 and dy != 0: return "Vertical" # 垂直移動 (向上或向下)
        if dx != 0 and dy == 0: return "Horizontal" # 水平移動 (向左或向右)
        return "Center/Undefined" # 無明顯移動或起始結束點相同

    # 提取群組起始和結束幀對應的 Yaw 和 Pitch 值 (使用 .loc 或 .iloc 配合索引列表)
    # 確保索引是整數類型
    start_indices = grouped_df["Frame Start"].values.astype(int)
    end_indices = grouped_df["Frame End"].values.astype(int)

    # 使用 .iloc 批量提取數據，比 apply 或 iterrows 快得多
    start_yaw = df["cum_yaw_deg"].iloc[start_indices].values
    end_yaw = df["cum_yaw_deg"].iloc[end_indices].values
    start_pitch = df["cum_pitch_deg"].iloc[start_indices].values
    end_pitch = df["cum_pitch_deg"].iloc[end_indices].values

    # 計算 Yaw 和 Pitch 的變化量 (向量化操作)
    delta_yaw = end_yaw - start_yaw     # 對應 dx
    delta_pitch = end_pitch - start_pitch # 對應 dy

    # 使用向量化的條件判斷 (例如 np.select) 來進行分類，避免使用 apply
    conditions = [
        (delta_yaw > 0) & (delta_pitch > 0), # Q1
        (delta_yaw < 0) & (delta_pitch > 0), # Q2
        (delta_yaw < 0) & (delta_pitch < 0), # Q3
        (delta_yaw > 0) & (delta_pitch < 0), # Q4
        (delta_yaw == 0) & (delta_pitch != 0),# Vertical
        (delta_yaw != 0) & (delta_pitch == 0),# Horizontal
    ]
    choices = ["Q1", "Q2", "Q3", "Q4", "Vertical", "Horizontal"]
    # np.select(條件列表, 選擇列表, 預設值)
    grouped_df["Direction Quadrant"] = np.select(conditions, choices, default="Center/Undefined")

    # === [6] 視覺化 ===
    # 目標：如果 show=True，繪製視角軌跡圖，用顏色深淺表示滑鼠移動速度，並標記 Z 最小值點。
    if show:
        plt.figure(figsize=(8, 8)) # 設定圖表大小

        # 繪製主要的視角軌跡散點圖
        # x 軸是 Pitch (垂直視角)，y 軸是 Yaw (水平視角)
        # c=df['speed'] 指定點的顏色由 'speed' 欄位決定
        # cmap='plasma' 指定顏色映射方案
        # alpha=0.7 設定透明度
        # s=5 設定點的大小
        sc = plt.scatter(
            df['cum_pitch_deg'], df['cum_yaw_deg'], # X, Y 座標
            c=df['speed'], cmap='plasma', alpha=0.7, s=5, # 顏色、透明度、大小
            label='View Angle Trajectory (colored by speed)' # 圖例標籤
        )
        # 添加顏色條 (colorbar) 並標註其代表的意義 ('Mouse Speed')
        plt.colorbar(sc, label='Mouse Speed (°/s or unit of speed column)')

        # 在圖上特別標記出所有的 Z 軸局部最小值點 (通常是紅色)
        # 提取這些點的 Pitch 和 Yaw 座標
        # 這裡再次使用了 iloc，如果 final_minima_idx 很大，也可能稍慢，但通常可接受
        minima_pitch = df['cum_pitch_deg'].iloc[final_minima_idx].values
        minima_yaw = df['cum_yaw_deg'].iloc[final_minima_idx].values
        plt.scatter(minima_pitch, minima_yaw,
                    color='red', s=20, label='Z Minima', zorder=3) # zorder=3 讓紅點在最上層

        # 設定圖表的標籤、標題、網格線和坐標軸比例
        plt.xlabel('Pitch Angle (°)')
        plt.ylabel('Yaw Angle (°)')
        plt.title('View Angle Trajectory Colored by Mouse Speed with Z Minima')
        plt.grid(True) # 顯示網格線
        plt.axis('equal') # 讓 X 和 Y 軸具有相同的單位長度比例，避免角度變形
        plt.legend() # 顯示圖例
        plt.show() # 顯示圖表

    # 返回最終處理好的包含群組資訊的 DataFrame
    return grouped_df
# %%

def excludeCenter(df, filtered_minima_data, grouped_df,
                  target_length = 101,
                  show=True):
    """
   排除起點不在中央視角範圍的擊殺群組，並標記各群組的速度方向象限，
   可視化剩餘群組的最終 Z 軸最小值與分群結果。

   參數:
       df (pd.DataFrame): 包含 X, Y, Z 與視角欄位(cum_yaw_deg, cum_pitch_deg, speed)
       filtered_minima_data (pd.DataFrame): Z 軸最小值資料 (含 Frame 欄位)
       grouped_df (pd.DataFrame): 原始分群結果 (含 Frames, Frame Start, End 等欄位)
       target_length (int): 標準化速度序列的長度
       show (bool): 是否繪製可視化圖

   返回:
       excldueCen_grouped_df (pd.DataFrame): 排除後的分群結果
   """
    # === o. 篩選機制，只有從中心出發才會計算 ===
    yaw_center = (df["cum_yaw_deg"].max() + df["cum_yaw_deg"].min()) / 2
    pitch_center = (df["cum_pitch_deg"].max() + df["cum_pitch_deg"].min()) / 2
    
    yaw_range = 10   # 水平方向 ±10°
    pitch_range = 10  # 垂直方向 ±10°
    
    filtered_minima_idx = filtered_minima_data["Frame"].tolist()
    
    central_minima_frames = []
    
    for idx in filtered_minima_idx:
        yaw = df["cum_yaw_deg"].iloc[idx]
        pitch = df["cum_pitch_deg"].iloc[idx]
        
        if (abs(yaw - yaw_center) <= yaw_range) and (abs(pitch - pitch_center) <= pitch_range):
            central_minima_frames.append(idx)
    
    
    # 1. central_minima_frames 本身就是一维的 frame 索引列表
    central_flat = set(central_minima_frames)
    
    # 2. 只要该群组的第一帧不在 central_flat，就排除它
    excludeCen_idx = []
    for i, frames in enumerate(grouped_df["Frames"]):
        first_frame = int(frames[0])
        if first_frame not in central_flat:
            excludeCen_idx.append(i)
    
    # 3. （可选）去重
    excludeCen_idx = list(dict.fromkeys(excludeCen_idx))
    
    
    excldueCen_grouped_df = grouped_df.iloc[excludeCen_idx, :].reset_index(drop=True)
    
    
    standardized_data = pd.DataFrame(np.zeros([target_length,
                                               len(grouped_df)]))
    direction_labels = []  # 用來儲存象限
    for idx in range(len(grouped_df)):
        # 取出路徑
        # 從速度為正值在開始取
        start_frame = int(grouped_df["Frames"][idx][0])
        end_frame = int(grouped_df["Frames"][idx][-1])
        signal_trial = df.iloc[start_frame:end_frame, :]
        
        original_length = len(signal_trial["angle_speed_dps"])
        sequence = signal_trial["angle_speed_dps"]
        if original_length == target_length:
            standardized_data.iloc[:, idx] = sequence
        elif original_length > 1:
            x_original = np.linspace(0, 1, original_length)
            x_target = np.linspace(0, 1, target_length)
            interp_func = interp1d(x_original, sequence, kind='cubic', fill_value="extrapolate")
            standardized_data.iloc[:, idx] = interp_func(x_target).tolist()
        # 計算速度方向，並區分為四個象限，新增註記
        # --- 計算速度方向 ΔX, ΔY ---
        start_x = signal_trial["X"].iloc[0]
        start_y = signal_trial["Y"].iloc[0]
        end_x = signal_trial["X"].iloc[-1]
        end_y = signal_trial["Y"].iloc[-1]
        delta_x = end_x - start_x
        delta_y = end_y - start_y
    
        # --- 分象限 ---
        if delta_x > 0 and delta_y > 0:
            direction = "Q1"
        elif delta_x < 0 and delta_y > 0:
            direction = "Q2"
        elif delta_x < 0 and delta_y < 0:
            direction = "Q3"
        elif delta_x > 0 and delta_y < 0:
            direction = "Q4"
        else:
            direction = "Undefined"
    
        direction_labels.append(direction)
    
    # 新增欄位至 grouped_df
    grouped_df["Direction Quadrant"] = direction_labels
    
    
    if show:
        excldueCen_minima_idx = excldueCen_grouped_df["Frames"].tolist()
        last_values = [int(sublist[-1]) for sublist in excldueCen_minima_idx]
        """
        這裡有問題，剛剛修改到這裡 2025.05.01 13:46
        """
        all_minima_deg_x = df["cum_pitch_deg"].iloc[last_values]
        all_minima_deg_y = df["cum_yaw_deg"].iloc[last_values]
        plt.figure(figsize=(10, 8))
        
        # 背景點（全視角軌跡）
        plt.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"], alpha=0.3, s=5, label="All Points")
        
        # 所有最小值點
        plt.scatter(all_minima_deg_x, all_minima_deg_y, color='blue', s=40, label="Z Minima")
        
        
        # ✅ 使用 grouped_df 分群畫圓（轉為視角單位）
        for group in excldueCen_grouped_df["Frames"]:
            print(group)
            group_x = df["cum_pitch_deg"].iloc[group]
            group_y = df["cum_yaw_deg"].iloc[group]
            plt.scatter(group_x, group_y, facecolors='none', edgecolors='red',
                        s=120, linewidths=2)
        
        
        # === 圖例與標籤 ===
        plt.xlabel("Yaw Angle (°)")
        plt.ylabel("Pitch Angle (°)")
        plt.title("Z 最小值分群視覺化（以視角為單位）")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.show()
    return excldueCen_grouped_df

# %%

# 將單位從mm轉換成視角
df = ConverUnit2Angle(combine_dict, descriptions)
# 2.1. 找出每一次目標擊殺的開槍數 -> 找出Z axis local minimal
# 2.1.1. 以滑鼠點擊次數計算，使用Z軸局部最小值，如果兩次Z軸局部最小值的視角差
#         小於某個閾值，則視為仍在瞄準同一個目標
filtered_minima_data = find_Zaxis_min(df, show=True)
filtered_minima_idx = filtered_minima_data["Frame"].tolist()
# 2.1.2. 找出完成擊殺的 frame 以及上一個視角大於閾值的視角位置
"""
仍然需要再加上依照速度方向的多重條件
"""
grouped_df = findZminGroup(df, filtered_minima_idx)
# 2.2. 找出從中心出發的開槍軌跡
# 2025.04.30 接下來從這邊開始
excldueCen_grouped_df = excludeCenter(df, filtered_minima_data, grouped_df)



# %%
"""
2. 計算
    2.1. 找出每一次目標擊殺的開槍數 -> 找出Z axis local minimal
        2.1.1. 以滑鼠點擊次數計算，使用Z軸局部最小值，如果兩次Z軸局部最小值的視角差
                小於某個閾值，則視為仍在瞄準同一個目標
        2.1.2. 找出完成擊殺的 frame 以及上一個視角大於閾值的視角位置
                data format
            	GroupID   Frames          Shot Count   Frame Start   Frame End   Frame Span
                -------   --------------  -----------  ------------  ----------  -----------
                1       [53.0, 71.0]          1           53.0         71.0        18.0
    2.2. 找出從中心出發的開槍軌跡
    2.3. 定義開槍軌跡: 多重條件
        2.3.1. 只有速度方向往目標方向才算開始
        2.3.2. 速度達到一定閾值？ 速度與目標方向的偏差角度？
    2.4. 計算初始偏移角度
"""

# === 滑鼠移動轉視角（整段軌跡） ===
# delta_x_mm = df["X"].diff().fillna(0)
# delta_y_mm = df["Y"].diff().fillna(0)




# %%
"""
    2.2. 計算
        2.2.1. 指標
            o. (廢棄)擊殺數, 命中率？
            a. Throughput (Mouse Travel Efficiency): 
            b. Mouse Speed (°/s): 找出整段時間內的最大值 or 平均值，單位換算成視角
            c. Initial Move Angle: 初始 5 個 frame 的移動方向與最終擊殺目標位置的視角差
                修改條件: 1. 排除所有Initial Move Angle大於45度的trial
                         2. Frame Span 要大於 20
            d. Full Path Time: 
                使用 Frame Span/descriptions['motion info']['frame_rate']
            e. Reaction Time: 從這次目標擊殺到某個 frame 移動速度超過一個閾值 
                扣掉直接回中的反應時間
            i. 一槍擊殺的次數, 二槍, 三槍...
            j. 超過目標的次數， 還沒到目標就開槍的次數
            k. Mouse Travel Efficiency: idea path/real path
        2.2.2. 不同方向的計算: 全部方向綜合, 分四個方向 (四象限)
"""




# 修正 不應該使用初始角度作為區分

# 排除所有初始角度大於45度的trial    
final_grouped_df = cen_grouped_df[(cen_grouped_df["Initial Move Angle (°)"] <= 45) \
                                  & (cen_grouped_df["Frame Span"] > 20)].reset_index(drop=True)
# 多做一個統計 去掉outline

# 將每一筆資料都標準化成固定長度




# === b. Mouse Speed (°/s) ===
max_angle_speed = max(df["angle_speed_dps"])
mean_angle_speed = np.mean(df["angle_speed_dps"])
# === c. Initial Move Angle: ===


mean_initial_move_angle = np.mean(cen_grouped_df["Initial Move Angle (°)"]\
                                  [cen_grouped_df["Initial Move Angle (°)"] <= 45])
# === d. Full Path Time (單位 Second)===
path_time = np.mean(cen_grouped_df["Frame Span"])\
    /descriptions['motion info']['frame_rate']

# === e. Reaction Time ===
# 只計算從中心出發，並且 initial move angle 小於 45 度

# === x. 量化速度 ===



    

# === k. Mouse Travel Efficiency
# Mouse Travel Efficiency: idea path/real path

# %% mean std cloud
palette = plt.get_cmap('Set1')
fig, axs = plt.subplots(1, 1, figsize = (8, 6), sharex='col')

# x, y = i - n*math.floor(abs(i)/n), math.floor(abs(i)/n)
color = palette(0) # 設定顏色
# 都改成100個點
iters = list(np.linspace(0,
                         len(standardized_data[0]),
                         len(standardized_data[0])))
# 設定計算資料
avg1 = np.mean(standardized_data, axis=1) # 計算平均
std1 = np.std(standardized_data, axis=1) # 計算標準差
r1 = list(map(lambda x: x[0]-x[1], zip(avg1, std1))) # 畫一個標準差以內的線
r2 = list(map(lambda x: x[0]+x[1], zip(avg1, std1)))
axs.plot(iters, avg1, color=color, label='before', linewidth=3)
axs.fill_between(iters, r1, r2, color=color, alpha=0.2)

# 畫第二條線
color = palette(1) # 設定顏色
avg2 = np.mean(standardized_data, axis=1) # 計畫平均
std2 = np.std(standardized_data, axis=1) # 計算標準差
r1 = list(map(lambda x: x[0]-x[1], zip(avg2, std2))) # 畫一個標準差以內的線
r2 = list(map(lambda x: x[0]+x[1], zip(avg2, std2)))

axs.plot(iters, avg2, color=color, label='after', linewidth=3) # 畫平均線
axs.fill_between(iters, r1, r2, color=color, alpha=0.2) # 塗滿一個正負標準差以內的區塊
# 圖片的格式設定
# axs.set_title(example_data.columns[i+1], fontsize=12)
axs.legend(loc="lower left") # 圖例位置
axs.grid(True, linestyle='-.')
# 畫放箭時間
# axs[x, y].set_xlim(-(release[0]), release[1])
# axs.axvline(x=0, color = 'darkslategray', linewidth=1, linestyle = '--')
    
plt.suptitle(str("mean std cloud: "), fontsize=16)
plt.tight_layout()
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("time (%)", fontsize = 14)
plt.ylabel("Velocity (°/s)", fontsize = 14)
# plt.savefig(save, dpi=200, bbox_inches = "tight")
plt.show()

"""
1. 待解決問題，分成四象限

"""




















