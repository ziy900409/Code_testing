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
def find_Zaxis_min(combine_dict, order=5, min_frame_gap =8,
                   min_z_diff=0.2, threshold=0.05, show=True):
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
    z_values = combine_dict["markers"]["R.I.Finger3"][:, 2]
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
    return final_minima_idx, filtered_minima_data
# %%
def ConverUnit2Angle(combine_dict, descriptions, final_minima_idx,
                     DPI=800, sens=1, yaw=0.022,
                     show=True, showVel=True):
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
    
    # 7️⃣ 取得篩選過的 Z 軸局部最小值對應的視角位置
    filtered_yaw = df.loc[final_minima_idx, "cum_yaw_deg"]
    filtered_pitch = df.loc[final_minima_idx, "cum_pitch_deg"]
    
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
    
            # 若夾角小於 90 度，接受此向量
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
# 2.1. 找出每一次目標擊殺的開槍數 -> 找出Z axis local minimal
# 2.1.1. 以滑鼠點擊次數計算，使用Z軸局部最小值，如果兩次Z軸局部最小值的視角差
#         小於某個閾值，則視為仍在瞄準同一個目標
filtered_minima_data = find_Zaxis_min(combine_dict, show=True)
final_minima_idx = filtered_minima_data["Frame"]
# 將單位從mm轉換成視角
df = ConverUnit2Angle(combine_dict, descriptions, filtered_minima_data["Frame"])
# 2.1.2. 找出完成擊殺的 frame 以及上一個視角大於閾值的視角位置
grouped_df = findZminGroup(df, final_minima_idx)
# 2.2. 找出從中心出發的開槍軌跡
2025.04.30 接下來從這邊開始



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

# === o. 篩選機制，只有從中心出發才會計算 ===
yaw_center = (df["cum_yaw_deg"].max() + df["cum_yaw_deg"].min()) / 2
pitch_center = (df["cum_pitch_deg"].max() + df["cum_pitch_deg"].min()) / 2

yaw_range = 10   # 水平方向 ±10°
pitch_range = 10  # 垂直方向 ±10°

central_minima_frames = []

for idx in final_minima_idx:
    yaw = df["cum_yaw_deg"].iloc[idx]
    pitch = df["cum_pitch_deg"].iloc[idx]
    
    if (abs(yaw - yaw_center) <= yaw_range) and (abs(pitch - pitch_center) <= pitch_range):
        central_minima_frames.append(idx)
cen_idx = []
for idx in range(len(grouped_df)):
    for num in range(len(central_minima_frames)):
        if int(grouped_df["Frames"][idx][0]) == central_minima_frames[num]:
            cen_idx.append(idx)
            
cen_grouped_df = grouped_df.iloc[cen_idx, :].reset_index(drop=True)


# 修正 不應該使用初始角度作為區分

# 排除所有初始角度大於45度的trial    
final_grouped_df = cen_grouped_df[(cen_grouped_df["Initial Move Angle (°)"] <= 45) \
                                  & (cen_grouped_df["Frame Span"] > 20)].reset_index(drop=True)
# 多做一個統計 去掉outline

# 將每一筆資料都標準化成固定長度
target_length = 101
standardized_data = pd.DataFrame(np.zeros([target_length,
                                               len(final_grouped_df)]))
direction_labels = []  # 用來儲存象限
for idx in range(len(final_grouped_df)):
    # 取出路徑
    # 從速度為正值在開始取
    start_frame = int(final_grouped_df["Frames"][idx][0])
    end_frame = int(final_grouped_df["Frames"][idx][-1])
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

# 新增欄位至 final_grouped_df
final_grouped_df["Direction Quadrant"] = direction_labels

plt.figure(figsize=(10, 8))

# 背景點（全視角軌跡）
plt.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"], alpha=0.3, s=5, label="All Points")

# 所有最小值點
plt.scatter(all_minima_deg_x, all_minima_deg_y, color='blue', s=40, label="Z Minima")


# ✅ 使用 final_grouped_df 分群畫圓（轉為視角單位）
for group in final_grouped_df["Frames"]:
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




















