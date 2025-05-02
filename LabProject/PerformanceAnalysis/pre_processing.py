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


def findZminGroup(df, final_minima_idx, angle_merge_threshold=10, show=True):
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


def excludeCenter_and_plot_separately(df: pd.DataFrame, grouped_df: pd.DataFrame,
                                      yaw_range: float = 10, pitch_range: float = 10,
                                      show: bool = True) -> pd.DataFrame:
    """
    過濾擊殺動作群組，僅保留結束點位於視角中心區域之外的群組。
    如果 show=True，則會分別顯示兩張視覺化圖表：
    1. 點分佈圖：顯示所有點、保留的群組點及中心排除區域。
    2. 箭頭圖：顯示保留群組的移動方向箭頭 (從起始幀到結束幀)。

    Args:
        df (pd.DataFrame): 包含原始數據的 DataFrame ('cum_yaw_deg', 'cum_pitch_deg')。
        grouped_df (pd.DataFrame): 預先計算好的擊殺群組 DataFrame (包含 'Frames' 列表)。
        yaw_range (float): 中心區域的水平半徑 (度)。
        pitch_range (float): 中心區域的垂直半徑 (度)。
        show (bool): 是否顯示視覺化圖表。

    Returns:
        pd.DataFrame: 經過濾後的 DataFrame，僅包含結束點不在中心的群組。
    """

    # === 1. 定義視角中心區域 ===
    # (這裡繼續使用中位數，你可根據需要更改為 mean 或 min/max 中點)
    try:
        yaw_center = df["cum_yaw_deg"].median()
        pitch_center = df["cum_pitch_deg"].median()
        print(f"【基於中位數】視角中心計算結果: Yaw={yaw_center:.2f}°, Pitch={pitch_center:.2f}°")
        print(f"中心區域定義: Yaw ±{yaw_range}°, Pitch ±{pitch_range}°")
    except KeyError as e:
        print(f"錯誤：輸入的 df 缺少必要的欄位 {e}")
        return pd.DataFrame() # 返回空的 DataFrame 或拋出異常

    # === 2. 向量化過濾 ===
    temp_grouped = grouped_df.copy()

    def get_last_frame(frames_list):
        if isinstance(frames_list, list) and len(frames_list) > 0:
             # 確保幀索引是有效的數值類型，並處理可能的錯誤
             try:
                 return int(frames_list[-1])
             except (ValueError, TypeError):
                 return np.nan # 如果轉換失敗，返回 NaN
        return np.nan

    if 'Frames' not in temp_grouped.columns:
        print("錯誤：grouped_df 缺少 'Frames' 欄位")
        return pd.DataFrame()

    temp_grouped['last_frame'] = temp_grouped['Frames'].apply(get_last_frame)

    # 檢查必要的座標欄位是否存在
    if 'cum_yaw_deg' not in df.columns or 'cum_pitch_deg' not in df.columns:
        print("錯誤：df 缺少 'cum_yaw_deg' 或 'cum_pitch_deg' 欄位")
        return pd.DataFrame()

    yaw_map = df['cum_yaw_deg']
    pitch_map = df['cum_pitch_deg']

    temp_grouped['last_yaw'] = temp_grouped['last_frame'].map(yaw_map)
    temp_grouped['last_pitch'] = temp_grouped['last_frame'].map(pitch_map)

    valid_coords_mask = temp_grouped['last_yaw'].notna() & temp_grouped['last_pitch'].notna()

    is_in_center_mask = pd.Series(False, index=temp_grouped.index)
    # 僅在有效座標上計算是否在中心
    if valid_coords_mask.any():
        is_in_center_mask.loc[valid_coords_mask] = (
            (abs(temp_grouped.loc[valid_coords_mask, 'last_yaw'] - yaw_center) <= yaw_range) &
            (abs(temp_grouped.loc[valid_coords_mask, 'last_pitch'] - pitch_center) <= pitch_range)
        )

    # 保留條件：座標有效 且 不在中心
    keep_mask = valid_coords_mask & (~is_in_center_mask)
    filtered_grouped_df = grouped_df.loc[keep_mask].reset_index(drop=True)

    # === 3. 計算與報告排除數量 ===
    original_count = len(grouped_df)
    filtered_count = len(filtered_grouped_df)
    excluded_count = original_count - filtered_count
    print(f"原始群組數量: {original_count}")
    print(f"因 **結束點在中心區域** 或 **資料無效/缺失** 而被排除的群組數量: {excluded_count}")
    print(f"過濾後剩餘群組數量: {filtered_count}")

    # === 4. 視覺化 (明確分為兩張圖) ===
    if show:
        if filtered_grouped_df.empty:
            print("沒有可供顯示的過濾後群組。")
        else:
            # --- 圖 1: 點分佈與中心區域 ---
            try:
                plt.figure(figsize=(10, 8))
                ax1 = plt.gca()
                # 背景點
                ax1.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"], alpha=0.1, s=5, color='gray', label="所有數據點")
                # 保留群組的所有點
                all_retained_frames_plot1 = [frame for frames_list in filtered_grouped_df["Frames"] for frame in frames_list if isinstance(frames_list, list)]
                # 過濾掉無效的幀索引 (例如 NaN 或非數字)
                valid_frame_indices = [f for f in all_retained_frames_plot1 if pd.notna(f) and isinstance(f, (int, float))]
                valid_retained_frames_plot1 = df.index.intersection(valid_frame_indices)

                if not valid_retained_frames_plot1.empty:
                     ax1.scatter(df.loc[valid_retained_frames_plot1, "cum_pitch_deg"], df.loc[valid_retained_frames_plot1, "cum_yaw_deg"],
                                 color='blue', s=20, alpha=0.6, label="保留群組的點", zorder=3)
                # 繪製每個被保留群組的範圍
                for idx, row in filtered_grouped_df.iterrows(): # 使用 idx 避免與 plt 變數衝突
                    group_frames = row["Frames"]
                    if not group_frames: continue

                    try:
                        group_pitch = df["cum_pitch_deg"].loc[group_frames]
                        group_yaw = df["cum_yaw_deg"].loc[group_frames]
                        # 使用半透明紅色邊框標記群組
                        plt.scatter(group_pitch, group_yaw, facecolors='none', edgecolors='red',
                                    s=80, linewidths=1.5, alpha=0.7, label="保留的群組範圍" if idx == 0 else "", zorder=2)
                    except KeyError:
                        print(f"警告：繪製群組 {idx} 時無法在 df 中找到部分幀索引，該群組可能未完整繪製。")
                
                # 中心區域框
                rect_pitch = [pitch_center - pitch_range, pitch_center + pitch_range, pitch_center + pitch_range, pitch_center - pitch_range, pitch_center - pitch_range]
                rect_yaw = [yaw_center - yaw_range, yaw_center - yaw_range, yaw_center + yaw_range, yaw_center + yaw_range, yaw_center - yaw_range]
                ax1.plot(rect_pitch, rect_yaw, color='green', linestyle='--', linewidth=2, label="中心區域 (排除用)")
                # 圖表元素
                ax1.set_xlabel("Pitch Angle (°)")
                ax1.set_ylabel("Yaw Angle (°)")
                ax1.set_title("過濾後的擊殺群組視覺化 (點分佈與排除區域)")
                ax1.grid(True)
                ax1.axis("equal")
                handles, labels = ax1.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax1.legend(by_label.values(), by_label.keys())
                plt.show() # 顯示第一張圖
            except Exception as e:
                print(f"繪製第一張圖時發生錯誤: {e}")


            # --- 圖 2: 移動方向箭頭 ---
            try:
                plt.figure(figsize=(10, 8))
                ax2 = plt.gca()
                # 可選: 背景點
                ax2.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"], alpha=0.05, s=5, color='gray', label="所有數據點 (背景)")
                # 可選: 保留群組的點
                if not valid_retained_frames_plot1.empty: # 使用上面計算過的索引
                     ax2.scatter(df.loc[valid_retained_frames_plot1, "cum_pitch_deg"], df.loc[valid_retained_frames_plot1, "cum_yaw_deg"],
                                 color='blue', s=10, alpha=0.3, label="保留群組的點 (參考)", zorder=2)

                arrow_drawn = False # 圖例標籤控制
                # 遍歷繪製箭頭
                for idx, row in filtered_grouped_df.iterrows():
                    group_frames = row["Frames"]
                    if isinstance(group_frames, list) and len(group_frames) >= 2:
                        try:
                            start_frame = int(group_frames[0])
                            end_frame = int(group_frames[-1])

                            # 獲取座標 (增加錯誤檢查)
                            if start_frame not in df.index or end_frame not in df.index:
                                print(f"警告：群組 {idx} 的開始幀 {start_frame} 或結束幀 {end_frame} 不在 df 的索引中。")
                                continue

                            pitch_start = df.loc[start_frame, "cum_pitch_deg"]
                            yaw_start = df.loc[start_frame, "cum_yaw_deg"]
                            pitch_end = df.loc[end_frame, "cum_pitch_deg"]
                            yaw_end = df.loc[end_frame, "cum_yaw_deg"]

                            # 檢查座標是否有效 (非 NaN)
                            if pd.isna(pitch_start) or pd.isna(yaw_start) or pd.isna(pitch_end) or pd.isna(yaw_end):
                                print(f"警告：群組 {idx} 的開始或結束座標無效 (NaN)。")
                                continue

                            # 繪製箭頭
                            ax2.annotate(
                                '', xy=(pitch_end, yaw_end), xytext=(pitch_start, yaw_start),
                                arrowprops=dict(arrowstyle="->", color="lightsteelblue", lw=1, linestyle="--", shrinkA=5, shrinkB=5),
                                zorder=3 )
                            if not arrow_drawn:
                                ax2.plot([], [], color='red', lw=1, label='擊殺動作方向 (開始->結束)')
                                arrow_drawn = True
                        except (KeyError, ValueError, TypeError) as frame_err:
                             print(f"警告：處理群組 {idx} 的幀 {group_frames} 時出錯: {frame_err}")
                             continue # 跳過這個群組的箭頭繪製
                # 繪製每個被保留群組的範圍
                for idx, row in filtered_grouped_df.iterrows(): # 使用 idx 避免與 plt 變數衝突
                    group_frames = row["Frames"]
                    if not group_frames: continue
    
                    try:
                        group_pitch = df["cum_pitch_deg"].loc[group_frames]
                        group_yaw = df["cum_yaw_deg"].loc[group_frames]
                        # 使用半透明紅色邊框標記群組
                        plt.scatter(group_pitch, group_yaw, facecolors='none', edgecolors='red',
                                    s=80, linewidths=1.5, alpha=0.7, label="保留的群組範圍" if idx == 0 else "", zorder=2)
                    except KeyError:
                        print(f"警告：繪製群組 {idx} 時無法在 df 中找到部分幀索引，該群組可能未完整繪製。")
        
                # 中心區域框
                rect_pitch = [pitch_center - pitch_range, pitch_center + pitch_range, pitch_center + pitch_range, pitch_center - pitch_range, pitch_center - pitch_range]
                rect_yaw = [yaw_center - yaw_range, yaw_center - yaw_range, yaw_center + yaw_range, yaw_center + yaw_range, yaw_center - yaw_range]
                ax2.plot(rect_pitch, rect_yaw, color='green', linestyle='--', linewidth=2, label="中心區域 (排除用)")
                # 圖表元素
                ax2.set_xlabel("Pitch Angle (°)")
                ax2.set_ylabel("Yaw Angle (°)")
                ax2.set_title("過濾後擊殺群組的移動方向箭頭")
                ax2.grid(True)
                ax2.axis("equal")
                handles, labels = ax2.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                if by_label: ax2.legend(by_label.values(), by_label.keys())
                plt.show() # 顯示第二張圖
            except Exception as e:
                print(f"繪製第二張圖時發生錯誤: {e}")

    # === 5. 返回結果 ===
    return filtered_grouped_df
# %%



def standardize_group_signals(df, filtered_grouped_df, signal_column_name,
                              target_length=101,
                              start_col='Frame Start', # 或 'NEW Frame Start'
                              end_col='Frame End',
                              group_id_col='Group ID'):
    """
    為 filtered_grouped_df 中的每個擊殺群組提取指定的信號時間序列，
    並使用三次樣條插值 (cubic interpolation) 將其標準化為固定的長度。

    參數 (Parameters):
        df (pd.DataFrame): 包含原始時間序列數據的 DataFrame。
                           必須包含 `signal_column_name` 指定的欄位以及幀索引。
        filtered_grouped_df (pd.DataFrame): 經過濾後的群組 DataFrame (例如來自 excludeCenter)。
                                            必須包含 `start_col`, `end_col`, 和 `group_id_col` 指定的欄位。
        signal_column_name (str): 需要從 `df` 中提取並標準化的信號欄位名稱
                                  (例如 'angle_speed_dps', 'speed', 'X', 'Y', 'Z',
                                   'cum_yaw_deg', 'cum_pitch_deg')。
        target_length (int): 標準化後的目標序列長度。預設為 101。
        start_col (str): `filtered_grouped_df` 中代表群組起始幀的欄位名稱。
                         預設為 'Frame Start'。可改為 'NEW Frame Start' 等。
        end_col (str): `filtered_grouped_df` 中代表群組結束幀的欄位名稱。
                       預設為 'Frame End'。
        group_id_col (str): `filtered_grouped_df` 中代表群組唯一標識符的欄位名稱。
                            預設為 'Group ID'。結果字典的鍵將使用此欄位的值。

    返回 (Return):
        dict: 一個字典，其中：
              - 鍵 (key) 是每個群組的 ID (來自 `group_id_col`)。
              - 值 (value) 是對應群組提取並標準化後的信號 (NumPy array, 長度為 `target_length`)。
              如果某個群組的原始信號長度不足以進行插值 (少於2個點)，
              或者發生其他錯誤，其對應的值可能為全 NaN 的 NumPy 陣列。

    可能引發的錯誤 (Potential Errors):
        - KeyError: 如果 `df` 或 `filtered_grouped_df` 中找不到指定的欄位名稱。
        - IndexError: 如果 `start_col` 或 `end_col` 中的幀索引在 `df` 中無效。
        - ValueError: 如果原始信號長度不足以支持所選的插值方法 (例如，cubic 需要至少4個點，但 interp1d 可能會自動降級或處理邊界)。

    使用範例 (Example Usage):
    # 假設 df 是原始數據, final_groups 是 excludeCenter 的輸出
    # 標準化 'angle_speed_dps' 信號到 101 點
    standardized_speeds = standardize_group_signals(
        df=df,
        filtered_grouped_df=final_groups,
        signal_column_name='angle_speed_dps',
        target_length=101,
        start_col='NEW Frame Start', # 使用優化後的起始點
        end_col='Frame End',
        group_id_col='Group ID'
    )

    # 訪問第一個群組 (假設其 Group ID 是 1) 的標準化速度
    # group_1_speed = standardized_speeds[1]
    # print(group_1_speed.shape) # 應輸出 (101,)
    """
    standardized_signals = {} # 初始化結果字典

    # 檢查必要欄位是否存在
    required_df_cols = [signal_column_name]
    required_grouped_cols = [start_col, end_col, group_id_col]
    if not all(col in df.columns for col in required_df_cols):
        raise KeyError(f"指定的 signal_column_name '{signal_column_name}' 不在 df 中。")
    if not all(col in filtered_grouped_df.columns for col in required_grouped_cols):
        raise KeyError(f"指定的欄位 ({start_col}, {end_col}, {group_id_col}) 並非全部存在於 filtered_grouped_df 中。")


    print(f"開始標準化 '{signal_column_name}' 信號...")
    # 遍歷過濾後的群組 DataFrame
    for _, group_row in filtered_grouped_df.iterrows():
        group_id = group_row[group_id_col]
        try:
            # 獲取起始和結束幀索引 (轉換為整數)
            start_frame = int(group_row[start_col])
            end_frame = int(group_row[end_col])

            # --- 提取信號序列 ---
            # 使用 .iloc 進行基於整數位置的切片
            # 切片範圍 [start_frame, end_frame]，所以結束索引需要 +1
            # 確保 start_frame 不大於 end_frame
            if start_frame > end_frame:
                 print(f"警告：群組 {group_id} 的起始幀 {start_frame} 晚於結束幀 {end_frame}，跳過此群組。")
                 standardized_signals[group_id] = np.full(target_length, np.nan)
                 continue

            # 提取序列值
            # 添加 .values 將 Pandas Series 轉換為 NumPy array
            sequence = df[signal_column_name].iloc[start_frame : end_frame + 1].values

            original_length = len(sequence)

            # --- 處理邊界情況和插值 ---
            standardized_sequence = np.full(target_length, np.nan) # 預設為 NaN

            if original_length == target_length:
                # 長度已符合，直接使用
                standardized_sequence = sequence
            elif original_length >= 2: # 至少需要2個點才能進行插值
                # 創建原始數據和目標數據的 x 軸座標 (標準化到 0 到 1)
                x_original = np.linspace(0, 1, original_length)
                x_target = np.linspace(0, 1, target_length)

                try:
                    # 創建三次樣條插值函數
                    # kind='cubic': 指定三次樣條插值
                    # bounds_error=False: 允許插值目標超出原始數據範圍
                    # fill_value="extrapolate": 對超出範圍的點進行外插 (可能有風險，取決於數據特性)
                    #     如果不想外插，可設為 np.nan 或其他值
                    interp_func = interp1d(x_original, sequence, kind='cubic',
                                           bounds_error=False, fill_value="extrapolate")

                    # 應用插值函數到目標 x 軸座標
                    standardized_sequence = interp_func(x_target)

                except ValueError as e_interp:
                    # 如果 cubic 插值失敗 (例如點數不足4個，雖然 interp1d 可能會降級)
                    print(f"警告：群組 {group_id} (長度 {original_length}) 進行 Cubic 插值時出錯: {e_interp}。嘗試 Linear 插值。")
                    try:
                       # 嘗試使用線性插值作為備選
                       interp_func_linear = interp1d(x_original, sequence, kind='linear',
                                                     bounds_error=False, fill_value="extrapolate")
                       standardized_sequence = interp_func_linear(x_target)
                    except Exception as e_linear:
                       print(f"錯誤：群組 {group_id} 的 Linear 插值也失敗: {e_linear}。該群組結果將為 NaN。")
                       # standardized_sequence 保持為 np.full(target_length, np.nan)

            elif original_length == 1:
                # 只有一個點，用該點的值填充
                print(f"警告：群組 {group_id} 原始長度只有 1，使用該點的值填充目標序列。")
                standardized_sequence = np.full(target_length, sequence[0])
            else: # original_length == 0
                print(f"警告：群組 {group_id} 提取到的序列長度為 0 (可能因 start={start_frame}, end={end_frame} 導致)，結果為 NaN。")
                # standardized_sequence 保持為 np.full(target_length, np.nan)

            # 儲存結果到字典
            standardized_signals[group_id] = standardized_sequence

        except KeyError:
            # 處理在 df 中找不到 signal_column_name 的情況 (雖然前面檢查過，但以防萬一)
            print(f"錯誤：無法在 df 中找到欄位 '{signal_column_name}'。")
            standardized_signals[group_id] = np.full(target_length, np.nan) # 或拋出異常
        except IndexError:
            # 處理幀索引超出 df 範圍的情況
            print(f"錯誤：群組 {group_id} 的幀索引 [{start_frame}, {end_frame}] 超出 df 的範圍。")
            standardized_signals[group_id] = np.full(target_length, np.nan)
        except Exception as e:
            # 捕獲其他潛在錯誤
            print(f"錯誤：處理群組 {group_id} 時發生未知錯誤: {e}")
            standardized_signals[group_id] = np.full(target_length, np.nan)

    print(f"標準化完成。共處理 {len(standardized_signals)} 個群組。")
    return standardized_signals

# %%

# 將單位從mm轉換成視角
df = ConverUnit2Angle(combine_dict, descriptions)
# 2.1. 找出每一次目標擊殺的開槍數 -> 找出Z axis local minimal
# 2.1.1. 以滑鼠點擊次數計算，使用Z軸局部最小值，如果兩次Z軸局部最小值的視角差
#         小於某個閾值，則視為仍在瞄準同一個目標
filtered_minima_data = find_Zaxis_min(df, show=True)
filtered_minima_idx = filtered_minima_data["Frame"].tolist()
# 2.1.2. 找出完成擊殺的 frame 以及上一個視角大於閾值的視角位置
grouped_df = findZminGroup(df, filtered_minima_idx)
# 2.2. 找出從中心出發的開槍軌跡
# 2025.04.30 接下來從這邊開始
excldueCen_grouped_df = excludeCenter_and_plot_separately(df, grouped_df, show=True)

"""
準備做標準化處理

"""

# --- 如何使用 ---
# 假設 df 是包含 'angle_speed_dps' 的原始數據 DataFrame
# 假設 final_groups 是之前 excludeCenter 函數返回的 DataFrame
final_groups = excldueCen_grouped_df
# 檢查 final_groups 是否為空
if not final_groups.empty:
    try:
        standardized_speeds = standardize_group_signals(
            df=df,
            filtered_grouped_df=final_groups,
            signal_column_name='angle_speed_dps', # 指定要標準化的欄位
            target_length=101,                   # 指定目標長度
            start_col='Frame Start',             # 指定起始幀欄位
            end_col='Frame End',                 # 指定結束幀欄位
            group_id_col='Group ID'              # 指定群組ID欄位
        )

        # 查看第一個群組的標準化結果 (假設 Group ID 為 1 存在)
        # if 1 in standardized_speeds:
        #     print("第一個群組的標準化速度序列 (前10個點):")
        #     print(standardized_speeds[1][:10])
        #     print(f"序列長度: {len(standardized_speeds[1])}") # 應為 101

    except KeyError as e:
        print(f"執行標準化時出錯：{e}")
    except ImportError:
        print("錯誤：需要安裝 scipy 庫才能執行插值。請運行 pip install scipy")
else:
      print("沒有可供標準化的群組 (filtered_grouped_df is empty)。")


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

# 將每一筆資料都標準化成固定長度


# === b. Mouse Speed (°/s) ===
max_angle_speed = max(df["angle_speed_dps"])
mean_angle_speed = np.mean(df["angle_speed_dps"])
# === c. Initial Move Angle: ===

mean_initial_move_angle = np.mean(excldueCen_grouped_df['Initial Move Angle (°)'])

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

# %%

import matplotlib.pyplot as plt
import numpy as np
import math

def _process_and_calculate_stats(signals_dict, target_length):
    """(內部輔助函數) 處理信號字典並計算統計數據"""
    if not signals_dict:
        print("警告：提供的信號字典為空。")
        return None, None, None, 0 # 返回 None 表示失敗

    signals_list = list(signals_dict.values())
    if not signals_list:
        print("警告：未能從字典中提取任何有效的信號數組。")
        return None, None, None, 0

    # 過濾並堆疊信號
    valid_signals = [s for s in signals_list if isinstance(s, np.ndarray) and s.shape == (target_length,)]
    if not valid_signals:
         print(f"警告：未能找到任何有效（NumPy 數組且長度為 {target_length}）的信號。")
         return None, None, None, 0

    try:
        signals_array = np.stack(valid_signals, axis=1)
        num_signals = signals_array.shape[1]
    except Exception as e:
        print(f"錯誤：數據準備過程中無法堆疊數組 (檢查長度是否均為 {target_length})：{e}")
        return None, None, None, 0

    # 計算統計數據 (忽略 NaN)
    avg_signal = np.nanmean(signals_array, axis=1)
    std_signal = np.nanstd(signals_array, axis=1)

    # 檢查計算結果是否有效 (例如，如果所有輸入都是 NaN)
    if np.all(np.isnan(avg_signal)) or np.all(np.isnan(std_signal)):
        print(f"警告：計算得到的平均值或標準差全部為 NaN (可能所有輸入信號都無效或全為 NaN)。")
        return None, None, None, num_signals # 即使計算失敗也返回信號數量

    lower_bound = avg_signal - std_signal
    upper_bound = avg_signal + std_signal

    return avg_signal, lower_bound, upper_bound, num_signals

def plot_standardized_signals_cloud_compare(
        signals_dict1,             # 第一個數據集 (必需)
        target_length,             # 信號的標準化長度 (必需)
        signals_dict2=None,        # 第二個數據集 (可選)
        title="Comparison of Mean ± Std Dev Clouds",
        xlabel="Normalized Time (%)",
        ylabel="Signal Value (°/s or other units)",
        label1='Dataset 1',      # 第一個數據集的標籤
        label2='Dataset 2',      # 第二個數據集的標籤
        color_index1=0,            # 第一個數據集的顏色索引
        color_index2=1             # 第二個數據集的顏色索引
    ):
    """
    在同一張圖上繪製一個或兩個標準化信號數據集的平均值和標準差範圍圖。

    參數 (Parameters):
        signals_dict1 (dict):      第一個包含標準化信號的字典 (鍵: ID, 值: 1D NumPy array)。
        target_length (int):       標準化信號的長度。兩個數據集必須相同。
        signals_dict2 (dict, optional): 第二個包含標準化信號的字典。預設為 None。
        title (str):               圖表的標題。
        xlabel (str):              x 軸的標籤。
        ylabel (str):              y 軸的標籤。
        label1 (str):              第一個數據集在圖例中的標籤。
        label2 (str):              第二個數據集在圖例中的標籤 (如果提供 signals_dict2)。
        color_index1 (int):        第一個數據集使用的 'Set1' 調色板顏色索引。
        color_index2 (int):        第二個數據集使用的 'Set1' 調色板顏色索引。
    """

    # --- 創建圖表和 x 軸 ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    iters = np.linspace(0, 100, target_length) # x 軸：0% 到 100%
    palette = plt.get_cmap('Set1')

    plot_success_count = 0

    # --- 處理和繪製第一個數據集 ---
    print(f"處理數據集 1 ({label1})...")
    avg1, lower1, upper1, count1 = _process_and_calculate_stats(signals_dict1, target_length)

    if avg1 is not None: # 確保數據處理和計算成功
        color1 = palette(color_index1 % palette.N)
        ax.plot(iters, avg1, color=color1, label=f'{label1} (n={count1})', linewidth=2)
        ax.fill_between(iters, lower1, upper1, color=color1, alpha=0.2)
        plot_success_count += 1
    else:
        print(f"未能成功處理或計算數據集 1 ({label1}) 的統計數據。")


    # --- 處理和繪製第二個數據集 (如果存在) ---
    if signals_dict2 is not None:
        print(f"\n處理數據集 2 ({label2})...")
        avg2, lower2, upper2, count2 = _process_and_calculate_stats(signals_dict2, target_length)

        if avg2 is not None: # 確保數據處理和計算成功
            color2 = palette(color_index2 % palette.N)
            # 確保顏色不同
            if color_index1 == color_index2:
                print(f"警告：數據集 1 和 2 的顏色索引相同 ({color_index1})。將嘗試使用下一個顏色。")
                color2 = palette((color_index2 + 1) % palette.N)

            ax.plot(iters, avg2, color=color2, label=f'{label2} (n={count2})', linewidth=2)
            ax.fill_between(iters, lower2, upper2, color=color2, alpha=0.2)
            plot_success_count += 1
        else:
           print(f"未能成功處理或計算數據集 2 ({label2}) 的統計數據。")

    # --- 圖表格式設定 ---
    if plot_success_count > 0: # 只有成功繪製了至少一個數據集才進行格式化
        ax.set_title(title, fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, linestyle='-.')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print("\n沒有成功繪製任何數據集，圖表未顯示。")
        plt.close(fig) # 關閉空的圖表窗口


# --- 如何使用 ---
# 假設 standardized_speeds1 和 standardized_speeds2 是兩個包含標準化速度信號的字典
# 假設 target_length = 101
standardized_speeds1 = standardized_speeds
# 示例 1: 只繪製一個數據集
if standardized_speeds1:
      plot_standardized_signals_cloud_compare(
          signals_dict1=standardized_speeds1,
          target_length=101,
          title="數據集 1 的平均速度 ± 標準差",
          label1='實驗組 A',
          ylabel="速度 (°/s)"
      )

# 示例 2: 繪製兩個數據集進行比較
# if standardized_speeds1 and standardized_speeds2:
#      plot_standardized_signals_cloud_compare(
#          signals_dict1=standardized_speeds1,
#          target_length=101,
#          signals_dict2=standardized_speeds2, # 提供第二個字典
#          title="比較兩個數據集的平均速度 ± 標準差",
#          label1='實驗組 A',
#          label2='實驗組 B',        # 為第二個數據集提供標籤
#          ylabel="速度 (°/s)",
#          color_index1=0,         # 第一個用顏色 0
#          color_index2=1          # 第二個用顏色 1
#      )
# else:
#      print("至少需要一個有效的標準化信號字典才能繪圖。")


















