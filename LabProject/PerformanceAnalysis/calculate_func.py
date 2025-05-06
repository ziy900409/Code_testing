# -*- coding: utf-8 -*-
"""
Created on Mon May  5 19:07:40 2025

@author: Hsin.YH.Yang
"""
import pandas as pd
import numpy as np

# %%

def cal_tra_efficiency(df: pd.DataFrame,
                       grouped_df: pd.DataFrame) -> pd.DataFrame:
    """
    計算每個群組的實際視角軌跡長度與理想直線距離的比值。

    Args:
        df (pd.DataFrame): 包含原始數據的 DataFrame，必須包含 'cum_pitch_deg' 和 'cum_yaw_deg' 欄位，
                           且其索引應能對應 grouped_df 中 'Frames' 欄位的數值。
        grouped_df (pd.DataFrame): 包含擊殺群組資訊的 DataFrame，必須包含 'Frames' 欄位，
                                   其中 'Frames' 是一個包含幀索引 (float 或 int) 的列表。

    Returns:
        pd.DataFrame: 原始的 grouped_df，並增加了 'Actual Path Length',
                      'Ideal Path Length', 和 'Efficiency Ratio' 三個欄位。
                      如果計算失敗（例如幀數據不足或缺失），這些欄位的值可能為 NaN 或 Inf。
    """
    # 檢查必要欄位是否存在
    if not all(col in df.columns for col in ['cum_pitch_deg', 'cum_yaw_deg']):
        print("錯誤：主要 DataFrame (df) 缺少 'cum_pitch_deg' 或 'cum_yaw_deg' 欄位。")
        return grouped_df # 或者拋出異常

    if 'Frames' not in grouped_df.columns:
        print("錯誤：群組 DataFrame (grouped_df) 缺少 'Frames' 欄位。")
        return grouped_df

    actual_lengths = []
    ideal_lengths = []
    ratios = []
    mean_vel_1shot = []
    max_vel_1shot = []
    
    print("開始計算每個群組的軌跡效率...")
    # 迭代每個群組
    for index, row in grouped_df.iterrows():
        frames_list = row['Frames']
        act_len, ideal_len, ratio = np.nan, np.nan, np.nan # 初始化為 NaN

        # 檢查 Frames 列表是否有效且至少有兩個點
        if isinstance(frames_list, list) and len(frames_list) >= 2:
            try:
                # 將 frames 列表中的 float 轉為 int (假設它們代表幀索引)
                # 並確保它們存在於 df 的索引中
                # valid_frame_indices = df.index.intersection([int(f) for f in frames_list if pd.notna(f)])
                start_point = int(frames_list[0])
                first_end = int(frames_list[1])
                end_point = int(frames_list[-1])

                if (end_point - start_point) >= 2:
                    # 計算低一次開槍的移動速度
                    trial_speed = df.loc[start_point:first_end, "speed"]
                    # 提取這個群組所有有效幀的軌跡數據
                    trajectory = df.loc[start_point:end_point, ['cum_pitch_deg', 'cum_yaw_deg']]

                    # 計算實際路徑長度 (連續點之間的距離總和)
                    # diff() 計算相鄰行的差值，第一行為 NaN
                    # np.hypot(dx, dy) 計算每個點到前一個點的歐幾里得距離
                    segment_distances = np.hypot(trajectory['cum_pitch_deg'].diff(),
                                                 trajectory['cum_yaw_deg'].diff())
                    act_len = np.nansum(segment_distances) # 使用 nansum 忽略第一個 NaN

                    # 獲取起始點和結束點的座標
                    start_point = trajectory.iloc[0]
                    end_point = trajectory.iloc[-1]

                    # 計算理想路徑長度 (起始點到結束點的直線距離)
                    ideal_len = np.hypot(end_point['cum_pitch_deg'] - start_point['cum_pitch_deg'],
                                         end_point['cum_yaw_deg'] - start_point['cum_yaw_deg'])

                    # 計算比值，處理 ideal_len 為 0 的情況
                    if ideal_len > 1e-6: # 使用一個小閾值避免浮點數精度問題
                        ratio = ideal_len / act_len
                    elif act_len < 1e-6: # 如果實際長度也很小 (幾乎沒移動)
                        ratio = 1.0      # 可以定義為效率為 1
                    else: # 起點終點相同，但中間有移動
                        ratio = np.inf   # 效率視為無限大（或極低）
                else:
                    print(f"警告：群組 {index} 在 df 中有效的幀少於 2 個，無法計算路徑。")

            except (ValueError, TypeError) as e:
                print(f"警告：處理群組 {index} 的幀列表 {frames_list} 時出錯: {e}")
            except KeyError as e:
                 print(f"警告：嘗試從 df 獲取群組 {index} 的幀時出錯 (可能部分幀不在 df 索引中): {e}")

        else:
             print(f"警告：群組 {index} 的 'Frames' 數據無效或長度不足 2。")

        actual_lengths.append(act_len)
        ideal_lengths.append(ideal_len)
        ratios.append(ratio)
        mean_vel_1shot.append(trial_speed.mean())
        max_vel_1shot.append(trial_speed.max())

    # 將計算結果添加為新的欄位
    grouped_df_out = grouped_df.copy() # 避免修改原始傳入的 DataFrame
    grouped_df_out['Actual Path Length'] = actual_lengths
    grouped_df_out['Ideal Path Length'] = ideal_lengths
    grouped_df_out['Efficiency Ratio'] = ratios # 比值 < 1 表示實際路徑比直線長
    grouped_df_out['mean_vel_1shot'] = mean_vel_1shot
    grouped_df_out['max_vel_1shot'] = max_vel_1shot

    print("軌跡效率計算完成。")
    return grouped_df_out








