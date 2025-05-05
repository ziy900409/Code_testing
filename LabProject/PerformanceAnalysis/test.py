
z_file = r"C:\Users\Hsin.YH.Yang\Downloads\Z.txt"
manual_min_file = r"C:\Users\Hsin.YH.Yang\Downloads\localmin.txt"
 
 # %%
import numpy as np
from scipy.signal import savgol_filter, find_peaks, argrelextrema
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# --- 參數設定 ---
z_file = r"C:\Users\Hsin.YH.Yang\Downloads\Z.txt"
manual_min_file = r"C:\Users\Hsin.YH.Yang\Downloads\localmin.txt"

# Savitzky-Golay 濾波器參數 (試著調整看看)
baseline_window_length = 51 # 試試 101 或 31?
baseline_polyorder = 3     # 試試 2?

# find_peaks 參數 (啟用 prominence 並調整)
prominence_threshold = 0.1 # <--- *** 試著調整這個值 (例如 0.05, 0.2, 0.5 ...) ***

# argrelextrema 參數 (試著調整看看)
order_param = 5 # 試試 3 或 10?

# --- 1. 載入資料 ---
# (同前)
try:
    z_data = np.loadtxt(z_file, skiprows=1)
    print(f"成功從 '{z_file}' 載入 {len(z_data)} 個數據點。")
except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{z_file}'。")
    exit()
except Exception as e:
    print(f"讀取 '{z_file}' 時發生錯誤：{e}")
    exit()

try:
    manual_indices = np.loadtxt(manual_min_file, dtype=int) - 1
    print(f"成功從 '{manual_min_file}' 載入 {len(manual_indices)} 個手動標記的最小值索引。")
except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{manual_min_file}'。")
    exit()
except Exception as e:
    print(f"讀取 '{manual_min_file}' 時發生錯誤：{e}")
    exit()

if baseline_window_length >= len(z_data):
    print(f"錯誤：baseline_window_length ({baseline_window_length}) 必須小於數據點總數 ({len(z_data)})。")
    baseline_window_length = len(z_data) // 2 * 2 + 1
    if baseline_window_length < baseline_polyorder + 1:
         baseline_window_length = baseline_polyorder + 2 if (baseline_polyorder + 1) % 2 == 0 else baseline_polyorder + 1
    print(f"已自動調整 baseline_window_length 為 {baseline_window_length}")

# --- 2. 資料預處理 (去除基線) ---
# (同前)
try:
    z_baseline = savgol_filter(z_data, baseline_window_length, baseline_polyorder)
    z_detrended = z_data - z_baseline
    print("成功計算基線並進行去除。")
except Exception as e:
    print(f"計算 Savitzky-Golay 濾波時發生錯誤：{e}")
    exit()

# --- 3. 尋找局部最小值 ---
# 方法一：使用 find_peaks 並過濾
try:
    # 使用 prominence 參數，並獲取 properties
    peaks_indices_raw, properties = find_peaks(
        -z_detrended,
        prominence=prominence_threshold
        # 可以加入其他參數, e.g., width=width_threshold, distance=distance_threshold
    )
    print(f"使用 find_peaks (prominence={prominence_threshold}) 找到 {len(peaks_indices_raw)} 個原始峰。")

    # *** 新增：過濾 Z_detrended < 0 的點 ***
    if len(peaks_indices_raw) > 0: # 確保索引不為空
        negative_detrended_mask = z_detrended[peaks_indices_raw] < 0
        peaks_indices_filtered = peaks_indices_raw[negative_detrended_mask]
        # 如果需要，也可以過濾 properties
        # properties_filtered = {k: v[negative_detrended_mask] for k, v in properties.items()}
    else:
        peaks_indices_filtered = np.array([], dtype=int)

    print(f"--> 過濾後 (Z_detrended < 0)，剩下 {len(peaks_indices_filtered)} 個局部最小值。")

except Exception as e:
    print(f"執行 find_peaks 或過濾時發生錯誤：{e}")
    peaks_indices_raw = np.array([], dtype=int)
    peaks_indices_filtered = np.array([], dtype=int)

# 方法二：使用 argrelextrema
# (同前)
try:
    extrema_indices = argrelextrema(z_detrended, np.less, order=order_param)[0]
    print(f"使用 argrelextrema (order={order_param}) 找到 {len(extrema_indices)} 個局部最小值。")
except Exception as e:
    print(f"執行 argrelextrema 時發生錯誤：{e}")
    extrema_indices = np.array([], dtype=int)


# --- 4 & 5. 比較與評估 ---
# (同前 - 但注意 find_peaks 的評估要用 filtered indices)
def evaluate_algorithm(predicted_indices, true_indices, data_length):
    """計算 Precision, Recall, F1-score"""
    pred_set = set(predicted_indices)
    true_set = set(true_indices)
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    y_true = np.zeros(data_length)
    y_pred = np.zeros(data_length)
    if len(true_set) > 0:
      y_true[list(true_set)] = 1
    if len(pred_set) > 0:
      y_pred[list(pred_set)] = 1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return tp, fp, fn, precision, recall, f1

print("\n--- 演算法評估 (與 manual_min.txt 比較) ---")

# *** 評估 find_peaks (使用過濾後的索引) ***
tp_fp, fp_fp, fn_fp, precision_fp, recall_fp, f1_fp = evaluate_algorithm(peaks_indices_filtered, manual_indices, len(z_data))
print(f"Find_Peaks (prominence={prominence_threshold}, Z_detrended<0):") # 更新標題
print(f"  找到的索引: {peaks_indices_filtered[:20]} ... (前 20 個)")
print(f"  TP: {tp_fp}, FP: {fp_fp}, FN: {fn_fp}")
print(f"  Precision: {precision_fp:.4f}")
print(f"  Recall:    {recall_fp:.4f}")
print(f"  F1-score:  {f1_fp:.4f}")

# 評估 argrelextrema (同前)
tp_ar, fp_ar, fn_ar, precision_ar, recall_ar, f1_ar = evaluate_algorithm(extrema_indices, manual_indices, len(z_data))
print(f"\nArgrelextrema (order={order_param}):")
print(f"  找到的索引: {extrema_indices[:20]} ... (前 20 個)")
print(f"  TP: {tp_ar}, FP: {fp_ar}, FN: {fn_ar}")
print(f"  Precision: {precision_ar:.4f}")
print(f"  Recall:    {recall_ar:.4f}")
print(f"  F1-score:  {f1_ar:.4f}")


# --- 6. 判斷最佳演算法 ---
# (同前)
print("\n--- 結論 ---")
if f1_fp > f1_ar:
    print(f"基於 F1-score，find_peaks (prominence={prominence_threshold}, Z_detrended<0) 在此參數設定下表現較好。")
elif f1_ar > f1_fp:
    print(f"基於 F1-score，argrelextrema (order={order_param}) 在此參數設定下表現較好。")
else:
    # 考慮到 find_peaks 多了一步過濾，如果 F1 相同，argrelextrema 可能更直接
    if f1_fp == 0 and f1_ar == 0:
         print("兩種演算法的 F1-score 均為 0。")
    else:
         print("兩種演算法的 F1-score 相同。")


print("\n建議：")
print("1. **調整 `prominence_threshold`**：這是最可能改善 find_peaks 結果的參數。")
print("2. 調整 `baseline_window_length` 和 `order_param`。")
print("3. 觀察圖表：查看過濾後的 find_peaks 點和 argrelextrema 點是否更符合您的預期。")

# --- 可選：繪圖比較 ---
# (繪圖部分程式碼不變，但會顯示過濾後的 find_peaks 結果)
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(z_data, label='原始數據 (Z.txt)', alpha=0.7)
plt.plot(z_baseline, label=f'基線 (savgol win={baseline_window_length}, poly={baseline_polyorder})', linestyle='--')
# Handle potential empty manual_indices for plotting
if len(manual_indices) > 0:
    plt.scatter(manual_indices, z_data[manual_indices], color='red', marker='x', s=100, label='手動標記最小值', zorder=5)
else:
    plt.scatter([], [], color='red', marker='x', s=100, label='手動標記最小值', zorder=5) # Add label even if empty
plt.title('原始數據、基線和手動標記')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(z_detrended, label='去基線數據 (Z_detrended)')
plt.axhline(0, color='gray', linestyle=':', linewidth=0.8) # 添加 y=0 的參考線
# Handle potential empty indices for plotting
if len(manual_indices) > 0:
    plt.scatter(manual_indices, z_detrended[manual_indices], color='red', marker='x', s=100, label='手動標記 (對應去基線)', zorder=5)
else:
     plt.scatter([], [], color='red', marker='x', s=100, label='手動標記 (對應去基線)', zorder=5)
if len(peaks_indices_filtered) > 0:
    plt.scatter(peaks_indices_filtered, z_detrended[peaks_indices_filtered], color='purple', marker='v', s=60, label=f'find_peaks 過濾後 ({len(peaks_indices_filtered)})', alpha=0.7)
else:
    plt.scatter([], [], color='purple', marker='v', s=60, label=f'find_peaks 過濾後 (0)', alpha=0.7)
if len(extrema_indices) > 0:
    plt.scatter(extrema_indices, z_detrended[extrema_indices], color='green', marker='o', s=40, label=f'argrelextrema ({len(extrema_indices)})', alpha=0.7)
else:
    plt.scatter([], [], color='green', marker='o', s=40, label=f'argrelextrema (0)', alpha=0.7)

plt.title('去基線數據與演算法找到的最小值 (find_peaks 已過濾 Z_detrended < 0)')
plt.legend()
plt.grid(True)
plt.xlabel('索引')
plt.tight_layout()
plt.show()

# %%

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, argrelextrema
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings # To suppress potential warnings if needed
plt.rcParams['font.sans-serif'] = ['Noto Sans TC']  # 改為你實際有的
plt.rcParams['axes.unicode_minus'] = False    # 避免座標軸負號亂碼

# Helper function for evaluation (必須在主函數外部或內部定義)
def evaluate_algorithm(predicted_indices, true_indices, data_length):
    """計算 Precision, Recall, F1-score"""
    # Handle potential empty inputs
    if not isinstance(predicted_indices, (list, np.ndarray)) or len(predicted_indices) == 0:
        pred_set = set()
    else:
        pred_set = set(predicted_indices)

    if not isinstance(true_indices, (list, np.ndarray)) or len(true_indices) == 0:
        true_set = set()
    else:
        true_set = set(true_indices)

    if not true_set and not pred_set:
        # Handle case where both true and predicted are empty
        tp, fp, fn = 0, 0, 0
        precision, recall, f1 = 1.0, 1.0, 1.0 # Or 0.0 depending on convention
    elif not true_set:
        # Handle case where true is empty but predicted is not
        tp, fp, fn = 0, len(pred_set), 0
        precision, recall, f1 = 0.0, 0.0, 0.0
    elif not pred_set:
         # Handle case where predicted is empty but true is not
        tp, fp, fn = 0, 0, len(true_set)
        precision, recall, f1 = 0.0, 0.0, 0.0
    else:
        tp = len(pred_set.intersection(true_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        # Use sklearn calculation, requires binary arrays
        y_true = np.zeros(data_length, dtype=int)
        y_pred = np.zeros(data_length, dtype=int)
        y_true[list(true_set)] = 1
        y_pred[list(pred_set)] = 1

        # Use zero_division=0 to avoid warnings and return 0 in case of undefined metric
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": precision, "Recall": recall, "F1-score": f1
    }
    return metrics

def find_Zaxis_min_combined( # Renamed function
        df,                      # Input DataFrame
        manual_min_file,         # Path to manual labels file ('localmin.txt')
        method='argrelextrema',  # Algorithm: 'argrelextrema' or 'find_peaks'
        # --- Baseline Removal Params ---
        use_baseline_removal=True,
        baseline_window_length=51,
        baseline_polyorder=3,
        # --- Minima Finding Params ---
        # argrelextrema specific
        order=5,
        # find_peaks specific
        prominence_threshold=0.1,
        filter_find_peaks_below_zero=True,
        # --- Filtering Params ---
        # Original threshold logic replaced by z_value_threshold applied to detrended/raw Z
        z_value_threshold=None, # e.g., 0 or -0.1. If None, this filter is skipped.
        # Original custom filter logic
        use_custom_filter=True,
        min_frame_gap=8,
        min_z_diff=0.2,
        # --- Output Params ---
        show=True,
        showVel=True):
    """
    結合了基線移除、不同最小值搜尋演算法 (argrelextrema, find_peaks)、
    多重過濾條件以及與手動標籤比較評估的功能。

    Parameters:
        df: DataFrame, 包含 'Z' 欄位以及繪圖所需的 'cum_yaw_deg', 'cum_pitch_deg', 'speed'。
        manual_min_file: str, 手動標記最小值索引檔案路徑 (每行一個索引，基於 1)。
        method: str, 使用的演算法 ('argrelextrema' 或 'find_peaks')。
        use_baseline_removal: bool, 是否進行基線移除。
        baseline_window_length: int, Savitzky-Golay 濾波窗口。
        baseline_polyorder: int, Savitzky-Golay 濾波多項式階數。
        order: int, `argrelextrema` 的 order 參數。
        prominence_threshold: float, `find_peaks` 的 prominence 參數。
        filter_find_peaks_below_zero: bool, 是否只保留 `find_peaks` 找到的 Z < 0 的點。
        z_value_threshold: float or None, 最小值 Z 值的門檻 (作用於去基線後或原始 Z)。
        use_custom_filter: bool, 是否啟用自訂的間隔/差異過濾。
        min_frame_gap: int, 自訂過濾：最小 frame 間隔。
        min_z_diff: float, 自訂過濾：最小 Z 值差異。
        show: bool, 是否顯示 Z 值和視角軌跡圖。
        showVel: bool, 是否顯示速度著色的視角軌跡圖。

    Returns:
        final_minima_idx: list, 最終篩選出的最小值索引 (基於 0)。
        filtered_minima_data: DataFrame, 包含最終最小值點對應的視角等資訊。
        evaluation_metrics: dict, 與手動標籤比較的評估結果。
    """
    print(f"\n--- 開始分析：使用方法 '{method}' ---")
    if not all(col in df.columns for col in ['Z', 'cum_yaw_deg', 'cum_pitch_deg', 'speed']):
         warnings.warn("DataFrame 缺少必要的欄位 ('Z', 'cum_yaw_deg', 'cum_pitch_deg', 'speed')，部分功能可能無法運作或出錯。")

    z_values_raw = df["Z"].values.copy() # 使用 .copy() 避免修改原始 df
    data_length = len(z_values_raw)
    evaluation_metrics = {} # Initialize evaluation metrics

    # --- 1. Baseline Removal (Optional) ---
    z_baseline = np.zeros_like(z_values_raw) # Default baseline if not calculated
    if use_baseline_removal:
        print(f"步驟 1: 應用 Savitzky-Golay 基線移除 (窗口={baseline_window_length}, 階數={baseline_polyorder})")
        # Check window length validity
        if baseline_window_length >= data_length:
            original_wl = baseline_window_length
            baseline_window_length = data_length // 2 * 2 + 1 # Adjust to largest odd number <= length/2
            if baseline_window_length < 3: baseline_window_length = 3 # Minimum practical window
            # Ensure window > polyorder
            if baseline_window_length <= baseline_polyorder:
                 baseline_window_length = baseline_polyorder + 1 if baseline_polyorder % 2 == 0 else baseline_polyorder + 2
            print(f"  警告: baseline_window_length ({original_wl}) >= data length ({data_length})。已自動調整為 {baseline_window_length}")

        try:
            z_baseline = savgol_filter(z_values_raw, baseline_window_length, baseline_polyorder)
            z_values_processed = z_values_raw - z_baseline # Processed Z = Detrended Z
            print("  基線移除完成。")
        except Exception as e:
            print(f"  錯誤: 基線移除失敗: {e}。將使用原始 Z 值進行後續處理。")
            z_values_processed = z_values_raw # Fallback to raw Z
            use_baseline_removal = False # Disable flag to reflect reality
    else:
        print("步驟 1: 跳過基線移除。")
        z_values_processed = z_values_raw # Processed Z = Raw Z

    # --- 2. Find Initial Local Minima ---
    print(f"步驟 2: 使用 '{method}' 尋找初始局部最小值")
    initial_minima_idx = np.array([], dtype=int) # Ensure it's always an array

    if method == 'argrelextrema':
        try:
            initial_minima_idx = argrelextrema(z_values_processed, np.less, order=order)[0]
            print(f"  argrelextrema (order={order}) 找到 {len(initial_minima_idx)} 個初始點。")
        except Exception as e:
            print(f"  錯誤: argrelextrema 執行失敗: {e}")
    elif method == 'find_peaks':
        try:
            # find_peaks finds peaks, so use negative data for minima
            peaks_indices_raw, _ = find_peaks(-z_values_processed, prominence=prominence_threshold)
            print(f"  find_peaks (prominence={prominence_threshold}) 找到 {len(peaks_indices_raw)} 個原始點。")
            if filter_find_peaks_below_zero:
                if len(peaks_indices_raw) > 0:
                    # Filter based on the processed Z value (detrended or raw)
                    below_zero_mask = z_values_processed[peaks_indices_raw] < 0
                    initial_minima_idx = peaks_indices_raw[below_zero_mask]
                    print(f"  --> 已過濾 Z < 0 的點，剩下: {len(initial_minima_idx)}")
                # else: initial_minima_idx remains empty array
            else:
                initial_minima_idx = peaks_indices_raw
        except Exception as e:
            print(f"  錯誤: find_peaks 執行失敗: {e}")
    else:
        print(f"  錯誤: 未知的 method '{method}'。請選擇 'argrelextrema' 或 'find_peaks'.")
        # Return empty results if method is invalid
        return [], pd.DataFrame(), evaluation_metrics

    # Ensure initial_minima_idx is always a numpy array for consistency
    if not isinstance(initial_minima_idx, np.ndarray):
         initial_minima_idx = np.array(initial_minima_idx, dtype=int)

    # --- 3. Filter by Z Value Threshold (Optional) ---
    filtered_minima_idx_step3 = initial_minima_idx # Start with results from step 2
    if z_value_threshold is not None:
        print(f"步驟 3: 過濾 Z 值低於 {z_value_threshold:.4f} 的點")
        if len(initial_minima_idx) > 0:
            threshold_mask = z_values_processed[initial_minima_idx] < z_value_threshold
            filtered_minima_idx_step3 = initial_minima_idx[threshold_mask]
            print(f"  --> 過濾後剩下: {len(filtered_minima_idx_step3)}")
        else:
             print("  --> 無初始點可供過濾。")
        # else: filtered_minima_idx_step3 remains empty if initial_minima_idx was empty
    else:
        print("步驟 3: 跳過 Z 值門檻過濾。")


    # --- 4. Apply Custom Gap/Difference Filtering (Optional) ---
    final_minima_idx_step4 = filtered_minima_idx_step3 # Start with results from step 3
    if use_custom_filter:
        print(f"步驟 4: 應用自訂過濾 (最小間隔={min_frame_gap}, 最小 Z 差={min_z_diff})")
        if len(filtered_minima_idx_step3) > 0:
            # Sort indices first
            sorted_indices = np.sort(filtered_minima_idx_step3)
            temp_final_indices = [sorted_indices[0]] # Add the first one

            for i in range(1, len(sorted_indices)):
                current_idx = sorted_indices[i]
                last_added_idx = temp_final_indices[-1]
                frame_diff = current_idx - last_added_idx

                if frame_diff >= min_frame_gap:
                    temp_final_indices.append(current_idx)
                else:
                    # Compare based on processed Z value (detrended or raw)
                    z_diff = abs(z_values_processed[current_idx] - z_values_processed[last_added_idx])
                    if z_diff < min_z_diff:
                        # If difference is small, only keep the lower one
                        if z_values_processed[current_idx] < z_values_processed[last_added_idx]:
                            temp_final_indices[-1] = current_idx # Replace the last added index
                        # else: keep the last_added_idx (do nothing)
                    else:
                        # If difference is large enough, keep both
                        temp_final_indices.append(current_idx)
            final_minima_idx_step4 = temp_final_indices
            print(f"  --> 自訂過濾後剩下: {len(final_minima_idx_step4)}")
        else:
            print("  --> 無點可供自訂過濾。")
            final_minima_idx_step4 = [] # Ensure it's a list if empty
    else:
        print("步驟 4: 跳過自訂過濾。")
        # Ensure output is a list if custom filter skipped
        final_minima_idx_step4 = list(filtered_minima_idx_step3)


    # Final indices are the result of step 4
    final_minima_idx = final_minima_idx_step4

    # --- 5. Prepare Output DataFrame ---
    print("步驟 5: 準備輸出 DataFrame")
    if final_minima_idx and len(final_minima_idx) > 0:
        # Check if required columns exist before accessing iloc
        required_cols = ['cum_yaw_deg', 'cum_pitch_deg']
        if all(col in df.columns for col in required_cols):
             filtered_minima_data = pd.DataFrame({
                 "Frame": final_minima_idx,
                 "Z Value Raw": z_values_raw[final_minima_idx], # Always report raw Z
                 "Z Processed": z_values_processed[final_minima_idx], # Report processed Z (detrended or raw)
                 "Yaw Angle (°)": df["cum_yaw_deg"].iloc[final_minima_idx].values,
                 "Pitch Angle (°)": df["cum_pitch_deg"].iloc[final_minima_idx].values
             })
             print("  最終篩選出的最小值點 (部分):")
             print(filtered_minima_data.head())
        else:
             print("  警告: DataFrame 缺少 'cum_yaw_deg' 或 'cum_pitch_deg' 欄位，無法包含角度資訊。")
             filtered_minima_data = pd.DataFrame({
                 "Frame": final_minima_idx,
                 "Z Value Raw": z_values_raw[final_minima_idx],
                 "Z Processed": z_values_processed[final_minima_idx]
             })
             print(filtered_minima_data.head())
    else:
        print("  未找到符合所有條件的最終最小值點。")
        filtered_minima_data = pd.DataFrame(columns=["Frame", "Z Value Raw", "Z Processed", "Yaw Angle (°)", "Pitch Angle (°)"])


    # --- 6. Evaluation Against Manual Labels ---
    print(f"步驟 6: 與手動標籤檔案 '{manual_min_file}' 比較")
    try:
        # Load manual indices (assuming 1-based)
        manual_indices = np.loadtxt(manual_min_file, dtype=int) - 1
        print(f"  載入 {len(manual_indices)} 個手動標籤。")

        # Ensure indices are within bounds
        manual_indices = manual_indices[(manual_indices >= 0) & (manual_indices < data_length)]
        if len(manual_indices) == 0:
             print("  警告: 手動標籤檔案中無有效索引。")

        evaluation_metrics = evaluate_algorithm(final_minima_idx, manual_indices, data_length)
        print("  評估指標:")
        for key, value in evaluation_metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
    except FileNotFoundError:
        print(f"  錯誤: 找不到手動標籤檔案 '{manual_min_file}'。跳過評估。")
        evaluation_metrics = {} # Ensure it's empty dict on error
    except Exception as e:
        print(f"  錯誤: 評估過程中發生錯誤: {e}")
        evaluation_metrics = {} # Ensure it's empty dict on error


    # --- 7. Plotting ---
    if show:
        print("步驟 7: 產生圖表")
        plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True) # Share x-axis

        # Plot 1: Raw Data, Baseline, Thresholds, Final Minima
        axes[0].plot(z_values_raw, label='原始 Z 值', color='gray', alpha=0.7, linewidth=1)
        if use_baseline_removal:
            axes[0].plot(z_baseline, label=f'基線 (win={baseline_window_length}, poly={baseline_polyorder})', color='orange', linestyle='--', linewidth=1.5)
        # Plot Z threshold only if applied to raw Z
        if z_value_threshold is not None and not use_baseline_removal:
             axes[0].axhline(z_value_threshold, color='cyan', linestyle=':', label=f'Z 值門檻 ({z_value_threshold:.2f})', linewidth=1.5)
        # Plot final minima on raw data
        if final_minima_idx and len(final_minima_idx) > 0:
             axes[0].scatter(final_minima_idx, z_values_raw[final_minima_idx], color='red', label=f'最終最小值 ({len(final_minima_idx)})', zorder=5, s=60, marker='x')
        axes[0].set_title(f"原始 Z 值、基線與最終最小值點 (方法: {method})")
        axes[0].set_ylabel("原始 Z 值")
        axes[0].legend()
        axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)


        # Plot 2: Processed Data, Thresholds, Final Minima
        axes[1].plot(z_values_processed, label='處理後 Z 值' + (' (去基線)' if use_baseline_removal else ' (原始)'), color='blue', alpha=0.8, linewidth=1)
        # Plot Z=0 line if filtered below zero for find_peaks
        if method == 'find_peaks' and filter_find_peaks_below_zero:
             axes[1].axhline(0, color='magenta', linestyle=':', label='Z=0 過濾線 (find_peaks)', linewidth=1.5)
        # Plot Z threshold if applied to processed Z
        if z_value_threshold is not None: # Check if threshold exists, applies to processed Z here
            axes[1].axhline(z_value_threshold, color='cyan', linestyle=':', label=f'Z 值門檻 ({z_value_threshold:.2f})', linewidth=1.5)
        # Plot final minima on processed data
        if final_minima_idx and len(final_minima_idx) > 0:
             axes[1].scatter(final_minima_idx, z_values_processed[final_minima_idx], color='red', label=f'最終最小值 ({len(final_minima_idx)})', zorder=5, s=60, marker='x')
        axes[1].set_title(f"處理後 Z 值與最終最小值點")
        axes[1].set_ylabel("處理後 Z 值")
        axes[1].legend()
        axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)


        # Plot 3: Add manual labels for reference
        try:
             manual_indices_plot = np.loadtxt(manual_min_file, dtype=int) - 1
             manual_indices_plot = manual_indices_plot[(manual_indices_plot >= 0) & (manual_indices_plot < data_length)]
             if len(manual_indices_plot) > 0:
                  axes[1].scatter(manual_indices_plot, z_values_processed[manual_indices_plot],
                                  facecolors='none', edgecolors='lime', s=100, linewidth=1.5,
                                  label=f'手動標籤 ({len(manual_indices_plot)})', zorder=4, marker='o')
                  # Also show on raw plot for context
                  axes[0].scatter(manual_indices_plot, z_values_raw[manual_indices_plot],
                                  facecolors='none', edgecolors='lime', s=100, linewidth=1.5,
                                  label=f'手動標籤 ({len(manual_indices_plot)})', zorder=4, marker='o')
                  # Update legends after adding manual points potentially
                  axes[0].legend()
                  axes[1].legend()
        except Exception as e:
             print(f"  繪製手動標籤時出錯: {e}")

        axes[2].set_xlabel("Frame") # Only set x-label on the bottom plot
        plt.tight_layout()
        plt.show()

        # --- View Angle Plots ---
        required_angle_cols = ['cum_yaw_deg', 'cum_pitch_deg']
        has_angle_data = all(col in df.columns for col in required_angle_cols)

        if has_angle_data:
            if final_minima_idx and len(final_minima_idx) > 0:
                filtered_yaw = df["cum_yaw_deg"].iloc[final_minima_idx].values
                filtered_pitch = df["cum_pitch_deg"].iloc[final_minima_idx].values
            else:
                filtered_yaw, filtered_pitch = [], []

            # Plot 8: Basic Trajectory
            plt.figure(figsize=(8, 8))
            plt.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"], c=df.index, cmap="viridis", alpha=0.6, s=10, label="視角軌跡 (依 Frame)")
            plt.scatter(filtered_pitch, filtered_yaw, color="red", s=50, label="最終最小值", zorder=3, marker='x')
            plt.colorbar(label="Frame Index")
            plt.xlabel("Pitch Angle (Vertical) °")
            plt.ylabel("Yaw Angle (Horizontal, Rotated) °")
            plt.title(f"視角軌跡與 Z 軸最終最小值 ({method})")
            plt.legend()
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.show()

            # Plot 9: Velocity Colored Trajectory
            if showVel and 'speed' in df.columns:
                plt.figure(figsize=(8, 8))
                sc = plt.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"],
                                 c=df["speed"], cmap="plasma", alpha=0.7, s=10,
                                 label="視角軌跡 (依速度)")
                plt.scatter(filtered_pitch, filtered_yaw, color="red", s=50,
                            label="最終最小值", zorder=3, marker='x')
                plt.colorbar(sc, label="滑鼠速度 (單位未知)") # Speed unit might vary
                plt.xlabel("Pitch Angle (Vertical) °")
                plt.ylabel("Yaw Angle (Horizontal, Rotated) °")
                plt.title(f"視角軌跡 (依速度) 與 Z 軸最終最小值 ({method})")
                plt.legend()
                plt.grid(True, linestyle='--', linewidth=0.5)
                plt.show()
            elif showVel and 'speed' not in df.columns:
                 print("  警告: DataFrame 中缺少 'speed' 欄位，無法繪製速度著色圖。")
        else:
            print("  跳過視角軌跡繪圖，因缺少 'cum_yaw_deg' 或 'cum_pitch_deg' 欄位。")

    print(f"--- '{method}' 分析完成 ---")
    return final_minima_idx, filtered_minima_data, evaluation_metrics


# --- 範例使用 ---
if __name__ == "__main__":
    # 1. 載入您的 DataFrame (假設為 df)
    #    確保它包含 'Z', 'cum_yaw_deg', 'cum_pitch_deg', 'speed' 欄位
    #    這裡使用 dummy data 示範
    try:
        z_file = r"C:\Users\Hsin.YH.Yang\Downloads\Z.txt"
        manual_min_file = r"C:\Users\Hsin.YH.Yang\Downloads\localmin.txt"
        # 從 Z.txt 載入數據來創建基礎 DataFrame
        z_data_for_df = np.loadtxt(z_file, skiprows=1)
        n_points = len(z_data_for_df)
        df_main = pd.DataFrame({
            'Z': z_data_for_df,
            # 創建更真實的角度和速度數據 (例如模擬一些運動)
            'cum_yaw_deg': np.cumsum(np.random.randn(n_points) * 0.5) % 360,
            'cum_pitch_deg': np.cumsum(np.random.randn(n_points) * 0.3),
            'speed': np.abs(np.random.randn(n_points) * 10 + 5) # 模擬速度
        })
        # 限制 pitch 範圍
        df_main['cum_pitch_deg'] = np.clip(df_main['cum_pitch_deg'], -90, 90)
        print(f"已創建包含 {n_points} 點的範例 DataFrame。")
    except FileNotFoundError:
        print("錯誤: 找不到 'Z.txt' 檔案，無法創建範例 DataFrame。請將 Z.txt 放在腳本同目錄下。")
        exit()
    except Exception as e:
        print(f"創建範例 DataFrame 時發生錯誤: {e}")
        exit()

    # 2. 設定手動標籤檔案路徑
    manual_labels_file = manual_min_file

    # --- 3. 分別執行和比較不同設定 ---

    # 設定 1: 使用 argrelextrema，包含基線移除和自訂過濾
    print("\n" + "="*20 + " 設定 1: Argrelextrema (基線+自訂過濾) " + "="*20)
    indices_ar1, data_ar1, metrics_ar1 = find_Zaxis_min_combined(
        df=df_main.copy(), # 使用 .copy() 避免意外修改
        manual_min_file=manual_labels_file,
        method='argrelextrema',
        use_baseline_removal=True, baseline_window_length=51, baseline_polyorder=3,
        order=5,
        use_custom_filter=True, min_frame_gap=8, min_z_diff=0.2,
        z_value_threshold=None, # 不使用 Z 值門檻
        show=True, showVel=True
    )

    # 設定 2: 使用 find_peaks，包含基線移除和 Z<0 過濾，不使用自訂過濾
    print("\n" + "="*20 + " 設定 2: Find_Peaks (基線+Z<0過濾) " + "="*20)
    indices_fp1, data_fp1, metrics_fp1 = find_Zaxis_min_combined(
        df=df_main.copy(),
        manual_min_file=manual_labels_file,
        method='find_peaks',
        use_baseline_removal=True, baseline_window_length=51, baseline_polyorder=3,
        prominence_threshold=0.05, # 嘗試較小的 prominence
        filter_find_peaks_below_zero=True,
        use_custom_filter=False, # 關閉自訂過濾
        z_value_threshold=None,
        show=True, showVel=True
    )

    # 設定 3: 使用 find_peaks，包含基線移除、Z<0 過濾 和 自訂過濾
    print("\n" + "="*20 + " 設定 3: Find_Peaks (基線+Z<0+自訂過濾) " + "="*20)
    indices_fp2, data_fp2, metrics_fp2 = find_Zaxis_min_combined(
        df=df_main.copy(),
        manual_min_file=manual_labels_file,
        method='find_peaks',
        use_baseline_removal=True, baseline_window_length=101, # 嘗試不同基線窗口
        baseline_polyorder=3,
        prominence_threshold=0.1, # 調整 prominence
        filter_find_peaks_below_zero=True,
        use_custom_filter=True, min_frame_gap=10, min_z_diff=0.15, # 調整自訂過濾參數
        z_value_threshold=-0.05, # 加上 Z 值門檻試試
        show=True, showVel=True
    )


    # --- 4. 最終比較 ---
    print("\n" + "="*40 + " 最終指標比較 " + "="*40)
    print(f"設定 1 (Argrelextrema, Baseline, CustomFilter): F1 = {metrics_ar1.get('F1-score', 'N/A'):.4f}")
    print(f"設定 2 (Find_Peaks, Baseline, Z<0 Filter):      F1 = {metrics_fp1.get('F1-score', 'N/A'):.4f}")
    print(f"設定 3 (Find_Peaks, Baseline, Z<0, Custom, Z_thresh): F1 = {metrics_fp2.get('F1-score', 'N/A'):.4f}")

    # 可以在這裡加入邏輯來找出 F1 最高的設定
    best_f1 = -1
    best_setting = "N/A"
    results = {
        "Setting 1": metrics_ar1.get('F1-score'),
        "Setting 2": metrics_fp1.get('F1-score'),
        "Setting 3": metrics_fp2.get('F1-score'),
    }
    for setting, f1 in results.items():
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_setting = setting

    if best_f1 > -1:
         print(f"\n表現最佳的設定 (基於 F1-score): {best_setting} (F1 = {best_f1:.4f})")
    else:
         print("\n無法確定最佳設定 (可能評估失敗或 F1 為 0)。")




