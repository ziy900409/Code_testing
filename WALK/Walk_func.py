# -*- coding: utf-8 -*-
"""
Created on Tue May 20 21:32:50 2025

@author: User
"""
import re
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt 

plt.rcParams['font.sans-serif'] =  ['Noto Sans TC']  # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號



# %%

def parse_excel_expression_recursively(expression_str, df, eval_locals, col_name_map): # <--- 新增 col_name_map
    """
    (輔助函式) 遞迴地解析 Excel 表達式，處理 IF 組件中的巢狀結構。
    將儲存格參照轉換為適合 eval() 的變數。
    """
    processed_expression = expression_str
    # 正則表達式用於匹配儲存格參照，例如 A1, A$1, AB1, AB$1
    # 群組 1: 欄位字母 (例如 "D")
    # 群組 2: "$" 符號 (若存在，表示固定列)
    # 群組 3: 列號 (例如 "2")
    cell_pattern = re.compile(r'([A-Z]+)(\$?)(\d+)') # 假設公式中的代號仍為大寫字母

    replacements = []
    for match in cell_pattern.finditer(expression_str):
        formula_col_letter = match.group(1) # 例如 'D' from D2 or D$2
        fixed_marker = match.group(2)
        row_number_str = match.group(3)
        original_cell_ref = match.group(0)

        # 從 col_name_map 獲取實際的 DataFrame 欄位名稱
        actual_df_col_name = col_name_map.get(formula_col_letter)
        if not actual_df_col_name:
            raise KeyError(f"公式中的代號 '{formula_col_letter}' (來自 '{original_cell_ref}') 在 col_name 映射字典中找不到對應的實際欄位名稱。")

        if fixed_marker: # 固定列參照，例如 D$2
            excel_row_num = int(row_number_str)
            pandas_row_index = excel_row_num - 2

            if not (0 <= pandas_row_index < len(df)):
                raise ValueError(f"固定列 {excel_row_num} (來自 '{original_cell_ref}') 超出 DataFrame 的範圍 (0-{len(df)-1})。")
            if actual_df_col_name not in df.columns: # <--- 使用 actual_df_col_name 檢查
                raise KeyError(f"實際欄位 '{actual_df_col_name}' (對應公式代號 '{formula_col_letter}' 的固定參照 '{original_cell_ref}') 在 DataFrame 中找不到。")

            fixed_value = df.loc[pandas_row_index, actual_df_col_name] # <--- 使用 actual_df_col_name 讀取
            if not isinstance(fixed_value, (int, float, complex)):
                raise ValueError(f"來自固定儲存格 '{original_cell_ref}' (實際欄位 '{actual_df_col_name}'，值: {fixed_value}) 的資料非數值型。")

            # 為這個固定值產生一個唯一的變數名稱
            var_name_for_fixed_val = f"__fixed_{formula_col_letter}_r{excel_row_num}" # 保持使用公式代號以簡化
            if var_name_for_fixed_val not in eval_locals:
                 eval_locals[var_name_for_fixed_val] = fixed_value
            replacements.append({'start': match.start(), 'end': match.end(), 'replacement_text': var_name_for_fixed_val})
        else: # 相對列參照，例如 D2 (我們將其視為對應的整個欄位 Series)
            if actual_df_col_name not in df.columns: # <--- 使用 actual_df_col_name 檢查
                raise KeyError(f"實際欄位 '{actual_df_col_name}' (對應公式代號 '{formula_col_letter}' 的相對參照 '{original_cell_ref}') 在 DataFrame 中找不到。")

            # 在 eval_locals 中，我們希望公式中的 'D' 直接對應到 df[actual_df_col_name]
            # 所以，我們需要確保 eval_locals 有一個鍵 'D' (formula_col_letter)
            # 且其值是 df[actual_df_col_name] (對應的 Series)
            # 這一步會在 apply_excel_formulas_v4 的開頭處理 eval_locals 的初始化

            # 替換文字應該是公式中使用的代號 (例如 'D')，因為 eval_locals 將被設定為 'D': df[actual_df_col_name]
            replacements.append({'start': match.start(), 'end': match.end(), 'replacement_text': formula_col_letter})


    # 從後往前替換，以保持 start/end 索引的有效性
    replacements.sort(key=lambda x: x['start'], reverse=True)
    temp_expression_list = list(expression_str)
    for rep in replacements:
        temp_expression_list[rep['start']:rep['end']] = list(rep['replacement_text'])
    return "".join(temp_expression_list)


def find_if_components(expression_if_body):
    # (此輔助函式與先前版本相同，用於分離 IF 的條件、真值、偽值部分)
    balance = 0
    comma_indices = []
    start_search = 0
    for i, char in enumerate(expression_if_body):
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        elif char == ',' and balance == 0:
            comma_indices.append(i)
            if len(comma_indices) == 2:
                break
    if len(comma_indices) != 2:
        raise ValueError(f"無法正確解析 IF 組件: {expression_if_body}。預期應有2個頂層逗號。")
    cond_str = expression_if_body[:comma_indices[0]].strip()
    true_str = expression_if_body[comma_indices[0]+1:comma_indices[1]].strip()
    false_str = expression_if_body[comma_indices[1]+1:].strip()
    return cond_str, true_str, false_str

# %%
 

def apply_excel_formulas_v4(df_input, formulas_dict, COL_NAME_MAP): # <--- 新增 COL_NAME_MAP
    df = df_input.copy()
    # 建立 eval 函式的局部變數環境
    # *** 修改初始化 eval_locals 的方式 ***
    eval_locals = {}
    for formula_key, actual_col_name in COL_NAME_MAP.items():
        if actual_col_name in df.columns:
            eval_locals[formula_key] = df[actual_col_name] # 例如 eval_locals['D'] = df['plate 1 COPY analog']
        else:
            # 允許部分映射，但如果公式中用到未映射的欄位，會在 parse_excel_expression_recursively 中報錯
            print(f"警告：col_name 映射中的實際欄位 '{actual_col_name}' (對應公式代號 '{formula_key}') 在 DataFrame 中找不到。如果公式未使用 '{formula_key}' 則無影響。")

    eval_globals = {"pd": pd, "np": np}

    print("\n開始自動化公式運算 (v4)...")
    for new_calc_col_name, formula_str in formulas_dict.items():
        print(f"  處理公式 for '{new_calc_col_name}': {formula_str}")
        
        expression = formula_str.strip()
        if expression.startswith('='):
            expression = expression[1:]

        try:
            # 創建一個針對當前公式的 eval_locals 副本，以避免固定值在不同公式間互相污染
            # Series 的部分可以共享，因為它們是唯讀的
            current_eval_locals = eval_locals.copy() # 複製包含 Series 的基礎環境

            if expression.upper().startswith("IF(") and expression.endswith(")"):
                print(f"    偵測到 IF 函式 for '{new_calc_col_name}'.")
                if_body = expression[len("IF("):-1]
                cond_expr_str, true_expr_str, false_expr_str = find_if_components(if_body)
                
                # 遞迴解析每個部分，COL_NAME_MAP 會被傳遞下去
                # current_eval_locals 將在 parse_excel_expression_recursively 中被更新 (加入 __fixed_... 值)
                parsed_cond = parse_excel_expression_recursively(cond_expr_str, df, current_eval_locals, COL_NAME_MAP)
                parsed_true = parse_excel_expression_recursively(true_expr_str, df, current_eval_locals, COL_NAME_MAP)
                parsed_false = parse_excel_expression_recursively(false_expr_str, df, current_eval_locals, COL_NAME_MAP)
                
                print(f"      轉換後條件: {parsed_cond}")
                print(f"      轉換後真值: {parsed_true}")
                print(f"      轉換後偽值: {parsed_false}")

                condition_series = eval(parsed_cond, eval_globals, current_eval_locals)
                true_series_or_scalar = eval(parsed_true, eval_globals, current_eval_locals)
                false_series_or_scalar = eval(parsed_false, eval_globals, current_eval_locals)
                
                df[new_calc_col_name] = np.where(condition_series, true_series_or_scalar, false_series_or_scalar)
                print(f"    '{new_calc_col_name}' (IF) 計算完成。")

            else: # 基本算術運算
                # current_eval_locals 已包含 Series，將在 parse_excel_expression_recursively 中加入固定值
                processed_expression = parse_excel_expression_recursively(expression, df, current_eval_locals, COL_NAME_MAP)
                print(f"    轉換後表達式 for '{new_calc_col_name}': {processed_expression}")
                df[new_calc_col_name] = eval(processed_expression, eval_globals, current_eval_locals)
                print(f"    '{new_calc_col_name}' 計算完成。")

        except Exception as e:
            import traceback
            print(f"    !!! 錯誤 (欄位: {new_calc_col_name}): 處理表達式 '{expression}' 時發生錯誤 -> {e}")
            # print(traceback.format_exc()) # 取消註解以獲得更詳細的追蹤訊息
            df[new_calc_col_name] = pd.NA

    print("自動化公式運算 (v4) 完成。")
    return df
# %%


def calculate_custom_correlations(df, PERSON_CORR, COL_NAME_MAP, START=None):
    """
    根據提供的設定計算 DataFrame 中特定欄位組合之間的相關係數。

    Args:
        df (pd.DataFrame): 包含數據的 DataFrame。
        PERSON_CORR (dict): 定義相關係數計算組合的字典。
                                   鍵為相關係數結果的名稱 (例如 "right 前後_cor")，
                                   值為一個包含代號的列表 (例如 ["O", "J"])。
        COL_NAME_MAP (dict): 將 PERSON_CORR 中的代號映射到 df 中實際欄位名稱的字典。
                                 鍵為代號 (例如 "O")，
                                 值為實際欄位名 (例如 "right 前後")。

    Returns:
        dict: 一個字典，儲存計算出的相關係數。
              如果一組是兩個欄位，則值為它們之間的相關係數 (純量)。
              如果一組多於兩個欄位，則鍵會加上 "_matrix" 後綴，值為相關係數矩陣 (DataFrame)。
              如果某組無法計算，則不會包含在回傳結果中。
    """
    if START:
        df = df.iloc[START[0]:START[1], :]
    print("\n執行自訂相關係數計算...")
    all_correlation_results = {}

    if not isinstance(df, pd.DataFrame):
        print("  錯誤: 輸入的 'df' 不是一個有效的 Pandas DataFrame。")
        return all_correlation_results
    if not isinstance(PERSON_CORR, dict):
        print("  錯誤: 'PERSON_CORR' 不是一個有效的字典。")
        return all_correlation_results
    if not isinstance(COL_NAME_MAP, dict):
        print("  錯誤: 'COL_NAME_MAP' 不是一個有效的字典。")
        return all_correlation_results

    for result_name, key_list in PERSON_CORR.items():
        actual_cols_for_this_corr = []
        valid_keys_for_current_set = True # 標記目前這組的代號是否都有效

        if not len(key_list):
            print(f"  警告 (相關係數 for '{result_name}'): 提供的代號列表不是一個列表，已跳過。")
            continue

        for key in key_list:
            print(key)
            # actual_col_name = COL_NAME_MAP.get(key)
            actual_col_name = key
            if actual_col_name:
                if actual_col_name in df.columns:
                    if pd.api.types.is_numeric_dtype(df[actual_col_name]):
                        actual_cols_for_this_corr.append(actual_col_name)
                    else:
                        print(f"  警告 (相關係數 for '{result_name}'): 實際欄位 '{actual_col_name}' (代號 '{key}') 非數值型，將被忽略。")
                        # 即使被忽略，我們可能仍想計算其他有效欄位的相關性，所以不在此處將 valid_keys_for_current_set 設為 False
                        # 但如果因此導致少於2個有效欄位，後面會處理
                else:
                    print(f"  錯誤 (相關係數 for '{result_name}'): 實際欄位 '{actual_col_name}' (代號 '{key}') 在 DataFrame 中找不到。")
                    valid_keys_for_current_set = False
                    break # 一個欄位找不到，這組就無法計算
            else:
                print(f"  錯誤 (相關係數 for '{result_name}'): 代號 '{key}' 在 COL_NAME_MAP 中找不到對應的實際欄位名。")
                valid_keys_for_current_set = False
                break # 一個代號無效，這組就無法計算
        
        if not valid_keys_for_current_set: # 如果有任何代號或欄位查找失敗，則跳過這組計算
            print(f"  由於上述錯誤，跳過 '{result_name}' 的相關係數計算。")
            continue

        if len(actual_cols_for_this_corr) >= 2:
            try:
                correlation_matrix = df[actual_cols_for_this_corr].corr()
                if len(actual_cols_for_this_corr) == 2:
                    correlation_value = correlation_matrix.iloc[0, 1]
                    print(f"  計算完成: '{result_name}' ({actual_cols_for_this_corr[0]} vs {actual_cols_for_this_corr[1]}): {correlation_value:.4f}")
                    all_correlation_results[result_name] = correlation_value
                else: # 超過兩個欄位，儲存整個相關係數矩陣
                    print(f"  計算完成: '{result_name}' (欄位: {', '.join(actual_cols_for_this_corr)}) 相關係數矩陣:\n{correlation_matrix}")
                    all_correlation_results[result_name + "_matrix"] = correlation_matrix
            except Exception as e:
                print(f"  計算相關係數時發生錯誤 for '{result_name}' (欄位: {actual_cols_for_this_corr}): {e}")
        elif valid_keys_for_current_set: # 確保是因為有效欄位不足，而不是因為前面已經出錯跳過了
             print(f"  警告 (相關係數 for '{result_name}'): 至少需要兩個有效的數值型欄位來計算相關係數，目前找到 {len(actual_cols_for_this_corr)} 個有效數值欄位 ({', '.join(actual_cols_for_this_corr)})。")

    print("自訂相關係數計算完成。")
    return all_correlation_results
# %%

def create_custom_subplots(df, PLOTS_CONFIG, COL_NAME_MAP, output_dir, filename):
    """
    根據提供的設定，為 DataFrame 中的資料建立包含多個子圖的圖表，
    每個子圖可以繪製兩條線。

    Args:
        df (pd.DataFrame): 包含數據的 DataFrame。
        PLOTS_CONFIG (dict): 繪圖設定字典。鍵為子圖的標識，
                             值為包含 "x_col_key", "y1_col_key", "y2_col_key" 及可選 "title",
                             "xlabel", "ylabel" 的字典。
        COL_NAME_MAP (dict): 將 PLOTS_CONFIG 中的 *_col_key 映射到 df 中實際欄位名稱的字典。
        output_dir (str): 儲存輸出圖片的資料夾路徑。
        filename (str): 原始檔案的基本名稱 (不含副檔名)，用於命名輸出的圖片。

    Returns:
        str or None: 成功儲存圖片則回傳圖片檔案路徑，否則回傳 None。
    """
    print("\n執行自訂子圖繪製...")

    if not isinstance(df, pd.DataFrame):
        print("  錯誤: 輸入的 'df' 不是一個有效的 Pandas DataFrame。")
        return None
    if not isinstance(PLOTS_CONFIG, dict) or not PLOTS_CONFIG:
        print("  錯誤: 'PLOTS_CONFIG' 不是一個有效的字典或為空。無需繪圖。")
        return None
    if not isinstance(COL_NAME_MAP, dict):
        print("  錯誤: 'COL_NAME_MAP' 不是一個有效的字典。")
        return None

    num_plots = len(PLOTS_CONFIG)
    
    # 設定子圖布局，盡量讓圖片美觀
    if num_plots == 0:
        print("  沒有定義任何子圖，無需繪製。")
        return None
    
    ncols = 1 # 每行顯示1個子圖，可以調整為2或更多以獲得不同佈局
    # ncols = min(2, num_plots) # 例如，每行最多2個子圖
    nrows = (num_plots + ncols - 1) // ncols # 計算需要的行數

    # 設定每個子圖的建議尺寸 (寬, 高)，單位為英吋
    subplot_width = 10
    subplot_height = 5
    fig_width = subplot_width * ncols
    fig_height = subplot_height * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False)
    # squeeze=False 確保 axes 總是一個二維陣列，方便迭代
    
    axes_flat = axes.flatten() # 將 axes 攤平成一維陣列，方便迭代

    plot_idx = 0
    successful_plots = 0

    for plot_group_name, config in PLOTS_CONFIG.items():
        if plot_idx >= len(axes_flat): # 以防萬一 PLOTS_CONFIG 比 axes 多
            print(f"  警告: 子圖數量超出預期佈局，'{plot_group_name}' 將不會被繪製。")
            break

        ax = axes_flat[plot_idx]

        x_key = config.get("x_col_key")
        y1_key = config.get("y1_col_key")
        y2_key = config.get("y2_col_key")

        if not all([x_key, y1_key, y2_key]):
            print(f"  警告 (子圖 '{plot_group_name}'): x_col_key, y1_col_key, 或 y2_col_key 未完整定義。跳過此子圖。")
            # 仍然可以隱藏這個未使用的 ax
            ax.axis('off')
            plot_idx +=1 # 確保即使跳過也增加索引，以便下一個圖使用正確的ax
            continue

        # actual_x_col = COL_NAME_MAP.get(x_key)
        # actual_y1_col = COL_NAME_MAP.get(y1_key)
        # actual_y2_col = COL_NAME_MAP.get(y2_key)
        
        actual_x_col = config.get("x_col_key")
        actual_y1_col = config.get("y1_col_key")
        actual_y2_col = config.get("y2_col_key")

        # 檢查欄位是否存在及是否為數值型
        valid_cols_for_plot = True
        cols_to_check = {
            x_key: actual_x_col,
            y1_key: actual_y1_col,
            y2_key: actual_y2_col
        }
        
        numeric_series = {} # 儲存有效的數值型 Series

        for key, actual_col in cols_to_check.items():
            if not actual_col:
                print(f"  警告 (子圖 '{plot_group_name}'): 代號 '{key}' 在 COL_NAME_MAP 中找不到對應的實際欄位名。跳過此子圖。")
                valid_cols_for_plot = False
                break
            if actual_col not in df.columns:
                print(f"  警告 (子圖 '{plot_group_name}'): 實際欄位 '{actual_col}' (代號 '{key}') 在 DataFrame 中找不到。跳過此子圖。")
                valid_cols_for_plot = False
                break
            if not pd.api.types.is_numeric_dtype(df[actual_col]):
                print(f"  警告 (子圖 '{plot_group_name}'): 實際欄位 '{actual_col}' (代號 '{key}') 非數值型。跳過此子圖。")
                valid_cols_for_plot = False
                break
            numeric_series[key] = df[actual_col] # 儲存 Series 供後續使用

        if not valid_cols_for_plot:
            ax.axis('off') # 隱藏無效的子圖座標軸
            plot_idx += 1
            continue

        # 繪製兩條線
        try:
            # 使用代號 (key) 或實際欄位名 (actual_col) 作為圖例標籤
            line1_label = f"{y1_key} ({actual_y1_col})" if y1_key != actual_y1_col else actual_y1_col
            line2_label = f"{y2_key} ({actual_y2_col})" if y2_key != actual_y2_col else actual_y2_col

            ax.plot(numeric_series[x_key], numeric_series[y1_key], label=line1_label)
            ax.plot(numeric_series[x_key], numeric_series[y2_key], label=line2_label)

            # 設定標題和軸標籤
            plot_title = config.get("title", plot_group_name)
            ax.set_title(plot_title)
            
            xlabel_text = config.get("xlabel", actual_x_col) # 如果未提供，使用實際X欄位名
            ax.set_xlabel(xlabel_text)

            ylabel_text = config.get("ylabel", "數值") # 如果未提供，使用通用標籤 "數值"
            ax.set_ylabel(ylabel_text)
            
            ax.legend() # 顯示圖例
            ax.grid(True) # 加入網格線
            successful_plots += 1
        except Exception as e:
            print(f"  錯誤 (子圖 '{plot_group_name}'): 繪圖時發生錯誤: {e}")
            ax.axis('off') # 隱藏出錯的子圖座標軸
        
        plot_idx += 1

    # 隱藏剩餘未使用的子圖 (如果 num_plots 不是 nrows * ncols 的整數倍)
    for i in range(plot_idx, nrows * ncols):
        axes_flat[i].axis('off')

    if successful_plots == 0:
        print("  沒有任何子圖成功繪製，不儲存圖片。")
        plt.close(fig) # 關閉圖形以釋放記憶體
        return None

    # 自動調整子圖佈局以避免重疊
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.96]) # rect=[0, 0, 1, 0.96] 為了給主標題留空間
        fig.suptitle(f"{filename} - 圖表分析", fontsize=16)
    except Exception as e:
        print(f"  警告: tight_layout 失敗: {e}")


    # 儲存整個圖表
    plot_output_filename = f"{filename}_custom_plots.png"
    plot_output_path = os.path.join(output_dir, plot_output_filename)
    try:
        plt.savefig(plot_output_path)
        plt.show()
        print(f"  多子圖圖表已儲存至: {plot_output_path}")
        plt.close(fig) # 關閉圖形以釋放記憶體
        return plot_output_path
    except Exception as e:
        print(f"  儲存圖表時發生錯誤: {e}")
        plt.close(fig) # 關閉圖形以釋放記憶體
        return None
# %%

def Read_File(file_path, file_type, subfolder=None):
    """
    Parameters
    ----------
    file_path : str
        給予欲讀取資料之路徑.
    file_type : str
        給定欲讀取資料之副檔名.
    subfolder : boolean, optional
        是否子資料夾一起讀取. The default is 'None'.

    Returns
    -------
    csv_file_list : list
        回給所有路徑下的資料絕對路徑.

    """
    # if subfolder = True, the function will run with subfolder

    csv_file_list = []

    if subfolder:
        file_list_1 = []
        for dirPath, dirNames, fileNames in os.walk(file_path):
            # file_list = os.walk(folder_name)
            file_list_1.append(dirPath)
        # need to change here [1:]
        for ii in file_list_1[1:]:
            file_list = os.listdir(ii)
            for iii in file_list:
                if os.path.splitext(iii)[1] == file_type:
                    # replace "\\" to '/', due to MAC version
                    file_list_name = ii + "\\" + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(file_path)
        for i in folder_list:
            if os.path.splitext(i)[1] == file_type:
                # replace "\\" to '/', due to MAC version
                file_list_name = file_path + "\\" + i
                csv_file_list.append(file_list_name)

    return csv_file_list


# %%


def apply_butterworth_filter(df, FILTER_COL, COL_NAME_MAP,
                             cutoff_freq, sampling_freq, order=4, filter_type='low'):
    """
    對 DataFrame 中的指定欄位應用 Butterworth 濾波器。

    Args:
        df (pd.DataFrame): 輸入的 DataFrame。
        FILTER_COL (set or list): 包含要濾波的欄位「代號」的集合或列表。
                                             這些代號將透過 COL_NAME_MAP 轉換。
        COL_NAME_MAP (dict): 將代號映射到 DataFrame 中實際欄位名稱的字典。
        cutoff_freq (float): 濾波器的截止頻率 (Hz)。
        sampling_freq (float): 訊號的取樣頻率 (Hz)。
        order (int, optional): 濾波器的階數。預設為 4。
        filter_type (str, optional): 濾波器類型，可以是 'low' (低通), 'high' (高通),
                                     'bandpass' (帶通), 或 'bandstop' (帶阻)。
                                     如果是 'bandpass' 或 'bandstop'，cutoff_freq 應為
                                     一個包含兩個元素的列表或元組 [lowcut, highcut]。
                                     預設為 'low' (低通)。

    Returns:
        pd.DataFrame: 包含濾波後欄位的 DataFrame 副本。
                      濾波後的欄位會覆蓋原始欄位。
    """
    print(f"\n執行 Butterworth {filter_type}pass 濾波...")
    df_filtered = df.copy()
    # df_filtered = df_1.copy()
    nyquist_freq = 0.5 * sampling_freq

    if not isinstance(FILTER_COL, (set, list)):
        print("  錯誤: 'FILTER_COL' 必須是集合或列表。")
        return df # 回傳原始 df
    if not isinstance(COL_NAME_MAP, dict):
        print("  錯誤: 'COL_NAME_MAP' 必須是字典。")
        return df

    for key in FILTER_COL:
        actual_col_name = key

        if not actual_col_name:
            print(f"  警告: 代號 '{key}' 在 COL_NAME_MAP 中找不到對應的實際欄位名，跳過濾波。")
            continue

        if actual_col_name not in df_filtered.columns:
            print(f"  警告: 實際欄位 '{actual_col_name}' (代號 '{key}') 在 DataFrame 中找不到，跳過濾波。")
            continue

        if not pd.api.types.is_numeric_dtype(df_filtered[actual_col_name]):
            print(f"  警告: 實際欄位 '{actual_col_name}' (代號 '{key}') 非數值型，無法濾波，已跳過。")
            continue
        
        # 處理 NaN 值，例如使用前一個有效值填充或插值
        # 這裡使用 forward fill 然後 backfill，您也可以選擇其他策略如 .interpolate()
        # filtfilt 對 NaN 敏感
        original_series = df_filtered[actual_col_name].copy()
        if original_series.isnull().any():
            print(f"  資訊: 欄位 '{actual_col_name}' 包含 NaN 值，將嘗試使用 ffill 和 bfill 填充後進行濾波。")
            df_filtered[actual_col_name] = df_filtered[actual_col_name].ffill().bfill()
            if df_filtered[actual_col_name].isnull().any(): # 如果填充後仍有 NaN (例如整欄都是 NaN)
                print(f"  警告: 欄位 '{actual_col_name}' 填充後仍包含 NaN，無法濾波，已跳過。")
                df_filtered[actual_col_name] = original_series # 還原
                continue


        # 設計濾波器
        # 對於低通和高通，wn 是單個值；對於帶通和帶阻，wn 是 [lowcut, highcut]
        if filter_type in ['low', 'high']:
            if not isinstance(cutoff_freq, (int, float)):
                print(f"  錯誤: 對於 '{filter_type}' 濾波器，cutoff_freq 必須是單個數值。跳過欄位 '{actual_col_name}'。")
                df_filtered[actual_col_name] = original_series # 還原
                continue
            normalized_cutoff = cutoff_freq / nyquist_freq
            if not (0 < normalized_cutoff < 1):
                print(f"  錯誤: 正規化截止頻率 ({normalized_cutoff:.4f}) 必須介於 0 和 1 之間。請檢查 cutoff_freq ({cutoff_freq} Hz) 和 sampling_freq ({sampling_freq} Hz)。跳過欄位 '{actual_col_name}'。")
                df_filtered[actual_col_name] = original_series # 還原
                continue
            b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)
        elif filter_type in ['bandpass', 'bandstop']:
            if not (isinstance(cutoff_freq, (list, tuple)) and len(cutoff_freq) == 2):
                print(f"  錯誤: 對於 '{filter_type}' 濾波器，cutoff_freq 必須是包含兩個數值的列表/元組 [low, high]。跳過欄位 '{actual_col_name}'。")
                df_filtered[actual_col_name] = original_series # 還原
                continue
            lowcut, highcut = cutoff_freq
            normalized_low = lowcut / nyquist_freq
            normalized_high = highcut / nyquist_freq
            if not (0 < normalized_low < 1 and 0 < normalized_high < 1 and normalized_low < normalized_high):
                print(f"  錯誤: 正規化截止頻率 ([{normalized_low:.4f}, {normalized_high:.4f}]) 必須介於 0 和 1 之間，且 low < high。請檢查 cutoff_freq ({cutoff_freq} Hz) 和 sampling_freq ({sampling_freq} Hz)。跳過欄位 '{actual_col_name}'。")
                df_filtered[actual_col_name] = original_series # 還原
                continue
            b, a = butter(order, [normalized_low, normalized_high], btype=filter_type, analog=False)
        else:
            print(f"  錯誤: 不支援的濾波器類型 '{filter_type}'。跳過欄位 '{actual_col_name}'。")
            df_filtered[actual_col_name] = original_series # 還原
            continue
        
        # (可選) 繪製頻率響應曲線
        # w, h = freqz(b, a, worN=8000)
        # plt.figure()
        # plt.plot(0.5*sampling_freq*w/np.pi, np.abs(h), 'b')
        # plt.title(f"Butterworth Filter Frequency Response for {actual_col_name}")
        # plt.xlabel('Frequency [Hz]')
        # plt.ylabel('Gain')
        # plt.grid()
        # plt.show()

        # 應用濾波器 (filtfilt 進行零相位濾波)
        try:
            filtered_signal = filtfilt(b, a, df_filtered[actual_col_name])
            df_filtered[actual_col_name] = filtered_signal
            print(f"  欄位 '{actual_col_name}' (代號 '{key}') 已成功濾波。")
        except ValueError as ve: # filtfilt 對輸入資料長度有要求
            print(f"  錯誤: 對欄位 '{actual_col_name}' (代號 '{key}') 應用 filtfilt 時發生錯誤: {ve}。資料長度可能太短。跳過濾波。")
            df_filtered[actual_col_name] = original_series # 還原
            
    print("Butterworth 濾波完成。")
    return df_filtered

# %%


def process_file(data_path, CAL_FORMULAS, COL_NAME_MAP,
                 PERSON_CORR, PLOTS_CONFIG, FILTER_COL,
                 OUTPUT_DIRECTORY,
                 NeedFilter=True):
    """
    處理單個 Excel 檔案：欄位運算、計算相關係數、輸出結果、繪圖。
    """
    data_path = r'D:\\Hsin\\NTSU_lab\\WALK\\範例檔-20250519T142424Z-1-001\\範例檔\\S1皮爾森相關分析範例\\\\\\S1_ST_WALK_1 FP&SI extra v2.xlsx'
    # print(f"--- 正在處理檔案: {original_filename} ---")
    try:
        df = pd.read_excel(data_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {data_path}")
        return
    except Exception as e:
        print(f"讀取 Excel 檔案 {data_path} 時發生錯誤: {e}")
        return
    
    if df is not None and CAL_FORMULAS: # 確保 df 已載入且有公式要處理
        try:
            df = df.iloc[:, :10]
            df_1 = apply_excel_formulas_v4(df, CAL_FORMULAS, COL_NAME_MAP)
        except Exception as e:
            print(f"  處理自動化公式時發生嚴重錯誤: {e}")
    # 取得含副檔名的檔名
    file_name_with_ext = os.path.basename(data_path)  # → "S1_GOLF_1 FP&SI extra v2.xlsx"
    
    # 取得不含副檔名的檔名
    file_name = os.path.splitext(file_name_with_ext)[0]  # → "S1_GOLF_1 FP&SI extra v2"
    # 2. 將資料濾波
    if NeedFilter:
        df_1 = apply_butterworth_filter(df_1,
                                        FILTER_COL,
                                        COL_NAME_MAP,
                                        cutoff_freq = 25,
                                        sampling_freq = 100,
                                        order=4,
                                        filter_type='low')
        processed_output_filename = f"{os.path.splitext(file_name)[0]}_Filtered.xlsx"
    else:
        processed_output_filename = f"{os.path.splitext(file_name)[0]}_NoFiltered.xlsx"
    # 3. 計算其中數個欄位間的相關係數
    all_correlation_results = calculate_custom_correlations(df_1, PERSON_CORR, COL_NAME_MAP)

    # 4. 輸出欄位運算後的資料
    print("\n輸出運算後的資料...")
    
    processed_output_path = os.path.join(OUTPUT_DIRECTORY, processed_output_filename)
    try:
        df_1.to_excel(processed_output_path, index=False)
        print(f"  運算後的資料已儲存至: {processed_output_path}")
    except Exception as e:
        print(f"  儲存運算後的資料時發生錯誤: {e}")
    
    # 資料繪圖 
    create_custom_subplots(df_1, PLOTS_CONFIG, COL_NAME_MAP, OUTPUT_DIRECTORY, filename=processed_output_filename)
    
    return all_correlation_results