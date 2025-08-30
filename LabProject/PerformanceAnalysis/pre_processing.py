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
from scipy.signal import argrelextrema, savgol_filter, butter, filtfilt
from numpy.linalg import norm
from scipy.interpolate import interp1d
import warnings
import os
from typing import Dict, Any, Optional, Tuple, List
import ezc3d
import emg_function as emg

# import Spider_function as func
plt.rcParams['font.sans-serif'] = ['Roboto']  # 改為你實際有的
plt.rcParams['axes.unicode_minus'] = False    # 避免座標軸負號亂碼

# %% Reading all of data path
# using a recursive loop to traverse each folder
# and find the file extension has .csv
def Read_File(file_path, file_type, subfolder=None):
    '''
    Parameters
    ----------
    x : str
        給予欲讀取資料之路徑.
    y : str
        給定欲讀取資料之副檔名.
    subfolder : boolean, optional
        是否子資料夾一起讀取. The default is 'None'.

    Returns
    -------
    csv_file_list : list
        回給所有路徑下的資料絕對路徑.

    '''
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
                    file_list_name = ii + '\\' + iii
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
# --- Main Function ---
def read_c3d(path: str,
             process_forceplate: bool = True, # Renamed for clarity
             process_analog: bool = True,     # Renamed for clarity
             prefix_to_remove: Optional[List[str]] = None,
             rename_map: Optional[Dict[str, str]] = None,
             marker_cutoff: Optional[float] = 10.0, # Default cutoff 10Hz for markers
             analog_cutoff: Optional[float] = None, # Default no filter for general analog
             fp_cutoff: Optional[float] = 20.0,     # Default cutoff 20Hz for force plates
             filter_order: int = 4
             ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Reads a C3D file, processes marker, force plate, and analog data,
    including interpolation and optional low-pass filtering.

    Args:
        path (str): Path to the C3D file.
        process_forceplate (bool): Whether to process force plate data.
        process_analog (bool): Whether to process general analog data (excluding FP channels if processed separately).
        prefix_to_remove (Optional[List[str]]): List of prefixes to remove from marker labels.
        rename_map (Optional[Dict[str, str]]): Dictionary for renaming marker labels {old: new}.
        marker_cutoff (Optional[float]): Cutoff frequency (Hz) for marker data filtering. Set to None or 0 to disable.
        analog_cutoff (Optional[float]): Cutoff frequency (Hz) for general analog data filtering. Set to None or 0 to disable.
        fp_cutoff (Optional[float]): Cutoff frequency (Hz) for force plate data (Force, Moment, COP) filtering. Set to None or 0 to disable.
        filter_order (int): Order for the Butterworth filter.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            - combine_dict: Dictionary containing processed data ("markers", "FP", "analog").
            - descriptions: Dictionary containing metadata ("motion_info", "analog_info", "fp_info").
    """
    # --- Helper Function for Filtering ---
    def _lowpass_filter(data: np.ndarray, fs: float, cutoff: Optional[float], order: int = 4) -> np.ndarray:
        """
        Applies a zero-phase low-pass Butterworth filter to the data.

        Args:
            data (np.ndarray): Data to filter (1D or 2D, time along axis 0).
            fs (float): Sampling frequency.
            cutoff (Optional[float]): Cutoff frequency. If None or <= 0, no filtering is applied.
            order (int): Filter order.

        Returns:
            np.ndarray: Filtered data or original data if filtering is skipped.
        """
        if cutoff is None or cutoff <= 0:
            # print("Debug: Filtering skipped (cutoff is None or <= 0)")
            return data # No filtering needed
        if fs <= 0:
            print(f"Warning: Invalid sampling frequency ({fs}Hz). Skipping filter.")
            return data

        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        if normal_cutoff >= 1: # Cutoff frequency is too high
             print(f"Warning: Cutoff frequency ({cutoff}Hz) is >= Nyquist frequency ({nyq}Hz). Skipping filter.")
             return data
        if normal_cutoff <= 0: # Cutoff frequency is too low
            print(f"Warning: Cutoff frequency ({cutoff}Hz) results in non-positive normalized cutoff. Skipping filter.")
            return data

        try:
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
        except ValueError as e:
            print(f"Warning: Could not create Butterworth filter (fs={fs}, cutoff={cutoff}, order={order}). Error: {e}. Skipping filter.")
            return data

        # Apply filter column by column for 2D data (like markers [N_frames, 3] or FP components [N_frames, 3])
        # Ensure data is float for filtering
        data_float = data.astype(float)
        filtered_data = np.zeros_like(data_float)

        if data_float.ndim == 1:
             # Avoid filtering if data length is less than padlen (default is 3 * max(len(a), len(b)))
             padlen = 3 * max(len(b), len(a))
             if len(data_float) <= padlen:
                 print(f"Warning: Data length ({len(data_float)}) is too short for filter padlen ({padlen}). Skipping filter.")
                 return data
             filtered_data = filtfilt(b, a, data_float)
        elif data_float.ndim == 2:
             padlen = 3 * max(len(b), len(a))
             if data_float.shape[0] <= padlen:
                 print(f"Warning: Data length ({data_float.shape[0]}) is too short for filter padlen ({padlen}). Skipping filter.")
                 return data
             for i in range(data_float.shape[1]):
                 filtered_data[:, i] = filtfilt(b, a, data_float[:, i])
        else:
             print("Warning: Filtering currently only supported for 1D or 2D data. Skipping filter.")
             return data # Return original data if not 1D/2D

        # print(f"Debug: Filtering applied with fs={fs}, cutoff={cutoff}")
        return filtered_data

    # --- Helper Function for Interpolation ---
    def _interpolate_data(data: np.ndarray) -> np.ndarray:
        """
        Interpolates missing data (represented by 0 or NaN) using linear interpolation
        followed by forward and backward fill.

        Warning: Replaces ALL zeros with NaN before interpolation. This might be
                 undesirable if zero is a valid data point.

        Args:
            data (np.ndarray): Input data array (time along axis 0).

        Returns:
            np.ndarray: Interpolated data array.
        """
        if data is None or data.size == 0:
            return np.array([]) # Return empty if input is empty

        df = pd.DataFrame(data)
        # Warning: Replacing all zeros with NaN might affect valid zero data points.
        df.replace(0, np.nan, inplace=True)

        # Check if all values became NaN after replacing zeros
        if df.isnull().all().all():
             print("Warning: All data points became NaN after replacing zeros. Cannot interpolate.")
             # Return original data (or perhaps zeros/NaNs based on desired behavior)
             return data # Or df.fillna(0).values or data (which might be all zeros)

        # Use linear interpolation first
        df = df.interpolate(method='cubic', axis=0, limit_direction='both') # limit_direction helps with start/end NaNs
        valid_points_count = df.notna().sum()

        # 檢查是否有任何一個欄位的數據點少於 4 個
        if valid_points_count.min() < 4:
            # 數據不足，降級使用 'linear' 方法
            # print(f"Warning: Insufficient data for cubic interpolation. Falling back to linear.") # 可選：印出警告訊息
            df_interpolated = df.interpolate(method='linear', axis=0, limit_direction='both')
        else:
           # 數據充足，使用 'cubic' 方法
           df_interpolated = df.interpolate(method='cubic', axis=0, limit_direction='both')


        # Use ffill and bfill to handle any remaining NaNs (e.g., at the very start/end if limit_direction='both' wasn't enough)
        df_interpolated.ffill(inplace=True)
        df_interpolated.bfill(inplace=True)

        # Final check if any NaNs persist (shouldn't happen with ffill/bfill, but as a safeguard)
        if df_interpolated.isnull().values.any():
            print("Warning: NaNs remain after interpolation and fill. Filling with 0.")
            df_interpolated.fillna(0, inplace=True) # Fill any persistent NaNs with 0 as a last resort

        return df_interpolated.values

    # --- Helper Function for Marker Processing ---
    def _process_markers(c3d_data: ezc3d.c3d, marker_cutoff: Optional[float], filter_order: int,
                         prefix_to_remove: Optional[List[str]], rename_map: Optional[Dict[str, str]]) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
        """Processes marker data: extraction, renaming, interpolation, filtering."""
        points_data = c3d_data['data']['points']
        points_header = c3d_data['header']['points']
        points_params = c3d_data['parameters']['POINT']

        marker_labels = points_params.get('LABELS', {}).get('value', [])
        marker_units = points_params.get('UNITS', {}).get('value', [""])[0] # Usually mm
        fs = points_header.get('frame_rate', 0.0)
        num_frames = points_data.shape[2]
        num_markers = points_data.shape[1]

        if len(marker_labels) != num_markers:
            print(f"Warning: Number of marker labels ({len(marker_labels)}) does not match number of markers in data ({num_markers}). Using generic names.")
            marker_labels = [f"Marker_{i+1}" for i in range(num_markers)]

        # --- Label Handling (Prefix Removal & Renaming) ---
        processed_labels = list(marker_labels) # Copy the list
        if prefix_to_remove:
            for prefix in prefix_to_remove:
                processed_labels = [label.replace(prefix, "") for label in processed_labels]
        if rename_map:
            temp_labels = list(processed_labels) # Work on a copy
            for original, new in rename_map.items():
                temp_labels = [label.replace(original, new) for label in temp_labels]
            processed_labels = temp_labels

        # --- Data Extraction & Initial Dictionary Creation ---
        marker_data_raw = {}
        for i, marker_name in enumerate(processed_labels):
            # Extract X, Y, Z coordinates. Data shape is (4, n_markers, n_frames)
            # The 4th row is usually camera contribution/residual, we only need first 3
            marker_data_raw[marker_name] = points_data[:3, i, :].T # Transpose to get (n_frames, 3)

        # --- Interpolation ---
        print("Interpolating marker data...")
        marker_data_interp = {key: _interpolate_data(value) for key, value in marker_data_raw.items()}
       

        # --- Filtering ---
        print("Filtering marker data...")
        marker_data_filt = {}
        if fs > 0 and marker_cutoff is not None and marker_cutoff > 0:
            for key, value in marker_data_interp.items():
                if value.ndim == 2 and value.shape[1] == 3: # Ensure it's (N, 3)
                     marker_data_filt[key] = _lowpass_filter(value, fs, marker_cutoff, order=filter_order)
                else:
                     print(f"Warning: Marker data '{key}' has unexpected shape {value.shape}. Skipping filter.")
                     marker_data_filt[key] = value # Keep original if shape is wrong
        else:
            print("Skipping marker filtering (fs invalid or cutoff not specified).")
            marker_data_filt = marker_data_interp # Use interpolated if not filtering

        # --- Time Vector ---
        last_frame_idx = points_header.get('last_frame', num_frames - 1) # Use actual last frame index if available
        duration = (last_frame_idx - points_header.get('first_frame', 0)) / fs if fs > 0 else 0
        # Ensure num matches the actual number of frames extracted
        time_vector = np.linspace(0, duration, num=num_frames)
        marker_data_filt["time"] = time_vector

        # --- Motion Info Dictionary ---
        motion_info = {
            "frame_rate": fs,
            "first_frame": points_header.get('first_frame', 0),
            "last_frame": last_frame_idx,
            "num_frames": num_frames,
            "num_markers": num_markers,
            "UNITS": marker_units,
            "LABELS": processed_labels # Store the final processed labels
        }

        return marker_data_filt, motion_info, time_vector # Return time_vector separately for potential use

    # --- Helper Function for Force Plate Processing ---
    def _process_force_plates(c3d_data: ezc3d.c3d, analog_fs: float, fp_cutoff: Optional[float], filter_order: int) -> Optional[Dict[str, Any]]:
        """Processes force plate data: extraction, unit conversion, filtering."""
        if 'FORCE_PLATFORM' not in c3d_data['parameters'] or 'platform' not in c3d_data['data']:
            print("No force plate parameter or data found.")
            return None

        fp_params = c3d_data['parameters']['FORCE_PLATFORM']
        fp_data = c3d_data['data']['platform']
        num_fp_used = fp_params.get('USED', {}).get('value', [0])[0]

        if num_fp_used <= 0:
            print("No force plates marked as 'used'.")
            return None

        print(f"Processing {num_fp_used} force plate(s)...")
        fp_data_processed = {}
        fp_info = {"num_plates": num_fp_used, "type": fp_params.get('TYPE', {}).get('value', [])}

        for i in range(num_fp_used):
            platform_idx = i # Assuming data corresponds directly to 'used' index
            if platform_idx >= len(fp_data):
                 print(f"Warning: Mismatch between 'used' count ({num_fp_used}) and available platform data ({len(fp_data)}). Skipping FP {i+1}.")
                 continue

            pf_label = f'FP{i+1}'
            platform = fp_data[platform_idx]

            # Extract raw data (transpose to get [N_frames, 3])
            force_raw = platform.get('force', np.array([])).T
            moment_raw = platform.get('moment', np.array([])).T
            cop_raw = platform.get('center_of_pressure', np.array([])).T

            # --- Unit Conversion (as per original comments, verify correctness for your system) ---
            # Force: N (assuming input is N)
            force_converted = force_raw
            # Moment: Nmm -> Nm (divide by 1000)
            moment_converted = moment_raw / 1000.0
            # COP: mm -> mm (No conversion needed if target unit is mm)
            # Original code divided by 10 (mm -> cm?), keeping it but it seems unusual.
            # If target is meters, divide by 1000. If target is mm, keep as is.
            cop_converted = cop_raw # / 10.0 # Uncomment and adjust if unit conversion is desired

            # --- Filtering ---
            # Warning: Filtering COP directly can be problematic. It's often better to
            # filter forces/moments and recalculate COP if high accuracy is needed.
            force_filt = _lowpass_filter(force_converted, analog_fs, fp_cutoff, filter_order)
            moment_filt = _lowpass_filter(moment_converted, analog_fs, fp_cutoff, filter_order)
            cop_filt = _lowpass_filter(cop_converted, analog_fs, fp_cutoff, filter_order) # Filter calculated COP

            fp_data_processed[pf_label] = {
                # Store corners if needed, transpose for easier interpretation [4, 3]
                "corners": fp_params.get('CORNERS', {}).get('value', np.array([]))[:, :, i].T if fp_params.get('CORNERS', {}).get('value', np.array([])).size > 0 else np.array([]),
                "force": force_filt,
                "moment": moment_filt,
                "cop": cop_filt
            }
            # Add origin info if available
            if 'ORIGIN' in fp_params and fp_params['ORIGIN']['value'].shape[1] > i:
                 fp_data_processed[pf_label]["origin"] = fp_params['ORIGIN']['value'][:, i]


        fp_info.update({
                "caution": "Units based on typical C3D export; verify for your system.",
                "Force_unit": "N",
                "Moment_unit": "Nm", # After conversion from Nmm
                "COP_unit": "mm" # Or 'cm' if divided by 10, or 'm' if divided by 1000
            })

        return {"data": fp_data_processed, "info": fp_info}

    # --- Helper Function for Analog Processing ---
    def _process_analog_data(c3d_data: ezc3d.c3d, analog_fs: float, analog_cutoff: Optional[float], filter_order: int) -> Optional[Dict[str, Any]]:
        """Processes general analog data: extraction, filtering."""
        if 'ANALOG' not in c3d_data['parameters'] or 'analogs' not in c3d_data['data']:
            print("No analog parameter or data found.")
            return None

        analog_params = c3d_data['parameters']['ANALOG']
        analog_data = c3d_data['data']['analogs'] # Shape (1, n_channels, n_analog_frames)
        num_analog_channels = analog_data.shape[1]
        analog_labels = analog_params.get('LABELS', {}).get('value', [])
        analog_units = analog_params.get('UNITS', {}).get('value', [])
        analog_scales = analog_params.get('SCALE', {}).get('value', np.ones(num_analog_channels))
        analog_offsets = analog_params.get('OFFSET', {}).get('value', np.zeros(num_analog_channels))

        if len(analog_labels) != num_analog_channels:
            print(f"Warning: Number of analog labels ({len(analog_labels)}) does not match number of channels ({num_analog_channels}). Using generic names.")
            analog_labels = [f"Analog_{i+1}" for i in range(num_analog_channels)]
        if len(analog_units) != num_analog_channels:
            analog_units = ["Unknown"] * num_analog_channels
        if len(analog_scales) != num_analog_channels:
            analog_scales = np.ones(num_analog_channels)
        if len(analog_offsets) != num_analog_channels:
             analog_offsets = np.zeros(num_analog_channels)


        print(f"Processing {num_analog_channels} analog channel(s)...")
        analog_data_processed = {}
        analog_info = {"labels": [], "units": []}

        for i, label in enumerate(analog_labels):
            # Extract data for the channel, apply scale factor and offset
            # Data shape is (1, n_channels, n_frames), so access [0, i, :]
            channel_data_raw = analog_data[0, i, :]
            # Apply scale and offset: final = (raw + offset) * scale
            # Note: ezc3d might apply this automatically depending on version/settings, verify if needed.
            # Assuming ezc3d provides raw data:
            channel_data_scaled = (channel_data_raw + analog_offsets[i]) * analog_scales[i]


            # --- Filtering ---
            channel_data_filt = _lowpass_filter(channel_data_scaled, analog_fs, analog_cutoff, filter_order)

            analog_data_processed[label] = channel_data_filt
            analog_info["labels"].append(label)
            analog_info["units"].append(analog_units[i])

        return {"data": analog_data_processed, "info": analog_info}
    # ------ main function Logic Start ------------
    print(f"Reading C3D file: {path}")
    try:
        # extract_forceplat_data=True helps ezc3d parse FP specific parameters
        # path = r"D:\BenQ_Project\01_UR_lab\00_BQE\2025_06 Lab Opening\motion\S1_Post_Spider30_EC.c3d"
        c = ezc3d.c3d(path, extract_forceplat_data=True)
    except FileNotFoundError:
        print(f"Error: C3D file not found at {path}")
        return {}, {}
    except Exception as e:
        print(f"Error reading C3D file {path}: {e}")
        return {}, {}

    # === 1. Basic Information ===
    descriptions = {
        # "c3d_header": c.get("header", {}), # Store the whole header for reference
        # "c3d_parameters": c.get("parameters", {}) # Store parameters for reference
    }
    marker_fs = c.get("header", {}).get("points", {}).get("frame_rate", 0.0)
    analog_fs = c.get("header", {}).get("analogs", {}).get("frame_rate", 0.0)
    # Check if frequencies are valid
    if marker_fs <= 0:
        print("Warning: Invalid marker frame rate in C3D header.")
    if analog_fs <= 0:
        print("Warning: Invalid analog frame rate in C3D header.")


    # === 2. Process Motion Data ===
    print("\n--- Processing Motion Data ---")
    markers_processed, motion_info, time_vector = _process_markers(
        c, marker_cutoff, filter_order, prefix_to_remove, rename_map
    )
    descriptions["motion_info"] = motion_info

    # === 3. Process Force Plate Data ===
    fp_processed_data = None
    if process_forceplate:
        print("\n--- Processing Force Plate Data ---")
        if analog_fs <= 0:
             print("Skipping Force Plate processing due to invalid analog frame rate.")
        else:
            fp_result = _process_force_plates(c, analog_fs, fp_cutoff, filter_order)
            if fp_result:
                fp_processed_data = fp_result["data"]
                descriptions["fp_info"] = fp_result["info"]
    else:
        print("\nSkipping Force Plate processing as requested.")


    # === 4. Process Analog Data ===
    analog_processed_data = None
    analog_channel_info = None
    if process_analog:
        print("\n--- Processing Analog Data ---")
        if analog_fs <= 0:
             print("Skipping Analog processing due to invalid analog frame rate.")
        else:
            analog_result = _process_analog_data(c, analog_fs, analog_cutoff, filter_order)
            if analog_result:
                analog_processed_data = analog_result["data"]
                analog_channel_info = analog_result["info"] # Store labels/units
                # Add general analog info from header
                descriptions["analog_info"] = {
                    "frame_rate": analog_fs,
                    "num_channels": c.get("header", {}).get("analogs", {}).get("nb_channels", 0),
                    "samples_per_frame": c.get("header", {}).get("analogs", {}).get("ratio", 0),
                    "channel_details": analog_channel_info # Add specific labels/units
                }

    else:
        print("\nSkipping Analog processing as requested.")
        # Still add basic analog info from header if available
        if "analogs" in c.get("header", {}):
             descriptions["analog_info"] = {
                 "frame_rate": analog_fs,
                 "num_channels": c.get("header", {}).get("analogs", {}).get("nb_channels", 0),
                 "samples_per_frame": c.get("header", {}).get("analogs", {}).get("ratio", 0),
             }


    # === 5. Combine Results ===
    combine_dict = {"markers": markers_processed} # Markers are always processed
    if fp_processed_data is not None:
        combine_dict["FP"] = fp_processed_data
    if analog_processed_data is not None:
        # Optional: Exclude FP channels from general analog if they were processed separately
        # This requires knowing the mapping from FP labels (FP1_Fx etc.) to analog channel labels
        # For simplicity now, we include all processed analog channels.
        combine_dict["analog"] = analog_processed_data

    print("\n--- C3D Processing Complete ---")
    return combine_dict, descriptions


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

def find_Zaxis_min_with_baseline( # Function name kept for consistency with last step
        df,
        # --- Baseline Removal Params ---
        use_baseline_removal=True,
        baseline_window_length=51,
        baseline_polyorder=3,
        # --- Original Params (Threshold logic modified) ---
        order=5,
        min_frame_gap=8,
        min_z_diff=0.2,
        # --- New Threshold Param ---
        z_processed_threshold=None,
        # --- Output Params ---
        show=True,
        showVel=True):
    """
    Finds local minima in Z-axis data, optionally using Savitzky-Golay baseline removal,
    and filters them based on time interval and Z value changes.
    
    Parameters：
        df: 
            DataFrame, containing 'Z' column and optionally 'cum_yaw_deg', 'cum_pitch_deg', 'speed' for plotting.
        use_baseline_removal: 
            bool, whether to enable baseline removal.
        baseline_window_length: 
            int, window size for Savitzky-Golay filter.
        baseline_polyorder: 
            int, polynomial order for Savitzky-Golay filter.
        order: 
            int, window size for local minima search (default 5).
        min_frame_gap: 
            int, minimum frame gap between minima (default 8).
        min_z_diff: 
            float, minimum Z difference required if frame gap is insufficient (default 0.2).
        z_processed_threshold: 
            float or None, threshold for filtering processed Z values. Only points below are kept.
        show: 
            bool, whether to plot visualization results.
        showVel: 
            bool, whether to show velocity-colored view angle trajectory plot.

    Return：
        final_minima_idx: 
            list, indices of the filtered valid Z-axis local minima (0-based).
        filtered_minima_data: 
            DataFrame containing information about the filtered minima points.
        """
    # --- 1. Get Raw Z Values ---
    if 'Z' not in df.columns:
        print("Error: DataFrame is missing the 'Z' column.")
        return [], pd.DataFrame()
    z_values_raw = df["Z"].values.copy()
    data_length = len(z_values_raw)
    z_baseline = np.zeros_like(z_values_raw)

    # --- 2. Baseline Removal (Optional) ---
    if use_baseline_removal:
        print(f"Step 1: Applying Savitzky-Golay baseline removal (window={baseline_window_length}, order={baseline_polyorder})")
        if baseline_window_length >= data_length:
            original_wl = baseline_window_length
            baseline_window_length = data_length // 2 * 2 + 1
            if baseline_window_length < 3: baseline_window_length = 3
            if baseline_window_length <= baseline_polyorder:
                  baseline_window_length = baseline_polyorder + 1 if baseline_polyorder % 2 == 0 else baseline_polyorder + 2
            print(f"  Warning: baseline_window_length ({original_wl}) >= data length ({data_length}). Auto-adjusted to {baseline_window_length}")
        try:
            z_baseline = savgol_filter(z_values_raw, baseline_window_length, baseline_polyorder)
            z_values_processed = z_values_raw - z_baseline
            print("  Baseline removal completed.")
        except Exception as e:
            print(f"  Error: Baseline removal failed: {e}. Using raw Z values for subsequent processing.")
            z_values_processed = z_values_raw
            use_baseline_removal = False
    else:
        print("Step 1: Skipping baseline removal.")
        z_values_processed = z_values_raw

    # --- 3. Find Initial Local Minima (using argrelextrema) ---
    print(f"Step 2: Finding initial local minima using argrelextrema (order={order})")
    try:
        local_minima_idx = argrelextrema(z_values_processed, np.less, order=order)[0]
        print(f"  Found {len(local_minima_idx)} initial points.")
    except Exception as e:
        print(f"  Error: argrelextrema execution failed: {e}")
        local_minima_idx = np.array([], dtype=int)

    # --- 4. Filter by Processed Z Value Threshold (Optional) ---
    filtered_minima_idx_step4 = local_minima_idx
    if z_processed_threshold is not None:
        print(f"Step 3: Filtering points with processed Z value below {z_processed_threshold:.4f}")
        if len(local_minima_idx) > 0:
            threshold_mask = z_values_processed[local_minima_idx] < z_processed_threshold
            filtered_minima_idx_step4 = local_minima_idx[threshold_mask]
            print(f"  --> Points remaining after Z threshold filter: {len(filtered_minima_idx_step4)}")
        else:
              print("  --> No initial points to filter.")
    else:
        print("Step 3: Skipping processed Z value threshold filter.")

    # --- 5. Custom Filtering: Gap and Difference ---
    print(f"Step 4: Applying custom filter (min_gap={min_frame_gap}, min_z_diff={min_z_diff})")
    final_minima_idx = []
    if len(filtered_minima_idx_step4) > 0:
        sorted_indices = np.sort(filtered_minima_idx_step4)
        final_minima_idx.append(sorted_indices[0])
        for i in range(1, len(sorted_indices)):
            idx = sorted_indices[i]
            last_idx = final_minima_idx[-1]
            frame_diff = idx - last_idx
            if frame_diff >= min_frame_gap:
                final_minima_idx.append(idx)
            else:
                z_diff = abs(z_values_processed[idx] - z_values_processed[last_idx])
                if z_diff < min_z_diff:
                    if z_values_processed[idx] < z_values_processed[last_idx]:
                        final_minima_idx[-1] = idx
                else:
                    final_minima_idx.append(idx)
        print(f"  --> Points remaining after custom filter: {len(final_minima_idx)}")
    else:
        print("  --> No points to apply custom filter to.")

    # --- 6. Prepare Output DataFrame ---
    print("Step 5: Preparing output DataFrame")
    required_view_cols = ['cum_yaw_deg', 'cum_pitch_deg']
    has_view_data = all(col in df.columns for col in required_view_cols)
    if final_minima_idx and len(final_minima_idx) > 0:
        output_data = {
            "Frame": final_minima_idx,
            "Z Value Raw": z_values_raw[final_minima_idx],
            "Z Processed": z_values_processed[final_minima_idx]
        }
        if has_view_data:
              output_data["Yaw Angle (°)"] = df["cum_yaw_deg"].iloc[final_minima_idx].values
              output_data["Pitch Angle (°)"] = df["cum_pitch_deg"].iloc[final_minima_idx].values
        else:
              warnings.warn("Missing angle columns, output DataFrame will not include angle information.")
        filtered_minima_data = pd.DataFrame(output_data)
        print("  Final filtered minima points (first few):")
        print(filtered_minima_data.head())
    else:
        print("  No final minima points found meeting all criteria.")
        cols = ["Frame", "Z Value Raw", "Z Processed"]
        if has_view_data: cols.extend(["Yaw Angle (°)", "Pitch Angle (°)"])
        filtered_minima_data = pd.DataFrame(columns=cols)

    # --- 7. Plotting ---
    if show:
        print("Step 6: Generating plots with custom style")

        # --- Style setup function (optional helper) ---
        def apply_custom_style(ax):
            """Applies the requested style to an Axes object."""
            ax.set_facecolor('white') # White background
            ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='grey') # Grey dashed grid
            # Black solid axis lines (spines)
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.0) # Explicitly set linewidth (optional)
                spine.set_linestyle('-') # Explicitly set linestyle to solid (optional, default)
            # Black ticks and labels
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            ax.title.set_color('black')
        # --- End of style setup function ---

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

        # Apply custom style to both subplots for Z values
        for ax in axes:
            apply_custom_style(ax)

        # Subplot 1: Raw Z, Baseline, Final Minima
        axes[0].plot(z_values_raw, label='Raw Z Value', color='gray', alpha=0.7, linewidth=1)
        if use_baseline_removal:
            axes[0].plot(z_baseline, label=f'Baseline (window={baseline_window_length}, order={baseline_polyorder})', color='orange', linestyle='--', linewidth=1.5)
        if final_minima_idx and len(final_minima_idx) > 0:
            axes[0].scatter(final_minima_idx, z_values_raw[final_minima_idx], color='red', label=f'Final Minima ({len(final_minima_idx)})', zorder=5, s=60, marker='x')
        axes[0].set_title("Raw Z Value, Baseline, and Final Minima Points")
        axes[0].set_ylabel("Raw Z Value")
        axes[0].legend()
        # Style applied by loop

        # Subplot 2: Processed Z, Threshold, Final Minima
        plot_label = 'Processed Z Value' + (' (Detrended)' if use_baseline_removal else ' (Raw)')
        axes[1].plot(z_values_processed, label=plot_label, color='blue', alpha=0.8, linewidth=1)
        if z_processed_threshold is not None:
            axes[1].axhline(z_processed_threshold, color='cyan', linestyle=':', label=f'Z Threshold ({z_processed_threshold:.2f})', linewidth=1.5)
        if final_minima_idx and len(final_minima_idx) > 0:
            axes[1].scatter(final_minima_idx, z_values_processed[final_minima_idx], color='red', label=f'Final Minima ({len(final_minima_idx)})', zorder=5, s=60, marker='x')
        axes[1].set_title("Processed Z Value and Final Minima Points")
        axes[1].set_ylabel("Processed Z Value")
        axes[1].set_xlabel("Frame")
        axes[1].legend()
        # Style applied by loop

        plt.tight_layout()
        plt.show()

        # --- View Angle Plots (Apply similar styling) ---
        if has_view_data:
            if final_minima_idx and len(final_minima_idx) > 0:
                filtered_yaw = df["cum_yaw_deg"].iloc[final_minima_idx].values
                filtered_pitch = df["cum_pitch_deg"].iloc[final_minima_idx].values
            else:
                filtered_yaw, filtered_pitch = [], []
                
            padding_factor = 0.1 # 設定 10% 的邊距
            pitch_min, pitch_max = df['cum_pitch_deg'].min(), df['cum_pitch_deg'].max()
            yaw_min, yaw_max = df['cum_yaw_deg'].min(), df['cum_yaw_deg'].max()
            pitch_range_val = pitch_max - pitch_min
            yaw_range_val = yaw_max - yaw_min
            

            # Plot 8: Basic Trajectory
            fig_traj, ax_traj = plt.subplots(figsize=(8, 8), dpi=300)
            apply_custom_style(ax_traj) # Apply the style
            ax_traj.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"], c=df.index, cmap="viridis", alpha=0.7, s=10, label="View Angle Trajectory (by Frame)")
            ax_traj.scatter(filtered_pitch, filtered_yaw, color="red", s=50, label="Final Minima", zorder=3, marker='x')
            # Create invisible scatter for colorbar mapping if needed (might be optional depending on matplotlib version)
            cbar_mappable = ax_traj.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"], c=df.index, cmap="viridis", alpha=0)
            plt.colorbar(cbar_mappable, ax=ax_traj, label="Frame Index")
            ax_traj.set_xlabel("Pitch Angle (Vertical) °")
            ax_traj.set_ylabel("Yaw Angle (Horizontal, Rotated) °")
            ax_traj.set_title("View Angle Trajectory with Final Z-Axis Minima")
            ax_traj.legend()
            # Grid and axis styles are applied by apply_custom_style
            ax_traj.set_xlim(pitch_min - padding_factor * pitch_range_val, pitch_max + padding_factor * pitch_range_val)
            ax_traj.set_ylim(yaw_min - padding_factor * yaw_range_val, yaw_max + padding_factor * yaw_range_val)
            plt.show()

            # Plot 9: Velocity Colored Trajectory
            if showVel and 'speed' in df.columns:
                fig_vel, ax_vel = plt.subplots(figsize=(8, 8), dpi=300)
                apply_custom_style(ax_vel) # Apply the style
                sc = ax_vel.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"],
                                  c=df["speed"], cmap="plasma", alpha=0.7, s=10,
                                  label="View Angle Trajectory (by Speed)")
                ax_vel.scatter(filtered_pitch, filtered_yaw, color="red", s=50,
                            label="Final Minima", zorder=3, marker='x')
                plt.colorbar(sc, ax=ax_vel, label="Speed (unit unknown)")
                ax_vel.set_xlabel("Pitch Angle (Vertical) °")
                ax_vel.set_ylabel("Yaw Angle (Horizontal, Rotated) °")
                ax_vel.set_title("View Angle Trajectory (Colored by Speed) with Final Z-Axis Minima")
                ax_vel.legend()
                # 設定邊距
                
                # Grid and axis styles are applied by apply_custom_style
                ax_vel.set_xlim(pitch_min - padding_factor * pitch_range_val, pitch_max + padding_factor * pitch_range_val)
                ax_vel.set_ylim(yaw_min - padding_factor * yaw_range_val, yaw_max + padding_factor * yaw_range_val)
                plt.show()
            elif showVel and 'speed' not in df.columns:
                  warnings.warn("DataFrame is missing the 'speed' column, cannot generate velocity-colored plot.")
        else:
              print("Skipping view angle plots due to missing angle columns.")

    print("--- Analysis Finished ---")
    return final_minima_idx, filtered_minima_data

# --- Example Usage (No changes needed here for style) ---
# if __name__ == "__main__":
#     # ... (Your example usage code) ...
# %%
def ConverUnit2Angle(combine_dict, descriptions,
                     marker="R.I.Finger3",
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
    # df = pd.DataFrame(combine_dict["markers"][marker],
    #                   columns=["X", "Y", "Z"])
    df = pd.DataFrame(combine_dict["markers"][marker][:, 0:2],
                      columns=["X", "Y"])
    df_2 = pd.DataFrame(combine_dict["markers"][marker][:, -1],
                        columns=["Z"])
    df = pd.concat([df, df_2], axis=1)
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
    sampling_rate = descriptions["motion_info"]["frame_rate"]
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
    
    
    # final_minima_idx = final_indices

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
    # --- Style setup function (copied from previous response) ---
    def apply_custom_style(ax):
        """Applies the requested style to an Axes object."""
        ax.set_facecolor('white') # White background
        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='grey') # Grey dashed grid
        # Black solid axis lines (spines)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0) # Explicitly set linewidth
            spine.set_linestyle('-') # Explicitly set linestyle to solid
        # Black ticks and labels
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')
    # 目標：如果 show=True，繪製視角軌跡圖，用顏色深淺表示滑鼠移動速度，並標記 Z 最小值點。
    if show:
        # Create figure and axes objects
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300) # Use subplots to get axes object
    
        # Apply the custom style to the axes
        apply_custom_style(ax)
    
        # Ensure 'speed' column exists before using it for color
        if 'speed' in df.columns:
            # Plot the main view angle trajectory scatter plot using the axes object
            sc = ax.scatter(
                df['cum_pitch_deg'], df['cum_yaw_deg'], # X, Y coordinates
                c=df['speed'], cmap='plasma', alpha=0.7, s=10, # Use s=10 for slightly larger points
                label='View Angle Trajectory (colored by speed)' # Legend label
            )
            # Add colorbar, associating it with the axes object
            plt.colorbar(sc, ax=ax, label='Mouse Speed (°/s or unit of speed column)')
        else:
            # Fallback if 'speed' column is missing: color by index
            print("Warning: 'speed' column not found in DataFrame. Coloring by frame index instead.")
            sc = ax.scatter(
                df['cum_pitch_deg'], df['cum_yaw_deg'], # X, Y coordinates
                c=df.index, cmap='viridis', alpha=0.7, s=10,
                label='View Angle Trajectory (colored by frame)' # Updated label
            )
            plt.colorbar(sc, ax=ax, label='Frame Index')
    
    
        # Mark the Z-axis local minima points on the plot
        if final_minima_idx is not None and len(final_minima_idx) > 0:
            # Ensure indices are valid before using iloc
            valid_minima_idx = df.index.intersection(final_minima_idx)
            if len(valid_minima_idx) > 0:
                  minima_pitch = df.loc[valid_minima_idx, 'cum_pitch_deg'].values
                  minima_yaw = df.loc[valid_minima_idx, 'cum_yaw_deg'].values
                  ax.scatter(minima_pitch, minima_yaw,
                            color='red', s=50, # Make minima markers larger
                            label='Z Minima', zorder=3, marker='x') # Use 'x' marker
            else:
                  print("Warning: None of the final_minima_idx were found in the DataFrame index.")
        # 設定邊距
        padding_factor = 0.1 # 設定 10% 的邊距
        pitch_min, pitch_max = df['cum_pitch_deg'].min(), df['cum_pitch_deg'].max()
        yaw_min, yaw_max = df['cum_yaw_deg'].min(), df['cum_yaw_deg'].max()
        pitch_range_val = pitch_max - pitch_min
        yaw_range_val = yaw_max - yaw_min
        ax.set_xlim(pitch_min - padding_factor * pitch_range_val, pitch_max + padding_factor * pitch_range_val)
        ax.set_ylim(yaw_min - padding_factor * yaw_range_val, yaw_max + padding_factor * yaw_range_val)
        # Set chart labels, title, and aspect ratio using the axes object
        ax.set_xlabel('Pitch Angle (°)')
        ax.set_ylabel('Yaw Angle (°)')
        ax.set_title('View Angle Trajectory Colored by Mouse Speed with Z Minima')
        # ax.axis('equal') # Ensure equal aspect ratio
        ax.legend() # Show legend
        # Grid and axis styles are already set by apply_custom_style
    
        plt.tight_layout() # Adjust layout
        plt.show() # Display the plot

    # 返回最終處理好的包含群組資訊的 DataFrame
    return grouped_df

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

# def excludeCenter(df: pd.DataFrame,
#                   grouped_df: pd.DataFrame,
#                   yaw_range: float = 10,
#                   pitch_range: float = 10,
#                   show: bool = True) -> pd.DataFrame:
#     """
#     Filters kill action groups, retaining only those whose endpoint is outside the central view area.
#     If show=True, displays two separate visualizations:
#     1. Point Distribution Plot: Shows all points, retained group points, and the central exclusion zone.
#     2. Arrow Plot: Shows movement direction arrows (from start frame to end frame) for retained groups.

#     Args:
#         df (pd.DataFrame): DataFrame containing the original data ('cum_yaw_deg', 'cum_pitch_deg').
#         grouped_df (pd.DataFrame): Pre-calculated kill group DataFrame (must include 'Frames' list).
#         yaw_range (float): Horizontal radius of the central area (degrees).
#         pitch_range (float): Vertical radius of the central area (degrees).
#         show (bool): Whether to display the visualization plots.

#     Returns:
#         pd.DataFrame: Filtered DataFrame containing only groups whose endpoint is not in the center.
#     """

#     # === 1. Define Central View Area ===
#     try:
#         # Using median as center calculation method
#         yaw_center = df["cum_yaw_deg"].median()
#         pitch_center = df["cum_pitch_deg"].median()
#         print(f"[Based on Median] View center calculated: Yaw={yaw_center:.2f}°, Pitch={pitch_center:.2f}°")
#         print(f"Central area defined: Yaw ±{yaw_range}°, Pitch ±{pitch_range}°")
#     except KeyError as e:
#         print(f"Error: Input df is missing required column {e}")
#         return pd.DataFrame() # Return empty DataFrame or raise exception

#     # === 2. Vectorized Filtering ===
#     temp_grouped = grouped_df.copy()

#     def get_last_frame(frames_list):
#         if isinstance(frames_list, list) and len(frames_list) > 0:
#             try:
#                 return int(frames_list[-1])
#             except (ValueError, TypeError):
#                 return np.nan # Return NaN if conversion fails
#         return np.nan

#     if 'Frames' not in temp_grouped.columns:
#         print("Error: grouped_df is missing the 'Frames' column")
#         return pd.DataFrame()

#     temp_grouped['last_frame'] = temp_grouped['Frames'].apply(get_last_frame)

#     # Check if necessary coordinate columns exist in df
#     if 'cum_yaw_deg' not in df.columns or 'cum_pitch_deg' not in df.columns:
#         print("Error: df is missing 'cum_yaw_deg' or 'cum_pitch_deg' column")
#         return pd.DataFrame()

#     # Use map for efficient lookup
#     yaw_map = df['cum_yaw_deg']
#     pitch_map = df['cum_pitch_deg']

#     temp_grouped['last_yaw'] = temp_grouped['last_frame'].map(yaw_map)
#     temp_grouped['last_pitch'] = temp_grouped['last_frame'].map(pitch_map)

#     # Mask for rows where coordinates could be successfully retrieved
#     valid_coords_mask = temp_grouped['last_yaw'].notna() & temp_grouped['last_pitch'].notna()

#     # Initialize mask for points within the center
#     is_in_center_mask = pd.Series(False, index=temp_grouped.index)
#     # Calculate 'is_in_center' only for rows with valid coordinates
#     if valid_coords_mask.any():
#         is_in_center_mask.loc[valid_coords_mask] = (
#             (abs(temp_grouped.loc[valid_coords_mask, 'last_yaw'] - yaw_center) <= yaw_range) &
#             (abs(temp_grouped.loc[valid_coords_mask, 'last_pitch'] - pitch_center) <= pitch_range)
#         )

#     # Keep rows that have valid coordinates AND are NOT in the center
#     keep_mask = valid_coords_mask & (~is_in_center_mask)
#     filtered_grouped_df = grouped_df.loc[keep_mask].reset_index(drop=True)

#     # === 3. Calculate and Report Excluded Count ===
#     original_count = len(grouped_df)
#     filtered_count = len(filtered_grouped_df)
#     excluded_count = original_count - filtered_count
#     print(f"Original group count: {original_count}")
#     print(f"Groups excluded due to **endpoint in center** or **invalid/missing data**: {excluded_count}")
#     print(f"Filtered group count remaining: {filtered_count}")

#     # === 4. Visualization (Clearly separated into two plots) ===
#     if show:
#         if filtered_grouped_df.empty:
#             print("No filtered groups available for plotting.")
#         else:
#             # --- Prepare plotting data (calculate only once if possible) ---
#             # Flatten list of lists, handle potential non-list entries or NaNs within lists
#             all_retained_frames_flat = []
#             for frames_list in filtered_grouped_df["Frames"]:
#                 if isinstance(frames_list, list):
#                     all_retained_frames_flat.extend([f for f in frames_list if pd.notna(f) and isinstance(f, (int, float))])

#             # Get unique, valid frame indices that exist in the original DataFrame
#             valid_retained_frames_idx = df.index.intersection(pd.unique(all_retained_frames_flat))

#             # --- Plot 1: Point Distribution, Group Extents, and Center Area ---
#             try:
#                 plt.figure(figsize=(10, 8))
#                 ax1 = plt.gca()
#                 # Background points
#                 ax1.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"], alpha=0.1, s=5, color='gray', label="All Data Points") # English Label

#                 # Points belonging to retained groups (blue)
#                 if not valid_retained_frames_idx.empty:
#                     ax1.scatter(df.loc[valid_retained_frames_idx, "cum_pitch_deg"], df.loc[valid_retained_frames_idx, "cum_yaw_deg"],
#                                 color='blue', s=20, alpha=0.6, label="Retained Group Points", zorder=3) # English Label

#                 # Optional: Mark retained group extents (red hollow circles)
#                 red_circle_legend_added = False
#                 for idx, row in filtered_grouped_df.iterrows():
#                     group_frames = row["Frames"]
#                     if not isinstance(group_frames, list) or not group_frames: continue
#                     valid_group_indices = [f for f in group_frames if pd.notna(f) and isinstance(f, (int, float))]
#                     group_indices_in_df = df.index.intersection(valid_group_indices)

#                     if group_indices_in_df.empty: continue

#                     try:
#                         group_pitch = df.loc[group_indices_in_df, "cum_pitch_deg"]
#                         group_yaw = df.loc[group_indices_in_df, "cum_yaw_deg"]
#                         # Mark group extent with semi-transparent red border
#                         ax1.scatter(group_pitch, group_yaw, facecolors='none', edgecolors='red',
#                                     s=80, linewidths=1.5, alpha=0.7,
#                                     label="Retained Group Extent" if not red_circle_legend_added else "", zorder=2) # English Label
#                         if not red_circle_legend_added: red_circle_legend_added = True
#                     except KeyError:
#                         warnings.warn(f"Could not find some frame indices {group_indices_in_df} in df when plotting red circle for group {idx}.") # Use warnings

#                 # Central exclusion zone (green dashed rectangle)
#                 rect_pitch = [pitch_center - pitch_range, pitch_center + pitch_range, pitch_center + pitch_range, pitch_center - pitch_range, pitch_center - pitch_range]
#                 rect_yaw = [yaw_center - yaw_range, yaw_center - yaw_range, yaw_center + yaw_range, yaw_center + yaw_range, yaw_center - yaw_range]
#                 ax1.plot(rect_pitch, rect_yaw, color='green', linestyle='--', linewidth=2, label="Central Zone (Excluded Endpoints)") # English Label

#                 # Chart elements
#                 ax1.set_xlabel("Pitch Angle (°)") # English Label
#                 ax1.set_ylabel("Yaw Angle (°)")   # English Label
#                 ax1.set_title("Filtered Kill Groups Visualization (Points, Extents & Exclusion Zone)") # English Title
#                 ax1.grid(True)
#                 ax1.axis("equal") # Maintain aspect ratio
#                 # Consolidate legend
#                 handles, labels = ax1.get_legend_handles_labels()
#                 by_label = dict(zip(labels, handles)) # Remove duplicate labels
#                 ax1.legend(by_label.values(), by_label.keys())
#                 plt.show() # Display the first plot
#             except Exception as e:
#                 print(f"Error occurred during plotting (Plot 1): {e}")


#             # --- Plot 2: Movement Direction Arrows ---
#             try:
#                 plt.figure(figsize=(10, 8))
#                 ax2 = plt.gca()
#                 # Optional: Background points
#                 ax2.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"], alpha=0.05, s=5, color='gray', label="All Data Points (Background)") # English Label
#                 # Optional: Retained group points (blue, for reference)
#                 if not valid_retained_frames_idx.empty:
#                     ax2.scatter(df.loc[valid_retained_frames_idx, "cum_pitch_deg"], df.loc[valid_retained_frames_idx, "cum_yaw_deg"],
#                                 color='blue', s=10, alpha=0.3, label="Retained Group Points (Reference)", zorder=2) # English Label

#                 arrow_drawn = False # Legend label control
#                 # Iterate and draw arrows
#                 for idx, row in filtered_grouped_df.iterrows():
#                     group_frames = row["Frames"]
#                     if isinstance(group_frames, list) and len(group_frames) >= 2:
#                         try:
#                             start_frame = int(group_frames[0])
#                             end_frame = int(group_frames[-1])

#                             # Get coordinates with error checking
#                             if start_frame not in df.index or end_frame not in df.index:
#                                 warnings.warn(f"Start frame {start_frame} or end frame {end_frame} for group {idx} not in DataFrame index.") # Use warnings
#                                 continue

#                             pitch_start = df.loc[start_frame, "cum_pitch_deg"]
#                             yaw_start = df.loc[start_frame, "cum_yaw_deg"]
#                             pitch_end = df.loc[end_frame, "cum_pitch_deg"]
#                             yaw_end = df.loc[end_frame, "cum_yaw_deg"]

#                             # Check for NaN coordinates
#                             if pd.isna(pitch_start) or pd.isna(yaw_start) or pd.isna(pitch_end) or pd.isna(yaw_end):
#                                 warnings.warn(f"Start or end coordinates are NaN for group {idx}.") # Use warnings
#                                 continue

#                             # Draw arrow (red dashed)
#                             ax2.annotate(
#                                 '', xy=(pitch_end, yaw_end), xytext=(pitch_start, yaw_start),
#                                 arrowprops=dict(arrowstyle="->", color="red", alpha=0.5,
#                                             linestyle="--", lw=1, shrinkA=5, shrinkB=5),
#                                 zorder=3)
#                             if not arrow_drawn:
#                                 # Add legend entry only once
#                                 ax2.plot([], [], color='red', alpha=0.5,
#                                          linestyle="--",
#                                          lw=1.5, label='Kill Action Direction (Start->End)') # English Label
#                                 arrow_drawn = True
#                         except (KeyError, ValueError, TypeError) as frame_err:
#                             warnings.warn(f"Error processing frames {group_frames} for group {idx}: {frame_err}") # Use warnings
#                             continue # Skip arrow for this group

#                 # Chart elements
#                 ax2.set_xlabel("Pitch Angle (°)") # English Label
#                 ax2.set_ylabel("Yaw Angle (°)")   # English Label
#                 ax2.set_title("Movement Direction Arrows of Filtered Kill Groups") # English Title
#                 ax2.grid(True)
#                 ax2.axis("equal") # Maintain aspect ratio
#                 # Consolidate legend
#                 handles, labels = ax2.get_legend_handles_labels()
#                 by_label = dict(zip(labels, handles)) # Remove duplicate labels
#                 if by_label: # Only show legend if there's something to show
#                     ax2.legend(by_label.values(), by_label.keys())
#                 plt.show() # Display the second plot
#             except Exception as e:
#                 print(f"Error occurred during plotting (Plot 2): {e}")

#     # === 5. Return Result ===
#     return filtered_grouped_df

# %%
def excludeCenter(df: pd.DataFrame,
                  grouped_df: pd.DataFrame,
                  yaw_range: float = 10,
                  pitch_range: float = 10,
                  rotation_angle: float = 0,  # <== 新增參數
                  show: bool = True) -> pd.DataFrame:
    """
    Filters kill action groups, retaining only those whose endpoint is outside the central view area (now supports rotated parallelogram).
    
    Args:
        df (pd.DataFrame): DataFrame containing the original data ('cum_yaw_deg', 'cum_pitch_deg').
        grouped_df (pd.DataFrame): Pre-calculated kill group DataFrame (must include 'Frames' list).
        yaw_range (float): Horizontal half-width of the central area (degrees).
        pitch_range (float): Vertical half-height of the central area (degrees).
        rotation_angle (float): Rotation angle (degrees) of the central area parallelogram.
        show (bool): Whether to display visualization plots.
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only groups whose endpoint is not in the central area.
    """

    # === 1. Define Central View Area ===
    try:
        yaw_center = df["cum_yaw_deg"].median()
        pitch_center = df["cum_pitch_deg"].median()
        print(f"[Based on Median] View center calculated: Yaw={yaw_center:.2f}°, Pitch={pitch_center:.2f}°")
        print(f"Central area defined: Yaw ±{yaw_range}°, Pitch ±{pitch_range}°, Rotated by {rotation_angle}°")
    except KeyError as e:
        print(f"Error: Input df is missing required column {e}")
        return pd.DataFrame()

    # === 2. Vectorized Filtering ===
    temp_grouped = grouped_df.copy()

    def get_last_frame(frames_list):
        if isinstance(frames_list, list) and len(frames_list) > 0:
            try:
                return int(frames_list[-1])
            except (ValueError, TypeError):
                return np.nan
        return np.nan

    if 'Frames' not in temp_grouped.columns:
        print("Error: grouped_df is missing the 'Frames' column")
        return pd.DataFrame()

    temp_grouped['last_frame'] = temp_grouped['Frames'].apply(get_last_frame)

    if 'cum_yaw_deg' not in df.columns or 'cum_pitch_deg' not in df.columns:
        print("Error: df is missing 'cum_yaw_deg' or 'cum_pitch_deg' column")
        return pd.DataFrame()

    yaw_map = df['cum_yaw_deg']
    pitch_map = df['cum_pitch_deg']

    temp_grouped['last_yaw'] = temp_grouped['last_frame'].map(yaw_map)
    temp_grouped['last_pitch'] = temp_grouped['last_frame'].map(pitch_map)

    valid_coords_mask = temp_grouped['last_yaw'].notna() & temp_grouped['last_pitch'].notna()

    # === 2b. Calculate rotated parallelogram corners ===
    from matplotlib.path import Path
    
    theta = np.deg2rad(rotation_angle)
    half_w = yaw_range
    half_h = pitch_range
    
    local_corners = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
    ])
    # rotation
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])
    rotated_corners = local_corners @ rotation_matrix.T
    # shift to center
    rotated_corners[:, 0] += yaw_center
    rotated_corners[:, 1] += pitch_center
    
    central_poly = Path(rotated_corners)

    # === 2c. New center test ===
    is_in_center_mask = pd.Series(False, index=temp_grouped.index)
    if valid_coords_mask.any():
        is_in_center_mask.loc[valid_coords_mask] = temp_grouped.loc[valid_coords_mask].apply(
            lambda row: central_poly.contains_point((row['last_yaw'], row['last_pitch'])),
            axis=1
        )

    keep_mask = valid_coords_mask & (~is_in_center_mask)
    filtered_grouped_df = grouped_df.loc[keep_mask].reset_index(drop=True)

    # === 3. Statistics ===
    original_count = len(grouped_df)
    filtered_count = len(filtered_grouped_df)
    excluded_count = original_count - filtered_count
    print(f"Original group count: {original_count}")
    print(f"Groups excluded due to endpoint in rotated center area or invalid data: {excluded_count}")
    print(f"Filtered group count remaining: {filtered_count}")

    # === 4. Visualization (unchanged, except central zone visualization updated) ===
    if show:
        if filtered_grouped_df.empty:
            print("No filtered groups available for plotting.")
        else:
            # retained group points
            all_retained_frames_flat = []
            for frames_list in filtered_grouped_df["Frames"]:
                if isinstance(frames_list, list):
                    all_retained_frames_flat.extend([f for f in frames_list if pd.notna(f) and isinstance(f, (int, float))])
            valid_retained_frames_idx = df.index.intersection(pd.unique(all_retained_frames_flat))

            # plot 1
            try:
                plt.figure(figsize=(10, 8))
                ax1 = plt.gca()
                ax1.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"], alpha=0.1, s=5, color='gray', label="All Data Points")
                if not valid_retained_frames_idx.empty:
                    ax1.scatter(df.loc[valid_retained_frames_idx, "cum_pitch_deg"], df.loc[valid_retained_frames_idx, "cum_yaw_deg"],
                                color='blue', s=20, alpha=0.6, label="Retained Group Points", zorder=3)
                # red hollow circles
                red_circle_legend_added = False
                for idx, row in filtered_grouped_df.iterrows():
                    group_frames = row["Frames"]
                    if not isinstance(group_frames, list) or not group_frames: continue
                    valid_group_indices = [f for f in group_frames if pd.notna(f) and isinstance(f, (int, float))]
                    group_indices_in_df = df.index.intersection(valid_group_indices)
                    if group_indices_in_df.empty: continue
                    group_pitch = df.loc[group_indices_in_df, "cum_pitch_deg"]
                    group_yaw = df.loc[group_indices_in_df, "cum_yaw_deg"]
                    ax1.scatter(group_pitch, group_yaw, facecolors='none', edgecolors='red', s=80, linewidths=1.5, alpha=0.7,
                                label="Retained Group Extent" if not red_circle_legend_added else "", zorder=2)
                    if not red_circle_legend_added:
                        red_circle_legend_added = True

                # draw rotated parallelogram
                ax1.plot(rotated_corners[:,1].tolist() + [rotated_corners[0,1]],
                         rotated_corners[:,0].tolist() + [rotated_corners[0,0]],
                         color='green', linestyle='--', linewidth=2, label="Rotated Central Zone")

                ax1.set_xlabel("Pitch Angle (°)")
                ax1.set_ylabel("Yaw Angle (°)")
                ax1.set_title("Filtered Kill Groups Visualization (Rotated Parallelogram Zone)")
                ax1.grid(True)
                ax1.axis("equal")
                handles, labels = ax1.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax1.legend(by_label.values(), by_label.keys())
                plt.show()
            except Exception as e:
                print(f"Error during plotting (Plot 1): {e}")

            # plot 2
            try:
                plt.figure(figsize=(10, 8))
                ax2 = plt.gca()
                ax2.scatter(df["cum_pitch_deg"], df["cum_yaw_deg"], alpha=0.05, s=5, color='gray', label="All Data Points (Background)")
                if not valid_retained_frames_idx.empty:
                    ax2.scatter(df.loc[valid_retained_frames_idx, "cum_pitch_deg"], df.loc[valid_retained_frames_idx, "cum_yaw_deg"],
                                color='blue', s=10, alpha=0.3, label="Retained Group Points (Reference)", zorder=2)
                arrow_drawn = False
                for idx, row in filtered_grouped_df.iterrows():
                    group_frames = row["Frames"]
                    if isinstance(group_frames, list) and len(group_frames) >= 2:
                        start_frame = int(group_frames[0])
                        end_frame = int(group_frames[-1])
                        if start_frame not in df.index or end_frame not in df.index:
                            continue
                        pitch_start = df.loc[start_frame, "cum_pitch_deg"]
                        yaw_start = df.loc[start_frame, "cum_yaw_deg"]
                        pitch_end = df.loc[end_frame, "cum_pitch_deg"]
                        yaw_end = df.loc[end_frame, "cum_yaw_deg"]
                        if pd.isna(pitch_start) or pd.isna(yaw_start) or pd.isna(pitch_end) or pd.isna(yaw_end):
                            continue
                        ax2.annotate(
                            '', xy=(pitch_end, yaw_end), xytext=(pitch_start, yaw_start),
                            arrowprops=dict(arrowstyle="->", color="red", alpha=0.5, linestyle="--", lw=1, shrinkA=5, shrinkB=5),
                            zorder=3)
                        if not arrow_drawn:
                            ax2.plot([], [], color='red', alpha=0.5, linestyle="--", lw=1.5, label='Kill Action Direction (Start->End)')
                            arrow_drawn = True
                ax2.set_xlabel("Pitch Angle (°)")
                ax2.set_ylabel("Yaw Angle (°)")
                ax2.set_title("Movement Direction Arrows of Filtered Kill Groups")
                ax2.grid(True)
                ax2.axis("equal")
                handles, labels = ax2.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                if by_label:
                    ax2.legend(by_label.values(), by_label.keys())
                plt.show()
            except Exception as e:
                print(f"Error during plotting (Plot 2): {e}")

    return filtered_grouped_df

# %%
def plot_standardized_signals_cloud_compare(
        datasets,                   # List of signal dictionaries (required)
        target_length,              # Standardized length of signals (required)
        title="Comparison of Mean ± Std Dev Clouds",
        xlabel="Normalized Time (%)",
        ylabel="Signal Value (°/s or other units)",
        # Optional: Provide lists for labels and colors, otherwise defaults are used
        labels=None,                # List of labels corresponding to datasets
        # color_indices=None,          # List of color indices corresponding to datasets
        color_hex_codes: Optional[List[str]] = None
    ):
    """
    Plots the mean and standard deviation range for one or more (up to N) datasets of
    standardized signals on the same figure.

    Parameters:
        datasets (list[dict]):  List where each element is a dictionary containing standardized signals
                                (Keys: ID, Values: 1D NumPy array).
        target_length (int):    The standardized length of the signals. Must be the same
                                for all datasets.
        title (str):            Title for the plot.
        xlabel (str):           Label for the x-axis.
        ylabel (str):           Label for the y-axis.
        labels (list[str], optional): List of legend labels for each dataset. If None, defaults
                                      like "Dataset 1", "Dataset 2", etc., will be used.
        color_indices (list[int], optional): List of indices for the color from 'Set1' palette
                                             for each dataset. If None, indices 0, 1, 2,... will be used.
    """
    # --- Define Inner Helper Function ---
    def _process_and_calculate_stats(signals_dict, target_length):
        """(Internal helper function) Process signal dictionary and calculate statistics"""
        if not signals_dict:
            print("Warning: Provided signal dictionary is empty.")
            return None, None, None, 0

        # Handle cases where values might not be lists/arrays directly
        try:
            signals_list = list(signals_dict.values())
        except AttributeError: # Handle if signals_dict is not dict-like
             print("Warning: Input signals_dict is not a dictionary or dictionary-like object.")
             return None, None, None, 0

        if not signals_list:
            print("Warning: Could not extract any valid signal arrays from the dictionary.")
            return None, None, None, 0

        # Filter and stack signals
        valid_signals = [s for s in signals_list if isinstance(s, np.ndarray) and s.ndim == 1 and s.shape[0] == target_length]
        if not valid_signals:
            print(f"Warning: Could not find any valid signals (1D NumPy array with length {target_length}).")
            return None, None, None, 0

        try:
            signals_array = np.stack(valid_signals, axis=1)
            num_signals = signals_array.shape[1]
        except Exception as e:
            print(f"Error: Could not stack arrays during data preparation (check if all lengths are {target_length}): {e}")
            return None, None, None, 0

        # Calculate statistics (ignore NaN)
        with np.errstate(all='ignore'):
            avg_signal = np.nanmean(signals_array, axis=1)
            std_signal = np.nanstd(signals_array, axis=1)

        if np.all(np.isnan(avg_signal)) or np.all(np.isnan(std_signal)):
            print("Warning: Calculated mean or standard deviation are all NaN (perhaps all input signals were invalid or all NaN).")
            return None, None, None, num_signals

        lower_bound = avg_signal - std_signal
        upper_bound = avg_signal + std_signal

        return avg_signal, lower_bound, upper_bound, num_signals
    # --- End of Inner Helper Function Definition ---


    # --- Main Function Logic Starts Here ---
    if not isinstance(datasets, list) or not datasets:
        print("Error: 'datasets' must be a non-empty list of dictionaries.")
        return

    num_datasets = len(datasets)

    # --- Setup Labels and Colors ---
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(num_datasets)]
    elif len(labels) != num_datasets:
        print(f"Warning: Number of labels ({len(labels)}) does not match number of datasets ({num_datasets}). Using default labels.")
        labels = [f'Dataset {i+1}' for i in range(num_datasets)]

    # if color_indices is None:
    #     color_indices = list(range(num_datasets))
    # elif len(color_indices) != num_datasets:
    #     print(f"Warning: Number of color_indices ({len(color_indices)}) does not match number of datasets ({num_datasets}). Using default indices.")
    #     color_indices = list(range(num_datasets))

    # --- Create figure and x-axis ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    iters = np.linspace(0, 100, target_length) # x-axis: 0% to 100%
    # try:
    #     # Use a colormap with more distinct colors if plotting many lines
    #     if color_hex_codes and num_datasets > plt.get_cmap('Set1').N:
    #         color = color_hex_codes[num_datasets]
    #         # palette = plt.get_cmap('tab10') # Or 'tab20', 'viridis', etc.
    #         # print(f"Warning: More datasets ({num_datasets}) than distinct colors in 'Set1'. Switched to '{palette.name}' colormap.")
    #     else:
    #         palette = plt.get_cmap('Set1')
    # except ValueError:
    #     print("Warning: Colormap 'Set1' not found. Using default 'viridis'.")
    #     palette = plt.get_cmap('viridis')

    plot_success_count = 0

    # --- Loop through datasets, process and plot ---
    # plotted_colors = set() # Keep track of used colors to avoid reuse if indices clash
    for i in range(num_datasets):
        if color_hex_codes:
        # and num_datasets > plt.get_cmap('Set1').N:
            color = color_hex_codes[i]
            # palette = plt.get_cmap('tab10') # Or 'tab20', 'viridis', etc.
            # print(f"Warning: More datasets ({num_datasets}) than distinct colors in 'Set1'. Switched to '{palette.name}' colormap.")
        else:
            palette = plt.get_cmap('Set1')
            color = palette(i % palette.N)
        signals_dict = datasets[i]
        label = labels[i]
        # color_idx = color_indices[i]

        print(f"\nProcessing {label}...")
        avg, lower, upper, count = _process_and_calculate_stats(signals_dict, target_length)

        if avg is not None:
            # Assign color, ensuring uniqueness if indices clash
            # current_color_idx = color_idx
            # while current_color_idx in plotted_colors:
            #      print(f"Warning: Color index {current_color_idx} for '{label}' already used. Trying next index.")
            #      current_color_idx += 1
            # color = palette(current_color_idx % palette.N) # Use modulo for safety
            # plotted_colors.add(current_color_idx)

            ax.plot(iters, avg, color=color, linewidth=2)
            ax.fill_between(iters, lower, upper, color=color, alpha=0.15)
            plot_success_count += 1
        else:
            print(f"Could not successfully process or calculate statistics for {label}.")

    # --- Plot Formatting ---
    if plot_success_count > 0:
        # 設置 title
        # ax.text(x=0.0, y=1.1, s=title, fontsize=32,
        #         transform=ax.transAxes,
        #         ha='left', va='bottom')
        # ax.set_title(title, fontsize=20, loc='left')
        # ax.legend(loc="best")
        # 設置背景格式
        ax.grid(True, linestyle=(0, (10, 5)), alpha=0.5, linewidth=0.5)
        # 設定邊框格式
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)         # 下方邊框加粗
        ax.spines['left'].set_linewidth(2)           # 左側邊框加粗
        ax.spines['bottom'].set_color('#959595')     # 下方邊框改為深灰色
        ax.spines['left'].set_color('#959595')     # 下方邊框改為深灰色
        
        # 設置 X 軸格式
        ax.set_xlim(left=0, right=100)
        ax.set_xlabel(xlabel, fontsize=20, color="#868686")
        ax.tick_params(axis='x', labelsize=12, which='both', length=0, labelcolor="#868686")
        # 設置 Y 軸格式
        ax.set_ylabel(ylabel, fontsize=20, labelpad=30, rotation=270, color="#868686")
        ax.yaxis.set_label_coords(-0.07, 0.5)  # (x, y) → y=0.0 對齊 X 軸
        
        ax.yaxis.tick_right()                  # 把刻度值也放右邊
        ax.tick_params(axis='y', labelsize=12, which='both', length=0, labelcolor="#868686")
        
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo datasets were plotted successfully, figure not shown.")
        plt.close(fig)




























