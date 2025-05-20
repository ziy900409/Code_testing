from typing import Dict, Any, Optional, Tuple, List
import ezc3d
import pandas as pd
import numpy as np

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

        # Use ffill and bfill to handle any remaining NaNs (e.g., at the very start/end if limit_direction='both' wasn't enough)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Final check if any NaNs persist (shouldn't happen with ffill/bfill, but as a safeguard)
        if df.isnull().values.any():
            print("Warning: NaNs remain after interpolation and fill. Filling with 0.")
            df.fillna(0, inplace=True) # Fill any persistent NaNs with 0 as a last resort

        return df.values

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