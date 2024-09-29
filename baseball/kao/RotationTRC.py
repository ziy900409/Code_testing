import numpy as np
import pandas as pd
import os
import csv
import math


# %%

def append_per_label_data(trc_data, markers, data):
    """
    根據標記點和資料，將資料附加到對應的標記點座標列表中。
    
    :param trc_data: 一個字典，包含 TRC 資料。
    :param markers: 標記點列表。
    :param data: 要附加的資料。
    """
    for index, marker_data in enumerate(data):
        trc_data[markers[index]] += [marker_data]

def _convert_coordinates(coordinate_data):
    """將座標數據轉換為浮點數列表，並處理無效數據。"""
    try:
        return [float(x) for x in coordinate_data]
    except ValueError:
        return [float('nan')] * len(coordinate_data)

def readTRC(filename):
    # 1. read data
    # filename = r"E:\Hsin\NTSU_lab\Kao\S2_2.trc"

    
    trc_data = {}
    with open(filename, 'rb') as f:
        contents = f.read().decode(encoding="utf-8", errors="strict")
    
    contents = contents.split('\n')
    # 2. processing data
    markers = []
    file_header_keys = []
    data_header_markers = []
    data_format_count = 0
    header_read_successfully = False
    current_line_number = 0
    
    for line in contents:
        current_line_number += 1
        print(current_line_number)
        line = line.strip()
        # print(line)
    
        if current_line_number == 1:
            # 檔案標頭第1行
            sections = line.split('\t')
            if len(sections) != 4:
                raise IOError('檔案格式無效：標頭第1行應包含四個以 tab 分隔的部分。')
            trc_data[sections[0]] = sections[1]
            trc_data['DataFormat'] = sections[2]
            data_format_count = len(sections[2].split('/'))
            trc_data['FileName'] = sections[3]
    
        elif current_line_number == 2:
            # 檔案標頭第2行
            file_header_keys = line.split('\t')
            print(file_header_keys)
    
        elif current_line_number == 3:
            # 檔案標頭第3行
            file_header_data = line.split('\t')
            if len(file_header_keys) == len(file_header_data):
                for index, key in enumerate(file_header_keys):
                    if key == 'Units':
                        trc_data[key] = file_header_data[index]
                    else:
                        trc_data[key] = float(file_header_data[index])
            else:
                raise IOError('檔案格式無效：標頭鍵數量與標頭資料數量不一致。')
    
        elif current_line_number == 4:
            # 資料標頭第1行
            data_header_markers = line.split('\t')
            if data_header_markers[0] != 'Frame#':
                raise IOError('檔案格式無效：資料標頭未以 "Frame#" 開頭。')
            if data_header_markers[1] != 'Time':
                raise IOError('檔案格式無效：資料標頭第2欄位不是 "Time"。')
    
            trc_data['Frame#'] = []
            trc_data['Time'] = []
    
        elif current_line_number == 5:
            # 資料標頭第2行
            data_header_sub_marker = line.split('\t')
            # if len(data_header_markers) != len(data_header_sub_marker):
            #     raise IOError('檔案格式無效：資料標頭的標記數量與子標記數量不一致。')
    
            # 移除 'Frame#' 和 'Time'
            data_header_markers.pop(0)
            data_header_markers.pop(0)
            markers = []
            for marker in data_header_markers:
                marker = marker.strip()
                if len(marker):
                    trc_data[marker] = []
                    markers.append(marker)
    
            trc_data['Markers'] = markers
    
        elif current_line_number == 6 and len(line) == 0:
            # 空行，標頭部分已讀取成功
            header_read_successfully = True
    
        else:
            # 有些檔案在第6行沒有空行
            if current_line_number == 6:
                header_read_successfully = True
    
            # 資料部分
            if header_read_successfully:
                sections = line.split('\t')
    
                try:
                    frame = int(sections.pop(0))
                    trc_data['Frame#'].append(frame)
                except ValueError:
                    if int(trc_data['NumFrames']) == len(trc_data['Frame#']):
                        # 已達到指定的幀數
                        continue
                    else:
                        raise IOError(f"檔案格式無效：資料幀 {len(trc_data['Frame#'])} 無效。")
    
                time = float(sections.pop(0))
                trc_data['Time'].append(time)
    
                line_data = [[float('nan')] * data_format_count] * int(trc_data['NumMarkers'])
                len_section = len(sections)
                expected_entries = len(line_data) * data_format_count
    
                if len_section > expected_entries:
                    print(f'數據行錯誤，幀：{frame}，時間：{time}，預期條目：{expected_entries}，實際條目：{len_section}')
                    trc_data[frame] = (time, line_data)
                    append_per_label_data(trc_data, markers, line_data)
                elif len_section % data_format_count == 0:
                    for index, place in enumerate(range(0, len_section, data_format_count)):
                        coordinates = _convert_coordinates(sections[place:place + data_format_count])
                        line_data[index] = coordinates
    
                    trc_data[frame] = (time, line_data)
                    append_per_label_data(trc_data, markers, line_data)
                else:
                    raise IOError(f'檔案格式無效：資料幀 {len_section} 與資料格式不符。')
    motion_info = {}
    for i in range(len(file_header_keys)):
        motion_info[file_header_keys[i]] = file_header_data[i]
    return motion_info, markers, trc_data

def transform_up_Z2Y(raw_data, dtype="DataFrame"):
    """
    transform the axis of motion data from Z-up to Y-up.

    Parameters
    ----------
    raw_data : pandas.DataFrame
        Input raw motion data.
    dtype : Str, optional
    The default is 'DataFrame'.

    Returns
    -------
    data : pandas.DataFrame
        output the motion data has been transformed.

    """
    if dtype == "DataFrame":
        data = pd.DataFrame(np.zeros([np.shape(raw_data)[0], np.shape(raw_data)[1]]))
        for i in range(int((np.shape(raw_data.iloc[:, :])[1] - 2) / 3)):
            data.iloc[:, 2 + 3 * i] = raw_data.iloc[:, 2 + 3 * i]  # x -> x
            data.iloc[:, 3 + 3 * i] = raw_data.iloc[:, 4 + 3 * i]  # y -> z
            data.iloc[:, 4 + 3 * i] = -raw_data.iloc[:, 3 + 3 * i]  # z -> -y
    elif dtype == "np.array":
        print("rotation np.array")
        # raw_data = new_trun_motion
        data = np.empty(shape=(np.shape(raw_data)), dtype=float)
        for i in range(int((np.shape(raw_data)[0]))):
            data[i, :, 0] = raw_data[i, :, 0]  # x -> x
            data[i, :, 1] = raw_data[i, :, 2]  # y -> z
            data[i, :, 2] = -raw_data[i, :, 1]  # z -> -y
    elif dtype == "list":
        data = []
        for i in range(len(raw_data)):
            data.append([raw_data[i][0], raw_data[i][2], -raw_data[i][1]])
    else:
        print("rotation fail")
    return data

def trc_write(save_path, file_name, motion_info, filtered_data_dict):
    markers = list(filtered_data_dict.keys())
    
    data = dict()
    data["Timestamps"] = np.arange(0,
                                   len(filtered_data_dict[markers[0]]) * 1 / float(motion_info["CameraRate"]),
                                   1 / float(motion_info["CameraRate"]),
                                    )
    with open(str(save_path), "w") as file:
            # Write header
            file.write(str("PathFileType\t4\t(X/Y/Z)\t" + file_name + "\n"))
            file.write(
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
            )
            file.write(
                "%d\t%d\t%d\t%d\t%s\t%d\t%d\t%d\n"
                % (
                    float(motion_info["DataRate"]),
                    float(motion_info["CameraRate"]),
                    int(motion_info["NumFrames"]),
                    len(data),
                    motion_info["Units"],
                    float(motion_info["OrigDataRate"]),
                    int(motion_info["OrigDataStartFrame"]),
                    int(motion_info["OrigNumFrames"]),
                )
            )
            # Write labels
            file.write("Frame#\tTime\t")
            for i, label in enumerate(markers):
                # print(i, label)
                if i == 0:
                    file.write("%s" % (label))
                else:
                    file.write("\t")
                    file.write("\t\t%s" % (label))
            file.write("\n")
            file.write("\t")
            for i in range((len(markers) * 3)):
                # print((chr(ord('X')+(i%3)), math.floor((i+3)/3)))
                file.write("\t%c%d" % (chr(ord("X") + (i % 3)), math.floor((i + 3) / 3)))
            file.write("\n")
            file.write("\n")
            # Write data
            for i in range(len(filtered_data_dict[markers[0]])):
                print(i)
                # print(i, data["Timestamps"][i])
                file.write("%d\t%f" % ((i + 1), data["Timestamps"][i]))
                for l in range(len(markers)):
                    file.write("\t%f\t%f\t%f" % tuple(filtered_data_dict[markers[l]][i]))
                    # marker * frame * (x, y, z)
                    # file.write("\t%f\t%f\t%f" % tuple(motion_data[l, i, :]))
                file.write("\n")


# %% code 正式開始

motion_info, markers, trc_data = readTRC(r"E:\Hsin\NTSU_lab\Kao\S2_2.trc")
trans_data = {}

for marker in markers:
    trans_data[marker] = transform_up_Z2Y(trc_data[marker], dtype="list")
filtered_data_dict = {key: value for key, value in trans_data.items() if len(value) > 0}

trc_write(r"E:\Hsin\NTSU_lab\Kao\test.trc",
          "test",
          motion_info, filtered_data_dict)

        




















