# %% import package
import ezc3d
import os
import numpy as np
import pandas as pd

# import btk
import math
import logging
import csv


# %% read c3d
def read_c3d(path):
    """
    Parameters
    ----------
    path : str
        key in c3d data path.
    Returns
    -------
    motion_info : dict
        Contains: frame rate, first frame, last frame, size(number of infrared markers).
    motion_data : DataFrame
        data strcuture like .trc file.
    analog_info : dict
        Contains: frame rate, first frame, last frame, size(number of analog channel).
    analog_data : DataFrame
        data structure like .anc file.
    np_motion_data : np.array
        numpy.array with N marker x M frmae x 3.
    -------

    example:
        motion_info, motion_data, analog_info, analog_data, np_motion_data = read_c3d(Your_Path)

    Author: Hsin Yang. 2023.01.20
    """
    # 1. read c3d file
    # path = r"E:\Motion Analysis\U3 Research\S01\S01_1VS1_1.c3d"
    c = ezc3d.c3d(path)

    # 數據的基本資訊，使用dict儲存
    # 1.1 information of motion data
    motion_info = c["header"]["points"]
    # add Unit in motion information
    motion_info.update(
        {
            "UNITS": c["parameters"]["POINT"]["UNITS"]["value"],
            "LABELS": c["parameters"]["POINT"]["LABELS"]["value"],
        }
    )
    # 1.2 information of analog data
    analog_info = c["header"]["analogs"]
    # 2. convert c3d motion data to DataFrame format
    ## 2.1 create column's name of motion data
    motion_axis = ["x", "y", "z"]
    motion_markers = []
    for marker_name in c["parameters"]["POINT"]["LABELS"]["value"]:
        for axis in motion_axis:
            name = marker_name + "_" + axis
            motion_markers.append(name)
    # 2.2 create x, y, z matrix to store motion data
    motion_data = pd.DataFrame(
        np.zeros(
            [
                c["header"]["points"]["last_frame"] + 1,  # last frame + 1
                len(c["parameters"]["POINT"]["LABELS"]["value"]) * 3,
            ]
        ),  # marker * 3
        columns=motion_markers,
    )
    # 使用numpy.array來貯存資料
    np_motion_data = np.empty(
        shape=(
            len(c["parameters"]["POINT"]["LABELS"]["value"]),
            np.shape(c["data"]["points"])[-1],
            3,
        ),
        dtype=float,
    )

    for i in range(len(c["parameters"]["POINT"]["LABELS"]["value"])):
        np_motion_data[i, :, :] = np.transpose(c["data"]["points"][:3, i, :])
    # 2.3 key in data into matrix
    for i in range(len(c["parameters"]["POINT"]["LABELS"]["value"])):
        # print(1*i*3, 1*i*3+3)
        # transpose matrix to key in data
        motion_data.iloc[:, 1 * i * 3 : 1 * i * 3 + 3] = np.transpose(
            c["data"]["points"][:3, i, :]
        )
    # 2.4 insert time frame
    ## 2.4.1 create time frame
    motion_time = np.linspace(
        0,  # start
        (
            (c["header"]["points"]["last_frame"]) / c["header"]["points"]["frame_rate"]
        ),  # stop = last_frame/frame_rate
        num=(np.shape(c["data"]["points"])[-1]),  # num = last_frame
    )
    # ### 2.4.2 insert time frame to motion data
    motion_data.insert(0, "Frame", motion_time)
    # 3. convert c3d analog data to DataFrame format
    #    force plate data (analog = force plate)
    ## 3.1 create force plate channel name
    analog_channel = c["parameters"]["ANALOG"]["LABELS"]["value"]
    ## 3.2 create a matrix to store force plate data
    analog_data = pd.DataFrame(
        np.zeros(
            [np.shape(c["data"]["analogs"])[-1], len(analog_channel)]  # last frame + 1
        ),
        columns=analog_channel,
    )
    analog_data.iloc[:, :] = np.transpose(c["data"]["analogs"][0, :, :])
    ## 3.3 insert time frame
    ### 3.3.1 create time frame
    analog_time = np.linspace(
        0,  # start
        (
            (c["header"]["analogs"]["last_frame"])
            / c["header"]["analogs"]["frame_rate"]
        ),  # stop = last_frame/frame_rate
        num=(np.shape(c["data"]["analogs"])[-1]),  # num = last_frame
    )
    analog_data.insert(0, "Frame", analog_time)
    # synchronize data (optional)
    return motion_info, motion_data, analog_info, analog_data, np_motion_data


# %% Reading all of data path
# using a recursive loop to traverse each folder
# and find the file extension has .csv
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

# %% 開啟 .trc 獲得 marker name


def open_trc(data_path):
    show_msg = True
    with open(file=data_path, mode="rt", encoding="utf-8", newline="") as f:
        if show_msg:
            print('Opening file "{}" ... '.format(data_path), end="")
            # get header information
        read = csv.reader(f, delimiter="\t")
        header = [next(read) for x in range(5)]
        # actual number of markers
        nmarkers = int((len(header[3]) - 2) / 3 + 1)
        # column labels
        markers = np.asarray(header[3])[np.arange(2, 2 + 3 * nmarkers, 3)].tolist()
        # find marker duplicates if they exist and add suffix to the duplicate markers:
        if len(markers) != len(set(markers)):
            from collections import Counter

            d = {
                a: [""] + list(range(2, b + 1)) if b > 1 else ""
                for a, b in Counter(markers).items()
            }
            markers = [i + str(d[i].pop(0)) if len(d[i]) else i for i in markers]
        markers3 = [m for m in markers for i in range(3)]
        # XYZ order
        # XYZ = [i[0] for i in header[4][2:5]]
        xyz = [i[0].lower() for i in header[4][2:5]]
        # XYZ = header[0][2].strip('()').split('/')
        # xyz = header[0][2].strip('()').lower().split('/')
        markersxyz = [a + b for a, b in zip(markers3, xyz * nmarkers)]
        columns_name = ["Frame", "Time"] + markersxyz

    return columns_name