# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:18:26 2024


• E1: 當L.Wrist.Rad Z軸高度超過T10 Z軸高度 擷取此段資料。
• E5: 擷取直到弓身低於T10 Z軸高度，停止擷取。
• 資料分期點:
• E2 舉弓頂點時間:根據全段資料，以L.Wrist.Rad Z軸判定，回傳位置峰值
數值與對應時間點，即時運算角度後取角度峰直數值與對應時間點
• E3 當L.Wrist.Rad Z軸高度等於L. Acromion 進行標記
• E4 放箭時間:根據資料末端2000點判定，即時運算移動平均, R. Elbow Lat
X軸超出前1秒數據3個標準差，判定為放箭
    
@author: Hsin.Yang 05.May.2024
"""
# %%
import ezc3d
import numpy as np
import pandas as pd

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
