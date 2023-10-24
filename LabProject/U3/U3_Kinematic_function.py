# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:10:51 2023

@author: Hsin.YH.Yang
"""
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


# %% Transformation between Coordinate System
"""
1. define Local Coordinate System.
    1.1. define LCS orgin. 
    1.2. calculate Local Coordinate.
Transformation matrix :
    1. Linear transformation.
    2. Rotational Transformation.
"""


def transformation_matrix(LCS_0, LCS_1, LCS_2, p, O, rotation="GCStoLCS"):
    """
    Parameters
    ----------
    LCS_0 : np.array
        The orgin of LCS.
    LCS_1 : np.array
        To create long axis with respect to orgin point.
    LCS_2 : np.array
        To create plane axis with respect to orgin point.
    p : np.array
        the specific point coordinate with respect to GCS/LCS.
    rotation : Str, optional
        To determinate to rotation sequence of transform matrix. The default is 'GCStoLCS'.

    Returns
    -------
    p1 : np.array
        the specific point coordinate with respect to LCS/GCS.

    """
    # determinate the axis
    v1 = LCS_1 - LCS_0  # long axis
    v2 = np.cross(v1, (LCS_2 - LCS_0))  # superior direction
    v3 = np.cross(v1, v2)  # lateral direction

    # normalize the vector
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    # calculate rotation matrix
    rotation_LCS = np.array([v1, v2, v3])  # or np.vstack((v1, v2, v3)).T
    # print("\nRotation matrix rotation_LCS:\n", rotation_LCS)

    if rotation == "GCStoLCS":
        p1 = np.matmul(rotation_LCS, (p - LCS_0))
    elif rotation == "LCStoGCS":
        p1 = np.matmul((np.transpose(rotation_LCS)), p) + LCS_0

    return p1


# %% transform Z axis up to Y axis up


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
    else:
        print("rotation fail")
    return data


# %% 將資料 .c3d 轉成 .trc


def c3d_to_trc(save_path, file_name, motion_data, motion_info):
    """
    referce :
        1. https://github.com/IISCI/c3d_2_trc/blob/master/extractMarkers.py

    convert c3d formate to trc formate.
    1. input motion data and information from c3d file.
    2. writting data to trc file
    3. reload trc data to bkt tool to output trc again
        (due to first version of trc can not read by opensim)

    Parameters
    ----------
    save_path : str
        define the save path of data.
    file_name : str
        Defines the filename to which data needs to be written.
    motion_data : pandas.DataFrame
        the motion data from c3d file.
    motion_info : dict
        the motion information from c3d file.

    Returns
    -------
    None.

    """
    # # 配置日志记录
    # logging.basicConfig(filename = save_path + 'error_log.txt',
    #                     level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. 取得 motion data information
    # motion_data = new_np_motion_data
    # motion_axis = ['X', 'Y', 'Z']
    # motion_markers_axis = []
    # for num in range(len(new_labels)): # +1 因為新增R.Elbow.Med
    #     for axis in motion_axis:
    #         motion_markers_axis.append(axis + str(int(num + 1)))
    # # # 重新更新時間軸
    data = dict()
    data["Timestamps"] = np.arange(
        0,
        np.shape(motion_data)[1] * 1 / motion_info["frame_rate"],
        1 / motion_info["frame_rate"],
    )

    # 旋轉矩陣 transform Z axis up to Y axis up
    # trans_np_motion_data = transform_up_Z2Y(motion_data, dtype="np.array")
    # 2. 將資料寫進 .trc file
    # -----------------numpy做輸入參數-----------------------------
    # save_path = r"E:\Motion Analysis\U3 Research\error_log.trc"
    # file_name = r"E:\Motion Analysis\U3 Research\error_log.trc"
    with open(str(save_path), "w") as file:
        # Write header
        file.write(str("PathFileType\t4\t(X/Y/Z)\t" + file_name + "\n"))
        file.write(
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
        )
        file.write(
            "%d\t%d\t%d\t%d\tmm\t%d\t%d\t%d\n"
            % (
                motion_info["frame_rate"],
                motion_info["frame_rate"],
                np.shape(motion_data)[1],
                len(motion_info["LABELS"]),
                motion_info["frame_rate"],
                1,
                np.shape(motion_data)[1],
            )
        )
        # Write labels
        file.write("Frame#\tTime\t")
        for i, label in enumerate(motion_info["LABELS"]):
            # print(i, label)
            if i == 0:
                file.write("%s" % (label))
            else:
                file.write("\t")
                file.write("\t\t%s" % (label))
        file.write("\n")
        file.write("\t")
        for i in range((len(motion_info["LABELS"]) * 3)):
            # print((chr(ord('X')+(i%3)), math.floor((i+3)/3)))
            file.write("\t%c%d" % (chr(ord("X") + (i % 3)), math.floor((i + 3) / 3)))
        file.write("\n")
        file.write("\n")
        # Write data
        for i in range(np.shape(motion_data)[1]):
            # print(i, data["Timestamps"][i])
            file.write("%d\t%f" % ((i + 1), data["Timestamps"][i]))
            for l in range(np.shape(motion_data)[0]):
                file.write("\t%f\t%f\t%f" % tuple(motion_data[l, i, :]))
            file.write("\n")

    print("数據已成功寫入文本文件：", save_path)
    # 3. 使用 btk package 協助轉檔
    # try:
    #     reader = btk.btkAcquisitionFileReader()
    #     reader.SetFilename(str(save_path))
    #     reader.Update()
    #     acq = reader.GetOutput()
    #     writer = btk.btkAcquisitionFileWriter()
    #     clone = acq.Clone()
    #     writer.SetInput(clone)
    #     writer.SetFilename(str(save_path))
    #     writer.Update()
    #     print("数據已成功使用 btk 轉檔：", save_path)
    # except RuntimeError:
    #     print("An unexpected error occurred: %s", save_path)


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


# %%
if __name__ == "__main__":
    print(__name__)

# %% 廢棄功能區

"""
廢棄功能 : 
    1. 抓取不同種類的 Fitts law data，並計算所花費時間
"""
# Fitts law
# load merge file table of Fitts law
# Fitts_table = pd.read_csv(r"E:\BenQ_Project\U3\08_Fitts\S01\S01.csv")
# # 計算有多少字串在列表中
# def string_count(lst):
#     string_count_list = {}
#     for item in lst:
#         if item in string_count_list:
#             string_count_list[item] += 1
#         else:
#             string_count_list[item] = 1
#     return string_count_list

# # 計算每個 Fitts law 測試所用的時間
# condition_list = string_count(Fitts_table.loc[:,'Condition'])
# # 給定兩個條件，四種 condition 選一、三次測試中選一
# for condi_key_1 in condition_list.keys():
#     print(condi_key_1)
#     condition_ind = Fitts_table.index[Fitts_table.loc[:,'Condition'] == condi_key_1]
#     block_key = string_count(Fitts_table.loc[condition_ind, 'Block'])
#     for condi_key_2 in block_key.keys():
#         total_time = 0
#         for index in condition_ind:
#             if Fitts_table.loc[index, 'Block'] == condi_key_2:
#                 total_time += Fitts_table.loc[index, 'MT(ms)']
#         # 因為共有 15 個區間，所以 MT(ms) * 15
#         # 四個block，區間中各給兩秒 = 6s
#         # trigger 按下之後給兩秒 = 2s
#         print(total_time*15)
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
"""
前處理資料
0. 使用 Tpose 計算手肘內上髁位置 :
    0.1. 使用上肢參考架 UA1, UA2, UA3 計算 LCS 座標系.
    0.2. 為避免補點所造成的資料長短不一，因此取最短數列做標準.
    0.3. 計算所有 Tpose 中手肘內上髁之位置，再計算平均.

"""
# 1. -------------read Tpose c3d file and calculate ---------------------------
# read c3d
# motion_info, motion_data, analog_info, FP_data = func.read_c3d(r"E:\Motion Analysis\U3 Research\S01\S01_Tpose_1.c3d")

# R_Elbow_Med = motion_data.loc[:, "EC2 Wight_Elbow:R.Elbow.Med_x":"EC2 Wight_Elbow:R.Elbow.Med_z"].dropna(axis=0)
# R_Elbow_Lat = motion_data.loc[:, "EC2 Wight_Elbow:R.Elbow.Lat_x":"EC2 Wight_Elbow:R.Elbow.Lat_z"].dropna(axis=0)
# UA1 = motion_data.loc[:, "EC2 Wight_Elbow:UA1_x":"EC2 Wight_Elbow:UA1_z"].dropna(axis=0)
# UA2 = motion_data.loc[:, "EC2 Wight_Elbow:UA2_x":"EC2 Wight_Elbow:UA2_z"].dropna(axis=0)
# UA3 = motion_data.loc[:, "EC2 Wight_Elbow:UA3_x":"EC2 Wight_Elbow:UA3_z"].dropna(axis=0)

# # 避免數量中出現NAN，請造成不同變數見長短不一致，因此找出最短數列的 index
# ind_frame = min([np.shape(R_Elbow_Med)[0], np.shape(UA1)[0], np.shape(UA2)[0], np.shape(UA3)[0]])
# for i in [R_Elbow_Med, UA1, UA2, UA3]:
#     if ind_frame == np.shape(i)[0]:
#         ind_frame = i.index
#         break
# # 計算手肘內上髁在 LCS 之位置
# p1_all = pd.DataFrame(np.zeros([len(ind_frame), 3]))
# for frame in ind_frame:
#     p1_all.iloc[frame :] = (func.transformation_matrix(R_Elbow_Lat.iloc[frame, :].values, UA1.iloc[frame, :].values, UA3.iloc[frame, :].values,
#                                                        R_Elbow_Med.iloc[frame, :].values, np.array([0, 0, 0]),
#                                                        rotation='GCStoLCS'))
# # 手肘內上髁在LCS之位置

# # 回算手肘內上髁在 GCS 之位置
# V_R_Elbow_Med = pd.DataFrame(np.zeros([len(ind_frame), 3]))
# for frame in range(np.shape(V_R_Elbow_Med)[0]):
#     V_R_Elbow_Med.iloc[frame, :] = (func.transformation_matrix(R_Elbow_Lat.iloc[frame, :].values, UA1.iloc[frame, :].values, UA3.iloc[frame, :].values,
#                                                 p1_all.iloc[5, :], np.array([0, 0, 0]), # p1_all 取相對穩定的一筆
#                                                 rotation='LCStoGCS'))
# # 驗證差異用
# # diff = R_Elbow_Med.values - R_Elbow_Lat.iloc[:334, :].values
# # from scipy.spatial import distance
# # dist_all = pd.DataFrame(np.zeros([len(ind_frame), 1]))
# # for i in range(np.shape(dist_all)[0]):
# #     dist_all.iloc[i, :] = distance.euclidean(R_Elbow_Med.iloc[i, :].values, R_Elbow_Lat.iloc[i, :].values)

# # 清除不需要的變數
# del motion_info, motion_data, analog_info, FP_data, R_Elbow_Med, UA1, UA2, UA3
# gc.collect()
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
"""
"""
# 2. 將資料寫進 .trc file
# 定義要寫進 .trc 的資料
# 設定 .trc file 的標頭檔，後續資料從第六行開始寫入
# trc_info = {'1': ["PathFileType", 4, "(X/Y/Z)", file_name],
#             '2': ["DataRate", "CameraRate", "NumFrames", "NumMarkers", "Units", "OrigDataRate", "OrigDataStartFrame", "OrigNumFrames"],
#             '3': [motion_info["frame_rate"], motion_info["frame_rate"], np.shape(motion_data)[0], len(new_labels),
#                   motion_info["UNITS"][0], motion_info["frame_rate"], 1, np.shape(motion_data)[0]],
#             '4': ["Frame#", "Time", new_labels],
#             '5': motion_markers_axis,
#             '6': new_motion_data.values}
# 開啟 .txt 令存檔名為 .trc


# for key_name, value in trc_info.items():
#     # 寫入 Marker name
#     if '4' in key_name:
#         for nested_list in trc_info[key_name]:
#             if "Frame#" in nested_list or "Time" in nested_list:
#                 file.write(str(nested_list) + '\t')
#             else:
#                 for marker_name in nested_list:
#                     file.write(str(marker_name) + '\t' + '\t' + '\t')
#     # 寫入 Marker axis
#     elif '5' in key_name:
#         file.write('\t' + '\t')
#         for nested_list in trc_info[key_name]:
#             file.write(str(nested_list) + '\t')
#     # 寫入 motion data
#     elif '6' in key_name:
#         file.write('\n')
#         for row in value:
#             for number in range(len(row)):
#                 if number == 0:
#                     file.write(str(int(row[number])) + '\t')
#                 else:
#                     file.write(str(row[number]) + '\t')
#             file.write('\n')
#     # 寫入標頭檔資訊
#     else:

#             # print(nested_list)
#             file.write(str(nested_list) + '\t')
#     file.write('\n')
"""
因為pd.DataFrame寫入資料時間太長
"""
# file_name = "123"
# with open(str(r"E:\Motion Analysis\U3 Research\error_log.trc"), 'w') as file:
#     # Write header
#     file.write(str("PathFileType\t4\t(X/Y/Z)\t" + file_name +"\n"))
#     file.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
#     file.write("%d\t%d\t%d\t%d\tmm\t%d\t%d\t%d\n" \
#                % (motion_info["frame_rate"], motion_info["frame_rate"], np.shape(motion_data)[0],
#                    len(new_labels), motion_info["frame_rate"],
#                    1, np.shape(motion_data)[1]))
#     # Write labels
#     file.write("Frame#\tTime\t")
#     for i, label in enumerate(new_labels):
#         if i != 0:
#             file.write("\t")
#         file.write("\t\t%s" % (label))
#     file.write("\n")
#     file.write("\t")
#     for i in range(len(new_labels*3)):
#         file.write("\t%c%d" % (chr(ord('X')+(i%3)), math.ceil((i+3)/3)))
#     file.write("\n")
#     # Write data
#     # data = dict()
#     # data["Timestamps"] = np.arange(0, np.shape(motion_data)[1]*1/motion_info["frame_rate"], 1/motion_info["frame_rate"])
#     for i in range(np.shape(new_motion_data)[0]):
#         file.write("%d\t%f" % (i, new_motion_data.iloc[i, 0]))
#         print(i, new_motion_data.iloc[i, 0])
#         for l in range(int((np.shape(new_motion_data)[1]-1)/3)):
#             file.write("\t%f\t%f\t%f" % tuple(new_motion_data.iloc[i, [3*l+1, 3*l+2, 3*l+3]].values))
#             # print(new_motion_data.iloc[i, [3*l+1, 3*l+2, 3*l+3]].values)
#         file.write("\n")
