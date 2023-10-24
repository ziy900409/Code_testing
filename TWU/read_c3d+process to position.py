# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.utils import resample
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.signal import freqs
from scipy import interpolate
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

"""
Created on Thu Jan 19 22:44:15 2023
error: ValueError: could not broadcast input array from shape (4132,12) into shape (4144,12)
bug 原因: header資訊與data資訊不重合的問題
@author: Hsin Yang. 2023.01.20
"""


def read_c3d(path):
    """
    Parameters
    ----------
    path : str
        kep in c3d data path.
    Returns
    -------
    motion_information : dict
        Contains: frame rate, first frame, last frame, size(number of infrared markers).
    motion_data : DataFrame
        data strcuture like .trc file.
    analog_information : dict
        Contains: frame rate, first frame, last frame, size(number of analog channel).
    FP_data : DataFrame
        data structure like .anc file.

    example:
        motion_information, motion_data, analog_information, FP_data = read_c3d(Your_Path)

    Author: Hsin Yang. 2023.01.20
    """
    # import library
    import ezc3d
    import numpy as np
    import pandas as pd

    # 1. read c3d file
    c = ezc3d.c3d(path)
    # 數據的基本資訊，使用dict儲存
    # 1.1 information of motion data
    motion_information = c["header"]["points"]
    # 1.2 information of analog data
    analog_information = c["header"]["analogs"]
    # 2. convert c3d motion data to DataFrame format
    ## 2.1 create column's name of motion data
    motion_axis = ["x", "y", "z"]
    motion_markers = []
    for marker_name in c["parameters"]["POINT"]["LABELS"]["value"]:
        for axis in motion_axis:
            name = marker_name + "_" + axis
            motion_markers.append(name)
    ## 2.2 create x, y, z matrix to store motion data
    # motion_data = pd.DataFrame(np.zeros([c['header']['points']['last_frame']+1, # last frame + 1
    #                                      len(c['parameters']['POINT']['LABELS']['value'])*3]), # marker * 3
    #                            columns=motion_markers)
    motion_data = pd.DataFrame(
        np.zeros(
            [
                np.shape(c["data"]["points"])[-1],  # last frame + 1
                len(c["parameters"]["POINT"]["LABELS"]["value"]) * 3,
            ]
        ),  # marker * 3
        columns=motion_markers,
    )
    ## 2.3 key in data into matrix
    for i in range(len(c["parameters"]["POINT"]["LABELS"]["value"])):
        # print(1*i*3, 1*i*3+3)
        # transpose matrix to key in data
        motion_data.iloc[:, 1 * i * 3 : 1 * i * 3 + 3] = np.transpose(
            c["data"]["points"][:3, i, :]
        )
    ## 2.4 insert time frame
    ### 2.4.1 create time frame
    motion_time = np.linspace(
        0,  # start
        (
            (c["header"]["points"]["last_frame"]) / c["header"]["points"]["frame_rate"]
        ),  # stop = last_frame/frame_rate
        num=(np.shape(c["data"]["points"])[-1]),  # num = last_frame
    )
    ### 2.4.2 insert time frame to motion data
    motion_data.insert(0, "Frame", motion_time)
    # 3. convert c3d analog data to DataFrame format
    #    force plate data (FP = force plate)
    ## 3.1 create force plate channel name
    FP_channel = c["parameters"]["ANALOG"]["LABELS"]["value"]
    ## 3.2 create a matrix to store force plate data
    FP_data = pd.DataFrame(
        np.zeros(
            [np.shape(c["data"]["analogs"])[-1], len(FP_channel)]  # last frame + 1
        ),
        columns=FP_channel,
    )
    FP_data.iloc[:, :] = np.transpose(c["data"]["analogs"][0, :, :])
    ## 3.3 insert time frame
    ### 3.3.1 create time frame
    FP_time = np.linspace(
        0,  # start
        (
            (c["header"]["analogs"]["last_frame"])
            / c["header"]["analogs"]["frame_rate"]
        ),  # stop = last_frame/frame_rate
        num=(np.shape(c["data"]["analogs"])[-1]),  # num = last_frame
    )
    FP_data.insert(0, "Frame", FP_time)
    # synchronize data (optional)
    return motion_information, motion_data, analog_information, FP_data


def unit_vector(point1, point2):
    length = np.array(point1 - point2)
    a = np.zeros([len(point1[:, :]), 3])
    for i in length:
        a = a + i**2
    unitvector = length / np.sqrt(a)
    return unitvector


# make transformation matrix
def TX(X, Y, Z):
    T = np.zeros([len(X[:, :]), 3, 3])
    for i in range(len(X[:, :])):
        aaa = np.vstack((X[0 + i, 0:3], Y[0 + i, 0:3], Z[0 + i, 0:3]))
        T[i, :, :] = aaa
    return T


# %%
# motion_information, motion_data, analog_information, FP_data = read_c3d(r'C:\Users\19402\Documents\Kwon3D Projects\disc_golf\Trials\Trials\disc_golf\2023-02-17\golf_Tpose.c3d')
# motion_information1, motion_data1, analog_information1, FP_data1 = read_c3d(r'C:\Users\19402\Documents\Kwon3D Projects\disc_golf\Trials\Trials\disc_golf\2023-02-17\Ball.c3d')
motion_information_s, motion_data_s, analog_information, FP_data = read_c3d(
    "E:\git\Code_testing\TWU\data\origin static markers.c3d"
)

motion_data_arrays = motion_data_s.to_numpy(dtype="float64")
ro = motion_data_arrays[:, 1:4]
rx = motion_data_arrays[:, 4:7]
ry = motion_data_arrays[:, 7:10]

motion_information_d, motion_data_d, analog_informations, FP_datas = read_c3d(
    "E:\git\Code_testing\TWU\data\dynamic markers.c3d"
)
motion_data_arrayd = motion_data_d.to_numpy(dtype="float64")
rO = motion_data_arrayd[:, 7:10]
rX = motion_data_arrayd[:, 10:13]
rY = motion_data_arrayd[:, 1:4]
rXY = motion_data_arrayd[:, 4:7]

s_X = unit_vector(rx, ro)
s_Y = unit_vector(ry, ro)
s_Z = np.cross(s_X, s_Y)
s_T = TX(s_X, s_Y, s_Z)
average_s_T = np.mean(s_T, axis=0)  # 计算平均值
num_frames = motion_data_arrayd.shape[0]
static_tm = np.tile(average_s_T, (num_frames, 1, 1))

rs = R.from_matrix(static_tm)
euler_angless = rs.as_euler("ZYX", degrees=True)
plt.plot(euler_angless[1:, 0])
plt.plot(euler_angless[1:, 1])
plt.plot(euler_angless[1:, 2])

d_X = unit_vector(rX, rO)
# 我原本用d_Y = unit_vector*(rY, rO) -> 但全部變成nan
d_F = unit_vector(rXY, rO)
d_Z = np.cross(d_X, d_F)
d_Y = np.cross(d_Z, d_X)
dynamic_tm = TX(d_X, d_Y, d_Z)

# 這邊轉出來超怪...
rd = R.from_matrix(dynamic_tm)
euler_anglesd = rd.as_euler("ZYX", degrees=True)
plt.plot(euler_anglesd[1:, 0])
plt.plot(euler_anglesd[1:, 1])
plt.plot(euler_anglesd[1:, 2])

marker_d_inv = inv_t(dynamic_T)
marker_tm_s_to_d = np.empty(tm_d.shape)

for i in range(tm_d.shape[0]):
    marker_tm_s_to_d[i, :, :] = np.dot(static_tm[i, :, :], marker_d_inv[i, :, :])

angle_stod = R.from_matrix(marker_tm_s_to_d)
euler_angles123 = angle_stod.as_euler("ZYX", degrees=True)
# 将旋转矩阵转换为ZYX顺序的欧拉角
plt.plot(euler_angles123[:, 0])
plt.plot(euler_angles123[:, 1])
plt.plot(euler_angles123[:, 2])

# change acc_d to acc_glo
glo_acc = np.empty(accelerometer_d.shape)  # 创建一个空数组以存储结果
for i in range(accelerometer_d.shape[0]):
    glo_acc[i, :] = np.dot(accelerometer_d[i, :], marker_tm_s_to_d[i, :, :])

plt.plot(glo_acc[:, 0])
plt.plot(glo_acc[:, 1])
plt.plot(glo_acc[:, 2])

# low pass filtering
lowcut = 6  # 低通滤波的截止频率
order = 4  # 滤波器阶数
nyq = 0.5 * 100.0  # 假设采样率为100 Hz
low = lowcut / nyq
b, a = butter(order, low, btype="low")
global_accelerometer = lfilter(b, a, glo_acc, axis=0)

plt.plot(global_accelerometer[1:, 0])
plt.plot(global_accelerometer[1:, 1])
plt.plot(global_accelerometer[1:, 2])
plt.title("global acc")

# method 1
# acc inegration to velocity
time_diff = np.diff(timestamp_d)
velocity = np.cumsum(glo_acc[:-1] * time_diff[:, np.newaxis], axis=0)

plt.plot(velocity[1:, 0])
plt.plot(velocity[1:, 1])
plt.plot(velocity[1:, 2])
plt.title("global velocity")

# # 高通滤波
# cutoff_frequency = 0.1
# order = 1
# sampling_frequency = 100.0
# nyquist_frequency = 0.5 * sampling_frequency
# high = cutoff_frequency / nyquist_frequency
# b, a = butter(order, high, btype='high')
# filtered_velocity_data = filtfilt(b, a, velocity, axis=0)

# #from Oscillatory-Motion-Tracking-With-x-IMU method of high pass for velocity
# order = 1
# filtCutOff = 0.1  # 截止频率，单位为 Hz
# sampleFreq = 100  # 采样频率，单位为 Hz
# normalized_cutoff = filtCutOff / (0.5 * sampleFreq)
# b, a = butter(order, normalized_cutoff, btype='high')
# linVelHP = np.zeros_like(velocity)  # 假设linVel是您的数据
# for i in range(velocity.shape[1]):  # 遍历数据的列
#     linVelHP[:, i] = filtfilt(b, a, velocity[:, i])

highpass_velocity = np.zeros_like(velocity)  # 假设linVel是您的数据
for i in range(velocity.shape[1]):  # 遍历数据的列
    highpass_velocity[:, i] = filtfilt(b, a, velocity[:, i])

# 速度积分以计算位置
time_diff = np.diff(timestamp_d)
position = np.cumsum(highpass_velocity[:] * time_diff[:, np.newaxis], axis=0)

# #from Oscillatory-Motion-Tracking-With-x-IMU method of high pass for position
# order = 1
# filtCutOff = 0.1  # 截止频率，单位为 Hz
# sampleFreq = 100  # 采样频率，单位为 Hz
# normalized_cutoff = filtCutOff / (0.5 * sampleFreq)
# b, a = butter(order, normalized_cutoff, btype='high')
# highpass_position = np.zeros_like(position)  # 假设linVel是您的数据
# for i in range(position.shape[1]):  # 遍历数据的列
#     highpass_position[:, i] = filtfilt(b, a, position[:, i])

# # 高通滤波
# highcut = 0.1
# order = 1
# nyq = 0.5 * 100.0  # 采样频率为100Hz
# high = highcut / nyq
# b, a = butter(order, high, btype='high')
# filtered_position = filtfilt(b, a, position, axis=0)

plt.plot(highpass_position[1:, 0])
plt.plot(highpass_position[1:, 1])
plt.plot(highpass_position[1:, 2])
plt.title("global position")

fig = plt.figure()
Axes3D = fig.add_subplot(111, projection="3d")

Axes3D.plot(highpass_position[:, 0], highpass_position[:, 1], highpass_position[:, 2])
Axes3D.set_xlabel("X Position")
Axes3D.set_ylabel("Y Position")
Axes3D.set_zlabel("Z Position")

# %%
