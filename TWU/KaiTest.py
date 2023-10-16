# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:09:26 2023

@author: Hsin Yang, 15.Oct.2023
"""

import numpy as np
from scipy.spatial.transform import Rotation
import imufusion
import matplotlib.pyplot as pyplot
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from mpl_toolkits.mplot3d import Axes3D


# %% define for math
def quatern2rotMat(q):
    """
    Converts a quaternion orientation to a rotation matrix.

    Parameters:
    q (numpy.array): Input quaternion in the form [w, x, y, z].

    Returns:
    R (numpy.array): Rotation matrix.
    """
    R = np.zeros((len(q), 3, 3))
    R[:, 0, 0] = 2 * (q[:, 0] ** 2) - 1 + 2 * (q[:, 1] ** 2)
    R[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    R[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    R[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    R[:, 1, 1] = 2 * (q[:, 0] ** 2 - 1 + 2 * (q[:, 2] ** 2))
    R[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    R[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
    R[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    R[:, 2, 2] = 2 * (q[:, 0] ** 2 - 1 + 2 * (q[:, 3] ** 2))

    return R


def double_integration(global_acc, frequency):
    num_samples = len(global_acc)
    time_intervals = 1 / frequency
    velocity_data = np.zeros((num_samples, 3))
    position_data = np.zeros((num_samples, 3))

    for i in range(1, num_samples):
        delta_t = time_intervals
        velocity_data[i, :] = velocity_data[i - 1, :] + global_acc[i, :] * delta_t
        position_data[i, :] = position_data[i - 1, :] + velocity_data[i, :] * delta_t

    return position_data


def quaternProd(a, b):
    """
    Calculates the quaternion product of two quaternions a and b.

    Parameters:
    a (numpy.array): First quaternion in the form [w, x, y, z].
    b (numpy.array): Second quaternion in the form [w, x, y, z].

    Returns:
    ab (numpy.array): Quaternion product of a and b.
    """
    ab = np.zeros_like(a)
    ab[:, 0] = (
        a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3]
    )
    ab[:, 1] = (
        a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2]
    )
    ab[:, 2] = (
        a[:, 0] * b[:, 2] - a[:, 1] * b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1]
    )
    ab[:, 3] = (
        a[:, 0] * b[:, 3] + a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0]
    )

    return ab


def quaternConj(q):
    """
    Converts a quaternion to its conjugate.

    Parameters:
    q (numpy.array): Input quaternion in the form [w, x, y, z].

    Returns:
    qConj (numpy.array): Conjugate of the input quaternion.
    """
    qConj = np.array([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]]).T
    return qConj


# %% code test
# read data
data = np.genfromtxt(r"E:\Hsin\git\TWU\KaiCode\golf.csv", delimiter=",", skip_header=1)
timestamp = data[:, 0]
q = data[:, 10:14]
accelerometer = data[:, 7:10]
freeaccelerometer = data[:, 4:7]
gyroscope = data[:, 1:4]

rm = np.empty((len(timestamp), 3, 3))

rm = quatern2rotMat(q)

results = np.empty((np.shape(freeaccelerometer)))

# 使用 for 迴圈處理每個 frame 並計算點積
for i in range(np.shape(freeaccelerometer)[0]):
    results[i, :] = np.dot(rm[i, :, :].T, freeaccelerometer[i, :])

# 使用 quternion 計算
# q_results = np.empty((len(timestamp), 3, 3))

acc_0 = np.concatenate(
    (np.zeros((np.shape(freeaccelerometer)[0], 1)), freeaccelerometer), axis=1
)

q_results = quaternProd(q, quaternProd(acc_0, quaternConj(q)))
qq_results = q_results[:, 1:]

qqq_results = quaternProd(quaternConj(q), quaternProd(acc_0, quaternConj(q)))
qqqq_results = qqq_results[:, 1:]

frequency = 120
num_samples = 192
position_data = double_integration(results, frequency)

q_position_data = double_integration(qq_results, frequency)


qq_position_data = double_integration(qqqq_results, frequency)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(position_data[:, 0], position_data[:, 1], position_data[:, 2])
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(q_position_data[:, 0], q_position_data[:, 1], q_position_data[:, 2])
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(qq_position_data[:, 0], qq_position_data[:, 1], qq_position_data[:, 2])
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")

# %%
import imufusion

# Process sensor data
ahrs = imufusion.Ahrs()
rotation_matrix = np.empty((len(timestamp), 3, 3))

for index in range(len(timestamp)):
    ahrs.update_no_magnetometer(
        gyroscope[index, :], accelerometer[index, :], 1 / 120
    )  # 120 Hz sample rate
    rotation_matrix[index, :, :] = ahrs.quaternion.to_matrix()

results = np.empty((np.shape(freeaccelerometer)))
for i in range(np.shape(freeaccelerometer)[0]):
    results[i, :] = np.dot(rotation_matrix[i, :, :], freeaccelerometer[i, :])

frequency = 120
num_samples = 192
position_data = double_integration(results, frequency)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(position_data[:, 0], position_data[:, 1], position_data[:, 2])
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
