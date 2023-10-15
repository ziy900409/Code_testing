# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:09:26 2023

@author: Hsin Yang, 15.10.2023
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
    R[:, 0, 0] = 2 * (q[:, 0]**2) - 1 + 2 * (q[:, 1]**2)
    R[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    R[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    R[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    R[:, 1, 1] = 2 * (q[:, 0]**2 - 1 + 2 * (q[:, 2]**2))
    R[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    R[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
    R[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    R[:, 2, 2] = 2 * (q[:, 0]**2 - 1 + 2 * (q[:, 3]**2))
    
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
    ab[:, 0] = a[:, 0]*b[:, 0] - a[:, 1]*b[:, 1] - a[:, 2]*b[:, 2] - a[:, 3]*b[:, 3]
    ab[:, 1] = a[:, 0]*b[:, 1] + a[:, 1]*b[:, 0] + a[:, 2]*b[:, 3] - a[:, 3]*b[:, 2]
    ab[:, 2] = a[:, 0]*b[:, 2] - a[:, 1]*b[:, 3] + a[:, 2]*b[:, 0] + a[:, 3]*b[:, 1]
    ab[:, 3] = a[:, 0]*b[:, 3] + a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1] + a[:, 3]*b[:, 0]
    
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

acc_0 = np.concatenate((np.zeros((np.shape(freeaccelerometer)[0], 1)), freeaccelerometer), axis=1)

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
ax = fig.add_subplot(111, projection='3d')

ax.plot(position_data[:, 0], position_data[:, 1], position_data[:, 2])
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(q_position_data[:, 0], q_position_data[:, 1], q_position_data[:, 2])
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(qq_position_data[:, 0], qq_position_data[:, 1], qq_position_data[:, 2])
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')

# %%
import imufusion
# Process sensor data
ahrs = imufusion.Ahrs()
rotation_matrix = np.empty((len(timestamp), 3, 3))

for index in range(len(timestamp)):
    ahrs.update_no_magnetometer(gyroscope[index, :], accelerometer[index, :], 1 / 120)  # 120 Hz sample rate
    rotation_matrix[index, :, :] = ahrs.quaternion.to_matrix()

results = np.empty((np.shape(freeaccelerometer)))
for i in range(np.shape(freeaccelerometer)[0]):
    results[i, :] = np.dot(rotation_matrix[i, :, :], freeaccelerometer[i, :])
    
frequency = 120
num_samples = 192
position_data = double_integration(results, frequency)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(position_data[:, 0], position_data[:, 1], position_data[:, 2])
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')

# %%
import imufusion
import matplotlib.pyplot as pyplot
import numpy
import sys

# Import sensor data
data = numpy.genfromtxt(r"E:\Hsin\git\TWU\Fusion\Python\sensor_data.csv", delimiter=",", skip_header=1)

sample_rate = 100  # 100 Hz

timestamp = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]
magnetometer = data[:, 7:10]

# Instantiate algorithms
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU,  # convention
                                   0.5,  # gain
                                   2000,  # gyroscope range
                                   10,  # acceleration rejection
                                   10,  # magnetic rejection
                                   5 * sample_rate)  # recovery trigger period = 5 seconds

# Process sensor data
delta_time = numpy.diff(timestamp, prepend=timestamp[0])

euler = numpy.empty((len(timestamp), 3))
quaternion = numpy.empty((len(timestamp), 4))

internal_states = numpy.empty((len(timestamp), 6))
flags = numpy.empty((len(timestamp), 4))

for index in range(len(timestamp)):
    gyroscope[index] = offset.update(gyroscope[index])

    ahrs.update(gyroscope[index], accelerometer[index], magnetometer[index], delta_time[index])
    # quaternion[index] = ahrs.quaternion.to_matrix()
    euler[index] = ahrs.quaternion.to_euler()

    
    ahrs_internal_states = ahrs.internal_states
    internal_states[index] = numpy.array([ahrs_internal_states.acceleration_error,
                                          ahrs_internal_states.accelerometer_ignored,
                                          ahrs_internal_states.acceleration_recovery_trigger,
                                          ahrs_internal_states.magnetic_error,
                                          ahrs_internal_states.magnetometer_ignored,
                                          ahrs_internal_states.magnetic_recovery_trigger])

    ahrs_flags = ahrs.flags
    flags[index] = numpy.array([ahrs_flags.initialising,
                                ahrs_flags.angular_rate_recovery,
                                ahrs_flags.acceleration_recovery,
                                ahrs_flags.magnetic_recovery])


def plot_bool(axis, x, y, label):
    axis.plot(x, y, "tab:cyan", label=label)
    pyplot.sca(axis)
    pyplot.yticks([0, 1], ["False", "True"])
    axis.grid()
    axis.legend()


# Plot Euler angles
figure, axes = pyplot.subplots(nrows=11, sharex=True, gridspec_kw={"height_ratios": [6, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]})

figure.suptitle("Euler angles, internal states, and flags")

axes[0].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
axes[0].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
axes[0].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
axes[0].set_ylabel("Degrees")
axes[0].grid()
axes[0].legend()

# Plot initialising flag
plot_bool(axes[1], timestamp, flags[:, 0], "Initialising")

# Plot angular rate recovery flag
plot_bool(axes[2], timestamp, flags[:, 1], "Angular rate recovery")

# Plot acceleration rejection internal states and flags
axes[3].plot(timestamp, internal_states[:, 0], "tab:olive", label="Acceleration error")
axes[3].set_ylabel("Degrees")
axes[3].grid()
axes[3].legend()

plot_bool(axes[4], timestamp, internal_states[:, 1], "Accelerometer ignored")

axes[5].plot(timestamp, internal_states[:, 2], "tab:orange", label="Acceleration recovery trigger")
axes[5].grid()
axes[5].legend()

plot_bool(axes[6], timestamp, flags[:, 2], "Acceleration recovery")

# Plot magnetic rejection internal states and flags
axes[7].plot(timestamp, internal_states[:, 3], "tab:olive", label="Magnetic error")
axes[7].set_ylabel("Degrees")
axes[7].grid()
axes[7].legend()

plot_bool(axes[8], timestamp, internal_states[:, 4], "Magnetometer ignored")

axes[9].plot(timestamp, internal_states[:, 5], "tab:orange", label="Magnetic recovery trigger")
axes[9].grid()
axes[9].legend()

plot_bool(axes[10], timestamp, flags[:, 3], "Magnetic recovery")

pyplot.show(block="no_block" not in sys.argv)  # don't block when script run by CI














