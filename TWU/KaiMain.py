# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:09:26 2023

@author: Hsin Yang, 18.Oct.2023
"""
# %% import package

import numpy as np
import sys

# 路徑改成你放自己code的資料夾
sys.path.append("E:\Hsin\git\git\Code_testing\TWU")
import KaiMadgwickAHRS as kai
import KaiMathFunction as fun

# %% load IMU data
data = np.genfromtxt(
    r"E:\git\Code_testing\TWU\data\golf.csv", delimiter=",", skip_header=1
)
timestamp = data[:, 0]
q = data[:, 10:14]
accelerometer = data[:, 7:10]
freeaccelerometer = data[:, 4:7]
gyroscope = data[:, 1:4]

# %%
ahrs = kai.MadgwickAHRS()

ahrs_q = np.empty([len(timestamp), 4])  # quaternion of Earth relative to sensor
ahrs_R = np.empty([len(timestamp), 3, 3])  # Rotation Matrix of Earth relative to sensor

# Using AHRS algorithm to calculate sensor's orientation
for i in range(len(timestamp)):
    ahrs_q[i, :] = ahrs.UpdateIMU(gyroscope[i, :], accelerometer[i, :])
    ahrs_R[i, :, :] = fun.quatern2rotMat(
        ahrs_q[i, :]
    ).T  # transpose because ahrs provides Earth relative to sensor
# %%
# using rotation matrix to calculate sensor acceleration with respect to Earth frame
earth_acc = np.empty([len(timestamp), 3])

for i in range(len(timestamp)):
    earth_acc[i, :] = np.dot(ahrs_R[i, :], earth_acc[i, :].T)


# %%
import matplotlib.pyplot as plt

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
plt.plot(earth_acc[:, 0], "r", label="X")
plt.plot(earth_acc[:, 1], "g", label="Y")
plt.plot(earth_acc[:, 2], "b", label="Z")
plt.xlabel("sample")
plt.ylabel("g")
plt.title("Tilt-compensated accelerometer")
plt.legend()
plt.show()

# %%
