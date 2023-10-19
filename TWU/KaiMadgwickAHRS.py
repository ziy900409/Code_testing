# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:09:26 2023

@author: Hsin Yang, 16.Oct.2023
"""
# %% import package

import numpy as np
import sys

# 路徑改成你放自己code的資料夾
sys.path.append("E:\Hsin\git\git\Code_testing\TWU")
import KaiMathFunction as func

# %% define function
import numpy as np


class MadgwickAHRS:
    def __init__(
        self, SamplePeriod=1.0 / 256, Quaternion=np.array([1, 0, 0, 0]), Beta=1
    ):
        self.SamplePeriod = SamplePeriod
        self.Quaternion = Quaternion
        self.Beta = Beta

    def quaternProd(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    def quaternConj(self, q):
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    def Update(self, Gyroscope, Accelerometer, Magnetometer):
        q = self.Quaternion

        # Normalise accelerometer measurement
        if np.linalg.norm(Accelerometer) == 0:
            return
        Accelerometer = Accelerometer / np.linalg.norm(Accelerometer)

        # Normalise magnetometer measurement
        if np.linalg.norm(Magnetometer) == 0:
            return
        Magnetometer = Magnetometer / np.linalg.norm(Magnetometer)

        # Reference direction of Earth's magnetic field
        h = self.quaternProd(
            q,
            self.quaternProd(
                [0, Magnetometer[0], Magnetometer[1], Magnetometer[2]],
                self.quaternConj(q),
            ),
        )
        b = np.array([0, np.linalg.norm([h[1], h[2]]), 0, h[3]])

        # Gradient descent algorithm corrective step
        F = np.array(
            [
                2 * (q[1] * q[3] - q[0] * q[2]) - Accelerometer[0],
                2 * (q[0] * q[1] + q[2] * q[3]) - Accelerometer[1],
                2 * (0.5 - q[1] ** 2 - q[2] ** 2) - Accelerometer[2],
                2 * b[1] * (0.5 - q[2] ** 2 - q[3] ** 2)
                + 2 * b[3] * (q[1] * q[3] - q[0] * q[2])
                - Magnetometer[0],
                2 * b[1] * (q[1] * q[2] - q[0] * q[3])
                + 2 * b[3] * (q[0] * q[1] + q[2] * q[3])
                - Magnetometer[1],
                2 * b[1] * (q[0] * q[2] + q[1] * q[3])
                + 2 * b[3] * (0.5 - q[1] ** 2 - q[2] ** 2)
                - Magnetometer[2],
            ]
        )
        J = np.array(
            [
                [-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
                [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
                [0, -4 * q[1], -4 * q[2], 0],
                [
                    -2 * b[3] * q[2],
                    2 * b[3] * q[3],
                    -4 * b[1] * q[2] - 2 * b[3] * q[0],
                    -4 * b[1] * q[3] + 2 * b[3] * q[1],
                ],
                [
                    -2 * b[1] * q[3] + 2 * b[3] * q[1],
                    2 * b[1] * q[2] + 2 * b[3] * q[0],
                    2 * b[1] * q[1] + 2 * b[3] * q[3],
                    -2 * b[1] * q[0] + 2 * b[3] * q[2],
                ],
                [
                    2 * b[1] * q[2],
                    2 * b[1] * q[3] - 4 * b[3] * q[1],
                    2 * b[1] * q[0] - 4 * b[3] * q[2],
                    2 * b[1] * q[1],
                ],
            ]
        )
        step = np.dot(J.T, F)
        step /= np.linalg.norm(step)

        # Compute rate of change of quaternion
        qDot = (
            0.5 * self.quaternProd(q, [0, Gyroscope[0], Gyroscope[1], Gyroscope[2]])
            - self.Beta * step
        )

        # Integrate to yield quaternion
        q = q + (qDot * self.SamplePeriod).astype("float64")
        self.Quaternion = q / np.linalg.norm(q)
        return self.Quaternion

    def UpdateIMU(self, Gyroscope, Accelerometer):
        q = self.Quaternion

        # Normalise accelerometer measurement
        if np.linalg.norm(Accelerometer) == 0:
            return
        Accelerometer = Accelerometer / np.linalg.norm(Accelerometer)

        # Gradient descent algorithm corrective step
        F = np.array(
            [
                2 * (q[1] * q[3] - q[0] * q[2]) - Accelerometer[0],
                2 * (q[0] * q[1] + q[2] * q[3]) - Accelerometer[1],
                2 * (0.5 - q[1] ** 2 - q[2] ** 2) - Accelerometer[2],
            ]
        )
        J = np.array(
            [
                [-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
                [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
                [0, -4 * q[1], -4 * q[2], 0],
            ]
        )
        step = np.dot(J.T, F)
        step /= np.linalg.norm(step)

        # Compute rate of change of quaternion
        qDot = (
            0.5
            * self.quaternProd(
                np.array(q), np.array([0, Gyroscope[0], Gyroscope[1], Gyroscope[2]])
            )
            - self.Beta * step
        )

        # Integrate to yield quaternion
        q = q + (qDot * self.SamplePeriod).astype("float64")
        self.Quaternion = q / np.linalg.norm(q)
        return self.Quaternion


# %%
# 創建 MadgwickAHRS 物件
# ahrs = MadgwickAHRS()

# # 模擬陀螺儀、加速度計和磁力計的數據
# gyroscope_data = np.array([0.1, 0.2, 0.3])  # 陀螺儀數據 (rad/s)
# accelerometer_data = np.array([0.9, 0.1, 0.2])  # 加速度計數據 (m/s^2)
# magnetometer_data = np.array([0.3, 0.2, 0.9])  # 磁力計數據 (uT)

# # 更新 AHRS 物件
# ahrs.Update(gyroscope_data, accelerometer_data, magnetometer_data)

# # 獲取更新後的四元數
# updated_quaternion = ahrs.Quaternion

# # 輸出更新後的四元數
# print("Updated Quaternion:", updated_quaternion)
