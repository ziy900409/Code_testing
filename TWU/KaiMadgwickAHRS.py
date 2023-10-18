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


class MadgwickAHRS:
    def __init__(self, sample_period=1 / 120, quaternion=[1, 0, 0, 0], beta=1):
        self.sample_period = sample_period
        self.quaternion = np.array(quaternion)
        self.beta = beta

    def update(self, gyroscope, accelerometer, magnetometer):
        q = self.quaternion

        # Normalise accelerometer measurement
        accelerometer /= np.linalg.norm(accelerometer)

        # Normalise magnetometer measurement
        magnetometer /= np.linalg.norm(magnetometer)

        # Reference direction of Earth's magnetic field
        h = self.quaternion_multiply(
            q,
            self.quaternion_multiply(
                [0, magnetometer[0], magnetometer[1], magnetometer[2]],
                self.quaternion_conjugate(q),
            ),
        )
        b = np.array([0, np.linalg.norm([h[1], h[2]]), 0, h[3]])

        # Gradient descent algorithm corrective step
        f = np.array(
            [
                2 * (q[1] * q[3] - q[0] * q[2]) - accelerometer[0],
                2 * (q[0] * q[1] + q[2] * q[3]) - accelerometer[1],
                2 * (0.5 - q[1] ** 2 - q[2] ** 2) - accelerometer[2],
                2 * b[1] * (0.5 - q[2] ** 2 - q[3] ** 2)
                + 2 * b[3] * (q[1] * q[3] - q[0] * q[2])
                - magnetometer[0],
                2 * b[1] * (q[1] * q[2] - q[0] * q[3])
                + 2 * b[3] * (q[0] * q[1] + q[2] * q[3])
                - magnetometer[1],
                2 * b[1] * (q[0] * q[2] + q[1] * q[3])
                + 2 * b[3] * (0.5 - q[1] ** 2 - q[2] ** 2)
                - magnetometer[2],
            ]
        )

        jacobian = np.array(
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

        step = np.dot(jacobian.T, f)
        step = step / np.linalg.norm(step)  # Normalize step magnitude

        # Compute rate of change of quaternion
        q_dot = (
            0.5
            * self.quaternion_multiply(q, [0, gyroscope[0], gyroscope[1], gyroscope[2]])
            - self.beta * step
        )

        # Integrate to yield quaternion
        q = q + (q_dot * self.sample_period).astype("float64")
        self.quaternion = q / np.linalg.norm(q)  # Normalize quaternion
        return self.quaternion

    def update_imu(self, gyroscope, accelerometer):
        q = self.quaternion

        accelerometer /= np.linalg.norm(accelerometer)

        # Gradient descent algorithm corrective step
        f = np.array(
            [
                2 * (q[1] * q[3] - q[0] * q[2]) - accelerometer[0],
                2 * (q[0] * q[1] + q[2] * q[3]) - accelerometer[1],
                2 * (0.5 - q[1] ** 2 - q[2] ** 2) - accelerometer[2],
            ]
        )

        jacobian = np.array(
            [
                [-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
                [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
                [0, -4 * q[1], -4 * q[2], 0],
            ]
        )

        step = np.dot(jacobian.T, f)
        step = step / np.linalg.norm(step)  # Normalize step magnitude
        # Compute rate of change of quaternion
        q_dot = (
            0.5
            * func.HessianProduct(
                np.array(q),
                np.array([0, gyroscope[0], gyroscope[1], gyroscope[2]]),
                shape="1D",
            )
            - self.beta * step
        )

        # Integrate to yield quaternion
        q = q + (q_dot * self.sample_period).astype("float64")
        quaternion = q / np.linalg.norm(q)
        return self.quaternion
