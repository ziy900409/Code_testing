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
        q += q_dot * self.sample_period
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
            * HessianProduct(
                np.array(q),
                np.array([0, gyroscope[0], gyroscope[1], gyroscope[2]]),
                shape="1D",
            )
            - self.beta * step
        )

        # Integrate to yield quaternion
        q += q_dot * self.sample_period
        quaternion = q / np.linalg.norm(q)
        return self.quaternion


# %%
data = np.genfromtxt(
    r"E:\git\Code_testing\TWU\data\golf.csv", delimiter=",", skip_header=1
)
timestamp = data[:, 0]
q = data[:, 10:14]
accelerometer = data[:, 7:10]
freeaccelerometer = data[:, 4:7]
gyroscope = data[:, 1:4]

ahrs = MadgwickAHRS()

ahrs_q = np.empty([len(timestamp), 4])  # quaternion of Earth relative to sensor
ahrs_R = np.empty([len(timestamp), 3])  # Rotation Matrix of Earth relative to sensor

for i in range(len(timestamp)):
    ahrs_q[i, :] = ahrs.update_imu(gyroscope[i, :], accelerometer[i, :])


# %%


# %%
def HessianProduct(q_1, q_2, shape="1D"):
    """
    calculate the quaternion product of quaternion q_1 and q_2

    Returns:
    Prod (numpy.array): Quaternion product of q_1 and q_2.
    """
    # q_1 = np.array(q)
    # q_2 = np.array([0, gyroscope[0], gyroscope[1], gyroscope[2]])
    # shape = "1D"
    if shape == "1D":
        q_1 = np.array(q_1)
        q_2 = np.array(q_2)
        Prod = np.zeros_like(q_1)
        Prod[0] = q_1[0] * q_2[0] - q_1[1] * q_2[1] - q_1[2] * q_2[2] - q_1[3] * q_2[3]
        Prod[1] = q_1[0] * q_2[1] + q_1[1] * q_2[0] + q_1[2] * q_2[3] - q_1[3] * q_2[2]
        Prod[2] = q_1[0] * q_2[2] - q_1[1] * q_2[3] + q_1[2] * q_2[0] + q_1[3] * q_2[1]
        Prod[3] = q_1[0] * q_2[3] + q_1[1] * q_2[2] - q_1[2] * q_2[1] + q_1[3] * q_2[0]
    elif shape == "2D":
        Prod = np.zeros_like(q_1)
        Prod[:, 0] = (
            q_1[:, 0] * q_2[:, 0]
            - q_1[:, 1] * q_2[:, 1]
            - q_1[:, 2] * q_2[:, 2]
            - q_1[:, 3] * q_2[:, 3]
        )
        Prod[:, 1] = (
            q_1[:, 0] * q_2[:, 1]
            + q_1[:, 1] * q_2[:, 0]
            + q_1[:, 2] * q_2[:, 3]
            - q_1[:, 3] * q_2[:, 2]
        )
        Prod[:, 2] = (
            q_1[:, 0] * q_2[:, 2]
            - q_1[:, 1] * q_2[:, 3]
            + q_1[:, 2] * q_2[:, 0]
            + q_1[:, 3] * q_2[:, 1]
        )
        Prod[:, 3] = (
            q_1[:, 0] * q_2[:, 3]
            + q_1[:, 1] * q_2[:, 2]
            - q_1[:, 2] * q_2[:, 1]
            + q_1[:, 3] * q_2[:, 0]
        )

    return Prod


# %%
def update_imu(q, gyroscope, accelerometer, beta, sample_period):
    # q = quaternion

    # Normalise accelerometer measurement
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

    step = np.linalg.solve(jacobian.T, f)
    step /= np.linalg.norm(step)  # Normalize step magnitude

    # Compute rate of change of quaternion
    q_dot = (
        0.5 * HessianProduct(q, [0, gyroscope[0], gyroscope[1], gyroscope[2]])
        - beta * step
    )

    # Integrate to yield quaternion
    q += q_dot * sample_period
    quaternion = q / np.linalg.norm(q)  # Normalize quaternion
    return quaternion


# %%
q = [1, 0, 0, 0]
# gyroscope = gyroscope[1, :]
# accelerometer = accelerometer[1, :]
# beta = 1
# sample_period = 1/120
# Normalise accelerometer measurement
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
    * HessianProduct(
        np.array(q), np.array([0, gyroscope[0], gyroscope[1], gyroscope[2]]), shape="1D"
    )
    - beta * step
)

# Integrate to yield quaternion
q += q_dot * sample_period
quaternion = q / np.linalg.norm(q)
# %%
