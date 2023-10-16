# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:09:26 2023

@author: Hsin Yang, 16.Oct.2023
"""
# %% import package
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

# %% define Math function


def QuaternionConjugate(q):
    """
    converts a quaternion to its conjugate
    """

    qConj = np.array([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]])
    return


def HessianProduct(q_1, q_2):
    """
    calculate the quaternion product of quaternion q_1 and q_2

    Returns:
    Prod (numpy.array): Quaternion product of q_1 and q_2.
    """

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


class Quaternion:
    """
    還需要再修改
    """

    def __init__(self):
        self = 1

    @staticmethod
    def quaternion_multiply(quaternion1, quaternion2):
        w1, x1, y1, z1 = quaternion1
        w2, x2, y2, z2 = quaternion2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])

    @staticmethod
    def quaternion_conjugate(quaternion):
        w, x, y, z = quaternion
        return np.array([w, -x, -y, -z])


class MadgwickAHRS:
    def __init__(self, sample_period=1 / 256, quaternion=[1, 0, 0, 0], beta=1):
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

        step = np.linalg.solve(jacobian.T, f)
        step /= np.linalg.norm(step)  # Normalize step magnitude

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
            0.5
            * self.quaternion_multiply(q, [0, gyroscope[0], gyroscope[1], gyroscope[2]])
            - self.beta * step
        )

        # Integrate to yield quaternion
        q += q_dot * self.sample_period
        self.quaternion = q / np.linalg.norm(q)  # Normalize quaternion
        return self.quaternion
